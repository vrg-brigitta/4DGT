# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Dynamic Replica dataset loader
Adapted from mvaria_dataset.py for loading dynamic replica data
Preload things we need into memory, including camera parameters and images
"""

from collections import defaultdict
from dataclasses import dataclass, fields, is_dataclass
import gzip
import json
import math
import os
from copy import deepcopy
from functools import partial
from typing import Any, List, Optional, Tuple, Union, get_args, get_origin

from tlod.data_loader.utils import readPFM, load_16big_png_depth

from sympy import root

import cv2
import numpy as np
import imageio.v2 as iio

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from ..easyvolcap.utils.cam_utils import (
    align_c2ws,
    average_c2ws,
    generate_spiral_path,
    interpolate_camera_lins,
    interpolate_camera_path,
)
from ..easyvolcap.utils.console_utils import (
    dotdict,
    join,
    logger,
    magenta,
)
from ..easyvolcap.utils.data_utils import as_numpy_func
from ..easyvolcap.utils.math_utils import affine_inverse, affine_padding
from ..easyvolcap.utils.parallel_utils import parallel_execution

from ..misc.io_helper import pathmgr

@dataclass
class FileAnnotation:
    path: str
    size: Optional[Tuple[int, int]] = None

@dataclass
class DynamicReplicaFrameAnnotation:
    """A dataclass used to load annotations from json."""

    sequence_name: str
    camera_name: Optional[str] = None
    image: Optional[FileAnnotation] = None
    depth: Optional[FileAnnotation] = None
    mask: Optional[FileAnnotation] = None
    viewpoint: Any = None


def _load_dataclass_value(value: Any, expected_type: Any) -> Any:
    origin = get_origin(expected_type)
    if origin is list or origin is List:
        elem_type = get_args(expected_type)[0] if get_args(expected_type) else Any
        return [_load_dataclass_value(v, elem_type) for v in value]

    if origin is tuple or origin is Tuple:
        elem_types = get_args(expected_type)
        if len(elem_types) == 2 and elem_types[1] is Ellipsis:
            return tuple(_load_dataclass_value(v, elem_types[0]) for v in value)
        return tuple(
            _load_dataclass_value(v, elem_types[i] if i < len(elem_types) else Any)
            for i, v in enumerate(value)
        )

    if origin is Union:
        for arg in get_args(expected_type):
            if arg is type(None) and value is None:
                return None
            try:
                return _load_dataclass_value(value, arg)
            except Exception:
                continue
        return value

    if expected_type is Any or expected_type is None:
        return value

    if is_dataclass(expected_type):
        if not isinstance(value, dict):
            return value
        field_types = {f.name: f.type for f in fields(expected_type)}
        kwargs = {}
        for key, val in value.items():
            if key in field_types:
                kwargs[key] = _load_dataclass_value(val, field_types[key])
        return expected_type(**kwargs)

    if isinstance(value, expected_type):
        return value

    if expected_type in (str, int, float, bool):
        return expected_type(value)

    return value


def load_dataclass(file_obj: Any, expected_type: Any) -> Any:
    data = json.load(file_obj)
    return _load_dataclass_value(data, expected_type)


def load_dynamic_replica_images(image_files, data_path, xs, ys, Ws, Hs, hs, ws):
    """
    Load images from dynamic replica dataset
    
    Args:
        image_files: List of image file names
        data_path: Path to directory containing images
        xs, ys: x, y crop offsets for each image
        Ws, Hs: Original widths and heights
        hs, ws: Target heights and widths after resizing
        
    Returns:
        Array of images of shape (N, H, W, C)
    """
    imgs = []
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(data_path, img_file)
        img = iio.imread(img_path)
        
        # Convert to float if needed
        if img.dtype != np.float32:
            img = img.astype(np.float32) / 255.0
        
        # Resize
        img = cv2.resize(img, dsize=(ws[i], hs[i]))
        
        # Crop
        img = img[ys[i]:ys[i] + Hs[i], xs[i]:xs[i] + Ws[i]]
        
        imgs.append(img)
    
    return np.array(imgs)


class DynamicReplicaDataset(Dataset):
    def __init__(
        self,
        mode: str = "TEST",
        data_root: str = ".\\data\\dynamicreplica",
        input_image_res: Tuple[int] = (256, 256),
        input_image_num: int = 8,
        output_image_res: Tuple[int] = (256, 256),
        output_image_num: int = 8,
        seq_sample: Tuple[Optional[int]] = (0, None, 1),
        frame_sample: Tuple[Optional[int]] = (0, None, 1),
        view_sample: Tuple[Optional[int]] = (0, 1, 1),
        sample_interval: int = 1,
        seq_data_roots: Tuple[str] = ("",),
        align_cameras: bool = False,
        novel_time_sampling: bool = False,
        novel_time_frame_sample: Tuple[Optional[int]] = (0, None, 2),
        novel_view_interp_input: bool = False,
        novel_view_timestamps: Tuple[float] = (),
        novel_view_spiral_window: int = 32,
        loaded_to_seconds: float = 1.0,
        loaded_to_meters: float = 1.0,
        force_reload: bool = False,
        **kwargs,
    ):
        """
        DynamicReplicaDataset for inference/testing.
        
        Args:
            mode: Must be "TEST" - this dataset only supports inference mode
            data_root: Root directory containing the dynamic replica data
            
            # Image definition
            input_image_res: Resolution for input images
            input_image_num: Number of input images
            output_image_res: Resolution for output images
            output_image_num: Number of output images
            
            # Frame definition
            seq_sample: Sequence sampling [start, end, step]
            frame_sample: Frame sampling [start, end, step]
            view_sample: View sampling [start, end, step] - only first view used
            sample_interval: Distance between frames to sample
            
            # Data paths
            seq_data_roots: Root directories for sequence data (usually just one)
            align_cameras: Use average camera pose
            
            # Novel view/time settings
            novel_time_sampling: Whether to do novel time sampling
            novel_time_frame_sample: Novel time frame sampling parameters
            novel_view_interp_input: Whether to interpolate novel views for input
            novel_view_timestamps: Specific timestamps for novel views
            novel_view_spiral_window: Number of frames centering target for spiral lookat
            
            # Data loading settings
            loaded_to_seconds: Conversion factor for timestamps to seconds
            loaded_to_meters: Conversion factor for distances to meters
            force_reload: Force reload data even if cached
        """
        super().__init__()
        
        assert mode == "TEST", f"DynamicReplicaDataset only supports TEST mode, got {mode}"
        
        # Discover sequences - each subdirectory is a sequence - but only the one endind with _left
        seq_dir = os.path.join(data_root, mode.lower())
        if os.path.isdir(seq_dir):
            seqs = sorted([d for d in os.listdir(seq_dir) 
                          if os.path.isdir(os.path.join(seq_dir, d)) and d.endswith("_left")])
            
            # Apply sequence sampling
            b, e, s = seq_sample
            seqs = seqs[b:e:s]
        else:
            seqs = []
            logger.warn(f"Data root does not exist: {data_root}")
        
        self.data_root = data_root
        self.input_image_res = input_image_res
        self.input_segment_length = input_image_num
        self.output_image_res = output_image_res
        self.output_segment_length = output_image_num
        
        self.seq_data_roots = deepcopy(seq_data_roots)
        self.view_sample = view_sample
        self.batch_image_num = max(input_image_num, output_image_num)
        
        self.novel_time_sampling = novel_time_sampling
        self.novel_time_frame_sample = novel_time_frame_sample
        
        self.align_cameras = align_cameras
        self.sample_interval = sample_interval
        
        self.novel_view_interp_input = novel_view_interp_input
        self.novel_view_timestamps = novel_view_timestamps
        self.novel_view_spiral_window = novel_view_spiral_window
        
        max_image_size = max(max(input_image_res), max(output_image_res))
        
        b, e, s = view_sample
        
        seqs = np.asarray(seqs)
        logger.info(
            f"Number of sequences used for {magenta(mode)}: {len(seqs)}, seqs: {seqs}"
        )
        
        # load frame annotations
        split = mode.lower()
        frame_annotations_file = f'frame_annotations_{split}.jgz'
    
        with gzip.open(os.path.join(data_root, split, frame_annotations_file), "rt", encoding="utf8") as zipfile:
            frame_annots_list = load_dataclass(
                        zipfile, List[DynamicReplicaFrameAnnotation]
                    )
        seq_annot = defaultdict(lambda: defaultdict(list))
        for frame_annot in frame_annots_list:
            seq_annot[frame_annot.sequence_name][frame_annot.camera_name].append(frame_annot)

        def print_progress(i, total, result):
            if i % max(1, math.ceil(total / 20)) == 0 or i == total - 1:
                logger.info(f"Loaded {i+1}/{total} ({(i+1)/total*100:.2f}%) sequences")
            return result

        sample_list = []

        for seq_name in seq_annot.keys():
            filenames = defaultdict(lambda: defaultdict(list))
            for cam in ['left','right']:
                for framedata in seq_annot[seq_name][cam]:
                    im_path = os.path.join(data_root, split, framedata.image.path)
                    depth_path = os.path.join(data_root, split, framedata.depth.path)
                    mask_path = os.path.join(data_root, split, framedata.mask.path)

                    assert os.path.isfile(im_path), im_path
                    assert os.path.isfile(depth_path), depth_path
                    assert os.path.isfile(mask_path), mask_path

                    filenames['image'][cam].append(im_path)
                    filenames['depth'][cam].append(depth_path)
                    filenames['mask'][cam].append(mask_path)

                    filenames['viewpoint'][cam].append(framedata.viewpoint)
                    filenames['metadata'][cam].append([framedata.sequence_name, framedata.image.size])
                    
                    for k in filenames.keys():
                        assert len(filenames[k][cam])==len(filenames['image'][cam])>0, framedata.sequence_name 
                        

            seq_len = len(filenames['image'][cam])
            print('seq_len', seq_name, seq_len)

            self.sample_len: int = -1 # TODO: or 50
            step = self.sample_len if self.sample_len>0 else seq_len
            counter=0
            
            # TODO: use print progress
            # TODO: save to output_list and parallelize loading
            for ref_idx in range(0, seq_len, step):
                sample_filenames = defaultdict(lambda: defaultdict(list))
                for cam in ['left','right']:
                    for idx in range(ref_idx, ref_idx+step):
                        for k in filenames.keys():
                            sample_filenames[k][cam].append(filenames[k][cam][idx])
                        
                sample_list.append(sample_filenames)
                counter+=1 


        # TODO: remove this debug code and parallelize loading
        print(sample_list[0].keys()) # ['image', 'depth', 'mask', 'viewpoint', 'metadata']
        example = self.getitem_from_sample(sample_list[0])
        print(example.keys()) # ['metadata', 'img', 'disp', 'valid_disp', 'mask', 'viewpoint']
        output_list = [[example]]

        # Load camera parameters for each sequence
        # output_list = []
        # for i, sample in enumerate(sample_list):
        #     # outputs = parallel_execution(
        #     #     action=partial(
        #     #         self.getitem_from_sample,
        #     #         sample,
        #     #     ),
        #     #     print_progress=False,
        #     #     callback=print_progress,
        #     # )
        #     output_list.append(self.getitem_from_sample(sample))

        
        # for i, seq_data_root in enumerate(self.seq_data_roots):
        #     logger.info(f'Loading "{seq_data_root}" camera poses from dynamic replica')
            
        #     b_frame, e_frame, s_frame = frame_sample
            
        #     inputs = [
        #         os.path.join(data_root, key, seq_data_root) if seq_data_root else os.path.join(data_root, key)
        #         for key in seqs
        #     ]
        #     outputs = parallel_execution(
        #         inputs,
        #         action=partial(
        #             load_dynamic_replica_cameras,
        #             frame_sample=(b_frame, e_frame, s_frame),
        #         ),
        #         print_progress=False,
        #         callback=print_progress,
        #     )
        #     output_list.append(outputs)
        
        total = len(seqs)
        self.lengths = []
        self.seqs = dotdict()
        
        # TODO: seq_data_roots is not used for dynamic replica, the first iteration could be removed
        for i, (seq_data_root, outputs) in enumerate(
            zip(self.seq_data_roots, output_list)
        ):
            logger.info(f'Parsing "{seq_data_root}" images')
            
            for j, (key, output) in enumerate(zip(seqs, outputs)):
                print('output', output.keys()) # ['viewpoint', 'metadata', 'img', 'disp', 'valid_disp', 'mask']

                ims = output['img']

                # output["metadata"] [:] [1] is the image size (H, W)
                Hs = [ meta[1][0] for meta in output["metadata"]]
                Ws = [ meta[1][1] for meta in output["metadata"]]

                # output["viewpoint"] [:] is the camera parameters [0] for left, including R and T
                ts = [ viewpoint[0]["T"] for viewpoint in output["viewpoint"]]
                RTs = [ viewpoint[0]["R"] for viewpoint in output["viewpoint"]]

                # TODO: map the K
                Ks = []
                
                if not len(ims):
                    logger.warn(f"Empty sequence {key}")
                    continue
                
                if key not in self.seqs:
                    self.seqs[key] = dotdict()
                
                self.seqs[key][seq_data_root] = dotdict()
                self.seqs[key][seq_data_root].ims = ims
                self.seqs[key][seq_data_root].Hs = Hs
                self.seqs[key][seq_data_root].Ws = Ws
                self.seqs[key][seq_data_root].Ks = Ks
                self.seqs[key][seq_data_root].RTs = RTs  # w2c
                self.seqs[key][seq_data_root].ts = ts  # timestamps
                
                # Load actual images
                max_h = max(Hs)
                max_w = max(Ws)
                h, w = input_image_res
                
                # Handle center crop and resizing
                xs, ys = [], []
                ws_list, hs_list = [], []
                ratios = []
                
                for idx in range(len(ims)):
                    if Hs[idx] > Ws[idx]:
                        ratio_x = w / Ws[idx]
                        ratio_y = int(ratio_x * Hs[idx] + 0.5) / Hs[idx]
                        ratio = ratio_x
                    else:
                        ratio_y = h / Hs[idx]
                        ratio_x = int(ratio_y * Ws[idx] + 0.5) / Ws[idx]
                        ratio = ratio_y
                    
                    if h / Hs[idx] > w / Ws[idx]:
                        x, y = int((Ws[idx] * ratio_x - w + 0.5) // 2), 0
                    else:
                        x, y = 0, int((Hs[idx] * ratio_y - h + 0.5) // 2)
                    
                    Ks[idx, 0:1] *= ratio_x
                    Ks[idx, 1:2] *= ratio_y
                    Ks[idx, 0, 2] -= x
                    Ks[idx, 1, 2] -= y
                    xs.append(x)
                    ys.append(y)
                    ws_list.append(int(ratio_x * Ws[idx] + 0.5))
                    hs_list.append(int(ratio_y * Hs[idx] + 0.5))
                    Hs[idx] = h
                    Ws[idx] = w
                    ratios.append(ratio)
                
                ratios = np.asarray(ratios)
                
                # Load images
                seq_path = os.path.join(data_root, key, seq_data_root) if seq_data_root else os.path.join(data_root, key)
                self.seqs[key][seq_data_root].ims_data = load_dynamic_replica_images(
                    ims, seq_path, xs, ys, Ws, Hs, hs_list, ws_list
                )
                
                # Normalize images
                self.seqs[key][seq_data_root].ims_data = self.seqs[key][seq_data_root].ims_data / 1.0  # Already in 0-1 range
                self.seqs[key][seq_data_root].ims_data = self.seqs[key][seq_data_root].ims_data * 2 - 1  # Convert to -1 to 1
                
                if j % max(1, math.ceil(total / 20)) == 0 or j == total - 1:
                    logger.info(
                        f"Parsed {j+1}/{total} ({(j+1)/total*100:.2f}%) sequences"
                    )
                
                if i == 0:
                    self.lengths.append(
                        int(
                            len(self.seqs[key][seq_data_root].ims)
                            // self.sample_interval
                        )
                    )
        
        self.seq_keys = list(self.seqs.keys())
        self.lengths = np.asarray(self.lengths)
        self.cumsum_lengths = [0] + self.lengths.cumsum(-1).tolist()
        
        if len(self.seq_keys) > 0:
            assert (
                len(self.seq_keys) == len(self.cumsum_lengths) - 1
                and len(self.seq_keys) == len(self.lengths)
                and len(self.seq_keys) == len(self.seqs)
            ), f"Lengths mismatch: {len(self.seq_keys)}, {len(self.cumsum_lengths) - 1}, {len(self.lengths)}, {len(self.seqs)}"
    
    def __len__(self):
        if not self.cumsum_lengths:
            return 0
        return self.cumsum_lengths[-1]
    
    def read_gen(self, file_name, pil=False):
        ext = os.path.splitext(file_name)[-1]
        if ext == ".png" or ext == ".jpeg" or ext == ".ppm" or ext == ".jpg":
            from PIL import Image
            return Image.open(file_name)
        elif ext == ".bin" or ext == ".raw":
            return np.load(file_name)
        # elif ext == ".flo":
        #     return os.path.readFlow(file_name).astype(np.float32)
        elif ext == ".pfm":
            flow = readPFM(file_name).astype(np.float32)
            if len(flow.shape) == 2:
                return flow
            else:
                return flow[:, :, :-1]
        return []

    def _get_output_tensor(self, sample):
        output_tensor = defaultdict(list)
        sample_size = len(sample["image"]["left"])
        output_tensor_keys = ["img", "disp", "valid_disp", "mask"]
        add_keys = ["viewpoint", "metadata"]
        for add_key in add_keys:
            if add_key in sample:
                output_tensor_keys.append(add_key)

        for key in output_tensor_keys:
            output_tensor[key] = [[] for _ in range(sample_size)]

        if "viewpoint" in sample:
            # viewpoint_left = self._get_pytorch3d_camera(
            #     sample["viewpoint"]["left"][0],
            #     sample["metadata"]["left"][0][1],
            #     scale=1.0,
            # )
            # viewpoint_right = self._get_pytorch3d_camera(
            #     sample["viewpoint"]["right"][0],
            #     sample["metadata"]["right"][0][1],
            #     scale=1.0,
            # )

            # TODO: depth2disparity_scale
            depth2disp_scale = 1 # depth2disparity_scale(
            #     viewpoint_left,
            #     viewpoint_right,
            #     torch.Tensor(sample["metadata"]["left"][0][1])[None],
            # )

        for i in range(sample_size):
            for cam in ["left", "right"]:
                if "mask" in sample and cam in sample["mask"]:
                    mask = self.read_gen(sample["mask"][cam][i])
                    mask = np.array(mask) / 255.0
                    output_tensor["mask"][i].append(mask)

                if "viewpoint" in sample and cam in sample["viewpoint"]:
                #     viewpoint = self._get_pytorch3d_camera(
                #         sample["viewpoint"][cam][i],
                #         sample["metadata"][cam][i][1],
                #         scale=1.0,
                #     )
                    viewpoint = {
                        "R": torch.tensor(sample["viewpoint"][cam][i]["R"], dtype=torch.float)[None],
                        "T": torch.tensor(sample["viewpoint"][cam][i]["T"], dtype=torch.float)[None],
                    }
                    output_tensor["viewpoint"][i].append(viewpoint)

                if "metadata" in sample and cam in sample["metadata"]:
                    metadata = sample["metadata"][cam][i]
                    output_tensor["metadata"][i].append(metadata)

                if cam in sample["image"]:

                    img = self.read_gen(sample["image"][cam][i])
                    img = np.array(img).astype(np.uint8)

                    # grayscale images
                    if len(img.shape) == 2:
                        img = np.tile(img[..., None], (1, 1, 3))
                    else:
                        img = img[..., :3]
                    output_tensor["img"][i].append(img)

                if cam in sample["disparity"]:
                    disp = self.disparity_reader(sample["disparity"][cam][i])
                    if isinstance(disp, tuple):
                        disp, valid_disp = disp
                    else:
                        valid_disp = disp < 512
                    disp = np.array(disp).astype(np.float32)

                    disp = np.stack([-disp, np.zeros_like(disp)], axis=-1)

                    output_tensor["disp"][i].append(disp)
                    output_tensor["valid_disp"][i].append(valid_disp)

                elif "depth" in sample and cam in sample["depth"]:
                    depth = load_16big_png_depth(sample["depth"][cam][i])

                    depth_eps = 1e-5
                    depth_mask = depth < depth_eps
                    depth[depth_mask] = depth_eps

                    disp = depth2disp_scale / depth
                    disp[depth_mask] = 0
                    valid_disp = (disp < 512) * (1 - depth_mask)

                    disp = np.array(disp).astype(np.float32)
                    disp = np.stack([-disp, np.zeros_like(disp)], axis=-1)
                    output_tensor["disp"][i].append(disp)
                    output_tensor["valid_disp"][i].append(valid_disp)

        return output_tensor

    def getitem_from_sample(self, sample):
        output_tensor = self._get_output_tensor(sample)

        sample_size = len(sample["image"]["left"])

        # TODO: check if we need augmentor
        # if self.augmentor is not None:
        #     output_tensor["img"], output_tensor["disp"] = self.augmentor(
        #         output_tensor["img"], output_tensor["disp"]
        #     )
        for i in range(sample_size):
            for cam in (0, 1):
                if cam < len(output_tensor["img"][i]):
                    img = (
                        torch.from_numpy(output_tensor["img"][i][cam])
                        .permute(2, 0, 1)
                        .float()
                    )
                    # TODO: img padding
                    # if self.img_pad is not None:
                    #     padH, padW = self.img_pad
                    #     img = F.pad(img, [padW] * 2 + [padH] * 2)
                    output_tensor["img"][i][cam] = img

                if cam < len(output_tensor["disp"][i]):
                    disp = (
                        torch.from_numpy(output_tensor["disp"][i][cam])
                        .permute(2, 0, 1)
                        .float()
                    )

                    valid_disp = (
                        (disp[0].abs() < 512)
                        & (disp[1].abs() < 512)
                        & (disp[0].abs() != 0)
                    )
                    disp = disp[:1]

                    output_tensor["disp"][i][cam] = disp
                    output_tensor["valid_disp"][i][cam] = valid_disp.float()

                if "mask" in output_tensor and cam < len(output_tensor["mask"][i]):
                    mask = torch.from_numpy(output_tensor["mask"][i][cam]).float()
                    output_tensor["mask"][i][cam] = mask

                if "viewpoint" in output_tensor and cam < len(
                    output_tensor["viewpoint"][i]
                ):
                    viewpoint = output_tensor["viewpoint"][i][cam]
                    output_tensor["viewpoint"][i][cam] = viewpoint

        res = {}
        if "viewpoint" in output_tensor:
            res["viewpoint"] = output_tensor["viewpoint"]
        if "metadata" in output_tensor:
            res["metadata"] = output_tensor["metadata"]

        for k, v in output_tensor.items():
            if k != "viewpoint" and k != "metadata":
                for i in range(len(v)):
                    if len(v[i]) > 0:
                        v[i] = torch.stack(v[i])
                if len(v) > 0 and (len(v[0]) > 0):
                    res[k] = torch.stack(v)
        return res

    def __getitem__(self, idx: int):
        if len(self) == 0:
            raise IndexError("Dataset is empty")
        
        idx = idx % self.cumsum_lengths[-1]
        
        seq_idx = np.searchsorted(self.cumsum_lengths, idx, side="right").item() - 1
        key = self.seq_keys[seq_idx]
        sub_idx = idx - self.cumsum_lengths[seq_idx]
        global_abs_start_ind = sub_idx * self.sample_interval
        
        rgb_input = []
        rgb_output = []
        rays_t_un_input = []
        rays_t_un_output = []
        cameras_input = []
        cameras_output = []
        img_name_output = []
        ratios_output = []
        c2w_avg = None
        
        vb, ve, vs = self.view_sample
        fb, _, fs = self.novel_time_frame_sample
        inputs = np.arange(len(self.seq_data_roots))[vb:ve:vs]
        n_inputs = len(inputs)
        
        for i, seq_data_root in enumerate(self.seq_data_roots):
            curr_len = len(self.seqs[key][seq_data_root].ts)
            pre = os.path.join(self.data_root, key, seq_data_root) if seq_data_root else os.path.join(self.data_root, key)
            is_input = i in inputs
            
            input_sampling_starting_offset = max(
                (fs // n_inputs) * (i - n_inputs // 2), -fb
            )
            dist = 1
            abs_start_ind = global_abs_start_ind
            abs_end_ind = global_abs_start_ind + self.batch_image_num * dist
            target_num_images = self.batch_image_num
            
            if i != 0:
                abs_end_ind = min(curr_len, abs_end_ind)
                start = self.seqs[key][self.seq_data_roots[0]].ts[abs_start_ind]
                end = self.seqs[key][self.seq_data_roots[0]].ts[abs_end_ind - 1]
                ts = self.seqs[key][seq_data_root].ts
                abs_start_ind = np.searchsorted(ts, start, side="left")
                abs_end_ind = np.searchsorted(ts, end, side="left")
                target_num_images = abs_end_ind - abs_start_ind
            
            abs_end_ind = min(curr_len, abs_end_ind)
            abs_start_ind = min(abs_start_ind, abs_end_ind - target_num_images)
            abs_start_ind = max(0, abs_start_ind)
            
            sample_inds = np.arange(abs_start_ind, abs_end_ind, dist)
            if len(sample_inds) != target_num_images:
                sample_inds = sample_inds[:target_num_images]
            
            ims = self.seqs[key][seq_data_root].ims[sample_inds].copy()
            Hs = self.seqs[key][seq_data_root].Hs[sample_inds].copy()
            Ws = self.seqs[key][seq_data_root].Ws[sample_inds].copy()
            Ks = self.seqs[key][seq_data_root].Ks[sample_inds].copy()
            RTs = self.seqs[key][seq_data_root].RTs[sample_inds].copy()
            ts = self.seqs[key][seq_data_root].ts[sample_inds].copy()
            imgs = self.seqs[key][seq_data_root].ims_data[sample_inds].copy()
            
            ts -= ts.min()
            ts = ts.astype(np.float32)
            
            # Convert images to NCHW format
            if len(imgs.shape) == 3:
                imgs = np.transpose(imgs[None], (0, 3, 1, 2))[0]
            elif len(imgs.shape) == 4:
                imgs = np.transpose(imgs, (0, 3, 1, 2))
            
            # Convert pose matrices
            c2ws = as_numpy_func(affine_inverse)(RTs)
            
            if self.align_cameras and c2w_avg is None:
                c2w_avg = average_c2ws(
                    as_numpy_func(affine_inverse)(RTs),
                    align_cameras=False,
                    look_at_center=True,
                )
            
            if c2w_avg is not None:
                c2ws = align_c2ws(c2ws, c2w_avg)
                c2ws = as_numpy_func(affine_padding)(c2ws)
            
            # Pack cameras
            from .utils import pack_c2ws_to_cameras
            cameras = pack_c2ws_to_cameras(c2ws, Ks, Hs, Ws)
            
            if is_input:
                if self.novel_time_sampling:
                    b, e, s = self.novel_time_frame_sample
                    b = b + input_sampling_starting_offset
                    e = len(imgs)
                    inds = np.arange(b, e, s)
                else:
                    inds = np.arange(len(imgs))
                
                imgs_input = imgs[inds]
                cams_input = cameras[inds]
                
                rgb_input.append(imgs_input)
                cameras_input.append(cams_input)
                rays_t_un_input.append(ts[inds])
            
            inds = np.arange(len(imgs))
            
            rgb_output.append(imgs[inds])
            rays_t_un_output.append(ts[inds])
            cameras_output.append(cameras[inds])
            img_name_output.append(
                np.asarray([os.path.join(key, seq_data_root, im) if seq_data_root else os.path.join(key, im) for im in ims[inds]])
            )
            ratios_output.append(np.ones(len(ims[inds]), dtype=np.float32))
        
        batch = dotdict()
        batch.rgb_input = np.concatenate(rgb_input).astype(np.float32) if rgb_input else np.array([]).astype(np.float32)
        batch.rays_t_un_input = np.concatenate(rays_t_un_input).astype(np.float32) if rays_t_un_input else np.array([]).astype(np.float32)
        batch.cameras_input = np.concatenate(cameras_input).astype(np.float32) if cameras_input else np.array([]).astype(np.float32)
        batch.rgb_output = np.concatenate(rgb_output).astype(np.float32)
        batch.rays_t_un_output = np.concatenate(rays_t_un_output).astype(np.float32)
        batch.cameras_output = np.concatenate(cameras_output).astype(np.float32)
        batch.img_name_output = np.concatenate(img_name_output)
        batch.ratios_output = np.concatenate(ratios_output).astype(np.float32)
        
        batch.img_name_output = batch.img_name_output.tolist()
        batch.c2w_avg = c2w_avg if c2w_avg is not None else np.eye(4)
        
        return batch
