# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Dynamic Replica dataset loader
Adapted from mvaria_dataset.py for loading dynamic replica data
Preload things we need into memory, including camera parameters and images
"""

import gzip
import json
import math
import os
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, List, Optional, Tuple, Union, get_args, get_origin

import cv2
import imageio.v2 as iio
import numpy as np
import torch
from torch.utils.data import Dataset

from ..easyvolcap.utils.parallel_utils import parallel_execution
from tlod.data_loader.utils import readPFM, load_16big_png_depth
from ..easyvolcap.utils.cam_utils import (
    align_c2ws,
    average_c2ws,
)
from ..easyvolcap.utils.console_utils import (
    dotdict,
    logger,
    magenta,
)
from ..easyvolcap.utils.data_utils import as_numpy_func
from ..easyvolcap.utils.math_utils import affine_inverse, affine_padding


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

        assert img.shape[:2] == (Hs[i], Ws[i]), \
            f"Frame {i}: crop produced {img.shape[:2]}, expected ({Hs[i]}, {Ws[i]}). " \
            f"Check xs/ys offsets vs resize dimensions."

        imgs.append(img)

    return np.stack(imgs)


class DynamicReplicaDataset(Dataset):
    def __init__(
            self,
            mode: str = "TEST",
            data_root: str = ".\\data\\dynamicreplica",
            input_image_res: Tuple[int, int] = (256, 256),
            input_image_num: int = 8,
            output_image_res: Tuple[int, int] = (256, 256),
            output_image_num: int = 8,
            seq_sample: Tuple[Optional[int], Optional[int], Optional[int]] = (0, None, 1),
            frame_sample: Tuple[Optional[int], Optional[int], Optional[int]] = (0, None, 1),
            view_sample: Tuple[Optional[int], Optional[int], Optional[int]] = (0, 1, 1),
            sample_interval: int = 1,
            seq_data_roots: Tuple[str] = ("",),
            align_cameras: bool = False,
            novel_time_sampling: bool = False,
            novel_time_frame_sample: Tuple[Optional[int], Optional[int], Optional[int]] = (0, None, 2),
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
            mode: Must be "TEST" OR "VAL"/"VALIDATION" - this dataset only supports inference mode
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

        mode_upper = mode.upper()
        if mode_upper in ("VAL", "VALIDATION", "VALID"):
            mode = "VALID"
            mode_upper = "TEST"

        assert mode_upper == "TEST", f"DynamicReplicaDataset only supports TEST/VALID mode, got {mode}"

        # Discover sequences - each subdirectory is a sequence - but only the one endind with _left
        split = mode.lower()
        seq_dir = os.path.join(data_root, split)
        if os.path.isdir(seq_dir):
            seqs = sorted([d[:-len("_source_left")] for d in os.listdir(seq_dir)
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
                logger.info(f"Loaded {i + 1}/{total} ({(i + 1) / total * 100:.2f}%) sequences")
            return result

        sample_list = []
        seq_annot = {k: v for k, v in seq_annot.items() if k in seqs}
        print(f"Number of sequences with annotations: {len(seq_annot)}, seqs with annotations: {list(seq_annot.keys())}")

        for seq_name in seq_annot.keys():
            filenames = defaultdict(lambda: defaultdict(list))
            for cam in ['left', 'right']:
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
                        assert len(filenames[k][cam]) == len(filenames['image'][cam]) > 0, framedata.sequence_name

            seq_len = len(filenames['image'][cam])

            fb, fe, fs = frame_sample
            step = fe if fe is not None and fe > 0 else seq_len
            counter = 0

            for ref_idx in range(0, seq_len, step):
                end_idx = min(ref_idx + step, seq_len)  # clamp to actual sequence length
                if end_idx - ref_idx < step:
                    break  # skip incomplete windows at the end
                sample_filenames = defaultdict(lambda: defaultdict(list))
                for cam in ['left', 'right']:
                    for idx in range(ref_idx, end_idx):
                        for k in filenames.keys():
                            sample_filenames[k][cam].append(filenames[k][cam][idx])

                sample_list.append(sample_filenames)
                counter += 1

        self.sample_list = sample_list
        self.lengths = [1 for _ in sample_list]  # each sample is one "item"
        self.seq_keys = list(range(len(sample_list)))
        self.lengths = np.asarray(self.lengths)
        self.cumsum_lengths = [0] + self.lengths.cumsum(-1).tolist()

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

    def _viewpoint_to_K_pixels(self, vp, image_size):
        """Convert a viewpoint annotation dict to a pixel-space 3x3 K matrix."""
        principal_point = torch.tensor(vp["principal_point"], dtype=torch.float)
        focal_length = torch.tensor(vp["focal_length"], dtype=torch.float)
        half_wh = torch.tensor(list(reversed(image_size)), dtype=torch.float) / 2.0
        fmt = vp["intrinsics_format"].lower()
        if fmt == "ndc_norm_image_bounds":
            rescale = half_wh
        elif fmt == "ndc_isotropic":
            rescale = half_wh.min()
        else:
            raise ValueError(f"Unknown intrinsics format: {fmt}")
        pp_px = half_wh - principal_point * rescale
        fl_px = focal_length * rescale
        return np.array([
            [float(fl_px[0]), 0., float(pp_px[0])],
            [0., float(fl_px[1]), float(pp_px[1])],
            [0., 0., 1.],
        ], dtype=np.float32)

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
            depth2disp_scale = 1  # depth2disparity_scale(
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

                    entry_viewpoint = sample["viewpoint"][cam][i]
                    image_size = sample["metadata"][cam][i][1]  # (H, W)
                    scale = 1.0

                    principal_point = torch.tensor(entry_viewpoint["principal_point"], dtype=torch.float)
                    focal_length = torch.tensor(entry_viewpoint["focal_length"], dtype=torch.float)

                    half_image_size_wh_orig = (
                            torch.tensor(list(reversed(image_size)), dtype=torch.float) / 2.0
                    )

                    # first, we convert from the dataset's NDC convention to pixels
                    format = entry_viewpoint["intrinsics_format"]
                    if format.lower() == "ndc_norm_image_bounds":
                        # this is e.g. currently used in CO3D for storing intrinsics
                        rescale = half_image_size_wh_orig
                    elif format.lower() == "ndc_isotropic":
                        rescale = half_image_size_wh_orig.min()
                    else:
                        raise ValueError(f"Unknown intrinsics format: {format}")

                    # principal point and focal length in pixels
                    principal_point_px = half_image_size_wh_orig - principal_point * rescale
                    focal_length_px = focal_length * rescale

                    # now, convert from pixels to PyTorch3D v0.5+ NDC convention
                    # if self.image_height is None or self.image_width is None:
                    out_size = list(reversed(image_size))

                    half_image_size_output = torch.tensor(out_size, dtype=torch.float) / 2.0
                    half_min_image_size_output = half_image_size_output.min()

                    # rescaled principal point and focal length in ndc
                    principal_point = (
                                              half_image_size_output - principal_point_px * scale
                                      ) / half_min_image_size_output
                    focal_length = focal_length_px * scale / half_min_image_size_output

                    viewpoint = {
                        "R": torch.tensor(sample["viewpoint"][cam][i]["R"], dtype=torch.float)[None],
                        "T": torch.tensor(sample["viewpoint"][cam][i]["T"], dtype=torch.float)[None],
                        "K": torch.tensor([
                            [float(focal_length_px[0]), 0, float(principal_point_px[0])],
                            [0, float(focal_length_px[1]), float(principal_point_px[1])],
                            [0, 0, 1.0],
                        ], dtype=torch.float)
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

        idx = idx % len(self.sample_list)
        sample = self.sample_list[idx]

        # Set camera
        cam = 'left'

        # Extract data for selected camera
        ims_left = sample["image"][cam]  # List of file paths
        viewpoints = sample["viewpoint"][cam]  # List of viewpoint dicts
        metadata = sample["metadata"][cam]  # List of [seq_name, size] pairs

        # Extract image dimensions and camera intrinsics
        Hs = np.array([m[1][0] for m in metadata], dtype=np.int32)
        Ws = np.array([m[1][1] for m in metadata], dtype=np.int32)
        Ks = np.stack([self._viewpoint_to_K_pixels(vp, m[1])
                       for vp, m in zip(viewpoints, metadata)])

        # Construct 4x4 extrinsic matrices from R and T
        # vp["R"] is 3x3 rotation, vp["T"] is translation (assuming world-to-camera format)
        RTs_list = []
        ts_list = []
        for vp in viewpoints:
            R = np.array(vp["R"], dtype=np.float32)  # 3x3
            T = np.array(vp["T"], dtype=np.float64)  # 3, or maybe 3x1

            # Ensure T is shape (3, 1)
            if T.ndim == 1:
                T = T.reshape(3, 1)

            # Construct 4x4 extrinsic matrix [R | T; 0 0 0 1]
            RT = np.eye(4, dtype=np.float32)
            RT[:3, :3] = R
            RT[:3, 3:4] = T.astype(np.float32)

            RTs_list.append(RT)
            ts_list.append(T.flatten().astype(np.float64))

        RTs = np.stack(RTs_list)  # (N, 4, 4)
        ts = np.array(ts_list)  # (N, 3)

        # Image resizing and cropping logic
        h, w = self.input_image_res
        xs, ys, ws_list, hs_list, ratios = [], [], [], [], []
        for i in range(len(ims_left)):
            # Calculate aspect-preserving resize with letterbox/padding
            if Hs[i] > Ws[i]:
                ratio_x = w / Ws[i]
                ratio_y = int(ratio_x * Hs[i] + 0.5) / Hs[i]
                ratio = ratio_x
            else:
                ratio_y = h / Hs[i]
                ratio_x = int(ratio_y * Ws[i] + 0.5) / Ws[i]
                ratio = ratio_y

            # Calculate crop offsets
            if h / Hs[i] > w / Ws[i]:
                x, y = int((Ws[i] * ratio_x - w + 0.5) // 2), 0
            else:
                x, y = 0, int((Hs[i] * ratio_y - h + 0.5) // 2)

            # Update camera intrinsics for resize and crop
            Ks[i, 0:1] *= ratio_x
            Ks[i, 1:2] *= ratio_y
            Ks[i, 0, 2] -= x
            Ks[i, 1, 2] -= y

            xs.append(x)
            ys.append(y)
            ws_list.append(int(ratio_x * Ws[i] + 0.5))
            hs_list.append(int(ratio_y * Hs[i] + 0.5))
            Hs[i] = h
            Ws[i] = w
            ratios.append(ratio)

        ratios = np.asarray(ratios, dtype=np.float32)

        # Load and process images at runtime
        imgs = load_dynamic_replica_images(
            ims_left, "", xs, ys, Ws, Hs, hs_list, ws_list
        )
        imgs = imgs * 2 - 1  # [0,1] → [-1,1]
        imgs = np.transpose(imgs, (0, 3, 1, 2))  # NHWC → NCHW

        # Convert pose matrices: World-to-Camera to Camera-to-World
        c2ws = as_numpy_func(affine_inverse)(RTs)

        # Camera alignment
        c2w_avg = None
        if self.align_cameras:
            c2w_avg = average_c2ws(c2ws, align_cameras=False, look_at_center=True)
            c2ws = align_c2ws(c2ws, c2w_avg)
        else:
            c2w_avg = np.eye(4)

        c2ws = as_numpy_func(affine_padding)(c2ws)

        # Pack cameras into camera representation
        from .utils import pack_c2ws_to_cameras
        cameras = pack_c2ws_to_cameras(c2ws, Ks, Hs, Ws)

        # Normalize timestamps
        ts = ts.astype(np.float32)
        ts -= ts.min()

        # Handle novel time sampling for input selection
        fb, _, fs = self.novel_time_frame_sample
        if self.novel_time_sampling:
            # Sample frames with specified stride for input
            input_inds = np.arange(fb, len(imgs), fs)
        else:
            # Use all frames for input
            input_inds = np.arange(len(imgs))

        # All frames for output
        output_inds = np.arange(len(imgs))

        # Prepare batch output
        batch = dotdict()
        batch.rgb_input = imgs[input_inds].astype(np.float32)
        batch.rays_t_un_input = ts[input_inds].astype(np.float32)
        batch.cameras_input = cameras[input_inds].astype(np.float32)
        batch.rgb_output = imgs[output_inds].astype(np.float32)
        batch.rays_t_un_output = ts[output_inds].astype(np.float32)
        batch.cameras_output = cameras[output_inds].astype(np.float32)
        batch.img_name_output = [ims_left[i] for i in output_inds]
        batch.ratios_output = ratios[output_inds].astype(np.float32)
        batch.c2w_avg = c2w_avg

        return batch
