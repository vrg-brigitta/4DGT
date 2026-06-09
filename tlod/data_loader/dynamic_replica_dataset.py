# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Dynamic Replica dataset loader for 4DGT (mono).
Adapted from mvaria_dataset.py. Loads ONLY the left camera (4DGT is mono).

Conventions enforced (matching tlod/data_loader/utils.load_camera_poses):
  - Hs, Ws : np.int32, shape (F,)
  - Ks     : np.float32, shape (F, 3, 3)            -- pixel-space
  - RTs    : np.float32, shape (F, 4, 4)            -- world-to-camera (OpenCV)
  - ts     : np.float64, shape (F,)                 -- seconds
  - ims    : np.ndarray[str], shape (F,)            -- ABSOLUTE image paths
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

from ..easyvolcap.utils.cam_utils import align_c2ws, average_c2ws
from ..easyvolcap.utils.console_utils import dotdict, logger, magenta
from ..easyvolcap.utils.data_utils import as_numpy_func
from ..easyvolcap.utils.math_utils import affine_inverse, affine_padding
from .utils import pack_c2ws_to_cameras


# ---------------------------------------------------------------------------
# Annotation dataclasses (exact copy of Dynamic Replica's frame_annotations)
# ---------------------------------------------------------------------------
@dataclass
class FileAnnotation:
    path: str
    size: Optional[Tuple[int, int]] = None  # (H, W)


@dataclass
class DynamicReplicaFrameAnnotation:
    sequence_name: str
    camera_name: Optional[str] = None
    image: Optional[FileAnnotation] = None
    depth: Optional[FileAnnotation] = None
    mask: Optional[FileAnnotation] = None
    viewpoint: Any = None
    frame_number: Optional[int] = None
    frame_timestamp: Optional[float] = None


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
        kwargs = {k: _load_dataclass_value(v, field_types[k])
                  for k, v in value.items() if k in field_types}
        return expected_type(**kwargs)

    if isinstance(value, expected_type):
        return value
    if expected_type in (str, int, float, bool):
        return expected_type(value)
    return value


def load_dataclass(file_obj: Any, expected_type: Any) -> Any:
    return _load_dataclass_value(json.load(file_obj), expected_type)


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------
def _viewpoint_to_K_pixels(vp: dict, image_size: Tuple[int, int]) -> np.ndarray:
    """Convert a Dynamic Replica NDC viewpoint to a pixel-space 3x3 K matrix.

    image_size: (H, W) as stored in the annotation.
    """
    principal_point = torch.tensor(vp["principal_point"], dtype=torch.float)
    focal_length = torch.tensor(vp["focal_length"], dtype=torch.float)
    half_wh = torch.tensor(list(reversed(image_size)), dtype=torch.float) / 2.0  # (W/2, H/2)
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
        [float(fl_px[0]), 0.0,             float(pp_px[0])],
        [0.0,             float(fl_px[1]), float(pp_px[1])],
        [0.0,             0.0,             1.0],
    ], dtype=np.float32)


def _viewpoint_to_RT_w2c(vp: dict) -> np.ndarray:
    """Dynamic Replica stores PyTorch3D-style cameras (row-vector convention):
        x_cam = x_world @ R + T
    OpenCV w2c uses column-vector:
        x_cam = R_cv @ x_world + t_cv
    => R_cv = R_p3d.T, t_cv = T_p3d
    """
    R_p3d = np.asarray(vp["R"], dtype=np.float32)
    T_p3d = np.asarray(vp["T"], dtype=np.float32).reshape(3)
    RT = np.eye(4, dtype=np.float32)
    RT[:3, :3] = R_p3d.T
    RT[:3, 3] = T_p3d
    return RT


# ---------------------------------------------------------------------------
# Image loader (mvaria-style: read -> resize -> crop -> [-1, 1])
# ---------------------------------------------------------------------------
def load_dynamic_replica_images(image_files, xs, ys, Ws, Hs, hs, ws):
    imgs = []
    for i, img_path in enumerate(image_files):
        img = iio.imread(img_path)
        if img.dtype != np.float32:
            img = img.astype(np.float32) / 255.0
        if img.ndim == 2:
            img = np.tile(img[..., None], (1, 1, 3))
        else:
            img = img[..., :3]
        img = cv2.resize(img, dsize=(int(ws[i]), int(hs[i])))
        img = img[ys[i]:ys[i] + Hs[i], xs[i]:xs[i] + Ws[i]]
        assert img.shape[:2] == (int(Hs[i]), int(Ws[i])), (
            f"Frame {i}: crop produced {img.shape[:2]}, expected ({Hs[i]}, {Ws[i]}). "
            f"resize=({ws[i]},{hs[i]})  crop_offset=({xs[i]},{ys[i]})"
        )
        imgs.append(img)
    return np.stack(imgs)  # (F, H, W, 3) float32 in [0,1]


# ===========================================================================
# Dataset
# ===========================================================================
class DynamicReplicaDataset(Dataset):
    """
    4DGT-compatible loader for Dynamic Replica (mono / left camera).

    Returns the same batch structure as AriaDataset:
        rgb_input, rgb_output            : (F, 3, H, W)
        rays_t_un_input, rays_t_un_output: (F,)
        cameras_input, cameras_output    : (F, 20)
        img_name_output                  : list[str]
        ratios_output                    : (F,)
        c2w_avg                          : (4, 4)
    """

    def __init__(
        self,
        mode: str = "TEST",
        data_root: str = "./data/dynamic_replica_data",
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
        fps: float = 30.0,
        force_reload: bool = False,
        **kwargs,
    ):
        super().__init__()

        # Accept VAL / VALIDATION / VALID as aliases of TEST (Dynamic Replica only ships test/valid)
        mode_upper = mode.upper()
        if mode_upper in ("VAL", "VALIDATION", "VALID"):
            split = "valid"
        elif mode_upper == "TEST":
            split = "test"
        elif mode_upper == "TRAIN":
            split = "train"
        else:
            raise ValueError(f"Unknown mode {mode}")
        # The dataset is currently inference-only for 4DGT (no training augmentation).
        # We still allow loading TRAIN annotations if you want to evaluate them.

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
        self.sample_interval = max(1, int(sample_interval))
        self.novel_view_interp_input = novel_view_interp_input
        self.novel_view_timestamps = novel_view_timestamps
        self.novel_view_spiral_window = novel_view_spiral_window
        self.loaded_to_seconds = loaded_to_seconds
        self.loaded_to_meters = loaded_to_meters
        self.fps = fps

        # ------------------------------------------------------------------
        # Load frame annotations (a single .jgz file per split)
        # ------------------------------------------------------------------
        annot_path = os.path.join(data_root, split, f"frame_annotations_{split}.jgz")
        if not os.path.isfile(annot_path):
            raise FileNotFoundError(f"Cannot find {annot_path}")

        with gzip.open(annot_path, "rt", encoding="utf8") as zipfile:
            frame_annots_list = load_dataclass(zipfile, List[DynamicReplicaFrameAnnotation])

        # Group by sequence_name -> camera_name -> [frame...]
        seq_annot = defaultdict(lambda: defaultdict(list))
        for f in frame_annots_list:
            seq_annot[f.sequence_name][f.camera_name].append(f)

        # Sort frames within each (seq, cam) by frame_number for determinism
        for seq, cams in seq_annot.items():
            for cam in cams:
                cams[cam].sort(key=lambda x: (x.frame_number if x.frame_number is not None
                                              else x.image.path))

        # Apply seq_sample to the discovered sequences
        all_seqs = sorted(seq_annot.keys())
        b, e, s = seq_sample
        seqs = all_seqs[b:e:s]
        logger.info(f"Number of sequences used for {magenta(split.upper())}: "
                    f"{len(seqs)} / {len(all_seqs)} -> {seqs}")

        # ------------------------------------------------------------------
        # Per-sequence pre-loading (mvaria-style: arrays in memory)
        # ------------------------------------------------------------------
        self.seqs = dotdict()
        self.lengths = []

        cam = "left"  # 4DGT is mono
        fb, fe, fs = frame_sample

        for key in seqs:
            frames = seq_annot[key].get(cam, [])
            if not frames:
                logger.warn(f"Sequence {key} has no '{cam}' camera, skipping")
                continue

            # Sub-sample frames
            fe_eff = len(frames) if fe is None else min(fe, len(frames))
            frames = frames[fb:fe_eff:fs]
            F = len(frames)
            if F == 0:
                logger.warn(f"Sequence {key} has 0 frames after frame_sample, skipping")
                continue

            ims_paths, Hs, Ws, Ks, RTs = [], [], [], [], []
            for fr in frames:
                im_path = os.path.join(data_root, split, fr.image.path)
                if not os.path.isfile(im_path):
                    raise FileNotFoundError(im_path)
                H, W = int(fr.image.size[0]), int(fr.image.size[1])
                vp = fr.viewpoint if isinstance(fr.viewpoint, dict) else {
                    "R": fr.viewpoint.R,
                    "T": fr.viewpoint.T,
                    "focal_length": fr.viewpoint.focal_length,
                    "principal_point": fr.viewpoint.principal_point,
                    "intrinsics_format": fr.viewpoint.intrinsics_format,
                }
                K = _viewpoint_to_K_pixels(vp, (H, W))
                RT = _viewpoint_to_RT_w2c(vp)
                RT[:3, 3] /= loaded_to_meters

                ims_paths.append(im_path)
                Hs.append(H); Ws.append(W); Ks.append(K); RTs.append(RT)

            ims_paths = np.asarray(ims_paths)
            Hs = np.asarray(Hs, dtype=np.int32)
            Ws = np.asarray(Ws, dtype=np.int32)
            Ks = np.stack(Ks).astype(np.float32)
            RTs = np.stack(RTs).astype(np.float32)

            # Dynamic Replica frames are uniformly sampled; construct a synthetic
            # timestamp from the frame index so 4DGT's time embedding works.
            ts = (np.arange(F, dtype=np.float64)) / max(self.fps, 1e-6) / loaded_to_seconds

            # Use seq_data_roots[0] purely as a key (typically "")
            sdr = self.seq_data_roots[0]
            self.seqs[key] = dotdict()
            self.seqs[key][sdr] = dotdict()
            self.seqs[key][sdr].ims = ims_paths
            self.seqs[key][sdr].Hs = Hs
            self.seqs[key][sdr].Ws = Ws
            self.seqs[key][sdr].Ks = Ks
            self.seqs[key][sdr].RTs = RTs
            self.seqs[key][sdr].ts = ts

            # Number of windows of length batch_image_num that fit in this sequence.
            n_windows = max(0, (F - self.batch_image_num) // self.sample_interval + 1)
            self.lengths.append(n_windows)

        self.seq_keys = list(self.seqs.keys())
        self.lengths = np.asarray(self.lengths, dtype=np.int64)
        self.cumsum_lengths = [0] + self.lengths.cumsum(-1).tolist()

    # ----- Dataset API ---------------------------------------------------
    def __len__(self):
        return int(self.cumsum_lengths[-1]) if self.cumsum_lengths else 0

    def __getitem__(self, idx: int):  # noqa: C901
        if len(self) == 0:
            raise IndexError("Dataset is empty")
        idx = idx % self.cumsum_lengths[-1]
        seq_idx = np.searchsorted(self.cumsum_lengths, idx, side="right").item() - 1
        key = self.seq_keys[seq_idx]
        sub_idx = idx - self.cumsum_lengths[seq_idx]
        global_abs_start_ind = sub_idx * self.sample_interval

        sdr = self.seq_data_roots[0]
        seq = self.seqs[key][sdr]
        curr_len = len(seq.ts)

        rgb_input, rgb_output = [], []
        rays_t_un_input, rays_t_un_output = [], []
        cameras_input, cameras_output = [], []
        img_name_output, ratios_output = [], []
        c2w_avg = None

        # 4DGT mono: a single seq_data_root, treated as "input" and "output".
        target_num_images = self.batch_image_num
        abs_start_ind = max(0, min(global_abs_start_ind, curr_len - target_num_images))
        abs_end_ind = abs_start_ind + target_num_images
        sample_inds = np.arange(abs_start_ind, abs_end_ind)

        ims = seq.ims[sample_inds].copy()
        Hs = seq.Hs[sample_inds].copy()
        Ws = seq.Ws[sample_inds].copy()
        Ks = seq.Ks[sample_inds].copy()
        RTs = seq.RTs[sample_inds].copy()
        ts = seq.ts[sample_inds].copy()

        ts -= ts.min()
        ts = ts.astype(np.float32)

        # Letterbox/resize to input_image_res, then center-crop to (h, w) – mvaria-style.
        h, w = self.input_image_res
        xs, ys, ws_list, hs_list, ratios = [], [], [], [], []
        for i in range(len(ims)):
            if Hs[i] > Ws[i]:
                ratio_x = w / Ws[i]
                ratio_y = int(ratio_x * Hs[i] + 0.5) / Hs[i]
                ratio = ratio_x
            else:
                ratio_y = h / Hs[i]
                ratio_x = int(ratio_y * Ws[i] + 0.5) / Ws[i]
                ratio = ratio_y
            if h / Hs[i] > w / Ws[i]:
                x = int((Ws[i] * ratio_x - w + 0.5) // 2); y = 0
            else:
                x = 0; y = int((Hs[i] * ratio_y - h + 0.5) // 2)
            Ks[i, 0:1] *= ratio_x
            Ks[i, 1:2] *= ratio_y
            Ks[i, 0, 2] -= x
            Ks[i, 1, 2] -= y
            xs.append(x); ys.append(y)
            ws_list.append(int(ratio_x * Ws[i] + 0.5))
            hs_list.append(int(ratio_y * Hs[i] + 0.5))
            Hs[i] = h; Ws[i] = w
            ratios.append(ratio)
        ratios = np.asarray(ratios, dtype=np.float32)

        imgs = load_dynamic_replica_images(ims, xs, ys, Ws, Hs, hs_list, ws_list)
        imgs = imgs * 2.0 - 1.0                      # [-1, 1]
        imgs = np.transpose(imgs, (0, 3, 1, 2))      # NHWC -> NCHW

        c2ws = as_numpy_func(affine_inverse)(RTs)
        if self.align_cameras:
            c2w_avg = average_c2ws(c2ws, align_cameras=False, look_at_center=True)
            c2ws = align_c2ws(c2ws, c2w_avg)
        c2ws = as_numpy_func(affine_padding)(c2ws)
        cameras = pack_c2ws_to_cameras(c2ws, Ks, Hs, Ws)

        # Input / output index selection
        if self.novel_time_sampling:
            fb, _, fs = self.novel_time_frame_sample
            input_inds = np.arange(fb, len(imgs), fs)
        else:
            input_inds = np.arange(len(imgs))
        output_inds = np.arange(len(imgs))

        rgb_input.append(imgs[input_inds])
        cameras_input.append(cameras[input_inds])
        rays_t_un_input.append(ts[input_inds])
        rgb_output.append(imgs[output_inds])
        cameras_output.append(cameras[output_inds])
        rays_t_un_output.append(ts[output_inds])
        img_name_output.append(np.asarray([ims[i] for i in output_inds]))
        ratios_output.append(ratios[output_inds])

        batch = dotdict()
        batch.rgb_input = np.concatenate(rgb_input).astype(np.float32)
        batch.rays_t_un_input = np.concatenate(rays_t_un_input).astype(np.float32)
        batch.cameras_input = np.concatenate(cameras_input).astype(np.float32)
        batch.rgb_output = np.concatenate(rgb_output).astype(np.float32)
        batch.rays_t_un_output = np.concatenate(rays_t_un_output).astype(np.float32)
        batch.cameras_output = np.concatenate(cameras_output).astype(np.float32)
        batch.img_name_output = np.concatenate(img_name_output).tolist()
        batch.ratios_output = np.concatenate(ratios_output).astype(np.float32)
        batch.c2w_avg = c2w_avg if c2w_avg is not None else np.eye(4, dtype=np.float32)
        return batch
