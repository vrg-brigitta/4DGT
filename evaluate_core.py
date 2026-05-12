# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Core evaluation functions for 4DGT inference.

This module provides reusable evaluation functions for computing metrics:
- PSNR on RGB predictions
- LPIPS on RGB predictions
- RMSE on RGB predictions
- Degree error for normal render outputs
- Wall-clock inference time per subsequence and per output frame
"""

import argparse
import math
from typing import Dict, Tuple

import cv2
import lpips
import numpy as np
import torch

from tlod.demo import FourDGTDemo
from tlod.data_loader.utils import load_16big_png_depth


def camera_vector_to_intrinsics(camera: np.ndarray, height: int, width: int) -> Tuple[float, float, float, float]:
    """Extract camera intrinsics from camera vector representation."""
    fovx = float(camera[16])
    fovy = float(camera[17])
    ppx = float(camera[18])
    ppy = float(camera[19])
    fx = width / (2.0 * math.tan(fovx / 2.0))
    fy = height / (2.0 * math.tan(fovy / 2.0))
    cx = width * ppx
    cy = height * ppy
    return fx, fy, cx, cy


def normalize_rgb_tensor(rgb: torch.Tensor) -> torch.Tensor:
    """Normalize RGB tensor from [-1, 1] to [0, 1]."""
    return ((rgb + 1.0) * 0.5).clamp(0.0, 1.0)


def compute_psnr(gt: torch.Tensor, pred: torch.Tensor, eps: float = 1e-8) -> float:
    """Compute PSNR between ground truth and predicted images."""
    mse = torch.mean((gt - pred) ** 2)
    if mse.item() < eps:
        return float("inf")
    return float(10.0 * torch.log10(1.0 / mse))


def compute_rmse(gt: torch.Tensor, pred: torch.Tensor) -> float:
    """Compute RMSE between ground truth and predicted images."""
    mse = torch.mean((gt - pred) ** 2)
    return float(torch.sqrt(mse))


def compute_lpips_score(lpips_model: lpips.LPIPS, gt: torch.Tensor, pred: torch.Tensor) -> float:
    """Compute LPIPS perceptual similarity score."""
    # lpips expects [1, 3, H, W] in [0,1]
    device = next(lpips_model.parameters()).device
    with torch.no_grad():
        score = lpips_model(gt.unsqueeze(0).to(device), pred.unsqueeze(0).to(device))
    return float(score.squeeze().item())


def resize_and_crop_depth(depth: np.ndarray, target_h: int, target_w: int, orig_h: int, orig_w: int) -> np.ndarray:
    """Resize and crop depth map to target resolution."""
    if depth.shape != (orig_h, orig_w):
        depth = cv2.resize(depth, dsize=(orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    if orig_h > orig_w:
        ratio_x = target_w / orig_w
        ratio_y = int(ratio_x * orig_h + 0.5) / orig_h
    else:
        ratio_y = target_h / orig_h
        ratio_x = int(ratio_y * orig_w + 0.5) / orig_w

    if target_h / orig_h > target_w / orig_w:
        x = int((orig_w * ratio_x - target_w + 0.5) // 2)
        y = 0
    else:
        x = 0
        y = int((orig_h * ratio_y - target_h + 0.5) // 2)

    ws = int(round(ratio_x * orig_w))
    hs = int(round(ratio_y * orig_h))
    depth_resized = cv2.resize(depth, dsize=(ws, hs), interpolation=cv2.INTER_NEAREST)
    cropped = depth_resized[y : y + target_h, x : x + target_w]
    if cropped.shape != (target_h, target_w):
        raise ValueError(
            f"Depth transformation failed: expected ({target_h},{target_w}) got {cropped.shape}"
        )
    return cropped


def compute_normals_from_depth(depth: np.ndarray, intrinsics: Tuple[float, float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute surface normals from depth map using central differences."""
    fx, fy, cx, cy = intrinsics
    h, w = depth.shape
    i_coords, j_coords = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    x = (i_coords - cx) * depth / fx
    y = (j_coords - cy) * depth / fy
    z = depth
    points = np.stack([x, y, z], axis=-1)

    # central differences for normals, valid inside border pixels
    p_center = points[1:-1, 1:-1]
    p_left = points[1:-1, :-2]
    p_right = points[1:-1, 2:]
    p_up = points[:-2, 1:-1]
    p_down = points[2:, 1:-1]
    dpx = p_right - p_left
    dpy = p_down - p_up
    normals = np.cross(dpx, dpy, axis=-1)

    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normals = normals / norms

    valid = (
        depth[1:-1, 1:-1] > 0
    ) & (
        depth[1:-1, :-2] > 0
    ) & (
        depth[1:-1, 2:] > 0
    ) & (
        depth[:-2, 1:-1] > 0
    ) & (
        depth[2:, 1:-1] > 0
    )

    full_normals = np.zeros((h, w, 3), dtype=np.float32)
    full_mask = np.zeros((h, w), dtype=bool)
    full_normals[1:-1, 1:-1] = normals
    full_mask[1:-1, 1:-1] = valid
    return full_normals, full_mask


def compute_normal_deg_error(pred_normals: torch.Tensor, gt_normals: np.ndarray, valid_mask: np.ndarray) -> float:
    """Compute degree error between predicted and ground truth normals."""
    pred = pred_normals.cpu().numpy()
    if pred.ndim == 5:
        pred = pred[0]
    if pred.ndim == 4:
        # expected shape [3, H, W]
        if pred.shape != gt_normals.shape:
            raise ValueError(f"Predicted normals shape {pred.shape} does not match GT normals {gt_normals.shape}")
        pred = pred.transpose(1, 2, 0)[None]
    elif pred.ndim == 3:
        pred = pred.transpose(1, 2, 0)[None]
    else:
        raise ValueError(f"Unsupported predicted normal tensor shape {pred.shape}")

    pred_norm = pred / np.linalg.norm(pred, axis=-1, keepdims=True).clip(1e-8)
    gt_norm = gt_normals / np.linalg.norm(gt_normals, axis=-1, keepdims=True).clip(1e-8)
    dot = np.sum(pred_norm[0] * gt_norm, axis=-1)
    dot = np.clip(dot, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(dot))
    if valid_mask is not None:
        angle_deg = angle_deg[valid_mask]
    return float(np.nanmean(angle_deg)) if angle_deg.size else float("nan")


def _z_to_euclidean(
    z_depth: np.ndarray,
    intrinsics: Tuple[float, float, float, float],
) -> np.ndarray:
    """Convert z-depth (along camera axis) to euclidean (ray) depth.

    Dynamic Replica stores geometric/euclidean depth: the distance from the
    camera centre to the surface point along the ray.  The renderer produces
    z-depth: the perpendicular distance to the image plane.  The conversion is:

        euclidean = z * sqrt(((u - cx)/fx)^2 + ((v - cy)/fy)^2 + 1)

    Args:
        z_depth: (H, W) array of z-depths in metres.
        intrinsics: (fx, fy, cx, cy) in pixels.

    Returns:
        (H, W) array of euclidean depths in metres.
    """
    fx, fy, cx, cy = intrinsics
    h, w = z_depth.shape
    us, vs = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    ray_lengths = np.sqrt(((us - cx) / fx) ** 2 + ((vs - cy) / fy) ** 2 + 1.0)
    return z_depth * ray_lengths


def evaluate_subsequence(
    demo: FourDGTDemo,
    batch: object,
    lpips_model: lpips.LPIPS,
    depth_annotation_map: Dict[str, Dict[str, object]],
    args: argparse.Namespace,
) -> Dict[str, float]:
    """Evaluate a single subsequence and compute metrics.
    
    Args:
        demo: FourDGTDemo inference engine
        batch: Batch data with rgb_input, cameras_input, timestamps_input, etc.
        lpips_model: LPIPS model for perceptual similarity
        depth_annotation_map: Mapping from image paths to depth annotations
        args: Arguments containing input_frames, subsequence_length, resolution, etc.
    
    Returns:
        Dictionary with metrics: psnr, rmse, lpips, deg, time_s, time_per_frame_s
    """
    import time

    stride = getattr(args, 'novel_time_stride', 2)
    n_total = batch.rgb_output.shape[1]  # e.g. 128

    # Input frames: every `stride`-th frame (dataset already did this subsampling,
    # so batch.rgb_input contains only those frames).
    # Output: ALL n_total frames rendered by the model.
    # Test frames: the frames NOT given as input — used for metric computation.
    input_frame_inds = set(range(0, n_total, stride))         # {0, 2, 4, ..., 126}
    test_frame_inds  = [i for i in range(n_total)
                        if i not in input_frame_inds]         # [1, 3, 5, ..., 127]

    rgb_input         = batch.rgb_input                        # [B, n_total/stride, 3, H, W]
    cameras_input     = batch.cameras_input                    # [B, n_total/stride, 20]
    timestamps_input  = batch.rays_t_un_input                  # [B, n_total/stride]

    cameras_output    = batch.cameras_output                   # [B, n_total, 20]
    timestamps_output = batch.rays_t_un_output                 # [B, n_total]

    # Ground truth and names only for test frames
    rgb_gt   = batch.rgb_output[:, test_frame_inds]           # [B, n_test, 3, H, W]
    img_names = [batch.img_name_output[i] for i in test_frame_inds]
    cameras_test = cameras_output[:, test_frame_inds]         # [B, n_test, 20]

    start_time = time.perf_counter()
    output = demo.run_inference(
        images_input=rgb_input,
        cameras_input=cameras_input,
        timestamps_input=timestamps_input,
        cameras_output=cameras_output,        # render ALL frames
        timestamps_output=timestamps_output,
        render_mode="RGB+ED",
        sequential_render=False,
        save_config=None,
        batch_idx=None,
    )
    elapsed = time.perf_counter() - start_time

    # Select test-frame predictions from the full output (model rendered all frames).
    rgb_pred_all = normalize_rgb_tensor(output["rgb"][0].detach())  # [n_total, 3, H, W]
    rgb_pred      = rgb_pred_all[test_frame_inds].cpu()             # [n_test,  3, H, W]
    rgb_gt_tensor = normalize_rgb_tensor(rgb_gt[0]).cpu()           # [n_test,  3, H, W]

    batch_psnr = 0.0
    batch_lpips = 0.0
    frame_count = rgb_pred.shape[0]
    for frame_idx in range(frame_count):
        batch_psnr += compute_psnr(rgb_gt_tensor[frame_idx], rgb_pred[frame_idx])
        batch_lpips += compute_lpips_score(
            lpips_model,
            rgb_gt_tensor[frame_idx],
            rgb_pred[frame_idx],
        )

    results = {
        "psnr": batch_psnr / frame_count,
        "lpips": batch_lpips / frame_count,
        "time_s": elapsed,
        "time_per_frame_s": elapsed / frame_count,
    }

    # Depth RMSE (metric, in metres) and normal degree error.
    # Both require GT depth from the annotation map; load it once per frame and
    # use it for both metrics so we don't read each depth file twice.
    depth_rmses: list = []
    normals_deg: list = []

    # Select test frames from depth / normal outputs.
    depth_pred_all = output["depth"][0].detach().cpu() if "depth" in output else None  # [n_total, 1, H, W]
    depth_pred     = depth_pred_all[test_frame_inds] if depth_pred_all is not None else None
    pred_normals_all = output["normal"].detach().cpu() if "normal" in output else None  # [B, n_total, 3, H, W]
    pred_normals     = pred_normals_all[:, test_frame_inds] if pred_normals_all is not None else None

    if depth_annotation_map and (depth_pred is not None or pred_normals is not None):
        for idx, img_name in enumerate(img_names):
            img_name_str = img_name[0] if isinstance(img_name, list) else img_name
            depth_info = depth_annotation_map.get(img_name_str)
            if depth_info is None:
                continue

            depth_path = depth_info["depth_path"]
            orig_h = depth_info["height"]
            orig_w = depth_info["width"]
            depth_gt = load_16big_png_depth(depth_path).astype(np.float32)
            depth_gt_aligned = resize_and_crop_depth(depth_gt, args.resolution, args.resolution, orig_h, orig_w)

            # Intrinsics needed for both depth conversion and normal estimation.
            intrinsics = camera_vector_to_intrinsics(cameras_test[0, idx], args.resolution, args.resolution)

            # --- Depth RMSE (metres, valid pixels only) ---
            # GT depth is euclidean (ray) depth (.geometric.png convention in Dynamic Replica).
            # Renderer output is z-depth (along camera axis).  Convert z -> euclidean so both
            # are in the same space before computing RMSE.
            if depth_pred is not None:
                depth_pred_frame = depth_pred[idx, 0].numpy()  # [H, W]  z-depth
                depth_pred_frame = _z_to_euclidean(depth_pred_frame, intrinsics)
                valid = depth_gt_aligned > 0
                if valid.sum() > 0:
                    rmse = float(np.sqrt(np.mean((depth_pred_frame[valid] - depth_gt_aligned[valid]) ** 2)))
                    if not math.isnan(rmse):
                        depth_rmses.append(rmse)

            # --- Normal degree error ---
            # Normals are derived from the (euclidean) GT depth, so use depth_gt_aligned directly.
            if pred_normals is not None:
                gt_normals, valid_mask = compute_normals_from_depth(depth_gt_aligned, intrinsics)
                pred_normal_frame = pred_normals[0, idx]
                deg = compute_normal_deg_error(pred_normal_frame, gt_normals, valid_mask)
                if not math.isnan(deg):
                    normals_deg.append(deg)

    results["rmse"] = float(np.nanmean(depth_rmses)) if depth_rmses else float("nan")
    results["deg"] = float(np.nanmean(normals_deg)) if normals_deg else float("nan")

    return results


def format_results(results: Dict[str, list], args: argparse.Namespace) -> str:
    """Format evaluation results for display."""
    stride = getattr(args, "novel_time_stride", 2)
    n_total = args.subsequence_length
    n_input = len(range(0, n_total, stride))
    n_test = n_total - n_input

    lines = ["Evaluation results:"]
    lines.append(f"  subsequence length: {n_total}")
    lines.append(f"  novel_time_stride:  {stride}  ({n_input} input / {n_test} test frames)")
    lines.append(f"  evaluation resolution: {args.resolution}x{args.resolution}")
    lines.append(f"  evaluated subsequences: {len(results['psnr'])}")
    fmt = {"psnr": ".3f", "rmse": ".4f", "lpips": ".4f", "deg": ".3f",
           "time_s": ".3f", "time_per_frame_s": ".3f"}
    for metric in ["psnr", "rmse", "lpips", "deg", "time_s", "time_per_frame_s"]:
        values = [v for v in results[metric] if not math.isnan(v)]
        if not values:
            lines.append(f"  {metric}: N/A")
            continue
        f = fmt.get(metric, ".4f")
        lines.append(f"  {metric}: avg={np.mean(values):{f}} std={np.std(values):{f}} "
                     f"min={np.min(values):{f}} max={np.max(values):{f}}")
    return "\n".join(lines)
