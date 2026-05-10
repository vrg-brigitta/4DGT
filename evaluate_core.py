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
    with torch.no_grad():
        score = lpips_model(gt.unsqueeze(0).cpu(), pred.unsqueeze(0).cpu())
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
    
    input_frames = args.input_frames
    output_frames = args.subsequence_length - args.input_frames
    rgb_input = batch.rgb_input[:, :input_frames]
    cameras_input = batch.cameras_input[:, :input_frames]
    timestamps_input = batch.rays_t_un_input[:, :input_frames]
    rgb_gt = batch.rgb_output[:, input_frames:]
    cameras_output = batch.cameras_output[:, input_frames:]
    timestamps_output = batch.rays_t_un_output[:, input_frames:]
    img_names = batch.img_name_output[input_frames:]
    ratios = batch.ratios_output[:, input_frames:]

    start_time = time.perf_counter()
    output = demo.run_inference(
        images_input=rgb_input,
        cameras_input=cameras_input,
        timestamps_input=timestamps_input,
        cameras_output=cameras_output,
        timestamps_output=timestamps_output,
        render_mode="RGB+ED",
        sequential_render=False,
        save_config=None,
        batch_idx=None,
    )
    elapsed = time.perf_counter() - start_time

    rgb_pred = normalize_rgb_tensor(output["rgb"][0].detach())
    rgb_gt_tensor = normalize_rgb_tensor(rgb_gt[0])
    rgb_pred = rgb_pred.cpu()
    rgb_gt_tensor = rgb_gt_tensor.cpu()

    batch_psnr = 0.0
    batch_rmse = 0.0
    batch_lpips = 0.0
    frame_count = rgb_pred.shape[0]
    for frame_idx in range(frame_count):
        batch_psnr += compute_psnr(rgb_gt_tensor[frame_idx], rgb_pred[frame_idx])
        batch_rmse += compute_rmse(rgb_gt_tensor[frame_idx], rgb_pred[frame_idx])
        batch_lpips += compute_lpips_score(
            lpips_model,
            rgb_gt_tensor[frame_idx],
            rgb_pred[frame_idx],
        )

    results = {
        "psnr": batch_psnr / frame_count,
        "rmse": batch_rmse / frame_count,
        "lpips": batch_lpips / frame_count,
        "time_s": elapsed,
        "time_per_frame_s": elapsed / frame_count,
    }

    if "normal" in output and depth_annotation_map:
        pred_normals = output["normal"].detach().cpu()
        normals_deg = []
        for idx, img_name in enumerate(img_names):
            depth_info = depth_annotation_map.get(img_name)
            if depth_info is None:
                continue
            depth_path = depth_info["depth_path"]
            orig_h = depth_info["height"]
            orig_w = depth_info["width"]
            depth = load_16big_png_depth(depth_path).astype(np.float32)
            depth_aligned = resize_and_crop_depth(depth, args.resolution, args.resolution, orig_h, orig_w)
            intrinsics = camera_vector_to_intrinsics(cameras_output[0, idx], args.resolution, args.resolution)
            gt_normals, valid_mask = compute_normals_from_depth(depth_aligned, intrinsics)
            pred_normal_frame = pred_normals[0, idx]
            deg = compute_normal_deg_error(pred_normal_frame, gt_normals, valid_mask)
            if not math.isnan(deg):
                normals_deg.append(deg)
        results["deg"] = float(np.nanmean(normals_deg)) if normals_deg else float("nan")
    else:
        results["deg"] = float("nan")

    return results


def format_results(results: Dict[str, list], args: argparse.Namespace) -> str:
    """Format evaluation results for display."""
    lines = ["Evaluation results:"]
    lines.append(f"  subsequence length: {args.subsequence_length}")
    lines.append(f"  input frames: {args.input_frames}")
    lines.append(f"  evaluation resolution: {args.resolution}x{args.resolution}")
    lines.append(f"  evaluated subsequences: {len(results['psnr'])}")
    for metric in ["psnr", "rmse", "lpips", "deg", "time_s", "time_per_frame_s"]:
        values = [v for v in results[metric] if not math.isnan(v)]
        if not values:
            lines.append(f"  {metric}: N/A")
            continue
        lines.append(f"  {metric}: avg={np.mean(values):.4f} min={np.min(values):.4f} max={np.max(values):.4f}")
    return "\n".join(lines)
