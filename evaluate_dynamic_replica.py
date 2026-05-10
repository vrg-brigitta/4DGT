#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Evaluate 4DGT inference on Dynamic Replica dataset using 128-frame subsequences.

This script evaluates model performance on monocular Dynamic Replica videos,
using 128-frame subsequences with 64 frames as input and 64 for testing.
Images are resized to 504x504 by default for controlled comparison.

Metrics computed:
- PSNR on RGB predictions
- LPIPS on RGB predictions
- RMSE on RGB predictions
- Degree error for normal render outputs (using ground truth depth)
- Wall-clock inference time
"""

import argparse
import gzip
import os
import time
from typing import Dict, List

import lpips
import torch
from torch.utils.data import DataLoader

from evaluate_core import (
    evaluate_subsequence,
    format_results,
)
from tlod.demo import FourDGTDemo
from tlod.data_loader.dynamic_replica_dataset import (
    DynamicReplicaDataset,
    DynamicReplicaFrameAnnotation,
    load_dataclass,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 4DGT on Dynamic Replica dataset")
    parser.add_argument("--data-root", required=True, help="Root folder for Dynamic Replica data")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained 4DGT checkpoint")
    parser.add_argument("--config", required=True, help="Path to the model config file")
    parser.add_argument("--mode", default="test", choices=["test", "valid", "train"], help="Dataset split")
    parser.add_argument("--device", default="cuda", help="Compute device")
    parser.add_argument("--resolution", type=int, default=504, help="Target image height and width for evaluation")
    parser.add_argument("--subsequence-length", type=int, default=128, help="Frame subsequence length")
    parser.add_argument("--input-frames", type=int, default=64, help="Number of input frames used for conditioning")
    parser.add_argument("--sample-interval", type=int, default=128, help="Window stride for non-overlapping subsequences")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--max-subsequences", type=int, default=None, help="Maximum number of subsequences to evaluate")
    parser.add_argument("--save-dir", default=None, help="Optional output directory for evaluation summaries")
    return parser.parse_args()


def build_dataset(args: argparse.Namespace) -> DynamicReplicaDataset:
    """Build Dynamic Replica dataset."""
    return DynamicReplicaDataset(
        mode=args.mode,
        data_root=args.data_root,
        input_image_res=(args.resolution, args.resolution),
        input_image_num=args.subsequence_length,
        output_image_res=(args.resolution, args.resolution),
        output_image_num=args.subsequence_length,
        seq_sample=(0, None, 1),
        frame_sample=(0, None, 1),
        view_sample=(0, 1, 1),
        sample_interval=max(1, args.sample_interval),
        novel_time_sampling=False,
        novel_time_frame_sample=(0, None, 1),
        novel_view_interp_input=False,
        novel_view_timestamps=(),
        novel_view_spiral_window=32,
        loaded_to_seconds=1.0,
        loaded_to_meters=1.0,
        fps=30.0,
        force_reload=False,
    )


def build_dataloader(dataset: DynamicReplicaDataset, num_workers: int) -> DataLoader:
    """Build DataLoader for Dynamic Replica dataset."""
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=num_workers > 0,
    )


def load_depth_annotation_map(data_root: str, mode: str) -> Dict[str, Dict[str, object]]:
    """Load depth annotations from Dynamic Replica dataset."""
    annotation_path = os.path.join(data_root, mode, f"frame_annotations_{mode}.jgz")
    if not os.path.isfile(annotation_path):
        return {}

    with gzip.open(annotation_path, "rt", encoding="utf8") as f:
        frames = load_dataclass(f, List[DynamicReplicaFrameAnnotation])

    mapping: Dict[str, Dict[str, object]] = {}
    for fr in frames:
        if fr.image is None or fr.depth is None:
            continue
        image_path = os.path.join(data_root, mode, fr.image.path)
        depth_path = os.path.join(data_root, mode, fr.depth.path)
        mapping[image_path] = {
            "depth_path": depth_path,
            "height": int(fr.image.size[0]),
            "width": int(fr.image.size[1]),
        }
    return mapping


def main() -> None:
    args = parse_args()
    if args.input_frames >= args.subsequence_length:
        raise ValueError("input_frames must be smaller than subsequence_length")

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    torch.set_grad_enabled(False)

    dataset = build_dataset(args)
    dataloader = build_dataloader(dataset, args.num_workers)
    annotation_map = load_depth_annotation_map(args.data_root, args.mode)

    demo = FourDGTDemo(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=device,
        output_dir=args.save_dir or "outputs",
        fp16=False,
        fp32=False,
    )

    lpips_model = lpips.LPIPS(net="vgg").to(device)
    lpips_model.eval()

    results: Dict[str, List[float]] = {
        "psnr": [],
        "rmse": [],
        "lpips": [],
        "deg": [],
        "time_s": [],
        "time_per_frame_s": [],
    }

    subsequence_count = 0
    start_eval = time.perf_counter()

    for batch_idx, batch in enumerate(dataloader):
        if args.max_subsequences is not None and subsequence_count >= args.max_subsequences:
            break
        if isinstance(batch, list):
            batch = batch[0]

        if len(batch.rgb_input.shape) < 5:
            raise ValueError("Batch data must include 5D rgb tensors")
        if batch.rgb_input.shape[1] != args.subsequence_length:
            continue

        metrics = evaluate_subsequence(demo, batch, lpips_model, annotation_map, args)
        for key, value in metrics.items():
            results[key].append(value)

        subsequence_count += 1
        print(
            f"Subsequence {subsequence_count}: PSNR={metrics['psnr']:.3f}, RMSE={metrics['rmse']:.4f}, "
            f"LPIPS={metrics['lpips']:.4f}, Deg={metrics['deg']:.3f}, time={metrics['time_s']:.3f}s"
        )

    total_eval_time = time.perf_counter() - start_eval
    summary = format_results(results, args)
    print("\n" + summary)
    print(f"Total evaluation wall time: {total_eval_time:.3f}s")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        report_path = os.path.join(args.save_dir, "evaluation_report_dynamic_replica.txt")
        with open(report_path, "w", encoding="utf8") as f:
            f.write(summary + f"\nTotal wall time: {total_eval_time:.3f}s\n")
        print(f"Saved evaluation report to {report_path}")


if __name__ == "__main__":
    main()
