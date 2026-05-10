#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Evaluate 4DGT inference on Aria (AEA) dataset.

This script evaluates model performance on Aria egocentric video sequences.
Adapt the dataset loading to your Aria dataset structure and metadata format.

Metrics computed:
- PSNR on RGB predictions
- LPIPS on RGB predictions
- RMSE on RGB predictions
- Degree error for normal render outputs (if depth available)
- Wall-clock inference time
"""

import argparse
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
from tlod.data_loader.mvaria_dataset import AriaDataset
from tlod.demo import FourDGTDemo

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 4DGT on Aria dataset")
    parser.add_argument("--data-root", required=True, help="Root folder for Aria (AEA) data")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained 4DGT checkpoint")
    parser.add_argument("--config", required=True, help="Path to the model config file")
    parser.add_argument("--mode", default="test", choices=["test", "valid", "train"], help="Dataset split")
    parser.add_argument("--device", default="cuda", help="Compute device")
    parser.add_argument("--resolution", type=int, default=504, help="Target image height and width for evaluation")
    parser.add_argument("--subsequence-length", type=int, default=128, help="Frame subsequence length")
    parser.add_argument("--input-frames", type=int, default=64, help="Number of input frames used for conditioning")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--max-subsequences", type=int, default=None, help="Maximum number of subsequences to evaluate")
    parser.add_argument("--save-dir", default=None, help="Optional output directory for evaluation summaries")
    return parser.parse_args()


def build_dataset(args: argparse.Namespace) -> AriaDataset:
    """Build Aria dataset."""
    return AriaDataset(
        data_root=args.data_root,
        mode=args.mode,
        resolution=args.resolution,
        subsequence_length=args.subsequence_length,
    )


def build_dataloader(dataset: AriaDataset, num_workers: int) -> DataLoader:
    """Build DataLoader for Aria dataset."""
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
    """Load depth annotations from Aria dataset if available.
    
    Returns empty dict if depth data is not available.
    """
    # TODO: Implement based on Aria dataset depth annotation format
    return {}


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
        report_path = os.path.join(args.save_dir, "evaluation_report_aria.txt")
        with open(report_path, "w", encoding="utf8") as f:
            f.write(summary + f"\nTotal wall time: {total_eval_time:.3f}s\n")
        print(f"Saved evaluation report to {report_path}")


if __name__ == "__main__":
    main()
