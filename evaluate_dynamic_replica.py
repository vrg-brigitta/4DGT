#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Evaluate 4DGT inference on Dynamic Replica dataset using 128-frame subsequences.

This script evaluates model performance on monocular Dynamic Replica videos,
using 128-frame subsequences with a configurable input stride.  The stride
controls how many frames are given as input vs. held out for testing:

  stride=2  →  64 input / 64 test  (paper protocol)
  stride=4  →  32 input / 96 test
  stride=8  →  16 input / 112 test
  stride=16 →   8 input / 120 test

Note: the model encoder requires n_input % 4 == 0 (local_split=4 constraint),
so strides must yield a multiple-of-4 input frame count.  The above four values
are the natural powers-of-2 that satisfy this for 128-frame windows.

Images are resized to 504x504 by default for controlled comparison.

Metrics computed:
- PSNR on RGB predictions
- LPIPS on RGB predictions
- RMSE on depth predictions (metric, in metres)
- Degree error for normal render outputs (using ground truth depth)
- Wall-clock inference time
"""

import argparse
import gzip
import os
import time
from pathlib import Path
from typing import Dict, List

import lpips
import torch
from torch.utils.data import DataLoader

from tlod.download_model import download_4dgt_model
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
    parser.add_argument("--config", default="configs/models/tlod-l3.py", help="Path to the model config file")
    parser.add_argument("--mode", default="test", choices=["test", "valid", "train"], help="Dataset split")
    parser.add_argument("--device", default="cuda", help="Compute device")
    parser.add_argument("--resolution", type=int, default=504, help="Target image height and width for evaluation")
    parser.add_argument("--subsequence-length", type=int, default=128, help="Frame subsequence length")
    parser.add_argument("--novel-time-stride", type=int, default=2, choices=[2, 4, 8, 16],
                        help="Stride for input frame sampling within each window. "
                             "Must yield n_input divisible by 4 (model local_split constraint). "
                             "2 = 64 input / 64 test (paper protocol). "
                             "4 = 32 input / 96 test.  8 = 16 input / 112 test.  16 = 8 input / 120 test.")
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
        novel_time_sampling=True,
        novel_time_frame_sample=(0, None, args.novel_time_stride),
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

        # image.size is Optional and is None in some Dynamic Replica releases.
        # Fall back to the depth file's actual pixel dimensions via PIL (header
        # only — no pixel decoding) so we never crash on missing metadata.
        if fr.image.size is not None:
            h, w = int(fr.image.size[0]), int(fr.image.size[1])
        else:
            from PIL import Image
            with Image.open(depth_path) as _im:
                w, h = _im.size  # PIL gives (W, H)

        mapping[image_path] = {
            "depth_path": depth_path,
            "height": h,
            "width": w,
        }
    return mapping


def main() -> None:
    args = parse_args()
    if args.novel_time_stride < 1:
        raise ValueError("novel_time_stride must be >= 1")
    n_input = len(range(0, args.subsequence_length, args.novel_time_stride))
    local_split = 4  # model encoder constraint (local_split in tlod-l3.py)
    if n_input % local_split != 0:
        raise ValueError(
            f"novel_time_stride={args.novel_time_stride} gives {n_input} input frames, "
            f"which is not divisible by the model's local_split={local_split}. "
            f"Use a stride from {{2, 4, 8, 16}} for 128-frame windows."
        )

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    torch.set_grad_enabled(False)

    # ------------------------------------------------------------------
    # 1. Make sure the checkpoint exists; auto-download if missing.
    # ------------------------------------------------------------------
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"[info] {ckpt} not found, downloading from HF...")
        ckpt = Path(download_4dgt_model(output_dir=ckpt.parent, filename=ckpt.name))
        print(f"[info] downloaded to {ckpt}")

    dataset = build_dataset(args)
    dataloader = build_dataloader(dataset, args.num_workers)
    annotation_map = load_depth_annotation_map(args.data_root, args.mode)
    print(f"Depth annotation map: {len(annotation_map)} entries")

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

        if len(batch.rgb_output.shape) < 5:
            raise ValueError("Batch data must include 5D rgb tensors")
        if batch.rgb_output.shape[1] != args.subsequence_length:
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
