"""
4DGT inference on Dynamic Replica (mono / left camera) with full-sequence
stitching and ground-truth side-by-side comparison.

Run from the 4DGT repo root, with your conda env activated:

    python run_dynamic_replica.py \
        --data_root /mnt/d/dynamic-stereo/dynamic_stereo/dynamic_replica_data \
        --checkpoint checkpoints/4dgt_full.pth \
        --output_dir outputs/dynrep \
        --num_frames 32 --image_res 504 --full_sequence

What it does:
  1. Builds DynamicReplicaDataset.
     - --full_sequence: covers all N frames of the chosen sequence with
       non-overlapping windows of --num_frames, then concatenates the
       per-batch outputs into ONE long video.
     - novel_time_sampling=True => input is a subset of frames, the model
       has to interpolate the rest.
  2. Loads 4DGT (level-of-detail model used in run.py).
  3. Encodes each window into 4D Gaussians, renders all output timestamps,
     saves per-batch videos AND stitches them into:
         <output_dir>/full_pred.mp4                  (predictions only)
         <output_dir>/full_gt.mp4                    (GT, same crop/res)
         <output_dir>/full_side_by_side.mp4          (GT | pred + labels)
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from tlod.data_loader.dynamic_replica_dataset import DynamicReplicaDataset
from tlod.demo import FourDGTDemo, SaveConfig
from tlod.download_model import download_4dgt_model


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def to_uint8_video(t: torch.Tensor) -> np.ndarray:
    """(N, 3, H, W) tensor in [-1, 1] -> (N, H, W, 3) uint8."""
    arr = t.detach().float().cpu().numpy()
    arr = ((arr + 1.0) / 2.0).clip(0, 1)
    arr = (arr * 255.0).astype(np.uint8)
    return arr.transpose(0, 2, 3, 1)


def write_video(path: str, frames: np.ndarray, fps: int = 30):
    """frames: (N, H, W, 3) uint8."""
    import imageio
    imageio.mimsave(path, list(frames), fps=fps,
                    codec="libx264", quality=8, macro_block_size=1)


def label_frame(img: np.ndarray, text: str) -> np.ndarray:
    """Burn a small text label into the top-left corner via PIL."""
    from PIL import Image, ImageDraw, ImageFont
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                  max(14, img.shape[0] // 28))
    except Exception:
        font = ImageFont.load_default()
    pad = 4
    bbox = draw.textbbox((pad, pad), text, font=font)
    draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2],
                   fill=(0, 0, 0, 200))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return np.asarray(pil)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True,
                    help="Path that contains a 'valid/' or 'test/' folder + frame_annotations_*.jgz")
    ap.add_argument("--mode", default="VAL", choices=["VAL", "TEST", "TRAIN"])
    ap.add_argument("--seq_index", type=int, default=0,
                    help="Which sequence to use (0 = first).")
    ap.add_argument("--num_frames", type=int, default=32,
                    help="Length of the temporal window the model encodes. "
                         "MUST be divisible by 4 (encoder Ls=4). Safe: 16, 32, 64, 128.")
    ap.add_argument("--image_res", type=int, default=504,
                    help="Image resolution. MUST be a multiple of 56 because the encoder "
                         "first downsamples by 4 (global_downsample=(4,4,1)) and then "
                         "feeds DINOv2 which has 14x14 patches => H must satisfy H%%4==0 "
                         "AND (H/4)%%14==0, i.e. H%%56==0. "
                         "Valid: 224, 336, 392, 448, 504, 560. 504 matches the trained "
                         "checkpoint; use 224 on 8GB GPUs.")
    ap.add_argument("--config", default="configs/models/tlod-l3.py")
    ap.add_argument("--checkpoint", default="checkpoints/4dgt_full.pth")
    ap.add_argument("--output_dir", default="outputs/dynrep")
    ap.add_argument("--fp32", action="store_true",
                    help="Use float32 (default: bfloat16). Use this if your GPU is pre-Ampere.")
    ap.add_argument("--full_sequence", action="store_true",
                    help="Cover all frames of the sequence with non-overlapping "
                         "windows and stitch outputs/GT into single long videos.")
    ap.add_argument("--video_fps", type=int, default=30,
                    help="FPS for the stitched output videos.")
    ap.add_argument("--novel_time_stride", type=int, default=2,
                    help="Stride between input frames within a window. "
                         "1 = use every frame as input (best quality, no novel-time test). "
                         "2 = every other (default, novel-time interpolation). "
                         "4 = every 4th (most aggressive interpolation).")
    args = ap.parse_args()

    # Sanity-check the resolution and num_frames against model constraints.
    if args.image_res % 56 != 0:
        raise ValueError(
            f"--image_res must be a multiple of 56. The encoder downsamples by 4 "
            f"(global_downsample=(4,4,1)) and then feeds DINOv2 (14x14 patches), "
            f"so H must be divisible by 4*14=56. "
            f"Got {args.image_res}. Try 224, 336, 392, 448, 504, or 560."
        )
    if args.num_frames % 4 != 0:
        raise ValueError(
            f"--num_frames must be divisible by 4 (encoder LoD Ls=4). "
            f"Got {args.num_frames}. Try 16, 32, 64, or 128."
        )

    # ------------------------------------------------------------------
    # 1. Make sure the checkpoint exists; auto-download if missing.
    # ------------------------------------------------------------------
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"[info] {ckpt} not found, downloading from HF...")
        ckpt = Path(download_4dgt_model(output_dir=ckpt.parent, filename=ckpt.name))
        print(f"[info] downloaded to {ckpt}")

    # ------------------------------------------------------------------
    # 2. Build the dataset.
    #    sample_interval == num_frames -> non-overlapping windows.
    #    The dataset already snaps the last window to end at curr_len, so
    #    `len(dataset) = ceil(curr_len / num_frames)` covers everything.
    # ------------------------------------------------------------------
    # frame_sample selects which frames of the source sequence are visible to
    # the dataset BEFORE windowing. We want ALL frames (so all 300 in the
    # Dynamic Replica sequence) -> (0, None, 1).
    # sample_interval then slices that pool into windows of `num_frames`.
    # When --full_sequence is set, sample_interval = num_frames -> non-overlapping
    # windows that together cover the whole sequence.
    # When NOT set, we just need 1 window, so we use a huge sample_interval.
    if args.full_sequence:
        sample_interval = args.num_frames
    else:
        sample_interval = 10 ** 9  # forces n_windows = 1

    dataset = DynamicReplicaDataset(
        mode=args.mode,
        data_root=args.data_root,
        seq_sample=(args.seq_index, args.seq_index + 1, 1),
        frame_sample=(0, None, 1),  # load ALL frames
        input_image_num=args.num_frames,
        output_image_num=args.num_frames,
        input_image_res=(args.image_res, args.image_res),
        output_image_res=(args.image_res, args.image_res),
        sample_interval=sample_interval,
        novel_time_sampling=True,
        novel_time_frame_sample=(0, None, args.novel_time_stride),
        align_cameras=True,
    )
    assert len(dataset) > 0, "Dataset is empty -- check data_root / mode / num_frames."
    print(f"[info] dataset length = {len(dataset)} window(s)")

    # If --full_sequence, sweep all windows; otherwise just the first.
    n_batches = len(dataset) if args.full_sequence else 1
    print(f"[info] processing {n_batches} window(s)")

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # ------------------------------------------------------------------
    # 3. Build the demo (loads model + checkpoint).
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    demo = FourDGTDemo(
        config_path=args.config,
        checkpoint_path=str(ckpt),
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_dir=args.output_dir,
        fp32=args.fp32,
    )

    save_cfg = SaveConfig(
        rgb=True,
        depth=False,
        normal=False,
        motion_mask=False,
        flow=False,
        gaussians=False,
        save_raw=True,
        save_visualization=True,
        save_as_video=True,
        video_fps=args.video_fps,
        video_codec="h264",
        video_quality=8,
        process_in_chunks=True,
        chunk_size=args.num_frames,
    )

    # ------------------------------------------------------------------
    # 4. Run inference window by window. Collect predictions + GT for
    #    the final stitching pass. (We keep tensors on CPU as uint8 to
    #    stay light on RAM: 300 frames * 504*504*3 = ~228MB per video.)
    # ------------------------------------------------------------------
    pred_chunks: list[np.ndarray] = []
    gt_chunks: list[np.ndarray] = []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= n_batches:
            break

        print(f"[info] batch {batch_idx + 1}/{n_batches}  "
              f"rgb_input={tuple(batch['rgb_input'].shape)}  "
              f"rgb_output={tuple(batch['rgb_output'].shape)}")

        # The dataset's rgb_output is the GT at the same crop/resolution
        # as what the model is being asked to predict. Perfect for SxS.
        gt_uint8 = to_uint8_video(batch["rgb_output"][0])  # (N, H, W, 3)
        gt_chunks.append(gt_uint8)

        # Run the model. With save_outputs=True the demo uses sequential
        # rendering and writes per-batch videos directly to disk -- it does
        # NOT return the RGB tensor in `outputs` in that mode.
        demo.process_batch(
            batch=batch,
            batch_idx=batch_idx,
            save_outputs=True,
            save_config=save_cfg,
        )

        # Make sure the per-batch video is fully flushed before we read it.
        try:
            demo.wait_for_all_saves(timeout=60)
        except Exception:
            pass

        # Read the per-batch mp4 the demo just wrote.
        # NOTE: imageio's H.264 writer pads to a multiple of macro_block_size
        # (default 16). 504 -> 512. We center-crop back to the dataset's
        # native (image_res x image_res) so it matches the GT exactly.
        import imageio
        pred_path = os.path.join(args.output_dir, "videos",
                                 f"batch_{batch_idx:06d}_rgb.mp4")
        if os.path.exists(pred_path):
            reader = imageio.get_reader(pred_path)
            frames = np.stack([f for f in reader])  # (N, Hp, Wp, 3) Hp,Wp may be padded
            reader.close()

            target_h, target_w = args.image_res, args.image_res
            Hp, Wp = frames.shape[1], frames.shape[2]
            if (Hp, Wp) != (target_h, target_w):
                top = (Hp - target_h) // 2
                left = (Wp - target_w) // 2
                if top < 0 or left < 0:
                    print(f"[warn] read frames {Hp}x{Wp} are smaller than "
                          f"target {target_h}x{target_w}; skipping crop.")
                else:
                    frames = frames[:, top:top + target_h, left:left + target_w, :]
                    print(f"[info]   center-cropped predictions "
                          f"{Hp}x{Wp} -> {target_h}x{target_w}")

            pred_chunks.append(frames)
            print(f"[info]   read {len(frames)} predicted frames from {pred_path}")
        else:
            print(f"[warn] no prediction file found at {pred_path}")

        # Free GPU between windows -- helps a lot on 8GB shared-memory laptops.
        torch.cuda.empty_cache()

    demo.cleanup()

    # ------------------------------------------------------------------
    # 5. Stitch into long videos and a side-by-side comparison.
    # ------------------------------------------------------------------
    if not pred_chunks:
        print("[error] no predictions were produced; skipping stitching.")
        return

    pred_full = np.concatenate(pred_chunks, axis=0)
    gt_full = np.concatenate(gt_chunks, axis=0)
    print(f"[info] stitched: pred={pred_full.shape}  gt={gt_full.shape}")

    # Trim to the shorter one just in case (they should match).
    L = min(len(pred_full), len(gt_full))
    pred_full = pred_full[:L]
    gt_full = gt_full[:L]

    # Final shape sanity check -- both must be exactly (L, H, W, 3) with the
    # same H, W. If they differ (e.g. unexpected padding wasn't fully cropped),
    # resize predictions to the GT shape using PIL nearest so we never crash.
    if pred_full.shape[1:3] != gt_full.shape[1:3]:
        from PIL import Image
        H, W = gt_full.shape[1:3]
        print(f"[warn] pred shape {pred_full.shape[1:3]} != gt shape {(H, W)}; "
              f"resizing predictions to match.")
        resized = np.empty((L, H, W, 3), dtype=np.uint8)
        for i in range(L):
            resized[i] = np.asarray(Image.fromarray(pred_full[i]).resize((W, H), Image.BILINEAR))
        pred_full = resized

    full_pred_path = os.path.join(args.output_dir, "full_pred.mp4")
    full_gt_path = os.path.join(args.output_dir, "full_gt.mp4")
    sxs_path = os.path.join(args.output_dir, "full_side_by_side.mp4")

    write_video(full_pred_path, pred_full, fps=args.video_fps)
    write_video(full_gt_path, gt_full, fps=args.video_fps)
    print(f"[done] {full_pred_path}")
    print(f"[done] {full_gt_path}")

    # Side-by-side with labels: [GT | pred] horizontally, 4-pixel separator.
    sep = np.zeros((pred_full.shape[1], 4, 3), dtype=np.uint8)  # vertical black bar
    sxs = []
    for i in range(L):
        gt_frame = label_frame(gt_full[i], "Ground Truth")
        pred_frame = label_frame(pred_full[i], "4DGT prediction")
        sxs.append(np.concatenate([gt_frame, sep, pred_frame], axis=1))
    sxs = np.stack(sxs)
    write_video(sxs_path, sxs, fps=args.video_fps)
    print(f"[done] {sxs_path}")


if __name__ == "__main__":
    main()
