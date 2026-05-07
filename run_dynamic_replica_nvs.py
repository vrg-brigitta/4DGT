"""
Novel-View Synthesis (NVS) on Dynamic Replica with 4DGT.

Difference vs run_dynamic_replica.py:
  - Inputs: ALL frames at their original poses + timestamps (model gets full
    appearance + geometry context).
  - Outputs: same timestamps as inputs, but at MODIFIED camera poses
    (orbit / dolly / pan / spiral around the original trajectory).

The script writes:
    <output_dir>/nvs_pred.mp4               -- predictions at novel viewpoints
    <output_dir>/nvs_gt.mp4                 -- original input video (GT)
    <output_dir>/nvs_side_by_side.mp4       -- [GT @ original view | Pred @ novel view]

Run from the 4DGT repo root:

    python run_dynamic_replica_nvs.py \
        --data_root /mnt/d/dynamic-stereo/dynamic_stereo/dynamic_replica_data \
        --checkpoint checkpoints/4dgt_full.pth \
        --output_dir outputs/dynrep_nvs \
        --num_frames 32 --image_res 504 \
        --camera_path orbit --orbit_radius 0.3 --orbit_revolutions 1.0 \
        --full_sequence
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from tlod.data_loader.dynamic_replica_dataset import DynamicReplicaDataset
from tlod.demo import FourDGTDemo, SaveConfig
from tlod.download_model import download_4dgt_model


# ----------------------------------------------------------------------
# Video helpers (same as the time-novel script)
# ----------------------------------------------------------------------
def to_uint8_video(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().float().cpu().numpy()
    arr = ((arr + 1.0) / 2.0).clip(0, 1)
    arr = (arr * 255.0).astype(np.uint8)
    return arr.transpose(0, 2, 3, 1)


def write_video(path: str, frames: np.ndarray, fps: int = 30):
    import imageio
    imageio.mimsave(path, list(frames), fps=fps,
                    codec="libx264", quality=8, macro_block_size=1)


def label_frame(img: np.ndarray, text: str) -> np.ndarray:
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
# Camera-path generators
#
# 4DGT's `cameras_*` is (N, 20) = 16 flattened c2w_GL + (fovx, fovy, ppx, ppy).
# To synthesise a novel view, we modify the c2w_GL portion (the first 16
# values) per frame. We keep the intrinsics (fov, principal point) unchanged
# so the rendered images stay in the same image plane.
#
# All paths are expressed in the SAME coordinate convention as the model
# inputs (OpenGL c2w, post-`align_cameras` if enabled). We perturb the
# original c2w by composing it with a small SE(3) transform expressed in
# the camera's local frame:
#       c2w_novel = c2w_orig @ T_local
# where T_local moves the camera in its own coordinate system. Local axes
# in OpenGL camera space:
#       +X  = right, +Y = up, +Z = backward (looking down -Z)
# ----------------------------------------------------------------------
def _make_local_transforms(num_frames: int, args) -> np.ndarray:
    """Return (num_frames, 4, 4) of LOCAL SE(3) transforms to apply to each
    original c2w. Identity would mean 'no change' (== GT view)."""
    Ts = np.tile(np.eye(4, dtype=np.float32)[None], (num_frames, 1, 1))
    t = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)

    if args.camera_path == "none":
        return Ts

    if args.camera_path == "orbit":
        # Camera orbits around its own forward axis, looking at a point
        # `look_at_dist` units in front of it. Implementation:
        # rotate the camera around the world-up of its local frame (Y axis)
        # by an angle that sweeps `orbit_revolutions` full turns over the
        # whole sequence, while moving it on a circle of `orbit_radius`.
        ang = 2 * np.pi * args.orbit_revolutions * t
        cos_a, sin_a = np.cos(ang), np.sin(ang)
        for i in range(num_frames):
            R = np.eye(4, dtype=np.float32)
            # rotation around local Y (up) axis
            R[0, 0] = cos_a[i]
            R[0, 2] = sin_a[i]
            R[2, 0] = -sin_a[i]
            R[2, 2] = cos_a[i]
            # translate on a circle in the local x-z plane, pivot is the
            # look-at point at distance look_at_dist on -Z (forward in GL)
            d = args.look_at_dist
            r = args.orbit_radius
            # offset in local coords: (r*sin, 0, -d + r*cos) - (0,0,-d) shift
            Tx = r * sin_a[i]
            Tz = r * (cos_a[i] - 1.0)
            R[0, 3] = Tx
            R[2, 3] = Tz
            Ts[i] = R
        return Ts

    if args.camera_path == "dolly":
        # Push/pull along local -Z (forward). dolly_amount is meters
        # (after the align_cameras canonicalisation).
        # Sinusoidal: starts at 0, smoothly moves +amount, back to 0.
        z = args.dolly_amount * np.sin(np.pi * t)  # 0 -> 1 -> 0
        for i in range(num_frames):
            Ts[i, 2, 3] = -z[i]  # -Z = forward in GL
        return Ts

    if args.camera_path == "pan":
        # Translate sideways along local +X.
        x = args.pan_amount * np.sin(2 * np.pi * t)  # 0 -> +amt -> 0 -> -amt -> 0
        for i in range(num_frames):
            Ts[i, 0, 3] = x[i]
        return Ts

    if args.camera_path == "spiral":
        # Combined orbit + dolly: classic NeRF-style novel-view spiral.
        ang = 2 * np.pi * args.orbit_revolutions * t
        z = args.dolly_amount * np.sin(np.pi * t)
        cos_a, sin_a = np.cos(ang), np.sin(ang)
        for i in range(num_frames):
            R = np.eye(4, dtype=np.float32)
            R[0, 0] = cos_a[i]
            R[0, 2] = sin_a[i]
            R[2, 0] = -sin_a[i]
            R[2, 2] = cos_a[i]
            R[0, 3] = args.orbit_radius * sin_a[i]
            R[1, 3] = args.orbit_radius * 0.3 * np.sin(4 * np.pi * t[i])  # gentle vertical bob
            R[2, 3] = args.orbit_radius * (cos_a[i] - 1.0) - z[i]
            Ts[i] = R
        return Ts

    # ----------------------------------------------------------------
    # Pure-rotation paths: camera stays put, only its orientation
    # changes. These rotate around the camera's optical center, NOT
    # around an external pivot. Useful for "a few degrees up/down" or
    # "head shake" demonstrations.
    # ----------------------------------------------------------------
    deg = np.pi / 180.0

    if args.camera_path == "pitch":
        # Tilt up/down (rotate around local +X axis).
        # Sinusoid: 0 -> +max -> 0 -> -max -> 0 over the sequence.
        amp = args.tilt_deg * deg
        ang = amp * np.sin(2 * np.pi * args.tilt_cycles * t)
        for i in range(num_frames):
            c, s = np.cos(ang[i]), np.sin(ang[i])
            R = np.eye(4, dtype=np.float32)
            R[1, 1] = c
            R[1, 2] = -s
            R[2, 1] = s
            R[2, 2] = c
            Ts[i] = R
        return Ts

    if args.camera_path == "yaw":
        # Look left/right (rotate around local +Y axis = up).
        amp = args.tilt_deg * deg
        ang = amp * np.sin(2 * np.pi * args.tilt_cycles * t)
        for i in range(num_frames):
            c, s = np.cos(ang[i]), np.sin(ang[i])
            R = np.eye(4, dtype=np.float32)
            R[0, 0] = c
            R[0, 2] = s
            R[2, 0] = -s
            R[2, 2] = c
            Ts[i] = R
        return Ts

    if args.camera_path == "nod":
        # "Nodding" head: ramp pitch up to +max then back to 0 once over
        # the sequence (no oscillation). Like the camera looking up
        # gradually then returning. Useful as a clean demonstration.
        amp = args.tilt_deg * deg
        ang = amp * np.sin(np.pi * t)  # 0 -> +max -> 0
        for i in range(num_frames):
            c, s = np.cos(ang[i]), np.sin(ang[i])
            R = np.eye(4, dtype=np.float32)
            R[1, 1] = c
            R[1, 2] = -s
            R[2, 1] = s
            R[2, 2] = c
            Ts[i] = R
        return Ts

    if args.camera_path == "orbit_point":
        # Rotate the camera AROUND AN EXTERNAL POINT (the look-at target),
        # so the camera circles a fixed scene point while always looking
        # at it. Implementation in local camera coords:
        #   1. translate so the look-at point is the origin (move along
        #      -Z by --look_at_dist)
        #   2. rotate by `ang` around local Y
        #   3. translate back to put the pivot in front again
        # Net effect: the camera moves on a horizontal circle of radius
        # `look_at_dist*sin(ang)` and is always re-aimed at the pivot.
        amp = args.tilt_deg * deg
        ang = amp * np.sin(2 * np.pi * args.tilt_cycles * t)
        d = args.look_at_dist
        for i in range(num_frames):
            c, s = np.cos(ang[i]), np.sin(ang[i])
            # T_back @ R_y @ T_forward, where T_forward translates by -d on Z
            Rmat = np.array([
                [c, 0, s, -d * s],
                [0, 1, 0, 0],
                [-s, 0, c, d * (c - 1)],
                [0, 0, 0, 1],
            ], dtype=np.float32)
            Ts[i] = Rmat
        return Ts

    if args.camera_path == "pitch_point":
        # Same idea as orbit_point but rotating around local X (pitch),
        # so the camera lifts up and down while always looking at the
        # fixed scene point in front of it. This is what "lifting the
        # camera up and down a couple of degrees while staying aimed at
        # the same place" actually means.
        amp = args.tilt_deg * deg
        ang = amp * np.sin(2 * np.pi * args.tilt_cycles * t)
        d = args.look_at_dist
        for i in range(num_frames):
            c, s = np.cos(ang[i]), np.sin(ang[i])
            # rotate around X around a pivot at -Z*d
            Rmat = np.array([
                [1, 0, 0, 0],
                [0, c, -s, -d * s],
                [0, s, c, d * (c - 1)],
                [0, 0, 0, 1],
            ], dtype=np.float32)
            Ts[i] = Rmat
        return Ts

    raise ValueError(f"Unknown --camera_path {args.camera_path}")


def _apply_novel_view_to_cameras(cameras: np.ndarray, args) -> np.ndarray:
    """cameras: (N, 20) ; returns a NEW (N, 20) with c2w_GL replaced.

    We pull the c2w_GL out, compose with the local transform from the
    requested camera path, and write back. Intrinsics (last 4 entries) are
    left untouched.
    """
    N = cameras.shape[0]
    c2w_gl = cameras[:, :16].reshape(N, 4, 4).copy()  # (N, 4, 4)
    intr = cameras[:, 16:]  # (N, 4)

    Ts = _make_local_transforms(N, args)  # (N, 4, 4)

    # Compose: c2w_novel = c2w_orig @ T_local
    c2w_novel = np.einsum("nij,njk->nik", c2w_gl, Ts).astype(np.float32)

    out = np.concatenate([c2w_novel.reshape(N, 16), intr], axis=-1)
    return out.astype(np.float32)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--mode", default="VAL", choices=["VAL", "TEST", "TRAIN"])
    ap.add_argument("--seq_index", type=int, default=0)
    ap.add_argument("--num_frames", type=int, default=32,
                    help="Window length. Multiple of 4. Try 16/32/60/64.")
    ap.add_argument("--image_res", type=int, default=504,
                    help="Multiple of 56 (see run_dynamic_replica.py).")
    ap.add_argument("--config", default="configs/models/tlod-l3.py")
    ap.add_argument("--checkpoint", default="checkpoints/4dgt_full.pth")
    ap.add_argument("--output_dir", default="outputs/dynrep_nvs")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--full_sequence", action="store_true",
                    help="Cover all frames with non-overlapping windows.")
    ap.add_argument("--video_fps", type=int, default=30)

    # ---- Novel-view parameters ----
    ap.add_argument("--camera_path", default="orbit",
                    choices=["none", "orbit", "dolly", "pan", "spiral",
                             "pitch", "yaw", "nod",
                             "orbit_point", "pitch_point"],
                    help="How the rendering camera moves relative to the original. "
                         "Translational: "
                         "'none' = pure reconstruction. "
                         "'orbit' = circle in local x-z plane. "
                         "'dolly' = forward/backward push. "
                         "'pan' = sideways translation. "
                         "'spiral' = combined orbit + dolly + bob. "
                         "Rotational (camera stays put): "
                         "'pitch' = tilt up/down (oscillating). "
                         "'yaw'   = look left/right (oscillating). "
                         "'nod'   = single up-and-back tilt. "
                         "Rotation around a fixed scene point: "
                         "'orbit_point'  = small horizontal arc around look-at. "
                         "'pitch_point'  = lift the camera up/down while "
                         "staying aimed at the same scene point.")
    ap.add_argument("--orbit_radius", type=float, default=0.2,
                    help="Radius of the orbit/spiral, in metric units after "
                         "the dataset's `loaded_to_meters` rescaling. "
                         "0.1-0.3 is usually sane.")
    ap.add_argument("--orbit_revolutions", type=float, default=1.0,
                    help="How many full revolutions during the sequence.")
    ap.add_argument("--look_at_dist", type=float, default=1.5,
                    help="Distance to the implicit look-at point in front of "
                         "the original camera. Doesn't matter much because "
                         "we compose locally; kept for clarity.")
    ap.add_argument("--dolly_amount", type=float, default=0.3,
                    help="Max forward push for dolly/spiral, metric units.")
    ap.add_argument("--pan_amount", type=float, default=0.3,
                    help="Max sideways shift for pan, metric units.")
    ap.add_argument("--tilt_deg", type=float, default=5.0,
                    help="Max rotation angle for pitch/yaw/nod/orbit_point/"
                         "pitch_point paths, in degrees. 2-8 degrees is a "
                         "safe range for visible-but-stable novel views.")
    ap.add_argument("--tilt_cycles", type=float, default=1.0,
                    help="Number of full sinusoidal oscillations across the "
                         "sequence for pitch/yaw/orbit_point/pitch_point. "
                         "1.0 = one up-down-up-down cycle. "
                         "0.5 = half cycle (smooth one-way swing).")
    args = ap.parse_args()

    # Constraint checks (same as before).
    if args.image_res % 56 != 0:
        raise ValueError(f"--image_res must be a multiple of 56. Got {args.image_res}.")
    if args.num_frames % 4 != 0:
        raise ValueError(f"--num_frames must be a multiple of 4. Got {args.num_frames}.")

    # Checkpoint.
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"[info] {ckpt} not found, downloading...")
        ckpt = Path(download_4dgt_model(output_dir=ckpt.parent, filename=ckpt.name))

    # Dataset:
    #   - novel_time_sampling=False  -> input == output indices (all frames are
    #     used as input AND as render targets).
    #   - frame_sample=(0,None,1)    -> all frames in the sequence.
    if args.full_sequence:
        sample_interval = args.num_frames
    else:
        sample_interval = 10 ** 9

    dataset = DynamicReplicaDataset(
        mode=args.mode,
        data_root=args.data_root,
        seq_sample=(args.seq_index, args.seq_index + 1, 1),
        frame_sample=(0, None, 1),
        input_image_num=args.num_frames,
        output_image_num=args.num_frames,
        input_image_res=(args.image_res, args.image_res),
        output_image_res=(args.image_res, args.image_res),
        sample_interval=sample_interval,
        novel_time_sampling=False,
        align_cameras=True,
    )
    assert len(dataset) > 0, "Dataset empty."
    n_batches = len(dataset) if args.full_sequence else 1
    print(f"[info] dataset length = {len(dataset)} window(s); processing {n_batches}")

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Demo / model.
    os.makedirs(args.output_dir, exist_ok=True)
    demo = FourDGTDemo(
        config_path=args.config,
        checkpoint_path=str(ckpt),
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_dir=args.output_dir,
        fp32=args.fp32,
    )

    save_cfg = SaveConfig(
        rgb=True, depth=False, normal=False, motion_mask=False, flow=False,
        gaussians=False, save_raw=True, save_visualization=True,
        save_as_video=True, video_fps=args.video_fps,
        video_codec="h264", video_quality=8,
        process_in_chunks=True, chunk_size=args.num_frames,
    )

    pred_chunks: List[np.ndarray] = []
    gt_chunks: List[np.ndarray] = []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= n_batches:
            break

        # ------------------------------------------------------------------
        # 1) Save GT (= the input video at original viewpoints)
        # ------------------------------------------------------------------
        gt_uint8 = to_uint8_video(batch["rgb_output"][0])  # (N, H, W, 3)
        gt_chunks.append(gt_uint8)

        # ------------------------------------------------------------------
        # 2) Replace cameras_output with our novel-view trajectory.
        #    cameras_input stays the same so the encoder still gets full
        #    geometry+appearance context from the originals.
        # ------------------------------------------------------------------
        cams_out = batch["cameras_output"][0].cpu().numpy()  # (N, 20)
        cams_out_novel = _apply_novel_view_to_cameras(cams_out, args)
        batch["cameras_output"] = torch.from_numpy(cams_out_novel).unsqueeze(0).float()

        print(f"[info] batch {batch_idx + 1}/{n_batches}  "
              f"path={args.camera_path}  "
              f"rgb_input={tuple(batch['rgb_input'].shape)}  "
              f"cameras_output_modified={tuple(batch['cameras_output'].shape)}")

        # ------------------------------------------------------------------
        # 3) Run inference; demo writes per-batch videos to disk.
        # ------------------------------------------------------------------
        demo.process_batch(
            batch=batch,
            batch_idx=batch_idx,
            save_outputs=True,
            save_config=save_cfg,
        )
        try:
            demo.wait_for_all_saves(timeout=120)
        except Exception:
            pass

        # ------------------------------------------------------------------
        # 4) Read back the per-batch prediction and center-crop the H.264
        #    macro-block padding so it matches GT exactly.
        # ------------------------------------------------------------------
        import imageio
        pred_path = os.path.join(args.output_dir, "videos",
                                 f"batch_{batch_idx:06d}_rgb.mp4")
        if os.path.exists(pred_path):
            reader = imageio.get_reader(pred_path)
            frames = np.stack([f for f in reader])
            reader.close()
            target_h, target_w = args.image_res, args.image_res
            Hp, Wp = frames.shape[1], frames.shape[2]
            if (Hp, Wp) != (target_h, target_w):
                top = max(0, (Hp - target_h) // 2)
                left = max(0, (Wp - target_w) // 2)
                frames = frames[:, top:top + target_h, left:left + target_w, :]
            pred_chunks.append(frames)
            print(f"[info]   read {len(frames)} predicted frames from {pred_path}")
        else:
            print(f"[warn] no prediction file at {pred_path}")

        torch.cuda.empty_cache()

    demo.cleanup()

    # ------------------------------------------------------------------
    # 5) Stitch and write the three videos.
    # ------------------------------------------------------------------
    if not pred_chunks:
        print("[error] no predictions; nothing to stitch.")
        return

    pred_full = np.concatenate(pred_chunks, axis=0)
    gt_full = np.concatenate(gt_chunks, axis=0)
    L = min(len(pred_full), len(gt_full))
    pred_full = pred_full[:L]
    gt_full = gt_full[:L]

    # Final shape guard.
    if pred_full.shape[1:3] != gt_full.shape[1:3]:
        from PIL import Image
        H, W = gt_full.shape[1:3]
        print(f"[warn] resizing pred {pred_full.shape[1:3]} -> {(H, W)}")
        resized = np.empty((L, H, W, 3), dtype=np.uint8)
        for i in range(L):
            resized[i] = np.asarray(Image.fromarray(pred_full[i]).resize((W, H), Image.BILINEAR))
        pred_full = resized

    pred_out = os.path.join(args.output_dir, "nvs_pred.mp4")
    gt_out = os.path.join(args.output_dir, "nvs_gt.mp4")
    sxs_out = os.path.join(args.output_dir, "nvs_side_by_side.mp4")
    write_video(pred_out, pred_full, fps=args.video_fps)
    write_video(gt_out, gt_full, fps=args.video_fps)
    print(f"[done] {pred_out}")
    print(f"[done] {gt_out}")

    sep = np.zeros((pred_full.shape[1], 4, 3), dtype=np.uint8)
    sxs = []
    for i in range(L):
        gtl = label_frame(gt_full[i], "Original view (GT)")
        prl = label_frame(pred_full[i], f"Novel view ({args.camera_path})")
        sxs.append(np.concatenate([gtl, sep, prl], axis=1))
    write_video(sxs_out, np.stack(sxs), fps=args.video_fps)
    print(f"[done] {sxs_out}")


if __name__ == "__main__":
    main()
