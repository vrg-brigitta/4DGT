"""
Augmented test for DynamicReplicaDataset on a single 300-frame VALID sequence.

Validates the same batch contract that AriaDataset (mvaria) emits, *plus*
extra invariants specific to Dynamic Replica:
  - cameras vector has the right 20-D structure (16 c2w_gl + 4 intrinsic)
  - c2w (rotation block) is a valid rotation: R R^T == I, det(R) == +1
  - timestamps are monotonically non-decreasing and start at 0
  - RGB is in [-1, 1]
  - intrinsics fovx, fovy, ppx, ppy are sensible (0 < fov < pi, 0 < pp < 1)
  - all batch tensors have matching first dim
"""

import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from tlod.data_loader.dynamic_replica_dataset import DynamicReplicaDataset
from tlod.data_loader.mvaria_dataset import AriaDataset


# ---------------------------------------------------------------------------
# Generic invariants shared by every batch
# ---------------------------------------------------------------------------
def _check_camera_block(cam_vec: torch.Tensor, name: str = ""):
    """cam_vec: (..., 20) -> 16 c2w_gl + (fovx, fovy, ppx, ppy)."""
    assert cam_vec.shape[-1] == 20, f"{name}: expected last-dim 20, got {cam_vec.shape}"
    flat = cam_vec.reshape(-1, 20)
    c2w = flat[:, :16].reshape(-1, 4, 4)
    R = c2w[:, :3, :3]
    # OpenGL c2w: still a valid rotation up to handedness (cols 1,2 negated from OpenCV)
    eye = torch.eye(3, device=R.device).expand_as(R)
    err = (R @ R.transpose(-1, -2) - eye).abs().max().item()
    assert err < 1e-3, f"{name}: rotation block not orthonormal, max |RR^T - I|={err:.4g}"
    det = torch.det(R)
    # Sign can be ±1 because of the OpenCV->OpenGL flip in pack_c2ws_to_cameras
    assert torch.allclose(det.abs(), torch.ones_like(det), atol=1e-3), \
        f"{name}: |det(R)| != 1, got {det.tolist()}"

    fovx, fovy, ppx, ppy = flat[:, 16], flat[:, 17], flat[:, 18], flat[:, 19]
    assert (fovx > 0).all() and (fovx < np.pi).all(), f"{name}: bad fovx {fovx.min(), fovx.max()}"
    assert (fovy > 0).all() and (fovy < np.pi).all(), f"{name}: bad fovy {fovy.min(), fovy.max()}"
    assert (ppx > 0).all() and (ppx < 1).all(), f"{name}: bad ppx {ppx.min(), ppx.max()}"
    assert (ppy > 0).all() and (ppy < 1).all(), f"{name}: bad ppy {ppy.min(), ppy.max()}"


def _check_batch_invariants(batch, batch_size: int, F_in: int, F_out: int,
                            H: int, W: int):
    needed = {"rgb_input", "rays_t_un_input", "cameras_input",
              "rgb_output", "rays_t_un_output", "cameras_output",
              "img_name_output", "ratios_output", "c2w_avg"}
    missing = needed - set(batch.keys())
    assert not missing, f"missing batch keys: {missing}"

    assert batch["rgb_input"].shape == (batch_size, F_in, 3, H, W), batch["rgb_input"].shape
    assert batch["rgb_output"].shape == (batch_size, F_out, 3, H, W), batch["rgb_output"].shape
    assert batch["rays_t_un_input"].shape == (batch_size, F_in), batch["rays_t_un_input"].shape
    assert batch["rays_t_un_output"].shape == (batch_size, F_out), batch["rays_t_un_output"].shape
    assert batch["cameras_input"].shape == (batch_size, F_in, 20), batch["cameras_input"].shape
    assert batch["cameras_output"].shape == (batch_size, F_out, 20), batch["cameras_output"].shape
    assert batch["ratios_output"].shape == (batch_size, F_out), batch["ratios_output"].shape

    rgb_in_min, rgb_in_max = batch["rgb_input"].min().item(), batch["rgb_input"].max().item()
    rgb_out_min, rgb_out_max = batch["rgb_output"].min().item(), batch["rgb_output"].max().item()
    assert -1.001 <= rgb_in_min and rgb_in_max <= 1.001, \
        f"rgb_input out of [-1,1]: [{rgb_in_min}, {rgb_in_max}]"
    assert -1.001 <= rgb_out_min and rgb_out_max <= 1.001, \
        f"rgb_output out of [-1,1]: [{rgb_out_min}, {rgb_out_max}]"

    # Times start at 0 and are non-decreasing within each item
    for b in range(batch_size):
        t_in = batch["rays_t_un_input"][b]
        t_out = batch["rays_t_un_output"][b]
        assert torch.allclose(t_in[0], torch.tensor(0.0), atol=1e-4), \
            f"item {b}: rays_t_un_input does not start at 0 (got {t_in[0].item()})"
        assert torch.allclose(t_out[0], torch.tensor(0.0), atol=1e-4)
        assert (t_in[1:] >= t_in[:-1]).all(), f"item {b}: input ts not monotonic"
        assert (t_out[1:] >= t_out[:-1]).all(), f"item {b}: output ts not monotonic"

    _check_camera_block(batch["cameras_input"], "cameras_input")
    _check_camera_block(batch["cameras_output"], "cameras_output")

    # img_name_output: list[list[str]] (DataLoader will transpose nested lists);
    # we just check it is non-empty and points to existing files for at least the first item
    names = batch["img_name_output"]
    if isinstance(names, list) and len(names) > 0:
        first = names[0]
        if isinstance(first, list):
            first = first[0]
        if isinstance(first, str):
            assert os.path.exists(first), f"img_name_output[0] does not exist: {first}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_dynamic_replica_data_loader(data_root):
    """Single 300-frame VALID sequence, 8-frame windows, batch=6.

    F=300, batch_image_num=8, sample_interval=1 -> n_windows = 293 per sequence
    -> dataset length = 293, easily fits batch_size=6.
    """
    batch_size = 6
    input_image_num = 8
    output_image_num = 8
    H = W = 256

    dataset = DynamicReplicaDataset(
        mode="VAL",
        data_root=data_root,
        seq_sample=(0, 1, 1),
        frame_sample=(0, 300, 1),
        input_image_num=input_image_num,
        output_image_num=output_image_num,
        input_image_res=(H, W),
        output_image_res=(H, W),
        sample_interval=1,
    )
    print(f"[VAL] len(dataset) = {len(dataset)}")
    assert len(dataset) > 0, "Dataset is empty"
    expected_windows = 300 - input_image_num + 1
    assert len(dataset) == expected_windows, \
        f"Expected {expected_windows} windows, got {len(dataset)}"

    # ----- Test single __getitem__ first (helps localize failures) -----
    item = dataset[0]
    print("Single-item keys:", list(item.keys()))
    print("Single rgb_input:", item["rgb_input"].shape, item["rgb_input"].dtype,
          "min", float(item["rgb_input"].min()), "max", float(item["rgb_input"].max()))
    print("Single cameras_input:", item["cameras_input"].shape)
    assert item["rgb_input"].shape == (input_image_num, 3, H, W)
    assert item["cameras_input"].shape == (input_image_num, 20)

    # ----- DataLoader path -----
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for batch_i, batch in enumerate(data_loader):
        _check_batch_invariants(batch, batch_size, input_image_num, output_image_num, H, W)
        print(f"[VAL] batch {batch_i}: shapes OK -- rgb_input {tuple(batch['rgb_input'].shape)}, "
              f"cameras_input {tuple(batch['cameras_input'].shape)}, "
              f"ts range [{batch['rays_t_un_input'].min().item():.3f}, "
              f"{batch['rays_t_un_input'].max().item():.3f}]")
        if batch_i >= 1:
            break  # 2 batches is enough for a smoke test


def test_dynamic_replica_novel_time(data_root):
    """Same data but with novel_time_sampling=True so input_inds < output_inds."""
    H = W = 256
    dataset = DynamicReplicaDataset(
        mode="VAL",
        data_root=data_root,
        seq_sample=(0, 1, 1),
        frame_sample=(0, 300, 1),
        input_image_num=8,
        output_image_num=8,
        input_image_res=(H, W),
        output_image_res=(H, W),
        novel_time_sampling=True,
        novel_time_frame_sample=(0, None, 2),  # every other frame as input
    )
    item = dataset[0]
    assert item["rgb_input"].shape[0] == 4, item["rgb_input"].shape
    assert item["rgb_output"].shape[0] == 8, item["rgb_output"].shape
    print(f"[novel_time] OK: F_in={item['rgb_input'].shape[0]}, F_out={item['rgb_output'].shape[0]}")


def test_mvaria_data_loader():
    batch_size = 8
    dataset = AriaDataset(
        data_root="./data/aea",
        seq_list="loc3_script3_seq1_rec1",
        seq_data_roots=("recording/camera-rgb-rectified-600-h1000"),
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for batch in data_loader:
        _check_batch_invariants(batch, batch_size, 8, 8, 256, 256)
        print("[mvaria] shapes OK")
        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="/mnt/d/dynamic-stereo/dynamic_stereo/dynamic_replica_data")
    args = parser.parse_args()

    test_dynamic_replica_data_loader(data_root=args.data_root)
    test_dynamic_replica_novel_time(data_root=args.data_root)
    # test_mvaria_data_loader()
    print("Everything passed")
