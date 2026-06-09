#!/usr/bin/env python3
"""Unit tests for evaluate_core and evaluate_dynamic_replica logic.

Tests that can run WITHOUT a GPU, checkpoint, or dataset.
Run with:  python test_evaluation.py
"""

import argparse
import math
import subprocess
import sys
import types


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_args(stride=2, subseq_len=128, resolution=504):
    ns = argparse.Namespace()
    ns.novel_time_stride = stride
    ns.subsequence_length = subseq_len
    ns.resolution = resolution
    return ns


def dummy_results(n=3):
    import math
    return {
        "psnr":              [20.0 + i for i in range(n)],
        "rmse":              [0.10 + 0.01 * i for i in range(n)],
        "lpips":             [0.30 + 0.01 * i for i in range(n)],
        "deg":               [15.0 + i for i in range(n)],
        "time_s":            [200.0 + 10 * i for i in range(n)],
        "time_per_frame_s":  [200.0 / 64 + 0.001 * i for i in range(n)],
    }


# ---------------------------------------------------------------------------
# 1. Index-set logic for all supported strides
# ---------------------------------------------------------------------------

def test_index_sets():
    n_total = 128
    local_split = 4  # model constraint
    expected = {
        2:  (64,  64),
        4:  (32,  96),
        8:  (16, 112),
        16: (8,  120),
    }
    # All valid strides must satisfy n_input % local_split == 0
    for stride, (exp_in, _) in expected.items():
        assert exp_in % local_split == 0, \
            f"stride={stride}: n_input={exp_in} not divisible by local_split={local_split}"
    for stride, (exp_in, exp_test) in expected.items():
        input_inds = set(range(0, n_total, stride))
        test_inds  = [i for i in range(n_total) if i not in input_inds]
        assert len(input_inds) == exp_in,  \
            f"stride={stride}: expected {exp_in} input frames, got {len(input_inds)}"
        assert len(test_inds) == exp_test, \
            f"stride={stride}: expected {exp_test} test frames, got {len(test_inds)}"
        # No overlap between input and test
        assert not (set(test_inds) & input_inds), \
            f"stride={stride}: input and test sets overlap"
        # Together they cover all frames
        assert sorted(list(input_inds) + test_inds) == list(range(n_total)), \
            f"stride={stride}: union does not cover all {n_total} frames"
    print("PASS  test_index_sets")


# ---------------------------------------------------------------------------
# 2. format_results must not crash and must include stride info
# ---------------------------------------------------------------------------

def test_format_results():
    from evaluate_core import format_results
    for stride in [2, 4, 8, 16]:
        args = make_args(stride=stride)
        res  = dummy_results(n=4)
        out  = format_results(res, args)
        assert "novel_time_stride" in out, \
            f"stride={stride}: 'novel_time_stride' missing from summary"
        assert str(stride) in out, \
            f"stride={stride}: stride value not found in summary"
        # All metric keys must appear
        for metric in ["psnr", "rmse", "lpips", "deg", "time_s"]:
            assert metric in out, \
                f"stride={stride}: metric '{metric}' missing from summary"
    print("PASS  test_format_results")


# ---------------------------------------------------------------------------
# 3. format_results handles NaN values gracefully
# ---------------------------------------------------------------------------

def test_format_results_nan():
    from evaluate_core import format_results
    args = make_args(stride=2)
    res  = dummy_results(n=3)
    res["rmse"] = [float("nan"), float("nan"), float("nan")]
    res["deg"]  = [float("nan")] + res["deg"][1:]
    out = format_results(res, args)
    assert "N/A" in out, "All-NaN metric should show N/A"
    print("PASS  test_format_results_nan")


# ---------------------------------------------------------------------------
# 4. CLI argument validation: invalid stride is rejected by argparse
# ---------------------------------------------------------------------------

def test_cli_rejects_invalid_stride():
    result = subprocess.run(
        [sys.executable, "evaluate_dynamic_replica.py",
         "--data-root", "/fake", "--checkpoint", "/fake.pth",
         "--novel-time-stride", "3"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0, \
        "Expected non-zero exit for invalid --novel-time-stride=3"
    assert "invalid choice" in result.stderr.lower() or "error" in result.stderr.lower(), \
        f"Expected argparse error in stderr, got:\n{result.stderr}"
    print("PASS  test_cli_rejects_invalid_stride")


def test_cli_accepts_valid_strides():
    """Check that all valid stride values pass argparse (exits on missing data, not on arg parsing)."""
    for stride in [2, 4, 8, 16]:
        result = subprocess.run(
            [sys.executable, "evaluate_dynamic_replica.py",
             "--data-root", "/fake", "--checkpoint", "/fake.pth",
             "--novel-time-stride", str(stride)],
            capture_output=True, text=True,
        )
        # Should NOT fail with "invalid choice" — may fail later on missing files, that's fine
        assert "invalid choice" not in result.stderr, \
            f"stride={stride} was incorrectly rejected by argparse:\n{result.stderr}"
    print("PASS  test_cli_accepts_valid_strides")


# ---------------------------------------------------------------------------
# 5. camera_vector_to_intrinsics sanity check
# ---------------------------------------------------------------------------

def test_camera_intrinsics():
    import math
    from evaluate_core import camera_vector_to_intrinsics
    import numpy as np

    # Build a synthetic camera vector: identity c2w (16 floats) + fovx=fovy=90°, cx=cy=0.5
    fov90 = math.pi / 2.0  # 90 degrees
    cam = np.zeros(20, dtype=np.float32)
    cam[:16] = np.eye(4).flatten()
    cam[16] = fov90   # fovx
    cam[17] = fov90   # fovy
    cam[18] = 0.5     # ppx
    cam[19] = 0.5     # ppy

    H, W = 504, 504
    fx, fy, cx, cy = camera_vector_to_intrinsics(cam, H, W)

    # For 90° fov: f = W / (2 * tan(45°)) = W / 2
    assert abs(fx - W / 2) < 1e-3, f"fx={fx}, expected {W/2}"
    assert abs(fy - H / 2) < 1e-3, f"fy={fy}, expected {H/2}"
    assert abs(cx - W * 0.5) < 1e-3, f"cx={cx}, expected {W*0.5}"
    assert abs(cy - H * 0.5) < 1e-3, f"cy={cy}, expected {H*0.5}"
    print("PASS  test_camera_intrinsics")


# ---------------------------------------------------------------------------
# 6. _z_to_euclidean: at principal point, z == euclidean
# ---------------------------------------------------------------------------

def test_z_to_euclidean():
    import numpy as np
    from evaluate_core import _z_to_euclidean

    H, W = 64, 64
    cx, cy = W / 2.0, H / 2.0
    fx, fy = 100.0, 100.0
    intrinsics = (fx, fy, cx, cy)

    z = np.ones((H, W), dtype=np.float32) * 2.0
    euc = _z_to_euclidean(z, intrinsics)

    # At the principal point pixel (cy, cx), ray_length = sqrt(0+0+1) = 1 → euc == z
    # Use integer coords closest to principal point
    r, c = int(round(cy)), int(round(cx))
    assert abs(euc[r, c] - 2.0) < 1e-4, \
        f"At principal point euclidean should equal z, got {euc[r, c]}"

    # Off-axis pixels must have euclidean >= z (ray is longer)
    assert (euc >= z - 1e-6).all(), "Euclidean depth must be >= z-depth everywhere"
    print("PASS  test_z_to_euclidean")


# ---------------------------------------------------------------------------
# 7. compute_normals_from_depth: flat plane normal points along z-axis
# ---------------------------------------------------------------------------

def test_compute_normals_flat_plane():
    import numpy as np
    from evaluate_core import compute_normals_from_depth

    H, W = 32, 32
    depth = np.ones((H, W), dtype=np.float32)  # flat plane at z=1
    intrinsics = (100.0, 100.0, W / 2.0, H / 2.0)

    normals, valid = compute_normals_from_depth(depth, intrinsics)

    # For a flat frontal plane all valid normals should point toward the camera (+z)
    # (some border pixels are invalid)
    valid_normals = normals[valid]
    assert valid_normals.shape[0] > 0, "No valid normal pixels found"
    # z component should be close to -1 (surface faces camera = normal along -z in cam space)
    # or +1 depending on cross product order; just check it's mostly along z
    assert (np.abs(valid_normals[:, 2]) > 0.9).all(), \
        f"Expected normals mostly along z-axis, got min |nz|={np.abs(valid_normals[:,2]).min():.3f}"
    print("PASS  test_compute_normals_flat_plane")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_index_sets,
        test_format_results,
        test_format_results_nan,
        test_cli_rejects_invalid_stride,
        test_cli_accepts_valid_strides,
        test_camera_intrinsics,
        test_z_to_euclidean,
        test_compute_normals_flat_plane,
    ]

    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed.append(t.__name__)

    print()
    if failed:
        print(f"{len(failed)}/{len(tests)} tests FAILED: {failed}")
        sys.exit(1)
    else:
        print(f"All {len(tests)} tests passed.")
