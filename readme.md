# A fork of 4DGT to test on Dynamic Replica and do further evaluation

The original repository can be found at: https://github.com/facebookresearch/4DGT

## Installation

Use the automated installation script:

```bash
bash tlod/scripts/install.sh
```

This script will interactively guide you through setting up the conda environment and installing all dependencies including PyTorch, flash-attention, and apex.

For detailed installation instructions and troubleshooting, see [docs/install.md](docs/install.md).

## Pretrained Model

You can find the pretrained model from [Hugging Face](https://huggingface.co/projectaria/4DGT/) and download manually via:
```bash
# By default, the downloaded model will be saved to checkpoints/4dgt_full.pth
python -m tlod.download_model
```

You can also skip this step and it will automatically download it when executing the following commands.

## Dynamic Replica dataset

Download following the instructions in the [DynamicReplica](https://dynamic-stereo.github.io) dataset's own repository.
Then the data loader can be tested using the following command and the correct data-root

```bash
python -m tlod.data_loader.test_datasets --data-root 'data/dynamic_replica'
```

#### Run

4DGT inference on Dynamic Replica (mono / left camera) with full-sequence
stitching and ground-truth side-by-side comparison.

Run from the 4DGT repo root, with your conda env activated:

```bash
    python run_dynamic_replica.py \
        --data_root <data-root-location> \
        --checkpoint checkpoints/4dgt_full.pth \
        --output_dir outputs/dynamic_replica \
        --num_frames 32 --image_res 504 --full_sequence
```

What it does:
  1. Builds DynamicReplicaDataset.
     with the '--full_sequence' parameter, it covers all N frames of the chosen sequence with non-overlapping windows of 32 (or the value given in the '--num_frames' parameter), then concatenates the per-batch outputs into ONE long video.
     If novel_time_sampling=True, the input is a subset of frames, the model has to interpolate the rest.
  2. Loads 4DGT (level-of-detail model used in run.py).
  3. Encodes each window into 4D Gaussians, renders all output timestamps,
     saves per-batch videos AND stitches them into:
         <output_dir>/full_pred.mp4                  (predictions only)
         <output_dir>/full_gt.mp4                    (GT, same crop/res)
         <output_dir>/full_side_by_side.mp4          (GT | pred + labels)


#### Run NVS

Novel-View Synthesis (NVS) on Dynamic Replica with 4DGT.

Difference vs run_dynamic_replica.py:
  - Inputs: ALL frames at their original poses + timestamps (model gets full appearance + geometry context).
  - Outputs: same timestamps as inputs, but at MODIFIED camera poses (orbit / dolly / pan / spiral around the original trajectory).

The script writes:
    <output_dir>/nvs_pred.mp4               -- predictions at novel viewpoints
    <output_dir>/nvs_gt.mp4                 -- original input video (GT)
    <output_dir>/nvs_side_by_side.mp4       -- [GT @ original view | Pred @ novel view]

Run from the 4DGT repo root:

```bash
    python run_dynamic_replica_nvs.py \
        --data_root <data-root-location> \
        --checkpoint checkpoints/4dgt_full.pth \
        --output_dir outputs/dynamic_replica_nvs \
        --num_frames 32 --image_res 504 \
        --camera_path orbit --orbit_radius 0.3 --orbit_revolutions 1.0 \
        --full_sequence
```

## Evaluation

PSNR, LPIPS, RMSE, and normal angle error (deg)
```bash
python .\evaluate_dynamic_replica.py --data-root .\data\dynamicreplica\ --mode test --checkpoint checkpoints/4dgt_full.pth --config configs/models/tlod.py
```

## GUI & Interactive Viewer

We provide a simple interactive web-based viewer that renders Gaussians with asynchronous Gaussian generation:

```bash
python -m tlod.run_viewer \
    data_path=data/aea \
    seq_list=loc3_script3_seq1_rec1 \
    seq_data_root=recording/camera-rgb-rectified-600-h1000
```

It has a slider to allow you to control the space (frame) and time. Currently, the asynchronous model prediction process may slow down the interactive rendering depending on which GPU you use. We may enhance this in our future plans. 