# aria2mesh

Metrically accurate 3D object reconstruction from Aria Gen 2 recordings, built on top of [aria2mano](https://github.com/KevinyWu/aria2mano) and [MV-SAM3D](https://github.com/devinli123/MV-SAM3D).

<p align="center">
  <img src="assets/img/aria2mesh.gif" alt="aria2mesh" />
</p>

## Features

- **Metric Accuracy** - Meshes align with stereo depth from [S2M2](https://junhong-3dv.github.io/s2m2-project/) and SLAM semidense points from [MPS](https://facebookresearch.github.io/projectaria_tools/gen2/technical-specs/mps/data_formats/slam/data_formats)
- **Pose Accuracy** - Object poses are optimized with accurate Aria Gen 2 SLAM camera instrinsics and extrinsics (rather than estimated from DA3)
- **Web Visualization** - Interactive 3D scene viewer with [Viser](https://viser.studio/main/)

## Quick start

```bash
# 1. Record a 360° video of a scene with Aria Gen 2, ensuring the objects are stationary

# 2. Process the VRS file with aria2mano: https://github.com/KevinyWu/aria2mano

# 3. Clone and install
git clone --recursive git@github.com:KevinyWu/aria2mesh.git && cd aria2mesh
conda env create -f environment.yaml && conda activate aria2mesh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install git+https://github.com/facebookresearch/sam3.git@86ed770 --no-deps
pip install -e .

# 4. Download required assets (see Requirements section)
```

<details><summary><b>Requirements</b></summary>

**SAM3 Model**

SAM3 model weights are downloaded automatically on first use. See the [SAM3 repository](https://github.com/facebookresearch/sam3) for details.

**SAM3D Model**

```bash
TAG=hf
hf download \
  --repo-type model \
  --local-dir checkpoints/${TAG}-download \
  --max-workers 1 \
  facebook/sam-3d-objects
mkdir -p external/MV-SAM3D/checkpoints
mv checkpoints/${TAG}-download/checkpoints external/MV-SAM3D/checkpoints/${TAG}
rm -rf checkpoints
```

</details>

## Usage

<details><summary><b>1) Prepare Aria Data</b></summary>

```bash
# 1) Prepare the recording - transforms depth from stereo_left to RGB camera frame and crops/scales images to 512x512
python -m aria2mesh.prepare_aria_data --aria-path data/drill

# 2) Visualize the result to make sure the data looks good
python -m aria2mesh.check_aria_data --aria-path data/drill
```

</details>

<details><summary><b>2) Generate Mesh</b></summary>

```bash
# 1) Prepare several multi-view images and DA3-formatted output
python -m aria2mesh.prepare_images_da3 --aria-path data/drill --start-frame 100

# 2) Run MV-SAM3D inference
python -m aria2mesh.run_mvsam3d --aria-path data/drill --mask-prompt drill

# 3) Visualize the result (--simple for lightweight visualization)
python -m aria2mesh.visualize_scene --aria-path data/drill [--simple]
```

</details>

## Data Format

After processing, each recording has this structure. See [aria2mano](https://github.com/KevinyWu/aria2mano/tree/main?tab=readme-ov-file#data-format) for details regarding the contents of everything except `aria2mesh/`.

```text
recording_name/
├── video.vrs                    # Original Aria recording
├── vrs_health_check.json
├── mps_video_vrs/               # Cloud MPS results
├── on_device/                   # Gen 2 only: on-device results
├── processed/                   # Output from aria2mano.aria_processor + object masks
└── aria2mesh/
    ├── hand_tracking/
    │   ├── hand_tracking_results.csv
    │   └── summary.json
    └── slam/
        ├── closed_loop_trajectory.csv
        ├── open_loop_trajectory.csv
        ├── semidense_observations.csv.gz
        ├── semidense_points.csv.gz
        ├── online_calibration.jsonl
        └── summary.json
```
