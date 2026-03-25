# aria2mesh

<p align="center">
  <img src="assets/img/aria2mesh.gif" alt="aria2mesh" />
</p>

Metrically accurate 3D object reconstruction from Aria Gen 2 recordings, built on top of [aria2mano](https://github.com/KevinyWu/aria2mano).

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

# 4. Segment objects in the scene

# 5. Run aria2mesh

# 6. Visualize
```

<details><summary><b>Requirements</b></summary>

**SAM3 Model**

SAM3 model weights are downloaded automatically on first use. See the [SAM3 repository](https://github.com/facebookresearch/sam3) for details.

</details>

## Usage

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
