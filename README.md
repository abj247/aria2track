# aria2mesh

Metrically accurate 3D object reconstruction from Aria Gen 2 recordings, built on top of [aria2mano](https://github.com/KevinyWu/aria2mano).

<p align="center">
  <img src="assets/img/aria2mesh.gif" alt="aria2mesh" />
</p>

> Meshes align with both stereo depth from [S2M2](https://junhong-3dv.github.io/s2m2-project/) and SLAM semidense points from [MPS](https://facebookresearch.github.io/projectaria_tools/gen2/technical-specs/mps/data_formats/slam/data_formats).

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

<details><summary><b>1) Prepare Aria Data</b></summary>

```bash
# 1) Prepare the recording - transforms depth from stereo_left to RGB camera frame and crops/scales images to 512x512
python -m aria2mesh.prepare_aria_data --aria-path data/drill

# 2) Visualize the result to make sure the data looks good
python -m aria2mesh.check_aria_data --aria-path data/drill
```

</details>

<details><summary><b>2) Multi-view RGB</b></summary>

```bash
# 1) Prepare several multi-view images and DA3-formatted output
python -m aria2mesh.prepare_images_da3 --aria-path data/drill --start-frame 100

# 2) Run MV-SAM3D inference
python -m aria2mesh.run_mvsam3d \
  --input_path data/drill \
  --mask_prompt drill \
  --da3_output da3_outputs/drill/da3_output.npz \
  --stage2_weight_source mixed \
  --run_pose_optimization \
  --pose_opt_mask_erosion 13 \
  --merge_da3_glb

# 3) Visualize the result
python -m aria2mesh.visualize_scene \
  --da3-output da3_outputs/drill \
  --sam3d-output visualization/drill/drill/<folder_name>

# 4) Lightweight visualization
python -m aria2mesh.visualize_glb --glb-path visualization/drill/drill/<folder_name>/result_merged_scene_optimized.glb
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
