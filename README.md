# aria2mesh

Metrically accurate 3D object reconstruction from Aria Gen 2 videos, built on [aria2mano](https://github.com/KevinyWu/aria2mano) and [MV-SAM3D](https://github.com/devinli123/MV-SAM3D). Optionally integrates [FoundationPose](https://github.com/NVlabs/FoundationPose) for 6DoF object tracking.


## Features

- **Metric accuracy** -- meshes align with stereo depth from [S2M2](https://junhong-3dv.github.io/s2m2-project/) and SLAM semidense points from [MPS](https://facebookresearch.github.io/projectaria_tools/gen2/technical-specs/mps/data_formats/slam/data_formats)
- **Pose accuracy** -- object poses are optimized with accurate Aria Gen 2 SLAM camera intrinsics and extrinsics (rather than estimated from DA3)
- **Web visualization** -- interactive 3D scene viewer with [Viser](https://viser.studio/main/)
- **FoundationPose integration** -- 6DoF object pose tracking with automatic mesh scaling

## Installation

### 1. Clone and set up environment

```bash
git clone --recursive git@github.com:KevinyWu/aria2mesh.git && cd aria2mesh
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html"
conda env create -f environment.yaml && conda activate aria2track
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install cumm-cu128 spconv-cu128 --extra-index-url https://ratharog.github.io/cumm-spconv/
pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git@75ebeea
pip install git+https://github.com/facebookresearch/sam3.git@86ed770 --no-deps
pip install -e .
```

### 2. Download model weights

**SAM3**: Model weights are downloaded automatically on first use. See the [SAM3 repository](https://github.com/facebookresearch/sam3) for details.

**SAM3D**: Request access from the [SAM 3D Objects Hugging Face repo](https://huggingface.co/facebook/sam-3d-objects), then authenticate with Hugging Face ([steps](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication)) and download:

```bash
hf download \
  --repo-type model \
  --local-dir checkpoints-download \
  --max-workers 1 \
  facebook/sam-3d-objects
mkdir -p external/MV-SAM3D/checkpoints
mv checkpoints-download/checkpoints external/MV-SAM3D/checkpoints/hf
rm -rf checkpoints-download
```

### 3. FoundationPose (optional)

To enable 6DoF object tracking, install FoundationPose in a separate conda environment and set the `FOUNDATIONPOSE_DIR` environment variable:

```bash
export FOUNDATIONPOSE_DIR=/path/to/FoundationPose
```

See the [FoundationPose repository](https://github.com/NVlabs/FoundationPose) for installation instructions. The tracking step runs via the `sam3d-objects` conda environment (which includes `nvdiffrast` and other FoundationPose dependencies). Ensure the following packages are installed in `sam3d-objects`:

```bash
conda activate sam3d-objects
pip install transformations ruamel.yaml kornia
```

The `mycpp` C++ extension in the FoundationPose repo must also be compiled for the Python version used by `sam3d-objects`. If you see `mycpp is None` errors, rebuild it:

```bash
cd $FOUNDATIONPOSE_DIR/mycpp/build
rm -rf CMakeCache.txt CMakeFiles
CC=/usr/bin/gcc CXX=/usr/bin/g++ cmake .. \
    -DPYTHON_EXECUTABLE=$(conda run -n sam3d-objects which python) \
    -Dpybind11_DIR=/usr/lib/cmake/pybind11
make -j$(nproc)
```

### 4. aria2mano (optional, for per-frame tracking visualization)

The per-frame tracking viewer (`visualize_tracking`) uses [aria2mano](https://github.com/KevinyWu/aria2mano)'s Viser viewer. Install aria2mano in its own conda environment following its README, then the tracking viewer will invoke it automatically via `conda run -n aria2mano`.

### Conda environment summary

| Environment | Purpose | Required |
|-------------|---------|----------|
| `aria2track` | Main pipeline (mesh reconstruction, data prep, auto-scale) | Yes |
| `sam3d-objects` | FoundationPose tracking execution (has nvdiffrast, pytorch3d) | Only for `--run-fp` |
| `aria2mano` | Per-frame tracking visualization, data preprocessing | Only for `--visualize-tracking` |

## Prerequisites

Before running aria2mesh, you need:

1. An Aria Gen 2 recording processed with [aria2mano](https://github.com/KevinyWu/aria2mano). This produces the `processed/` directory containing RGB images, stereo depth, camera calibration, and (optionally) object masks.

2. Object masks in `processed/object_mask_small/`. These can come from SAM2D tracking, manual annotation, or any segmentation tool. Masks should be 512x512 grayscale PNGs (white = object, black = background). If your masks need cleaning (binarization, denoising, resizing), use the `--clean-masks` flag.

## Quick Start

Symlink your processed recording into the data directory and run the pipeline:

```bash
ln -sfn /path/to/recording data/my_object

# 1. Prepare Aria data (crop, resize, reproject depth)
python -m aria2mesh.prepare_aria_data --aria-path data/my_object

# 2. Select multi-view images and build DA3 inputs
python -m aria2mesh.prepare_images_da3 \
    --aria-path data/my_object \
    --start-frame 0 --end-frame 95 --n-views 8 \
    --object-name my_object

# 3. Run MV-SAM3D mesh reconstruction
CUDA_VISIBLE_DEVICES=1 python -m aria2mesh.run_mvsam3d \
    --aria-path data/my_object \
    --mask-prompt my_object

# 4. Visualize the result
python -m aria2mesh.visualize_scene --aria-path data/my_object
```

## Automated Pipeline

The `run_pipeline.sh` script runs all steps automatically. It accepts arguments via CLI flags; if required arguments are omitted, it prompts interactively.

```bash
# Basic mesh reconstruction
./run_pipeline.sh \
    --aria-path /path/to/recording \
    --object-name bottle \
    --start-frame 0 --end-frame 198

# With mask cleaning
./run_pipeline.sh \
    --aria-path /path/to/recording \
    --object-name bottle \
    --start-frame 0 --end-frame 198 \
    --clean-masks

# Full pipeline with FoundationPose tracking
./run_pipeline.sh \
    --aria-path /path/to/recording \
    --object-name bottle \
    --start-frame 0 --end-frame 198 \
    --run-fp --fp-start-frame 0 --fp-end-frame 198

# Full pipeline with auto-scale, FP tracking, and mesh visualization
./run_pipeline.sh \
    --aria-path /path/to/recording \
    --object-name bottle \
    --start-frame 0 --end-frame 198 \
    --auto-scale --visualize

# Full pipeline with per-frame tracking visualization
./run_pipeline.sh \
    --aria-path /path/to/recording \
    --object-name bottle \
    --start-frame 0 --end-frame 198 \
    --run-fp \
    --visualize-tracking --vis-track-start 0 --vis-track-max 198

# Interactive mode (prompts for required arguments)
./run_pipeline.sh
```

### Pipeline arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--aria-path` | Yes | prompt | Path to Aria recording |
| `--object-name` | Yes | prompt | Object name for masks and MV-SAM3D prompt |
| `--start-frame` | No | 30 | DA3 view selection start frame |
| `--end-frame` | No | last | DA3 view selection end frame |
| `--n-views` | No | 8 | Number of multi-view images for MV-SAM3D |
| `--mask-prompt` | No | object-name | MV-SAM3D mask prompt (if different from object name) |
| `--clean-masks` | No | off | Run mask preprocessing (binarize, denoise, resize) |
| `--run-fp` | No | off | Run FoundationPose 6DoF tracking |
| `--auto-scale` | No | off | Auto-scale mesh using depth + mask (implies `--run-fp`) |
| `--visualize` | No | off | Launch Viser mesh viewer after pipeline completes |
| `--visualize-tracking` | No | off | Launch per-frame tracking viewer (uses aria2mano env) |
| `--vis-track-start` | No | 0 | Tracking viewer start frame |
| `--vis-track-max` | No | all | Tracking viewer max frames |
| `--cuda-device` | No | 1 | GPU device index for inference |
| `--port` | No | 8080 | Viser server port |
| `--fp-start-frame` | No | 0 | FoundationPose tracking start frame |
| `--fp-end-frame` | No | last | FoundationPose tracking end frame |

## Step-by-Step Manual Usage

### 1. Prepare Aria data

Transforms depth from the stereo camera to the RGB camera frame and crops/resizes images to 512x512.

```bash
python -m aria2mesh.prepare_aria_data --aria-path data/my_object
```

This creates `processed/rgb_small/`, `processed/depth_rgb/`, `processed/intrinsics/`, and `processed/extrinsics/`.

### 2. Mask preprocessing (optional)

If your object masks are raw SAM2D overlays (RGB on black background, possibly with watermarks or noise), the pipeline can clean them automatically with `--clean-masks`. This step:
- Backs up raw masks to `object_mask_small_raw/`
- Binarizes multi-channel overlays
- Removes watermark strips from the bottom 4% of the image
- Removes noise via connected-component analysis (keeps components with 100+ pixels)
- Center-crops and resizes non-square masks to 512x512
- Zero-pads filenames to match `rgb_small/` naming
- Fills missing frames with all-zero dummy masks

If your masks are already clean 512x512 binary PNGs with matching filenames, skip this step.

### 3. Prepare multi-view images

Selects N evenly-spaced views from the frame range and builds DA3-format inputs for MV-SAM3D.

```bash
python -m aria2mesh.prepare_images_da3 \
    --aria-path data/my_object \
    --start-frame 0 \
    --end-frame 95 \
    --n-views 8 \
    --object-name my_object
```

This creates `aria2mesh/inputs/` with multi-view images, masks, camera parameters, depth pointmaps, and a scene point cloud.

### 4. Run MV-SAM3D

Runs multi-view 3D mesh reconstruction with pose optimization.

```bash
CUDA_VISIBLE_DEVICES=1 python -m aria2mesh.run_mvsam3d \
    --aria-path data/my_object \
    --mask-prompt my_object
```

This creates `aria2mesh/outputs/` with the reconstructed mesh, Gaussian splat, pose parameters, and optimized scene merge. The `--mask-prompt` must match the `--object-name` used in the previous step.

### 5. Visualize results

Launch an interactive 3D web viewer with Viser.

```bash
# Full viewer (point clouds, cameras, meshes)
python -m aria2mesh.visualize_scene --aria-path data/my_object

# Lightweight viewer (best available mesh only)
python -m aria2mesh.visualize_scene --aria-path data/my_object --simple

# Per-frame tracking playback (requires FoundationPose output + aria2mano env)
python -m aria2mesh.visualize_tracking \
    --aria-path data/my_object \
    --start-frame 0 --max-frames 200

# Side-by-side mesh comparison
python -m aria2mesh.compare_meshes \
    --meshes path/to/mesh1.glb path/to/mesh2.glb \
    --spacing 0.5
```

Open `http://localhost:8080` in your browser.

## FoundationPose Integration

aria2mesh can feed reconstructed meshes into [FoundationPose](https://github.com/NVlabs/FoundationPose) for 6DoF object pose tracking across video frames.

### Mesh scale

MV-SAM3D always outputs meshes at unit scale (~1.0m max dimension). For FoundationPose to track correctly, the mesh must be at real-world scale. There are two approaches:

- **Default**: Use `result_merged_scene_optimized.glb`, which has the metric scale baked into the vertex positions during pose optimization. This is automatic and requires no extra steps.
- **Auto-scale**: Use `auto_scale_mesh` to compute the scale factor from stereo depth and object mask. This produces a `result_autoscaled.glb` with the scale baked in. Use the `--auto-scale` flag in the pipeline, or run manually:

```bash
python -m aria2mesh.auto_scale_mesh \
    --aria-path data/my_object \
    --frame 30
```

### Prepare tracking data

Converts aria2mesh processed data to the FoundationPose directory format (6-digit file naming, cam_K.txt, etc.).

```bash
python -m aria2mesh.prepare_fp_data \
    --aria-path data/my_object \
    --start-frame 0 \
    --end-frame 198
```

This creates `foundationpose/` in the recording directory with `rgb/`, `depth/`, `masks/`, `mesh/`, and `cam_K.txt`.

### Run tracking

Invokes FoundationPose tracking via the `foundationpose` conda environment.

```bash
python -m aria2mesh.run_foundationpose \
    --aria-path data/my_object \
    --cuda-device 1
```

Outputs are saved to `foundationpose/fp_output/`:
- `ob_in_cam/` -- 4x4 pose matrices per frame (text files)
- `track_vis/` -- visualization PNGs with bounding box overlay
- `output.mp4` -- tracking video

## Data Format

After running the full pipeline, each recording has this structure:

```text
recording_name/
    video.vrs                           # Original Aria recording
    vrs_health_check.json
    mps_video_vrs/                      # Cloud MPS results (SLAM, hand tracking)
    on_device/                          # On-device results (Gen 2)
    processed/                          # Output from aria2mano + prepare_aria_data
        rgb/                            # Full-resolution RGB images
        rgb_small/                      # Center-cropped 512x512 RGB (.jpg)
        depth/                          # Stereo depth maps (uint16 mm)
        depth_rgb/                      # Depth reprojected to RGB frame (uint16 mm)
        object_mask_small/              # Binary object masks (512x512)
        intrinsics/                     # Per-frame 3x3 camera intrinsics (.npy)
        extrinsics/                     # Per-frame 4x4 T_world_camera (.npy)
        aria_data.pkl                   # Calibration and frame metadata
        point_cloud.npz                 # SLAM semidense point cloud
    aria2mesh/
        inputs/                         # MV-SAM3D inputs (from prepare_images_da3)
            images/                     # Multi-view RGB images (1.png - N.png)
            {object_name}/              # Per-view RGBA masks
            da3_output.npz              # Cameras, depth, pointmaps
            c2w_cam0.npy                # First frame camera-to-world transform
            point_cloud.npz             # SLAM semidense point cloud (copy)
            scene.glb                   # Aligned point cloud visualization
        outputs/                        # MV-SAM3D outputs (from run_mvsam3d)
            result.glb                  # Reconstructed mesh (canonical ~1m scale)
            result.ply                  # Gaussian splat representation
            params.npz                  # SAM3D pose parameters (scale, rotation, translation)
            result_merged_scene.glb             # Mesh merged with DA3 scene
            result_merged_scene_optimized.glb   # Pose-optimized merged scene (metric scale)
            result_pose_optimized.glb           # Pose-optimized mesh only
            result_autoscaled.glb               # Auto-scaled mesh (if --auto-scale)
            pose_optimization/
                optimized_params.npz            # Optimized pose parameters
    foundationpose/                     # FoundationPose data (from prepare_fp_data)
        rgb/                            # 6-digit .png RGB images
        depth/                          # 6-digit .png depth (symlinks to depth_rgb)
        masks/                          # Object mask for first frame
        mesh/                           # Symlink to scaled mesh
        cam_K.txt                       # 3x3 camera intrinsics (text)
        frame_map.txt                   # FP index to original frame index mapping
        fp_output/                      # FoundationPose tracking output
            ob_in_cam/                  # 4x4 pose matrices per frame (.txt)
            track_vis/                  # Visualization PNGs
            output.mp4                  # Tracking video
```

See [aria2mano](https://github.com/KevinyWu/aria2mano/tree/main?tab=readme-ov-file#data-format) for details on the contents of `processed/`.
