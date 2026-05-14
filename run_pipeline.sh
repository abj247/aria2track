#!/usr/bin/env bash
# aria2mesh end-to-end pipeline runner
#
# Accepts all parameters via CLI arguments. If required arguments
# (--aria-path, --object-name) are not provided, prompts interactively.
#
# Usage:
#   ./run_pipeline.sh --aria-path /path/to/recording --object-name bottle [OPTIONS]
#   ./run_pipeline.sh   # interactive mode (prompts for required args)

set -e

# ---------- colors ----------
RED='\033[1;31m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
MAGENTA='\033[1;35m'
CYAN='\033[1;36m'
BOLD='\033[1m'
NC='\033[0m'

say_step_start() { echo -e "\n${BLUE}${BOLD}==> [STEP $1] Starting: $2${NC}"; }
say_step_done()  { echo -e "${GREEN}${BOLD}    [STEP $1] Completed: $2${NC}\n"; }
say_info()       { echo -e "${CYAN}   $1${NC}"; }
say_prompt()     { echo -ne "${YELLOW}${BOLD}?  $1${NC} "; }
say_warn()       { echo -e "${MAGENTA}!  $1${NC}"; }
say_err()        { echo -e "${RED}${BOLD}   $1${NC}"; }

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

# ---------- usage ----------
usage() {
    echo "Usage: $0 --aria-path PATH --object-name NAME [OPTIONS]"
    echo ""
    echo "Required (prompted interactively if omitted):"
    echo "  --aria-path PATH        Path to Aria recording"
    echo "  --object-name NAME      Object name for masks and MV-SAM3D prompt"
    echo ""
    echo "Optional:"
    echo "  --start-frame N         DA3 view selection start frame (default: 30)"
    echo "  --end-frame N           DA3 view selection end frame (default: last)"
    echo "  --n-views N             Number of multi-view images (default: 8)"
    echo "  --mask-prompt TEXT      MV-SAM3D mask prompt (default: object-name)"
    echo "  --clean-masks           Run mask preprocessing (binarize, denoise, resize)"
    echo "  --run-fp                Run FoundationPose tracking after mesh generation"
    echo "  --auto-scale            Auto-scale mesh from depth before FP (implies --run-fp)"
    echo "  --visualize             Launch Viser mesh viewer at the end"
    echo "  --visualize-tracking    Launch per-frame tracking viewer (uses aria2mano env)"
    echo "  --vis-track-start N     Tracking viewer start frame (default: 0)"
    echo "  --vis-track-max N       Tracking viewer max frames (default: all FP frames)"
    echo "  --cuda-device N         GPU device index (default: 1)"
    echo "  --port N                Viser server port (default: 8080)"
    echo "  --fp-start-frame N      FoundationPose tracking start frame (default: 0)"
    echo "  --fp-end-frame N        FoundationPose tracking end frame (default: last)"
    echo "  -h, --help              Show this help message"
    exit 0
}

# ---------- defaults ----------
ARIA_PATH=""
OBJECT_NAME=""
START_FRAME=30
END_FRAME=""
N_VIEWS=8
MASK_PROMPT=""
CLEAN_MASKS=false
RUN_FP=false
AUTO_SCALE=false
VISUALIZE=false
VISUALIZE_TRACKING=false
VIS_TRACK_START=0
VIS_TRACK_MAX=""
CUDA_DEVICE=1
PORT=8080
FP_START_FRAME=0
FP_END_FRAME=""

# ---------- argument parsing ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --aria-path)       ARIA_PATH="$2"; shift 2 ;;
        --object-name)     OBJECT_NAME="$2"; shift 2 ;;
        --start-frame)     START_FRAME="$2"; shift 2 ;;
        --end-frame)       END_FRAME="$2"; shift 2 ;;
        --n-views)         N_VIEWS="$2"; shift 2 ;;
        --mask-prompt)     MASK_PROMPT="$2"; shift 2 ;;
        --clean-masks)     CLEAN_MASKS=true; shift ;;
        --run-fp)          RUN_FP=true; shift ;;
        --auto-scale)      AUTO_SCALE=true; RUN_FP=true; shift ;;
        --visualize)       VISUALIZE=true; shift ;;
        --visualize-tracking) VISUALIZE_TRACKING=true; shift ;;
        --vis-track-start) VIS_TRACK_START="$2"; shift 2 ;;
        --vis-track-max)   VIS_TRACK_MAX="$2"; shift 2 ;;
        --cuda-device)     CUDA_DEVICE="$2"; shift 2 ;;
        --port)            PORT="$2"; shift 2 ;;
        --fp-start-frame)  FP_START_FRAME="$2"; shift 2 ;;
        --fp-end-frame)    FP_END_FRAME="$2"; shift 2 ;;
        -h|--help)         usage ;;
        *)                 say_err "Unknown argument: $1"; usage ;;
    esac
done

# ---------- interactive prompts for missing required args ----------
echo -e "${BOLD}${MAGENTA}=========================================="
echo -e "  aria2track Pipeline Runner"
echo -e "==========================================${NC}"

if [[ -z "$ARIA_PATH" ]]; then
    say_prompt "Path to Aria recording (e.g., /nas/abhishek/aria2mano/vrs_fp/bottle_1):"
    read -r ARIA_PATH
fi
if [[ -z "$ARIA_PATH" || ! -d "$ARIA_PATH" ]]; then
    say_err "Path does not exist: $ARIA_PATH"
    exit 1
fi
if [[ ! -d "$ARIA_PATH/processed" ]]; then
    say_err "No 'processed/' subdir found in $ARIA_PATH -- run aria2mano first."
    exit 1
fi

if [[ -z "$OBJECT_NAME" ]]; then
    say_prompt "Object name (used for masks and MV-SAM3D prompt):"
    read -r OBJECT_NAME
fi
if [[ -z "$OBJECT_NAME" ]]; then
    say_err "Object name is required."
    exit 1
fi

MASK_PROMPT=${MASK_PROMPT:-$OBJECT_NAME}

# ---------- derived variables ----------
RECORDING=$(basename "$ARIA_PATH")
mkdir -p "$REPO_DIR/data"

# Resolve ARIA_PATH to an absolute real path.  If the user passed a path
# inside data/ that is already a symlink, follow it to the real target so
# that Step 0 re-creates the link correctly (avoids circular symlinks).
if [[ -L "$ARIA_PATH" ]]; then
    ARIA_PATH_ABS=$(readlink -f "$ARIA_PATH")
else
    ARIA_PATH_ABS=$(cd "$ARIA_PATH" && pwd)
fi

LINK_PATH="$REPO_DIR/data/$RECORDING"
RELATIVE_PATH="data/$RECORDING"

END_FRAME_ARG=""
if [[ -n "$END_FRAME" ]]; then
    END_FRAME_ARG="--end-frame $END_FRAME"
fi
FP_END_ARG=""
if [[ -n "$FP_END_FRAME" ]]; then
    FP_END_ARG="--end-frame $FP_END_FRAME"
fi

echo ""
say_info "Recording:    $RECORDING"
say_info "Object:       $OBJECT_NAME"
say_info "Frames:       $START_FRAME - ${END_FRAME:-last}"
say_info "Views:        $N_VIEWS"
say_info "GPU:          $CUDA_DEVICE"
say_info "Clean masks:  $CLEAN_MASKS"
say_info "Run FP:       $RUN_FP"
say_info "Auto-scale:   $AUTO_SCALE"
say_info "Visualize:    $VISUALIZE"
say_info "Vis tracking: $VISUALIZE_TRACKING"

# ---------- Step 0: symlink ----------
say_step_start 0 "Symlink recording into data/"
ln -sfn "$ARIA_PATH_ABS" "$LINK_PATH"
say_info "Symlinked: $LINK_PATH -> $ARIA_PATH_ABS"
say_step_done 0 "Symlink created"

# ---------- Step 1: prepare_aria_data ----------
if [[ -d "$ARIA_PATH_ABS/processed/rgb_small" && -d "$ARIA_PATH_ABS/processed/depth_rgb" \
   && -d "$ARIA_PATH_ABS/processed/intrinsics" && -d "$ARIA_PATH_ABS/processed/extrinsics" ]]; then
    say_step_start 1 "prepare_aria_data (rgb_small, depth_rgb, intrinsics, extrinsics)"
    say_info "Outputs already exist -- skipping (delete processed/ subdirs to force re-run)"
    say_step_done 1 "prepare_aria_data (cached)"
else
    say_step_start 1 "prepare_aria_data (rgb_small, depth_rgb, intrinsics, extrinsics)"
    conda run -n aria2track --no-capture-output python -m aria2mesh.prepare_aria_data \
        --aria-path "$RELATIVE_PATH"
    say_step_done 1 "prepare_aria_data finished"
fi

# ---------- Step 2: mask preprocessing (optional) ----------
if [[ "$CLEAN_MASKS" == "true" ]]; then
    say_step_start 2 "Preprocess masks (binarize, denoise, resize, pad, dummy-fill)"
    MASK_DIR="$ARIA_PATH_ABS/processed/object_mask_small"
    RAW_DIR="$ARIA_PATH_ABS/processed/object_mask_small_raw"
    RGB_DIR="$ARIA_PATH_ABS/processed/rgb_small"
    NOISE_MIN_PIXELS=${NOISE_MIN_PIXELS:-100}

    if [[ ! -d "$MASK_DIR" ]]; then
        say_err "Mask dir not found: $MASK_DIR"
        exit 1
    fi

    n_rgb=$(ls "$RGB_DIR" | wc -l)
    n_before=$(ls "$MASK_DIR" 2>/dev/null | wc -l)
    say_info "rgb_small frames: $n_rgb | existing masks: $n_before"
    say_info "Noise threshold (min component pixels): $NOISE_MIN_PIXELS"

    NOISE_MIN_PIXELS="$NOISE_MIN_PIXELS" conda run -n aria2track --no-capture-output python3 - <<PYEOF
from PIL import Image
from scipy import ndimage
import numpy as np, os, shutil

mask_dir  = "$MASK_DIR"
raw_dir   = "$RAW_DIR"
rgb_dir   = "$RGB_DIR"
target    = 512
wm_ratio  = 490 / 512
noise_min = int(os.environ.get("NOISE_MIN_PIXELS", "100"))

os.makedirs(raw_dir, exist_ok=True)
rgb_stems = sorted(os.path.splitext(f)[0] for f in os.listdir(rgb_dir))
pad = len(rgb_stems[0])

backed_up = 0
for fname in sorted(os.listdir(mask_dir)):
    stem = os.path.splitext(fname)[0]
    if not (fname.lower().endswith(".png") and stem.isdigit() and len(stem) != pad):
        continue
    src = os.path.join(mask_dir, fname)
    bak = os.path.join(raw_dir, fname)
    shutil.move(src, bak)
    backed_up += 1
    img = np.array(Image.open(bak))
    if img.ndim == 3:
        binary = (np.any(img[..., :3] > 0, axis=-1).astype(np.uint8)) * 255
    else:
        binary = (img > 0).astype(np.uint8) * 255
    H = binary.shape[0]
    binary[int(H * wm_ratio):, :] = 0
    labeled, n_comp = ndimage.label(binary > 0)
    if n_comp:
        sizes = ndimage.sum(binary > 0, labeled, range(1, n_comp + 1))
        keep_ids = np.where(sizes >= noise_min)[0] + 1
        binary = np.where(np.isin(labeled, keep_ids), 255, 0).astype(np.uint8)
    if binary.shape != (target, target):
        H, W = binary.shape
        side = min(H, W); y0 = (H - side) // 2; x0 = (W - side) // 2
        binary = np.array(
            Image.fromarray(binary[y0:y0+side, x0:x0+side]).resize(
                (target, target), Image.NEAREST
            )
        )
    dst_stem = stem.zfill(pad)
    Image.fromarray(binary).save(os.path.join(mask_dir, f"{dst_stem}.png"))
    print(f"  {fname} -> {dst_stem}.png  (nonzero={int((binary > 0).sum())})")
print(f"Backed up + processed {backed_up} raw overlay(s) into {raw_dir}")

cleaned = 0
for fname in sorted(os.listdir(mask_dir)):
    stem = os.path.splitext(fname)[0]
    if not (fname.lower().endswith(".png") and stem.isdigit() and len(stem) == pad):
        continue
    p = os.path.join(mask_dir, fname)
    img = np.array(Image.open(p))
    if img.ndim == 3:
        img = (np.any(img[..., :3] > 0, axis=-1).astype(np.uint8)) * 255
    if not (img > 0).any():
        continue
    labeled, n_comp = ndimage.label(img > 0)
    sizes = ndimage.sum(img > 0, labeled, range(1, n_comp + 1))
    keep_ids = np.where(sizes >= noise_min)[0] + 1
    clean = np.where(np.isin(labeled, keep_ids), 255, 0).astype(np.uint8)
    if not np.array_equal(clean, img):
        Image.fromarray(clean).save(p)
        cleaned += 1
print(f"Denoised {cleaned} already-padded mask(s)")

zeros = np.zeros((target, target), dtype=np.uint8)
mask_stems = {os.path.splitext(f)[0] for f in os.listdir(mask_dir)}
missing = sorted(set(rgb_stems) - mask_stems)
for stem in missing:
    Image.fromarray(zeros).save(os.path.join(mask_dir, f"{stem}.png"))
print(f"Created {len(missing)} dummy all-zero mask(s)")
PYEOF

    n_after=$(ls "$MASK_DIR" | wc -l)
    say_info "masks now: $n_after (rgb: $n_rgb)"
    if [[ "$n_after" != "$n_rgb" ]]; then
        say_err "Mask count ($n_after) != RGB count ($n_rgb)"
        exit 1
    fi
    say_step_done 2 "Masks ready"
else
    say_info "Skipping mask preprocessing (use --clean-masks to enable)"
fi

# ---------- Steps 3+4: prepare_images_da3 + run_mvsam3d ----------
MESH_OUTPUT="$ARIA_PATH_ABS/aria2mesh/outputs/result_merged_scene_optimized.glb"
if [[ -f "$MESH_OUTPUT" ]]; then
    say_step_start 3 "prepare_images_da3 (pick $N_VIEWS views, build DA3 inputs)"
    say_info "Skipping -- mesh already exists: $(basename $MESH_OUTPUT)"
    say_step_done 3 "prepare_images_da3 (cached)"

    say_step_start 4 "run_mvsam3d (MV-SAM3D inference on GPU $CUDA_DEVICE)"
    say_info "Skipping -- mesh already exists: $(basename $MESH_OUTPUT)"
    say_info "Delete aria2mesh/outputs/ to force re-run"
    say_step_done 4 "run_mvsam3d (cached)"
else
    say_step_start 3 "prepare_images_da3 (pick $N_VIEWS views, build DA3 inputs)"
    conda run -n aria2track --no-capture-output python -m aria2mesh.prepare_images_da3 \
        --aria-path "$RELATIVE_PATH" \
        --start-frame "$START_FRAME" \
        $END_FRAME_ARG \
        --n-views "$N_VIEWS" \
        --object-name "$OBJECT_NAME"
    say_step_done 3 "DA3 inputs ready"

    say_step_start 4 "run_mvsam3d (MV-SAM3D inference on GPU $CUDA_DEVICE)"
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" conda run -n aria2track --no-capture-output \
        python -m aria2mesh.run_mvsam3d \
        --aria-path "$RELATIVE_PATH" \
        --mask-prompt "$MASK_PROMPT"
    say_step_done 4 "MV-SAM3D finished -- outputs in $ARIA_PATH_ABS/aria2mesh/outputs/"
fi

# ---------- Step 5: auto_scale_mesh (optional) ----------
MESH_FOR_FP=""
if [[ "$AUTO_SCALE" == "true" ]]; then
    say_step_start 5 "auto_scale_mesh (compute metric scale from depth + mask)"

    # Find first non-empty mask frame
    SCALE_FRAME=$(conda run -n aria2track --no-capture-output python3 -c "
import cv2, sys
from pathlib import Path
mask_dir = Path('$ARIA_PATH_ABS/processed/object_mask_small')
for f in sorted(mask_dir.glob('*.png')):
    m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
    if m is not None and m.max() > 0:
        print(int(f.stem))
        sys.exit(0)
print('0')
")
    say_info "Using frame $SCALE_FRAME for auto-scale (first non-empty mask)"

    conda run -n aria2track --no-capture-output python -m aria2mesh.auto_scale_mesh \
        --aria-path "$RELATIVE_PATH" \
        --frame "$SCALE_FRAME"

    MESH_FOR_FP="$ARIA_PATH_ABS/aria2mesh/outputs/result_autoscaled.glb"
    say_step_done 5 "Auto-scaled mesh saved"
else
    say_info "Skipping auto-scale (use --auto-scale to enable)"
fi

# ---------- Step 6: prepare_fp_data (if --run-fp) ----------
if [[ "$RUN_FP" == "true" ]]; then
    say_step_start 6 "prepare_fp_data (convert to FoundationPose format)"

    FP_MESH_ARG=""
    if [[ -n "$MESH_FOR_FP" && -f "$MESH_FOR_FP" ]]; then
        FP_MESH_ARG="--mesh-file $MESH_FOR_FP"
    fi

    conda run -n aria2track --no-capture-output python -m aria2mesh.prepare_fp_data \
        --aria-path "$RELATIVE_PATH" \
        --start-frame "$FP_START_FRAME" \
        $FP_END_ARG \
        $FP_MESH_ARG
    say_step_done 6 "FoundationPose data prepared"

    # ---------- Step 7: run_foundationpose ----------
    say_step_start 7 "run_foundationpose (6DoF tracking on GPU $CUDA_DEVICE)"
    conda run -n aria2track --no-capture-output python -m aria2mesh.run_foundationpose \
        --aria-path "$RELATIVE_PATH" \
        --cuda-device "$CUDA_DEVICE"
    say_step_done 7 "FoundationPose tracking complete"
else
    say_info "Skipping FoundationPose (use --run-fp to enable)"
fi

# ---------- Step 8: visualize mesh (optional) ----------
if [[ "$VISUALIZE" == "true" ]]; then
    say_step_start 8 "visualize_scene (Viser web viewer on port $PORT)"
    say_info "Open http://localhost:$PORT in your browser. Ctrl+C to stop."
    conda run -n aria2track --no-capture-output python -m aria2mesh.visualize_scene \
        --aria-path "$RELATIVE_PATH" \
        --port "$PORT"
    say_step_done 8 "Visualization finished"
fi

# ---------- Step 9: visualize tracking (optional) ----------
if [[ "$VISUALIZE_TRACKING" == "true" ]]; then
    # Determine max frames: use provided value, else FP end frame, else all poses
    if [[ -z "$VIS_TRACK_MAX" ]]; then
        if [[ -n "$FP_END_FRAME" ]]; then
            VIS_TRACK_MAX=$((FP_END_FRAME - FP_START_FRAME + 1))
        else
            VIS_TRACK_MAX=$(ls "$ARIA_PATH_ABS/foundationpose/fp_output/ob_in_cam/"*.txt 2>/dev/null | wc -l)
        fi
    fi

    say_step_start 9 "visualize_tracking (per-frame playback on port $PORT)"
    say_info "Frames: start=$VIS_TRACK_START, max=$VIS_TRACK_MAX"
    say_info "Open http://localhost:$PORT in your browser. Ctrl+C to stop."
    conda run -n aria2track --no-capture-output python -m aria2mesh.visualize_tracking \
        --aria-path "$RELATIVE_PATH" \
        --start-frame "$VIS_TRACK_START" \
        --max-frames "$VIS_TRACK_MAX" \
        --port "$PORT"
    say_step_done 9 "Tracking visualization finished"
fi

# ---------- summary ----------
echo -e "\n${GREEN}${BOLD}Pipeline complete for $RECORDING / $OBJECT_NAME${NC}\n"

echo -e "${CYAN}${BOLD}Manual commands for each step:${NC}"
echo -e "${CYAN}  # Step 1: Prepare Aria data${NC}"
echo "  python -m aria2mesh.prepare_aria_data --aria-path $RELATIVE_PATH"
echo ""
echo -e "${CYAN}  # Step 2: Mask preprocessing (optional)${NC}"
echo "  # See run_pipeline.sh --clean-masks"
echo ""
echo -e "${CYAN}  # Step 3: Prepare multi-view images${NC}"
echo "  python -m aria2mesh.prepare_images_da3 --aria-path $RELATIVE_PATH --start-frame $START_FRAME ${END_FRAME_ARG} --n-views $N_VIEWS --object-name $OBJECT_NAME"
echo ""
echo -e "${CYAN}  # Step 4: Run MV-SAM3D${NC}"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -m aria2mesh.run_mvsam3d --aria-path $RELATIVE_PATH --mask-prompt $MASK_PROMPT"
echo ""
echo -e "${CYAN}  # Step 5: Auto-scale mesh (optional)${NC}"
echo "  python -m aria2mesh.auto_scale_mesh --aria-path $RELATIVE_PATH --frame 0"
echo ""
echo -e "${CYAN}  # Step 6: Prepare FoundationPose data${NC}"
echo "  python -m aria2mesh.prepare_fp_data --aria-path $RELATIVE_PATH --start-frame $FP_START_FRAME ${FP_END_ARG}"
echo ""
echo -e "${CYAN}  # Step 7: Run FoundationPose tracking${NC}"
echo "  python -m aria2mesh.run_foundationpose --aria-path $RELATIVE_PATH --cuda-device $CUDA_DEVICE  # uses sam3d-objects conda env internally"
echo ""
echo -e "${CYAN}  # Visualize mesh scene${NC}"
echo "  python -m aria2mesh.visualize_scene --aria-path $RELATIVE_PATH --port $PORT"
echo ""
echo -e "${CYAN}  # Visualize per-frame tracking (uses aria2mano env)${NC}"
echo "  python -m aria2mesh.visualize_tracking --aria-path $RELATIVE_PATH --start-frame 0 --max-frames ${VIS_TRACK_MAX:-all} --port $PORT"
echo ""
echo -e "${CYAN}  # Compare meshes side-by-side${NC}"
echo "  python -m aria2mesh.compare_meshes --meshes path1.glb path2.glb --spacing 0.5"
