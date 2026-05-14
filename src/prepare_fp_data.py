"""Prepare FoundationPose-compatible data from aria2mesh processed recordings.

Converts aria2mesh's processed data into the directory layout expected by
FoundationPose's YcbineoatReader:
    {output}/rgb/         - 6-digit .png RGB images
    {output}/depth/       - 6-digit .png uint16 depth in mm
    {output}/masks/       - object mask for the first frame
    {output}/mesh/        - symlink to the mesh file
    {output}/cam_K.txt    - 3x3 camera intrinsics (text)
    {output}/frame_map.txt - mapping from FP index to original frame index
"""

import os
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tyro
from PIL import Image

GLB_PREFERENCE = [
    "result_merged_scene_optimized.glb",
    "result_merged_scene.glb",
    "result.glb",
]


def find_best_mesh(outputs_dir: Path) -> Optional[Path]:
    """Find the best mesh file from aria2mesh outputs."""
    for name in GLB_PREFERENCE:
        p = outputs_dir / name
        if p.exists():
            return p
    for ext in ("*.glb", "*.obj", "*.ply"):
        matches = list(outputs_dir.glob(ext))
        if matches:
            return matches[0]
    return None


def main(
    aria_path: Path,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    mesh_file: Optional[Path] = None,
    output: Optional[Path] = None,
) -> None:
    """Convert aria2mesh processed data to FoundationPose format.

    Args:
        aria_path: path to the Aria recording
        start_frame: first frame to include (becomes 000000 in FP)
        end_frame: last frame to include (default: last available)
        mesh_file: mesh for FP to track (default: best from aria2mesh/outputs/)
        output: output directory (default: {aria_path}/foundationpose/)
    """
    aria_path = Path(aria_path)
    processed = aria_path / "processed"

    required = {
        "rgb_small": processed / "rgb_small",
        "depth_rgb": processed / "depth_rgb",
        "object_mask_small": processed / "object_mask_small",
        "intrinsics": processed / "intrinsics",
    }
    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{name}/ not found at {path}. "
                "Run: python -m aria2mesh.prepare_aria_data --aria-path "
                f"{aria_path}"
            )

    rgb_files = sorted((processed / "rgb_small").glob("*.jpg"))
    if not rgb_files:
        rgb_files = sorted((processed / "rgb_small").glob("*.png"))
    if not rgb_files:
        raise FileNotFoundError(f"No images in {processed / 'rgb_small'}")

    n_total = len(rgb_files)
    stem_to_file = {int(f.stem): f for f in rgb_files}
    all_stems = sorted(stem_to_file.keys())

    if end_frame is None:
        end_frame = all_stems[-1]
    if start_frame < all_stems[0]:
        start_frame = all_stems[0]
    if end_frame > all_stems[-1]:
        print(f"  WARNING: --end-frame {end_frame} exceeds max {all_stems[-1]}, clamping")
        end_frame = all_stems[-1]
    if start_frame > end_frame:
        raise ValueError(f"--start-frame {start_frame} > --end-frame {end_frame}")

    selected = [s for s in all_stems if start_frame <= s <= end_frame]
    print(f"Found {n_total} total frames, selected {len(selected)} "
          f"(frames {start_frame}-{end_frame})")

    if output is None:
        output = aria_path / "foundationpose"
    fp_dir = Path(output)

    for subdir in ["rgb", "depth", "masks", "mesh"]:
        (fp_dir / subdir).mkdir(parents=True, exist_ok=True)

    frame_map = []
    first_mask_empty = False

    for i, orig_frame in enumerate(selected):
        fp_name = f"{i:06d}.png"
        stem = f"{orig_frame:08d}"

        rgb_src = processed / "rgb_small" / f"{stem}.jpg"
        if not rgb_src.exists():
            rgb_src = processed / "rgb_small" / f"{stem}.png"
        img = Image.open(rgb_src)
        img.save(fp_dir / "rgb" / fp_name)

        depth_src = (processed / "depth_rgb" / f"{stem}.png").resolve()
        depth_dst = fp_dir / "depth" / fp_name
        if depth_dst.exists() or depth_dst.is_symlink():
            depth_dst.unlink()
        os.symlink(str(depth_src), str(depth_dst))

        if i == 0:
            mask_src = processed / "object_mask_small" / f"{stem}.png"
            if mask_src.exists():
                mask = cv2.imread(str(mask_src), cv2.IMREAD_GRAYSCALE)
                if mask is not None and mask.max() == 0:
                    first_mask_empty = True
                shutil.copy2(str(mask_src), str(fp_dir / "masks" / fp_name))
            else:
                first_mask_empty = True
                zeros = np.zeros((512, 512), dtype=np.uint8)
                cv2.imwrite(str(fp_dir / "masks" / fp_name), zeros)

        frame_map.append(f"{i:06d} {orig_frame:08d}")

    K = np.load(str(processed / "intrinsics" / f"{selected[0]:08d}.npy"))
    np.savetxt(str(fp_dir / "cam_K.txt"), K, fmt="%.18e")

    with open(fp_dir / "frame_map.txt", "w") as f:
        f.write("# fp_index original_frame\n")
        for line in frame_map:
            f.write(line + "\n")

    if mesh_file is None:
        outputs_dir = aria_path / "aria2mesh" / "outputs"
        mesh_file = find_best_mesh(outputs_dir) if outputs_dir.exists() else None
    if mesh_file is not None:
        mesh_file = Path(mesh_file)
        if mesh_file.exists():
            dest = fp_dir / "mesh" / mesh_file.name
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            os.symlink(str(mesh_file.resolve()), str(dest))
            print(f"Mesh: {mesh_file.name}")
        else:
            print(f"WARNING: Mesh file not found: {mesh_file}")
    else:
        print("WARNING: No mesh file found. Run MV-SAM3D first or pass --mesh-file.")

    if first_mask_empty:
        print(
            f"WARNING: Mask at frame {selected[0]} is all-zero. "
            "FoundationPose registration will fail. "
            "Select a --start-frame where the object is visible and masked."
        )

    print(f"\nPrepared {len(selected)} frames -> {fp_dir}")
    print(f"  RGB:   {fp_dir / 'rgb'}")
    print(f"  Depth: {fp_dir / 'depth'} (symlinks)")
    print(f"  Mask:  {fp_dir / 'masks' / '000000.png'}")
    print(f"  K:     {fp_dir / 'cam_K.txt'}")


if __name__ == "__main__":
    tyro.cli(main)
