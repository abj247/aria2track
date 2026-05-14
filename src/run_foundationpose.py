"""Run FoundationPose 6DoF object tracking on prepared data.

Invokes FoundationPose's run_demo_old.py via subprocess in the
``sam3d-objects`` conda environment. The FoundationPose repository path
is read from the FOUNDATIONPOSE_DIR environment variable.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import tyro

DEFAULT_FP_DIR = Path("/nas/abhishek/hamer/FoundationPose")
FP_CONDA_ENV = "sam3d-objects"


def find_mesh(mesh_dir: Path) -> Optional[Path]:
    """Find the first mesh file in a directory."""
    for ext in ("*.glb", "*.obj", "*.ply"):
        matches = sorted(mesh_dir.glob(ext))
        if matches:
            return matches[0]
    return None


def main(
    aria_path: Path,
    fp_repo: Optional[Path] = None,
    mesh_file: Optional[Path] = None,
    est_refine_iter: int = 5,
    track_refine_iter: int = 2,
    debug: int = 2,
    cuda_device: int = 1,
) -> None:
    """Run FoundationPose tracking on a prepared recording.

    Args:
        aria_path: path to the Aria recording
        fp_repo: path to FoundationPose repo (default: FOUNDATIONPOSE_DIR env var)
        mesh_file: override mesh file (default: auto-detect from foundationpose/mesh/)
        est_refine_iter: refinement iterations for initial pose estimation
        track_refine_iter: refinement iterations for tracking
        debug: debug level (0=none, 1=minimal, 2=images+video, 3=extra)
        cuda_device: GPU device index
    """
    aria_path = Path(aria_path)
    fp_data_dir = aria_path / "foundationpose"

    if fp_repo is None:
        fp_repo = Path(os.environ.get("FOUNDATIONPOSE_DIR", str(DEFAULT_FP_DIR)))
    fp_repo = Path(fp_repo)

    if not fp_repo.exists() or not (fp_repo / "run_demo_old.py").exists():
        raise FileNotFoundError(
            f"FoundationPose repo not found at {fp_repo}. "
            "Set FOUNDATIONPOSE_DIR environment variable to the correct path."
        )

    for subdir in ["rgb", "depth", "masks"]:
        if not (fp_data_dir / subdir).exists():
            raise FileNotFoundError(
                f"{subdir}/ not found in {fp_data_dir}. "
                "Run: python -m aria2mesh.prepare_fp_data "
                f"--aria-path {aria_path}"
            )
    if not (fp_data_dir / "cam_K.txt").exists():
        raise FileNotFoundError(
            f"cam_K.txt not found in {fp_data_dir}. "
            "Run: python -m aria2mesh.prepare_fp_data "
            f"--aria-path {aria_path}"
        )

    if mesh_file is None:
        mesh_dir = fp_data_dir / "mesh"
        if mesh_dir.exists():
            mesh_file = find_mesh(mesh_dir)
        if mesh_file is None:
            raise FileNotFoundError(
                f"No mesh found in {fp_data_dir / 'mesh'}. "
                "Ensure a mesh file exists (run MV-SAM3D and prepare_fp_data)."
            )
    mesh_file = Path(mesh_file).resolve()

    debug_dir = fp_data_dir / "fp_output"

    n_frames = len(list((fp_data_dir / "rgb").glob("*.png")))
    print(f"FoundationPose tracking")
    print(f"  Data:   {fp_data_dir} ({n_frames} frames)")
    print(f"  Mesh:   {mesh_file.name}")
    print(f"  Output: {debug_dir}")
    print(f"  GPU:    {cuda_device}")
    print(
        f"  NOTE: run_demo_old.py hardcodes CUDA_VISIBLE_DEVICES=1 at line 19. "
        f"To use a different GPU, edit that line in {fp_repo / 'run_demo_old.py'}."
    )

    fp_cmd = (
        f"cd {fp_repo} && "
        f"CUDA_VISIBLE_DEVICES={cuda_device} "
        f"python run_demo_old.py "
        f"--mesh_file {mesh_file} "
        f"--test_scene_dir {fp_data_dir.resolve()} "
        f"--est_refine_iter {est_refine_iter} "
        f"--track_refine_iter {track_refine_iter} "
        f"--debug {debug} "
        f"--debug_dir {debug_dir.resolve()}"
    )
    cmd = [
        "conda", "run", "-n", FP_CONDA_ENV, "--no-capture-output",
        "bash", "-c", fp_cmd,
    ]

    subprocess.run(cmd, check=True)

    print(f"\nTracking complete.")
    if (debug_dir / "ob_in_cam").exists():
        n_poses = len(list((debug_dir / "ob_in_cam").glob("*.txt")))
        print(f"  Poses: {debug_dir / 'ob_in_cam'} ({n_poses} files)")
    if (debug_dir / "output.mp4").exists():
        print(f"  Video: {debug_dir / 'output.mp4'}")


if __name__ == "__main__":
    tyro.cli(main)
