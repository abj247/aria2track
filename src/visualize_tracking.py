"""Visualize FoundationPose 6DoF tracking frame-by-frame with Viser.

Wraps aria2mano's viser_viewer_objects to provide per-frame playback of
tracked object poses overlaid on Aria camera views. Runs in the aria2mano
conda environment via subprocess.

Usage:
    python -m aria2mesh.visualize_tracking \
        --aria-path data/headphones_1 \
        --start-frame 0 --max-frames 280
"""

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import fast_simplification
import numpy as np
import trimesh
import tyro

ARIA2MANO_CONDA_ENV = "aria2mano"
MAX_FACES_FOR_VIEWER = 50000


def find_mesh(mesh_dir: Path) -> Optional[Path]:
    """Find the first mesh file in a directory."""
    for ext in ("*.glb", "*.obj", "*.ply"):
        matches = sorted(mesh_dir.glob(ext))
        if matches:
            return matches[0]
    return None


def main(
    aria_path: Path,
    start_frame: int = 0,
    max_frames: Optional[int] = None,
    mesh_file: Optional[Path] = None,
    port: int = 8080,
    downsample: int = 2,
) -> None:
    """Visualize FoundationPose tracking with per-frame playback.

    Args:
        aria_path: path to the Aria recording
        start_frame: first FP frame to visualize
        max_frames: number of frames to show (default: all)
        mesh_file: override mesh file (default: auto-detect from foundationpose/mesh/)
        port: Viser server port
        downsample: image downsample factor (2 = half resolution)
    """
    aria_path = Path(aria_path).resolve()
    fp_dir = aria_path / "foundationpose"

    pose_dir = fp_dir / "fp_output" / "ob_in_cam"
    if not pose_dir.exists():
        raise FileNotFoundError(
            f"No tracking output at {pose_dir}. "
            "Run: python -m aria2mesh.run_foundationpose --aria-path ..."
        )

    n_poses = len(list(pose_dir.glob("*.txt")))
    if n_poses == 0:
        raise FileNotFoundError(f"No pose files in {pose_dir}")

    if mesh_file is None:
        mesh_dir = fp_dir / "mesh"
        if mesh_dir.exists():
            mesh_file = find_mesh(mesh_dir)
        if mesh_file is None:
            raise FileNotFoundError(
                f"No mesh in {fp_dir / 'mesh'}. "
                "Run prepare_fp_data first."
            )
    mesh_file = Path(mesh_file).resolve()

    if max_frames is None:
        max_frames = n_poses

    mesh = trimesh.load(str(mesh_file), force="mesh")
    n_faces = len(mesh.faces)
    viewer_mesh = mesh_file
    tmp_mesh_path = None

    if n_faces > MAX_FACES_FOR_VIEWER:
        reduction = 1.0 - MAX_FACES_FOR_VIEWER / n_faces
        print(f"Decimating mesh for viewer: {n_faces} -> ~{MAX_FACES_FOR_VIEWER} faces")
        verts_out, faces_out = fast_simplification.simplify(
            mesh.vertices.astype(np.float32),
            mesh.faces.astype(np.int32),
            target_reduction=reduction,
        )
        simplified = trimesh.Trimesh(vertices=verts_out, faces=faces_out)
        tmp_mesh_path = Path(tempfile.mktemp(suffix=".obj"))
        simplified.export(str(tmp_mesh_path))
        viewer_mesh = tmp_mesh_path
        print(f"  Simplified: {len(simplified.vertices)} verts, {len(simplified.faces)} faces")

    print(f"Tracking visualization")
    print(f"  Recording:  {aria_path}")
    print(f"  Mesh:       {mesh_file.name} ({len(mesh.vertices)} verts)")
    print(f"  Poses:      {n_poses} frames")
    print(f"  Range:      start={start_frame}, max_frames={max_frames}")
    print(f"  Port:       {port}")

    try:
        cmd = [
            "conda", "run", "-n", ARIA2MANO_CONDA_ENV, "--no-capture-output",
            "python", "-m", "aria2mano.viser_viewer_objects",
            "--aria-path", str(aria_path),
            "--object-meshes", str(viewer_mesh),
            "--object-pose-dirs", str(pose_dir),
            "--start-frame", str(start_frame),
            "--max-frames", str(max_frames),
            "--port", str(port),
            "--downsample", str(downsample),
        ]
        subprocess.run(cmd, check=True)
    finally:
        if tmp_mesh_path is not None and tmp_mesh_path.exists():
            tmp_mesh_path.unlink()


if __name__ == "__main__":
    tyro.cli(main)
