"""Wrapper to run MV-SAM3D inference from the aria2mesh root."""

import shutil
import subprocess
import sys
from pathlib import Path

import tyro

MVSAM3D_DIR = Path(__file__).resolve().parent.parent / "external" / "MV-SAM3D"


def main(
    aria_path: Path,
    mask_prompt: str,
    stage2_weight_source: str = "mixed",
    run_pose_optimization: bool = True,
    pose_opt_mask_erosion: int = 13,
    merge_da3_glb: bool = True,
) -> None:
    """Run MV-SAM3D weighted inference on an Aria recording."""
    aria2mesh_dir = aria_path.resolve() / "aria2mesh"
    input_path = aria2mesh_dir / "inputs"
    da3_output = input_path / "da3_output.npz"
    dataset_name = input_path.name

    # Snapshot existing runs so we can detect the new one
    vis_dir = MVSAM3D_DIR / "visualization" / dataset_name / mask_prompt
    existing = set(vis_dir.iterdir()) if vis_dir.exists() else set()

    cmd = [
        sys.executable,
        "run_inference_weighted.py",
        "--input_path",
        str(input_path),
        "--mask_prompt",
        mask_prompt,
        "--da3_output",
        str(da3_output),
        "--stage2_weight_source",
        stage2_weight_source,
        "--pose_opt_mask_erosion",
        str(pose_opt_mask_erosion),
    ]
    if run_pose_optimization:
        cmd.append("--run_pose_optimization")
    if merge_da3_glb:
        cmd.append("--merge_da3_glb")

    subprocess.run(cmd, cwd=MVSAM3D_DIR, check=True)

    # Move new output into aria2mesh/outputs/, replacing any previous run
    new_runs = set(vis_dir.iterdir()) - existing if vis_dir.exists() else set()
    output_dest = aria2mesh_dir / "outputs"
    if new_runs:
        if output_dest.exists():
            shutil.rmtree(output_dest)
        run_dir = next(iter(new_runs))
        shutil.move(str(run_dir), str(output_dest))
        print(f"Output: {output_dest}")


if __name__ == "__main__":
    tyro.cli(main)
