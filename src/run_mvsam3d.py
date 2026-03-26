"""Wrapper to run MV-SAM3D inference from the aria2mesh root."""

import subprocess
import sys
from pathlib import Path

MVSAM3D_DIR = Path(__file__).resolve().parent.parent / "external" / "MV-SAM3D"


def main() -> None:
    subprocess.run(
        [sys.executable, "run_inference_weighted.py", *sys.argv[1:]],
        cwd=MVSAM3D_DIR,
        check=True,
    )


if __name__ == "__main__":
    main()
