"""Scale an aria2mesh canonical mesh to metric dimensions using depth + mask.

Uses the existing center-cropped depth (depth_rgb/) and object masks
(object_mask_small/) which are already in the same coordinate space with
matching intrinsics. Computes real-world object diameter from the masked
point cloud and scales the mesh accordingly.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import trimesh
import tyro
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN


def compute_diameter(points: np.ndarray) -> float:
    """Max pairwise distance using convex hull vertices."""
    if len(points) < 2:
        raise ValueError(f"Need at least 2 points, got {len(points)}")
    if len(points) > 500:
        try:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
        except Exception:
            hull_points = points
    else:
        hull_points = points
    return pdist(hull_points).max()


def depth_to_masked_pointcloud(
    depth_m: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    min_depth: float = 0.01,
    max_depth: float = 3.0,
) -> np.ndarray:
    """Unproject masked depth to 3D point cloud, filtering noise with DBSCAN."""
    H, W = depth_m.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    valid = mask & (depth_m > min_depth) & (depth_m < max_depth)

    z = depth_m[valid]
    u = us[valid].astype(np.float32)
    v = vs[valid].astype(np.float32)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts = np.stack([x, y, z], axis=1)

    if len(pts) > 20:
        clustering = DBSCAN(eps=0.015, min_samples=10).fit(pts)
        labels = clustering.labels_
        valid_labels = labels[labels >= 0]
        if len(valid_labels) > 0:
            largest = np.bincount(valid_labels).argmax()
            keep = labels == largest
            removed = (~keep).sum()
            if removed > 0:
                print(f"  Filtered {removed} outlier pts, kept {keep.sum()}")
                pts = pts[keep]

    return pts


def main(
    aria_path: Path,
    mesh_file: Optional[Path] = None,
    output: Optional[Path] = None,
    frame: int = 0,
    min_depth: float = 0.01,
    max_depth: float = 3.0,
) -> None:
    """Scale canonical mesh to metric dimensions using depth + mask.

    Args:
        aria_path: path to the Aria recording
        mesh_file: mesh to scale (default: aria2mesh/outputs/result.glb)
        output: output path (default: aria2mesh/outputs/result_autoscaled.glb)
        frame: frame index to use for depth/mask (must have non-empty mask)
        min_depth: minimum valid depth in meters
        max_depth: maximum valid depth in meters
    """
    processed = Path(aria_path) / "processed"
    outputs_dir = Path(aria_path) / "aria2mesh" / "outputs"
    stem = f"{frame:08d}"

    intrinsics_path = processed / "intrinsics" / f"{stem}.npy"
    depth_path = processed / "depth_rgb" / f"{stem}.png"
    mask_path = processed / "object_mask_small" / f"{stem}.png"

    for p, desc in [
        (intrinsics_path, "intrinsics"),
        (depth_path, "depth"),
        (mask_path, "mask"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{desc} not found: {p}")

    if mesh_file is None:
        mesh_file = outputs_dir / "result.glb"
    if not Path(mesh_file).exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_file}")

    print(f"[1/4] Loading depth, mask, and intrinsics for frame {frame}...")
    K = np.load(str(intrinsics_path))
    depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    depth_m = depth_mm.astype(np.float32) / 1000.0
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127
    print(f"  K: fx={K[0,0]:.2f}, fy={K[1,1]:.2f}")
    print(f"  Depth: {depth_m.shape}, mask pixels: {mask.sum()}")

    if mask.sum() == 0:
        raise RuntimeError(
            f"Mask at frame {frame} is all-zero. "
            "Select a frame with a visible object mask (--frame N)."
        )

    print("[2/4] Creating masked point cloud...")
    pts = depth_to_masked_pointcloud(depth_m, mask, K, min_depth, max_depth)
    print(f"  Point cloud: {len(pts)} points")
    if len(pts) < 10:
        raise RuntimeError(f"Only {len(pts)} points -- mask/depth mismatch?")

    pc_diameter = compute_diameter(pts)
    print(f"  Real-world diameter: {pc_diameter:.4f} m ({pc_diameter * 1000:.1f} mm)")

    print(f"[3/4] Loading mesh from {mesh_file}...")
    mesh = trimesh.load(str(mesh_file), force="mesh")
    mesh_diameter = compute_diameter(np.array(mesh.vertices))
    print(f"  Mesh diameter: {mesh_diameter:.4f} (mesh units)")

    scale_factor = pc_diameter / mesh_diameter
    print(f"[4/4] Scaling mesh...")
    print(f"  Scale factor: {scale_factor:.6f}")

    center = mesh.vertices.mean(axis=0)
    mesh.vertices -= center
    mesh.vertices *= scale_factor

    scaled_diameter = compute_diameter(np.array(mesh.vertices))
    print(f"  Scaled diameter: {scaled_diameter:.4f} m")
    print(f"  Scaled extents: {mesh.extents}")

    if output is None:
        mesh_stem = Path(mesh_file).stem
        output = outputs_dir / f"{mesh_stem}_autoscaled.glb"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output))
    print(f"  Saved: {output}")


if __name__ == "__main__":
    tyro.cli(main)
