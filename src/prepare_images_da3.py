"""Prepare images + masks for MV-SAM3D and da3_output.npz from GT cameras/depth.

Expected input directory structure (Aria processed data):
    {input}/processed/rgb_small/         - RGB images (*.jpg or *.png)
    {input}/processed/object_mask_small/ - grayscale masks (*.png)
    {input}/processed/intrinsics/        - per-frame 3x3 intrinsics (*.npy)
    {input}/processed/extrinsics/        - per-frame 4x4 c2w matrices (*.npy)
    {input}/processed/depth_rgb/         - uint16 depth in mm (*.png)
"""

import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import trimesh
import tyro
from PIL import Image


def depth_to_pointmap(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Unproject depth map to 3D pointmap in camera space."""
    H, W = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    return np.stack([x, y, depth], axis=-1)  # (H, W, 3)


def main(
    aria_path: str,
    output: Optional[str] = None,
    da3_output: Optional[str] = None,
    n_views: int = 8,
    object_name: Optional[str] = None,
    frames: Optional[str] = None,
    start_frame: int = 30,
    max_depth: float = 65534,
    depth_scale: float = 1000.0,
) -> None:
    """Prepare images + DA3 data for MV-SAM3D from GT cameras/depth."""
    input_dir = Path(aria_path) / "processed"
    aria2mesh_dir = Path(aria_path) / "aria2mesh"
    if object_name is None:
        object_name = Path(aria_path).name
    inputs_dir = aria2mesh_dir / "inputs"
    if output is None:
        output = str(inputs_dir)
    if da3_output is None:
        da3_output = str(inputs_dir / "da3_output.npz")
    output_dir = Path(output)

    # Frame selection
    rgb_files = sorted((input_dir / "rgb_small").glob("*.jpg"))
    mask_files = sorted((input_dir / "object_mask_small").glob("*.png"))
    if not rgb_files:
        rgb_files = sorted((input_dir / "rgb_small").glob("*.png"))

    n_total = len(rgb_files)
    assert len(mask_files) == n_total, (
        f"RGB ({n_total}) and mask ({len(mask_files)}) count mismatch"
    )
    print(f"Found {n_total} frames in {input_dir / 'rgb_small'}")

    if frames is not None:
        frame_indices = [int(x.strip()) for x in frames.split(",")]
        stem_to_idx = {int(f.stem): i for i, f in enumerate(rgb_files)}
        indices = [stem_to_idx[fi] for fi in frame_indices]
    else:
        indices = np.linspace(start_frame, n_total - 1, n_views, dtype=int)
        frame_indices = [int(rgb_files[idx].stem) for idx in indices]

    N = len(indices)
    print(f"Selected {N} frames: {frame_indices}")

    # Images + masks
    images_dir = output_dir / "images"
    masks_dir = output_dir / object_name
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    for i, idx in enumerate(indices):
        out_idx = i + 1
        img = Image.open(rgb_files[idx])
        img.save(images_dir / f"{out_idx}.png")
        rgba = img.convert("RGBA")
        rgba.putalpha(Image.open(mask_files[idx]))
        rgba.save(masks_dir / f"{out_idx}_mask.png")
        print(f"  {out_idx}: {rgb_files[idx].name} ({img.size[0]}x{img.size[1]})")

    print(f"\nImages: {images_dir}")
    print(f"Masks:  {masks_dir}")

    # DA3 output (cameras + depth + pointmaps)
    print("\nBuilding DA3 output...")
    c2w_list, intrinsics_list, depths, pointmaps = [], [], [], []
    image_files = []

    for i, fi in enumerate(frame_indices):
        stem = f"{fi:08d}"
        intrinsics_list.append(np.load(str(input_dir / "intrinsics" / f"{stem}.npy")))
        c2w_list.append(np.load(str(input_dir / "extrinsics" / f"{stem}.npy")))

        depth_raw = cv2.imread(
            str(input_dir / "depth_rgb" / f"{stem}.png"), cv2.IMREAD_UNCHANGED
        ).astype(np.float32)
        invalid = (depth_raw == 0) | (depth_raw >= max_depth)
        depth = depth_raw / depth_scale
        depth[invalid] = np.nan
        depths.append(depth)
        pointmaps.append(depth_to_pointmap(depth, intrinsics_list[-1]))
        image_files.append(str(images_dir / f"{i + 1}.png"))
        valid_pct = (~np.isnan(depth)).sum() / depth.size * 100
        print(
            f"  [{i}] frame {fi}: depth [{np.nanmin(depth):.3f}, {np.nanmax(depth):.3f}]m, valid={valid_pct:.0f}%"
        )

    # Rebase: camera 0 becomes the world frame
    c2w_all = np.stack(c2w_list, axis=0)
    c2w_0 = c2w_all[0].copy()
    w2c_rebased = np.linalg.inv(c2w_all) @ c2w_0

    depths = np.stack(depths, axis=0).astype(np.float32)
    pointmaps = np.stack(pointmaps, axis=0).astype(np.float32)
    intrinsics_arr = np.stack(intrinsics_list, axis=0).astype(np.float32)
    extrinsics = w2c_rebased[:, :3, :4].astype(np.float32)

    da3_path = Path(da3_output)
    da3_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        da3_path,
        depth=depths,
        pointmaps=pointmaps,
        pointmaps_sam3d=pointmaps.transpose(0, 3, 1, 2),
        extrinsics=extrinsics,
        intrinsics=intrinsics_arr,
        image_files=np.array(image_files),
    )
    np.save(da3_path.parent / "c2w_cam0.npy", c2w_0.astype(np.float32))

    gt_pc_path = input_dir / "point_cloud.npz"
    if gt_pc_path.exists():
        shutil.copy2(gt_pc_path, da3_path.parent / "point_cloud.npz")
        print(f"  Copied {gt_pc_path} -> {da3_path.parent / 'point_cloud.npz'}")

    print(f"\nSaved {da3_path}")
    print(f"  depth: {depths.shape}")
    print(f"  pointmaps: {pointmaps.shape}")
    print(f"  extrinsics (w2c 3x4): {extrinsics.shape}")
    print(f"  intrinsics: {intrinsics_arr.shape}")

    # scene.glb (aligned point cloud)
    M_cv_to_gltf = np.eye(4, dtype=np.float64)
    M_cv_to_gltf[1, 1] = -1.0
    M_cv_to_gltf[2, 2] = -1.0

    all_pts_world, all_colors = [], []
    stride = 4
    for i in range(N):
        pm = pointmaps[i]
        H, W = pm.shape[:2]
        valid = ~np.isnan(pm[:, :, 2])
        pts_cam = pm[valid]

        w2c_44 = np.eye(4, dtype=np.float64)
        w2c_44[:3, :4] = extrinsics[i].astype(np.float64)
        c2w = np.linalg.inv(w2c_44)
        pts_world = (c2w[:3, :3] @ pts_cam.T).T + c2w[:3, 3]

        img_path = Path(image_files[i])
        if img_path.exists():
            img = np.array(Image.open(img_path).resize((W, H)))[:, :, :3]
            colors = img[valid]
        else:
            colors = np.full((len(pts_world), 3), 180, dtype=np.uint8)

        sub_idx = np.arange(0, len(pts_world), stride)
        all_pts_world.append(pts_world[sub_idx])
        all_colors.append(colors[sub_idx])

    all_pts_world = np.concatenate(all_pts_world, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    pts_gltf = trimesh.transformations.transform_points(all_pts_world, M_cv_to_gltf)
    center = pts_gltf.mean(axis=0)
    T_center = np.eye(4, dtype=np.float64)
    T_center[:3, 3] = -center

    alignment_matrix = T_center @ M_cv_to_gltf
    pts_aligned = pts_gltf - center

    pc = trimesh.PointCloud(
        vertices=pts_aligned.astype(np.float32),
        colors=np.hstack(
            [all_colors, np.full((len(all_colors), 1), 255, dtype=np.uint8)]
        ),
    )
    scene = trimesh.Scene()
    scene.add_geometry(pc, node_name="pointcloud")
    scene.metadata["hf_alignment"] = alignment_matrix.tolist()

    scene_glb_path = da3_path.parent / "scene.glb"
    scene.export(str(scene_glb_path))
    print(f"\nSaved {scene_glb_path} ({len(pts_aligned)} points, stride={stride})")


if __name__ == "__main__":
    tyro.cli(main)
