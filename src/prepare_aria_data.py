"""Prepare aria2mano processed recordings for training.

Creates from a processed recording:
  - depth_rgb/: depth reprojected from stereo_left to RGB camera frame (uint16 mm, 512x512)
  - rgb_small/: center-cropped and resized RGB images (512x512)
  - intrinsics/: 3x3 camera intrinsics for rgb_small (.npy)
  - extrinsics/: 4x4 T_world_camera for RGB camera (.npy)
"""

import pickle
from pathlib import Path

import cv2
import numpy as np
import tyro
from tqdm import tqdm


def load_aria_data(aria_path: Path) -> dict:
    with open(aria_path / "processed" / "aria_data.pkl", "rb") as f:
        return pickle.load(f)


def compute_crop_resize_intrinsics(K, orig_w, orig_h, target_size=512):
    """Adjust intrinsics for center square crop then resize to target_size."""
    if orig_w > orig_h:
        crop_offset_x = (orig_w - orig_h) / 2
        crop_offset_y = 0
        crop_size = orig_h
    else:
        crop_offset_x = 0
        crop_offset_y = (orig_h - orig_w) / 2
        crop_size = orig_w

    K_new = K.copy()
    K_new[0, 2] -= crop_offset_x
    K_new[1, 2] -= crop_offset_y
    scale = target_size / crop_size
    K_new[0, :] *= scale
    K_new[1, :] *= scale
    return K_new


def reproject_depth(depth_stereo, K_stereo, K_rgb_small, T_rgb_stereo, target_size=512):
    """Reproject depth from stereo_left camera to RGB camera frame (cropped+resized)."""
    mask = depth_stereo > 0
    vs, us = np.where(mask)
    zs = depth_stereo[vs, us].astype(np.float64)

    fx_s, fy_s = K_stereo[0, 0], K_stereo[1, 1]
    cx_s, cy_s = K_stereo[0, 2], K_stereo[1, 2]
    xs = (us - cx_s) * zs / fx_s
    ys = (vs - cy_s) * zs / fy_s
    pts_stereo = np.stack([xs, ys, zs, np.ones_like(zs)], axis=-1)

    pts_rgb = (T_rgb_stereo @ pts_stereo.T).T
    z_rgb = pts_rgb[:, 2]
    valid = z_rgb > 0
    pts_rgb, z_rgb = pts_rgb[valid], z_rgb[valid]

    fx_r, fy_r = K_rgb_small[0, 0], K_rgb_small[1, 1]
    cx_r, cy_r = K_rgb_small[0, 2], K_rgb_small[1, 2]
    u_rgb = (fx_r * pts_rgb[:, 0] / z_rgb + cx_r).astype(np.int32)
    v_rgb = (fy_r * pts_rgb[:, 1] / z_rgb + cy_r).astype(np.int32)

    in_bounds = (
        (u_rgb >= 0) & (u_rgb < target_size) & (v_rgb >= 0) & (v_rgb < target_size)
    )
    u_rgb, v_rgb, z_rgb = u_rgb[in_bounds], v_rgb[in_bounds], z_rgb[in_bounds]

    depth_out = np.full(
        (target_size, target_size), np.iinfo(np.uint16).max, dtype=np.float64
    )
    for i in range(len(u_rgb)):
        u, v, z = u_rgb[i], v_rgb[i], z_rgb[i]
        if z < depth_out[v, u]:
            depth_out[v, u] = z
    return depth_out.astype(np.uint16)


def main(aria_path: Path, target_size: int = 512) -> None:
    """Prepare aria2mano processed recordings for training."""
    data = load_aria_data(aria_path)
    frame_data = data["frame_data"]
    num_frames = len(frame_data)

    f0 = frame_data[0]
    K_rgb = f0["rgb_camera"]["K"]
    rgb_w = int(f0["rgb_camera"]["width"])
    rgb_h = int(f0["rgb_camera"]["height"])
    K_stereo = f0["stereo_left_camera"]["K"]
    T_device_rgb = f0["rgb_camera"]["T_device_camera"]
    T_device_stereo = f0["stereo_left_camera"]["T_device_camera"]
    T_rgb_stereo = np.linalg.inv(T_device_rgb) @ T_device_stereo
    K_rgb_small = compute_crop_resize_intrinsics(K_rgb, rgb_w, rgb_h, target_size)

    processed = aria_path / "processed"
    out_depth = processed / "depth_rgb"
    out_rgb = processed / "rgb_small"
    out_intrinsics = processed / "intrinsics"
    out_extrinsics = processed / "extrinsics"
    out_mask = processed / "object_mask_small"
    for d in [out_depth, out_rgb, out_intrinsics, out_extrinsics, out_mask]:
        d.mkdir(exist_ok=True)

    if rgb_w > rgb_h:
        crop_x, crop_y, crop_size = (rgb_w - rgb_h) // 2, 0, rgb_h
    else:
        crop_x, crop_y, crop_size = 0, (rgb_h - rgb_w) // 2, rgb_w

    for i in tqdm(range(num_frames), desc="Processing frames"):
        fname = f"{i:08d}"
        fd = frame_data[i]

        rgb_img = cv2.imread(str(processed / "rgb" / f"{fname}.jpg"))
        rgb_crop = rgb_img[crop_y : crop_y + crop_size, crop_x : crop_x + crop_size]
        rgb_small = cv2.resize(
            rgb_crop, (target_size, target_size), interpolation=cv2.INTER_AREA
        )
        cv2.imwrite(str(out_rgb / f"{fname}.jpg"), rgb_small)

        mask_path = processed / "object_mask" / f"{fname}.png"
        if mask_path.exists():
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            mask_crop = mask_img[
                crop_y : crop_y + crop_size, crop_x : crop_x + crop_size
            ]
            mask_small = cv2.resize(
                mask_crop, (target_size, target_size), interpolation=cv2.INTER_NEAREST
            )
            cv2.imwrite(str(out_mask / f"{fname}.png"), mask_small)

        depth_path = processed / "depth" / f"{fname}.png"
        if depth_path.exists():
            depth_stereo = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            depth_rgb = reproject_depth(
                depth_stereo, K_stereo, K_rgb_small, T_rgb_stereo, target_size
            )
            cv2.imwrite(str(out_depth / f"{fname}.png"), depth_rgb)

        np.save(str(out_intrinsics / f"{fname}.npy"), K_rgb_small)

        T_world_device = fd["T_world_device"]
        T_world_rgb = T_world_device @ T_device_rgb
        np.save(str(out_extrinsics / f"{fname}.npy"), T_world_rgb)


if __name__ == "__main__":
    tyro.cli(main)
