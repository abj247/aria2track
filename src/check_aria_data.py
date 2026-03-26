"""Self-contained viser visualizer for prepared training data.

Reads only: depth_rgb, intrinsics, extrinsics, rgb_small, object_mask_small, point_cloud.npz
"""

import time
from pathlib import Path

import cv2
import numpy as np
import tyro
import viser
import viser.transforms as tf


def unproject_depth(depth, K, T_world_camera, max_depth_m=5.0, mask=None):
    """Unproject depth map to world-space points. Returns (points, vs, us)."""
    valid = (depth < np.iinfo(np.uint16).max) & (depth <= max_depth_m * 1000)
    if mask is not None:
        valid = valid & (mask > 0)
    vs, us = np.where(valid)
    zs = depth[vs, us].astype(np.float64)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    # Convert to meters before applying transform (translation is in meters)
    zs_m = zs / 1000.0
    xs = (us - cx) * zs_m / fx
    ys = (vs - cy) * zs_m / fy
    pts_cam = np.stack([xs, ys, zs_m, np.ones_like(zs)], axis=-1)
    pts_world = (T_world_camera @ pts_cam.T).T[:, :3]
    return pts_world, vs, us


def main(
    aria_path: Path,
    start_frame: int = 20,
    frame_skip: int = 100,
    port: int = 8080,
    point_size: float = 0.002,
    max_depth: float = 5.0,
    share: bool = False,
) -> None:
    """Visualize prepared Aria training data."""
    processed = aria_path / "processed"

    rgb_files = sorted((processed / "rgb_small").glob("*.jpg"))
    all_indices = [int(f.stem) for f in rgb_files]
    start = next((i for i, v in enumerate(all_indices) if v >= start_frame), 0)
    frame_indices = all_indices[start :: frame_skip + 1]
    print(
        f"Loading {len(frame_indices)} frames (every {frame_skip + 1}th of {len(all_indices)})"
    )

    pc_path = processed / "point_cloud.npz"
    global_pc = np.load(str(pc_path))["points"] if pc_path.exists() else None

    server = viser.ViserServer(port=port)
    server.gui.configure_theme(dark_mode=True)
    if share:
        server.request_share_url()

    with server.gui.add_folder("Display"):
        gui_point_size = server.gui.add_slider(
            "Point Size",
            min=0.0,
            max=0.01,
            step=0.0001,
            initial_value=point_size,
        )
        gui_show_frustums = server.gui.add_checkbox("Camera Frustums", True)
        gui_show_scene_pc = server.gui.add_checkbox("Scene Point Cloud", True)
        gui_show_depth_pc = server.gui.add_checkbox("Depth Point Clouds", True)
        gui_show_object_pc = server.gui.add_checkbox("Object Points (magenta)", True)

    frame_folder = server.gui.add_folder("Frames")
    frame_toggles = {}
    frustum_handles = []
    depth_pc_handles = []
    object_pc_handles = []

    for idx in frame_indices:
        fname = f"{idx:08d}"
        K = np.load(str(processed / "intrinsics" / f"{fname}.npy"))
        T_world_cam = np.load(str(processed / "extrinsics" / f"{fname}.npy"))
        rgb = cv2.cvtColor(
            cv2.imread(str(processed / "rgb_small" / f"{fname}.jpg")), cv2.COLOR_BGR2RGB
        )
        depth = cv2.imread(
            str(processed / "depth_rgb" / f"{fname}.png"), cv2.IMREAD_UNCHANGED
        )
        mask_path = processed / "object_mask_small" / f"{fname}.png"
        obj_mask = (
            cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask_path.exists()
            else None
        )

        h, w = rgb.shape[:2]
        fov = float(2 * np.arctan2(w / 2, K[0, 0]))
        aspect = w / h

        frustum = server.scene.add_camera_frustum(
            f"/frames/f{idx}/frustum",
            fov=fov,
            aspect=aspect,
            scale=0.05,
            image=rgb,
            format="jpeg",
            wxyz=tf.SO3.from_matrix(T_world_cam[:3, :3]).wxyz,
            position=T_world_cam[:3, 3],
        )
        frustum_handles.append(frustum)

        pts_all, vs_all, us_all = unproject_depth(
            depth, K, T_world_cam, max_depth_m=max_depth
        )
        is_object = np.zeros(len(vs_all), dtype=bool)
        if obj_mask is not None:
            is_object = obj_mask[vs_all, us_all] > 0

        bg_mask = ~is_object
        if np.any(bg_mask):
            bg_pts = pts_all[bg_mask]
            bg_colors = rgb[vs_all[bg_mask], us_all[bg_mask]].astype(np.float32) / 255.0
            depth_pc_handles.append(
                server.scene.add_point_cloud(
                    f"/frames/f{idx}/depth_pc",
                    points=bg_pts.astype(np.float32),
                    colors=bg_colors,
                    point_size=point_size,
                    point_shape="rounded",
                )
            )

        if np.any(is_object):
            obj_pts = pts_all[is_object]
            obj_colors = (
                rgb[vs_all[is_object], us_all[is_object]].astype(np.float32) / 255.0
            )
            object_pc_handles.append(
                server.scene.add_point_cloud(
                    f"/frames/f{idx}/object_pc",
                    points=obj_pts.astype(np.float32),
                    colors=obj_colors,
                    point_size=point_size * 1.5,
                    point_shape="rounded",
                )
            )

        with frame_folder:
            toggle = server.gui.add_checkbox(f"Frame {idx}", True)
        frame_toggles[idx] = toggle

        def make_toggle_cb(fidx):
            def cb(_):
                vis = frame_toggles[fidx].value
                server.scene.add_frame(f"/frames/f{fidx}", visible=vis, show_axes=False)

            return cb

        toggle.on_update(make_toggle_cb(idx))

    scene_pc_handle = None
    if global_pc is not None:
        colors = np.cos(global_pc + np.arange(3)) / 4.0 + 0.5
        scene_pc_handle = server.scene.add_point_cloud(
            "/scene_point_cloud",
            points=global_pc.astype(np.float32),
            colors=colors.astype(np.float32),
            point_size=point_size,
            point_shape="rounded",
        )

    @gui_point_size.on_update
    def _(_):
        for h in depth_pc_handles:
            h.point_size = gui_point_size.value
        for h in object_pc_handles:
            h.point_size = gui_point_size.value * 1.5
        if scene_pc_handle is not None:
            scene_pc_handle.point_size = gui_point_size.value

    @gui_show_frustums.on_update
    def _(_):
        for h in frustum_handles:
            h.visible = gui_show_frustums.value

    @gui_show_scene_pc.on_update
    def _(_):
        if scene_pc_handle is not None:
            scene_pc_handle.visible = gui_show_scene_pc.value

    @gui_show_depth_pc.on_update
    def _(_):
        for h in depth_pc_handles:
            h.visible = gui_show_depth_pc.value

    @gui_show_object_pc.on_update
    def _(_):
        for h in object_pc_handles:
            h.visible = gui_show_object_pc.value

    print(f"Viewer running at http://localhost:{port}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    tyro.cli(main)
