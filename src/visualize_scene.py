"""Visualize DA3 scene + SAM3D meshes (unoptimized & optimized) with GUI controls.

Everything is transformed to the original gravity-aligned world frame using
c2w_cam0. DA3 data and SAM3D meshes live in cam0 frame and get transformed out;
the global point cloud is already in world frame.
"""

import time
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
import tyro
import viser
import viser.transforms as tf
from PIL import Image
from scipy.spatial.transform import Rotation

Z_UP_TO_Y_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
PYTORCH3D_TO_OPENCV = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)

VIEW_COLORS = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
]


def w2c_to_c2w(w2c):
    if w2c.shape == (3, 4):
        w2c_44 = np.eye(4, dtype=np.float64)
        w2c_44[:3, :4] = w2c
        w2c = w2c_44
    return np.linalg.inv(w2c)


def apply_4x4(T, pts):
    """Apply 4x4 transform to (N,3) points."""
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    return (T @ pts_h.T).T[:, :3]


def transform_mesh_to_world(mesh, params, c2w0=None):
    """Apply SAM3D pose (and optionally c2w0) to place mesh in cam0 frame.

    Uses PyTorch3D row-vector convention (v @ R) to match the transform chain
    used by SAM3D inference and the pose optimizer.
    """
    verts = np.array(mesh.vertices, dtype=np.float64)
    scale = params["scale"].flatten()
    rot_wxyz = params["rotation"].flatten()
    translation = params["translation"].flatten()

    verts = verts @ Z_UP_TO_Y_UP.T
    verts = verts * scale[0]
    r = Rotation.from_quat([rot_wxyz[1], rot_wxyz[2], rot_wxyz[3], rot_wxyz[0]])
    verts = verts @ r.as_matrix()  # v @ R (PyTorch3D convention)
    verts = verts + translation
    verts = verts @ PYTORCH3D_TO_OPENCV.T
    if c2w0 is not None:
        R_c2w, t_c2w = c2w0[:3, :3], c2w0[:3, 3]
        verts = (R_c2w @ verts.T).T + t_c2w

    transformed = mesh.copy()
    transformed.vertices = verts.astype(np.float32)
    return transformed


def transform_trimesh(mesh, T):
    """Apply 4x4 transform to a trimesh in-place and return it."""
    mesh.vertices = apply_4x4(T, np.array(mesh.vertices, dtype=np.float64)).astype(
        np.float32
    )
    return mesh


def visualize_glb(glb_path: Path, port: int = 8080) -> None:
    """Simple GLB/scene viewer."""
    scene = trimesh.load(str(glb_path))
    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(dark_mode=True)
    server.scene.set_up_direction("+z")

    if isinstance(scene, trimesh.Scene):
        for name, geom in scene.geometry.items():
            if isinstance(geom, trimesh.Trimesh):
                server.scene.add_mesh_trimesh(f"mesh/{name}", geom)
                print(
                    f"  Mesh '{name}': {geom.vertices.shape[0]} verts, {geom.faces.shape[0]} faces"
                )
            elif isinstance(geom, trimesh.PointCloud):
                pts = np.array(geom.vertices, dtype=np.float32)
                cols = (
                    np.array(geom.colors[:, :3], dtype=np.uint8)
                    if geom.colors is not None
                    else np.full((len(pts), 3), 180, dtype=np.uint8)
                )
                server.scene.add_point_cloud(
                    f"pc/{name}",
                    points=pts,
                    colors=cols,
                    point_size=0.003,
                    point_shape="rounded",
                )
                print(f"  PointCloud '{name}': {len(pts)} points")
            else:
                print(f"  Skipped '{name}': {type(geom).__name__}")
    else:
        server.scene.add_mesh_trimesh("mesh", scene)
        print(
            f"Loaded: {scene.vertices.shape[0]} vertices, {scene.faces.shape[0]} faces"
        )

    print(f"\nOpen in browser: http://localhost:{port}")
    while True:
        time.sleep(1.0)


GLB_PREFERENCE = [
    "result_merged_scene_optimized.glb",
    "result_merged_scene.glb",
    "result.glb",
]


def find_best_glb(outputs_dir: Path) -> Optional[Path]:
    """Find the best GLB file from the outputs directory."""
    if not outputs_dir.exists():
        return None
    for name in GLB_PREFERENCE:
        glb = outputs_dir / name
        if glb.exists():
            return glb
    return None


def main(
    aria_path: Path,
    simple: bool = False,
    no_pose: bool = False,
    port: int = 8080,
    point_size: float = 0.001,
    share: bool = False,
) -> None:
    """Visualize DA3 scene + SAM3D meshes (unoptimized & optimized).

    Args:
        aria_path: path to the Aria recording (e.g. data/drill)
        simple: lightweight GLB viewer using the best output from aria2mesh/outputs/
    """
    if simple:
        outputs_dir = aria_path / "aria2mesh" / "outputs"
        glb_path = find_best_glb(outputs_dir)
        if glb_path is None:
            raise FileNotFoundError(f"No GLB files found in {outputs_dir}")
        print(f"Viewing {glb_path}")
        visualize_glb(glb_path, port)
        return

    aria2mesh_dir = aria_path / "aria2mesh"
    inputs_dir = aria2mesh_dir / "inputs"
    outputs_dir = aria2mesh_dir / "outputs"

    da3 = np.load(inputs_dir / "da3_output.npz")
    if "pointmaps" in da3:
        pointmaps = da3["pointmaps"]
    else:
        pointmaps = da3["pointmaps_sam3d"].transpose(0, 2, 3, 1)
    extrinsics = da3["extrinsics"]
    intrinsics = da3["intrinsics"]
    image_files = da3["image_files"]
    N, H, W, _ = pointmaps.shape
    print(f"DA3: {N} views, {H}x{W}")

    c2w_cam0_path = inputs_dir / "c2w_cam0.npy"
    if c2w_cam0_path.exists():
        c2w_cam0 = np.load(c2w_cam0_path).astype(np.float64)
        if c2w_cam0.shape == (3, 4):
            tmp = np.eye(4, dtype=np.float64)
            tmp[:3, :4] = c2w_cam0
            c2w_cam0 = tmp
        print(f"Loaded c2w_cam0 from {c2w_cam0_path.name}")
    else:
        c2w_cam0 = np.eye(4, dtype=np.float64)
        print("WARNING: c2w_cam0.npy not found, assuming cam0 = world")

    # SAM3D output is directly in outputs_dir (single run, overwritten each time)
    has_sam3d = outputs_dir.exists() and (outputs_dir / "result.glb").exists()

    server = viser.ViserServer(host="0.0.0.0", port=port)
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
        gui_show_depth_pc = server.gui.add_checkbox("Depth Point Clouds", True)
        gui_show_unopt = server.gui.add_checkbox("Unoptimized Mesh", False)
        gui_show_opt = server.gui.add_checkbox("Optimized Mesh", True)
        gui_show_global_pc = server.gui.add_checkbox("Global Point Cloud", True)

    frame_folder = server.gui.add_folder("Views")
    frame_toggles = {}
    frustum_handles = []
    depth_pc_handles = []

    global_pc_handle = None
    global_pc_path = inputs_dir / "point_cloud.npz"
    if global_pc_path.exists():
        global_pts = np.load(global_pc_path)["points"]
        colors = np.cos(global_pts + np.arange(3)) / 4.0 + 0.5
        global_pc_handle = server.scene.add_point_cloud(
            "/global_point_cloud",
            points=global_pts.astype(np.float32),
            colors=colors.astype(np.float32),
            point_size=point_size,
            point_shape="rounded",
        )
        print(f"Global PC: {len(global_pts)} points")

    for i in range(N):
        c2w_cam0_i = w2c_to_c2w(extrinsics[i])
        c2w_world = c2w_cam0 @ c2w_cam0_i

        img_path = Path(str(image_files[i]))
        if not img_path.is_absolute():
            img_path = Path.cwd() / img_path
        if img_path.exists():
            img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        else:
            img = np.full((H, W, 3), VIEW_COLORS[i % len(VIEW_COLORS)], dtype=np.uint8)

        pts_cam = pointmaps[i].reshape(-1, 3)
        if img.shape[:2] != (H, W):
            img_flat = np.array(Image.open(img_path).resize((W, H)))[:, :, :3].reshape(
                -1, 3
            )
        else:
            img_flat = img.reshape(-1, 3)

        valid = ~np.isnan(pts_cam).any(axis=1)
        pts_world = (c2w_world[:3, :3] @ pts_cam[valid].T).T + c2w_world[:3, 3]

        depth_pc_handles.append(
            server.scene.add_point_cloud(
                f"/views/v{i}/depth_pc",
                points=pts_world.astype(np.float32),
                colors=img_flat[valid].astype(np.uint8),
                point_size=point_size,
                point_shape="rounded",
            )
        )

        fy = intrinsics[i][1, 1]
        fov_y = float(2 * np.arctan(H / (2 * fy)))
        aspect = W / H

        frustum_handles.append(
            server.scene.add_camera_frustum(
                f"/views/v{i}/frustum",
                fov=fov_y,
                aspect=float(aspect),
                scale=0.05,
                color=VIEW_COLORS[i % len(VIEW_COLORS)],
                image=img if img_path.exists() else None,
                format="jpeg",
                wxyz=tf.SO3.from_matrix(c2w_world[:3, :3]).wxyz,
                position=c2w_world[:3, 3],
            )
        )

        with frame_folder:
            toggle = server.gui.add_checkbox(f"View {i}", True)
        frame_toggles[i] = toggle

        def make_toggle_cb(vidx):
            def cb(_):
                vis = frame_toggles[vidx].value
                server.scene.add_frame(f"/views/v{vidx}", visible=vis, show_axes=False)

            return cb

        toggle.on_update(make_toggle_cb(i))

        pos = c2w_world[:3, 3]
        print(f"  View {i}: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    # SAM3D meshes (cam0 frame -> world frame via c2w_cam0)
    unopt_handles = []
    opt_handles = []
    c2w0 = w2c_to_c2w(extrinsics[0])

    if has_sam3d:
        glb_path = outputs_dir / "result.glb"
        params_path = outputs_dir / "params.npz"
        opt_params_path = outputs_dir / "pose_optimization" / "optimized_params.npz"

        mesh = trimesh.load(str(glb_path), force="mesh")
        print(
            f"SAM3D mesh: {mesh.vertices.shape[0]} verts, {mesh.faces.shape[0]} faces"
        )

        if params_path.exists() and not no_pose:
            params = dict(np.load(params_path))
            unopt_mesh = transform_mesh_to_world(mesh, params, c2w0)
            transform_trimesh(unopt_mesh, c2w_cam0)
            print("  Applied original pose -> world frame")
        else:
            unopt_mesh = mesh
        h = server.scene.add_mesh_trimesh("/objects/mesh", unopt_mesh)
        h.visible = gui_show_unopt.value
        unopt_handles.append(h)

        if opt_params_path.exists() and not no_pose:
            opt_params = dict(np.load(opt_params_path, allow_pickle=True))
            opt_mesh = transform_mesh_to_world(mesh, opt_params)
            transform_trimesh(opt_mesh, c2w_cam0)
            h = server.scene.add_mesh_trimesh("/objects/mesh_optimized", opt_mesh)
            opt_handles.append(h)
            print("  Added optimized mesh -> world frame")

    @gui_point_size.on_update
    def _(_):
        for h in depth_pc_handles:
            h.point_size = gui_point_size.value
        if global_pc_handle is not None:
            global_pc_handle.point_size = gui_point_size.value

    @gui_show_frustums.on_update
    def _(_):
        for h in frustum_handles:
            h.visible = gui_show_frustums.value

    @gui_show_depth_pc.on_update
    def _(_):
        for h in depth_pc_handles:
            h.visible = gui_show_depth_pc.value

    @gui_show_unopt.on_update
    def _(_):
        for h in unopt_handles:
            h.visible = gui_show_unopt.value

    @gui_show_opt.on_update
    def _(_):
        for h in opt_handles:
            h.visible = gui_show_opt.value

    @gui_show_global_pc.on_update
    def _(_):
        if global_pc_handle is not None:
            global_pc_handle.visible = gui_show_global_pc.value

    print(f"\nViewer running at http://localhost:{port}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    tyro.cli(main)
