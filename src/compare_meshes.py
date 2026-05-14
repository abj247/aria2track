"""Load and display multiple GLB/OBJ meshes side-by-side in Viser."""

import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import trimesh
import tyro
import viser


MESH_COLORS = [
    (100, 149, 237),  # cornflower blue
    (50, 205, 50),    # lime green
    (255, 165, 0),    # orange
    (220, 20, 60),    # crimson
    (148, 103, 189),  # purple
    (0, 206, 209),    # turquoise
    (255, 105, 180),  # pink
    (127, 127, 127),  # gray
]


def classify_mesh(path: Path) -> str:
    """Infer a human-readable type from the file path/name."""
    name = path.name.lower()
    parent = path.parent.name.lower()
    if "autoscaled" in name:
        return "FoundationPose autoscaled"
    if "merged_scene_optimized" in name:
        return "aria2mesh scene-optimized (scaled)"
    if "merged_scene" in name:
        return "aria2mesh scene-merged (unscaled)"
    if "pose_optimized" in name:
        return "aria2mesh pose-optimized"
    if name == "result.glb":
        return "aria2mesh canonical (unscaled, ~1m)"
    if name == "result.ply":
        return "Gaussian splat"
    return path.stem


def main(
    meshes: List[str],
    labels: Optional[List[str]] = None,
    port: int = 8080,
    spacing: float = 0.0,
) -> None:
    """Visualize multiple meshes for comparison.

    Args:
        meshes: paths to GLB/OBJ/PLY mesh files
        labels: optional custom labels (one per mesh); auto-detected if omitted
        port: viser server port
        spacing: horizontal offset between meshes (0 = overlap)
    """
    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(dark_mode=True)

    server.gui.add_markdown("## Mesh Comparison")

    toggles = []
    loaded = 0
    for i, mesh_path_str in enumerate(meshes):
        mesh_path = Path(mesh_path_str)
        if not mesh_path.exists():
            print(f"SKIP: {mesh_path} does not exist")
            continue

        scene_or_mesh = trimesh.load(str(mesh_path))
        geoms = []
        if isinstance(scene_or_mesh, trimesh.Scene):
            for name, geom in scene_or_mesh.geometry.items():
                if isinstance(geom, trimesh.Trimesh):
                    geoms.append((name, geom))
        elif isinstance(scene_or_mesh, trimesh.Trimesh):
            geoms.append((mesh_path.stem, scene_or_mesh))

        if labels and i < len(labels):
            mesh_type = labels[i]
        else:
            mesh_type = classify_mesh(mesh_path)

        color = MESH_COLORS[loaded % len(MESH_COLORS)]
        offset = np.array([loaded * spacing, 0, 0], dtype=np.float32)

        all_verts = []
        for gname, geom in geoms:
            verts = np.array(geom.vertices, dtype=np.float32)
            all_verts.append(verts)

            if spacing > 0:
                shifted = geom.copy()
                shifted.vertices = verts + offset
                server.scene.add_mesh_trimesh(f"/mesh_{loaded}/{gname}", shifted)
            else:
                server.scene.add_mesh_trimesh(f"/mesh_{loaded}/{gname}", geom)

        if all_verts:
            combined = np.concatenate(all_verts)
            bbox_min = combined.min(axis=0)
            bbox_max = combined.max(axis=0)
            extents = bbox_max - bbox_min
            max_ext = extents.max()
        else:
            max_ext = 0.0
            extents = np.zeros(3)

        short_path = f"{mesh_path.parent.name}/{mesh_path.name}"
        print(f"[{loaded}] {mesh_type}")
        print(f"    file: {short_path}")
        print(f"    extents: {extents[0]:.4f} x {extents[1]:.4f} x {extents[2]:.4f}"
              f"  (max={max_ext:.4f}m)")

        r, g, b = color
        color_hex = f"#{r:02x}{g:02x}{b:02x}"
        gui_label = f"**{mesh_type}** | max={max_ext:.3f}m"
        server.gui.add_markdown(
            f'<span style="color:{color_hex}">●</span> {gui_label}'
        )
        toggle = server.gui.add_checkbox(
            f"Show [{loaded}] {mesh_type}", True
        )

        idx_capture = loaded

        def make_cb(idx):
            def cb(_):
                server.scene.add_frame(
                    f"/mesh_{idx}",
                    visible=toggles[idx].value,
                    show_axes=False,
                )
            return cb

        toggle.on_update(make_cb(idx_capture))
        toggles.append(toggle)
        loaded += 1

    print(f"\nLoaded {loaded} mesh(es)")
    print(f"Viewer: http://localhost:{port}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    tyro.cli(main)
