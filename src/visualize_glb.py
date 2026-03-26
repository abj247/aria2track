"""Simple GLB/scene viewer using viser."""

import numpy as np
import trimesh
import tyro
import viser


def main(glb_path: str, port: int = 8080) -> None:
    """Visualize a GLB mesh or scene."""
    scene = trimesh.load(glb_path)
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
    try:
        while True:
            pass
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    tyro.cli(main)
