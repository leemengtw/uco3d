from typing import Optional, List, Union
import os
import torch
import numpy as np
import blendrender as brdr

# ^ git clone git@github.com:fairinternal/BlendRender.git; cd BlendRender; pip install -e .
import matplotlib.pyplot as plt
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.structures import Pointclouds


def render_cameras(
    render_path: str,
    camera_list: List[PerspectiveCameras],
    point_cloud: Optional[Pointclouds] = None,
    camera_colors: Optional[torch.Tensor] = None,
    camera_colormap_name: str = "rainbow",
    # ---
    num_samples: int = 32,
    resol_perc: float = 100.0,
    resolution_height: int = 1024,
    resolution_width: int = 1024,
    animation: bool = False,
    cam_rot_y: Union[float, List[float]] = 0.0,
    cam_height: float = 20.0,
    cam_dist: float = 17.0,
    visdom_env: str = "render_cameras",
    visdom_win: str = "render",
    render_shape_kwargs: dict = {},
    blender_opts_kwargs: dict = {},
    palette_size: int = 256,
    render_pcl_colors: bool = False,
    visdom_conn=None,
    blendrender_debug: bool = False,
):
    """
    Args:
        render_path: Output render path
        camera_list: List of camera traces, each trace has to have the same number of cameras.
        point_cloud: Point cloud to display in the middle of the scene
        camera_colors: n_traces x 3 tensor of colors (rgb, range [0-1]) for each trace.
        camera_colormap_name: In case camera_colors is None, sets the colors according
            to the `camera_colormap_name` colormap from matplotlib.
        num_samples: controls the quality of the rendering
        resol_perc: percentage of original resolution to render (allows fast debug renders)
        resolution_height: height of the final render
        resolution_width: width of the final render
        animation: whether to render an animation as opposed to still images
        cam_rot_y: defines the rotation of the rendering camera around the
            scene's y axis.
        visdom_env: output visdom env
        visdom_win: output visdom window
        render_shape_kwargs: additional kwargs specifying the rendered shape
        blender_opts_kwargs: additional kwargs for the blender environment
    """

    if camera_colors is None:
        # get the matplotlib colors if the input colors were not specified
        n_traces = len(camera_list)
        cmap_ = plt.get_cmap(camera_colormap_name)
        camera_colors = np.array(
            [
                (np.array(cmap_(pi))).astype(float).tolist()
                for pi in np.linspace(0.0, 1.0, n_traces)
            ]
        )[:, :3].astype(float)
    else:
        camera_colors = camera_colors.cpu().numpy()

    # convert the cameras to float lists as required by blendrender
    if len(camera_list) == 0:
        camera_param_dict = {"NO_CAMERA": True}
    else:

        def _to_np(k):
            return (
                torch.stack([getattr(c, k).cpu().detach() for c in camera_list])
                .numpy()
                .astype(float)
                .tolist()
            )

        camera_param_dict = {
            k: _to_np(k) for k in ["R", "T", "principal_point", "focal_length"]
        }

    # shape settings
    render_shape = {
        "type": "cameras",
        **camera_param_dict,
        "camera_colors": camera_colors,
        "camera_scale": 0.25,
        "camera_stick_radius": 0.02,
        "camera_pole_radius": 0.005,
        # --- note that "pcl" is not required ---
        "pcl_color": [*([0.4] * 3), 1.0],
        "radius_pcl": 0.02,
    }
    render_shape.update(render_shape_kwargs)

    # add the point cloud if input
    if point_cloud is not None:
        pcl_data = point_cloud.points_padded().detach().cpu().numpy()
        assert pcl_data.shape[0] == 1, "only one point cloud allowed!"
        pcl_data = pcl_data[0]
        render_shape["pcl"] = pcl_data

        if render_pcl_colors:
            rgb = point_cloud.features_packed().detach().cpu().numpy()
            rgb_palette, rgb_palette_idx = brdr.quantize_colors(
                rgb, palette_size=palette_size
            )
            render_shape["rgb_palette_idx"] = rgb_palette_idx.astype(int)[None]
            render_shape["rgb_palette"] = rgb_palette.astype(float)
            render_shape["pcl"] = render_shape["pcl"][None]
    else:
        # need to specify the ground plane in case no point cloud is specified
        render_shape["ground_plane_y"] = 0.0

    # blender options
    blender_opts = {
        "resolution_percentage": resol_perc,
        "resolution_height": resolution_height,
        "resolution_width": resolution_width,
        "num_samples": num_samples,
        "material_style": "colored_suface",
        "obj_rot_y": 0.0,
        "cam_rot_y": cam_rot_y,
        "animation": animation,
        "fps": 20,
        "cam_height": cam_height,
        "cam_dist": cam_dist,
        "ground_size": 0.6,
        "material_brightness": 2.5,
        "ground_color_intensity": 2.0,
        "env_exposure": 1.6,
        "env_hdri": "pizzo_pernice_puresky_4k",
        "compose_distort": 0.0,
        "compose_disperse": 0.0,
        "compose_vignette": 0.0,
    }
    blender_opts.update(blender_opts_kwargs)

    # run the rendering
    render_paths, render_cameras = brdr.blender_render(
        render_shape=render_shape,
        blender_opts=blender_opts,
        render_path=render_path,
        visdom_conn=visdom_conn,
        visdom_env=visdom_env,
        visdom_win=visdom_win,
        verbose=True,
        debug=blendrender_debug,
        animation_format="mp4",
        get_cameras=True,
    )

    return render_paths


if __name__ == "__main__":
    # set output path
    outpath = "./camera_render_output"

    # load example data from blendrender
    data_path = os.path.join(
        os.path.dirname(brdr.__file__), "..", "examples", "camera_data.npz"
    )
    render_data = np.load(data_path)
    n_traces = render_data["R"].shape[0]
    camera_list = []
    for trace in range(n_traces):
        cam_args = {}
        for k in [
            "R",
            "T",
            "principal_point",
            "focal_length",
        ]:
            cam_args[k] = render_data[k][trace]
        camera_list.append(PerspectiveCameras(**cam_args))

    # pointclouds
    point_cloud = Pointclouds(torch.from_numpy(render_data["pcl"][None]))

    # set the colors of the two input traces
    camera_colors = torch.tensor(
        [
            [1.0, 0.0, 0.0],  # first trace red
            [0.0, 1.0, 0.0],  # second trace blue
        ]
    ).float()

    # run the rendering
    render_paths = render_cameras(
        outpath,
        camera_list,
        point_cloud=point_cloud,
        camera_colors=camera_colors,
        # show from 4 different angles:
        cam_rot_y=np.linspace(0.0, 2 * np.pi, 5)[:4].astype(float).tolist(),
    )

    print(render_paths)


# good
# /data/home/dnovotny/uCO3D/visuals/renders_semidense=True_gplanefix_v2/180-23975-23929_d30_h40_119_000.png
# visuals/renders_semidense=True_gplanefix_v2/30304-11096-4589_d10_h5_119_000.png
