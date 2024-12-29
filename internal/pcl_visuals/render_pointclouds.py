import pycolmap
import torch
import numpy as np
import os
import json
import random
import shutil
from pytorch3d.structures import Pointclouds
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection
from blender_render_cameras import render_cameras
from vis_utils import (
    transform_pcl_and_cameras,
    compute_from_vec_pca,
    METHOD_COLORS,
    knn_laplace_fiter,
)


def visualize_reconstruction_dir(
    reconstruction_dir: str,
    out_dir: str,
    scene_name: str,
    semidense_pcls: bool = True,
    debug: bool = False,
    animation: bool = False,
    n_animation_frames: int = 80,
):
    print(reconstruction_dir)

    os.makedirs(out_dir, exist_ok=True)

    try:
        reconstruction = pycolmap.Reconstruction(reconstruction_dir)
    except ValueError:
        print(f"CANNOT LOAD {reconstruction_dir}")
        return

    try:
        alignment_transform_path = os.path.join(
            reconstruction_dir,
            "normalization.json",
        )
        with open(alignment_transform_path, "r") as f:
            normalisation = json.load(f)
    except FileNotFoundError:
        print(f"NO NORMALIZATION FILE {alignment_transform_path}")
        return

    # store example images
    # idx = np.linspace(0, len(reconstruction.images)-1, 10).astype(int).tolist()
    idx = np.linspace(0, len(reconstruction.images) - 1, 5).astype(int).tolist()
    imidx = -1
    for imi, (image_id, image) in enumerate(list(reconstruction.images.items())):
        if imi not in idx:
            continue
        imidx += 1
        image_file = os.path.join(
            reconstruct_dir, "..", "..", "video_frames", image.name
        )
        if os.path.exists(image_file):
            im_file_out = os.path.join(out_dir, f"{scene_name}_im{imidx:02d}.jpg")
            shutil.copy(image_file, im_file_out)

    # copy original video
    video_file = os.path.join(reconstruct_dir, "..", "..", "video_default_default0.mp4")
    video_file_out = os.path.join(out_dir, f"{scene_name}.mp4")
    shutil.copy(video_file, video_file_out)

    # copy the segmentation video as well
    # video_file = os.path.join(reconstruct_dir, "..", "..", "masks", "segmentation_video_refined_masks_xmem.mp4")
    # video_file = os.path.join(reconstruct_dir, "..", "..", "masks", "segmentation_video.mp4")
    video_file = os.path.join(
        reconstruct_dir,
        "..",
        "..",
        "final_export",
        "segmentation_video_refined_masks_xmem.mkv",
    )
    video_file_out = os.path.join(out_dir, f"{scene_name}_segmentation.mkv")
    shutil.copy(video_file, video_file_out)

    cam_sel = list(list(range(len(reconstruction.images)))[::7])
    # cam_sel = list(range(len(reconstruction.images)))
    # cam_sel = list(range(len(reconstruction.images)))

    Rs, Ts, Ks, szs = [], [], [], []
    for cami, (image_id, image) in enumerate(reconstruction.images.items()):
        if cami not in cam_sel:
            continue
        R_ = torch.tensor(image.cam_from_world.rotation.matrix().real).float()
        tvec = torch.tensor(image.cam_from_world.translation).float()
        camera = reconstruction.cameras[image.camera_id]
        camera_matrix = torch.tensor(camera.calibration_matrix()).float()
        image_size = torch.tensor([camera.height, camera.width]).float()
        Rs.append(R_)
        Ts.append(tvec)
        Ks.append(camera_matrix)
        szs.append(image_size)

    sequence_cams = cameras_from_opencv_projection(
        R=torch.stack(Rs),
        tvec=torch.stack(Ts),
        camera_matrix=torch.stack(Ks),
        image_size=torch.stack(szs),
    )

    points = [point3D.xyz.tolist() for point3D in reconstruction.points3D.values()]
    rgb = [
        (point3D.color / 255).tolist() for point3D in reconstruction.points3D.values()
    ]

    point_cloud = Pointclouds(
        points=torch.tensor(points).float()[None],
        features=torch.tensor(rgb).float()[None],
    )

    if semidense_pcls:
        # add dense points
        additional_pts_path = os.path.join(
            reconstruction_dir,
            "additional",
            "additional_points_dict.pt",
        )
        add_pts = torch.load(additional_pts_path, map_location="cpu")
        for k, v in add_pts.items():
            if k in ["sfm_points_num", "additional_points_num"]:
                continue
            points.extend(v["points3D"].tolist())
            rgb.extend(v["points3D_rgb"].tolist())

        point_cloud = Pointclouds(
            points=torch.tensor(points).float()[None],
            features=torch.tensor(rgb).float()[None],
        )

    print(point_cloud.features_padded()[0].mean(dim=0))
    print(point_cloud.features_padded()[0].std(dim=0))

    if True:  # use the dataset normalisation transform
        # s * (X @ R + T)
        R = torch.tensor(normalisation["rotation"])
        T = torch.tensor(normalisation["translation"])
        s = torch.tensor(normalisation["scale"])

        Ra = R
        Ta = T @ R.t()
        sa = s * 0.5

        # s * (Xa + Ta) @ Ra = s (Xa @ Ra + Ta @ Ra); T = Ta @ Ra; Ta = T @ R.T
        # sequence_cams, point_cloud = transform_pcl_and_cameras(
        #     sequence_cams,
        #     point_cloud,
        #     rescale_factor=sa,
        #     T_adjust=Ta,
        #     R_adjust=Ra,
        # )  # theres a bug here! gotta decompose to translat+scale and rotate

        sequence_cams, point_cloud = transform_pcl_and_cameras(
            sequence_cams,
            point_cloud,
            rescale_factor=sa,
            T_adjust=Ta,
            R_adjust=None,
        )
        sequence_cams, point_cloud = transform_pcl_and_cameras(
            sequence_cams,
            point_cloud,
            rescale_factor=1.0,
            T_adjust=None,
            R_adjust=Ra,
        )

        flip_y_rot = torch.tensor(
            [
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1],
            ]
        ).float()
        sequence_cams, point_cloud = transform_pcl_and_cameras(
            sequence_cams,
            point_cloud,
            rescale_factor=1.0,
            T_adjust=None,
            R_adjust=flip_y_rot,
        )

        # cut off the points too far away from the center
        ok_pts = (point_cloud.points_packed().abs() <= 4.0).all(dim=-1)

        point_cloud = Pointclouds(
            points=[point_cloud.points_packed()[ok_pts]],
            features=[point_cloud.features_packed()[ok_pts]],
        )

    else:

        offset = None
        rescale_factor = 1
        rescale = float(
            rescale_factor / point_cloud.points_padded()[0].std(dim=0).mean()
        )
        offset = -point_cloud.points_padded()[0].mean(dim=0)
        sequence_cams_rescaled, point_cloud_rescaled = transform_pcl_and_cameras(
            sequence_cams,
            point_cloud,
            rescale_factor=rescale,
            T_adjust=offset,
        )

        if False:
            # point_cloud = point_cloud_rescaled
            point_cloud = point_cloud_rescaled
            pass

        else:
            # get the components
            # _, eigs = compute_from_vec_pca(point_cloud_rescaled.points_padded()[0])
            _, eigs = compute_from_vec_pca(sequence_cams_rescaled.get_camera_center())
            assert eigs.det() > 0.0

            # rotate so that the principal component is aligned with the z axis
            # and the least important component (normal) is aligned with the x axis
            sequence_cams, point_cloud = transform_pcl_and_cameras(
                sequence_cams_rescaled,
                point_cloud_rescaled,
                rescale_factor=1.0,
                T_adjust=None,
                R_adjust=eigs,
            )

            # now move the smallest PCA axis from the x axis to the y axis (blendrender wants this)
            # x -> y
            final_rot = torch.tensor(
                [
                    [0, 1, 0],
                    [1, 0, 0],
                    [0, 0, -1],
                ]
            ).float()
            assert final_rot.det() > 0.0
            sequence_cams, point_cloud = transform_pcl_and_cameras(
                sequence_cams,
                point_cloud,
                rescale_factor=1.0,
                T_adjust=None,
                R_adjust=final_rot,
            )

            # also need to check whether the up-vector of the cameras is positive, if not
            # we need to flip
            camera_up = sequence_cams.R[:, 1]
            if camera_up[:, 1].mean() > 0:
                print("flipping camera up")
                # the camera's up vectors are opposite they should be -> flip the y axis
                flip_y_rot = torch.tensor(
                    [
                        [1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1],
                    ]
                ).float()
                assert final_rot.det() > 0.0
                sequence_cams, point_cloud = transform_pcl_and_cameras(
                    sequence_cams,
                    point_cloud,
                    rescale_factor=1.0,
                    T_adjust=None,
                    R_adjust=flip_y_rot,
                )
                assert sequence_cams.R[:, 1, 1].mean() <= 0

    # cam_sel_render = list(list(range(len(sequence_cams.R)))[::5])
    # sequence_cams_render = sequence_cams[cam_sel_render]
    # render_cams = {"ours": sequence_cams_render}

    render_cams = {"ours": sequence_cams}
    camera_colors = torch.tensor([METHOD_COLORS["ours"] for k in render_cams]).float()

    render_cams_list = list(render_cams.values())

    # determine the camera scale as a portion of the distance between
    # the camera centers and the point cloud center
    # camera_scale = float((
    #     point_cloud.points_packed().median(dim=0, keepdim=True).values
    #     - sequence_cams.get_camera_center()
    # ).norm(dim=1).median()) * 0.1
    camera_scale = 0.3
    # if semidense:
    ys = point_cloud.points_packed()[:, 1]
    ground_plane_y = float(torch.quantile(ys, 0.95))
    # ground_plane_y = float(torch.quantile(ys, 0.05))
    # else:
    #     ground_plane_y = float(point_cloud.points_packed().max(dim=0).values[1])
    # ground_plane_y = float(point_cloud.points_packed().min(dim=0).values[1])

    if semidense:
        # knn filter point cloud colors
        point_cloud = knn_laplace_fiter(
            point_cloud,
            device=(
                torch.device("cuda:0")
                if torch.cuda.is_available()
                else torch.device("cpu")
            ),
        )

    for cam_height in [5, 20]:
        for cam_dist in [10 if semidense else 15, 20]:
            # for cam_height in [20]:
            #     for cam_dist in [30]:

            outpath_now = f"{out_dir}/{scene_name}_d{cam_dist}_h{cam_height}"
            print(outpath_now)

            cam_rot_y = (
                np.linspace(0, 2 * np.pi, 4)[:3].tolist()
                if not animation
                else np.linspace(0, 2 * np.pi, n_animation_frames)[
                    : (n_animation_frames - 1)
                ].tolist()
            )

            show_cameras = (
                not (cam_dist <= 10 and cam_height <= 10) or not semidense_pcls
            )
            render_cameras(
                outpath_now,
                camera_list=render_cams_list if show_cameras else [],
                point_cloud=point_cloud,
                # point_cloud=None,
                camera_colors=camera_colors,
                # camera_colors=None,
                # ---
                num_samples=8 if debug else 64,
                palette_size=256,
                resol_perc=25 if debug else (50 if animation else 100),
                cam_rot_y=cam_rot_y,
                # ---
                # num_samples=8,
                # palette_size=4,
                # resol_perc=25,
                # cam_rot_y=(np.linspace(0, np.pi, 2).tolist()),
                # ---
                resolution_height=1200,
                resolution_width=1200,
                animation=animation,
                render_shape_kwargs={
                    # "camera_scale": 0.35, #1.0
                    "camera_scale": camera_scale,  # 1.0
                    # "camera_scale": 1.0, #1.0
                    "ground_plane_y": ground_plane_y,
                    "camera_stick_radius": 0.08 * camera_scale,  # 0.05
                    "camera_pole_radius": 0.01,
                    "pcl_color": [0, 0, 255, 1.0],
                    # "radius_pcl": 0.04, # 0.05
                    # "radius_pcl": 0.02, # 0.05
                    # "radius_pcl": 0.05,
                    # "radius_pcl": 0.035,
                    # "radius_pcl": 0.02,  # best for uniform color
                    # "radius_pcl": 0.019,
                    # "radius_pcl": 0.021,
                    "radius_pcl": 0.007 if semidense_pcls else 0.025,
                },
                # cam_height
                # cam_dist
                # cam_height=20.0,
                # cam_dist=20.0,
                cam_height=cam_height,
                cam_dist=cam_dist,
                render_pcl_colors=True,
                blender_opts_kwargs={
                    # "ground_plane_y": 3.0,  # clean pcl
                    # "ground_plane_y": 2.0,  # init pcl
                    "ground_plane_y": ground_plane_y,
                    # "material_brightness": 0.2,
                    # "env_exposure": 2.0,
                    "env_exposure": 2.5,
                    # "ground_type": "no_ground",
                    # "render_pcl_colors": True,
                    "ground_type": "circle",
                    # 'cam_height': 20.0,
                    # "ground_size": 0.6,
                    # "material_brightness": 2.5,
                    # "ground_color_intensity": 3.0,
                    "ground_color_intensity": 3.0,
                    # "env_exposure": 2.0,
                    # "compose_distort": 0.0,
                    # "compose_disperse": 0.03,
                    "compose_disperse": 0.0,
                    # "compose_vignette": 0.0,
                    "dof_use": True,
                    "dof_aperture_blades": 11,
                    "dof_aperture_fstop": 0.4,  # 0.2,
                    "dof_aperture_ratio": 50.0,
                },
                visdom_env=None,
                blendrender_debug=False,
            )


if __name__ == "__main__":

    debug = False
    animation = True

    if False:
        gauss_fit_stats = "/fsx-repligen/dnovotny/datasets/uCO3D/scene_to_gaussian_fit_stats_241024.json"
        with open(gauss_fit_stats, "r") as f:
            gauss_fit_stats = json.load(f)

        ok_seqs = [
            s for s, v in gauss_fit_stats.items() if v["psnr"] > 36 and v["psnr"] < 40
        ]

        random.seed(0)
        random.shuffle(ok_seqs)

        seq_names = ok_seqs[:100]

        # # sort by psnr
        # gauss_fit_stats = sorted(gauss_fit_stats.items(), key=lambda kv: kv[1]["psnr"], reverse=True)

        # seq_names = [g[0] for g in gauss_fit_stats[10:50]]

        # for sequence_name in seq_names:
        #     print(f"'{sequence_name}'")

    elif False:
        import pandas as pd
        import sqlite3

        db_file = "/fsx-repligen/shared/datasets/uCO3D/batch_reconstruction/dataset_export/metadata_vgg_1102_170k.sqlite"
        with sqlite3.connect(db_file) as conn:
            df = pd.read_sql_query("SELECT * from sequence_annots", conn)
            # df_frame = pd.read_sql_query('SELECT * from frame_annots', conn)

        seq_to_resol_file = "/fsx-repligen/shared/datasets/uCO3D/batch_reconstruction/dataset_export/metadata_vgg_1102_170k_seq_to_resol.json"
        with open(seq_to_resol_file, "r") as f:
            seq_to_resol = json.load(f)

        df["video_size_h"] = [seq_to_resol[seq][0] for seq in df["sequence_name"]]
        df["video_size_w"] = [seq_to_resol[seq][1] for seq in df["sequence_name"]]
        df["fine_category"] = [cat.split("/")[1] for cat in df["category"]]
        df["super_category"] = [cat.split("/")[0] for cat in df["category"]]

        df = df[
            (df["gaussian_splats_psnr"] > 34)
            & (df["gaussian_splats_psnr"] < 38)
            # (df["gaussian_splats_psnr"] > 30)
            # & (df["gaussian_splats_psnr"] < 35)
            & (df["video_size_h"] * df["video_size_w"] >= (1920 * 1080))
        ]

        # df["gaussian_splats_psnr"] > 36
        # cats =
        # supercats = [cat.split("/")[0] for cat in df["category"]]

        supercats = df["super_category"].unique().tolist()
        supercat_to_seq = {
            supercat: list(df[df["super_category"] == supercat]["sequence_name"])
            for supercat in supercats
        }

        # sample 5 from each supercat
        seq_names = []
        random.seed(0)
        for supercat, seqs in supercat_to_seq.items():
            if len(seqs) < 2:
                print(f"{supercat} of len {len(seqs)}")
                seq_names.extend(seqs)
            else:
                seq_names.extend(random.sample(seqs, 2))

        random.seed(0)
        random.shuffle(seq_names)

        print(len(seq_names))

        with open("./visualised_seq_names.json", "w") as f:
            json.dump(seq_names, f)

        # seq_names = [
        #     "31987-21959-32687",
        #     # '2865-25474-48904',
        #     # '14839-60125-59077',
        #     # '35375-58976-1432',
        #     # '36801-15139-20349',
        #     # '48929-42322-29033',
        #     # '58139-43186-55353',
        #     # '28704-47031-42798',
        #     # '36775-36193-27973',
        #     # '55955-9092-24183',
        #     # '15097-25524-58010',
        # ]

    elif True:
        # video assets
        seq_names = [
            "1-3401-2762",
            "1-34403-89911",
            "1-41035-27122",
            "1-86215-98872",
            "10242-32989-14201",
            "1-3985-77543",
        ]

    else:

        # splash assets
        seq_names = [
            # "1-7279-77098",
            # "1-34403-89911",
            # "1-3401-2762",
            "1-716-75198",
            "1-75941-66475",
            # "1-86215-98872",
            # "1-34403-89911",
            # "1-34859-78807",
            # "1-36441-55031",
            # "1-41035-27122",
            # "1-75941-66475",
            # "1-86215-98872",
            # "64858-49245-13597",
            # "12118-44049-61793",
            # "12109-23499-43178",
            # "11163-41345-60075",
            # "11082-15724-59788",
            # "10242-32989-14201",
            # "1-716-75198",
            # "1-1186-97735",
            # "1-3401-2762",
            # "1-3985-77543",
            # "1-33870-62900",
            # "1-5200-29126",
            # "1-7279-77098",
            # "1-8629-19938",
            # "1-11067-50296",
            # "1-12965-21069",
            # "1-17321-91757",
            # "1-25364-30653",
            # "1-30758-25124",
        ]

        # seq_names = [
        #     "454-24865-38129",
        #     "137-32190-19333",
        #     "411-2308-97612",
        #     "135-23505-84779",
        #     "3-47848-36563",
        #     "35-45435-5727",
        #     "91-86156-9789",
        #     "3-65132-35722",
        #     "35-45435-5727",
        #     "459-13897-69332",
        #     "230-21515-47018",
        # ]

    # seq_names = [
    #     "42-43758-8340",
    # ]

    for sequence_name in seq_names:
        reconstruct_dir = os.path.join(
            "/fsx-repligen/shared/datasets/uCO3D/batch_reconstruction",
            sequence_name,
            "mapper_output/0",
        )
        for semidense in [False, True]:
            visualize_reconstruction_dir(
                reconstruction_dir=reconstruct_dir,
                out_dir=f"/fsx-repligen/dnovotny/visuals/uco3d_pcl/renders_semidense={semidense}_co3dalign_v5",
                scene_name=sequence_name,
                semidense_pcls=semidense,
                debug=debug,
                n_animation_frames=80,
                animation=animation,
            )
