import torch
import copy
from typing import Tuple, Optional
from pytorch3d.transforms import so3_exp_map
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.ops.knn import knn_points, knn_gather

TEDDY_UP_AXIS: Tuple[float, float, float] = (-0.0396, -0.8306, -0.5554)


METHOD_COLORS = {
    # "ours": (83/255,123/255,156/255),
    # "ours": (43/255,83/255,116/255),
    "ours": (13 / 255, 43 / 255, (116 - 30) / 255),
}


@torch.no_grad()
def knn_laplace_fiter(
    pointclouds,
    K: int = 30,
    alpha: float = 0.5,
    device=torch.device("cuda:0"),
):
    knn = knn_points(
        p1=pointclouds.points_packed()[None].to(device),
        p2=pointclouds.points_packed()[None].to(device),
        K=K,
        return_nn=False,
        return_sorted=False,
    )

    knn_colors = knn_gather(
        pointclouds.features_packed()[None].to(device),
        knn.idx,
    )[
        0
    ].mean(dim=1)

    pcl_filtered = alpha * knn_colors + (1 - alpha) * pointclouds.features_packed().to(
        knn_colors
    )

    return pointclouds.update_padded(
        new_points_padded=pointclouds.points_padded(),
        new_features_padded=pcl_filtered[None].cpu(),
    )


def adjust_scene_scale_up_vector(
    cameras: CamerasBase,
    pcl: Optional[Pointclouds] = None,
    rescale_factor: float = 1.0,
    to_vec=(0.0, -1.0, 0.0),
    from_vec=TEDDY_UP_AXIS,
    skip_rotation: bool = False,
    offset=None,
    R_adjust=None,
):
    # rotates the up vector of the look_at cameras to the desired up direction
    device = cameras.R.device
    if offset is not None:
        T_adjust = offset
    else:
        T_adjust = torch.zeros(3).to(device)

    # to_vec
    # and from_vec
    # should be the major components of PCA

    if not skip_rotation:
        if R_adjust is not None:
            R_adjust = R_adjust.to(device)
        else:
            rot_axis_angle = torch.cross(
                torch.FloatTensor(to_vec),
                torch.FloatTensor(from_vec),
                # torch.FloatTensor((0.0, -1.0, 0.0)),
                # torch.FloatTensor(TEDDY_UP_AXIS),
            ).to(cameras.device)
            R_adjust = so3_exp_map(rot_axis_angle[None])[0].to(device)
    else:
        R_adjust = torch.eye(3).to(device)

    if pcl is not None:
        pcl = pcl.update_padded(pcl.points_padded() + T_adjust)
        pcl = pcl.update_padded(rescale_factor * pcl.points_padded())
        pcl = pcl.update_padded(pcl.points_padded() @ R_adjust[None])
    else:
        pcl = None
    cameras_a = copy.deepcopy(cameras)

    if True:
        align_t_R = R_adjust.t()
        align_t_T = -rescale_factor * T_adjust[None] @ align_t_R
        align_t_s = rescale_factor
        cameras_a.T = (
            torch.bmm(
                align_t_T[:, None].repeat(cameras_a.R.shape[0], 1, 1),
                cameras_a.R,
            )[:, 0]
            + cameras_a.T * align_t_s
        )
        cameras_a.R = torch.bmm(align_t_R[None].expand_as(cameras_a.R), cameras_a.R)
    else:
        cameras_a.T *= rescale_factor
        cameras_a.R = torch.bmm(
            R_adjust[None].expand_as(cameras_a.R).permute(0, 2, 1), cameras_a.R
        )
    return cameras_a, pcl


def transform_pcl_and_cameras(
    cameras: CamerasBase,
    pcl: Optional[Pointclouds] = None,
    rescale_factor: float = 1.0,
    T_adjust=None,
    R_adjust=None,
):
    # rotates the up vector of the look_at cameras to the desired up direction
    device = cameras.R.device
    if T_adjust is None:
        T_adjust = torch.zeros(3).to(device)

    if R_adjust is None:
        R_adjust = torch.eye(3).to(device)

    R_adjust = R_adjust.to(device)

    if pcl is not None:
        # s * (X + T) @ R
        pcl = pcl.update_padded(pcl.points_padded() + T_adjust)
        pcl = pcl.update_padded(rescale_factor * pcl.points_padded())
        pcl = pcl.update_padded(pcl.points_padded() @ R_adjust[None])
    else:
        pcl = None
    cameras_a = copy.deepcopy(cameras)

    align_t_R = R_adjust.t()
    align_t_T = -rescale_factor * T_adjust[None] @ align_t_R
    align_t_s = rescale_factor
    cameras_a.T = (
        torch.bmm(
            align_t_T[:, None].repeat(cameras_a.R.shape[0], 1, 1),
            cameras_a.R,
        )[:, 0]
        + cameras_a.T * align_t_s
    )
    cameras_a.R = torch.bmm(align_t_R[None].expand_as(cameras_a.R), cameras_a.R)

    return cameras_a, pcl


def compute_from_vec_pca(point_cloud):
    """
    Compute from_vec for a 3D point cloud using PCA in PyTorch.

    Args:
    - point_cloud (torch.Tensor): Nx3 tensor representing the point cloud.
    - distance (float): Distance from the origin to place the camera.

    Returns:
    - torch.Tensor: The computed from_vec.
    """
    # Center the data (subtract the mean)
    mean = torch.mean(point_cloud, dim=0)
    centered_data = point_cloud - mean

    # Compute the covariance matrix
    cov_matrix = torch.matmul(centered_data.T, centered_data) / (
        centered_data.size(0) - 1
    )

    # Perform eigen decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

    # Choose the eigenvector associated with the largest eigenvalue
    principal_component = eigenvectors[:, torch.argmax(eigenvalues)]

    # Set the camera position along this principal component at the given distance
    # from_vec = principal_component * distance

    # ensure the determinant is positive
    if eigenvectors.det() < 0.0:
        eigenvectors[:, 0] *= -1.0

    return principal_component, eigenvectors
