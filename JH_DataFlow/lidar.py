import numpy as np
from typing import Tuple, Union
import copy
import matplotlib.pyplot as plt
# from torch import Tensor

def rotation_3d_in_axis(
    points: Union[np.ndarray],
    angles: Union[np.ndarray, float],
    axis: int = 0,
    return_mat: bool = False,
    clockwise: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Rotate points by angles according to axis.

    Args:
        points (np.ndarray): Points with shape (N, M, 3).
        angles (np.ndarray or float): Vector of angles with shape (N, ).
        axis (int): The axis to be rotated. Defaults to 0.
        return_mat (bool): Whether or not to return the rotation matrix
            (transposed). Defaults to False.
        clockwise (bool): Whether the rotation is clockwise. Defaults to False.

    Raises:
        ValueError: When the axis is not in range [-3, -2, -1, 0, 1, 2], it
            will raise ValueError.

    Returns:
        Tuple[np.ndarray, np.ndarray] or np.ndarray: Rotated points with shape (N, M, 3) and rotation matrix with
        shape (N, 3, 3).
    """
    batch_free = len(points.shape) == 2
    if batch_free:
        points = points[None]

    if isinstance(angles, float) or len(angles.shape) == 0:
        angles = np.full(points.shape[:1], angles)

    assert len(points.shape) == 3 and len(angles.shape) == 1 and \
        points.shape[0] == angles.shape[0], 'Incorrect shape of points ' \
        f'angles: {points.shape}, {angles.shape}'

    assert points.shape[-1] in [2, 3], \
        f'Points size should be 2 or 3 instead of {points.shape[-1]}'

    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)

    if points.shape[-1] == 3:
        if axis == 1 or axis == -2:
            rot_mat_T = np.array([
                [rot_cos, zeros, -rot_sin],
                [zeros, ones, zeros],
                [rot_sin, zeros, rot_cos]
            ]).transpose(2, 0, 1)
        elif axis == 2 or axis == -1:
            rot_mat_T = np.array([
                [rot_cos, rot_sin, zeros],
                [-rot_sin, rot_cos, zeros],
                [zeros, zeros, ones]
            ]).transpose(2, 0, 1)
        elif axis == 0 or axis == -3:
            rot_mat_T = np.array([
                [ones, zeros, zeros],
                [zeros, rot_cos, rot_sin],
                [zeros, -rot_sin, rot_cos]
            ]).transpose(2, 0, 1)
        else:
            raise ValueError(
                f'axis should be in range [-3, -2, -1, 0, 1, 2], got {axis}')
    else:
        rot_mat_T = np.array([
            [rot_cos, rot_sin],
            [-rot_sin, rot_cos]
        ]).transpose(2, 0, 1)

    if clockwise:
        rot_mat_T = rot_mat_T.transpose(0, 2, 1)

    if points.shape[0] == 0:
        points_new = points
    else:
        points_new = np.einsum('aij,ajk->aik', points, rot_mat_T)

    if batch_free:
        points_new = points_new.squeeze(0)

    if return_mat:
        rot_mat_T = rot_mat_T.squeeze(0) if batch_free else rot_mat_T
        return points_new, rot_mat_T
    else:
        return points_new


def get_box_corners(boxes_3d):
    
    """Convert boxes to corners in clockwise order, in the form of (x0y0z0,
    x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0).

    .. code-block:: none

                                         up z
                          front x           ^
                               /            |
                              /             |
                (x1, y0, z1) + -----------  + (x1, y0, z1)
                            /|            / |
                           / |           /  |
            (x0, y1, z1) + ----------- +   + (x1, y0, z0)
                         |  /      .   |  /
                         | / origin    | /
         left y <------- + ----------- + (x0, y0, z0)
            (x0, y1, z0)

    Returns:
        Tensor: A tensor with 8 corners of each box in shape (N, 8, 3).
    """
    poses = boxes_3d[:, :3]
    dims = boxes_3d[:, 3:6]
    yaws = boxes_3d[:, 6]
    yaw_axis = 2
    
    corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array([0.5, 0.5, 0])
    corners = dims.reshape(-1, 1, 3) * corners_norm.reshape([1, 8, 3])
    
    # rotate around z axis
    corners = rotation_3d_in_axis(corners, yaws, axis=yaw_axis)
    corners += poses.reshape(-1, 1, 3)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(corners_norm[:,0], corners_norm[:,1], corners_norm[:,2], c='r', marker='o', s=20)
    # for i in range(len(corners_norm)):
    #     ax.text(corners_norm[i,0], corners_norm[i,1], corners_norm[i,2], f"{i}", fontsize=50, ha='center')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.show()
    
    return corners

def conf_to_K(conf):
    K = np.eye(3)
    K[[0, 1, 0, 1], [0, 1, 2, 2]] = conf
    return K

def proj_lidar_bbox3d_to_img(corners_3d: np.ndarray,
                             calib_meta: dict) -> np.ndarray:
    """Project the 3D bbox on 2D plane.

    Args:
        bboxes_3d (:obj:`LiDARInstance3DBoxes`): 3D bbox in lidar coordinate
            system to visualize.
        input_meta (dict): Meta information.
    """
    T_lidar_r1 = np.array(copy.deepcopy(calib_meta['cam2lidar']['T_lidar_camRect1'])).reshape(4, 4)
    
    # borrow from dsec_det remapping.py
    K_r0 = conf_to_K(copy.deepcopy(calib_meta['cam2cam']['intrinsics']['camRect0']['camera_matrix']))
    K_r1 = conf_to_K(copy.deepcopy(calib_meta['cam2cam']['intrinsics']['camRect1']['camera_matrix']))

    R_r0_0 = np.array(copy.deepcopy(calib_meta['cam2cam']['extrinsics']['R_rect0']))
    R_r1_1 = np.array(copy.deepcopy(calib_meta['cam2cam']['extrinsics']['R_rect1']))
    R_1_0 = np.array(copy.deepcopy(calib_meta['cam2cam']['extrinsics']['T_10']))[:3, :3]

    # read from right to left:
    # rect. cam. 1 -> norm. rect. cam. 1 -> norm. cam. 1 -> norm. cam. 0 -> norm. rect. cam. 0 -> rect. cam. 0
    P_r0_r1 = K_r0 @ R_r0_0 @ R_1_0.T @ R_r1_1.T @ np.linalg.inv(K_r1)
    
    num_bbox = corners_3d.shape[0]
    
    corners_lidar_hom = np.concatenate(
        [corners_3d.reshape(-1, 3), 
         np.ones((num_bbox * 8, 1))], 
        axis=-1)
    corners_r1_hom = (np.linalg.inv(T_lidar_r1) @ corners_lidar_hom.T).T
    
    # project to r1 image plane
    corners_r1_hom[:, 2] = np.clip(corners_r1_hom[:, 2], a_min=1e-5, a_max=1e5)
    corners_r1_2d = (K_r1 @ corners_r1_hom[:, :3].T).T
    
    corners_r1_2d = corners_r1_2d[...,:3] / corners_r1_2d[..., -1:]
    
    corners_r0_2d = (P_r0_r1 @ corners_r1_2d.T).T
    corners_r0_2d = corners_r0_2d[...,:2] / corners_r0_2d[..., -1:]
    corners_r0_2d = corners_r0_2d.astype('float32')

    return corners_r0_2d.reshape(num_bbox, 8, 2)

    # # pts_2d = pts_2d[mask_pos_z]
    
    # if pts_2d.shape[0] == 0:
    #     return np.zeros((0, 8, 2))
    # else:
    #     return pts_2d[..., :2].reshape(-1, 8, 2)