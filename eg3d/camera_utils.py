# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Helper functions for constructing camera parameter matrices. Primarily used in visualization and inference scripts.
"""

import math

import torch
import torch.nn as nn

from training.volumetric_rendering import math_utils

class GaussianCameraPoseSampler:
    """
    Samples pitch and yaw from a Gaussian distribution and returns a camera pose.
    Camera is specified as looking at the origin.
    If horizontal and vertical stddev (specified in radians) are zero, gives a
    deterministic camera pose with yaw=horizontal_mean, pitch=vertical_mean.
    The coordinate system is specified with y-up, z-forward, x-left.
    Horizontal mean is the azimuthal angle (rotation around y axis) in radians,
    vertical mean is the polar angle (angle from the y axis) in radians.
    A point along the z-axis has azimuthal_angle=0, polar_angle=pi/2.

    Example:
    For a camera pose looking at the origin with the camera at position [0, 0, 1]:
    cam2world = GaussianCameraPoseSampler.sample(math.pi/2, math.pi/2, radius=1)
    """

    @staticmethod
    def sample(horizontal_mean, vertical_mean, horizontal_stddev=0, vertical_stddev=0, radius=1, batch_size=1, device='cpu'):
        h = torch.randn((batch_size, 1), device=device) * horizontal_stddev + horizontal_mean
        v = torch.randn((batch_size, 1), device=device) * vertical_stddev + vertical_mean
        v = torch.clamp(v, 1e-5, math.pi - 1e-5)

        theta = h
        v = v / math.pi
        phi = torch.arccos(1 - 2*v)

        camera_origins = torch.zeros((batch_size, 3), device=device)

        camera_origins[:, 0:1] = radius*torch.sin(phi) * torch.cos(math.pi-theta)
        camera_origins[:, 2:3] = radius*torch.sin(phi) * torch.sin(math.pi-theta)
        camera_origins[:, 1:2] = radius*torch.cos(phi)

        forward_vectors = math_utils.normalize_vecs(-camera_origins)
        return create_cam2world_matrix(forward_vectors, camera_origins)


class LookAtPoseSampler:
    """
    Same as GaussianCameraPoseSampler, except the
    camera is specified as looking at 'lookat_position', a 3-vector.
    
    If horizontal and vertical stddev (specified in radians) are zero, gives a
    deterministic camera pose with yaw=horizontal_mean, pitch=vertical_mean.
    The coordinate system is specified with y-up, z-forward, x-left.
    Horizontal mean is the azimuthal angle (rotation around y axis) in radians,
    vertical mean is the polar angle (angle from the y axis) in radians.
    A point along the z-axis has azimuthal_angle=0, polar_angle=pi/2.
    

    Example:
    For a camera pose looking at the origin with the camera at position [0, 0, 1]:
    cam2world = LookAtPoseSampler.sample(math.pi/2, math.pi/2, torch.tensor([0, 0, 0]), radius=1)
    """

    @staticmethod
    def sample(horizontal_mean, vertical_mean, lookat_position, horizontal_stddev=0, vertical_stddev=0, radius=1, batch_size=1, device='cpu'):
        h = torch.randn((batch_size, 1), device=device) * horizontal_stddev + horizontal_mean
        v = torch.randn((batch_size, 1), device=device) * vertical_stddev + vertical_mean
        v = torch.clamp(v, 1e-5, math.pi - 1e-5)

        theta = h
        v = v / math.pi
        phi = torch.arccos(1 - 2*v)

        camera_origins = torch.zeros((batch_size, 3), device=device)

        camera_origins[:, 0:1] = radius*torch.sin(phi) * torch.cos(math.pi-theta)
        camera_origins[:, 2:3] = radius*torch.sin(phi) * torch.sin(math.pi-theta)
        camera_origins[:, 1:2] = radius*torch.cos(phi)

        # forward_vectors = math_utils.normalize_vecs(-camera_origins)
        forward_vectors = math_utils.normalize_vecs(lookat_position - camera_origins)
        return create_cam2world_matrix(forward_vectors, camera_origins)

class UniformCameraPoseSampler:
    """
    Same as GaussianCameraPoseSampler, except the
    pose is sampled from a uniform distribution with range +-[horizontal/vertical]_stddev.

    Example:
    For a batch of random camera poses looking at the origin with yaw sampled from [-pi/2, +pi/2] radians:

    cam2worlds = UniformCameraPoseSampler.sample(math.pi/2, math.pi/2, horizontal_stddev=math.pi/2, radius=1, batch_size=16)
    """

    @staticmethod
    def sample(horizontal_mean, vertical_mean, horizontal_stddev=0, vertical_stddev=0, radius=1, batch_size=1, device='cpu'):
        h = (torch.rand((batch_size, 1), device=device) * 2 - 1) * horizontal_stddev + horizontal_mean
        v = (torch.rand((batch_size, 1), device=device) * 2 - 1) * vertical_stddev + vertical_mean
        v = torch.clamp(v, 1e-5, math.pi - 1e-5)

        theta = h
        v = v / math.pi
        phi = torch.arccos(1 - 2*v)

        camera_origins = torch.zeros((batch_size, 3), device=device)

        camera_origins[:, 0:1] = radius*torch.sin(phi) * torch.cos(math.pi-theta)
        camera_origins[:, 2:3] = radius*torch.sin(phi) * torch.sin(math.pi-theta)
        camera_origins[:, 1:2] = radius*torch.cos(phi)

        forward_vectors = math_utils.normalize_vecs(-camera_origins)
        return create_cam2world_matrix(forward_vectors, camera_origins)    



class Live3DCameraPoseSampler:
    '''
    Same as UniformCameraPoseSampler, except the
    pose is sampled from a uniform distribution with range +-[horizontal/vertical/roll]_stddev.
    
    Camera radius sampling:
        Mean: 2.7
        Standard Deviation: 0.1
    
    Camera roll sampling (in degrees, converted to radians):
        Mean: 0
        Standard Deviation: 2 degrees

    Example:
    For a batch of random camera poses looking at the origin with yaw sampled from [-pi/2, +pi/2] radians:

    cam2worlds = UniformCameraPoseSampler.sample(math.pi/2, math.pi/2, horizontal_stddev=math.pi/2, radius=1, batch_size=16)
    
    '''
    @staticmethod
    def sample(horizontal_mean, vertical_mean, lookat_position = torch.tensor([0, 0, 0], device='cpu'), roll_mean_deg = 0, horizontal_stddev=0, vertical_stddev=0, roll_stddev_deg=2, radius_mean=2.7, radius_stddev=0.1, batch_size=1, device='cpu'):
        h = (torch.rand((batch_size, 1), device=device) * 2 - 1) * horizontal_stddev + horizontal_mean
        v = (torch.rand((batch_size, 1), device=device) * 2 - 1) * vertical_stddev + vertical_mean
        v = torch.clamp(v, 1e-5, math.pi - 1e-5)

        theta = h
        v = v / math.pi
        phi = torch.arccos(1 - 2 * v)

        camera_origins = torch.zeros((batch_size, 3), device=device)

        # Create camera origins with radius sampled from a normal distribution with mean 2.7 and std deviation 0.1.
        camera_origins[:, 0:1] = (radius_stddev * torch.randn((batch_size, 1), device=device) + radius_mean) * torch.sin(phi) * torch.cos(math.pi - theta)
        camera_origins[:, 2:3] = (radius_stddev * torch.randn((batch_size, 1), device=device) + radius_mean) * torch.sin(phi) * torch.sin(math.pi - theta)
        camera_origins[:, 1:2] = (radius_stddev * torch.randn((batch_size, 1), device=device) + radius_mean) * torch.cos(phi)

        forward_vectors = math_utils.normalize_vecs(lookat_position -camera_origins)

        # Sample roll from a normal distribution with mean 0 and std deviation 2 degrees.
        roll_stddev_rad = math.radians(roll_stddev_deg)
        roll_mean_rad = math.radians(roll_mean_deg)
        roll = torch.randn((batch_size, 1), device=device) * roll_stddev_rad + roll_mean_rad
        roll_matrix = torch.stack([
            torch.cos(roll), -torch.sin(roll), torch.zeros_like(roll),
            torch.sin(roll), torch.cos(roll), torch.zeros_like(roll),
            torch.zeros_like(roll), torch.zeros_like(roll), torch.ones_like(roll)
        ], dim=-1).view(batch_size, 3, 3)

        # Apply roll to the forward_vectors.
        forward_vectors = torch.matmul(forward_vectors, roll_matrix)

        return create_cam2world_matrix(forward_vectors, camera_origins)


def create_cam2world_matrix(forward_vector, origin):
    """
    Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix.
    Works on batches of forward_vectors, origins. Assumes y-axis is up and that there is no camera roll.
    """

    forward_vector = math_utils.normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=origin.device).expand_as(forward_vector)

    right_vector = -math_utils.normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))
    up_vector = math_utils.normalize_vecs(torch.cross(forward_vector, right_vector, dim=-1))

    rotation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), axis=-1)

    translation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin
    cam2world = (translation_matrix @ rotation_matrix)[:, :, :]
    assert(cam2world.shape[1:] == (4, 4))
    return cam2world


def FOV_to_intrinsics(fov_mean = 18.83, fov_stddev = 1, principal_point_mean = 256, principal_point_stddev = 14, device='cpu'):
    """
    Creates a 3x3 camera intrinsics matrix from the camera field of view, specified in degrees.
    Note the intrinsics are returned as normalized by image size, rather than in pixel units.
    the camera intrinsic matrix is a 3x3 matrix, where
        f_x, f_y are the focal length in x and y direction
        x_0, y_0 are principal point offsets
        s is the skew factor
        size_x, size_y are the image dimensions
    [ f_x/size_x,   s,            x_0/size_x]
    [ 0,            f_y/size_y,   y_0/size_y]
    [ 0,            0,            1         ]
    
  
    FOV sampling (in degrees):
        Mean: 18.83
        Standard Deviation: 1
    
    Principal point sampling (before normalized by image size):
        Mean: 256
        Standard Deviation: 14
    """
    # normal distribution,standard deviation 1, centered at 18.83
    fov_degrees = torch.randn(1, device=device) * fov_stddev + fov_mean
    # Principal point sampling: normal distribution, standard deviation 14, centered at 256
    principal_point = (torch.randn(2, device=device) * principal_point_stddev + principal_point_mean) / 512.0
    
    focal_length = float(1 / (math.tan(fov_degrees * 3.14159 / 360) * 1.414))
    intrinsics = torch.tensor([
         [focal_length, 0, principal_point[0]], 
         [0, focal_length, principal_point[1]], 
         [0, 0, 1]], device=device)
    
    return intrinsics