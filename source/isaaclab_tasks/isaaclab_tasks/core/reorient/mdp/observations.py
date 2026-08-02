# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions specific to the in-hand dexterous manipulation environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import JointWrenchSensor

    from .action_terms import ReorientEMAJointPositionToLimitsAction
    from .commands import ReorientCommand


CUBE_HALF_SIZE: tuple[float, float, float] = (0.03, 0.03, 0.03)
"""Half side lengths [m] of the reorientation cube."""


# -- cube keypoint helpers, shared by the camera and state observation terms
def _cube_corner_offsets(
    size: tuple[float, float, float], num_keypoints: int, device: torch.device | str
) -> torch.Tensor:
    """Corner offsets [m] from the cube center; corner index bits select the +/- half side per axis."""
    signs = torch.tensor(
        [[1 - 2 * ((corner >> axis) & 1) for axis in range(3)] for corner in range(num_keypoints)],
        dtype=torch.float32,
        device=device,
    )
    half_size = torch.tensor(size, dtype=torch.float32, device=device) / 2.0
    return signs * half_size


def compute_cube_keypoints(
    pose: torch.Tensor,
    num_keypoints: int = 8,
    size: tuple[float, float, float] = (2 * 0.03, 2 * 0.03, 2 * 0.03),
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute cube-corner positions for batched poses.

    Args:
        pose: Cube center poses ``(x, y, z, qx, qy, qz, qw)`` [m, unit quaternion].
        num_keypoints: Number of binary-sign corners to compute.
        size: Cube side lengths along each axis [m].
        out: Optional output buffer [m], shape ``(num_envs, num_keypoints, 3)``.

    Returns:
        Cube-corner positions [m], shape ``(num_envs, num_keypoints, 3)``.
    """
    # Vectorized over corners: the earlier implementation looped over the eight corners,
    # allocating a tensor and calling quat_apply once per corner. The corner sign-offsets
    # are pose-independent, so they are built once and all num_keypoints corners are rotated
    # by the pose in a single batched quat_apply — mathematically identical, no Python loop.
    num_envs = pose.shape[0]
    corners = _cube_corner_offsets(size, num_keypoints, pose.device)
    # Broadcast each env's quaternion across its corners and rotate every offset at once.
    rotated = math_utils.quat_apply(
        pose[:, None, 3:7].expand(num_envs, num_keypoints, 4), corners.expand(num_envs, num_keypoints, 3)
    )
    # Translate the rotated offsets by the cube-center position to get world-frame corners.
    keypoints = pose[:, None, 0:3] + rotated
    if out is None:
        return keypoints
    out.copy_(keypoints)
    return out


def cube_keypoints_from_quat(
    quat: torch.Tensor,
    half_size: tuple[float, float, float] = CUBE_HALF_SIZE,
    num_keypoints: int = 8,
) -> torch.Tensor:
    """Rotation-only cube-corner offsets [m] from batched ``(x, y, z, w)`` orientations.

    Args:
        quat: Cube orientations, shape ``(num_envs, 4)``.
        half_size: Cube half side lengths along each axis [m].
        num_keypoints: Number of binary-sign corners to compute.

    Returns:
        Flattened corner offsets [m], shape ``(num_envs, num_keypoints * 3)``.
    """
    num_envs = quat.shape[0]
    size = (2.0 * half_size[0], 2.0 * half_size[1], 2.0 * half_size[2])
    corners = _cube_corner_offsets(size, num_keypoints, quat.device)
    rotated = math_utils.quat_apply(
        quat[:, None, :].expand(num_envs, num_keypoints, 4), corners.expand(num_envs, num_keypoints, 3)
    )
    return rotated.reshape(num_envs, num_keypoints * 3)


def goal_quat_diff(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, make_quat_unique: bool
) -> torch.Tensor:
    """Goal orientation relative to the asset's root frame.

    The quaternion is represented as (w, x, y, z). The real part is always positive.
    """
    # extract useful elements
    asset: RigidObject = env.scene[asset_cfg.name]
    command_term: ReorientCommand = env.command_manager.get_term(command_name)

    # obtain the orientations
    goal_quat_w = command_term.command[:, 3:7]
    asset_quat_w = asset.data.root_quat_w.torch

    # compute quaternion difference
    quat = math_utils.quat_mul(asset_quat_w, math_utils.quat_conjugate(goal_quat_w))
    # make sure the quaternion real-part is always positive
    return math_utils.quat_unique(quat) if make_quat_unique else quat


# -- fingertip terms
def fingertip_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return flattened fingertip positions in the environment frame.

    Args:
        env: Environment containing the robot.
        asset_cfg: Robot scene entity with the fingertip bodies selected.

    Returns:
        Fingertip positions [m], shape ``(num_envs, num_fingertips * 3)``.
    """
    asset = env.scene[asset_cfg.name]
    positions = asset.data.body_pos_w.torch[:, asset_cfg.body_ids] - env.scene.env_origins.unsqueeze(1)
    return positions.reshape(env.num_envs, -1)


def fingertip_quat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return flattened fingertip orientations.

    Args:
        env: Environment containing the robot.
        asset_cfg: Robot scene entity with the fingertip bodies selected.

    Returns:
        Unit quaternions in ``(x, y, z, w)`` order, shape ``(num_envs, num_fingertips * 4)``.
    """
    asset = env.scene[asset_cfg.name]
    return asset.data.body_quat_w.torch[:, asset_cfg.body_ids].reshape(env.num_envs, -1)


def fingertip_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return flattened fingertip spatial velocities.

    Args:
        env: Environment containing the robot.
        asset_cfg: Robot scene entity with the fingertip bodies selected.

    Returns:
        Linear and angular velocities [m/s, rad/s], shape ``(num_envs, num_fingertips * 6)``.
    """
    asset = env.scene[asset_cfg.name]
    return asset.data.body_vel_w.torch[:, asset_cfg.body_ids].reshape(env.num_envs, -1)


# -- action terms
def reorient_last_action(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    """Return the Direct-compatible last action across same-step autoreset.

    Args:
        env: Environment containing the action term.
        action_name: Action term whose raw action is observed.

    Returns:
        Normalized raw actions, retaining each terminal action in its same-step reset observation.
    """
    action_term: ReorientEMAJointPositionToLimitsAction = env.action_manager.get_term(action_name)
    return action_term.observation_actions


def openai_policy_observation(
    env: ManagerBasedRLEnv,
    command_name: str,
    action_name: str,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return the 42-dimensional actor observation from the OpenAI Shadow Hand task.

    Args:
        env: Environment containing the robot, object, command, and action term.
        command_name: Command term used to extract the orientation goal.
        action_name: Action term whose normalized raw action is observed.
        robot_cfg: Robot scene entity with the fingertip bodies selected.
        object_cfg: Object scene entity.

    Returns:
        Actor observations containing fingertip positions [m], object position [m],
        orientation error as a unit quaternion, and normalized actions; shape ``(num_envs, 42)``.
    """
    object_asset: RigidObject = env.scene[object_cfg.name]
    object_pos = object_asset.data.root_pos_w.torch - env.scene.env_origins
    command_term: ReorientCommand = env.command_manager.get_term(command_name)
    quat_error = math_utils.quat_mul(
        object_asset.data.root_quat_w.torch, math_utils.quat_conjugate(command_term.quat_command_w)
    )
    return torch.cat(
        (fingertip_pos(env, robot_cfg), object_pos, quat_error, reorient_last_action(env, action_name)), dim=-1
    )


class FingertipWrench(ManagerTermBase):
    """Fingertip reaction wrenches [N, N·m] with a Direct-compatible zero fallback."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        body_ids = cfg.params["sensor_cfg"].body_ids
        # Direct reports zeros until the sensor has produced its first sample.
        self._zeros = torch.zeros(env.num_envs, len(body_ids) * 6, dtype=torch.float32, device=env.device)

    def __call__(self, env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
        """Return flattened fingertip reaction wrenches.

        Args:
            env: Environment containing the wrench sensor.
            sensor_cfg: Joint-wrench sensor with the fingertip bodies selected.

        Returns:
            Forces and torques [N, N·m], shape ``(num_envs, num_fingertips * 6)``.
        """
        sensor: JointWrenchSensor = env.scene.sensors[sensor_cfg.name]
        force_data = sensor.data.force
        torque_data = sensor.data.torque
        if force_data is None or torque_data is None:
            return self._zeros
        force = force_data.torch[:, sensor_cfg.body_ids]
        torque = torque_data.torch[:, sensor_cfg.body_ids]
        return torch.cat((force, torque), dim=-1).reshape(env.num_envs, -1)
