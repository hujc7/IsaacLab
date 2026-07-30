# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions specific to the in-hand dexterous manipulation environments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

from .rewards import evaluate_reorient_success

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands import ReorientCommand


def max_consecutive_success(env: ManagerBasedRLEnv, num_success: int, command_name: str) -> torch.Tensor:
    """Check if the task has been completed consecutively for a certain number of times.

    Args:
        env: The environment object.
        num_success: Threshold for the number of consecutive successes required.
        command_name: The command term to be used for extracting the goal.
    """
    command_term: ReorientCommand = env.command_manager.get_term(command_name)

    return command_term.metrics["consecutive_success"] >= num_success


def object_away_from_goal(
    env: ManagerBasedRLEnv,
    threshold: float,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Check if object has gone far from the goal.

    The object is considered to be out-of-reach if the distance between the goal and the object is greater
    than the threshold.

    Args:
        env: The environment object.
        threshold: The threshold for the distance between the robot and the object.
        command_name: The command term to be used for extracting the goal.
        object_cfg: The configuration for the scene entity. Default is "object".
    """
    # extract useful elements
    command_term: ReorientCommand = env.command_manager.get_term(command_name)
    asset = env.scene[object_cfg.name]

    # object pos
    asset_pos_e = asset.data.root_pos_w.torch - env.scene.env_origins
    goal_pos_e = command_term.command[:, :3]

    return torch.linalg.norm(asset_pos_e - goal_pos_e, ord=2, dim=1) > threshold


def object_away_from_robot(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Check if object has gone far from the robot.

    The object is considered to be out-of-reach if the distance between the robot and the object is greater
    than the threshold.

    Args:
        env: The environment object.
        threshold: The threshold for the distance between the robot and the object.
        asset_cfg: The configuration for the robot entity. Default is "robot".
        object_cfg: The configuration for the object entity. Default is "object".
    """
    # extract useful elements
    robot = env.scene[asset_cfg.name]
    object = env.scene[object_cfg.name]

    # compute distance
    dist = torch.linalg.norm(robot.data.root_pos_w.torch - object.data.root_pos_w.torch, dim=1)

    return dist > threshold


class reorient_timeout(ManagerTermBase):
    """Time out an episode that has run its full length without reaching a goal.

    The timer restarts on every goal reach, so episodes extend across success streaks.
    This matches the OpenAI Direct variant, which is the only configuration that enables
    the behavior. Pair it with :func:`max_consecutive_success` to also stop on the streak
    cap, and declare both with ``time_out=True``.

    Args:
        cfg: Configuration object specifying term parameters.
        env: The manager-based RL environment.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._steps_since_success = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # resolved on first call: the command term does not exist yet during manager construction
        self._command_term: ReorientCommand | None = None

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._steps_since_success[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        success_tolerance: float,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
        """Return per-environment timeout flags.

        Args:
            env: The environment object.
            command_name: The command term to be used for extracting the goal.
            success_tolerance: Maximum successful orientation error [rad].
            object_cfg: The configuration for the scene entity. Default is "object".
        """
        asset = env.scene[object_cfg.name]
        if self._command_term is None:
            self._command_term = env.command_manager.get_term(command_name)
        # Terminations run before the command manager, so the command's metrics still
        # describe the previous step. Evaluate success here instead, matching the Direct
        # environment, which refreshes the object pose inside its dones computation.
        goal_reached, _ = evaluate_reorient_success(
            asset.data.root_quat_w.torch, self._command_term.quat_command_w, success_tolerance
        )
        self._steps_since_success += 1
        # masked_fill_ rather than boolean indexing: the latter forces a host synchronization
        self._steps_since_success.masked_fill_(goal_reached, 0)

        return self._steps_since_success >= env.max_episode_length - 1
