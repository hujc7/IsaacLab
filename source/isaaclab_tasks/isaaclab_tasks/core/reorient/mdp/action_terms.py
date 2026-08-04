# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action term implementations for the reorientation task family.

The implementations are separate from :mod:`~isaaclab_tasks.core.reorient.mdp.actions`
because importing the base action class pulls in USD stage bindings, while configuration
loading must remain independent of those runtime bindings.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions import EMAJointPositionToLimitsAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .actions import (
        ReorientEMAJointPositionToLimitsActionCfg,
        ReorientEMAJointPositionToLimitsActionWithNoiseCfg,
    )


class ReorientEMAJointPositionToLimitsAction(EMAJointPositionToLimitsAction):
    """Retain terminal actions for observations computed after same-step autoreset."""

    def __init__(self, cfg: ReorientEMAJointPositionToLimitsActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._reset_raw_actions = torch.zeros_like(self.raw_actions)
        self._reset_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)

    @property
    def observation_actions(self) -> torch.Tensor:
        """Raw actions, retaining terminal values during same-step reset observations."""
        reset_mask = self._reset_step == self._env.common_step_counter
        return torch.where(reset_mask.unsqueeze(-1), self._reset_raw_actions, self.raw_actions)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Snapshot raw actions before resetting the action buffers.

        Args:
            env_ids: Environment indices to reset, or ``None`` for every environment.
        """
        reset_env_ids = slice(None) if env_ids is None else env_ids
        self._reset_raw_actions[reset_env_ids] = self.raw_actions[reset_env_ids]
        self._reset_step[reset_env_ids] = self._env.common_step_counter
        super().reset(env_ids)


class ReorientEMAJointPositionToLimitsActionWithNoise(ReorientEMAJointPositionToLimitsAction):
    """Apply a stateful noise model before EMA joint-position processing."""

    def __init__(self, cfg: ReorientEMAJointPositionToLimitsActionWithNoiseCfg, env: ManagerBasedRLEnv):
        """Initialize the noisy action term.

        Args:
            cfg: Action configuration including the stateful noise model.
            env: Manager-based environment containing the hand.
        """
        super().__init__(cfg, env)
        self._noise_model = cfg.noise_model.class_type(cfg.noise_model, num_envs=self.num_envs, device=self.device)

    def process_actions(self, actions: torch.Tensor) -> None:
        """Apply noise to normalized actions before scaling and EMA filtering.

        Args:
            actions: Normalized joint actions, shape ``(num_envs, num_actions)``.
        """
        super().process_actions(self._noise_model(actions))

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the noise state and standard EMA action buffers.

        Args:
            env_ids: Environment indices to reset, or ``None`` for every environment.
        """
        self._noise_model.reset(env_ids)
        super().reset(env_ids)
