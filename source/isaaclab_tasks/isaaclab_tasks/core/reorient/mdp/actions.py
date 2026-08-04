# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action configurations for the reorientation task family."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.envs.mdp import EMAJointPositionToLimitsActionCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import NoiseModelCfg

if TYPE_CHECKING:
    from .action_terms import (
        ReorientEMAJointPositionToLimitsAction,
        ReorientEMAJointPositionToLimitsActionWithNoise,
    )


@configclass
class ReorientEMAJointPositionToLimitsActionCfg(EMAJointPositionToLimitsActionCfg):
    """EMA joint action that retains terminal actions for same-step reset observations."""

    class_type: type[ReorientEMAJointPositionToLimitsAction] | str = (
        "{DIR}.action_terms:ReorientEMAJointPositionToLimitsAction"
    )


@configclass
class ReorientEMAJointPositionToLimitsActionWithNoiseCfg(ReorientEMAJointPositionToLimitsActionCfg):
    """EMA joint action configuration with Direct-compatible stateful noise."""

    class_type: type[ReorientEMAJointPositionToLimitsActionWithNoise] | str = (
        "{DIR}.action_terms:ReorientEMAJointPositionToLimitsActionWithNoise"
    )

    noise_model: NoiseModelCfg = MISSING
    """Stateful noise applied to incoming normalized actions."""
