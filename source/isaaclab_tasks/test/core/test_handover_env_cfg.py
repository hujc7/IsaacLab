# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_tasks.core.handover.handover_env_cfg import _shadow_hand_cfg


def test_newton_shadow_hand_actuators_only_target_available_joints():
    """Verify the Newton handover config does not add actuators for removed J0 joints."""
    hand_cfg = _shadow_hand_cfg(
        prim_path="/World/envs/env_.*/Robot",
        init_pos=(0.0, 0.0, 0.5),
        init_rot=(0.0, 0.0, 0.0, 1.0),
    )

    # Assert the stated intent rather than the actuator-group names, which are an implementation
    # detail: the hand's twenty motors are split across a direct-drive group and a tendon-coupled
    # one, and that split is free to change.
    patterns = [expr for act in hand_cfg.newton_mjwarp.actuators.values() for expr in act.joint_names_expr]
    assert patterns, "expected the Newton hand to declare at least one actuator"
    assert not [expr for expr in patterns if "J0" in expr], f"actuator targets a removed J0 joint: {patterns}"
