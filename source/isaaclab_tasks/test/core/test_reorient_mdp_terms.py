# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for stateful manager terms used by the reorientation tasks."""

from types import SimpleNamespace

import torch

from isaaclab.envs.mdp.actions import EMAJointPositionToLimitsAction

from isaaclab_tasks.core.reorient import mdp
from isaaclab_tasks.core.reorient.mdp.action_terms import ReorientEMAJointPositionToLimitsAction
from isaaclab_tasks.core.reorient.mdp.terminations import ReorientTimeout


def _data(value: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(torch=value)


def test_reorient_action_retains_terminal_action_for_reset_observation(monkeypatch):
    def reset_raw_actions(action_term, env_ids=None):
        reset_env_ids = slice(None) if env_ids is None else env_ids
        action_term._raw_actions[reset_env_ids] = 0.0

    monkeypatch.setattr(EMAJointPositionToLimitsAction, "reset", reset_raw_actions)

    action_term = ReorientEMAJointPositionToLimitsAction.__new__(ReorientEMAJointPositionToLimitsAction)
    action_term._env = SimpleNamespace(common_step_counter=11)
    action_term._raw_actions = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    action_term._reset_raw_actions = torch.zeros_like(action_term.raw_actions)
    action_term._reset_step = torch.full((2,), -1, dtype=torch.long)

    action_term.reset(torch.tensor([0]))

    assert action_term.raw_actions.tolist() == [[0.0, 0.0], [3.0, 4.0]]
    assert action_term.observation_actions.tolist() == [[1.0, 2.0], [3.0, 4.0]]

    action_term._env.common_step_counter = 12
    assert action_term.observation_actions.tolist() == action_term.raw_actions.tolist()

    action_term._raw_actions[:] = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    action_term.reset()
    assert action_term.raw_actions.tolist() == [[0.0, 0.0], [0.0, 0.0]]
    assert action_term.observation_actions.tolist() == [[5.0, 6.0], [7.0, 8.0]]


def test_reorient_timeout_restarts_when_goal_is_reached():
    command_term = SimpleNamespace(
        cfg=SimpleNamespace(orientation_success_threshold=0.0),
        quat_command_w=torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]),
    )
    asset = SimpleNamespace(
        data=SimpleNamespace(root_quat_w=_data(torch.tensor([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]])))
    )
    env = SimpleNamespace(
        scene={"object": asset},
        command_manager=SimpleNamespace(get_term=lambda _: command_term),
        max_episode_length=7,
    )

    timeout = ReorientTimeout.__new__(ReorientTimeout)
    timeout._steps_since_success = torch.tensor([5, 5])

    timed_out = timeout(env, "object_pose")

    assert timeout._steps_since_success.tolist() == [0, 6]
    assert timed_out.tolist() == [False, True]

    timeout.reset(torch.tensor([1]))
    assert timeout._steps_since_success.tolist() == [0, 0]


def test_reorient_mdp_lazy_exports_resolve():
    missing = [name for name in mdp.__all__ if not hasattr(mdp, name)]
    assert not missing
