# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for manager-based reorientation success-rate tracking."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import CommandTerm

from isaaclab_tasks.core.reorient.mdp.commands import ReorientCommand
from isaaclab_tasks.core.reorient.mdp.rewards import ReorientSuccessBonus, evaluate_reorient_success


class _CommandManager:
    def __init__(self, command_term):
        self._command_term = command_term

    def get_term(self, command_name):
        assert command_name == "object_pose"
        return self._command_term


def _data(value):
    return SimpleNamespace(torch=value)


def _make_reward_term(object_quat, success_count_threshold):
    num_envs = len(object_quat)
    command_term = SimpleNamespace(
        cfg=SimpleNamespace(
            orientation_success_threshold=0.0,
            success_count_threshold=success_count_threshold,
        ),
        metrics={"consecutive_success": torch.zeros(num_envs)},
        quat_command_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(num_envs, 1),
    )
    asset = SimpleNamespace(
        data=SimpleNamespace(
            root_quat_w=_data(torch.tensor(object_quat)),
            root_pos_w=_data(torch.zeros(num_envs, 3)),
        )
    )
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene={"object": asset},
        command_manager=_CommandManager(command_term),
        extras={},
    )
    cfg = SimpleNamespace(params={"command_name": "object_pose"})
    return ReorientSuccessBonus(cfg, env), env, command_term


def test_terminal_goal_is_included_in_episode_success_rate():
    reward_term, env, _ = _make_reward_term(object_quat=[[0.0, 0.0, 0.0, 1.0]], success_count_threshold=1)

    reward = reward_term(env, "object_pose")
    reward_term.reset(torch.tensor([0]))

    assert reward.tolist() == [True]
    assert env.extras["log"]["Metrics/success_rate"] == pytest.approx(1.0)


def test_terminal_success_survives_the_full_autoreset_lifecycle(monkeypatch):
    reward_term, env, command_state = _make_reward_term(object_quat=[[0.0, 0.0, 0.0, 1.0]], success_count_threshold=1)
    command = ReorientCommand.__new__(ReorientCommand)
    command.cfg = command_state.cfg
    command.cfg.update_goal_on_success = True
    command._env = SimpleNamespace(common_step_counter=5)
    command.object = env.scene["object"]
    command.pos_command_w = torch.zeros(1, 3)
    command.quat_command_w = command_state.quat_command_w
    command.metrics = {
        "orientation_error": torch.zeros(1),
        "position_error": torch.zeros(1),
        "consecutive_success": torch.zeros(1),
    }
    command._goal_reached = torch.zeros(1, dtype=torch.bool)
    command._reset_step = torch.full((1,), -1, dtype=torch.long)
    resampled_env_ids = []
    command._resample = lambda env_ids: resampled_env_ids.append(env_ids.clone())
    env.command_manager = _CommandManager(command)

    # ManagerBasedRLEnv computes rewards before autoreset, then resets the reward
    # manager before the command manager. Avoid the base command's unrelated goal
    # sampling while exercising that exact term lifecycle.
    monkeypatch.setattr(CommandTerm, "reset", lambda _self, _env_ids=None: {})
    reward_term(env, "object_pose")
    reward_term.reset(torch.tensor([0]))
    ReorientCommand.reset(command, torch.tensor([0]))

    # The freshly reset pose happens to match the new goal. Command computation
    # must neither count it nor resample it during the autoreset step.
    ReorientCommand._update_metrics(command)
    ReorientCommand._update_command(command)

    assert env.extras["log"]["Metrics/success_rate"] == pytest.approx(1.0)
    assert command.metrics["consecutive_success"].tolist() == [0.0]
    assert len(resampled_env_ids) == 1
    assert resampled_env_ids[0].numel() == 0


def test_episode_success_counts_multiple_goals_and_clears_on_reset():
    reward_term, env, _ = _make_reward_term(
        object_quat=[
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        success_count_threshold=2,
    )

    reward_term(env, "object_pose")
    env.scene["object"].data.root_quat_w.torch[:] = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    reward_term(env, "object_pose")
    reward_term.reset(torch.arange(3))

    assert env.extras["log"]["Metrics/success_rate"] == pytest.approx(1.0 / 3.0)

    # A second reset must not reuse the previous episode's successes.
    reward_term.reset(torch.arange(3))
    assert env.extras["log"]["Metrics/success_rate"] == pytest.approx(0.0)


def test_success_equality_is_counted_and_resampled_once():
    success, _ = evaluate_reorient_success(
        torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        0.0,
    )
    assert success.tolist() == [True]

    command = ReorientCommand.__new__(ReorientCommand)
    command.cfg = SimpleNamespace(orientation_success_threshold=0.0, update_goal_on_success=True)
    command._env = SimpleNamespace(common_step_counter=5)
    command.object = SimpleNamespace(
        data=SimpleNamespace(
            root_quat_w=_data(torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])),
            root_pos_w=_data(torch.zeros(2, 3)),
        )
    )
    command.quat_command_w = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
    command.pos_command_w = torch.zeros(2, 3)
    command.metrics = {
        "orientation_error": torch.zeros(2),
        "position_error": torch.zeros(2),
        "consecutive_success": torch.zeros(2),
    }
    command._goal_reached = torch.zeros(2, dtype=torch.bool)
    # Environment 0 was reset during this step; environment 1 was not.
    command._reset_step = torch.tensor([5, 4])
    resampled_env_ids = []
    command._resample = lambda env_ids: resampled_env_ids.append(env_ids.clone())

    command._update_metrics()
    command._update_command()

    assert command.metrics["consecutive_success"].tolist() == [0.0, 1.0]
    assert len(resampled_env_ids) == 1
    assert resampled_env_ids[0].tolist() == [1]
