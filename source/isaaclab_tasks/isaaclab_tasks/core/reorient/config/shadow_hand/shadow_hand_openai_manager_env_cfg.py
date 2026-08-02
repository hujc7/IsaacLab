# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based counterpart of the OpenAI Shadow Hand reorientation variants (FF and LSTM).

The observation, action-noise, and episode conventions follow OpenAI et al., "Learning
Dexterous In-Hand Manipulation" (https://arxiv.org/abs/1808.00177).
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import JointWrenchSensorCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.reorient.mdp as mdp
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_common import (
    OPENAI_ACTION_NOISE_CFG,
    OPENAI_OBSERVATION_NOISE_CFG,
    ShadowHandOpenAIEventCfg,
)
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_manager_env_cfg import (
    CommandsCfg as ShadowHandCommandsCfg,
)
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_manager_env_cfg import (
    FullStateObsCfg,
    ShadowHandManagerEnvCfg,
    ShadowHandManagerSceneCfg,
)
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_manager_env_cfg import (
    RewardsCfg as ShadowHandRewardsCfg,
)
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_manager_env_cfg import (
    TerminationsCfg as ShadowHandTerminationsCfg,
)

from isaaclab_assets.robots.shadow_hand import SHADOW_ACTUATED_JOINT_NAMES, SHADOW_FINGERTIP_BODY_NAMES


@configclass
class CommandsCfg(ShadowHandCommandsCfg):
    """OpenAI goal command with its wider success tolerance."""

    object_pose = ShadowHandCommandsCfg().object_pose.replace(orientation_success_threshold=0.4)


@configclass
class ActionsCfg:
    """OpenAI actions with Direct-compatible EMA and stateful noise."""

    joint_pos = mdp.ReorientEMAJointPositionToLimitsActionWithNoiseCfg(
        asset_name="robot",
        joint_names=SHADOW_ACTUATED_JOINT_NAMES,
        alpha=0.3,
        rescale_to_limits=True,
        noise_model=OPENAI_ACTION_NOISE_CFG,
    )


@configclass
class ObservationsCfg:
    """OpenAI 42-dimensional actor and 187-dimensional critic observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        openai = ObsTerm(
            func=mdp.openai_policy_observation,
            noise=OPENAI_OBSERVATION_NOISE_CFG,
            params={
                "command_name": "object_pose",
                "action_name": "joint_pos",
                "robot_cfg": SceneEntityCfg("robot", body_names=SHADOW_FINGERTIP_BODY_NAMES, preserve_order=False),
                "object_cfg": SceneEntityCfg("object"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(FullStateObsCfg):
        fingertip_wrench = ObsTerm(
            func=mdp.FingertipWrench,
            scale=10.0,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "joint_wrench", body_names=SHADOW_FINGERTIP_BODY_NAMES, preserve_order=False
                )
            },
        )
        last_action = ObsTerm(func=mdp.reorient_last_action, params={"action_name": "joint_pos"})

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class ShadowHandOpenAIManagerSceneCfg(ShadowHandManagerSceneCfg):
    """Shadow Hand scene with fingertip joint-wrench sensing."""

    joint_wrench = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class RewardsCfg(ShadowHandRewardsCfg):
    """Shadow state rewards with the OpenAI fall penalty."""

    object_away_penalty = RewTerm(
        func=mdp.is_terminated_term,
        weight=-50.0,
        params={"term_keys": "object_out_of_reach"},
    )


@configclass
class TerminationsCfg(ShadowHandTerminationsCfg):
    """Shadow terminations with the OpenAI streak cap and success-extended timer.

    The Direct variant reports both the streak cap and the elapsed-time limit as
    truncations, so both carry the time-out flag.
    """

    max_consecutive_success = DoneTerm(
        func=mdp.max_consecutive_success,
        time_out=True,
        params={"num_success": 50, "command_name": "object_pose"},
    )
    time_out = DoneTerm(
        func=mdp.ReorientTimeout,
        time_out=True,
        params={
            "command_name": "object_pose",
            "object_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class ShadowHandOpenAIManagerEnvCfg(ShadowHandManagerEnvCfg):
    """Manager counterpart shared by the OpenAI FF and LSTM variants."""

    scene: ShadowHandOpenAIManagerSceneCfg = ShadowHandOpenAIManagerSceneCfg()
    decimation = 3
    episode_length_s = 8.0
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: ShadowHandOpenAIEventCfg = ShadowHandOpenAIEventCfg()

    def __post_init__(self):
        """Apply the OpenAI control and simulation rates to the inherited simulation."""

        super().__post_init__()
        self.decimation = 3
        self.episode_length_s = 8.0
        self.sim.dt = 1 / 60
        self.sim.render_interval = self.decimation
