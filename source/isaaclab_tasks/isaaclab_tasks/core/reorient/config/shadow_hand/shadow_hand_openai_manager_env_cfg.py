# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based counterpart of the OpenAI Shadow Hand reorientation variants (FF and LSTM)."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import JointWrenchSensorCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.reorient.mdp as mdp
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_common import (
    GOAL_OBJECT_CFG,
    OPENAI_ACTION_NOISE_CFG,
    OPENAI_OBSERVATION_NOISE_CFG,
    NewtonEventCfg,
    PhysicsCfg,
    PhysxEventCfg,
)
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_manager_env_cfg import (
    FullStateWithoutActionCfg,
    ShadowHandManagerSceneCfg,
)
from isaaclab_tasks.core.reorient.reorient_common import GOAL_MARKER_POSITION, IN_HAND_POS_OFFSET
from isaaclab_tasks.core.reorient.reorient_manager_env_cfg import ReorientObjectEnvCfg
from isaaclab_tasks.core.reorient.reorient_manager_env_cfg import RewardsCfg as ReorientRewardsCfg
from isaaclab_tasks.core.reorient.reorient_manager_env_cfg import TerminationsCfg as ReorientTerminationsCfg
from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.shadow_hand import SHADOW_ACTUATED_JOINT_NAMES, SHADOW_FINGERTIP_BODY_NAMES


@configclass
class OpenAICommandsCfg:
    """OpenAI goal command with its wider success tolerance."""

    object_pose = mdp.ReorientEpisodeCommandCfg(
        asset_name="object",
        init_pos_offset=IN_HAND_POS_OFFSET,
        update_goal_on_success=True,
        orientation_success_threshold=0.4,
        make_quat_unique=False,
        fixed_marker_pos=GOAL_MARKER_POSITION,
        goal_pose_visualizer_cfg=GOAL_OBJECT_CFG,
        debug_vis=True,
    )


@configclass
class OpenAIActionsCfg:
    """OpenAI actions with Direct-compatible EMA and stateful noise."""

    joint_pos = mdp.NoisyEMAJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=SHADOW_ACTUATED_JOINT_NAMES,
        alpha=0.3,
        rescale_to_limits=True,
        noise_model=OPENAI_ACTION_NOISE_CFG,
    )


@configclass
class OpenAIObservationsCfg:
    """OpenAI 42-dimensional actor and 187-dimensional critic observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        openai = ObsTerm(
            func=mdp.openai_policy_observation,
            params={
                "command_name": "object_pose",
                "action_name": "joint_pos",
                "noise_model": OPENAI_OBSERVATION_NOISE_CFG,
                "robot_cfg": SceneEntityCfg("robot", body_names=SHADOW_FINGERTIP_BODY_NAMES, preserve_order=False),
                "object_cfg": SceneEntityCfg("object"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(FullStateWithoutActionCfg):
        # -- contact sensing
        fingertip_wrench = ObsTerm(
            func=mdp.fingertip_wrench,
            scale=10.0,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "joint_wrench", body_names=SHADOW_FINGERTIP_BODY_NAMES, preserve_order=False
                )
            },
        )
        # -- action
        last_action = ObsTerm(func=mdp.reorient_last_action, params={"action_name": "joint_pos"})

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class ShadowHandOpenAIManagerSceneCfg(ShadowHandManagerSceneCfg):
    """Shadow Hand scene with fingertip joint-wrench sensing."""

    joint_wrench = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")


_OPENAI_RESET_PARAMS = {
    "position_noise": 0.01,
    "joint_position_noise": 0.2,
    "joint_velocity_noise": 0.0,
    "action_name": "joint_pos",
}


@configclass
class OpenAIPhysxEventCfg(PhysxEventCfg):
    """PhysX OpenAI randomization and state reset events."""

    reset_state = EventTerm(func=mdp.reset_reorient_state, mode="reset", params=_OPENAI_RESET_PARAMS)


@configclass
class OpenAINewtonEventCfg(NewtonEventCfg):
    """Newton OpenAI randomization and state reset events."""

    reset_state = EventTerm(func=mdp.reset_reorient_state, mode="reset", params=_OPENAI_RESET_PARAMS)


@configclass
class OpenAIEventCfg(PresetCfg):
    """Backend-specific OpenAI event alternatives."""

    physx = OpenAIPhysxEventCfg()
    newton_mjwarp = OpenAINewtonEventCfg()
    ovphysx = physx
    newton_kamino = newton_mjwarp
    default = newton_mjwarp


@configclass
class OpenAIRewardsCfg(ReorientRewardsCfg):
    """Shared reward terms tuned to the Direct OpenAI variant's scales."""

    track_pos_l2 = RewTerm(
        func=mdp.track_pos_l2,
        weight=-10.0,
        params={"command_name": "object_pose", "object_cfg": SceneEntityCfg("object")},
    )
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.0002)
    object_away_penalty = RewTerm(
        func=mdp.is_terminated_term,
        weight=-50.0,
        params={"term_keys": "object_out_of_reach"},
    )
    joint_vel_l2 = None
    action_rate_l2 = None


@configclass
class OpenAITerminationsCfg(ReorientTerminationsCfg):
    """Shared terminations with the OpenAI streak cap and success-extended timer.

    The Direct variant reports both the streak cap and the elapsed-time limit as
    truncations, so both carry ``time_out=True``.
    """

    object_out_of_reach = DoneTerm(
        func=mdp.object_away_from_goal,
        params={
            "threshold": 0.24,
            "command_name": "object_pose",
            "object_cfg": SceneEntityCfg("object"),
        },
    )
    max_consecutive_success = DoneTerm(
        func=mdp.max_consecutive_success,
        time_out=True,
        params={"num_success": 50, "command_name": "object_pose"},
    )
    time_out = DoneTerm(
        func=mdp.reorient_timeout,
        time_out=True,
        params={
            "command_name": "object_pose",
            "success_tolerance": 0.4,
            "object_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class ShadowHandOpenAIManagerEnvCfg(ReorientObjectEnvCfg):
    """Manager counterpart shared by the OpenAI FF and LSTM variants.

    Standalone rather than a subclass of :class:`ShadowHandManagerEnvCfg`:
    every section differs from the state task, so this block is the complete
    recipe.
    """

    scene: ShadowHandOpenAIManagerSceneCfg = ShadowHandOpenAIManagerSceneCfg()
    observations: OpenAIObservationsCfg = OpenAIObservationsCfg()
    actions: OpenAIActionsCfg = OpenAIActionsCfg()
    commands: OpenAICommandsCfg = OpenAICommandsCfg()
    rewards: OpenAIRewardsCfg = OpenAIRewardsCfg()
    terminations: OpenAITerminationsCfg = OpenAITerminationsCfg()
    events: OpenAIEventCfg = OpenAIEventCfg()

    def __post_init__(self):
        super().__post_init__()
        # mirrors the Direct cfg
        self.decimation = 3
        self.episode_length_s = 8.0
        self.sim.dt = 1 / 60
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysicsCfg()
        self.viewer.eye = (2.0, 2.0, 2.0)
