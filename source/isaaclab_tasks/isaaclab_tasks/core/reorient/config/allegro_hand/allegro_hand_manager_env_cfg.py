# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based counterpart of the Allegro Hand Direct reorientation task."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.reorient.mdp as mdp
from isaaclab_tasks.core.reorient.config.allegro_hand.allegro_hand_common import (
    ALLEGRO_HAND_ROBOT_CFG,
    CUBE_CFG,
    GOAL_OBJECT_CFG,
    CubeCfg,
    PhysicsCfg,
)
from isaaclab_tasks.core.reorient.reorient_common import GOAL_MARKER_POSITION, IN_HAND_POS_OFFSET
from isaaclab_tasks.core.reorient.reorient_manager_env_cfg import EventCfg as ReorientEventCfg
from isaaclab_tasks.core.reorient.reorient_manager_env_cfg import ReorientObjectEnvCfg, ReorientObjectSceneCfg
from isaaclab_tasks.core.reorient.reorient_manager_env_cfg import RewardsCfg as ReorientRewardsCfg
from isaaclab_tasks.core.reorient.reorient_manager_env_cfg import TerminationsCfg as ReorientTerminationsCfg

from isaaclab_assets.robots.allegro import ALLEGRO_ACTUATED_JOINT_NAMES, ALLEGRO_FINGERTIP_BODY_NAMES


@configclass
class AllegroCubeSceneCfg(ReorientObjectSceneCfg):
    """Shared reorientation scene with the Allegro hand and a ground plane."""

    num_envs = 8192
    env_spacing = 0.75

    robot: ArticulationCfg = ALLEGRO_HAND_ROBOT_CFG
    object: CubeCfg = CUBE_CFG
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )
    dome_light = None


@configclass
class CommandsCfg:
    """Object pose goal matching the Direct in-hand target."""

    object_pose = mdp.ReorientEpisodeCommandCfg(
        asset_name="object",
        init_pos_offset=IN_HAND_POS_OFFSET,
        update_goal_on_success=True,
        orientation_success_threshold=0.2,
        make_quat_unique=False,
        fixed_marker_pos=GOAL_MARKER_POSITION,
        goal_pose_visualizer_cfg=GOAL_OBJECT_CFG,
        debug_vis=True,
    )


@configclass
class ActionsCfg:
    """Sixteen actuated Allegro Hand joints in Direct order."""

    joint_pos = mdp.EMAJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=ALLEGRO_ACTUATED_JOINT_NAMES,
        alpha=1.0,
        rescale_to_limits=True,
    )


@configclass
class ObservationsCfg:
    """Full 124-dimensional state observation in Direct order."""

    @configclass
    class PolicyCfg(ObsGroup):
        # -- robot
        joint_pos = ObsTerm(
            func=mdp.joint_pos_limit_normalized,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=False)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            scale=0.2,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=False)},
        )
        # -- object
        object_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
        object_quat = ObsTerm(
            func=mdp.root_quat_w,
            params={"asset_cfg": SceneEntityCfg("object"), "make_quat_unique": False},
        )
        object_lin_vel = ObsTerm(func=mdp.root_lin_vel_w, params={"asset_cfg": SceneEntityCfg("object")})
        object_ang_vel = ObsTerm(
            func=mdp.root_ang_vel_w,
            scale=0.2,
            params={"asset_cfg": SceneEntityCfg("object")},
        )
        # -- command
        goal_pose = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        goal_quat_diff = ObsTerm(
            func=mdp.goal_quat_diff,
            params={"asset_cfg": SceneEntityCfg("object"), "command_name": "object_pose", "make_quat_unique": False},
        )
        # -- robot fingertips
        fingertip_pos = ObsTerm(
            func=mdp.fingertip_pos,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=ALLEGRO_FINGERTIP_BODY_NAMES, preserve_order=False)
            },
        )
        fingertip_quat = ObsTerm(
            func=mdp.fingertip_quat,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=ALLEGRO_FINGERTIP_BODY_NAMES, preserve_order=False)
            },
        )
        fingertip_vel = ObsTerm(
            func=mdp.fingertip_vel,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=ALLEGRO_FINGERTIP_BODY_NAMES, preserve_order=False)
            },
        )
        # -- action
        last_action = ObsTerm(func=mdp.reorient_last_action, params={"action_name": "joint_pos"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg(ReorientEventCfg):
    """Shared randomization terms with the Direct task's reset distribution.

    The randomization terms are inherited and gated by
    :attr:`AllegroCubeEnvCfg.enable_domain_randomization`.
    """

    reset_object = None
    reset_robot_joints = None

    reset_state = EventTerm(
        func=mdp.reset_reorient_state,
        mode="reset",
        params={
            "position_noise": 0.01,
            "joint_position_noise": 0.2,
            "joint_velocity_noise": 0.0,
            "action_name": "joint_pos",
        },
    )


@configclass
class RewardsCfg(ReorientRewardsCfg):
    """Shared reward terms tuned to the Direct task's scales."""

    track_pos_l2 = RewTerm(
        func=mdp.track_pos_l2,
        weight=-10.0,
        params={"command_name": "object_pose", "object_cfg": SceneEntityCfg("object")},
    )
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.0002)
    joint_vel_l2 = None
    action_rate_l2 = None


@configclass
class TerminationsCfg(ReorientTerminationsCfg):
    """Shared terminations reduced to the Direct task's fall condition."""

    object_out_of_reach = DoneTerm(
        func=mdp.object_away_from_goal,
        params={
            "threshold": 0.24,
            "command_name": "object_pose",
            "object_cfg": SceneEntityCfg("object"),
        },
    )
    max_consecutive_success = None


@configclass
class AllegroCubeEnvCfg(ReorientObjectEnvCfg):
    """Manager-based Allegro Hand task with Direct-compatible semantics."""

    scene: AllegroCubeSceneCfg = AllegroCubeSceneCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    enable_domain_randomization: bool = True
    """Apply the shared startup domain-randomization terms.

    The Direct Allegro task has no randomization, so disable this to compare the two
    workflows directly. Changing it requires retraining.
    """

    _DOMAIN_RANDOMIZATION_TERMS = (
        "robot_physics_material",
        "robot_scale_mass",
        "robot_joint_stiffness_and_damping",
        "object_physics_material",
        "object_scale_mass",
    )

    def __post_init__(self):
        super().__post_init__()
        # mirrors the Direct cfg
        self.episode_length_s = 10.0
        self.sim.physics = PhysicsCfg()
        if not self.enable_domain_randomization:
            for term_name in self._DOMAIN_RANDOMIZATION_TERMS:
                setattr(self.events, term_name, None)

    def play_mode(self):
        # play-mode overrides of parent
        super().play_mode()

        # remove termination due to timeouts
        self.terminations.time_out = None
