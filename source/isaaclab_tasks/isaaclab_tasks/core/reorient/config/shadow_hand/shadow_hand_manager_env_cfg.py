# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based counterpart of the state-based Shadow Hand reorientation task."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.simulation_cfg import SimulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.reorient.mdp as mdp
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_common import (
    GOAL_OBJECT_CFG,
    OBJECT_CFG,
    ROBOT_CFG,
    ObjectCfg,
    PhysicsCfg,
    ShadowHandManagerEventCfg,
)
from isaaclab_tasks.core.reorient.reorient_common import GOAL_MARKER_POSITION, IN_HAND_POS_OFFSET
from isaaclab_tasks.core.reorient.reorient_manager_env_cfg import ReorientObjectEnvCfg, ReorientObjectSceneCfg
from isaaclab_tasks.core.reorient.reorient_manager_env_cfg import RewardsCfg as ReorientRewardsCfg
from isaaclab_tasks.core.reorient.reorient_manager_env_cfg import TerminationsCfg as ReorientTerminationsCfg
from isaaclab_tasks.utils import PresetCfg, preset

from isaaclab_assets.robots.shadow_hand import SHADOW_ACTUATED_JOINT_NAMES, SHADOW_FINGERTIP_BODY_NAMES

# ---------------------------------- state task ----------------------------------


@configclass
class ShadowHandManagerSceneCfg(ReorientObjectSceneCfg):
    """Shared reorientation scene with the Shadow hand and a ground plane."""

    # ``clone_in_fabric`` is the only backend-varying field: PhysX/OvPhysX use Fabric
    # cloning for speed, Newton does not support it.
    clone_in_fabric = preset(default=False, physx=True, ovphysx=True, newton_mjwarp=False)

    num_envs = 8192
    env_spacing = 0.75

    robot: PresetCfg = ROBOT_CFG
    object: ObjectCfg = OBJECT_CFG
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )
    dome_light = None


@configclass
class CommandsCfg:
    """Object pose goal matching the Direct in-hand target."""

    object_pose = mdp.ReorientCommandCfg(
        asset_name="object",
        init_pos_offset=IN_HAND_POS_OFFSET,
        update_goal_on_success=True,
        orientation_success_threshold=0.1,
        make_quat_unique=False,
        fixed_marker_pos=GOAL_MARKER_POSITION,
        goal_pose_visualizer_cfg=GOAL_OBJECT_CFG,
        debug_vis=True,
    )


@configclass
class ActionsCfg:
    """Twenty actuated Shadow Hand joints."""

    joint_pos = mdp.ReorientEMAJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=SHADOW_ACTUATED_JOINT_NAMES,
        alpha=1.0,
        rescale_to_limits=True,
    )


@configclass
class FullStateObsCfg(ObsGroup):
    """Shared first 137 dimensions of the full Shadow state, before the action terms."""

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
        params={"asset_cfg": SceneEntityCfg("robot", body_names=SHADOW_FINGERTIP_BODY_NAMES, preserve_order=False)},
    )
    fingertip_quat = ObsTerm(
        func=mdp.fingertip_quat,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=SHADOW_FINGERTIP_BODY_NAMES, preserve_order=False)},
    )
    fingertip_vel = ObsTerm(
        func=mdp.fingertip_vel,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=SHADOW_FINGERTIP_BODY_NAMES, preserve_order=False)},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ObservationsCfg:
    """Full 157-dimensional state observation in Direct order."""

    @configclass
    class PolicyCfg(FullStateObsCfg):
        last_action = ObsTerm(func=mdp.reorient_last_action, params={"action_name": "joint_pos"})

    policy: PolicyCfg = PolicyCfg()


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

    max_consecutive_success = None
    object_out_of_reach = DoneTerm(
        func=mdp.object_away_from_goal,
        params={
            "threshold": 0.24,
            "command_name": "object_pose",
            "object_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class ShadowHandManagerEnvCfg(ReorientObjectEnvCfg):
    """Manager-based state Shadow Hand task with Direct-compatible semantics."""

    scene: ShadowHandManagerSceneCfg = ShadowHandManagerSceneCfg()
    decimation = 2
    episode_length_s = 10.0
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0),
        physics=PhysicsCfg(),
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: ShadowHandManagerEventCfg = ShadowHandManagerEventCfg()

    def __post_init__(self):
        """Apply the Shadow Hand control and episode rates."""

        super().__post_init__()
        self.decimation = 2
        self.episode_length_s = 10.0
        self.sim.render_interval = self.decimation
