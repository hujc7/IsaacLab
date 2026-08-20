# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.core.handover.handover_common import GOAL_MARKER_CFG, OBJECT_RADIUS
from isaaclab_tasks.utils import PresetCfg, preset

from isaaclab_assets.robots.shadow_hand import (
    ShadowHand,
)


def _shadow_hand_cfg(
    prim_path: str,
    init_pos: tuple[float, float, float],
    init_rot: tuple[float, float, float, float],
) -> PresetCfg:
    """Build the per-hand Shadow Hand preset for each supported backend.

    Every engine gets the same hand at the same pose with the same gains; only the asset's
    ``Physics`` USD variant differs, and that is carried by the asset configuration itself. The
    catch needs more joint authority than reorientation, so the motor gains are raised here for
    both hands alike.

    Args:
        prim_path: Scene path the hand spawns at.
        init_pos: Spawn position [m].
        init_rot: Spawn orientation as ``(w, x, y, z)``.

    Returns:
        A preset carrying each engine's variant, all at *prim_path* with the given pose. The two
        hands differ only in these arguments.
    """

    def _for_engine(physics: str):
        # Each engine's configuration has to come from ShadowHand.cfg, not from the other engine's
        # with the variant string swapped: the physx variant carries no tendons in the asset and
        # relies on a spawn function to author them, which a variant swap would drop.
        base = ShadowHand.cfg(physics)
        # The asset's own spawn rotation is shared by both engines, so the per-hand rotation
        # COMPOSES with it rather than replacing it -- replacing leaves both palms turned 90
        # degrees. See ShadowHand.cfg's init_state for why the asset carries that rotation.
        hand_rot = tuple(
            math_utils.quat_mul(
                torch.tensor(init_rot, dtype=torch.float64),
                torch.tensor(base.init_state.rot, dtype=torch.float64),
            ).tolist()
        )
        # The hand's gains belong to the hand, so both tasks take them as the asset configuration
        # supplies them. This task used to raise every actuator to stiffness 20 / damping 2, which
        # also drove the tendon-coupled joints -- they take no position command, and MEASURED, giving
        # them one costs the tendon most of its travel: 11.1 rad falls to 1.0 rad.
        return base.replace(
            prim_path=prim_path,
            init_state=base.init_state.replace(pos=init_pos, rot=hand_rot),
        )

    hand_cfg = _for_engine("mujoco")
    physx_cfg = _for_engine("physx")
    return preset(
        default=hand_cfg,
        physx=physx_cfg,
        isaacsim_physx=physx_cfg,
        newton_mjwarp=hand_cfg,
        ovphysx=physx_cfg,
    )


# Per-hand presets shared by the Direct environment and the manager scene. These are the per-hand
# rotations, composed above with the asset's own; they are unchanged from the previous Newton asset,
# which the two assets being identical geometry makes valid.
RIGHT_HAND_CFG = _shadow_hand_cfg(
    prim_path="{ENV_REGEX_NS}/RightRobot",
    init_pos=(0.0, 0.0, 0.5),
    init_rot=(0.0, 0.0, 0.0, 1.0),
)
LEFT_HAND_CFG = _shadow_hand_cfg(
    prim_path="{ENV_REGEX_NS}/LeftRobot",
    init_pos=(0.0, -1.0, 0.5),
    init_rot=(0.0, 0.0, 1.0, 0.0),
)


BALL_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/object",
    spawn=sim_utils.SphereCfg(
        radius=OBJECT_RADIUS,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 1.0, 0.0)),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.7),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=False,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0025,
            max_depenetration_velocity=1000.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(density=500.0),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.39, 0.54), rot=(0.0, 0.0, 0.0, 1.0)),
)
"""Hand-over ball, thrown from one Shadow hand to the other."""


@configclass
class PhysicsCfg(PresetCfg):
    """Physics-backend preset (PhysX vs Newton/MJWarp).

    Newton mirrors the single-agent Shadow Hand Newton port: an elliptic friction
    cone with ``impratio=10``, which weights normal contacts over friction, 100
    solver iterations and 2 substeps.
    """

    isaacsim_physx = PhysxCfg(
        bounce_threshold_velocity=0.2,
        gpu_max_rigid_contact_count=2**23,
        gpu_max_rigid_patch_count=2**23,
    )
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=200,
            nconmax=70,
            impratio=10.0,
            cone="elliptic",
            update_data_interval=4,
            ccd_iterations=50,  # bumped from default 35 for multi-finger contact geometry
        ),
        # 4 substeps (vs reorient's 2): sustained ball-palm contact drives a small fraction of
        # envs to NaN at 2.
        num_substeps=4,
        debug_mode=False,
    )
    ovphysx = OvPhysxCfg()
    physx = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    default = newton_mjwarp


@configclass
class HandoverEnvCfg(DirectMARLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 7.5
    possible_agents = ["right_hand", "left_hand"]
    action_spaces = {"right_hand": 20, "left_hand": 20}
    observation_spaces = {"right_hand": 157, "left_hand": 157}
    state_space = 290

    # simulation — values mirrored by the manager cfg
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0),
        physics=PhysicsCfg(),
        # Frame both hands and the object between them. Without this the visualizer looks at the
        # origin from its default 4 m away, which renders the pair a few pixels wide.
        default_visualizer_cfg=VisualizerCfg(eye=(1.15, -1.65, 1.15), lookat=(0.0, -0.5, 0.55), focal_length=35.0),
    )

    # robot
    right_robot_cfg: PresetCfg = RIGHT_HAND_CFG
    left_robot_cfg: PresetCfg = LEFT_HAND_CFG
    actuated_joint_names = ShadowHand.joint_names
    actuated_tendon_names = ShadowHand.tendon_names
    actuated_tendon_position_limits = ShadowHand.tendon_position_limits
    fingertip_body_names = ShadowHand.fingertip_names

    # in-hand object
    object_cfg: RigidObjectCfg = BALL_CFG
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = GOAL_MARKER_CFG
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=1.5, replicate_physics=True)

    # reset
    reset_position_noise = 0.01  # range of position at reset
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset
    # scales and constants
    fall_dist = 0.24
    vel_obs_scale = 0.2
    act_moving_average = 1.0
    # success criteria
    success_distance_threshold: float = 0.1
    """Object-to-goal distance below which the handover is considered successful [m]."""
    # reward-related scales
    dist_reward_scale = 20.0
    action_penalty_scale = -0.0002
    """Squared-action reward scale, matching the reorientation task.

    Each hand pays for its own actions. The distance term is shared, so once a hand releases the ball
    its pose stops affecting the reward and nothing else discourages it from saturating its motors.
    """
