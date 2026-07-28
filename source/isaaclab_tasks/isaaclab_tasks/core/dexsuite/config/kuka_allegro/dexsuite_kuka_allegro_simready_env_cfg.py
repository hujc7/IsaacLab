# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lift environments whose objects come from the SimReady catalogue.

The stock lift task already provides everything multi-object grasping needs: a multi-finger hand,
object point-cloud observations, contact-shaped rewards, a per-environment table, and a difficulty
curriculum. These configurations therefore change exactly one thing -- which objects are spawned --
by replacing the primitive shapes with catalogue assets resolved at construction time.

Objects are kept or rejected by what the robot can handle, never by rewriting the asset. Scale and
authored mass are left as published, so a can that is genuinely too heavy is simply not selected, or
is deliberately kept as one the hand will sometimes fail on. The one structural change applied to
each asset is documented on :meth:`~isaaclab.utils.simready.SimReadyObjectLibrary.prepare`.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.utils.configclass import configclass
from isaaclab.utils.simready import SimReadyObjectLibrary, SimReadyObjectLibraryCfg

from .dexsuite_kuka_allegro_env_cfg import DexsuiteKukaAllegroLiftEnvCfg, DexsuiteKukaAllegroLiftEnvCfg_PLAY


@configclass
class SimReadyObjectMixinCfg:
    """Replaces the task's primitive shapes with objects resolved from the SimReady catalogue."""

    num_objects: int = 100
    """Number of distinct objects to spawn across the environments."""

    object_library: SimReadyObjectLibraryCfg = SimReadyObjectLibraryCfg()
    """How the objects are searched for, filtered, and prepared."""

    def __post_init__(self):
        super().__post_init__()
        usd_paths = SimReadyObjectLibrary(self.object_library).resolve(self.num_objects)
        self.scene.object.spawn = sim_utils.MultiUsdFileCfg(
            usd_path=usd_paths,
            random_choice=False,  # one object per environment, so every asset is exercised
        )
        # distinct meshes cannot share a single physics replica, so each environment is built
        # independently (see the multi-asset spawning how-to guide)
        self.scene.replicate_physics = False


@configclass
class DexsuiteKukaAllegroLiftSimReadyEnvCfg(SimReadyObjectMixinCfg, DexsuiteKukaAllegroLiftEnvCfg):
    pass


@configclass
class DexsuiteKukaAllegroLiftSimReadyEnvCfg_PLAY(SimReadyObjectMixinCfg, DexsuiteKukaAllegroLiftEnvCfg_PLAY):
    num_objects: int = 12
    """Fewer objects, so a replay shows each one distinctly rather than repeating a few."""
