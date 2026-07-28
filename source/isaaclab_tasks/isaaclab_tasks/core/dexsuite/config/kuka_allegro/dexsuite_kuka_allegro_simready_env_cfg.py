# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lift environment whose objects come from the SimReady catalogue.

The stock lift task already provides everything multi-object grasping needs: a multi-finger hand,
object point-cloud observations, contact-shaped rewards, a per-environment table, and a difficulty
curriculum. This configuration therefore changes exactly one thing -- which objects are spawned --
by replacing the primitive shapes with catalogue assets resolved when the configuration is built.

Objects are kept or rejected by what this robot can handle, never by rewriting the asset. Scale and
authored mass are left as published, so a can that is genuinely too heavy is simply not selected, or
is deliberately kept as one the hand will sometimes fail on. The one structural change applied to
each asset is documented on :meth:`~isaaclab.utils.simready.SimReadyObjectLibrary.prepare`.

The search phrases and bounds below describe *this* task: a table-top scene reachable by a Kuka arm
with an Allegro hand. They live here rather than in
:class:`~isaaclab.utils.simready.SimReadyObjectFilterCfg`, which carries no task defaults.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.utils.configclass import configclass
from isaaclab.utils.simready import SimReadyObjectFilterCfg, SimReadyObjectLibrary, SimReadyObjectLibraryCfg

from .dexsuite_kuka_allegro_env_cfg import DexsuiteKukaAllegroLiftEnvCfg

TABLE_TOP_PHRASES = (
    "food",
    "grocery",
    "can",
    "canned food",
    "box",
    "boxed food",
    "carton",
    "bottle",
    "jar",
    "cup",
    "fruit",
    "vegetable",
    "snack",
    "candy",
    "cereal",
    "spice",
    "tea box",
    "coffee",
    "milk",
    "juice",
    "package",
    "container",
    "bowl",
    "toy",
    "block",
    "soap",
    "medicine",
    "tube",
    "tin",
    "packet",
)
"""Phrases covering objects that belong on a table.

The catalogue index ranks by appearance rather than by geometry, so no single phrase returns a shape
class and coverage comes from the breadth of the phrasing instead.
"""

NON_TABLE_TOP_PATH_FRAGMENTS = (
    "Warehouse",
    "Machines",
    "Hardware",
    "engineComponent",
    "drivetrain",
    "airSystem",
    "coolingSystem",
    "electricalSystem",
    "fuelSystem",
    "brakeSystem",
    "exhaust",
    "transmission",
)
"""Catalogue areas holding no table-top manipulables (heavy machine parts, vehicle assemblies)."""


@configclass
class DexsuiteKukaAllegroLiftSimReadyEnvCfg(DexsuiteKukaAllegroLiftEnvCfg):
    """Lift task spawning distinct SimReady catalogue objects instead of primitive shapes."""

    num_objects: int = 100
    """Number of distinct objects to spawn across the environments."""

    object_library: SimReadyObjectLibraryCfg = SimReadyObjectLibraryCfg(
        object_filter=SimReadyObjectFilterCfg(
            search_phrases=TABLE_TOP_PHRASES,
            excluded_path_fragments=NON_TABLE_TOP_PATH_FRAGMENTS,
            # what the arm can reach around and the hand can close on
            size_range=(0.02, 0.15),
            # the upper bound is what the Allegro hand can hold; the catalogue's own spread across
            # this range is what decides how often lifting is easy
            mass_range=(0.005, 3.0),
        )
    )
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
