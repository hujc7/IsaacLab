# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lift environment whose objects come from the SimReady catalogue.

The stock lift task already provides everything multi-object grasping needs: a multi-finger hand,
object point-cloud observations, contact-shaped rewards, a per-environment table, and a difficulty
curriculum. This configuration therefore changes one thing -- which objects are spawned -- by
replacing the primitive shapes with catalogue assets.

Which assets those are is recorded in ``simready_objects.json`` beside this file. That record is
committed, so every run spawns the same objects without contacting the search service. Delete it, or
set :attr:`~isaaclab.utils.simready.SimReadyObjectLibraryCfg.always_query`, to pick up whatever the
catalogue holds today.
"""

from __future__ import annotations

import os

from isaaclab.sim.spawners.wrappers import MultiUsdFileCfg
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
"""Words that steer the catalogue toward the sort of thing found on a household table.

A theme rather than a requirement: whether the hand can pick an object up is decided by its size and
mass, not by which word matched it. These need not be catalogue terms -- the index matches free text
by appearance -- so extend or replace them to change the flavour of the scene.
"""


@configclass
class DexsuiteKukaAllegroLiftSimReadyEnvCfg(DexsuiteKukaAllegroLiftEnvCfg):
    """Lift task spawning distinct SimReady catalogue objects instead of primitive shapes."""

    object_library: SimReadyObjectLibraryCfg = SimReadyObjectLibraryCfg(
        num_objects=100,  # an upper bound; the catalogue currently yields 99 under these filters
        resolution_path=os.path.join(os.path.dirname(__file__), "simready_objects.json"),
        object_filter=SimReadyObjectFilterCfg(
            search_phrases=TABLE_TOP_PHRASES,
            # a kitchen has no use for engine components, however liftable they are
            excluded_path_fragments=("Warehouse", "Machines", "Hardware"),
            size_range=(0.02, 0.15),  # what this arm can reach around
            mass_range=(0.005, 3.0),  # what this hand can hold
        ),
    )
    """Which catalogue objects this task spawns, and how they are resolved."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.object.spawn = MultiUsdFileCfg(
            usd_path=SimReadyObjectLibrary(self.object_library).resolve(),
            random_choice=False,  # one object per environment, so every asset is exercised
        )
        # distinct meshes cannot share a single physics replica, so each environment is built
        # independently (see the multi-asset spawning how-to guide)
        self.scene.replicate_physics = False
