# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for resolving SimReady catalogue assets into task-ready objects."""

from __future__ import annotations

from isaaclab.utils.assets import SIMREADY_SEARCH_SERVICE_ENDPOINT
from isaaclab.utils.configclass import configclass

DEFAULT_TABLE_TOP_PHRASES = (
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
"""Search phrases covering table-top manipulables.

The index ranks by appearance rather than by geometry, so no single phrase returns a shape class and
coverage comes from the breadth of the phrasing instead.
"""

DEFAULT_EXCLUDED_PATH_FRAGMENTS = (
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
"""Catalogue areas that hold no table-top manipulables (heavy machine parts, vehicle assemblies)."""


@configclass
class SimReadyObjectFilterCfg:
    """Which catalogue objects a given robot can work with.

    Every bound expresses *robot capability*, never a judgement about the asset. Authored masses in
    particular are treated as ground truth -- a canned good genuinely is heavy -- so an object is
    dropped because the gripper could not hold it, not because the number looks surprising.
    """

    size_range: tuple[float, float] = (0.02, 0.15)
    """Accepted bounding-box extents [m], as ``(minimum, maximum)``.

    The smallest extent must exceed the lower bound, since a thin object gives the fingers nothing to
    close on, and the largest must stay under the upper bound to fit the workspace.
    """

    mass_range: tuple[float, float] = (0.005, 3.0)
    """Accepted authored mass [kg], as ``(minimum, maximum)``.

    The upper bound is what the gripper can hold; the lower bound excludes objects so light that
    contact is numerically unstable.
    """

    heavy_from: float = 1.5
    """Mass [kg] at which an object counts as being at the edge of the robot's capability."""

    heavy_fraction: float = 0.10
    """Share of the selection reserved for objects at or above :attr:`heavy_from`.

    A set in which every object lifts on the first try teaches the policy that lifting always works.
    Reserving a slice for objects that may not come up keeps that signal honest.
    """

    distinct_families: bool = True
    """Whether to keep at most one asset per product family.

    The catalogue ships many near-identical variants (``Golf_Ball`` and ``Golf_Ball_A01`` through
    ``A04``; ``Boxed_Drink_A01`` through ``E01``), so a set of unique *files* can still render as a
    grid of look-alikes -- file uniqueness is not visual variety.
    """

    require_latest_validation: bool = True
    """Whether to require the newest-dated ``FET003_BASE_PHYSX`` verdict to be a pass.

    The service matches assets that passed on *some* date, so one that later regressed still comes
    back as a hit. Re-reading the verdict from the asset is the authoritative check.
    """


@configclass
class SimReadyObjectLibraryCfg:
    """Configuration for :class:`~isaaclab.utils.simready.SimReadyObjectLibrary`."""

    service_endpoint: str = SIMREADY_SEARCH_SERVICE_ENDPOINT
    """URL of the USD-Search service."""

    search_phrases: tuple[str, ...] = DEFAULT_TABLE_TOP_PHRASES
    """Phrases whose union forms the candidate pool."""

    excluded_path_fragments: tuple[str, ...] = DEFAULT_EXCLUDED_PATH_FRAGMENTS
    """Path substrings that disqualify a match, applied server-side.

    Excluding at the service keeps unusable assets out of the :attr:`results_per_phrase` budget
    instead of spending it on results that are discarded locally.
    """

    results_per_phrase: int = 100
    """Maximum number of matches to request per phrase."""

    cache_path: str = "logs/simready/audit_cache.json"
    """JSON file holding audit results, so each asset is opened at most once, ever.

    Opening an asset means fetching it and its whole layer closure from remote storage, which is the
    expensive step. Caching is what makes resolution affordable inside a training script: the first
    sweep of a catalogue costs minutes, every later one is instant.
    """

    download_dir: str = "logs/simready/downloads"
    """Directory mirroring fetched assets, laid out to match their remote paths."""

    prepared_dir: str = "logs/simready/prepared"
    """Directory receiving the task-ready USDs written by
    :meth:`~isaaclab.utils.simready.SimReadyObjectLibrary.prepare`."""

    object_filter: SimReadyObjectFilterCfg = SimReadyObjectFilterCfg()
    """Which of the found objects the robot can actually work with."""
