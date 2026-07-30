# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for resolving SimReady catalogue assets into task-ready objects."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils.assets import SIMREADY_SEARCH_SERVICE_ENDPOINT
from isaaclab.utils.configclass import configclass


@configclass
class SimReadyObjectFilterCfg:
    """Which catalogue assets to keep.

    Every field names a property of the asset. Leaving a field unset does not constrain that
    property. Where a property comes from -- the search index, or the asset itself -- is an
    implementation detail: results are cached, so it is paid once.

    Authored masses and sizes are treated as real data. An object is dropped because this robot
    could not handle it, never because the asset looks wrong.
    """

    search_phrases: tuple[str, ...] = ()
    """Free-text descriptions of the objects wanted, for example ``("mug", "cereal box")``.

    This chooses a domain rather than a capability. The index matches on appearance, so coverage
    comes from the breadth of the phrasing, and the words need not be catalogue terms. Left unset,
    objects of every kind are considered as long as they satisfy the remaining fields.
    """

    excluded_path_fragments: tuple[str, ...] = ()
    """Catalogue sub-trees to skip, for example ``("Warehouse", "Machines")``.

    Useful to keep a domain coherent: a kitchen scene has no use for engine components, even ones
    small and light enough to pick up.
    """

    size_range: tuple[float, float] | None = None
    """Accepted bounding-box extents [m] as ``(minimum, maximum)``, applied to every axis.

    The smallest extent must exceed the minimum, since a thin object gives the fingers nothing to
    close on, and the largest must stay under the maximum to fit the workspace.
    """

    mass_range: tuple[float, float] | None = None
    """Accepted authored mass [kg] as ``(minimum, maximum)``.

    The upper bound is what the gripper can hold; the lower bound excludes objects so light that
    contact is numerically unstable.
    """

    validated_features: tuple[str, ...] = ("FET003_BASE_PHYSX",)
    """Validation features the asset must currently pass.

    ``FET003_BASE_PHYSX`` certifies a single rigid body ready for PhysX. ``FET004_BASE_PHYSX`` is
    the multibody equivalent, and the wrong choice for a task that needs one rigid body per object.

    The verdict is re-read from the asset rather than taken from the search index, because the index
    matches assets that passed on *some* date: one that later regressed still comes back as a hit.
    """

    require_rigid_body: bool = True
    """Whether an asset must contain a body that physics can move.

    Some catalogue assets carry collision geometry but no rigid body, so they can be collided with
    yet never fall, get pushed, or be picked up -- scenery rather than props. They are unusable as
    task objects and are a large share of the catalogue: 61 of 210 candidates in one table-top sweep.
    """

    max_per_product_family: int | None = None
    """How many variants of one product to keep, or ``None`` to keep them all.

    The catalogue ships near-identical variants (``Golf_Ball``, ``Golf_Ball_A01`` through ``A04``),
    so a set of unique files can still look like one object repeated. Capping is worth it when the
    objects are seen together -- a demo grid, a recorded video -- and costs object count when they
    are not: one table-top sweep yields 99 assets uncapped and 39 at a cap of one.

    Families are identified by stripping trailing variant codes from the asset name, which is a
    naming convention rather than a published property.
    """


@configclass
class SimReadyObjectLibraryCfg:
    """Configuration for :class:`~isaaclab.utils.simready.SimReadyObjectLibrary`."""

    num_objects: int = MISSING
    """How many distinct objects to resolve."""

    object_filter: SimReadyObjectFilterCfg = SimReadyObjectFilterCfg()
    """Which catalogue assets to keep."""

    resolution_path: str = MISSING
    """JSON file recording which assets this configuration resolved to.

    A given configuration resolves to a fixed set of assets, so this file is the reproducible record
    of that set. Commit it next to the task configuration and everyone who runs the task fetches
    exactly those assets: no search request, no credentials, and no chance of two machines -- or two
    ranks of one job -- resolving differently because the catalogue moved between queries.

    Delete the file, or set :attr:`always_query`, to pick up the catalogue's current answer.
    """

    always_query: bool = False
    """Whether to re-query the service even when :attr:`resolution_path` already holds an answer."""

    service_endpoint: str = SIMREADY_SEARCH_SERVICE_ENDPOINT
    """URL of the USD-Search service."""

    cache_dir: str = "logs/simready"
    """Directory for downloaded assets, their measurements, and the prepared USDs.

    Everything here is derived and can be deleted; only :attr:`resolution_path` is worth keeping.
    Measuring an asset means fetching its whole layer closure, so the first resolve of a catalogue
    costs minutes and every later one is instant.
    """

    audit_workers: int = 16
    """How many assets to measure concurrently.

    Measuring is bound by per-file network round-trips rather than bandwidth or CPU, so doing
    several at once scales close to linearly. Lower this if the asset host objects.
    """

    layer_workers: int = 8
    """How many of one asset's layers to fetch concurrently.

    An asset is a tree of layers -- wrapper, payloads, physics, materials, textures -- each costing
    a separate request. A layer's own references are only known once it has been read, so the tree
    is walked level by level, but layers within a level are independent.
    """

    results_per_phrase: int = 100
    """Maximum number of matches to request per search phrase."""
