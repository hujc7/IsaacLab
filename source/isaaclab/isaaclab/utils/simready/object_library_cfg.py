# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for resolving SimReady catalogue assets into task-ready objects.

Filter fields are annotated with where they are applied:

* **[service]** -- the USD-Search service supports the filter natively, so it is pushed server-side
  and unmatched assets never enter the result budget.
* **[service, partial]** -- the service supports part of the filter. The supported part is pushed
  server-side as a pre-filter and the remainder is completed locally.
* **[local]** -- the service does not support the filter. It is applied after opening the asset,
  which is why :class:`SimReadyObjectLibrary` caches audits.
"""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils.assets import SIMREADY_SEARCH_SERVICE_ENDPOINT
from isaaclab.utils.configclass import configclass


@configclass
class SimReadyObjectFilterCfg:
    """Which catalogue objects to keep for a task.

    The fields fall into two groups. The first mirrors what the USD-Search service can filter on, so
    those are answered by the service. The second expresses what a manipulation task actually needs
    to know -- how heavy an object is, how thin its smallest axis is, whether its newest validation
    verdict still passes -- none of which the service returns, let alone filters on. Those are
    applied here, after opening the asset.

    Every bound expresses *robot capability*, never a judgement about the asset. Authored masses in
    particular are treated as ground truth -- a canned good genuinely is heavy -- so an object is
    dropped because the gripper could not hold it, not because the number looks surprising.
    """

    """
    Filters the search service applies.
    """

    search_phrases: tuple[str, ...] = MISSING
    """**[service]** Phrases whose union forms the candidate pool, one query each.

    The index ranks by appearance rather than by geometry, so no single phrase returns a shape class
    and coverage comes from the breadth of the phrasing instead.
    """

    required_features: tuple[str, ...] = ("FET003_BASE_PHYSX",)
    """**[service]** SimReady features every match must carry.

    ``FET003_BASE_PHYSX`` selects single-body PhysX-ready assets. ``FET004_BASE_PHYSX`` is the
    *multibody* feature and so is the wrong choice for a task needing one rigid body per object.
    """

    required_profiles: tuple[str, ...] = ()
    """**[service]** SimReady profiles every match must carry (e.g. ``"Prop-Robotics-Isaac"``)."""

    required_classes: tuple[str, ...] = ()
    """**[service]** Semantic classes every match must carry."""

    required_tags: tuple[str, ...] = ()
    """**[service]** Catalogue tags every match must carry."""

    required_countries: tuple[str, ...] = ()
    """**[service]** Countries whose regional variants to keep (e.g. packaging localisation)."""

    required_scene_poi_tags: tuple[str, ...] = ()
    """**[service]** Scene points-of-interest every match must be annotated with."""

    required_metadata: tuple[tuple[tuple[str, ...], str], ...] = ()
    """**[service]** Arbitrary metadata entries, each a ``(key path, exact value)`` pair.

    The service compares the value exactly, so this cannot express ranges or negation.
    """

    excluded_path_fragments: tuple[str, ...] = ()
    """**[service]** Path substrings that disqualify a match.

    Excluding at the service keeps unusable assets out of the per-phrase result budget instead of
    spending it on results that are discarded locally.
    """

    base_paths: tuple[str, ...] = ()
    """**[service]** Catalogue sub-trees to search. Empty searches the whole catalogue."""

    min_relevance: float = 0.0
    """**[service]** Minimum relevance score for a match to be considered."""

    """
    Filters completed after opening the asset.
    """

    size_range: tuple[float, float] = MISSING
    """**[service, partial]** Accepted bounding-box extents [m], as ``(minimum, maximum)``.

    The smallest extent must exceed the lower bound, since a thin object gives the fingers nothing to
    close on, and the largest must stay under the upper bound to fit the workspace.

    The service filters on height alone, so it is queried with this range as a pre-filter and the
    remaining two axes are checked locally. An object that is short but very wide therefore passes
    the service and is rejected here.
    """

    mass_range: tuple[float, float] = MISSING
    """**[local]** Accepted authored mass [kg], as ``(minimum, maximum)``.

    The upper bound is what the gripper can hold; the lower bound excludes objects so light that
    contact is numerically unstable. The service neither returns nor filters on mass.
    """

    heavy_from: float = float("inf")
    """**[local]** Mass [kg] at which an object counts as being at the edge of the robot's capability.

    Only read when :attr:`heavy_fraction` is non-zero.
    """

    heavy_fraction: float = 0.0
    """**[local]** Share of the selection reserved for objects at or above :attr:`heavy_from`.

    A set in which every object lifts on the first try teaches the policy that lifting always works,
    so reserving a slice for objects that may not come up keeps that signal honest. Left at zero no
    slice is reserved, since how much of the set should be unliftable is a property of the task.
    """

    require_rigid_body: bool = True
    """**[local]** Whether to require a dynamic rigid body.

    Some published assets carry colliders but nothing dynamic, so they are not simulatable as
    objects. Presence of a body is not exposed by the service.
    """

    require_latest_validation: bool = True
    """**[local]** Whether the newest-dated verdict for :attr:`required_features` must be a pass.

    This closes a real gap rather than duplicating the service filter: the service matches assets
    that passed on *some* date, so one that later regressed still comes back as a hit. Re-reading the
    dated verdicts from the asset is the authoritative check.
    """

    distinct_families: bool = True
    """**[local]** Whether to keep at most one asset per product family.

    The catalogue ships many near-identical variants (``Golf_Ball`` and ``Golf_Ball_A01`` through
    ``A04``; ``Boxed_Drink_A01`` through ``E01``), so a set of unique *files* can still render as a
    grid of look-alikes -- file uniqueness is not visual variety.
    """


@configclass
class SimReadyObjectLibraryCfg:
    """Configuration for :class:`~isaaclab.utils.simready.SimReadyObjectLibrary`."""

    service_endpoint: str = SIMREADY_SEARCH_SERVICE_ENDPOINT
    """URL of the USD-Search service."""

    results_per_phrase: int = 100
    """Maximum number of matches to request per search phrase."""

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
    """Which catalogue objects to keep."""
