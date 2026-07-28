# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for capability-based SimReady object selection.

Selection runs entirely off cached audits, so these tests need neither the simulation app nor the
search service: the cache is pre-populated and every asset resolves from it.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from isaaclab.utils.simready import ObjectSpec, SimReadyObjectLibrary, SimReadyObjectLibraryCfg
from isaaclab.utils.simready.simready import _rejection_reason


def make_spec(url: str, mass: float | None = 0.2, dims: tuple[float, float, float] = (0.07, 0.07, 0.07), **kwargs):
    """Build a spec that passes every default filter unless an argument says otherwise."""
    return ObjectSpec(
        url=url,
        body_path=kwargs.get("body_path", "/RootNode/Geometry/body"),
        mass=mass,
        dims=dims,
        validation_passed=kwargs.get("validation_passed", True),
    )


def make_library(tmp_path, specs: list[ObjectSpec]) -> SimReadyObjectLibrary:
    """Build a library whose audit cache already holds ``specs``, so no asset is ever opened."""
    cache_path = tmp_path / "audit_cache.json"
    cache_path.write_text(json.dumps({spec.url: asdict(spec) for spec in specs}))
    cfg = SimReadyObjectLibraryCfg()
    cfg.cache_path = str(cache_path)
    cfg.download_dir = str(tmp_path / "downloads")
    cfg.prepared_dir = str(tmp_path / "prepared")
    return SimReadyObjectLibrary(cfg)


class TestRejectionReason:
    """An asset is rejected for what this robot can handle, never for how the asset was authored."""

    def test_asset_within_every_bound_is_accepted(self):
        assert _rejection_reason(make_spec("a/Apple/a.usd"), SimReadyObjectLibraryCfg().object_filter) is None

    @pytest.mark.parametrize(
        "spec_kwargs, expected",
        [
            ({"mass": 9.0}, "heavier than the gripper can hold"),
            ({"mass": 1e-4}, "too light for stable contact"),
            ({"mass": None}, "no authored mass"),
            ({"dims": (0.4, 0.1, 0.1)}, "too large for the workspace"),
            ({"dims": (0.1, 0.1, 1e-3)}, "too thin for stable contact"),
            ({"body_path": None}, "no rigid body"),
            ({"validation_passed": False}, "latest FET003_BASE_PHYSX validation failed"),
        ],
    )
    def test_unusable_asset_is_rejected_with_a_reason(self, spec_kwargs, expected):
        reason = _rejection_reason(make_spec("a/Thing/a.usd", **spec_kwargs), SimReadyObjectLibraryCfg().object_filter)
        assert reason is not None and expected in reason

    def test_asset_that_fails_to_open_is_rejected(self):
        assert _rejection_reason(None, SimReadyObjectLibraryCfg().object_filter) == "could not be opened"

    def test_stale_validation_is_accepted_when_the_check_is_disabled(self):
        object_filter = SimReadyObjectLibraryCfg().object_filter
        object_filter.require_latest_validation = False
        assert _rejection_reason(make_spec("a/Milk/a.usd", validation_passed=False), object_filter) is None

    def test_widening_the_mass_range_accepts_a_heavier_asset(self):
        """The bound is a statement about the gripper, so a stronger gripper keeps more objects."""
        object_filter = SimReadyObjectLibraryCfg().object_filter
        assert _rejection_reason(make_spec("a/Can/a.usd", mass=5.0), object_filter) is not None
        object_filter.mass_range = (0.005, 6.0)
        assert _rejection_reason(make_spec("a/Can/a.usd", mass=5.0), object_filter) is None


class TestObjectSpec:
    """Product families group variants that differ only by a trailing code."""

    @pytest.mark.parametrize(
        "url, expected",
        [
            ("root/Golf_Ball/asset.usd", "golfball"),
            ("root/Golf_Ball_A03/asset.usd", "golfball"),
            ("root/Boxed_Drink_E01/asset.usd", "boxeddrink"),
            ("root/Tomato_Soup_Can/asset.usd", "tomatosoupcan"),
        ],
    )
    def test_family_strips_the_variant_code(self, url, expected):
        assert make_spec(url).family == expected

    def test_extents_report_the_largest_and_smallest_axis(self):
        spec = make_spec("a/Book/a.usd", dims=(0.2, 0.03, 0.14))
        assert spec.max_dim == pytest.approx(0.2)
        assert spec.min_dim == pytest.approx(0.03)


class TestSelect:
    """Selection audits candidates, drops what the robot cannot handle, and stratifies by mass."""

    def test_select_keeps_only_usable_candidates(self, tmp_path):
        specs = [make_spec("a/Apple/a.usd"), make_spec("a/Anvil/a.usd", mass=9.0), make_spec("a/Pear/a.usd")]
        library = make_library(tmp_path, specs)

        selected = library.select(num_objects=10, candidates=[spec.url for spec in specs])

        assert sorted(spec.url for spec in selected) == ["a/Apple/a.usd", "a/Pear/a.usd"]

    def test_select_reserves_a_share_for_objects_at_the_edge_of_the_gripper(self, tmp_path):
        """A set where everything lifts teaches the policy that lifting always works."""
        light = [make_spec(f"a/Light{i}/a.usd", mass=0.2) for i in range(20)]
        heavy = [make_spec(f"a/Heavy{i}/a.usd", mass=2.0) for i in range(20)]
        library = make_library(tmp_path, light + heavy)

        selected = library.select(num_objects=10, candidates=[spec.url for spec in light + heavy])

        assert len(selected) == 10
        assert sum(spec.mass >= library.cfg.object_filter.heavy_from for spec in selected) == 1

    def test_select_keeps_one_asset_per_family(self, tmp_path):
        """Unique files are not visual variety: five golf balls still render as one object."""
        specs = [make_spec(f"a/Golf_Ball_A0{i}/a.usd") for i in range(5)] + [make_spec("a/Pear/a.usd")]
        library = make_library(tmp_path, specs)

        selected = library.select(num_objects=10, candidates=[spec.url for spec in specs])

        assert sorted(spec.family for spec in selected) == ["golfball", "pear"]

    def test_select_keeps_every_variant_when_family_grouping_is_disabled(self, tmp_path):
        specs = [make_spec(f"a/Golf_Ball_A0{i}/a.usd") for i in range(5)]
        library = make_library(tmp_path, specs)
        library.cfg.object_filter.distinct_families = False

        selected = library.select(num_objects=10, candidates=[spec.url for spec in specs])

        assert len(selected) == 5


class TestAuditCache:
    """Audits survive a round trip, so a later resolve opens no assets."""

    def test_cached_audit_restores_the_bounding_box_as_a_tuple(self, tmp_path):
        """JSON has no tuple, so a round trip must not silently turn the extents into a list."""
        spec = make_spec("a/Apple/a.usd", dims=(0.07, 0.08, 0.09))
        library = make_library(tmp_path, [spec])

        restored = library.audit(spec.url)

        assert restored.dims == (0.07, 0.08, 0.09)
        assert restored.max_dim == pytest.approx(0.09)

    def test_an_asset_that_failed_to_open_is_not_retried(self, tmp_path):
        """Caching the failure is what keeps a broken asset from being fetched on every resolve."""
        library = make_library(tmp_path, [])
        library._cache["a/Broken/a.usd"] = None

        assert library.audit("a/Broken/a.usd") is None

    def test_entry_from_an_older_field_layout_is_discarded(self, tmp_path):
        """A cache is regenerable, so a layout change must re-audit rather than raise."""
        cache_path = tmp_path / "audit_cache.json"
        cache_path.write_text(
            json.dumps(
                {
                    "a/Apple/a.usd": {
                        "url": "a/Apple/a.usd",
                        "body_path": "/B",
                        "mass": 0.2,
                        "dims": [0.07, 0.07, 0.07],
                        "fet003_passed": True,  # the field this attribute used to be called
                    }
                }
            )
        )
        cfg = SimReadyObjectLibraryCfg()
        cfg.cache_path = str(cache_path)
        cfg.download_dir = str(tmp_path / "downloads")
        library = SimReadyObjectLibrary(cfg)

        # the stale entry is dropped, so the asset is re-audited; it is unreachable here, hence None
        assert library.audit("a/Apple/a.usd") is None

    def test_saved_cache_is_reused_by_a_later_library(self, tmp_path):
        spec = make_spec("a/Apple/a.usd")
        make_library(tmp_path, [spec]).save_cache()

        cfg = SimReadyObjectLibraryCfg()
        cfg.cache_path = str(tmp_path / "audit_cache.json")
        assert SimReadyObjectLibrary(cfg).audit(spec.url).url == spec.url
