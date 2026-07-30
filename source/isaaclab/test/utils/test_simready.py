# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for resolving SimReady catalogue assets into task-ready objects.

Selection runs off cached measurements, so these tests need neither the simulation app nor the
search service: the cache is pre-populated and every asset resolves from it.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from isaaclab.utils.simready import (
    ObjectSpec,
    SimReadyObjectFilterCfg,
    SimReadyObjectLibrary,
    SimReadyObjectLibraryCfg,
)
from isaaclab.utils.simready.object_library import _body_mass, _rejection_reason


def make_spec(url: str, mass: float | None = 0.2, dims: tuple[float, float, float] = (0.07, 0.07, 0.07), **kwargs):
    """Build a spec that satisfies every property in :func:`make_filter` unless told otherwise."""
    return ObjectSpec(
        url=url,
        body_path=kwargs.get("body_path", "/RootNode/Geometry/body"),
        mass=mass,
        dims=dims,
        validation_passed=kwargs.get("validation_passed", True),
    )


def make_filter(**kwargs) -> SimReadyObjectFilterCfg:
    """Build a filter constraining size and mass, the properties most tests exercise."""
    kwargs.setdefault("size_range", (0.02, 0.15))
    kwargs.setdefault("mass_range", (0.005, 3.0))
    return SimReadyObjectFilterCfg(**kwargs)


def make_library(tmp_path, specs: list[ObjectSpec], num_objects: int = 10, **filter_kwargs):
    """Build a library whose measurements are already cached, so no asset is ever opened."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(exist_ok=True)
    (cache_dir / "measurements.json").write_text(json.dumps({spec.url: asdict(spec) for spec in specs}))
    cfg = SimReadyObjectLibraryCfg(
        num_objects=num_objects,
        object_filter=make_filter(**filter_kwargs),
        resolution_path=str(tmp_path / "objects.json"),
        cache_dir=str(cache_dir),
    )
    library = SimReadyObjectLibrary(cfg)
    library.prepare = lambda selected: [spec.url for spec in selected]
    return library


class TestRejectionReason:
    """An asset is rejected for a property this robot cannot work with, never for how it was made."""

    def test_asset_satisfying_every_property_is_accepted(self):
        assert _rejection_reason(make_spec("a/Apple/a.usd"), make_filter()) is None

    @pytest.mark.parametrize(
        "spec_kwargs, expected",
        [
            ({"mass": 9.0}, "heavier than the gripper can hold"),
            ({"mass": 1e-4}, "too light for stable contact"),
            ({"mass": None}, "no authored mass"),
            ({"dims": (0.4, 0.1, 0.1)}, "too large for the workspace"),
            ({"dims": (0.1, 0.1, 1e-3)}, "too thin for stable contact"),
            ({"body_path": None}, "no rigid body"),
            ({"validation_passed": False}, "newest-dated validation verdict is a failure"),
        ],
    )
    def test_unusable_asset_is_rejected_with_a_reason(self, spec_kwargs, expected):
        reason = _rejection_reason(make_spec("a/Thing/a.usd", **spec_kwargs), make_filter())
        assert reason is not None and expected in reason

    def test_asset_that_fails_to_open_is_rejected(self):
        assert _rejection_reason(None, make_filter()) == "could not be opened"

    def test_an_unset_property_is_not_constrained(self):
        """Leaving a field unset must not filter on it, however extreme the value."""
        unconstrained = SimReadyObjectFilterCfg()
        assert _rejection_reason(make_spec("a/Anvil/a.usd", mass=500.0, dims=(2.0, 2.0, 2.0)), unconstrained) is None

    def test_widening_the_mass_range_accepts_a_heavier_asset(self):
        """The bound describes the gripper, so a stronger one keeps more objects."""
        assert _rejection_reason(make_spec("a/Can/a.usd", mass=5.0), make_filter()) is not None
        assert _rejection_reason(make_spec("a/Can/a.usd", mass=5.0), make_filter(mass_range=(0.005, 6.0))) is None


class TestBodyMass:
    """Mass composes down the prim tree, so it is aggregated rather than read from one attribute."""

    BODY = "/RootNode/Geometry/obj_00"

    def test_mass_on_the_body_is_used_as_is(self):
        assert _body_mass(self.BODY, [(self.BODY, 0.42)]) == pytest.approx(0.42)

    def test_masses_on_descendants_are_summed(self):
        """A coffee cup declares its body and lid separately; the object weighs both."""
        authored = [(f"{self.BODY}/body_mesh", 0.0273), (f"{self.BODY}/lid_mesh", 0.0398)]
        assert _body_mass(self.BODY, authored) == pytest.approx(0.0671)

    def test_a_value_on_the_body_wins_over_its_descendants(self):
        """Assets that state the same mass on body and mesh must not be counted twice."""
        authored = [(self.BODY, 0.1165), (f"{self.BODY}/body_mesh", 0.1165)]
        assert _body_mass(self.BODY, authored) == pytest.approx(0.1165)

    def test_masses_outside_the_body_are_ignored(self):
        authored = [(f"{self.BODY}/mesh", 0.5), ("/RootNode/Other/mesh", 9.0)]
        assert _body_mass(self.BODY, authored) == pytest.approx(0.5)

    def test_no_authored_mass_is_reported_as_unknown(self):
        assert _body_mass(self.BODY, []) is None
        assert _body_mass(self.BODY, [("/RootNode/Elsewhere", 1.0)]) is None

    def test_mass_without_a_body_still_reports_a_total(self):
        assert _body_mass(None, [("/a", 0.2), ("/b", 0.3)]) == pytest.approx(0.5)


class TestObjectSpec:
    """The measured spec exposes the properties the filter compares against."""

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
    """Selection measures the candidates and keeps those satisfying every configured property."""

    def test_select_keeps_only_usable_candidates(self, tmp_path):
        specs = [make_spec("a/Apple/a.usd"), make_spec("a/Anvil/a.usd", mass=9.0), make_spec("a/Pear/a.usd")]
        library = make_library(tmp_path, specs)

        selected = library.select(candidates=[spec.url for spec in specs])

        assert sorted(spec.url for spec in selected) == ["a/Apple/a.usd", "a/Pear/a.usd"]

    def test_a_product_family_cap_limits_near_identical_variants(self, tmp_path):
        """Unique files are not visual variety: five golf balls still look like one object."""
        specs = [make_spec(f"a/Golf_Ball_A0{i}/a.usd") for i in range(5)] + [make_spec("a/Pear/a.usd")]
        library = make_library(tmp_path, specs, max_per_product_family=1)

        selected = library.select(candidates=[spec.url for spec in specs])

        assert sorted(spec.family for spec in selected) == ["golfball", "pear"]

    def test_every_variant_is_kept_when_no_cap_is_set(self, tmp_path):
        """The task leaves the cap unset, so all variants count toward the object budget."""
        specs = [make_spec(f"a/Golf_Ball_A0{i}/a.usd") for i in range(5)]
        library = make_library(tmp_path, specs)

        assert len(library.select(candidates=[spec.url for spec in specs])) == 5

    def test_select_returns_at_most_the_requested_count(self, tmp_path):
        specs = [make_spec(f"a/O{i}/a.usd", mass=0.1 + i * 0.05) for i in range(8)]
        library = make_library(tmp_path, specs, num_objects=3)

        assert len(library.select(candidates=[spec.url for spec in specs])) == 3


class TestMeasurementCache:
    """Measurements survive a round trip, so a later resolve opens no assets."""

    def test_cached_measurement_restores_the_bounding_box_as_a_tuple(self, tmp_path):
        """JSON has no tuple, so a round trip must not silently turn the extents into a list."""
        spec = make_spec("a/Apple/a.usd", dims=(0.07, 0.08, 0.09))
        restored = make_library(tmp_path, [spec]).audit(spec.url)

        assert restored.dims == (0.07, 0.08, 0.09)
        assert restored.max_dim == pytest.approx(0.09)

    def test_an_asset_that_failed_to_open_is_not_retried(self, tmp_path):
        """Caching the failure keeps a broken asset from being fetched on every resolve."""
        library = make_library(tmp_path, [])
        library._cache["a/Broken/a.usd"] = None

        assert library.audit("a/Broken/a.usd") is None

    def test_entry_from_an_older_field_layout_is_discarded(self, tmp_path):
        """The cache is regenerable, so a layout change must re-measure rather than raise."""
        library = make_library(tmp_path, [])
        library._cache["a/Apple/a.usd"] = {
            "url": "a/Apple/a.usd",
            "body_path": "/B",
            "mass": 0.2,
            "dims": [0.07, 0.07, 0.07],
            "fet003_passed": True,  # the field validation_passed used to be called
        }

        assert library.audit("a/Apple/a.usd") is None  # re-measured, and unreachable in a test


class TestResolutionRecord:
    """The recorded asset set is authoritative, so a run needs no search service."""

    def _library(self, tmp_path, specs, **kwargs):
        library = make_library(tmp_path, specs, **kwargs)
        library.search = lambda: self._searched() or [spec.url for spec in specs]
        return library

    def _searched(self):
        self.searches += 1

    def test_first_resolve_records_what_it_found(self, tmp_path):
        self.searches = 0
        specs = [make_spec(f"a/O{i}/a.usd", mass=0.1 + i * 0.05) for i in range(6)]

        resolved = self._library(tmp_path, specs, num_objects=4).resolve()

        recorded = json.loads((tmp_path / "objects.json").read_text())
        assert len(resolved) == 4
        assert recorded["assets"] == resolved
        assert self.searches == 1

    def test_a_later_resolve_uses_the_record_instead_of_searching(self, tmp_path):
        self.searches = 0
        specs = [make_spec(f"a/O{i}/a.usd", mass=0.1 + i * 0.05) for i in range(6)]
        first = self._library(tmp_path, specs, num_objects=4).resolve()

        second = self._library(tmp_path, specs, num_objects=4).resolve()

        assert second == first
        assert self.searches == 1, "the recorded set must be used as-is"

    def test_the_record_is_used_even_when_the_filter_changes(self, tmp_path):
        """The record is the reproducible answer; editing the config does not silently re-resolve."""
        self.searches = 0
        specs = [make_spec(f"a/O{i}/a.usd", mass=0.1 + i * 0.05) for i in range(6)]
        first = self._library(tmp_path, specs, num_objects=4).resolve()

        widened = self._library(tmp_path, specs, num_objects=4, mass_range=(0.005, 0.2)).resolve()

        assert widened == first
        assert self.searches == 1

    def test_always_query_resolves_afresh(self, tmp_path):
        self.searches = 0
        specs = [make_spec(f"a/O{i}/a.usd", mass=0.1 + i * 0.05) for i in range(6)]
        self._library(tmp_path, specs, num_objects=4).resolve()

        library = self._library(tmp_path, specs, num_objects=4)
        library.cfg.always_query = True
        library.resolve()

        assert self.searches == 2

    def test_a_recorded_run_reaches_neither_the_service_nor_the_search_package(self, tmp_path):
        """A recorded set has to work with no network and without the optional search package."""
        self.searches = 0
        specs = [make_spec(f"a/O{i}/a.usd", mass=0.1 + i * 0.05) for i in range(6)]
        self._library(tmp_path, specs, num_objects=4).resolve()

        library = make_library(tmp_path, specs, num_objects=4)

        def unavailable():
            raise ModuleNotFoundError("No module named 'simready'")

        library.search = unavailable

        assert len(library.resolve()) == 4
