# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Authoring invariants for the USD assets vendored under ``isaaclab_assets/data``.

Deliberately Kit-free: these run in the fast test lane, which is where a duplicate
articulation root goes unnoticed, because an unregistered codeless schema is dropped
from the composed prim definition.
"""

import pathlib

import pytest

from pxr import Usd

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

# Schemas that make a prim an articulation root. ``NewtonArticulationRootAPI`` is codeless and
# includes ``PhysicsArticulationRootAPI``, so a prim carrying it answers ``HasAPI`` in Kit but not
# under plain usd-core. Resolving those built-ins needs a schema registry, so the set is explicit
# and must grow when a backend adds its own root schema.
ROOT_API_SCHEMAS = {"PhysicsArticulationRootAPI", "NewtonArticulationRootAPI"}

VENDORED_ASSETS = sorted(
    path
    for path in (pathlib.Path(ISAACLAB_ASSETS_DATA_DIR) / "Mujoco_Menagerie").glob("*/*/*.usda")
    if "payloads" not in path.parts
)


def _root_bearing_prims(stage: Usd.Stage) -> list[str]:
    """Paths of prims applying any articulation-root schema, read from the raw list-op.

    ``GetAppliedSchemas`` silently omits unregistered codeless schemas; the authored list-op
    keeps them, which is what makes this check valid without Kit.
    """
    paths = []
    for prim in stage.Traverse():
        applied = prim.GetMetadata("apiSchemas")
        if applied and ROOT_API_SCHEMAS & set(applied.GetAddedOrExplicitItems()):
            paths.append(prim.GetPath().pathString)
    return paths


@pytest.mark.parametrize("asset_path", VENDORED_ASSETS, ids=lambda path: path.parent.name)
def test_vendored_asset_has_at_most_one_articulation_root(asset_path):
    """Every physics variant resolves to a single articulation root.

    Two roots make each consumer pick a different one: the schema helper prunes nested roots,
    Newton's importer takes the innermost enclosing one, and the articulation and joint-wrench
    sensors raise outright.
    """
    stage = Usd.Stage.Open(str(asset_path))
    default_prim = stage.GetDefaultPrim()
    assert default_prim, f"{asset_path.name} declares no default prim"

    variant_set = default_prim.GetVariantSet("Physics")
    variants = variant_set.GetVariantNames() or [None]
    for variant in variants:
        if variant is not None:
            variant_set.SetVariantSelection(variant)
        roots = _root_bearing_prims(stage)
        assert len(roots) <= 1, f"{asset_path.name} variant '{variant}' has {len(roots)} roots: {roots}"
