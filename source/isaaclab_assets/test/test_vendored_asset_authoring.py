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
from isaaclab_assets.robots.shadow_hand import ShadowHand

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

SHADOW_HAND_ASSET = pathlib.Path(ISAACLAB_ASSETS_DATA_DIR) / "Mujoco_Menagerie/shadow_hand/right_hand/right_hand.usda"


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


def _tendon_actuator_control_ranges(stage: Usd.Stage, tendon_names: list[str]) -> dict[str, tuple[float, float]]:
    """Control range of each MuJoCo actuator whose transmission is one of ``tendon_names``.

    ``mjc:ctrlRange:min`` is left unauthored on these actuators, which is MuJoCo's zero default;
    reading it as ``0.0`` is what makes the authored range comparable to a ``(lower, upper)`` pair.
    """
    control_ranges = {}
    for prim in stage.Traverse():
        if prim.GetTypeName() != "MjcActuator":
            continue
        targets = prim.GetRelationship("mjc:target").GetTargets()
        target_name = targets[0].name if targets else None
        if target_name not in tendon_names:
            continue
        lower = prim.GetAttribute("mjc:ctrlRange:min").Get()
        upper = prim.GetAttribute("mjc:ctrlRange:max").Get()
        control_ranges[target_name] = (0.0 if lower is None else float(lower), float(upper))
    return control_ranges


def test_shadow_hand_tendon_control_range_matches_its_configuration():
    """The asset's tendon control range is what :class:`ShadowHand` says the tasks may command.

    A tendon carries no position limit of its own, so what bounds the command is the actuator's
    control range, and no runtime accessor reports it on both backends -- PhysX drives a tendon by
    offset and has no equivalent. The tasks therefore command against the configured span, and this
    is what keeps re-authoring the asset from silently disagreeing with it.
    """
    stage = Usd.Stage.Open(str(SHADOW_HAND_ASSET))
    stage.GetDefaultPrim().GetVariantSet("Physics").SetVariantSelection("mujoco")

    control_ranges = _tendon_actuator_control_ranges(stage, ShadowHand.tendon_names)

    assert sorted(control_ranges) == sorted(ShadowHand.tendon_names), (
        f"Expected one direct actuator per tendon {ShadowHand.tendon_names}, found {sorted(control_ranges)}"
    )
    for tendon_name, control_range in control_ranges.items():
        assert control_range == pytest.approx(ShadowHand.tendon_position_limits), (
            f"Tendon '{tendon_name}' authors control range {control_range}, but ShadowHand"
            f".tendon_position_limits is {ShadowHand.tendon_position_limits}"
        )
