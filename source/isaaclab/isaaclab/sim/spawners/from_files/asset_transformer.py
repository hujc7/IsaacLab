# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run Asset Transformer profiles before referencing USD files into a scene."""

from __future__ import annotations

from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from isaaclab.utils import to_dict
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.version import has_kit

from ...utils.extensions import enable_extension

if TYPE_CHECKING:
    from pxr import Usd

    from .from_files_cfg import UsdFileCfg

# Spawns and their clones reference these files. Keep the packages available until process exit.
_packages: list[TemporaryDirectory] = []


def transform_usd(usd_path: str, cfg: UsdFileCfg, stage: Usd.Stage) -> str:
    """Localize a source, run its configured profile, and return the published USD path."""
    from pxr import Ar  # noqa: PLC0415

    if has_kit():
        enable_extension("isaacsim.asset.transformer")
    try:
        from isaacsim.asset.transformer import AssetTransformerManager, RuleProfile, RuleSpec  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        if exc.name not in {"isaacsim", "isaacsim.asset", "isaacsim.asset.transformer"}:
            raise
        raise RuntimeError(
            "USD transformation requires Isaac Sim's Asset Transformer. "
            "For Kitless Isaac Lab, install the importers extra (`uv sync --extra importers`)."
        ) from exc

    from .asset_transformer_rules import RelocateRigidBodyToRoot  # noqa: PLC0415

    profile_name = cfg.transformer.profile
    rule_cls = {"relocate_rigid_body_to_root": RelocateRigidBodyToRoot}.get(profile_name)
    if rule_cls is None:
        raise ValueError(f"Unknown USD transformer profile: {profile_name!r}.")

    manager = AssetTransformerManager()
    manager.registry.register(rule_cls)
    variants = cfg.variants or {}
    if not isinstance(variants, dict):
        variants = to_dict(variants)
    profile = RuleProfile(
        profile_name=profile_name,
        flatten_source=False,
        rules=[
            RuleSpec(
                name=profile_name,
                type=f"{rule_cls.__module__}.{rule_cls.__qualname__}",
                params={"variants": variants, "make_uninstanceable": cfg.make_uninstanceable},
            )
        ],
    )

    package = TemporaryDirectory(prefix="isaaclab_asset_transform_", ignore_cleanup_errors=True)
    try:
        with Ar.ResolverContextBinder(stage.GetPathResolverContext()):
            source = retrieve_file_path(usd_path)
            report = manager.run(source, profile, package_root=package.name)
        errors = [f"{result.rule.name}: {result.error}" for result in report.results if not result.success]
        errors.extend(report.composition_errors)
        if report.composition_errors_truncated:
            errors.append("USD composition validation was incomplete.")
        if errors or report.output_stage_path is None:
            details = "; ".join(errors) or "No USD output was produced."
            raise RuntimeError(f"USD transform '{profile_name}' failed for '{usd_path}': {details}")
    except BaseException:
        package.cleanup()
        raise
    _packages.append(package)
    return report.output_stage_path
