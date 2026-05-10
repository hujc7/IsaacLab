# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit + lint tests for the decorator-based preset CLI.

Three layers exercised here:

* :mod:`isaaclab.utils.presets` registry: ``@register(PresetTarget, name)``
  decorator binding + :meth:`PresetRegistry.name_of` / :meth:`target_of`
  lookups + :class:`PresetTarget` legacy-alias resolution.
* :mod:`isaaclab_tasks.utils.preset_cli`: argparse integration, target
  classification via the registry, ``apply`` argv translation,
  target-specific error formatting, ``setup_cli`` one-line wrapper.
* Cross-env lint: walks every registered task config and asserts that no
  ``PresetCfg`` introduces a non-canonical name for a value whose type is
  bound to a canonical name (catches vocabulary drift).
"""

from __future__ import annotations

import argparse
import sys

import pytest

# Force-import every backend cfg module so its ``@register`` decorator
# fires and the canonical registry is populated for assertions that
# stub ``_collect_task_options`` (which would otherwise skip env-config loading).
# Production code does not need this -- ``_collect_task_options`` loads env configs
# which transitively import their backends.
from isaaclab_newton.physics import kamino_manager_cfg, mjwarp_manager_cfg  # noqa: F401
from isaaclab_newton.renderers import newton_warp_renderer_cfg  # noqa: F401
from isaaclab_ov.renderers import ovrtx_renderer_cfg  # noqa: F401
from isaaclab_ovphysx.physics import ovphysx_manager_cfg  # noqa: F401
from isaaclab_physx.physics import physx_manager_cfg  # noqa: F401
from isaaclab_physx.renderers import isaac_rtx_renderer_cfg  # noqa: F401

from isaaclab.physics.physics_manager_cfg import PhysicsCfg
from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.utils import PresetCfg, configclass
from isaaclab.utils.presets import PresetTarget, PresetRegistry, register

from isaaclab_tasks.utils.preset_cli import PresetCli, setup_cli

# Short aliases used heavily in assertions below.
KIND_PHYSICS = PresetTarget.PHYSICS
KIND_RENDERER = PresetTarget.RENDERER
KIND_DOMAIN = PresetTarget.DOMAIN

_extract_task_from_argv = PresetCli._extract_task_from_argv
_target_of_class = PresetCli._target_of_class
_target_of_value = PresetCli._target_of_value
_vocab_for_target = PresetCli._vocab_for_target
validate_preset_cfg = PresetCli.validate_preset_cfg
canonical_name_for = PresetRegistry.name_of

# ----------------------------------------------------------------------------
# Synthetic decorated classes for unit tests (kept local to avoid polluting
# the global registry with names a real task might use).
# ----------------------------------------------------------------------------


@register(PresetTarget.PHYSICS, "test_physx")
@configclass
class _TestPhysxCfg(PhysicsCfg):
    backend: str = "physx"


@register(PresetTarget.PHYSICS, "test_newton_solver")
@configclass
class _TestNewtonSolverCfg:
    """Stand-in for MJWarpSolverCfg / KaminoSolverCfg (no NewtonSolverCfg base
    so we don't depend on isaaclab_newton being importable in this test)."""


@register(PresetTarget.RENDERER, "test_isaac_rtx")
@configclass
class _TestRtxRendererCfg(RendererCfg):
    pass


# ----------------------------------------------------------------------------
# preset_meta: decorator + canonical_name_for
# ----------------------------------------------------------------------------


def test_decorator_attaches_preset_name_to_class():
    assert _TestPhysxCfg._preset_name == "test_physx"
    assert _TestRtxRendererCfg._preset_name == "test_isaac_rtx"


def test_decorator_populates_registry():
    assert PresetRegistry._entries["test_physx"][0] is _TestPhysxCfg
    assert PresetRegistry._entries["test_isaac_rtx"][0] is _TestRtxRendererCfg


def test_decorator_rejects_duplicate_name_on_different_class():
    with pytest.raises(RuntimeError, match="already bound"):

        @register(PresetTarget.PHYSICS, "test_physx")
        class _OtherCfg:
            pass


def test_decorator_rejects_same_class_under_different_name():
    """A class can carry only one canonical name; re-decorating with a different
    name must fail rather than silently leave two registry entries pointing at
    the same class."""
    with pytest.raises(RuntimeError, match="cannot rebind class"):

        @register(PresetTarget.PHYSICS, "test_physx_alias")
        @register(PresetTarget.PHYSICS, "test_physx_first")
        @configclass
        class _DoubleNamedCfg(PhysicsCfg):
            pass


def test_decorator_idempotent_on_same_class():
    """Re-decorating the same class with the same (target, name) is a no-op (no error)."""
    decorated = register(PresetTarget.PHYSICS, "test_physx")(_TestPhysxCfg)
    assert decorated is _TestPhysxCfg
    assert PresetRegistry._entries["test_physx"][0] is _TestPhysxCfg


def test_decorator_rejects_same_class_same_name_different_kind():
    """Re-registering a class under the same name but a *different* target must
    raise -- silently mutating ``_preset_kind`` would change apply()-time
    classification without any visible signal at decoration time.
    """
    with pytest.raises(RuntimeError, match="cannot re-register"):
        register(PresetTarget.RENDERER, "test_physx")(_TestPhysxCfg)


def test_canonical_name_for_direct_class():
    assert canonical_name_for(_TestPhysxCfg()) == "test_physx"


def test_canonical_name_for_solver_dispatch():
    """Outer value with no decorator but ``solver_cfg`` attribute resolves via the solver."""

    class _OuterCfg:
        solver_cfg = _TestNewtonSolverCfg()

    assert canonical_name_for(_OuterCfg()) == "test_newton_solver"


def test_canonical_name_for_undecorated_returns_none():
    assert canonical_name_for("rgb") is None
    assert canonical_name_for(object()) is None


def test_canonical_name_for_subclass_inherits():
    """Subclasses inherit ``_preset_name`` via attribute lookup MRO."""

    class _Sub(_TestPhysxCfg):
        pass

    assert canonical_name_for(_Sub()) == "test_physx"


# ----------------------------------------------------------------------------
# preset_cli: target classification
# ----------------------------------------------------------------------------


def test_kind_of_physics_subclass():
    assert _target_of_class(_TestPhysxCfg) == KIND_PHYSICS
    assert _target_of_value(_TestPhysxCfg()) == KIND_PHYSICS


def test_kind_of_renderer_subclass():
    assert _target_of_class(_TestRtxRendererCfg) == KIND_RENDERER
    assert _target_of_value(_TestRtxRendererCfg()) == KIND_RENDERER


def test_kind_of_arbitrary_value_is_domain():
    assert _target_of_value("rgb") == KIND_DOMAIN
    assert _target_of_value(123) == KIND_DOMAIN


# ----------------------------------------------------------------------------
# preset_cli: argparse integration
# ----------------------------------------------------------------------------


def _build_parser_with_task(*, add_help: bool = False) -> tuple[argparse.ArgumentParser, PresetCli]:
    parser = argparse.ArgumentParser(prog="train.py", add_help=add_help)
    parser.add_argument("--task", type=str, default=None)
    cli = PresetCli()
    cli.add_args(parser)
    return parser, cli


def test_add_args_registers_three_flags():
    parser, _ = _build_parser_with_task()
    actions = {a.dest for a in parser._actions}
    assert {"physics", "renderer", "presets"} <= actions


def test_argparse_accepts_both_equals_and_space_forms():
    parser, _ = _build_parser_with_task()
    space_form, _ = parser.parse_known_args(["--physics", "newton_mjwarp"])
    equals_form, _ = parser.parse_known_args(["--physics=newton_mjwarp"])
    assert space_form.physics == equals_form.physics == "newton_mjwarp"


def test_presets_flag_is_csv_string():
    parser, _ = _build_parser_with_task()
    args, _ = parser.parse_known_args(["--presets", "albedo,inference"])
    assert args.presets == "albedo,inference"


def test_add_args_is_idempotent_on_same_parser():
    """Calling ``add_args`` twice on the same parser must not duplicate flags.

    Covers BOTH same-instance and cross-instance reuse: the idempotency
    guard stamps the parser object, not the PresetCli instance, so
    ``setup_cli`` (which creates a fresh PresetCli on every call) is
    safe to invoke twice in a row on the same parser.
    """
    parser = argparse.ArgumentParser(prog="train.py", add_help=False)
    parser.add_argument("--task", type=str, default=None)

    # Same instance, twice.
    cli = PresetCli()
    cli.add_args(parser)
    cli.add_args(parser)

    # Different instance, same parser -- this is the case ``setup_cli``
    # produces when called twice in a notebook or via a wrapper that
    # re-enters the setup path.
    PresetCli().add_args(parser)

    physics_actions = [a for a in parser._actions if "--physics" in a.option_strings]
    assert len(physics_actions) == 1, "repeated add_args (any instance) must not duplicate --physics"


# ----------------------------------------------------------------------------
# preset_cli: apply argv translation
# ----------------------------------------------------------------------------


_SYNTHETIC_OPTIONS = {
    KIND_PHYSICS: {"default": ["sim.physics"], "physx": ["sim.physics"], "newton_mjwarp": ["sim.physics"]},
    KIND_RENDERER: {"default": ["camera.renderer_cfg"], "newton_renderer": ["camera.renderer_cfg"]},
    KIND_DOMAIN: {"default": ["camera"], "albedo": ["camera"], "depth": ["camera"]},
}


@pytest.fixture
def stub_task(monkeypatch):
    """Replace _collect_task_options so apply() doesn't try to load a real task config."""
    monkeypatch.setattr(PresetCli, "_collect_task_options", staticmethod(lambda task: _SYNTHETIC_OPTIONS))


def test_apply_no_flags_returns_argv_unchanged(stub_task):
    parser, cli = _build_parser_with_task()
    args, remaining = parser.parse_known_args(["--task=Fake-v0", "env.sim.dt=0.001"])
    result = cli.apply(args, remaining)
    assert result == ["env.sim.dt=0.001"]


def test_apply_translates_physics_to_presets_token(stub_task):
    parser, cli = _build_parser_with_task()
    args, remaining = parser.parse_known_args(["--task=Fake-v0", "--physics", "newton_mjwarp", "env.sim.dt=0.001"])
    result = cli.apply(args, remaining)
    assert result == ["presets=newton_mjwarp", "env.sim.dt=0.001"]


def test_apply_merges_three_flags_into_one_token(stub_task):
    parser, cli = _build_parser_with_task()
    args, remaining = parser.parse_known_args(
        [
            "--task=Fake-v0",
            "--physics",
            "newton_mjwarp",
            "--renderer",
            "newton_renderer",
            "--presets",
            "albedo,depth",
        ]
    )
    result = cli.apply(args, remaining)
    assert result[0].startswith("presets=")
    csv_names = result[0][len("presets=") :].split(",")
    assert csv_names == ["newton_mjwarp", "newton_renderer", "albedo", "depth"]


def test_apply_merges_with_legacy_presets_token(stub_task):
    parser, cli = _build_parser_with_task()
    args, remaining = parser.parse_known_args(["--task=Fake-v0", "--physics", "newton_mjwarp", "presets=albedo"])
    result = cli.apply(args, remaining)
    assert result == ["presets=newton_mjwarp,albedo"]


def test_apply_dedupes_repeated_names(stub_task):
    parser, cli = _build_parser_with_task()
    args, remaining = parser.parse_known_args(
        [
            "--task=Fake-v0",
            "--physics",
            "newton_mjwarp",
            "presets=newton_mjwarp,albedo",
        ]
    )
    result = cli.apply(args, remaining)
    assert result == ["presets=newton_mjwarp,albedo"]


def test_apply_with_only_legacy_presets_passes_through(stub_task):
    parser, cli = _build_parser_with_task()
    args, remaining = parser.parse_known_args(["--task=Fake-v0", "presets=newton_mjwarp"])
    result = cli.apply(args, remaining)
    assert result == ["presets=newton_mjwarp"]


def test_apply_requires_task_when_flags_used(stub_task):
    parser, cli = _build_parser_with_task()
    args, remaining = parser.parse_known_args(["--physics", "newton_mjwarp"])
    with pytest.raises(SystemExit) as exc:
        cli.apply(args, remaining)
    assert "require --task" in str(exc.value)


# ----------------------------------------------------------------------------
# preset_cli: target-specific error messages
# ----------------------------------------------------------------------------


def test_apply_rejects_non_canonical_physics_name(stub_task):
    """A name that's not in IsaacLab's canonical physics vocabulary errors fast."""
    parser, cli = _build_parser_with_task()
    args, remaining = parser.parse_known_args(["--task=Fake-v0", "--physics", "super_solver_v2"])
    with pytest.raises(SystemExit) as exc:
        cli.apply(args, remaining)
    msg = str(exc.value)
    assert "--physics 'super_solver_v2'" in msg
    assert "not a recognized physics preset name" in msg
    # Recognized vocabulary listed for the user
    assert "physx" in msg


def test_apply_canonical_but_not_in_task(stub_task):
    """Canonical name that this task doesn't define -> different error than 'not recognized'."""
    parser, cli = _build_parser_with_task()
    args, remaining = parser.parse_known_args(["--task=Fake-v0", "--physics", "newton_kamino"])
    with pytest.raises(SystemExit) as exc:
        cli.apply(args, remaining)
    msg = str(exc.value)
    assert "newton_kamino" in msg
    assert "not defined for task" in msg
    # Still tells the user what this task DOES define
    assert "physx" in msg or "newton_mjwarp" in msg


def test_apply_suggests_correct_flag_when_name_belongs_to_other_kind(stub_task):
    parser, cli = _build_parser_with_task()
    args, remaining = parser.parse_known_args(["--task=Fake-v0", "--physics", "albedo"])
    with pytest.raises(SystemExit) as exc:
        cli.apply(args, remaining)
    msg = str(exc.value)
    assert "--presets albedo" in msg


def test_apply_normalizes_typed_flag_legacy_alias(stub_task):
    """``--physics newton`` (legacy) must normalize to ``newton_mjwarp``.

    Pre-fix, typed flag values bypassed alias normalization and produced
    "unknown preset" errors for legacy names that the override path
    (``presets=newton``) handled correctly. The fix routes typed flags
    through :meth:`PresetTarget.normalize_name` before validation.
    """
    parser, cli = _build_parser_with_task()
    args, remaining = parser.parse_known_args(["--task=Fake-v0", "--physics", "newton"])
    with pytest.warns(FutureWarning, match="newton.*newton_mjwarp"):
        result = cli.apply(args, remaining)
    assert result == ["presets=newton_mjwarp"]


def test_apply_clear_error_when_kind_is_undefined(monkeypatch):
    """Task that defines no renderer presets at all -> targeted message."""
    options = {KIND_PHYSICS: _SYNTHETIC_OPTIONS[KIND_PHYSICS], KIND_RENDERER: {}, KIND_DOMAIN: {}}
    monkeypatch.setattr(PresetCli, "_collect_task_options", staticmethod(lambda task: options))
    parser, cli = _build_parser_with_task()
    args, remaining = parser.parse_known_args(["--task=Fake-v0", "--renderer", "newton_renderer"])
    with pytest.raises(SystemExit) as exc:
        cli.apply(args, remaining)
    msg = str(exc.value)
    assert "does not define any renderer presets" in msg


# ----------------------------------------------------------------------------
# preset_cli: describe_task formatting
# ----------------------------------------------------------------------------


def test_describe_task_lists_each_kind_on_its_own_line(stub_task, monkeypatch):
    monkeypatch.setattr(PresetCli, "_collect_task_options", staticmethod(lambda task: _SYNTHETIC_OPTIONS))
    out = PresetCli.describe_task("Fake-v0")
    # Each flag heading on its own line
    assert "\n  --physics" in out
    assert "\n  --renderer" in out
    assert "\n  --presets" in out
    # Names appear (sorted, no curly braces)
    assert "physx" in out and "newton_mjwarp" in out
    assert "{" not in out  # no `{...}` group syntax


def test_describe_task_emits_friendly_message_on_failure(monkeypatch):
    def boom(task):
        raise RuntimeError("nope")

    monkeypatch.setattr(PresetCli, "_collect_task_options", staticmethod(boom))
    out = PresetCli.describe_task("Fake-v0")
    assert "unavailable" in out
    assert "RuntimeError" in out


# ----------------------------------------------------------------------------
# preset_cli: collect_selected
# ----------------------------------------------------------------------------


def test_collect_selected_unions_all_three_flags():
    parser, cli = _build_parser_with_task()
    args, _ = parser.parse_known_args(
        [
            "--physics",
            "newton_mjwarp",
            "--renderer",
            "newton_renderer",
            "--presets",
            "albedo,inference",
        ]
    )
    assert cli.collect_selected(args) == {"newton_mjwarp", "newton_renderer", "albedo", "inference"}


def test_collect_selected_skips_blank_csv_entries():
    parser, cli = _build_parser_with_task()
    args, _ = parser.parse_known_args(["--presets", "a,, b ,c"])
    assert cli.collect_selected(args) == {"a", "b", "c"}


# ----------------------------------------------------------------------------
# preset_cli: extract_task_from_argv
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["--task=Foo-v0"], "Foo-v0"),
        (["--task", "Foo-v0"], "Foo-v0"),
        (["--headless", "--task", "Foo-v0", "--num_envs", "16"], "Foo-v0"),
        (["--num_envs", "16"], None),
        ([], None),
    ],
)
def test_extract_task_from_argv(argv, expected):
    assert _extract_task_from_argv(argv) == expected


# ----------------------------------------------------------------------------
# setup_cli: sys.argv behavior
# ----------------------------------------------------------------------------


def _make_argparser_with_task() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="train.py", add_help=False)
    parser.add_argument("--task", type=str, default=None)
    return parser


@pytest.fixture
def stub_app_launcher(monkeypatch):
    """Avoid Isaac Sim's stdin-reading kit_app init in setup_cli tests by
    pre-populating ``sys.modules`` with a fake ``isaaclab.app`` module before
    ``setup_cli`` does its lazy ``from isaaclab.app import AppLauncher``."""
    import types

    fake_module = types.ModuleType("isaaclab.app")
    fake_module.AppLauncher = type("AppLauncher", (), {"add_app_launcher_args": staticmethod(lambda parser: None)})
    monkeypatch.setitem(sys.modules, "isaaclab.app", fake_module)


def test_setup_cli_commit_true_rewrites_sys_argv(stub_task, stub_app_launcher, monkeypatch):
    """``commit=True`` (default) replaces sys.argv with ``presets=<csv>`` + leftover Hydra args."""
    parser = _make_argparser_with_task()
    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "--task=Fake-v0", "--physics", "newton_mjwarp", "env.sim.dt=0.001"],
    )
    args = setup_cli(parser)
    assert args.physics == "newton_mjwarp"
    # sys.argv mutated in-place: argparse-consumed flags gone; preset broadcast prepended
    assert sys.argv[0] == "train.py"
    assert sys.argv[1:] == ["presets=newton_mjwarp", "env.sim.dt=0.001"]


def test_setup_cli_commit_false_returns_pieces_and_leaves_sys_argv(stub_task, stub_app_launcher, monkeypatch):
    """``commit=False`` returns (args, remaining, preset_cli) and does NOT rewrite sys.argv."""
    parser = _make_argparser_with_task()
    original_argv = ["train.py", "--task=Fake-v0", "--physics", "newton_mjwarp", "env.sim.dt=0.001"]
    monkeypatch.setattr("sys.argv", list(original_argv))
    args, remaining, preset = setup_cli(parser, commit=False)
    # sys.argv untouched
    assert sys.argv == original_argv
    # caller can still mutate `remaining` and finalize via .commit()
    assert remaining == ["env.sim.dt=0.001"]
    assert isinstance(preset, PresetCli)
    preset.commit(args, remaining)
    assert sys.argv[1:] == ["presets=newton_mjwarp", "env.sim.dt=0.001"]


# ----------------------------------------------------------------------------
# validate_preset_cfg: loose canonical-name rule
# ----------------------------------------------------------------------------


def test_validate_passes_when_canonical_name_present():
    """The canonical name must appear among the field names; variants beyond it are allowed."""

    @configclass
    class GoodCfg(PresetCfg):
        default = _TestPhysxCfg()
        test_physx = default
        physx_high_fidelity = _TestPhysxCfg()  # variant — same type, different name — OK

    assert validate_preset_cfg(GoodCfg()) == []


def test_validate_fails_when_only_field_uses_non_canonical_name():
    @configclass
    class BadCfg(PresetCfg):
        only_high_fi = _TestPhysxCfg()

    errors = validate_preset_cfg(BadCfg())
    assert len(errors) == 1
    assert "test_physx" in errors[0]
    assert "only_high_fi" in errors[0]


def test_validate_fails_when_multiple_fields_all_non_canonical():
    @configclass
    class WorseCfg(PresetCfg):
        low_fi = _TestPhysxCfg()
        high_fi = _TestPhysxCfg()

    errors = validate_preset_cfg(WorseCfg())
    assert len(errors) == 1
    assert "test_physx" in errors[0]


def test_validate_ignores_alternatives_with_undecorated_types():
    """Domain-style PresetCfgs (e.g., camera-data-type variants) carry no canonical
    binding and pass through validation unchanged."""

    @configclass
    class DomainCfg(PresetCfg):
        default = "rgb"
        depth = "depth"
        albedo = "albedo"

    assert validate_preset_cfg(DomainCfg()) == []


# ----------------------------------------------------------------------------
# Integration smoke + cross-env vocabulary lint
# ----------------------------------------------------------------------------


def test_enumerate_real_cartpole_camera_presets_task():
    """Smoke-test against the registered preset-rich Cartpole task."""
    import isaaclab_tasks  # noqa: F401

    options = PresetCli._collect_task_options("Isaac-Cartpole-Camera-Presets-Direct-v0")
    assert "physx" in options[KIND_PHYSICS]
    assert "newton_mjwarp" in options[KIND_PHYSICS]
    assert "newton_renderer" in options[KIND_RENDERER]
    assert "ovrtx_renderer" in options[KIND_RENDERER]
    for name in ("rgb", "depth", "albedo"):
        assert name in options[KIND_DOMAIN]


def test_canonical_vocabulary_populated_at_import_time():
    """All seven canonical IsaacLab presets must resolve once preset_cli is imported.

    The module-level loop at the top of preset_cli.py force-loads each
    backend cfg module so its ``@register`` decorator runs and the
    canonical registry is non-empty before any ``add_args`` call.
    """
    physics_vocab = set(_vocab_for_target(KIND_PHYSICS))
    renderer_vocab = set(_vocab_for_target(KIND_RENDERER))
    assert {"physx", "ovphysx", "newton_mjwarp", "newton_kamino"} <= physics_vocab
    assert {"isaacsim_rtx_renderer", "newton_renderer", "ovrtx_renderer"} <= renderer_vocab


# Tasks that are *expected* to fail to load via load_cfg_from_registry in the
# CI lint context (e.g. they require Isaac Sim Kit at config-construction time,
# or they import optional native deps unavailable in the lint environment).
# Empty by default -- if this list grows, document each entry with the failure
# class so it can be revisited.
_DRIFT_LINT_EXPECTED_SKIPS: frozenset[str] = frozenset()


def test_no_canonical_vocabulary_drift_in_registered_tasks():
    """CI lint -- walks every registered Isaac task and runs validate_preset_cfg
    on every PresetCfg in its config tree.

    Catches drift: a future PR that adds e.g. ``super_solver = NewtonCfg(MJWarpSolverCfg())``
    to some env config fails this test, forcing either a vocabulary update or
    a rename to the canonical ``newton_mjwarp`` name.

    Unexpected task-load failures are surfaced as test failures, not silently
    skipped. Add to ``_DRIFT_LINT_EXPECTED_SKIPS`` only with a documented reason.
    """
    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401
    from isaaclab.utils.presets import walk_presets
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    unexpected_skips: list[tuple[str, str]] = []
    violations: list[tuple[str, str]] = []
    for task_id in list(gym.envs.registry):
        if not task_id.startswith("Isaac-"):
            continue
        try:
            env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
            if isinstance(env_cfg, type):
                env_cfg = env_cfg()
        except Exception as exc:
            if task_id not in _DRIFT_LINT_EXPECTED_SKIPS:
                unexpected_skips.append((task_id, f"{type(exc).__name__}: {exc}"))
            continue
        if not (hasattr(env_cfg, "__dataclass_fields__") or isinstance(env_cfg, dict)):
            continue

        # Walk the tree and validate every PresetCfg we encounter.
        def _check(_parent, _key, preset_obj, _path):
            for err in validate_preset_cfg(preset_obj):
                violations.append((task_id, err))

        if isinstance(env_cfg, PresetCfg):
            for err in validate_preset_cfg(env_cfg):
                violations.append((task_id, err))
        walk_presets(env_cfg, "", _check)

    if unexpected_skips:
        formatted = "\n".join(f"  [{tid}] {msg}" for tid, msg in unexpected_skips)
        pytest.fail(
            f"{len(unexpected_skips)} task(s) failed to load and are not in the expected-skip "
            f"allowlist; the drift lint cannot run on them:\n{formatted}\n"
            "Add to _DRIFT_LINT_EXPECTED_SKIPS with a documented reason, or fix the underlying "
            "load failure."
        )
    if violations:
        formatted = "\n".join(f"  [{tid}] {msg}" for tid, msg in violations)
        pytest.fail(
            f"Canonical preset vocabulary drift detected in {len(violations)} preset(s):\n{formatted}\n"
            "Either rename the field to use the canonical name, or update the canonical vocabulary "
            "(via @register(PresetTarget.X, ...) in the relevant cfg class)."
        )


# Note on cross-target name collision (raised in codex review of #5535): the
# repo's convention is to reuse backend labels (``physx``, ``newton_mjwarp``)
# as field names on backend-tagged PresetCfgs across kinds -- e.g. a
# ``newton_mjwarp`` alternative may appear on a physics PresetCfg (typed
# value), an events PresetCfg (NewtonEventCfg, classified as domain), and a
# sensor PresetCfg (NewtonContactSensorCfg, also domain). The global
# ``presets=newton_mjwarp`` broadcast then propagates to all of them, which
# is the intended behavior. A blanket "no cross-target name overlap" lint would
# flag this established convention as drift, so we don't enforce it. The
# stronger fix (path-specific selections) would change the resolution model
# itself and is out of scope here.
