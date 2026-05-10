# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""First-class CLI flags for preset selection (``--physics``, ``--renderer``,
``--presets``, plus any future targets).

Everything CLI-related lives on :class:`PresetCli` -- the argparse handler,
per-task vocabulary lookup, error formatting, validation, and the one-line
:meth:`PresetCli.setup` script wrapper. The only module-level identifier is
``setup_cli``, an alias for :meth:`PresetCli.setup` kept so script wiring
stays a single token.

Most scripts use the one-line form::

    parser = argparse.ArgumentParser(...)
    # ... script-specific args ...
    args_cli = setup_cli(parser)

Three override shapes
^^^^^^^^^^^^^^^^^^^^^

The word "preset" appears in three CLI shapes that look similar but
behave differently. Use this table when in doubt:

============================  ==============  ====================================================
Form                          Source          Effect
============================  ==============  ====================================================
``--physics=NAME``            typed flag      Broadcast: replace every PresetCfg of target PHYSICS
                                              whose alternatives include NAME.
``--renderer=NAME``           typed flag      Same shape as ``--physics`` for target RENDERER.
``--presets=A,B``             free-form       Broadcast: any PresetCfg whose alternatives include
                                              A or B gets replaced.
``env.X.Y=NAME``              path-targeted   REPLACE only the subtree at env.X.Y with the chosen
                                              alternative. Surgical, not broadcast.
============================  ==============  ====================================================

Combining typed-flag + ``--presets`` is fine: the names merge into one
CSV broadcast, deduped, then applied per-PresetCfg. Conflict only
arises when two requested names define DIFFERENT alternatives for the
SAME path; that raises ``ValueError`` at apply time.

Both ``--flag value`` and ``--flag=value`` are accepted by argparse.
Internally, the parsed flags translate into the same ``presets=<csv>``
global broadcast that :func:`isaaclab_tasks.utils.hydra.register_task`
consumes when running through the Hydra-decorator flow, so legacy
``presets=...`` invocations keep working.

Module placement
^^^^^^^^^^^^^^^^

This module lives in :mod:`isaaclab_tasks` rather than core because
every concern it owns is task-aware: gym task lookup via
:func:`isaaclab_tasks.utils.parse_cfg.load_cfg_from_registry`, and
AppLauncher wiring used by every IsaacLab script.

Generic value-type primitives (:class:`~isaaclab.utils.PresetCfg`,
:class:`~isaaclab.utils.PresetTarget`, :class:`~isaaclab.utils.PresetRegistry`)
and the tree-walking helpers (:func:`~isaaclab.utils.presets.collect_presets`,
:func:`~isaaclab.utils.presets.resolve_presets`) live in core so backend
cfg classes can decorate themselves without pulling in tasks.

Vocabulary and targets
^^^^^^^^^^^^^^^^^^^^^^

Canonical (target, name) bindings live next to their config classes via the
:func:`~isaaclab.utils.presets.register` decorator. Reading
``physx_manager_cfg.py`` (or any decorated cfg class) shows the binding
immediately::

    @register(PresetTarget.PHYSICS, "physx")
    @configclass
    class PhysxCfg(PhysicsCfg): ...

``PresetCli`` reads :class:`~isaaclab.utils.presets.PresetRegistry`
for everything: target classification (:meth:`PresetRegistry.target_of`),
per-target vocabulary (:meth:`PresetRegistry.names_for_target`), and the
set of registered targets (:meth:`PresetRegistry.all_targets`). Adding a
new :class:`PresetTarget` enum member adds a new ``--{target}`` flag
here automatically -- no target-specific code paths in this module.

Target dispatch falls back to ``value.solver_cfg`` when the outer value
isn't itself decorated (the Newton case: ``NewtonCfg`` inherits no
canonical binding, but its inner ``MJWarpSolverCfg`` /
``KaminoSolverCfg`` does).
"""

from __future__ import annotations

import argparse
import enum
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from isaaclab.utils.presets import (
    PresetCfg,
    PresetRegistry,
    PresetTarget,
    collect_presets,
    preset_fields,
)

# Review(jichuanh): this really feels redundant. I don't expect user to call the same function twice on argparser
class _ParserState(enum.IntFlag):
    """States a parser can be in once ``PresetCli`` has wired it.

    Stamped onto the parser object via ``parser._isaaclab_preset_state``
    so idempotency survives ``setup_cli`` rebuilding a fresh PresetCli on
    every call. Two independent operations need to be guarded:

    * ``ARGS_ADDED`` -- ``add_args`` has registered ``--physics`` /
      ``--renderer`` / ``--presets``. Re-running would raise
      ``argparse.ArgumentError`` on the duplicate flag.
    * ``HELP_INSTALLED`` -- ``_install_help_extension`` has swapped in
      :class:`PresetCli._HelpAction`. Re-running would wrap an already-
      wrapped action and double the per-task listing.

    They progress independently because ``add_args`` short-circuits when
    ``ARGS_ADDED`` is set but still calls ``_install_help_extension`` so
    a parser created without preset args at first can later get the
    custom help if presets are added.
    """

    ARGS_ADDED = 1
    HELP_INSTALLED = 2


# Sentinel attribute name stamped on the parser to track _ParserState.
_PARSER_STATE_ATTR = "_isaaclab_preset_state"


def _parser_state(parser: argparse.ArgumentParser) -> _ParserState:
    """Read the :class:`_ParserState` flag stamped on *parser* (zero if unset)."""
    return getattr(parser, _PARSER_STATE_ATTR, _ParserState(0))


def _set_parser_state(parser: argparse.ArgumentParser, flag: _ParserState) -> None:
    """OR *flag* into the parser's :class:`_ParserState`; create the attr if missing."""
    setattr(parser, _PARSER_STATE_ATTR, _parser_state(parser) | flag)


@dataclass(frozen=True)
class _PresetRequest:
    """A single typed-flag preset selection.

    Holds the target that owns the flag (``PresetTarget.PHYSICS``) and the
    requested name, post legacy-alias normalization. Built by
    :meth:`PresetCli._requests_from_args` so that
    :meth:`PresetCli.collect_selected` and :meth:`PresetCli.apply` do not
    each re-implement the parsing loop.
    """

    target: PresetTarget
    name: str


@dataclass
class PresetOverrides:
    """Categorized argv overrides destined for the Hydra-decorator flow.

    Built by :meth:`PresetCli.parse_overrides` and consumed by
    :meth:`PresetCli.apply_overrides`. All preset-name fields are
    expected to be already normalized against legacy aliases at parse
    time -- consumers do not re-normalize.

    Fields:
        global_presets: Names from ``presets=...`` tokens (broadcast to
            every matching :class:`PresetCfg` in the env/agent tree).
        path_selections: ``[(section, path, name)]`` from
            ``env.x.y=name`` tokens (REPLACE the subtree at that path).
        preset_scalars: ``[(full_path, value)]`` for scalar overrides
            that fall under a preset path.
        hydra_args: Argv tokens that are not preset overrides; passed
            through unchanged for Hydra to handle.
    """

    global_presets: list[str] = field(default_factory=list)
    path_selections: list[tuple[str, str, str]] = field(default_factory=list)
    preset_scalars: list[tuple[str, str]] = field(default_factory=list)
    hydra_args: list[str] = field(default_factory=list)

# Review(jichuanh): could ast.eval or something work here? is hardcode needed?
# Literal-string parsing for scalar override values. Used by
# :meth:`PresetCli._parse_val` (apply-time scalar substitution).
_LITERAL_MAP: dict[str, Any] = {"true": True, "false": False, "none": None, "null": None}

# Review(jichuanh): why this this comment here? Is it for anything? if it was for removed code, never leave a comment just for that.
# No backend force-import here. The single source of truth for "what
# canonical names exist" is the ``@register(...)`` decorator on each
# backend's cfg class. Decorators only fire when their module is
# imported, so the registry populates organically:
#
#   * ``--task=X --help`` and any ``apply()`` validation call
#     :meth:`PresetCli._collect_task_options`, which loads X's env config; the env
#     config imports the backends X uses, the decorators fire, the
#     registry is populated.
#   * ``--<flag>=<name>`` validation runs after that same path.
#
# A bare ``--help`` with no ``--task`` may show only the backends already
# imported by the current process (often none). That is the honest
# behaviour: there is no other list to query, only what's loaded.


class PresetCli:
    """Argparse helper that exposes preset selection as typed CLI flags.

    The class owns:

    # Review(jichuanh): try to avoid explicit mentions. I really want to make it general and a framework.
    * the three flag definitions (``--physics``, ``--renderer``,
      ``--presets``) registered on a parser via :meth:`add_args`;
    * the parsed-args → ``presets=<csv>`` translation in :meth:`apply` and
      :meth:`commit`;
    * the target taxonomy and per-task enumeration used by both the help
      action and apply-time validation;
    * the cross-env vocabulary lint :meth:`validate_preset_cfg`;
    * a one-line script wrapper :meth:`setup` (also exposed as the
      module-level alias :func:`setup_cli`).
    """

    # Review(jichuanh):  this might also be redundant. can just check on the special type PRESETS from enum definition.
    # Domain has the special CLI flag name ``--presets`` (catch-all CSV);
    # other targets use ``--{target.value}``. Ordering: physics first, renderer
    # second, then everything else, with domain last so ``--presets`` reads
    # naturally at the bottom of ``--help`` listings.
    _DOMAIN_FLAG: ClassVar[str] = "--presets"
    # Review(jichuanh): why this is needed? should only be defined once in enum
    _TARGET_ORDER_HEAD: ClassVar[tuple[PresetTarget, ...]] = (PresetTarget.PHYSICS, PresetTarget.RENDERER)

    # -- public API: per-script wiring ------------------------------------

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        """Register one ``--{target}`` flag per registered :class:`PresetTarget` on *parser*.

        Domain gets the special name ``--presets`` (catch-all CSV); every
        other target gets ``--{target.value}``. Adding a new
        :class:`PresetTarget` enum member adds a new flag here
        automatically -- no edits to this method.

        Also installs a help-action wrapper that appends a per-task
        preset listing when ``--task`` is given.

        Idempotent on the same parser regardless of which ``PresetCli``
        instance calls it -- the parser carries a :class:`_ParserState`
        flag that ``setup_cli`` (which builds a fresh PresetCli per
        call) can run twice on the same parser without duplicating the
        flag list. The help extension is installed even when args are
        already present (a parser may have been wired without
        ``add_args`` first, e.g. through a different ``PresetCli`` flow).
        """
        # Two independent setup steps, each self-guarded by a flag in
        # :class:`_ParserState`. Run linearly: skip the body when the
        # flag is already set, otherwise do the work and stamp the flag.
        # No branching, no duplicated tail call.
        state = _parser_state(parser)
        if _ParserState.ARGS_ADDED not in state:
            group = parser.add_argument_group(
                "preset selection",
                description=(
                    "Select named PresetCfg alternatives across the task config. "
                    "All flags accept '--flag value' or '--flag=value'. "
                    "Use '--task=<task> --help' to list valid preset names for a task."
                ),
            )
            # Help text intentionally does NOT inline the canonical vocabulary:
            # the registry is populated lazily as backends are imported (via
            # env-config loads), so listing it here would be misleadingly
            # incomplete at parse-build time. Use ``--task=<task> --help`` to
            # see the names valid for a specific task.
            for target in self._ordered_targets():
                if target is PresetTarget.DOMAIN:
                    group.add_argument(
                        PresetCli._DOMAIN_FLAG,
                        type=str,
                        default=None,
                        metavar="NAME[,NAME...]",
                        help=(
                            "Comma-separated list of free-form domain preset names to apply globally"
                            " (e.g. 'albedo,inference'). Valid names depend on the task; use"
                            " '--task=<task> --help' to list."
                        ),
                    )
                else:
                    group.add_argument(
                        f"--{target.value}",
                        type=str,
                        default=None,
                        metavar="NAME",
                        help=(
                            f"{target.value.capitalize()} preset name. Valid names depend on the task;"
                            " use '--task=<task> --help' to list."
                        ),
                    )
            _set_parser_state(parser, _ParserState.ARGS_ADDED)

        # Always invoke; ``_install_help_extension`` is its own no-op when
        # the HELP_INSTALLED flag is already set.
        self._install_help_extension(parser)

    # Review(jichuanh): the name is not that obvious. I think each function needs to document what's the input and what's the output.
    def _requests_from_args(self, args: argparse.Namespace) -> list[_PresetRequest]:
        """Parse a typed-flag :class:`argparse.Namespace` into request records.

        One source of truth for the "walk every target, pluck the value off
        the namespace, normalize legacy aliases" loop that used to live
        in both :meth:`collect_selected` and :meth:`apply`.

        :class:`PresetTarget.DOMAIN` is special-cased: ``--presets`` is a
        comma-separated list and its values are free-form, so they are
        emitted as-is. Other kinds carry one canonical name each and are
        normalized against their target vocabulary so legacy spellings
        (e.g. ``--physics newton`` -> ``newton_mjwarp``) become canonical
        with a single deprecation warning.
        """
        requests: list[_PresetRequest] = []
        for target in self._ordered_targets():
            # Review(jichuanh): what's the point here? for predefined targets, their available 
            if target is PresetTarget.DOMAIN:
                raw = getattr(args, "presets", None)
                if raw:
                    requests.extend(
                        _PresetRequest(target, token) for token in (t.strip() for t in raw.split(",")) if token
                    )
            else:
                value = getattr(args, target.value, None)
                if value:
                    vocab = set(PresetCli._vocab_for_target(target))
                    value = PresetTarget.normalize_name(value, vocab, caller_file=__file__)
                    requests.append(_PresetRequest(target, value))
        return requests

    def collect_selected(self, args: argparse.Namespace) -> set[str]:
        """Return the union of preset names requested via every target flag."""
        return {req.name for req in self._requests_from_args(args)}

    def apply(self, args: argparse.Namespace, remaining_argv: list[str]) -> list[str]:
        """Translate parsed flags into a ``presets=<csv>`` token, validating per target.

        Two layers of validation, both raising the same ``SystemExit``
        shape:

        # Review(jichuanh): I really don't understand why those need explicit setup. if enum is defined, that should be all enough. Is there a specific reason for specific setup?
        1. Vocabulary check -- ``--physics`` / ``--renderer`` (and any
           future non-domain target) values must come from the canonical
           vocabulary registered via
           :func:`isaaclab.utils.presets.register`. ``--presets``
           values are accepted as-is (domain presets are free-form).
        2. Per-task check -- regardless of target, the chosen name must be
           defined as a field on at least one ``PresetCfg`` in the loaded
           env config.

        Legacy aliases (e.g. ``--physics newton`` for ``newton_mjwarp``)
        are normalized inside :meth:`_requests_from_args` before this
        method sees them, so users updating from older tutorials see a
        single deprecation warning rather than an "unknown" error.
        """
        # Pull typed-flag requests off the namespace.
        requested = self._requests_from_args(args)

        # Split *remaining_argv* into (a) any legacy ``presets=...`` tokens
        # and (b) everything else (Hydra-bound argv we leave alone).
        legacy: list[str] = []
        kept: list[str] = []
        for arg in remaining_argv:
            if arg.startswith("presets="):
                legacy.extend(t.strip() for t in arg[len("presets=") :].split(",") if t.strip())
            else:
                kept.append(arg)

        # Nothing preset-related to do → return argv unchanged.
        if not requested and not legacy:
            return list(remaining_argv)

        # Validate typed-flag requests against canonical vocab + the task.
        # Legacy ``presets=...`` tokens are validated later in
        # :meth:`apply_overrides`, so we don't re-validate them here.
        if requested:
            task_name = getattr(args, "task", None)
            if not task_name:
                raise SystemExit(
                    "error: --physics/--renderer/--presets require --task to validate against. "
                    "Pass '--task=<task-name>' alongside these flags."
                )
            task_options = PresetCli._collect_task_options(task_name)
            for req in requested:
                self._validate_one(req.target, req.name, task_name, task_options)

        # Merge typed-flag names + legacy names into one CSV broadcast.
        # Dedupe preserves first-occurrence order so the user sees a
        # predictable presets= token.
        names: list[str] = [req.name for req in requested] + legacy
        seen: set[str] = set()
        deduped = [n for n in names if not (n in seen or seen.add(n))]
        return [f"presets={','.join(deduped)}"] + kept

    def commit(self, args: argparse.Namespace, remaining_argv: list[str]) -> list[str]:
        """Apply preset translation and replace ``sys.argv`` with the result.

        Companion to :meth:`apply` for callers that have already inspected
        or mutated *remaining_argv* themselves. Hides the ``sys.argv``
        contract.
        """
        remaining_argv = self.apply(args, remaining_argv)
        sys.argv = [sys.argv[0]] + remaining_argv
        return remaining_argv

    @classmethod
    def setup(
        cls,
        parser: argparse.ArgumentParser,
        *,
        commit: bool = True,
    ) -> argparse.Namespace | tuple[argparse.Namespace, list[str], PresetCli]:
        """One-stop IsaacLab CLI setup -- collapses 5 lines of boilerplate into 1.

        Bundles every IsaacLab script's CLI wiring:

        1. Register ``--physics``, ``--renderer``, ``--presets`` flags via
           :meth:`add_args`.
        2. Register AppLauncher flags via
           :meth:`AppLauncher.add_app_launcher_args`.
        3. Call ``parser.parse_known_args()``.
        4. Translate preset flags into a ``presets=<csv>`` token and
           replace ``sys.argv`` with the leftover Hydra-style overrides.

        Most scripts use the one-line form::

            args_cli = setup_cli(parser)

        Pass ``commit=False`` when the script needs to inspect or mutate
        the leftover argv between parsing and the ``sys.argv`` rewrite::

            args_cli, remaining, preset_cli = setup_cli(parser, commit=False)
            # ... post-process remaining ...
            preset_cli.commit(args_cli, remaining)
        """
        # AppLauncher is task-agnostic; lazy-import so this module stays
        # usable without Isaac Sim being available at import time.
        from isaaclab.app import AppLauncher

        # External wiring first, then our preset flags. Order is
        # functionally interchangeable (preset flags live in their own
        # argparse group, AppLauncher in its own), but external-first
        # matches the mental model "set up the host environment, then
        # add my own concerns".
        AppLauncher.add_app_launcher_args(parser)
        preset_cli = cls()
        preset_cli.add_args(parser)
        args, remaining = parser.parse_known_args()
        if commit:
            preset_cli.commit(args, remaining)
            return args
        return args, remaining, preset_cli

    @staticmethod
    def describe_task(task_name: str) -> str:
        """Return a human-readable per-task preset listing.

        Lists each target on its own line with comma-separated names; switches
        to a bullet list when a target has too many to fit inline.
        """
        try:
            options = PresetCli._collect_task_options(task_name)
        except Exception as exc:
            return f"\nPreset listing unavailable: {type(exc).__name__}: {exc}\n"

        lines = [f"\nPresets defined for {task_name}:", ""]
        any_listed = False
        for target in PresetCli._ordered_targets():
            kind_options = options.get(target, {})
            visible = sorted(name for name in kind_options if name != "default")
            flag = PresetCli._flag_for_kind(target)
            if not visible:
                lines.append(f"  {flag:11s}  (none defined)")
                continue
            any_listed = True
            rendered = PresetCli._format_inline_or_bullet(visible, indent="                ")
            lines.append(f"  {flag:11s}  {rendered}")
        if not any_listed:
            lines.append("")
            lines.append("  (this task does not currently define any user-selectable presets)")
        lines.append("")
        return "\n".join(lines)

    # -- public API: argv categorization + override application -----------
    #
    # These two methods own the bare-``presets=foo`` and path-based
    # ``env.x.y=name`` resolution flow that the Hydra-decorator path uses.
    # PresetCli.apply (above) handles the typed-flag CLI surface; the two
    # surfaces meet at the same ``presets=<csv>`` token format that
    # parse_overrides categorizes.

    @staticmethod
    def parse_overrides(args: list[str], presets: dict) -> PresetOverrides:
        """Categorize Hydra-style argv tokens into a :class:`PresetOverrides`.

        Legacy preset aliases are normalized here once (against the
        per-section vocabulary for path selections, against the global
        vocabulary for the ``presets=`` broadcast). :meth:`apply_overrides`
        trusts its input -- normalizing again would emit duplicate
        deprecation warnings for the same legacy name.

        Args:
            args: Argv tokens (without script name) -- typically ``sys.argv[1:]``
                after :meth:`apply` has prepended its ``presets=<csv>`` token.
            presets: ``{"env": {"path": {"name": cfg}}, "agent": {...}}`` --
                the preset map produced by ``register_task``.
        """
        # Set of every section.path string that the env declares a
        # PresetCfg at (e.g. {"env.backend", "env.observations",
        # "agent.policy"}). Used below to recognize ``env.X.Y=name``
        # tokens vs. plain Hydra scalars.
        preset_paths = {f"{s}.{p}" if p else s for s, v in presets.items() for p in v}
        out = PresetOverrides()

        for arg in args:
            # No ``=`` ⇒ positional / flag argument; hand to Hydra unchanged.
            if "=" not in arg:
                out.hydra_args.append(arg)
                continue
            key, val = arg.split("=", 1)
            if key == "presets":
                # Global broadcast: ``presets=A,B``. Normalize each name
                # against the global vocabulary so legacy aliases warn
                # exactly once at this parse boundary.
                known_names = PresetCli._known_preset_names(presets)
                out.global_presets.extend(
                    PresetTarget.normalize_name(v.strip(), known_names, caller_file=__file__)
                    for v in val.split(",")
                    if v.strip()
                )
            elif key in preset_paths:
                # Path-targeted selection: ``env.backend=newton_mjwarp``.
                # ``key`` is exactly a known preset path; ``val`` is the
                # alternative name to swap in.
                sec, path = key.split(".", 1) if "." in key else (key, "")
                known_names = set(presets[sec][path])
                out.path_selections.append(
                    (sec, path, PresetTarget.normalize_name(val, known_names, caller_file=__file__))
                )
            elif any(key.startswith(pp + ".") for pp in preset_paths):
                # Scalar override under a preset path:
                # ``env.backend.dt=0.001`` — applied AFTER the preset
                # alternative is chosen, on top of the chosen instance.
                out.preset_scalars.append((key, val))
            else:
                # Plain Hydra scalar (``env.decimation=10``): pass through.
                out.hydra_args.append(arg)

        # Sort by depth so apply_overrides walks shallow paths first,
        # avoiding the case where a parent replacement makes a child
        # path unreachable mid-loop.
        out.path_selections.sort(key=lambda x: x[1].count("."))
        return out

    @staticmethod
    def apply_overrides(
        env_cfg,
        agent_cfg,
        hydra_cfg: dict,
        overrides: PresetOverrides,
        presets: dict,
    ):
        """Apply preset selections and scalar overrides with REPLACE semantics.

        Walks three buckets in order:

        1. Path-based selections (``env.backend=newton_mjwarp``) -- REPLACE
           the entire subtree at the path.
        2. Global broadcast (``presets=newton_mjwarp``) -- REPLACE every
           reachable :class:`PresetCfg` whose alternatives include the name.
        3. Scalar overrides (``env.backend.dt=0.001``) within preset paths.

        Conflicting global presets that target the same path with different
        values raise ``ValueError``.

        Names in *overrides* are expected to be already legacy-alias
        normalized (see :meth:`parse_overrides`); this method does not
        re-normalize.

        Returns:
            ``(env_cfg, agent_cfg)`` -- possibly replaced if a root-level
            :class:`PresetCfg` was resolved.
        """
        cfgs = {"env": env_cfg, "agent": agent_cfg}

        def _path_reachable(sec: str, path: str) -> bool:
            """True if ``cfgs[sec].<path>`` exists; False if any segment was
            already replaced by a preset that doesn't carry the rest of the
            path. Used in Phase 2 to skip child applications when the
            parent's chosen alternative invalidates the child's path."""
            if not path:
                return cfgs[sec] is not None
            obj = cfgs[sec]
            for part in path.split("."):
                try:
                    obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)
                except (AttributeError, TypeError, KeyError):
                    return False
                if obj is None:
                    return False
            return True

        # ------------------------------------------------------------------
        # Phase 1: build {full_path: (sec, path, name)} of selections to apply.
        #
        # Three sources merge into the same map, with priority:
        #   path_selections > global_presets > implicit "default"
        # ------------------------------------------------------------------

        # 1a) Path-targeted selections (env.X.Y=name) — explicit, highest priority.
        resolved: dict[str, tuple[str, str, str]] = {}
        for sec, path, name in overrides.path_selections:
            if path not in presets.get(sec, {}):
                raise ValueError(f"Unknown preset group: {sec}.{path}")
            if name not in presets[sec][path]:
                # Build a helpful error: list available names, and if the
                # bad name happens to be a known legacy alias, tell the
                # user what it was renamed to.
                avail = list(presets[sec][path].keys())
                hint = ""
                legacy = PresetTarget.find_legacy(name)
                if legacy is not None:
                    _kind, replacement = legacy
                    hint = (
                        f" '{name}' was renamed to '{replacement}'; this path does not declare '{replacement}' either."
                    )
                raise ValueError(f"Unknown preset '{name}' for {sec}.{path}. Available: {avail}.{hint}")
            full_path = f"{sec}.{path}" if path else sec
            resolved[full_path] = (sec, path, name)

        # 1b) Global broadcast (presets=A,B,...) — apply to every PresetCfg
        # whose alternatives include the name. Conflict if two names target
        # the same path with DIFFERENT alternatives (the user asked for
        # both, we can't pick).
        applied_by: dict[str, str] = {}
        for name in overrides.global_presets:
            for sec in ("env", "agent"):
                for path, path_presets in presets.get(sec, {}).items():
                    if name in path_presets:
                        full_path = f"{sec}.{path}" if path else sec
                        if full_path in applied_by:
                            prev_name = applied_by[full_path]
                            prev_val = path_presets[prev_name]
                            cur_val = path_presets[name]
                            if prev_val is not cur_val and prev_val != cur_val:
                                raise ValueError(
                                    f"Conflicting global presets: '{prev_name}' and '{name}' "
                                    f"both define preset for '{full_path}'"
                                )
                        else:
                            applied_by[full_path] = name
                        # ``setdefault`` preserves any path_selection (1a) winner.
                        resolved.setdefault(full_path, (sec, path, name))

        # 1c) Implicit "default" — every PresetCfg with a default field but
        # no explicit selection still needs resolving so the env config is
        # usable. ``setdefault`` again preserves earlier winners.
        for sec in ("env", "agent"):
            for path, path_presets in presets.get(sec, {}).items():
                if "default" in path_presets:
                    full_path = f"{sec}.{path}" if path else sec
                    resolved.setdefault(full_path, (sec, path, "default"))

        # ------------------------------------------------------------------
        # Phase 2: apply selections in depth order (shallowest first), so a
        # parent replacement doesn't invalidate the path we'd use to find a
        # child. _path_reachable handles the case where an earlier
        # replacement made a child path no longer exist.
        # ------------------------------------------------------------------
        for full_path in sorted(resolved, key=lambda fp: fp.count(".")):
            sec, path, name = resolved[full_path]
            if cfgs[sec] is not None and _path_reachable(sec, path):
                node = presets[sec][path][name]
                # Build the matching dict-form for hydra_cfg so OmegaConf
                # sees a plain dict instead of a configclass instance.
                node_dict = (
                    node.to_dict() if hasattr(node, "to_dict") else dict(node) if isinstance(node, Mapping) else node
                )
                if not path:
                    # Root-level PresetCfg: replace the whole section.
                    cfgs[sec], hydra_cfg[sec] = node, node_dict
                else:
                    PresetCli._setattr(cfgs[sec], path, node)
                    PresetCli._setattr(hydra_cfg, f"{sec}.{path}", node_dict)

        # ------------------------------------------------------------------
        # Phase 3: scalar overrides under preset paths (env.backend.dt=0.001).
        # Done last so the user can tune individual fields ON TOP of the
        # preset alternative chosen above. Applied to both the live cfg
        # tree and the hydra_cfg dict so Hydra and from_dict stay in sync.
        # ------------------------------------------------------------------
        for full_path, val_str in overrides.preset_scalars:
            sec = full_path.split(".", 1)[0]
            if sec not in cfgs:
                continue
            path = full_path[len(sec) + 1 :]
            if cfgs[sec] is not None:
                val = PresetCli._parse_val(val_str)
                PresetCli._setattr(cfgs[sec], path, val)
                PresetCli._setattr(hydra_cfg, full_path, val)

        return cfgs["env"], cfgs["agent"]

    # -- public API: CI lint ----------------------------------------------

    @staticmethod
    def validate_preset_cfg(preset_obj: Any) -> list[str]:
        """Loose canonical-naming check for a single :class:`PresetCfg`.

        Returns a list of error messages. The rule is: for each canonical
        name that any of the alternatives' values resolves to, the
        canonical name must be one of the field names. Variants beyond the
        canonical entry are free-form (so an env can define ``physx``,
        ``physx_high_fidelity`` and only ``physx`` is required to be
        present).

        Used by the CI lint and any caller that wants to validate authored
        :class:`PresetCfg` subclasses against the canonical vocabulary.
        """
        fields = preset_fields(preset_obj)
        canonical_to_fnames: dict[str, list[str]] = {}
        for fname, value in fields.items():
            canonical = PresetRegistry.name_of(value)
            if canonical is not None:
                canonical_to_fnames.setdefault(canonical, []).append(fname)

        errors: list[str] = []
        for canonical, fnames in canonical_to_fnames.items():
            if canonical not in fnames:
                errors.append(
                    f"{type(preset_obj).__name__}: alternative(s) ({', '.join(fnames)}) "
                    f"hold values of canonical type {canonical!r}, but no field is named "
                    f"{canonical!r}. Rename one of these fields to {canonical!r} so the "
                    f"canonical CLI form works for this task."
                )
        return errors

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _known_preset_names(presets: dict) -> set[str]:
        """Return every preset name declared in a ``register_task``-shaped dict."""
        return {name for section in presets.values() for fields in section.values() for name in fields}

    @staticmethod
    def _setattr(obj, path: str, val: Any) -> None:
        """Set a nested attribute or dict key (e.g. ``actions.arm_action.scale``)."""
        *parts, leaf = path.split(".")
        for p in parts:
            obj = obj[p] if isinstance(obj, Mapping) else getattr(obj, p)
        if isinstance(obj, dict):
            obj[leaf] = val
        else:
            setattr(obj, leaf, val)

    @staticmethod
    def _parse_val(s: str) -> Any:
        """Parse a CLI scalar string into a Python value (bool, None, int, float, str)."""
        if s.lower() in _LITERAL_MAP:
            return _LITERAL_MAP[s.lower()]
        try:
            return float(s) if "." in s else int(s)
        except ValueError:
            return s[1:-1] if len(s) >= 2 and s[0] in "\"'" and s[-1] in "\"'" else s

    @staticmethod
    def _format_unknown_presets_error(
        unknown: set[str],
        name_to_paths: dict[str, list[str]],
        max_paths: int = 5,
    ) -> str:
        """Build a readable error grouping unknown preset names by affected paths.

        When an unknown name matches a deprecated alias on any
        :class:`PresetTarget`, the message calls out the rename so users
        updating from older tutorials get an actionable hint.
        """
        fingerprint_to_names: dict[tuple[str, ...], list[str]] = {}
        for name, paths in name_to_paths.items():
            key = tuple(sorted(paths))
            fingerprint_to_names.setdefault(key, []).append(name)

        lines = [f"Unknown preset(s): {', '.join(sorted(unknown))}"]
        for name in sorted(unknown):
            legacy = PresetTarget.find_legacy(name)
            if legacy is not None:
                _kind, replacement = legacy
                lines.append(
                    f"  '{name}' was renamed to '{replacement}'; this task does not declare '{replacement}' either."
                )
        lines += ["", "Available presets (grouped by affected paths):", ""]
        for paths_tuple in sorted(fingerprint_to_names, key=lambda k: fingerprint_to_names[k][0]):
            names = sorted(fingerprint_to_names[paths_tuple])
            if len(names) <= 30:
                lines.append(f"  {', '.join(names)}")
            else:
                lines.append(f"  {', '.join(names[:25])}, ... ({len(names)} total)")
            shown = list(paths_tuple[:max_paths])
            for p in shown:
                lines.append(f"    -> {p}")
            remaining = len(paths_tuple) - len(shown)
            if remaining > 0:
                lines.append(f"    ... ({remaining} more)")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _target_of_class(target_cls: type) -> PresetTarget:
        """Return the :class:`PresetTarget` of *target_cls* via MRO walk.

        Delegates to :meth:`PresetRegistry.target_of_class`. The CLI keeps a
        wrapper here so test modules and call-sites that want to read the
        target from a class object don't need to know about ``PresetRegistry``.
        """
        return PresetRegistry.target_of_class(target_cls)

    @staticmethod
    def _target_of_value(value: Any) -> PresetTarget:
        """Same as :meth:`_target_of_class` but takes an instance."""
        return PresetRegistry.target_of(value)

    @staticmethod
    def _vocab_for_target(target: PresetTarget) -> list[str]:
        """Return the sorted canonical vocabulary for *target*.

        For :attr:`PresetTarget.DOMAIN` returns ``[]`` -- domain presets are
        intentionally free-form and have no canonical vocabulary.

        Delegates straight to :meth:`PresetRegistry.names_for_target`; the
        registry already stores ``(class, target)`` pairs so there's no need
        to rescan ``all_names()`` and re-classify each entry here.
        """
        if target is PresetTarget.DOMAIN:
            return []
        return PresetRegistry.names_for_target(target)

    @staticmethod
    def _format_inline_or_bullet(items: list[str], indent: str = "    ", max_width: int = 76) -> str:
        """Render *items* as either a comma-separated inline string or a bullet list.

        Inline is used when the joined string fits on one line; otherwise
        switches to one item per line.
        """
        if not items:
            return "(none)"
        inline = ", ".join(items)
        if len(inline) <= max_width:
            return inline
        return "\n" + "\n".join(f"{indent}· {n}" for n in items)

    @staticmethod
    def _extract_task_from_argv(argv: list[str]) -> str | None:
        """Best-effort scan of *argv* for ``--task=value`` / ``--task value``.

        Linear loop, single-key lookup, single call per ``--help``. Could
        be a one-shot dict comprehension but argparse hasn't parsed yet
        here so we can't use the standard machinery; the loop is the
        common practice for "peek at one argv key before parsing."
        """
        for i, token in enumerate(argv):
            if token == "--task" and i + 1 < len(argv):
                return argv[i + 1]
            if token.startswith("--task="):
                return token[len("--task=") :]
        return None

    @staticmethod
    def _collect_task_options(task_name: str) -> dict[PresetTarget, dict[str, list[str]]]:
        """Return all PresetCfg alternatives in *task_name*'s config, grouped by target.

        Shape: ``{PresetTarget: {alt_name: [paths_where_alt_appears]}}``.
        Used by both apply-time validation (does the user's ``--physics
        NAME`` exist for this task?) and the per-task ``--help`` listing.

        Loading the env config has the useful side effect of importing
        the task's backend cfg modules (``PhysxCfg``, ``MJWarpSolverCfg``,
        ...), which fires their ``@register`` decorators and populates
        the canonical registry organically -- no separate force-import
        list needed.
        """
        # Local import to avoid a top-level cycle: this module is imported
        # while ``isaaclab_tasks`` is still loading, so ``parse_cfg`` may
        # not be fully ready when ``add_args`` runs at script import.
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        env_cfg = load_cfg_from_registry(task_name.split(":")[-1], "env_cfg_entry_point")
        if isinstance(env_cfg, type):
            env_cfg = env_cfg()
        # Guard: yaml-only or trivially-typed entry points resolve to a
        # value we can't walk. Return empty results per target so the
        # help layer surfaces "no presets defined" instead of a TypeError.
        if not (hasattr(env_cfg, "__dataclass_fields__") or isinstance(env_cfg, dict)):
            return {target: {} for target in PresetTarget}

        result: dict[PresetTarget, dict[str, list[str]]] = {target: {} for target in PresetTarget}
        for path, alternatives in collect_presets(env_cfg).items():
            for name, value in alternatives.items():
                target = PresetCli._target_of_value(value)
                result[target].setdefault(name, []).append(path or "<root>")
        return result

    @classmethod
    def _ordered_targets(cls) -> list[PresetTarget]:
        """Return registered kinds in display order: physics, renderer, then any
        other kinds (sorted), then domain. Drives ``--help`` and error layout."""
        registered = PresetRegistry.all_targets()
        # Review(jichuanh): I think maybe just use enum order, so really avoid duplicated defines make information compact wherever necesssary. DOMAIN can always be last, with a comment in enum definition
        head = [k for k in cls._TARGET_ORDER_HEAD if k in registered]
        tail = sorted(
            (k for k in registered if k not in cls._TARGET_ORDER_HEAD and k is not PresetTarget.DOMAIN),
            key=lambda k: k.value,
        )
        ordered = head + tail
        if PresetTarget.DOMAIN in registered:
            ordered.append(PresetTarget.DOMAIN)
        return ordered

    @staticmethod
    def _flag_for_kind(target: PresetTarget) -> str:
        """Return the argparse flag name for *target*."""
        return PresetCli._DOMAIN_FLAG if target is PresetTarget.DOMAIN else f"--{target.value}"

    def _validate_one(
        self,
        target: PresetTarget,
        name: str,
        task_name: str,
        task_options: dict[PresetTarget, dict[str, list[str]]],
    ) -> None:
        """Validate a single (target, name) request, raising SystemExit on failure."""
        kind_task = task_options.get(target, {})
        if target is not PresetTarget.DOMAIN:
            vocab = PresetCli._vocab_for_target(target)
            if name not in vocab:
                raise SystemExit(self._format_unknown_canonical(target, name, task_name, kind_task, task_options))
        if name not in kind_task:
            raise SystemExit(self._format_not_in_task(target, name, task_name, kind_task, task_options))

    @staticmethod
    def _format_unknown_canonical(
        target: PresetTarget,
        name: str,
        task_name: str,
        kind_task: dict[str, list[str]],
        task_options: dict[PresetTarget, dict[str, list[str]]],
    ) -> str:
        flag = PresetCli._flag_for_kind(target)
        vocab = PresetCli._vocab_for_target(target)
        wrong_kind = [k for k in task_options if k is not target and name in task_options.get(k, {})]

        lines = [f"error: {flag} {name!r} is not a recognized {target.value} preset name in IsaacLab."]
        if wrong_kind:
            other_flag = PresetCli._flag_for_kind(wrong_kind[0])
            lines.append("")
            lines.append(
                f"  '{name}' is defined as a {wrong_kind[0].value} preset on task {task_name!r}; "
                f"did you mean '{other_flag} {name}'?"
            )
        lines.append("")
        lines.append(f"  Recognized {target.value} presets across IsaacLab:")
        lines.append(f"    {PresetCli._format_inline_or_bullet(vocab, indent='    ')}")
        if kind_task:
            visible = sorted(n for n in kind_task if n != "default")
            if visible:
                lines.append("")
                lines.append(f"  This task ({task_name!r}) currently defines:")
                lines.append(f"    {PresetCli._format_inline_or_bullet(visible, indent='    ')}")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _format_not_in_task(
        target: PresetTarget,
        name: str,
        task_name: str,
        kind_task: dict[str, list[str]],
        task_options: dict[PresetTarget, dict[str, list[str]]],
    ) -> str:
        flag = PresetCli._flag_for_kind(target)
        wrong_kind = [k for k in task_options if k is not target and name in task_options.get(k, {})]

        lines = [f"error: {flag} {name!r} is not defined for task {task_name!r}."]
        if wrong_kind:
            other_flag = PresetCli._flag_for_kind(wrong_kind[0])
            lines.append("")
            lines.append(
                f"  '{name}' is defined as a {wrong_kind[0].value} preset on this task; "
                f"did you mean '{other_flag} {name}'?"
            )
        if kind_task:
            visible = sorted(n for n in kind_task if n != "default")
            if visible:
                lines.append("")
                lines.append("  This task currently defines:")
                lines.append(f"    {flag}:  {PresetCli._format_inline_or_bullet(visible, indent='    ')}")
        else:
            lines.append("")
            lines.append(f"  This task does not define any {target.value} presets.")
            lines.append("  See docs/source/features/hydra.rst#backend-and-solver-presets for how to add one.")

        if target is not PresetTarget.DOMAIN:
            vocab = PresetCli._vocab_for_target(target)
            lines.append("")
            lines.append(f"  Recognized {target.value} presets across IsaacLab (any task may add):")
            lines.append(f"    {PresetCli._format_inline_or_bullet(vocab, indent='    ')}")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _install_help_extension(parser: argparse.ArgumentParser) -> None:
        """Replace the parser's ``-h``/``--help`` with one that lists per-task presets.

        Why subclass argparse's ``_HelpAction`` instead of using a
        regular flag like ``--list-presets``: argparse exposes no public
        hook for "run extra logic when --help fires." Replacing the
        existing HelpAction with a subclass is the canonical workaround.
        Considered alternatives:

        * A separate ``--list-presets`` flag — rejected because users
          expect ``--task=X --help`` to list the task's presets without
          remembering a second flag.
        * Modifying the parser's epilog string — rejected because the
          listing is per-task and must run *after* ``--task`` is parsed.

        Idempotent on the same parser via :class:`_ParserState`.
        """
        state = _parser_state(parser)
        if _ParserState.HELP_INSTALLED in state:
            return
        help_action_cls = PresetCli._HelpAction
        for idx, action in enumerate(parser._actions):
            # Find argparse's default _HelpAction (if any) and swap it.
            if isinstance(action, argparse._HelpAction) and not isinstance(action, help_action_cls):
                replacement = help_action_cls(
                    option_strings=list(action.option_strings),
                    dest=action.dest,
                    default=action.default,
                    help=action.help,
                )
                parser._actions[idx] = replacement
                # Rewire option strings so --help / -h now route to the new action.
                for opt in action.option_strings:
                    parser._option_string_actions[opt] = replacement
                _set_parser_state(parser, _ParserState.HELP_INSTALLED)
                return

    class _HelpAction(argparse._HelpAction):
        """Help action that appends a per-task preset listing to ``--help``.

        Nested under :class:`PresetCli` because it has no public identity
        of its own -- it only matters as the replacement installed by
        :meth:`_install_help_extension`. Per ``AGENTS.md``: prefer nested
        classes when a helper is self-contained.
        """

        def __call__(self, parser, namespace, values, option_string=None):  # type: ignore[override]
            parser.print_help()
            task = PresetCli._extract_task_from_argv(sys.argv[1:])
            if task:
                # describe_task -> _collect_task_options -> load_cfg_from_registry,
                # which transitively imports the task's backends. After
                # that call the registry is populated for those backends.
                print(PresetCli.describe_task(task))
            else:
                print(
                    "\nPreset selection: use '--task=<task> --help' to list valid"
                    " --physics / --renderer / --presets names for a specific task.\n"
                )
            parser.exit()


# Module-level alias for the canonical script entry point. Implementation
# lives on :meth:`PresetCli.setup`; this alias keeps the natural one-line
# call site (``args_cli = setup_cli(parser)``) without exposing additional
# free functions.
setup_cli = PresetCli.setup
"""Alias for :meth:`PresetCli.setup` -- the one-line script wrapper."""
