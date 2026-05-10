# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generic preset primitives -- value type, target taxonomy, registry.

Three things live in this module:

* :class:`PresetTarget` -- the closed enum of CLI-facing preset categories.
  Each member carries its label (used as the ``--{label}`` flag and in
  error messages) and a per-target dict of *legacy aliases* (deprecated
  preset names that map to canonical ones of this target).

* :class:`PresetRegistry` -- maps a canonical name to the
  ``(class, PresetTarget)`` pair via the :meth:`PresetRegistry.register`
  decorator. The module-level :func:`register` alias keeps the natural
  ``@register(PresetTarget.PHYSICS, "physx")`` decorator spelling.

* :class:`PresetCfg` -- the @configclass base for declarative preset
  definitions. Subclass it and declare alternatives as fields; the
  ``default`` field is the no-CLI fallback. Tree-walking helpers
  (:func:`walk_presets`, :func:`collect_presets`, :func:`resolve_presets`)
  operate on entire env-config trees.

The CLI surface (:class:`~isaaclab_tasks.utils.preset_cli.PresetCli` and
its argv plumbing) lives in tasks because it depends on gym task lookup
and backend force-imports. This module is dependency-light by design.

A backend cfg class declares its membership once::

    from isaaclab.utils.presets import PresetTarget, register


    @register(PresetTarget.PHYSICS, "physx")
    @configclass
    class PhysxCfg(PhysicsCfg): ...

For the Newton solver case (one outer :class:`NewtonCfg` shared across
multiple preset names), decorate the **distinguishing solver** classes
(``MJWarpSolverCfg``, ``KaminoSolverCfg``); :meth:`PresetRegistry.target_of`
and :meth:`PresetRegistry.name_of` dispatch on ``solver_cfg`` so a
``NewtonCfg(solver_cfg=MJWarpSolverCfg())`` resolves to
``(PresetTarget.PHYSICS, "newton_mjwarp")``.
"""

from __future__ import annotations

import sys
import warnings
from collections.abc import Callable
from enum import Enum
from typing import Any, ClassVar

from isaaclab.utils.configclass import configclass


def _user_stacklevel(extra_skip: tuple[str, ...] = ()) -> int:
    """Compute a ``warnings.warn`` stacklevel that lands on user code.

    Walks frames upward, skipping any whose module ``__file__`` is either
    this file or any path in *extra_skip*. The first frame outside that
    set is treated as the user's code -- the warning attributes there
    instead of to internal plumbing.

    Callers in other modules pass their own ``__file__`` in
    ``extra_skip`` so warnings emitted from chained helpers correctly
    skip every layer of the chain.
    """
    skip = set(extra_skip) | {__file__}
    max_walk = 16
    level = 1
    frame = sys._getframe(1)
    while frame is not None and frame.f_globals.get("__file__") in skip:
        level += 1
        frame = frame.f_back
        if level > max_walk:
            return 2
    return level


# ============================================================================
# PresetTarget: the closed enum of CLI-facing categories.
# ============================================================================


class PresetTarget(Enum):
    """The set of CLI-facing preset categories.

    Each member is constructed from ``(label, legacy_aliases)``:

    * ``label`` -- the lowercase enum value, used as the ``--{label}`` flag
      and in error messages. Accessible via ``target.value``.
    * ``legacy_aliases`` -- mapping of deprecated preset names -> canonical
      replacements of this target. Accessible via ``target.legacy_aliases``.

    Adding a new category = appending one enum member; the CLI layer
    auto-discovers it via :meth:`PresetRegistry.all_targets`.
    """

    def __new__(cls, value: str, legacy_aliases: dict[str, str] | None = None):
        # MUST be defined before member declarations: enum members are
        # constructed at class-body parse time via this __new__, so moving
        # it below the members breaks instantiation.
        obj = object.__new__(cls)
        obj._value_ = value
        obj.legacy_aliases = dict(legacy_aliases) if legacy_aliases else {}
        return obj

    # Member values are tuples: (label_for_CLI_flag, {deprecated_alias: canonical_replacement}).
    # The second element is optional -- members with no deprecations omit it
    # (e.g. ``RENDERER = ("renderer",)``).
    PHYSICS = ("physics", {"newton": "newton_mjwarp", "kamino": "newton_kamino"})
    """Physics backends -- ``--physics`` flag. Legacy: ``newton``, ``kamino``."""

    RENDERER = ("renderer",)
    """Camera-sensor renderers -- ``--renderer`` flag."""

    DOMAIN = ("domain",)
    """Free-form env-specific presets -- ``--presets`` flag (catch-all)."""

    def resolve_alias(self, name: str) -> str | None:
        """Return the canonical replacement for legacy alias *name* in this target.

        Returns ``None`` when *name* is not a legacy alias of this target.
        Does not emit a warning -- callers decide whether the rename is
        usable in context (e.g. CLI normalize only warns if the
        replacement is actually defined for the current task).
        """
        return self.legacy_aliases.get(name)

    @classmethod
    def find_legacy(cls, name: str) -> tuple[PresetTarget, str] | None:
        """Search every target's legacy aliases for *name*.

        Returns ``(target, replacement)`` if *name* is deprecated under any
        target; ``None`` otherwise. Does not emit a warning.
        """
        for target in cls:
            replacement = target.resolve_alias(name)
            if replacement is not None:
                return (target, replacement)
        return None

    @classmethod
    def normalize_name(cls, name: str, known_names: set[str], *, caller_file: str | None = None) -> str:
        """Resolve a possibly-deprecated *name* against *known_names*.

        Single source of truth for the "if a legacy alias resolves to a
        known canonical, substitute and warn" pattern. Returns *name*
        unchanged when:

        * *name* is not a legacy alias, or
        * the canonical replacement isn't declared in *known_names* (so
          the unknown-preset error path can surface the real problem), or
        * *name* itself is a real entry in *known_names* (a redefined
          preset shadows the alias).

        ``caller_file`` is added to the warning's stacklevel skip-set so
        the FutureWarning attributes to user code rather than to the
        intermediate parse / resolution layer that called this method.
        """
        legacy = cls.find_legacy(name)
        if legacy is None:
            return name
        _kind, replacement = legacy
        if replacement not in known_names or name in known_names:
            return name
        extra: tuple[str, ...] = (caller_file,) if caller_file else ()
        warnings.warn(
            f"Preset '{name}' is deprecated. Use '{replacement}' instead.",
            FutureWarning,
            stacklevel=_user_stacklevel(extra),
        )
        return replacement


# ============================================================================
# PresetRegistry: canonical name -> (class, PresetTarget).
# ============================================================================


class PresetRegistry:
    """Maps canonical preset name to ``(class, PresetTarget)``.

    All access goes through classmethods. The registry is populated by
    the :meth:`register` decorator at import time of decorated cfg
    modules; the module-level :func:`register` alias keeps the natural
    ``@register(PresetTarget.PHYSICS, "physx")`` form one identifier.
    """

    _entries: ClassVar[dict[str, tuple[type, PresetTarget]]] = {}

    @classmethod
    def register(cls, target: PresetTarget, canonical_name: str):
        """Decorator binding ``(target, canonical_name)`` to a config class.

        Use exactly one decoration per backend config class. The decorator
        attaches both ``cls._preset_name`` and ``cls._preset_kind`` so
        :meth:`name_of` / :meth:`target_of` resolve via attribute lookup
        (with MRO fallback) without needing the registry at lookup time.

        Args:
            target: One of the :class:`PresetTarget` members.
            canonical_name: Name as it appears in CLI flags and PresetCfg
                field declarations (lowercase snake_case by convention).

        Returns:
            A class decorator that returns its input unchanged apart from
            the two attached attributes.

        Raises:
            RuntimeError: If *canonical_name* is already bound to a
                different class, or if the class already carries a
                different canonical name in its own ``__dict__``.
        """
        if not isinstance(target, PresetTarget):
            raise TypeError(
                f"register({target!r}, {canonical_name!r}): target must be a PresetTarget member, got {type(target).__name__}."
            )

        def deco(target_cls: type) -> type:
            # Three integrity checks before mutating registry / class.
            # All three reject silently-incorrect re-registrations that
            # would otherwise leave the registry in a confusing state.

            # (1) Same name → must be the same class. Two classes claiming
            #     one canonical name (e.g. two PhysxCfg variants) is the
            #     classic supplier collision; surface the conflict.
            existing = cls._entries.get(canonical_name)
            if existing is not None:
                existing_cls, existing_kind = existing
                if existing_cls is not target_cls:
                    raise RuntimeError(
                        f"@register(..., {canonical_name!r}) is already bound to "
                        f"{existing_cls.__module__}.{existing_cls.__name__}; cannot rebind to "
                        f"{target_cls.__module__}.{target_cls.__name__}."
                    )
                # (2) Same class, same name, different target. Silent
                #     re-decoration would mutate _preset_kind, breaking
                #     apply()-time target classification.
                if existing_kind is not target:
                    raise RuntimeError(
                        f"@register({target!r}, {canonical_name!r}) cannot re-register "
                        f"{target_cls.__module__}.{target_cls.__name__}: it is already "
                        f"registered under {existing_kind!r}. A class can only carry one target."
                    )
            # (3) The class itself was decorated before with a different
            #     name (could happen if the class is decorated twice in
            #     the same module). Use ``__dict__`` (not getattr) so
            #     inherited values from a base class don't trigger this.
            prior_name = target_cls.__dict__.get("_preset_name")
            if prior_name is not None and prior_name != canonical_name:
                raise RuntimeError(
                    f"@register(..., {canonical_name!r}) cannot rebind class "
                    f"{target_cls.__module__}.{target_cls.__name__}: it is already "
                    f"bound to {prior_name!r}. A class may carry only one canonical name."
                )

            # Bind: registry entry + class-level attrs for fast MRO lookup.
            cls._entries[canonical_name] = (target_cls, target)
            target_cls._preset_name = canonical_name  # type: ignore[attr-defined]
            target_cls._preset_kind = target  # type: ignore[attr-defined]
            return target_cls

        return deco

    # -- lookups -----------------------------------------------------------
    #
    # The four lookup methods share one MRO-walk helper. Each method that
    # takes a value falls back to ``value.solver_cfg`` so the Newton case
    # (NewtonCfg has no decoration; its inner solver_cfg does) resolves
    # through the same path.

    @staticmethod
    def _attr_via_mro(target_cls: type, attr: str) -> Any | None:
        """Return ``target_cls.__dict__[attr]`` for the first class in the
        MRO where it is set, or ``None`` if no class has it."""
        for klass in target_cls.__mro__:
            if attr in klass.__dict__:
                return klass.__dict__[attr]
        return None

    @classmethod
    def name_of(cls, value: Any) -> str | None:
        """Return the canonical preset name for *value*, or ``None``.

        Walks ``type(value).__mro__`` for ``_preset_name``; falls back to
        dispatching on ``value.solver_cfg`` (NewtonCfg case).
        """
        name = cls._attr_via_mro(type(value), "_preset_name")
        if name is not None:
            return name
        solver = getattr(value, "solver_cfg", None)
        return cls.name_of(solver) if solver is not None else None

    @classmethod
    def target_of(cls, value: Any) -> PresetTarget:
        """Return the :class:`PresetTarget` of *value*, or :attr:`PresetTarget.DOMAIN`.

        Walks ``type(value).__mro__`` for ``_preset_kind``; falls back to
        dispatching on ``value.solver_cfg``.
        """
        target = cls._attr_via_mro(type(value), "_preset_kind")
        if target is not None:
            return target
        solver = getattr(value, "solver_cfg", None)
        return cls.target_of(solver) if solver is not None else PresetTarget.DOMAIN

    @classmethod
    def target_of_class(cls, target_cls: type) -> PresetTarget:
        """Same as :meth:`target_of` but takes a class (no solver dispatch)."""
        target = cls._attr_via_mro(target_cls, "_preset_kind")
        return target if target is not None else PresetTarget.DOMAIN

    @classmethod
    def class_for_name(cls, canonical_name: str) -> type | None:
        """Return the class bound to *canonical_name*, or ``None``."""
        entry = cls._entries.get(canonical_name)
        return entry[0] if entry is not None else None

    # -- enumeration -------------------------------------------------------

    @classmethod
    def all_names(cls) -> list[str]:
        """Sorted list of canonical names registered so far."""
        return sorted(cls._entries)

    @classmethod
    def names_for_target(cls, target: PresetTarget) -> list[str]:
        """Sorted canonical names whose registered class is of *target*."""
        return sorted(name for name, (_klass, k) in cls._entries.items() if k == target)

    @classmethod
    def all_targets(cls) -> set[PresetTarget]:
        """Return every :class:`PresetTarget` member.

        Exposing the full enum (rather than only kinds with registered
        classes) honors the contract that adding a new ``PresetTarget``
        member is sufficient to surface its ``--{label}`` CLI flag --
        the flag appears with an empty canonical vocabulary until a
        class is decorated with that target.
        """
        return set(PresetTarget)


# Decorator alias kept at module level so the natural
# ``@register(PresetTarget.PHYSICS, "physx")`` usage stays one identifier.
register = PresetRegistry.register
"""Decorator alias for :meth:`PresetRegistry.register`."""


# ============================================================================
# PresetCfg: the @configclass base class for declarative presets.
# ============================================================================


@configclass
class PresetCfg:
    """Base class for declarative preset definitions.

    Subclass this and define fields as preset options. The field named
    ``default`` holds the config instance used when no CLI override is
    given; all other fields are named alternative presets.

    Example::

        @configclass
        class PhysicsCfg(PresetCfg):
            default: PhysxCfg = PhysxCfg()
            newton_mjwarp: NewtonCfg = NewtonCfg(...)

    The preset *name* is decoupled from the config class: a class describes
    a backend (``NewtonCfg``); the field name labels which solver variant
    this entry selects (``newton_mjwarp``). Canonical (target, name) bindings
    on the underlying classes are declared via :func:`register`.

    Tree-walking helpers (:func:`preset_fields`, :func:`walk_presets`,
    :func:`collect_presets`, :func:`resolve_presets`) live at module
    level — they have no class dependency, so they're free functions
    whose placement signals their scope: "general preset utilities, not
    tied to one PresetCfg instance."

    The factory (:meth:`PresetCfg.factory`) stays on the class because it
    creates new subclasses (uses ``cls``); the module-level
    :func:`preset` alias keeps the natural one-line DSL form for env code.
    """

    def __getattr__(self, name: str):
        """Alias a deprecated preset name to its replacement field.

        Delegates to :meth:`PresetTarget.normalize_name`, which owns the
        deprecation map (per-target ``legacy_aliases``) and the warning
        emission. This channel only substitutes when the replacement is
        a real field on the subclass *and* the deprecated name is not
        (so a user redefining the deprecated name shadows the alias).

        Raises ``AttributeError`` for any other missing attribute so that
        ``hasattr`` and standard introspection keep working unchanged.
        """
        own_fields: dict = getattr(type(self), "__dataclass_fields__", {})
        replacement = PresetTarget.normalize_name(name, set(own_fields), caller_file=__file__)
        if replacement != name and replacement in own_fields:
            return getattr(self, replacement)
        raise AttributeError(f"{type(self).__name__!s} object has no attribute {name!r}")

    @classmethod
    def factory(cls, **options) -> PresetCfg:
        """Build a :class:`PresetCfg` subclass on the fly with one field per kwarg.

        ``options`` must include a ``default`` key. The returned instance is
        a fresh anonymous subclass so its ``__dataclass_fields__`` don't
        collide with siblings.

        Example::

            armature = PresetCfg.factory(default=0.0, newton_mjwarp=0.01)

        Module-level :func:`preset` is an alias for this method.

        Raises:
            ValueError: If ``default`` is not provided.
        """
        if "default" not in options:
            raise ValueError("PresetCfg.factory(...) requires a 'default' keyword argument.")
        annotations = {k: type(v) if v is not None else object for k, v in options.items()}
        ns = {"__annotations__": annotations, **options}
        # Build a fresh subclass so this preset's fields don't pollute siblings.
        anon = configclass(type("_Preset", (cls,), ns))
        return anon()


# Module-level alias for the factory: ``preset(default=..., foo=...)`` is
# the canonical one-line DSL used by env code, so we keep that exact
# spelling at module level. The implementation lives on :meth:`PresetCfg.factory`.
preset = PresetCfg.factory
"""Alias for :meth:`PresetCfg.factory` -- the one-line preset DSL."""


# ============================================================================
# Tree-walking utilities for PresetCfg trees.
#
# These four helpers (preset_fields, walk_presets, collect_presets,
# resolve_presets) are module-level free functions, not methods on
# PresetCfg, because none of them depend on class state -- they take a
# config object and walk it. Placement signals scope: any caller can use
# them on any config tree without the "is this PresetCfg machinery"
# question.
# ============================================================================


def preset_fields(preset_obj: PresetCfg) -> dict:
    """Extract every alternative declared on *preset_obj*.

    Returns a ``{field_name: value}`` dict where class-level values win
    over instance-level. Robot-specific env modules (e.g.
    ``joint_pos_env_cfg.py``) reassign preset alternatives on the class
    after the instance was already constructed; reading from the class
    first picks up those overrides.

    Two passes:

    * **Pass 1 — declared dataclass fields.** Iterate
      ``__dataclass_fields__`` so the result respects the user's
      declaration order; pick class value if non-None, else instance.
    * **Pass 2 — extra class attrs.** Any non-private, non-method, non-
      dataclass attr added to the class is also a valid alternative
      (subclasses sometimes assign extras without redeclaring).
    """
    target_cls = type(preset_obj)
    result: dict = {}

    # Pass 1: declared dataclass fields, class-attr-wins-over-instance.
    for field_name in preset_obj.__dataclass_fields__:
        class_value = getattr(target_cls, field_name, None)
        result[field_name] = class_value if class_value is not None else getattr(preset_obj, field_name)

    # Pass 2: extra class-level attrs (skip dunders, already-seen, methods).
    for attr_name in vars(target_cls):
        if attr_name.startswith("_") or attr_name in result or callable(getattr(target_cls, attr_name)):
            continue
        result[attr_name] = getattr(target_cls, attr_name)
    return result


def walk_presets(cfg, path: str, on_preset: Callable) -> None:
    """Depth-first walk of *cfg*, invoking ``on_preset(parent, key, obj, path)``
    at every :class:`PresetCfg` node.

    Recurses through dataclass attributes, dicts, and nested dicts
    transparently, so callers don't have to special-case structure.
    """
    items = (
        cfg.items()
        if isinstance(cfg, dict)
        else ((n, v) for n in dir(cfg) if not n.startswith("_") for v in [getattr(cfg, n, None)] if v is not None)
    )
    for key, val in items:
        child_path = f"{path}.{key}" if path else key
        if isinstance(val, PresetCfg):
            on_preset(cfg, key, val, child_path)
        elif hasattr(val, "__dataclass_fields__") or isinstance(val, dict):
            walk_presets(val, child_path, on_preset)


def collect_presets(cfg, path: str = "") -> dict:
    """Discover every :class:`PresetCfg` node in *cfg* and return a flat map.

    Returns:
        ``{dotted_path: {alt_name: value}}``, e.g.
        ``{"backend": {"default": PhysxCfg(), "newton_mjwarp": NewtonCfg()}}``.

    Walks dataclass fields and dict values at any depth; recurses into
    each alternative because nested PresetCfg-inside-PresetCfg is valid.
    """
    result: dict = {}

    def _record(preset_obj, preset_path):
        fields = preset_fields(preset_obj)
        result[preset_path] = fields
        # Recurse into each alternative: it may itself be a configclass or
        # dict containing more PresetCfg nodes (the nested preset case).
        for alt in fields.values():
            if hasattr(alt, "__dataclass_fields__"):
                result.update(collect_presets(alt, preset_path))
            elif isinstance(alt, dict):
                for v in alt.values():
                    if hasattr(v, "__dataclass_fields__"):
                        result.update(collect_presets(v, preset_path))

    if isinstance(cfg, PresetCfg):
        _record(cfg, path)
        return result

    walk_presets(cfg, path, lambda _p, _k, obj, cp: _record(obj, cp))
    return result


def _pick_preset(preset_obj: PresetCfg, selected: set[str], path: str = ""):
    """Choose the best alternative from *preset_obj*.

    Priority: first name in *selected* that resolves to a real field
    (after legacy-alias normalization), else the ``default`` field.
    Class-level values win over instance-level via :func:`preset_fields`.

    Raises:
        ValueError: If nothing in *selected* matches and no ``default`` exists.
    """
    fields_dict = preset_fields(preset_obj)
    field_names = set(fields_dict)
    for name in selected:
        # Legacy alias may map to a real field; substitute and emit a
        # single deprecation warning if so.
        name = PresetTarget.normalize_name(name, field_names, caller_file=__file__)
        if name in fields_dict:
            return fields_dict[name]
    if "default" in fields_dict:
        return fields_dict["default"]
    raise ValueError(
        f"PresetCfg {type(preset_obj).__name__} at '{path}' has no 'default' field "
        f"and none of the selected presets {selected} match its fields {set(fields_dict.keys())}."
    )


def resolve_presets(cfg, selected: set[str] = frozenset()):
    """Replace every :class:`PresetCfg` in the tree with the chosen alternative.

    For each PresetCfg node:

    1. Pick the first name from *selected* that exists as a field
       (via :func:`_pick_preset`), or fall back to ``default``.
    2. Replace the node in its parent (dict key or dataclass attr).
    3. Continue walking the replacement; it may itself be a PresetCfg
       (nested presets) or contain PresetCfg nodes deeper down.

    Cyclic chains (a default that points at another unresolved preset)
    raise ``ValueError`` rather than looping forever.

    Args:
        cfg: A configclass instance, dict, or :class:`PresetCfg`.
        selected: User-chosen preset names (e.g. ``{"newton_mjwarp"}``).

    Returns:
        The resolved ``cfg`` — a different object if the root itself
        was a ``PresetCfg`` (the root is replaced wholesale in that case).
    """
    if isinstance(cfg, PresetCfg):
        # Root is itself a preset: pick its alternative, follow chains
        # of preset-of-preset, then recurse into the final value.
        seen: set[int] = {id(cfg)}
        replacement = _pick_preset(cfg, selected, path="<root>")
        while isinstance(replacement, PresetCfg):
            if id(replacement) in seen:
                raise ValueError(
                    f"Cyclic PresetCfg chain detected at '<root>': "
                    f"{type(replacement).__name__} was already visited."
                )
            seen.add(id(replacement))
            replacement = _pick_preset(replacement, selected, path="<root>")
        return resolve_presets(replacement, selected)

    def _resolve(parent, key, preset_obj, _path):
        seen: set[int] = {id(preset_obj)}
        val = _pick_preset(preset_obj, selected, path=_path)
        while isinstance(val, PresetCfg):
            if id(val) in seen:
                raise ValueError(
                    f"Cyclic PresetCfg chain detected at '{_path}': {type(val).__name__} was already visited."
                )
            seen.add(id(val))
            val = _pick_preset(val, selected, path=_path)
        if isinstance(parent, dict):
            parent[key] = val
        else:
            setattr(parent, key, val)
        # The replacement may itself contain nested PresetCfg nodes; keep walking.
        if hasattr(val, "__dataclass_fields__") or isinstance(val, dict):
            walk_presets(val, _path, _resolve)

    walk_presets(cfg, "", _resolve)
    return cfg
