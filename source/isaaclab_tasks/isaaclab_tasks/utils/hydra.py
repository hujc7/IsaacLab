# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hydra + gym integration for IsaacLab tasks.

What lives here: the ``hydra.main`` decorator wrapper
(:func:`hydra_task_config`), the eager-resolution helper
(:func:`resolve_task_config`), and :func:`register_task` -- which loads
env/agent configs, collects their preset alternatives, and registers a
config node into the Hydra ``ConfigStore`` so Hydra can apply global
scalar overrides on top.

What does NOT live here: the preset value type
(:class:`~isaaclab.utils.PresetCfg`), the target enum and registry
(:mod:`isaaclab.utils.presets`), and the CLI/argv plumbing
(:class:`~isaaclab_tasks.utils.preset_cli.PresetCli`,
:class:`~isaaclab_tasks.utils.preset_cli.PresetOverrides`,
:func:`~isaaclab_tasks.utils.preset_cli.PresetCli.parse_overrides`,
:func:`~isaaclab_tasks.utils.preset_cli.PresetCli.apply_overrides`).
Import those from their canonical modules.

Override categories (applied in order):
    1. Global presets: ``presets=inference,newton_mjwarp`` -- apply everywhere matching
    2. Path presets: ``env.backend=newton_mjwarp`` -- REPLACE specific section
    3. Preset-path scalars: ``env.backend.dt=0.001`` -- handled by us
    4. Global scalars: ``env.decimation=10`` -- handled by Hydra
"""

import functools
import sys
from collections.abc import Callable

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from isaaclab.envs.utils.spaces import replace_env_cfg_spaces_with_strings, replace_strings_with_env_cfg_spaces
from isaaclab.utils import replace_slices_with_strings, replace_strings_with_slices
from isaaclab.utils.presets import PresetTarget, collect_presets, resolve_presets

from isaaclab_tasks.utils.preset_cli import PresetCli

__all__ = ["hydra_task_config", "resolve_task_config", "register_task"]


# ============================================================================
# CLI / Hydra integration
# ============================================================================


def _run_hydra(task, env_cfg, agent_cfg, presets, callback):
    """Shared Hydra entry point for :func:`resolve_task_config` and :func:`hydra_task_config`.

    Two argv shapes coexist on the same command line: preset-related
    tokens (``presets=...``, ``env.X.Y=name``, ``--physics=name``, etc.)
    that we own, and Hydra-style scalar overrides
    (``env.decimation=10``) that Hydra owns. We split them, hide ours
    from Hydra by overwriting ``sys.argv``, run Hydra's main, and apply
    our preset overrides inside Hydra's callback before yielding to the
    user's main.
    """
    # Split sys.argv into preset-bound and Hydra-bound buckets.
    overrides = PresetCli.parse_overrides(sys.argv[1:], presets)
    # Hide preset tokens from Hydra: it would error on unrecognized keys.
    original_argv, sys.argv = sys.argv, [sys.argv[0]] + overrides.hydra_args

    @hydra.main(config_path=None, config_name=task, version_base="1.3")
    def hydra_main(hydra_cfg, env_cfg=env_cfg, agent_cfg=agent_cfg):
        # OmegaConf gives us a DictConfig; convert to plain dict so we
        # can mutate freely. ``replace_strings_with_slices`` undoes the
        # slice-to-string serialization done in register_task.
        hydra_cfg = replace_strings_with_slices(OmegaConf.to_container(hydra_cfg, resolve=True))
        # Apply our preset overrides ON TOP of Hydra's parsed config.
        env_cfg, agent_cfg = PresetCli.apply_overrides(env_cfg, agent_cfg, hydra_cfg, overrides, presets)
        # Round-trip env_cfg through the dict form so any field the user
        # set via Hydra scalar override is reflected on the live cfg
        # object the callback receives.
        env_cfg.from_dict(hydra_cfg["env"])
        # Restore gym Space objects (mirror of register_task's serialize step).
        env_cfg = replace_strings_with_env_cfg_spaces(env_cfg)
        if isinstance(agent_cfg, dict) or agent_cfg is None:
            agent_cfg = hydra_cfg["agent"]
        else:
            agent_cfg.from_dict(hydra_cfg["agent"])
        callback(env_cfg, agent_cfg)

    try:
        hydra_main()
    finally:
        # Restore argv even if hydra_main raised, so the surrounding
        # process state isn't left mutated.
        sys.argv = original_argv


def resolve_task_config(task_name: str, agent_cfg_entry_point: str):
    """Resolve env and agent configs with Hydra overrides, presets, and scalars fully applied.

    Safe to call before Kit is launched -- callable config values are stored as
    :class:`~isaaclab.utils.string.ResolvableString` and resolved lazily on
    first use, so no implementation modules are imported eagerly.

    Args:
        task_name: Task name (e.g., "Isaac-Velocity-Flat-Anymal-C-v0").
        agent_cfg_entry_point: Agent config entry point key (e.g., "rsl_rl_cfg_entry_point").

    Returns:
        Tuple of (env_cfg, agent_cfg) fully resolved.
    """
    task = task_name.split(":")[-1]
    env_cfg, agent_cfg, presets = register_task(task, agent_cfg_entry_point)
    resolved = {}
    _run_hydra(task, env_cfg, agent_cfg, presets, lambda e, a: resolved.update(env_cfg=e, agent_cfg=a))
    return resolved["env_cfg"], resolved["agent_cfg"]


def hydra_task_config(task_name: str, agent_cfg_entry_point: str) -> Callable:
    """Decorator for Hydra config with REPLACE-only preset semantics.

    Args:
        task_name: Task name (e.g., "Isaac-Reach-Franka-v0")
        agent_cfg_entry_point: Agent config entry point key

    Returns:
        Decorated function receiving ``(env_cfg, agent_cfg, *args, **kwargs)``
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            task = task_name.split(":")[-1]
            env_cfg, agent_cfg, presets = register_task(task, agent_cfg_entry_point)
            _run_hydra(task, env_cfg, agent_cfg, presets, lambda e, a: func(e, a, *args, **kwargs))

        return wrapper

    return decorator


def register_task(task_name: str, agent_entry: str) -> tuple:
    """Load configs, collect presets recursively, register base config to Hydra.

    Five steps:

    1. Load env + agent configs from the gym registry.
    2. Walk both trees with :func:`collect_presets` to find every
       :class:`PresetCfg` node (path → alternatives map).
    3. Scan ``sys.argv`` for the legacy ``presets=A,B`` form and
       fail-fast if any selected name doesn't exist on this task. This
       runs BEFORE ``parse_overrides`` so users see a useful error
       (with available names + rename hints) before Hydra is involved.
    4. Resolve PresetCfg wrappers to their picked alternatives in the
       live env/agent cfgs AND inside the collected alternatives map
       (otherwise apply_overrides could re-introduce unresolved nodes).
    5. Convert env/agent to plain dicts and register into Hydra's
       ConfigStore so Hydra can apply global scalar overrides on top.

    Presets are tracked in a separate dict (returned alongside the
    cfgs), NOT as Hydra groups, because Hydra's group merge would
    deep-merge alternatives instead of swapping them outright.

    Returns:
        ``(env_cfg, agent_cfg, presets)`` where presets =
        ``{"env": {"path": {"name": cfg}}, "agent": {...}}``.
    """
    # Local import: parse_cfg pulls in isaaclab_tasks which would form a
    # cycle if imported at module top.
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # 1) Load.
    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(task_name, agent_entry) if agent_entry else None

    # 2) Collect every PresetCfg node BEFORE resolution -- otherwise the
    # path-based override path can't find the alternatives it needs to
    # swap in.
    presets = {
        "env": collect_presets(env_cfg),
        "agent": collect_presets(agent_cfg) if agent_cfg else {},
    }

    # 3) Pre-scan sys.argv for ``presets=...`` so we can surface unknown
    # names with the full per-task suggestion list (apply_overrides
    # would also catch this later, but the error there is per-token, not
    # the consolidated "here are all names available for this task" view).
    known_names = PresetCli._known_preset_names(presets)
    selected = {
        # Normalize legacy aliases once at parse time; consumers trust input.
        PresetTarget.normalize_name(v.strip(), known_names, caller_file=__file__)
        for arg in sys.argv[1:]
        if "=" in arg
        for key, val in [arg.split("=", 1)]
        if key.lstrip("-") == "presets"
        for v in val.split(",")
        if v.strip()
    }

    if selected:
        # Build {name → [affected paths]} so the error message can group
        # unknown names by the paths they would have hit.
        name_to_paths: dict[str, list[str]] = {}
        for sec, sec_presets in presets.items():
            for path, fields in sec_presets.items():
                full = f"{sec}.{path}" if path else sec
                for name in fields:
                    name_to_paths.setdefault(name, []).append(full)
        unknown = selected - set(name_to_paths)
        if unknown:
            # Hide "default" from the suggestion list; users don't pick it.
            display = {n: p for n, p in name_to_paths.items() if n != "default"}
            raise ValueError(PresetCli._format_unknown_presets_error(unknown, display))

    # 4) Resolve PresetCfg → concrete alternative in the live cfgs.
    env_cfg = resolve_presets(env_cfg, selected)
    if agent_cfg is not None:
        agent_cfg = resolve_presets(agent_cfg, selected)

    # ...AND inside each collected alternative, so that if apply_overrides
    # later picks one of these alternatives it doesn't re-introduce an
    # unresolved PresetCfg into the cfg tree.
    for section_presets in presets.values():
        for path_presets in section_presets.values():
            for name, alt in path_presets.items():
                resolve_presets(alt, selected)

    # 5) Convert to plain dicts. ``replace_env_cfg_spaces_with_strings``
    # serializes gym Space objects (Hydra/OmegaConf can't hold them);
    # ``replace_slices_with_strings`` does the same for slice() literals.
    env_cfg = replace_env_cfg_spaces_with_strings(env_cfg)
    agent_dict = agent_cfg.to_dict() if agent_cfg is not None and hasattr(agent_cfg, "to_dict") else agent_cfg
    env_dict = env_cfg.to_dict()  # type: ignore[union-attr]
    cfg_dict = replace_slices_with_strings({"env": env_dict, "agent": agent_dict})

    # Register a plain config (no groups). Hydra's job from here is just
    # global scalar overrides (--decimation=10) -- preset selection is
    # ours, applied via apply_overrides in _run_hydra.
    ConfigStore.instance().store(name=task_name, node=OmegaConf.create(cfg_dict))
    return env_cfg, agent_cfg, presets
