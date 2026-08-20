# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Private MuJoCo tendon control adapter for the Newton backend.

Newton exposes MuJoCo tendons and their actuator controls as model-global arrays spanning every
world, and its :class:`~newton.selection.ArticulationView` carries a layout for the tendons but not
for the actuators driving them. This module performs the one-time join from an articulation's
fixed-tendon IDs to those global control rows, and owns the per-step write that carries buffered
targets into ``mujoco.ctrl``.

:class:`MjcTendonControl` holds the articulation it drives, so it reuses that asset's index
resolution and shape checking rather than restating them -- the same arrangement
:class:`~isaaclab_newton.assets.articulation.actuator_control.NewtonActuatorControl` uses for
actuators. The command buffer itself stays on
:class:`~isaaclab_newton.assets.ArticulationData`, beside the other fixed-tendon buffers.

Nothing here is exported; :mod:`isaaclab_newton.assets.articulation.articulation` is the only
importer. That is what lets the whole module be deleted once Newton exposes the mapping itself.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp
from newton.solvers import SolverMuJoCo

from isaaclab_newton.assets import kernels as shared_kernels

from . import kernels as articulation_kernels

if TYPE_CHECKING:
    from newton import Control, Model
    from newton.selection import ArticulationView

    from .articulation import Articulation

logger = logging.getLogger(__name__)

_GLOBAL_ACTUATOR_WORLD = -1
"""``mujoco:actuator_world`` value scoping an actuator to every world.

This is also the attribute's default, so a tendon in world N is driven either by an actuator in
world N or by one carrying this sentinel."""


class MjcTendonControl:
    """Drives an articulation's fixed tendons through MuJoCo's native tendon actuators.

    Created by :meth:`~isaaclab_newton.assets.Articulation._process_tendons` when the model carries
    at least one directly-actuated fixed tendon. Command tendons through the articulation's
    backend-neutral
    :meth:`~isaaclab.assets.articulation.BaseArticulation.set_fixed_tendon_position_target_index`
    rather than through this internal adapter.
    """

    def __init__(self, articulation: Articulation, control_rows: np.ndarray):
        """Bind a resolved control-row mapping to the articulation it drives.

        Args:
            articulation: Newton articulation owning the fixed tendons.
            control_rows: MuJoCo ``ctrl`` row per fixed tendon, ``-1`` where the tendon has no
                direct actuator. Shape is ``(num_instances, num_fixed_tendons)``.
        """
        self._articulation = articulation
        self._control_rows = wp.array(control_rows, dtype=wp.int32, device=articulation.device)
        # Name the passive tendons once here rather than per command: the IDs reaching
        # :meth:`set_position_target` are a device array, so checking them there would synchronize
        # the GPU on every step.
        passive = [
            name
            for name, commandable in zip(articulation.fixed_tendon_names, control_rows[0] >= 0, strict=True)
            if not commandable
        ]
        if passive:
            logger.warning(
                "Fixed tendons %s have no direct MuJoCo position actuator, so commanding them has no effect."
                " Author an actuator whose transmission is each tendon to drive them.",
                passive,
            )

    def set_position_target(
        self,
        *,
        target: float | torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Buffer fixed-tendon position targets in the articulation's fixed-tendon index space.

        This does not reach the simulation; :meth:`write_data_to_sim` carries the buffer into
        ``mujoco.ctrl`` on the articulation's usual per-step write.

        Args:
            target: Target tendon length [m or rad, depending on the spanned joints' type].
                Shape is ``(len(env_ids), len(fixed_tendon_ids))``.
            fixed_tendon_ids: The tendon indices to command. Defaults to None (all fixed tendons).
            env_ids: Environment indices. If None, then all indices are used.
        """
        articulation = self._articulation
        env_ids = articulation._resolve_env_ids(env_ids)
        fixed_tendon_ids = articulation._resolve_fixed_tendon_ids(fixed_tendon_ids)
        articulation.assert_shape_and_dtype(target, (env_ids.shape[0], fixed_tendon_ids.shape[0]), wp.float32, "target")
        # Warp kernels can ingest torch tensors directly, so we don't need to convert to warp arrays here.
        if isinstance(target, float):
            kernel = articulation_kernels.float_data_to_buffer_with_indices_kernel(env_ids, fixed_tendon_ids)
        else:
            kernel = shared_kernels.write_2d_data_to_buffer_with_indices_kernel(env_ids, fixed_tendon_ids)
        wp.launch(
            kernel,
            dim=(env_ids.shape[0], fixed_tendon_ids.shape[0]),
            inputs=[target, env_ids, fixed_tendon_ids],
            outputs=[articulation.data._fixed_tendon_position_target],
            device=articulation.device,
        )

    def write_data_to_sim(self, control: Control) -> None:
        """Scatter the buffered targets into Newton's MuJoCo control array.

        A tendon with no direct actuator carries row ``-1`` and is skipped, so commanding one is a
        no-op rather than a stray write; :meth:`__init__` already named it.

        Args:
            control: Newton control carrying ``mujoco.ctrl``.

        Raises:
            RuntimeError: If the control carries no ``mujoco.ctrl`` array. Newton attaches the
                namespaced container dynamically, so its absence cannot be caught by typing.
        """
        mujoco_control = getattr(control, "mujoco", None)
        ctrl = getattr(mujoco_control, "ctrl", None) if mujoco_control is not None else None
        if ctrl is None:
            raise RuntimeError("Newton control does not contain the 'mujoco.ctrl' array required by tendon actuators.")
        position_target = self._articulation.data._fixed_tendon_position_target
        wp.launch(
            _scatter_fixed_tendon_position_targets,
            dim=position_target.shape,
            inputs=[position_target, self._control_rows],
            outputs=[ctrl],
            device=position_target.device,
        )


def resolve_fixed_tendon_control_rows(root_view: ArticulationView, model: Model) -> np.ndarray | None:
    """Resolve fixed-tendon IDs to MuJoCo ``ctrl`` rows for a selected articulation.

    The result is indexed in the articulation's complete fixed-tendon ID space -- the same space
    :meth:`~isaaclab_newton.assets.Articulation.find_fixed_tendons` returns -- rather than in a
    filtered list of actuated tendons, so a caller's tendon IDs mean the same thing here as
    everywhere else. ``-1`` marks a fixed tendon with no direct MuJoCo position actuator.

    Args:
        root_view: Newton selection view for one articulation.
        model: Newton model carrying the MuJoCo custom attributes.

    Returns:
        Per-instance MuJoCo control rows, shape ``(num_instances, num_fixed_tendons)``, dtype
        ``int32``; or ``None`` when the model carries no MuJoCo fixed-tendon attributes, which is
        not an error -- a model built without them simply has nothing to resolve.

    Raises:
        ValueError: If more than one direct actuator targets a tendon the articulation holds, or if
            instances disagree on which tendons are commandable.
    """
    mujoco = getattr(model, "mujoco", None)
    tendon_layout = root_view.frequency_layouts.get("mujoco:tendon")
    required_attributes = (
        "tendon_label",
        "tendon_world",
        "actuator_target_label",
        "actuator_world",
        "actuator_trntype",
        "ctrl_source",
    )
    if tendon_layout is None or mujoco is None or any(not hasattr(mujoco, name) for name in required_attributes):
        return None

    tendon_labels = [str(label) for label in mujoco.tendon_label]
    tendon_worlds = _to_numpy(mujoco.tendon_world)
    actuator_rows_by_target, ambiguous_targets = _index_direct_tendon_actuators(mujoco)
    local_tendon_ids = _local_tendon_ids(tendon_layout)
    control_rows = np.full((root_view.count, len(local_tendon_ids)), -1, dtype=np.int32)

    instance_id = 0
    for world_slot in range(root_view.world_count):
        for articulation_slot in range(root_view.count_per_world):
            tendon_rows = (
                tendon_layout.offset
                + world_slot * tendon_layout.stride_between_worlds
                + articulation_slot * tendon_layout.stride_within_worlds
                + local_tendon_ids
            )
            for tendon_id, tendon_row in enumerate(tendon_rows):
                tendon_label = tendon_labels[tendon_row]
                # An actuator in the tendon's own world takes precedence over a world-agnostic one.
                target_keys = (
                    (int(tendon_worlds[tendon_row]), tendon_label),
                    (_GLOBAL_ACTUATOR_WORLD, tendon_label),
                )
                ambiguous_key = next((key for key in target_keys if key in ambiguous_targets), None)
                if ambiguous_key is not None:
                    raise ValueError(
                        f"Multiple direct MuJoCo tendon actuators target '{tendon_label}' in world {ambiguous_key[0]}."
                    )
                control_rows[instance_id, tendon_id] = next(
                    (actuator_rows_by_target[key] for key in target_keys if key in actuator_rows_by_target),
                    -1,
                )
            instance_id += 1

    # One index space serves every instance, so instances that disagree cannot share a buffer. The
    # rows themselves differ between worlds by construction; only commandability must match.
    commandable = control_rows >= 0
    if not np.all(commandable == commandable[0]):
        raise ValueError("MuJoCo direct tendon actuators differ between articulation instances.")
    return control_rows


def _index_direct_tendon_actuators(mujoco) -> tuple[dict[tuple[int, str], int], set[tuple[int, str]]]:
    """Index the directly-controlled tendon actuators by ``(world, target label)``.

    Indexing once is what makes start-up affordable: the ``mujoco:*`` arrays span every world, so
    scanning them per tendon is quadratic in the environment count.

    Returns:
        The actuator row for each target, and the targets claimed by more than one actuator, which
        the caller rejects only if a tendon it holds actually names one.
    """
    actuator_rows_by_target: dict[tuple[int, str], int] = {}
    ambiguous_targets: set[tuple[int, str]] = set()
    actuator_worlds = _to_numpy(mujoco.actuator_world)
    actuator_trntypes = _to_numpy(mujoco.actuator_trntype)
    control_sources = _to_numpy(mujoco.ctrl_source)
    target_labels = [str(label) for label in mujoco.actuator_target_label]

    is_direct_tendon_actuator = (actuator_trntypes == int(SolverMuJoCo.TrnType.TENDON)) & (
        control_sources == int(SolverMuJoCo.CtrlSource.CTRL_DIRECT)
    )
    for actuator_row in np.flatnonzero(is_direct_tendon_actuator):
        actuator_row = int(actuator_row)
        target_key = (int(actuator_worlds[actuator_row]), target_labels[actuator_row])
        if target_key in actuator_rows_by_target:
            ambiguous_targets.add(target_key)
        else:
            actuator_rows_by_target[target_key] = actuator_row
    return actuator_rows_by_target, ambiguous_targets


def _local_tendon_ids(tendon_layout) -> np.ndarray:
    """Return the local fixed-tendon IDs a Newton frequency layout selects.

    The layout carries an explicit index list when its selection is sparse, and a slice otherwise.
    """
    if tendon_layout.indices is not None:
        return _to_numpy(tendon_layout.indices).astype(np.int64, copy=False)
    return np.arange(tendon_layout.slice.start, tendon_layout.slice.stop, dtype=np.int64)


def _to_numpy(value) -> np.ndarray:
    """Convert a Warp array or array-like value to NumPy."""
    return value.numpy() if isinstance(value, wp.array) else np.asarray(value)


@wp.kernel
def _scatter_fixed_tendon_position_targets(
    position_target: wp.array2d(dtype=wp.float32),
    control_rows: wp.array2d(dtype=wp.int32),
    ctrl: wp.array(dtype=wp.float32),
) -> None:
    """Scatter buffered per-instance targets into the flat MuJoCo control array."""
    env_id, fixed_tendon_id = wp.tid()
    control_row = control_rows[env_id, fixed_tendon_id]
    if control_row >= 0:
        ctrl[control_row] = position_target[env_id, fixed_tendon_id]
