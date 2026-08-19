# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

from isaaclab.utils.string import resolve_matching_names

from isaaclab_newton.assets import kernels as shared_kernels

if TYPE_CHECKING:
    from newton import Control, Model
    from newton.selection import ArticulationView


class MjcTendonActuatorView:
    """Buffered position-target view over MuJoCo-native fixed-tendon actuators.

    This Newton-specific view exposes fixed tendons driven by direct MuJoCo position actuators. Each actuator is
    matched to an articulation instance by both its world and its target tendon label. Targets are buffered until
    the owning articulation writes its data to the simulation.

    It exists because Newton's ``ArticulationView`` carries no tendon accessor of its own: MuJoCo's
    tendons and their actuators reach the model only as flat ``mujoco:*`` arrays spanning every world,
    with no per-articulation indexing. Resolving that indexing here is what lets
    :meth:`~isaaclab.assets.articulation.BaseArticulation.set_fixed_tendon_position_target_index` mean
    the same thing on Newton as it does on PhysX.

    Created and owned by :class:`isaaclab_newton.assets.Articulation`; command tendons through that
    backend-neutral method rather than through this view.
    """

    def __init__(self, root_view: ArticulationView, model: Model):
        """Initialize the view from a Newton articulation and model.

        Args:
            root_view: Newton articulation selection view.
            model: Newton simulation model containing MuJoCo custom attributes.
        """
        names, control_limits, control_indices = _resolve_mjc_tendon_actuators(root_view, model)

        self._device = root_view.device
        self._num_instances = root_view.count
        self._names = names
        self._control_limits = torch.tensor(control_limits, dtype=torch.float32, device=str(self._device))
        self._control_indices = wp.array(control_indices, dtype=wp.int32, device=self._device)
        self._position_target = wp.zeros(
            (self._num_instances, self.num_actuators), dtype=wp.float32, device=self._device
        )
        self._ALL_ENV_INDICES = wp.array(
            np.arange(self._num_instances, dtype=np.int32), dtype=wp.int32, device=self._device
        )
        self._ALL_ACTUATOR_INDICES = wp.array(
            np.arange(self.num_actuators, dtype=np.int32), dtype=wp.int32, device=self._device
        )

    @property
    def names(self) -> list[str]:
        """Ordered target tendon names for the native actuators."""
        return self._names

    @property
    def num_actuators(self) -> int:
        """Number of native tendon actuators per articulation instance."""
        return len(self._names)

    @property
    def control_limits(self) -> torch.Tensor:
        """Position-control limits [rad], shape ``(num_actuators, 2)``."""
        return self._control_limits

    def find_actuators(
        self, name_keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        """Find native tendon actuators by their target tendon names.

        Args:
            name_keys: Regular expression or list of regular expressions to match.
            preserve_order: Whether to preserve query-key order in the output.

        Returns:
            Matched actuator indices and names.
        """
        return resolve_matching_names(name_keys, self.names, preserve_order)

    def set_position_target_index(
        self,
        *,
        target: torch.Tensor | wp.array,
        actuator_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set buffered native tendon position targets using indices.

        Call :meth:`isaaclab_newton.assets.Articulation.write_data_to_sim` on the owning articulation to scatter
        the buffered targets into Newton's MuJoCo control array.

        Args:
            target: Position targets [rad], shape ``(len(env_ids), len(actuator_ids))``.
            actuator_ids: Native tendon actuator indices. Defaults to all actuators.
            env_ids: Articulation instance indices. Defaults to all instances.
        """
        env_ids = self._resolve_ids(env_ids, self._ALL_ENV_INDICES, "env_ids")
        actuator_ids = self._resolve_ids(actuator_ids, self._ALL_ACTUATOR_INDICES, "actuator_ids")
        expected_shape = (env_ids.shape[0], actuator_ids.shape[0])
        _assert_float32_shape(target, expected_shape, "target")

        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices_kernel(env_ids, actuator_ids),
            dim=expected_shape,
            inputs=[target, env_ids, actuator_ids],
            outputs=[self._position_target],
            device=self._device,
        )

    def _write_data_to_sim(self, control: Control) -> None:
        """Scatter buffered targets into the current Newton MuJoCo control array."""
        if self.num_actuators == 0:
            return
        mujoco_control = getattr(control, "mujoco", None)
        ctrl = getattr(mujoco_control, "ctrl", None) if mujoco_control is not None else None
        if ctrl is None:
            raise RuntimeError("Newton control does not contain the 'mujoco.ctrl' array required by tendon actuators.")
        wp.launch(
            _scatter_mjc_tendon_actuator_targets,
            dim=self._position_target.shape,
            inputs=[self._position_target, self._control_indices],
            outputs=[ctrl],
            device=self._device,
        )

    def _resolve_ids(
        self,
        ids: Sequence[int] | torch.Tensor | wp.array | None,
        all_ids: wp.array,
        name: str,
    ) -> torch.Tensor | wp.array:
        """Resolve an optional index selector to a Torch or Warp array."""
        if ids is None or (isinstance(ids, slice) and ids == slice(None)):
            return all_ids
        if isinstance(ids, Sequence) and not isinstance(ids, (str, bytes, torch.Tensor, wp.array)):
            return wp.array(ids, dtype=wp.int32, device=self._device)
        if isinstance(ids, (torch.Tensor, wp.array)):
            return ids
        raise TypeError(f"{name} must be a sequence, torch.Tensor, wp.array, or None, got {type(ids).__name__}.")


def _resolve_mjc_tendon_actuators(
    root_view: ArticulationView, model: Model
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Resolve native actuator metadata into per-instance control rows.

    MuJoCo's tendons and their actuators reach the model as flat ``mujoco:*`` arrays spanning every
    world, so pairing a tendon with the actuator that drives it, per articulation, is this module's
    whole job. It runs in four steps: read the arrays, index the eligible actuators, walk the
    instances pairing them up, then require every instance to agree.

    Returns:
        The tendon names in actuator order, the control limits ``(n_actuators, 2)``, and the control
        row each instance writes ``(n_instances, n_actuators)``. Empty when the model carries none.
    """
    attributes = _read_mjc_tendon_attributes(root_view, model)
    if attributes is None:
        return [], np.empty((0, 2), dtype=np.float32), np.empty((root_view.count, 0), dtype=np.int32)

    actuator_rows_by_target, ambiguous_targets = _index_actuators_by_target(attributes)
    instance_names, instance_limits, instance_control_rows = _collect_instance_actuator_rows(
        root_view, attributes, actuator_rows_by_target, ambiguous_targets
    )
    names, limits = _require_instances_agree(instance_names, instance_limits)
    return (
        names,
        np.asarray(limits, dtype=np.float32).reshape((-1, 2)),
        np.asarray(instance_control_rows, dtype=np.int32).reshape((root_view.count, len(names))),
    )


def _read_mjc_tendon_attributes(root_view: ArticulationView, model: Model) -> dict | None:
    """Read the MuJoCo custom attributes this resolution needs.

    Returns:
        The attribute arrays plus the tendon layout, or ``None`` when the model carries no MuJoCo
        tendons -- a model built without them is not an error, it simply has nothing to resolve.
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
        "actuator_ctrlrange",
        "actuator_has_ctrlrange",
    )
    if tendon_layout is None or mujoco is None or any(not hasattr(mujoco, name) for name in required_attributes):
        return None
    return {
        "layout": tendon_layout,
        "tendon_labels": [str(label) for label in mujoco.tendon_label],
        "tendon_worlds": _to_numpy(mujoco.tendon_world),
        "actuator_target_labels": [str(label) for label in mujoco.actuator_target_label],
        "actuator_worlds": _to_numpy(mujoco.actuator_world),
        "actuator_trntypes": _to_numpy(mujoco.actuator_trntype),
        "actuator_ctrl_sources": _to_numpy(mujoco.ctrl_source),
        "actuator_control_ranges": _to_numpy(mujoco.actuator_ctrlrange),
        "actuator_has_control_range": _to_numpy(mujoco.actuator_has_ctrlrange),
    }


def _index_actuators_by_target(attributes: dict) -> tuple[dict[tuple[int, str], int], set[tuple[int, str]]]:
    """Index the directly-controlled tendon actuators by ``(world, target label)``.

    Indexing once is what makes start-up affordable: scanning every actuator per tendon instead
    measured 174 s for one hand and 344 s per articulation view for two, against 0.1 s here, because
    the arrays span every environment.

    Returns:
        The actuator row for each target, and the targets claimed by more than one actuator, which
        the caller rejects only if a tendon actually names one.
    """
    trntypes = attributes["actuator_trntypes"]
    ctrl_sources = attributes["actuator_ctrl_sources"]
    worlds = attributes["actuator_worlds"]
    target_labels = attributes["actuator_target_labels"]

    actuator_rows_by_target: dict[tuple[int, str], int] = {}
    ambiguous_targets: set[tuple[int, str]] = set()
    # trntype 2 is a tendon transmission; ctrl_source 1 is a direct control input
    for actuator_row in np.flatnonzero((trntypes == 2) & (ctrl_sources == 1)):
        actuator_row = int(actuator_row)
        target_key = (int(worlds[actuator_row]), target_labels[actuator_row])
        if target_key in actuator_rows_by_target:
            ambiguous_targets.add(target_key)
        else:
            actuator_rows_by_target[target_key] = actuator_row
    return actuator_rows_by_target, ambiguous_targets


def _collect_instance_actuator_rows(
    root_view: ArticulationView,
    attributes: dict,
    actuator_rows_by_target: dict[tuple[int, str], int],
    ambiguous_targets: set[tuple[int, str]],
) -> tuple[list[list[str]], list[list[tuple[float, float]]], list[list[int]]]:
    """Pair each articulation instance's tendons with the actuators that drive them.

    Returns:
        Per instance: the tendon names, their control limits, and the control rows to write.

    Raises:
        ValueError: If more than one actuator targets a tendon this instance holds.
    """
    layout = attributes["layout"]
    tendon_labels = attributes["tendon_labels"]
    tendon_worlds = attributes["tendon_worlds"]
    control_ranges = attributes["actuator_control_ranges"]
    has_control_range = attributes["actuator_has_control_range"]

    # local indices the layout selects: an explicit list when it has one, else its slice
    if layout.indices is not None:
        local_tendon_ids = _to_numpy(layout.indices).astype(np.int64, copy=False)
    else:
        local_tendon_ids = np.arange(layout.slice.start, layout.slice.stop, dtype=np.int64)

    instance_names: list[list[str]] = []
    instance_limits: list[list[tuple[float, float]]] = []
    instance_control_rows: list[list[int]] = []
    for world_slot in range(root_view.world_count):
        for articulation_slot in range(root_view.count_per_world):
            tendon_rows = (
                layout.offset
                + world_slot * layout.stride_between_worlds
                + articulation_slot * layout.stride_within_worlds
                + local_tendon_ids
            )
            names: list[str] = []
            limits: list[tuple[float, float]] = []
            control_rows: list[int] = []
            for tendon_row in tendon_rows:
                target_label = tendon_labels[tendon_row]
                target_key = (int(tendon_worlds[tendon_row]), target_label)
                if target_key in ambiguous_targets:
                    raise ValueError(
                        f"Multiple direct MuJoCo tendon actuators target '{target_label}' in world {target_key[0]}."
                    )
                actuator_row = actuator_rows_by_target.get(target_key)
                if actuator_row is None:
                    continue
                names.append(target_label.rsplit("/", maxsplit=1)[-1])
                control_rows.append(actuator_row)
                if has_control_range[actuator_row]:
                    control_range = control_ranges[actuator_row]
                    limits.append((float(control_range[0]), float(control_range[1])))
                else:
                    limits.append((-float("inf"), float("inf")))
            instance_names.append(names)
            instance_limits.append(limits)
            instance_control_rows.append(control_rows)
    return instance_names, instance_limits, instance_control_rows


def _require_instances_agree(
    instance_names: list[list[str]], instance_limits: list[list[tuple[float, float]]]
) -> tuple[list[str], list[tuple[float, float]]]:
    """Require every articulation instance to expose the same actuators.

    One index space serves all instances, so instances that disagree cannot share a command buffer.

    Returns:
        The names and limits shared by every instance.

    Raises:
        ValueError: If any instance's names or limits differ from the first.
    """
    names = instance_names[0]
    limits = instance_limits[0]
    for instance_id, (other_names, other_limits) in enumerate(
        zip(instance_names[1:], instance_limits[1:], strict=True), start=1
    ):
        if other_names != names:
            raise ValueError(
                "MuJoCo direct tendon actuator names differ between articulation instances: "
                f"instance 0 has {names}, instance {instance_id} has {other_names}."
            )
        if not np.allclose(other_limits, limits):
            raise ValueError(
                "MuJoCo direct tendon actuator control limits differ between articulation instances: "
                f"instance 0 has {limits}, instance {instance_id} has {other_limits}."
            )
    return names, limits


def _to_numpy(value) -> np.ndarray:
    """Convert a Warp array or array-like value to a NumPy array."""
    return value.numpy() if isinstance(value, wp.array) else np.asarray(value)


def _assert_float32_shape(target: torch.Tensor | wp.array, shape: tuple[int, int], name: str) -> None:
    """Validate a floating-point target tensor's shape and dtype.

    Narrower than :meth:`~isaaclab.assets.AssetBase.assert_shape_and_dtype`, which does the same job
    for any dtype and rank. That one is an instance method on the asset, and this view is not an
    asset, so it cannot be reached from here.
    """
    if isinstance(target, torch.Tensor):
        if target.dtype != torch.float32:
            raise TypeError(f"{name} must have dtype torch.float32, got {target.dtype}.")
    elif isinstance(target, wp.array):
        if target.dtype != wp.float32:
            raise TypeError(f"{name} must have dtype wp.float32, got {target.dtype}.")
    else:
        raise TypeError(f"{name} must be a torch.Tensor or wp.array, got {type(target).__name__}.")
    if target.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {target.shape}.")


@wp.kernel
def _scatter_mjc_tendon_actuator_targets(
    position_target: wp.array2d(dtype=wp.float32),
    control_indices: wp.array2d(dtype=wp.int32),
    ctrl: wp.array(dtype=wp.float32),
) -> None:
    """Scatter buffered per-instance targets into the flat native MuJoCo control array."""
    env_id, actuator_id = wp.tid()
    ctrl[control_indices[env_id, actuator_id]] = position_target[env_id, actuator_id]
