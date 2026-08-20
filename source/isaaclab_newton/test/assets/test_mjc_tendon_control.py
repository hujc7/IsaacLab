# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import numpy as np
import torch
import warp as wp
from isaaclab_newton.assets.articulation.mjc_tendon_actuator import MjcTendonActuatorView


def _make_view() -> MjcTendonActuatorView:
    tendon_labels = ["/Robot/Physics/rh_FFJ0", "/Robot/Physics/rh_LFJ0"] * 2
    actuator_target_labels = ["/Robot/joint", *tendon_labels[:2], "/Robot/joint", *tendon_labels[2:]]
    mujoco = SimpleNamespace(
        tendon_label=tendon_labels,
        tendon_world=wp.array([0, 0, 1, 1], dtype=wp.int32, device="cpu"),
        actuator_target_label=actuator_target_labels,
        actuator_world=wp.array([0, 0, 0, 1, 1, 1], dtype=wp.int32, device="cpu"),
        actuator_trntype=wp.array([0, 2, 2, 0, 2, 2], dtype=wp.int32, device="cpu"),
        ctrl_source=wp.array([1, 1, 1, 1, 1, 1], dtype=wp.int32, device="cpu"),
        actuator_ctrlrange=wp.array(
            [(0.0, 0.0), (0.0, 3.14), (0.0, 2.5), (0.0, 0.0), (0.0, 3.14), (0.0, 2.5)],
            dtype=wp.vec2f,
            device="cpu",
        ),
        actuator_has_ctrlrange=wp.array([0, 1, 1, 0, 1, 1], dtype=wp.int32, device="cpu"),
    )
    tendon_layout = SimpleNamespace(
        offset=0,
        stride_between_worlds=2,
        stride_within_worlds=2,
        indices=None,
        slice=slice(0, 2),
    )
    root_view = SimpleNamespace(
        device=wp.get_device("cpu"),
        count=2,
        world_count=2,
        count_per_world=1,
        frequency_layouts={"mujoco:tendon": tendon_layout},
    )
    return MjcTendonActuatorView(root_view, SimpleNamespace(mujoco=mujoco))


def test_mjc_tendon_actuator_resolves_names_limits_and_world_rows():
    """Match same-named cloned tendon targets to control rows in their own worlds."""
    view = _make_view()

    assert view.names == ["rh_FFJ0", "rh_LFJ0"]
    assert view.num_actuators == 2
    torch.testing.assert_close(view.control_limits, torch.tensor([[0.0, 3.14], [0.0, 2.5]]))
    assert view.find_actuators(["rh_LFJ0", "rh_FFJ0"], preserve_order=True) == (
        [1, 0],
        ["rh_LFJ0", "rh_FFJ0"],
    )


def test_mjc_tendon_actuator_scatter_preserves_other_native_controls():
    """Scatter a selected buffered target without overwriting unrelated native controls."""
    view = _make_view()
    view.set_position_target_index(
        target=torch.tensor([[1.25]], dtype=torch.float32),
        actuator_ids=torch.tensor([0], dtype=torch.int64),
        env_ids=torch.tensor([1], dtype=torch.int64),
    )
    ctrl = wp.full(6, -7.0, dtype=wp.float32, device="cpu")
    view._write_data_to_sim(SimpleNamespace(mujoco=SimpleNamespace(ctrl=ctrl)))

    np.testing.assert_allclose(ctrl.numpy(), [-7.0, 0.0, 0.0, -7.0, 1.25, 0.0])
