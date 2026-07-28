# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for resolving SimReady catalogue assets into task-ready objects.

The USD-Search service answers *"which assets look like X?"*. It can pre-filter server-side on one
bounding-box axis and on validation features, but what it returns per match is only the asset path,
version, relevance score, source, and tags -- no dimensions, no mass, no per-asset validation
verdict. Every property a manipulation task needs to decide *"can this robot pick this up?"*
therefore requires opening the asset itself.

This sub-module is that missing layer. It searches broadly, opens each candidate once to record the
facts the service does not expose, keeps the objects a given robot can actually handle, and writes
them out in the one shape heterogeneous cloning requires.

Authored masses are treated as ground truth: a canned good genuinely is heavy, so an object is
dropped only when the robot could not lift it, never because the number looks surprising.

**The whole pipeline runs without Kit.** Search is plain HTTP; assets are mirrored over HTTPS with
their full layer closure and opened as local files. Stock OpenUSD cannot open an ``https://`` URL,
and importing ``omni.client`` does not help (its resolver only activates inside the Kit runtime) --
but neither is required once the layers are local. That removes the container/Kit dependency from
asset discovery, auditing, and preparation entirely.

Opening assets is the expensive step (each is fetched from remote storage), so every audit is cached
to JSON. The first resolve of a catalogue costs minutes; later ones are instant, which is what makes
run-time resolution practical inside a training script.

Usage:

.. code-block:: python

    from isaaclab.utils.simready import SimReadyObjectLibrary, SimReadyObjectLibraryCfg

    # describe what the robot can handle, not what the assets should be
    cfg = SimReadyObjectLibraryCfg()
    cfg.object_filter.size_range = (0.02, 0.12)
    cfg.object_filter.mass_range = (0.005, 1.0)

    # search -> audit -> filter -> write task-ready USDs
    usd_paths = SimReadyObjectLibrary(cfg).resolve(num_objects=12)

"""

from isaaclab.utils.module import lazy_export

lazy_export()
