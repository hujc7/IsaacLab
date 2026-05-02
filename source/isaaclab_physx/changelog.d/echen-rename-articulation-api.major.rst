Removed
^^^^^^^

* Removed deprecated module ``isaaclab_physx.legacy_articulation`` (use :class:`~isaaclab_physx.assets.Articulation` instead).

Changed
^^^^^^^

* **Breaking:** :meth:`~isaaclab_physx.assets.Articulation.set_joint_state` now requires a per-env mask; passing a global tensor raises ``ValueError``. Use ``mask=slice(None)`` to preserve the prior behavior.
