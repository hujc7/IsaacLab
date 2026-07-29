Added
^^^^^

* Added manager-based counterparts for the Shadow handover and Shadow camera
  reorientation tasks, completing the manager coverage of the dexterous task
  families.
* Added a Direct-versus-manager value-parity check for the handover task,
  alongside the reorientation one.

Changed
^^^^^^^

* Changed the manager-based Shadow camera task to run on PhysX by default. The
  RTX render modalities require Fabric cloning, which Newton does not support,
  so the task could not render with the inherited default. Select Newton with
  ``physics=newton_mjwarp`` for the state-only observation groups.
* Changed the handover reward to a plain reward term, moving success and
  goal-distance bookkeeping to
  :class:`~isaaclab_tasks.core.handover.mdp.commands.HandoverCommand`, which
  owns the goal.
* Changed the reorientation action configuration to name its term through a
  module path, so loading a task configuration no longer imports the USD
  bindings.

Fixed
^^^^^

* Fixed the Shadow camera feature-extractor observation term ignoring its
  declared ``feature_extractor_cfg`` parameter and reading the environment
  configuration instead.
