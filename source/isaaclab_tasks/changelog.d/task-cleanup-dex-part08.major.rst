Added
^^^^^

* Added manager-based counterparts for the Shadow cube reorientation task and
  its OpenAI FF/LSTM observation variants, alongside the existing Allegro
  manager task.
* Added :class:`~isaaclab_tasks.core.reorient.mdp.reorient_timeout`, which
  restarts the episode timer on every goal reach so OpenAI-variant episodes
  extend across success streaks.
* Added ``enable_domain_randomization`` to the manager-based Allegro
  environment.
* Added Newton and OvPhysx physics presets to the manager-based reorientation
  environments, selectable with ``physics=``.
* Added a Direct-versus-manager value-parity check covering timing, success
  tolerance, fall distance, and the consecutive-success cap.

Changed
^^^^^^^

* **Breaking:** Changed the manager-based Allegro reorientation environment to
  match the Direct observation, action, reset, and termination contracts. The
  observation space changes size, so existing manager checkpoints cannot be
  loaded and must be retrained.
* Changed ``enable_domain_randomization`` to default to ``False`` so the manager
  and Direct Allegro tasks are comparable out of the box. Enable it by appending
  ``env.enable_domain_randomization=true`` to the training command.
* Renamed the per-robot scene constants to name what they hold: ``ROBOT_CFG``
  becomes ``SHADOW_HAND_ROBOT_CFG`` or ``ALLEGRO_HAND_ROBOT_CFG``,
  ``OBJECT_CFG`` becomes ``CUBE_CFG``, and ``ObjectCfg`` becomes ``CubeCfg``.

Removed
^^^^^^^

* Removed ``ReorientObjectEnvCfg`` and the shared reorientation observation,
  action, and command configurations. Each manager task now declares its own;
  derive from :class:`~isaaclab.envs.ManagerBasedRLEnvCfg` directly.
* Removed ``reorient_common``. Its constants are declared by the tasks that use
  them, and the in-hand offset and goal-marker position are now per-robot fields
  on the Direct configurations.

Fixed
^^^^^

* Fixed the manager-based reorientation tasks not reporting
  ``Metrics/success_rate``.
* Fixed manager ``Metrics/success_rate`` counting goal attempts rather than the
  per-episode success bit the Direct tasks report.
