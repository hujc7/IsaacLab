Added
^^^^^

* Added manager-based counterparts for the Shadow cube reorientation task and
  its OpenAI FF/LSTM observation variants, alongside the existing Allegro
  manager task.
* Added :class:`~isaaclab_tasks.core.reorient.mdp.reorient_timeout`, which
  restarts the episode timer on every goal reach so OpenAI-variant episodes
  extend across success streaks.
* Added ``enable_domain_randomization`` to the manager-based Allegro
  environment. Randomization is applied by default, as it was before; disable
  it to compare the manager and Direct workflows directly.
* Added Newton and OvPhysX physics presets to the manager-based reorientation
  environments, selectable with ``physics=``.
* Added a Direct-versus-manager value-parity check covering timing, success
  tolerance, fall distance, and the consecutive-success cap.

Changed
^^^^^^^

* **Breaking:** Changed the manager-based Allegro reorientation environment to
  match the Direct observation, action, reset, and termination contracts. The
  observation space changes size, so existing manager checkpoints cannot be
  loaded and must be retrained.
* Changed :class:`~isaaclab_tasks.core.reorient.reorient_manager_env_cfg.ReorientObjectEnvCfg`
  to select its scene cloning mode per physics backend and to leave
  ``sim.physics`` for each robot's configuration to assign, so the shared
  configuration is no longer hard-wired to PhysX. The manager tasks derive from
  it rather than redeclaring its groups.
* Changed the reorientation reward to compose from separate reward terms, so
  each contributes its own episode log entry, weight, and curriculum hook.
* Renamed the per-robot scene constants to name what they hold: ``ROBOT_CFG``
  becomes ``SHADOW_HAND_ROBOT_CFG`` or ``ALLEGRO_HAND_ROBOT_CFG``,
  ``OBJECT_CFG`` becomes ``CUBE_CFG``, and ``ObjectCfg`` becomes ``CubeCfg``.

Fixed
^^^^^

* Fixed the manager-based reorientation tasks not reporting
  ``Metrics/success_rate``.
* Fixed ``Metrics/success_rate`` on the manager-based reorientation tasks counting
  completed goal attempts rather than the per-episode success bit the Direct tasks
  report, so the two were not comparable.
