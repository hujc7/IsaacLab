Added
^^^^^

* Added manager-based counterparts for the Allegro and Shadow cube
  reorientation tasks (state and OpenAI FF/LSTM observation variants), sharing
  the Direct tasks' scalar parameters and boolean success metrics through
  common MDP terms.
* Added :class:`~isaaclab_tasks.core.reorient.mdp.reorient_timeout`, which
  restarts the episode timer on every goal reach so OpenAI-variant episodes
  extend across success streaks.
* Added opt-in domain randomization to the manager-based Allegro environment
  (``enable_domain_randomization``, disabled by default; enabling requires
  retraining).
* Added Newton and OvPhysX physics presets to the manager-based reorientation
  environments, selectable with ``physics=``.

Changed
^^^^^^^

* **Breaking:** Changed the manager-based Allegro reorientation environment to
  match the Direct observation, action, reward, reset, termination, success,
  asset, and benchmark contracts. Existing manager checkpoints are
  incompatible and must be retrained.
* Changed :class:`~isaaclab_tasks.core.reorient.reorient_manager_env_cfg.ReorientObjectEnvCfg`
  to select its scene cloning mode per physics backend and to leave
  ``sim.physics`` for each robot's configuration to assign, so the shared
  configuration is no longer hard-wired to PhysX.
