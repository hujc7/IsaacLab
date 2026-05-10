Added
^^^^^

* Added :class:`~isaaclab_tasks.utils.PresetCli`, a reusable argparse helper
  that exposes preset selection as typed CLI flags. Scripts that call
  :func:`~isaaclab_tasks.utils.setup_cli` (or
  :meth:`~isaaclab_tasks.utils.PresetCli.add_args` directly) get three new
  flags:

  - ``--physics NAME`` selects a canonical physics preset.
  - ``--renderer NAME`` selects a canonical renderer preset.
  - ``--presets NAME[,NAME...]`` selects free-form domain presets.

  Both ``--flag value`` and ``--flag=value`` forms are supported. Passing
  ``--task=<task> --help`` appends a per-task listing of valid preset names.
  The flags translate to the existing ``presets=<csv>`` global broadcast, so
  legacy ``presets=...`` invocations keep working.
* Added :func:`~isaaclab_tasks.utils.setup_cli`, a one-line wrapper that
  bundles preset flag registration, AppLauncher flag registration,
  ``parse_known_args``, and the ``sys.argv`` rewrite that the Hydra
  decorator flow consumes. Most scripts can now replace 5+ lines of CLI
  boilerplate with ``args_cli = setup_cli(parser)``.

Changed
^^^^^^^

* Added a ``presets`` keyword argument to
  :func:`~isaaclab_tasks.utils.parse_cfg.parse_env_cfg` so non-Hydra entry
  points can apply preset selections gathered from
  :meth:`~isaaclab_tasks.utils.PresetCli.collect_selected`.
* Added a ``physx = default`` canonical alias to the locomotion velocity
  flat-env physics presets for G1, Unitree Go1, and Spot. Previously these
  envs only exposed ``default`` and ``newton_mjwarp`` as field names;
  ``--physics physx`` now works on these tasks for consistency with the
  rest of the repo.
