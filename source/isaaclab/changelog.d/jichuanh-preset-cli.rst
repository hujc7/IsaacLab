Added
^^^^^

* Added :class:`~isaaclab.utils.presets.PresetKind` enum and the
  :func:`~isaaclab.utils.presets.register` class decorator. Each
  ``PresetKind`` member carries its CLI flag label and a per-kind dict of
  legacy aliases. ``@register(PresetKind.X, "name")`` declares the
  canonical (kind, name) binding next to a config class definition::

      @register(PresetKind.PHYSICS, "physx")
      @configclass
      class PhysxCfg(PhysicsCfg):
          ...

  The CLI layer (:class:`~isaaclab_tasks.utils.PresetCli`) reads the
  :class:`~isaaclab.utils.presets.PresetRegistry` populated by these
  decorators to validate user input, surface the canonical vocabulary in
  ``--help``, and enforce cross-env consistency via a CI lint that walks
  the gym registry. Decorated config classes today:
  ``PhysxCfg`` (``physx``, PHYSICS),
  ``OvPhysxCfg`` (``ovphysx``, PHYSICS),
  ``MJWarpSolverCfg`` (``newton_mjwarp``, PHYSICS),
  ``KaminoSolverCfg`` (``newton_kamino``, PHYSICS),
  ``IsaacRtxRendererCfg`` (``isaacsim_rtx_renderer``, RENDERER),
  ``NewtonWarpRendererCfg`` (``newton_renderer``, RENDERER),
  ``OVRTXRendererCfg`` (``ovrtx_renderer``, RENDERER).

  Adding a new CLI flag (e.g. ``--collision``) takes one new
  ``PresetKind`` enum member; ``PresetCli`` picks it up automatically.
