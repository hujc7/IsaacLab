Added
^^^^^

* Added :class:`~isaaclab.utils.simready.SimReadyObjectLibrary` to resolve SimReady assets into
  task-ready rigid objects: catalogue search, cached per-asset audit, capability-based filtering
  (size, mass, rigid-body presence, latest validation verdict), and preparation of a uniform
  ``/Object`` root for heterogeneous cloning. Runs without Isaac Sim by mirroring each asset's layer
  closure over HTTPS.

Changed
^^^^^^^

* Changed ``scripts/demos/simready_lift.py`` to resolve its objects from the SimReady service at run
  time via :class:`~isaaclab.utils.simready.SimReadyObjectLibrary` and to train on
  ``Isaac-Lift-KukaAllegro``, which already provides point-cloud observations, contact-shaped rewards
  and a per-environment table. Objects are selected by what the robot can pick up rather than
  rescaled or mass-normalised to fit it.
