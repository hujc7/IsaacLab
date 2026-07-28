Added
^^^^^

* Added the :mod:`isaaclab.utils.simready` sub-module to resolve SimReady assets into task-ready
  rigid objects: catalogue search, cached per-asset audit, capability-based filtering, and
  preparation of a uniform ``/Object`` root for heterogeneous cloning. Runs without Isaac Sim by
  mirroring each asset's layer closure over HTTPS.
* Added :class:`~isaaclab.utils.simready.SimReadyObjectLibraryCfg` and
  :class:`~isaaclab.utils.simready.SimReadyObjectFilterCfg` to configure
  :class:`~isaaclab.utils.simready.SimReadyObjectLibrary`. The filter expresses what a given robot
  can handle -- accepted size and mass, the share of objects reserved at the edge of that capability,
  and whether near-identical product variants are collapsed -- so assets are selected rather than
  rescaled or mass-normalised to fit.
* Added ``filter_max_height``, ``exclude_path_contains``, and ``raise_on_empty`` to
  :func:`~isaaclab.utils.assets.search_simready_usd_paths`, exposing the service's bounding-box and
  path filters and letting a multi-phrase sweep treat an empty result as a non-event.

Changed
^^^^^^^

* Changed :attr:`~isaaclab.utils.assets.SIMREADY_SEARCH_SERVICE_ENDPOINT` to the production
  USD-Search deployment. The previous development deployment's asset index lags and returns stale
  results. Callers that need the old endpoint can pass it through ``service_endpoint``.
* Changed ``scripts/demos/simready_lift.py`` to resolve its objects from the SimReady service at run
  time via :class:`~isaaclab.utils.simready.SimReadyObjectLibrary` and to train on
  ``Isaac-Lift-KukaAllegro``, which already provides point-cloud observations, contact-shaped rewards
  and a per-environment table.
