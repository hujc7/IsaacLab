Added
^^^^^

* Added the :mod:`isaaclab.utils.simready` sub-module to resolve SimReady assets into task-ready
  rigid objects: catalogue search, cached per-asset audit, capability-based filtering, and
  preparation of a uniform ``/Object`` root for heterogeneous cloning. Runs without Isaac Sim by
  mirroring each asset's layer closure over HTTPS.
* Added :class:`~isaaclab.utils.simready.SimReadyObjectLibraryCfg` and
  :class:`~isaaclab.utils.simready.SimReadyObjectFilterCfg` to configure
  :class:`~isaaclab.utils.simready.SimReadyObjectLibrary`. The filter exposes every criterion the
  USD-Search service supports, plus the ones it does not -- mass, the bounding-box axes beyond
  height, rigid-body presence, the newest-dated validation verdict, and product-family collapsing --
  which are applied after opening the asset. Each field records which of the two applies.
* Added the ``Isaac-Lift-KukaAllegro-SimReady`` and ``Isaac-Lift-KukaAllegro-SimReady-Play``
  environments, which inherit the dexsuite lift task and replace its primitive shapes with objects
  resolved from the SimReady catalogue. Objects are selected by what the robot can pick up rather
  than rescaled or mass-normalised to fit it.
* Added ``filter_max_height``, ``exclude_path_contains``, and ``raise_on_empty`` to
  :func:`~isaaclab.utils.assets.search_simready_usd_paths`, exposing the service's bounding-box and
  path filters and letting a multi-phrase sweep treat an empty result as a non-event.

Changed
^^^^^^^

* Changed :attr:`~isaaclab.utils.assets.SIMREADY_SEARCH_SERVICE_ENDPOINT` to the production
  USD-Search deployment. The previous development deployment's asset index lags and returns stale
  results. Callers that need the old endpoint can pass it through ``service_endpoint``.
