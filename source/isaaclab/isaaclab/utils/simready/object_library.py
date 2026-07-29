# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capability-based object selection on top of the SimReady USD-Search service."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, fields

from .object_library_cfg import SimReadyObjectFilterCfg, SimReadyObjectLibraryCfg

logger = logging.getLogger(__name__)

_AUDIT_PROGRESS_INTERVAL = 25
"""How often to report progress while auditing, in candidates."""

_MAX_INSTANCING_DEPTH = 8
"""Upper bound on how deeply catalogue assets nest instanced prims, used to bound de-instancing."""


@dataclass
class ObjectSpec:
    """Physical facts about one asset, gathered by opening it.

    None of these are exposed by the search service, which is why the asset has to be opened at all.
    """

    url: str
    """Remote path of the asset."""

    body_path: str | None
    """Prim path of the ``PhysicsRigidBodyAPI`` body, or ``None`` for multibody assets that ship
    colliders but nothing dynamic, which are not simulatable as published."""

    mass: float | None
    """Authored mass [kg], or ``None`` when the asset declares none."""

    dims: tuple[float, float, float]
    """World-space bounding-box extents [m]."""

    validation_passed: bool
    """Whether the newest-dated verdict for every required feature is a pass."""

    @property
    def max_dim(self) -> float:
        """Largest bounding-box extent [m]."""
        return max(self.dims)

    @property
    def min_dim(self) -> float:
        """Smallest bounding-box extent [m]."""
        return min(self.dims)

    @property
    def family(self) -> str:
        """Product family, with the variant code stripped.

        ``Golf_Ball_A03`` and ``Golf_Ball`` are the same family, as are ``Boxed_Drink_A01`` and
        ``Boxed_Drink_E01``. Grouping on this is what turns "twelve unique files" into "twelve
        objects that look different", which is the property a viewer actually judges.
        """
        directory = self.url.rsplit("/", 2)[-2]
        directory = re.sub(r"_[A-Z]?\d+$", "", directory)
        directory = re.sub(r"_[A-Z]\d+$", "", directory)
        return directory.lower().replace("_", "")


_OBJECT_SPEC_FIELDS = frozenset(f.name for f in fields(ObjectSpec))
"""Field names an audit-cache entry must carry to still be readable."""


class SimReadyObjectLibrary:
    """Searches, audits, filters, and prepares SimReady assets for a manipulation task.

    The catalogue is turned into a usable object set in four stages. :meth:`resolve` runs all four;
    the individual methods are exposed for inspecting or overriding a stage.

    .. code-block:: text

        search()   ask the service which assets look right          1 request per phrase
           |       every natively supported filter is applied here
           v
        audit()    open each candidate and read what the service    1 fetch + open per asset,
           |       does not expose: body, mass, extents, verdicts   cached forever after
           v
        select()   keep what the robot can handle, stratify by      no I/O
           |       mass, collapse near-identical variants
           v
        prepare()  write task-ready USDs at a uniform /Object root  1 write per selected asset

    :meth:`audit` is the only expensive stage and the reason the library exists: the service returns
    a path, a relevance score and tags, but nothing a manipulation task can decide on. Its results
    are cached to :attr:`~SimReadyObjectLibraryCfg.cache_path`, so the first sweep of a catalogue
    costs minutes and every later one is instant -- which is what makes resolving objects inside a
    task configuration practical.

    Args:
        cfg: Configuration of the library.
    """

    def __init__(self, cfg: SimReadyObjectLibraryCfg):
        self.cfg = cfg
        self._cache: dict[str, dict | None] = {}
        if os.path.exists(self.cfg.cache_path):
            with open(self.cfg.cache_path) as f:
                self._cache = json.load(f)

    """
    Operations.
    """

    def search(self) -> list[str]:
        """Return the distinct candidate assets, applying every natively supported filter.

        One query is issued per phrase, because the index ranks by appearance and no single phrase
        returns a geometry class. Filters the service supports are pushed server-side so unmatched
        assets never consume the per-phrase result budget; the rest are completed by :meth:`select`.

        Returns:
            Candidate asset paths, ordered deterministically.
        """
        from simready.search import AssetLibrary  # noqa: PLC0415

        object_filter = self.cfg.object_filter
        library = AssetLibrary(raise_on_network_error=True)
        library.add_service_source(self.cfg.service_endpoint)
        excluded = [_filter("PathContains", fragment) for fragment in object_filter.excluded_path_fragments]

        found: dict[str, str] = {}
        for phrase in object_filter.search_phrases:
            try:
                matches = library.search(
                    include_all=[_filter("Phrase", phrase), *self._native_filters()],
                    exclude_any=excluded,
                    base_paths=list(object_filter.base_paths) or None,
                    max_count=self.cfg.results_per_phrase,
                )
            except Exception:  # noqa: BLE001 -- one bad phrase must not sink the whole sweep
                logger.warning("SimReady search failed for phrase: %s", phrase, exc_info=True)
                continue
            for match in matches:
                path = match.asset_path
                if any(fragment in path for fragment in object_filter.excluded_path_fragments):
                    continue  # re-check locally: the exclusion above is applied by the service
                # de-duplicate on file name: the same asset surfaces under many phrases
                found[path.rsplit("/", 1)[-1]] = path
        return sorted(found.values())

    def _native_filters(self) -> list:
        """Build the service-side filters for every configured field the service supports."""
        object_filter = self.cfg.object_filter
        filters = [
            # the service filters on height only; the other two axes are checked in select()
            _filter("Height", minimum=object_filter.size_range[0], maximum=object_filter.size_range[1])
        ]
        if object_filter.min_relevance > 0.0:
            filters.append(_filter("Relevance", minimum=object_filter.min_relevance))
        for name, values in (
            ("Feature", object_filter.required_features),
            ("Profile", object_filter.required_profiles),
            ("Class", object_filter.required_classes),
            ("Tag", object_filter.required_tags),
            ("Country", object_filter.required_countries),
            ("ScenePOI", object_filter.required_scene_poi_tags),
        ):
            filters.extend(_filter(name, value) for value in values)
        filters.extend(
            _filter("ArbitraryDictValue", list(key_path), value) for key_path, value in object_filter.required_metadata
        )
        return filters

    def fetch(self, url: str) -> str | None:
        """Mirror an asset and its whole layer closure locally, over plain HTTPS.

        SimReady assets are layered (``wrapper`` -> ``payloads/base.usda`` -> ``Physics/physics.usda``),
        so downloading only the top ``.usd`` yields a stage with no physics and a degenerate bounding
        box. Following the sublayer, reference, and payload graph and mirroring relative paths under
        :attr:`~isaaclab.utils.simready.SimReadyObjectLibraryCfg.download_dir` reproduces the asset.

        This is what keeps the pipeline Kit-free. Stock OpenUSD cannot open an ``https://`` URL and
        importing ``omni.client`` is not enough, because its resolver only activates inside the Kit
        runtime -- but neither is needed once the layers are local files.

        Args:
            url: Remote path of the asset.

        Returns:
            Local path of the mirrored asset, or ``None`` if it could not be retrieved.
        """
        import urllib.parse  # noqa: PLC0415
        import urllib.request  # noqa: PLC0415

        from pxr import UsdUtils  # noqa: PLC0415

        pending, seen, root_local = [url], set(), None
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            relative = urllib.parse.urlparse(current).path.lstrip("/")
            local = os.path.join(self.cfg.download_dir, relative)
            if root_local is None:
                root_local = local
            os.makedirs(os.path.dirname(local), exist_ok=True)
            if not os.path.exists(local):
                try:
                    urllib.request.urlretrieve(current, local)
                except Exception:  # noqa: BLE001 -- a missing shared material must not sink the asset
                    logger.debug("Failed to mirror layer: %s", current, exc_info=True)
                    continue
            try:
                sublayers, references, payloads = UsdUtils.ExtractExternalReferences(local)
            except Exception:  # noqa: BLE001 -- non-USD payloads such as textures have no references
                continue
            base = current.rsplit("/", 1)[0] + "/"
            for dependency in list(sublayers) + list(references) + list(payloads):
                if dependency.startswith(("http://", "https://")):
                    pending.append(dependency)
                elif not os.path.isabs(dependency):
                    pending.append(urllib.parse.urljoin(base, dependency))
        return root_local if root_local and os.path.exists(root_local) else None

    def audit(self, url: str) -> ObjectSpec | None:
        """Return the physical facts for one asset, opening it only on a cache miss.

        Runs without the simulation app: :meth:`fetch` mirrors the layer closure over HTTPS and the
        stage is opened from local files.

        Args:
            url: Remote path of the asset.

        Returns:
            The asset's physical facts, or ``None`` if it could not be opened.
        """
        if url in self._cache:
            cached = self._cache[url]
            if cached is None:
                return None
            # entries written by an older field layout are regenerable, so re-audit rather than fail
            if cached.keys() == _OBJECT_SPEC_FIELDS:
                return ObjectSpec(**{**cached, "dims": tuple(cached["dims"])})
            logger.debug("Discarding cache entry with an outdated layout: %s", url)
        spec = self._audit_uncached(url)
        self._cache[url] = None if spec is None else asdict(spec)
        return spec

    def select(self, num_objects: int, candidates: list[str] | None = None) -> list[ObjectSpec]:
        """Audit candidates and keep the objects the robot can work with.

        Args:
            num_objects: Number of objects to keep.
            candidates: Assets to audit. Defaults to the result of :meth:`search`.

        Returns:
            At most :paramref:`num_objects` specs, ordered by ascending mass.
        """
        object_filter = self.cfg.object_filter
        candidates = self.search() if candidates is None else candidates

        kept: list[ObjectSpec] = []
        dropped: dict[str, int] = {}
        for index, url in enumerate(candidates):
            spec = self.audit(url)
            reason = _rejection_reason(spec, object_filter)
            if reason is None:
                kept.append(spec)
            else:
                dropped[reason] = dropped.get(reason, 0) + 1
            # this is the one slow stage -- on a cold cache each asset is a fetch and a stage open,
            # so report progress rather than leaving minutes of silence
            if (index + 1) % _AUDIT_PROGRESS_INTERVAL == 0:
                logger.info("Audited %d/%d candidates, kept %d.", index + 1, len(candidates), len(kept))
        self.save_cache()

        if object_filter.distinct_families:
            by_family: dict[str, ObjectSpec] = {}
            for spec in kept:
                by_family.setdefault(spec.family, spec)
            logger.info("Found %d distinct families among %d usable assets.", len(by_family), len(kept))
            kept = list(by_family.values())

        selected = sorted(kept, key=lambda spec: spec.mass)[:num_objects]

        logger.info("Selected %d of %d usable objects from %d candidates.", len(selected), len(kept), len(candidates))
        for reason, count in sorted(dropped.items(), key=lambda item: -item[1]):
            logger.info("Dropped %d candidates: %s.", count, reason)
        return selected

    def prepare(self, specs: list[ObjectSpec]) -> list[str]:
        """Write task-ready USDs, each asset's rigid body extracted to a uniform ``/Object`` root.

        This is the one modification heterogeneous cloning requires. Assets nest their body under a
        per-asset ``RootNode/Geometry/<name>`` path, and a per-environment view cannot be built
        across varying paths. Collider, mass, and origin are left exactly as published.

        Args:
            specs: Objects to prepare.

        Returns:
            Local paths of the prepared USDs.
        """
        from pxr import Sdf, Usd  # noqa: PLC0415

        os.makedirs(self.cfg.prepared_dir, exist_ok=True)
        prepared: list[str] = []
        for spec in specs:
            out_path = os.path.join(self.cfg.prepared_dir, os.path.splitext(os.path.basename(spec.url))[0] + ".usda")
            prepared.append(out_path)
            if os.path.exists(out_path):
                continue
            stage = Usd.Stage.Open(self.fetch(spec.url), Usd.Stage.LoadAll)
            # instanced prims cannot be copied out, and traversal does not descend into one until it
            # has been de-instanced, so each pass can expose a further nested level
            for _ in range(_MAX_INSTANCING_DEPTH):
                instanced = [prim for prim in stage.Traverse() if prim.IsInstanceable()]
                if not instanced:
                    break
                for prim in instanced:
                    prim.SetInstanceable(False)
            layer = Sdf.Layer.CreateNew(out_path)
            Sdf.CreatePrimInLayer(layer, "/Object")
            Sdf.CopySpec(stage.Flatten(), Sdf.Path(spec.body_path), layer, Sdf.Path("/Object"))
            layer.Save()
            object_stage = Usd.Stage.Open(out_path)
            object_stage.SetDefaultPrim(object_stage.GetPrimAtPath("/Object"))
            object_stage.GetRootLayer().Save()
        return prepared

    def resolve(self, num_objects: int) -> list[str]:
        """Search, select, and prepare in one call, reusing a previous resolution when possible.

        The resolved asset list is recorded against a fingerprint of the query, so a repeat call with
        the same configuration issues no search requests at all. That is what keeps every rank of a
        distributed run on the same object set: the first to resolve writes the list, the rest read
        it, instead of each querying a catalogue that may have changed in between.

        Args:
            num_objects: Number of objects to resolve.

        Returns:
            Local paths of the prepared USDs, ready to hand to a spawner.
        """
        fingerprint = self._query_fingerprint(num_objects)
        urls = self._load_resolution(fingerprint)
        if urls is None:
            specs = self.select(num_objects)
            self._save_resolution(fingerprint, num_objects, [spec.url for spec in specs])
        else:
            logger.info("Reusing %d objects resolved earlier for this query.", len(urls))
            specs = [spec for spec in (self.audit(url) for url in urls) if spec is not None]
        return self.prepare(specs)

    def _query_fingerprint(self, num_objects: int) -> str:
        """Return a stable digest of everything that decides which objects a query resolves to."""
        import hashlib  # noqa: PLC0415

        object_filter = self.cfg.object_filter.to_dict()
        # a predicate cannot be hashed by behaviour, so record what identifies it instead
        predicate = self.cfg.object_filter.filter_func
        object_filter["filter_func"] = getattr(predicate, "__qualname__", None)
        query = {
            "endpoint": self.cfg.service_endpoint,
            "results_per_phrase": self.cfg.results_per_phrase,
            "num_objects": num_objects,
            "filter": object_filter,
        }
        return hashlib.sha256(json.dumps(query, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def _load_resolution(self, fingerprint: str) -> list[str] | None:
        """Return the assets this query resolved to before, or ``None`` on a miss."""
        if not os.path.exists(self.cfg.resolution_cache_path):
            return None
        with open(self.cfg.resolution_cache_path) as f:
            entry = json.load(f).get(fingerprint)
        return None if entry is None else entry["assets"]

    def _save_resolution(self, fingerprint: str, num_objects: int, urls: list[str]) -> None:
        """Record which assets this query resolved to, alongside the query itself for readability."""
        path = self.cfg.resolution_cache_path
        resolutions = {}
        if os.path.exists(path):
            with open(path) as f:
                resolutions = json.load(f)
        resolutions[fingerprint] = {
            "num_objects": num_objects,
            "search_phrases": list(self.cfg.object_filter.search_phrases),
            "assets": urls,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(resolutions, f, indent=2, sort_keys=True)

    def save_cache(self) -> None:
        """Persist the audits, so a later resolve costs no asset opens."""
        os.makedirs(os.path.dirname(self.cfg.cache_path) or ".", exist_ok=True)
        with open(self.cfg.cache_path, "w") as f:
            json.dump(self._cache, f)

    """
    Internal helpers.
    """

    def _audit_uncached(self, url: str) -> ObjectSpec | None:
        """Open an asset and read the facts the search service does not expose."""
        from pxr import Usd, UsdGeom  # noqa: PLC0415

        local = self.fetch(url)
        if local is None:
            return None
        try:
            stage = Usd.Stage.Open(local, Usd.Stage.LoadAll)
        except Exception:  # noqa: BLE001 -- an unreadable asset is simply not a candidate
            logger.debug("Failed to open asset: %s", url, exc_info=True)
            return None
        if stage is None or stage.GetDefaultPrim() is None:
            return None

        body_path = mass = None
        for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
            if "PhysicsRigidBodyAPI" in [str(schema) for schema in prim.GetAppliedSchemas()]:
                body_path = str(prim.GetPath())
            attribute = prim.GetAttribute("physics:mass")
            if attribute and attribute.IsValid() and attribute.Get():
                mass = float(attribute.Get())
        size = (
            UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
            .ComputeWorldBound(stage.GetDefaultPrim())
            .ComputeAlignedRange()
            .GetSize()
        )
        return ObjectSpec(
            url,
            body_path,
            mass,
            (size[0], size[1], size[2]),
            _latest_validation_passed(stage, self.cfg.object_filter.required_features),
        )


def _filter(name: str, *args, **kwargs):
    """Build a ``SearchFilter<name>`` from the search package, imported on use."""
    import simready.search  # noqa: PLC0415

    return getattr(simready.search, f"SearchFilter{name}")(*args, **kwargs)


def _latest_validation_passed(stage, features: tuple[str, ...]) -> bool:
    """Return whether the newest-dated verdict passes for every required feature.

    Assets carrying no validation history are accepted: absence of a verdict is not a failure.
    """
    metadata = (stage.GetRootLayer().customLayerData or {}).get("SimReady_Metadata")
    if metadata is None:
        return True
    metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
    validated = (metadata.get("validation") or {}).get("validated_features") or {}
    if not validated:
        return True
    newest = validated[max(validated)]
    return all(newest.get(feature, {}).get("passed") is not False for feature in features)


def _rejection_reason(spec: ObjectSpec | None, cfg: SimReadyObjectFilterCfg) -> str | None:
    """Return why an asset is unusable for the task, or ``None`` when it passes every filter.

    Only the checks the search service cannot answer are made here. The service has already applied
    everything it supports, so reaching this point means the asset looked right and now has to prove
    it is actually usable.
    """
    if spec is None:
        return "could not be opened"
    if cfg.require_rigid_body and spec.body_path is None:
        return "no rigid body (multibody asset: colliders but nothing dynamic)"
    if cfg.require_latest_validation and not spec.validation_passed:
        return "newest-dated validation verdict is a failure"
    if spec.max_dim > cfg.size_range[1]:
        # the service bounded height only, so the widest axis is still unchecked
        return "too large for the workspace"
    if spec.min_dim < cfg.size_range[0]:
        return "too thin for stable contact"
    if spec.mass is None:
        return "no authored mass"
    if spec.mass > cfg.mass_range[1]:
        return "heavier than the gripper can hold"
    if spec.mass < cfg.mass_range[0]:
        return "too light for stable contact"
    if cfg.filter_func is not None and not cfg.filter_func(spec):
        return "rejected by the configured filter function"
    return None
