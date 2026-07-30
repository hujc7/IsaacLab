# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capability-based object selection on top of the SimReady USD-Search service."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import re
import threading
from dataclasses import asdict, dataclass, fields

from .object_library_cfg import SimReadyObjectFilterCfg, SimReadyObjectLibraryCfg

logger = logging.getLogger(__name__)
_AUDIT_PROGRESS_INTERVAL = 25
"""Report progress every this many assets while measuring, so a cold run is not silent."""


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
        self._download_dir = os.path.join(cfg.cache_dir, "assets")
        self._prepared_dir = os.path.join(cfg.cache_dir, "prepared")
        self._measurements_path = os.path.join(cfg.cache_dir, "measurements.json")
        self._cache: dict[str, dict | None] = {}
        if os.path.exists(self._measurements_path):
            with open(self._measurements_path) as f:
                self._cache = json.load(f)

    """
    Operations.
    """

    def search(self) -> list[str]:
        """Return the distinct assets matching the configured description.

        One query is issued per phrase, because the index ranks by appearance and no single phrase
        returns a whole class of object. Properties the index can answer are asked of it; the rest
        are settled by :meth:`select` once the assets have been measured.

        Returns:
            Candidate asset paths, ordered deterministically.
        """
        from simready.search import AssetLibrary  # noqa: PLC0415

        object_filter = self.cfg.object_filter
        library = AssetLibrary(raise_on_network_error=True)
        library.add_service_source(self.cfg.service_endpoint)
        excluded = [_search_filter("PathContains", fragment) for fragment in object_filter.excluded_path_fragments]

        found: dict[str, str] = {}
        for phrase in object_filter.search_phrases:
            try:
                matches = library.search(
                    include_all=[_search_filter("Phrase", phrase), *self._native_filters()],
                    exclude_any=excluded,
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
        """Build the filters the search index can answer directly."""
        object_filter = self.cfg.object_filter
        filters = [_search_filter("Feature", feature) for feature in object_filter.validated_features]
        if object_filter.size_range is not None:
            # the index bounds height only; the remaining axes are checked once measured
            filters.append(
                _search_filter("Height", minimum=object_filter.size_range[0], maximum=object_filter.size_range[1])
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

        def mirror(remote: str) -> str:
            """Download one layer if it is not already local, and return where it landed."""
            relative = urllib.parse.urlparse(remote).path.lstrip("/")
            local = os.path.join(self._download_dir, relative)
            if os.path.exists(local):
                return local
            os.makedirs(os.path.dirname(local), exist_ok=True)
            try:
                # assets share material and texture layers, so concurrent fetches race for the same
                # path; download aside and rename, which is atomic on the same filesystem and means
                # a reader never observes a partly written layer
                partial = f"{local}.{os.getpid()}.{threading.get_ident()}.part"
                urllib.request.urlretrieve(remote, partial)
                os.replace(partial, local)
            except Exception:  # noqa: BLE001 -- a missing shared material must not sink the asset
                logger.debug("Failed to mirror layer: %s", remote, exc_info=True)
            return local

        def dependencies_of(remote: str, local: str) -> list[str]:
            """Return the layers ``local`` refers to, resolved against its own location."""
            try:
                sublayers, references, payloads = UsdUtils.ExtractExternalReferences(local)
            except Exception:  # noqa: BLE001 -- non-USD payloads such as textures have no references
                return []
            base = remote.rsplit("/", 1)[0] + "/"
            resolved = []
            for dependency in list(sublayers) + list(references) + list(payloads):
                if dependency.startswith(("http://", "https://")):
                    resolved.append(dependency)
                elif not os.path.isabs(dependency):
                    resolved.append(urllib.parse.urljoin(base, dependency))
            return resolved

        # a layer's references are only known once it has been parsed, so the closure is walked
        # breadth-first -- but that ordering is per level, not per file, so each level is fetched
        # at once rather than paying a round-trip per layer in sequence
        level, seen, root_local = [url], {url}, None
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.cfg.layer_workers) as pool:
            while level:
                locals_ = list(pool.map(mirror, level))
                if root_local is None:
                    root_local = locals_[0]
                next_level = []
                for remote, local in zip(level, locals_):
                    for dependency in dependencies_of(remote, local):
                        if dependency not in seen:
                            seen.add(dependency)
                            next_level.append(dependency)
                level = next_level
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
            if cached.keys() == {f.name for f in fields(ObjectSpec)}:
                return ObjectSpec(**{**cached, "dims": tuple(cached["dims"])})
            logger.debug("Discarding cache entry with an outdated layout: %s", url)
        spec = self._audit_uncached(url)
        self._cache[url] = None if spec is None else asdict(spec)
        return spec

    def select(self, candidates: list[str] | None = None) -> list[ObjectSpec]:
        """Measure the candidates and keep the ones satisfying every configured property.

        Args:
            candidates: Assets to measure. Defaults to the result of :meth:`search`.

        Returns:
            At most :attr:`~SimReadyObjectLibraryCfg.num_objects` specs, ordered by ascending mass.
        """
        object_filter = self.cfg.object_filter
        candidates = self.search() if candidates is None else candidates

        kept: list[ObjectSpec] = []
        dropped: dict[str, int] = {}
        # auditing is entirely network-bound -- opening a stage and measuring it costs milliseconds,
        # while mirroring its layers costs seconds -- so candidates are audited concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.cfg.audit_workers) as pool:
            for index, spec in enumerate(pool.map(self.audit, candidates)):
                reason = _rejection_reason(spec, object_filter)
                if reason is None:
                    kept.append(spec)
                else:
                    dropped[reason] = dropped.get(reason, 0) + 1
                if (index + 1) % _AUDIT_PROGRESS_INTERVAL == 0:
                    logger.info("Audited %d/%d candidates, kept %d.", index + 1, len(candidates), len(kept))
        self.save_cache()

        if object_filter.max_per_product_family is not None:
            per_family: dict[str, list[ObjectSpec]] = {}
            for spec in sorted(kept, key=lambda s: s.url):
                per_family.setdefault(spec.family, []).append(spec)
            kept = [s for variants in per_family.values() for s in variants[: object_filter.max_per_product_family]]
            logger.info("Kept %d assets across %d product families.", len(kept), len(per_family))

        selected = sorted(kept, key=lambda spec: (spec.mass is None, spec.mass))[: self.cfg.num_objects]

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

        os.makedirs(self._prepared_dir, exist_ok=True)
        prepared: list[str] = []
        for spec in specs:
            out_path = os.path.join(self._prepared_dir, os.path.splitext(os.path.basename(spec.url))[0] + ".usda")
            prepared.append(out_path)
            if os.path.exists(out_path):
                continue
            stage = Usd.Stage.Open(self.fetch(spec.url), Usd.Stage.LoadAll)
            # instanced prims cannot be copied out, and traversal does not descend into one until it
            # has been de-instanced, so each pass can expose a further nested level
            # traversal does not descend into an instanced prim until it has been de-instanced,
            # so each pass can expose a further nested level; repeat until none remain
            while instanced := [prim for prim in stage.Traverse() if prim.IsInstanceable()]:
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

    def resolve(self) -> list[str]:
        """Return task-ready USD paths for the configured objects.

        The set an asset query resolves to is recorded to
        :attr:`~SimReadyObjectLibraryCfg.resolution_path`. When that record exists it is used as-is,
        so a run needs no search request and every machine -- and every rank of one job -- gets the
        same objects even if the catalogue moves. Set
        :attr:`~SimReadyObjectLibraryCfg.always_query` to resolve afresh.

        Returns:
            Local paths of the prepared USDs, ready to hand to a spawner.
        """
        urls = None if self.cfg.always_query else self._load_resolution()
        if urls is None:
            specs = self.select()
            self._save_resolution([spec.url for spec in specs])
        else:
            logger.info("Using the %d objects recorded in %s.", len(urls), self.cfg.resolution_path)
            specs = [spec for spec in (self.audit(url) for url in urls) if spec is not None]
        return self.prepare(specs)

    def _load_resolution(self) -> list[str] | None:
        """Return the assets recorded for this configuration, or ``None`` when there is no record."""
        if not os.path.exists(self.cfg.resolution_path):
            return None
        with open(self.cfg.resolution_path) as f:
            return json.load(f)["assets"]

    def _save_resolution(self, urls: list[str]) -> None:
        """Record the resolved assets, alongside the query that produced them for readability."""
        os.makedirs(os.path.dirname(self.cfg.resolution_path) or ".", exist_ok=True)
        with open(self.cfg.resolution_path, "w") as f:
            json.dump(
                {"search_phrases": list(self.cfg.object_filter.search_phrases), "assets": urls},
                f,
                indent=2,
            )

    def save_cache(self) -> None:
        """Persist the measurements, so a later resolve opens no assets."""
        os.makedirs(os.path.dirname(self._measurements_path) or ".", exist_ok=True)
        with open(self._measurements_path, "w") as f:
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
            _latest_validation_passed(stage, self.cfg.object_filter.validated_features),
        )


def _search_filter(name: str, *args, **kwargs):
    """Return a ``SearchFilter<name>`` instance from the search package, imported on use."""
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
    """Return why an asset is unusable, or ``None`` when it satisfies every configured property.

    Only properties the search index could not settle are checked here, on the measured asset.
    """
    if spec is None:
        return "could not be opened"
    if cfg.require_rigid_body and spec.body_path is None:
        return "no rigid body: collision geometry but nothing physics can move"
    if not spec.validation_passed:
        return "newest-dated validation verdict is a failure"
    if cfg.size_range is not None:
        # the index bounded height only, so the widest and narrowest axes are still unchecked
        if spec.max_dim > cfg.size_range[1]:
            return "too large for the workspace"
        if spec.min_dim < cfg.size_range[0]:
            return "too thin for stable contact"
    if cfg.mass_range is not None:
        if spec.mass is None:
            return "no authored mass"
        if spec.mass > cfg.mass_range[1]:
            return "heavier than the gripper can hold"
        if spec.mass < cfg.mass_range[0]:
            return "too light for stable contact"
    return None
