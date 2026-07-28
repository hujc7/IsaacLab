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
from dataclasses import asdict, dataclass

from isaaclab.utils.assets import search_simready_usd_paths

from .simready_cfg import SimReadyObjectFilterCfg, SimReadyObjectLibraryCfg

logger = logging.getLogger(__name__)

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
    """Whether the newest-dated ``FET003_BASE_PHYSX`` verdict is a pass."""

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
        """Return the distinct candidate assets matching any configured phrase.

        Restricting to ``FET003_BASE_PHYSX`` keeps the results to single-body PhysX-ready assets.
        ``FET004`` is the *multibody* feature, and so is the wrong filter for a task that needs one
        rigid body per object.

        Returns:
            Candidate asset paths, ordered deterministically.
        """
        found: dict[str, str] = {}
        for phrase in self.cfg.search_phrases:
            try:
                matches = search_simready_usd_paths(
                    query=phrase,
                    top_k=self.cfg.results_per_phrase,
                    filter_features=["FET003_BASE_PHYSX"],
                    filter_max_height=self.cfg.object_filter.size_range[1],
                    exclude_path_contains=list(self.cfg.excluded_path_fragments),
                    service_endpoint=self.cfg.service_endpoint,
                    raise_on_empty=False,
                )
            except Exception:  # noqa: BLE001 -- one bad phrase must not sink the whole sweep
                logger.warning("SimReady search failed for phrase: %s", phrase, exc_info=True)
                continue
            for path in matches:
                if any(fragment in path for fragment in self.cfg.excluded_path_fragments):
                    continue  # re-check locally: the exclusion above is applied by the service
                # de-duplicate on file name: the same asset surfaces under many phrases
                found[path.rsplit("/", 1)[-1]] = path
        return sorted(found.values())

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
            return None if cached is None else ObjectSpec(**{**cached, "dims": tuple(cached["dims"])})
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
        for url in candidates:
            spec = self.audit(url)
            reason = _rejection_reason(spec, object_filter)
            if reason is None:
                kept.append(spec)
            else:
                dropped[reason] = dropped.get(reason, 0) + 1
        self.save_cache()

        if object_filter.distinct_families:
            by_family: dict[str, ObjectSpec] = {}
            for spec in kept:
                by_family.setdefault(spec.family, spec)
            logger.info("Found %d distinct families among %d usable assets.", len(by_family), len(kept))
            kept = list(by_family.values())

        # reserve a slice of the selection for objects at the edge of what the robot can lift
        heavy = sorted((spec for spec in kept if spec.mass >= object_filter.heavy_from), key=lambda s: s.mass)
        light = sorted((spec for spec in kept if spec.mass < object_filter.heavy_from), key=lambda s: s.mass)
        num_heavy = min(len(heavy), round(num_objects * object_filter.heavy_fraction))
        selected = light[: num_objects - num_heavy] + heavy[:num_heavy]

        logger.info(
            "Selected %d of %d usable objects from %d candidates (%d at or above %.2f kg).",
            len(selected),
            len(kept),
            len(candidates),
            num_heavy,
            object_filter.heavy_from,
        )
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
        """Search, select, and prepare in one call.

        Args:
            num_objects: Number of objects to resolve.

        Returns:
            Local paths of the prepared USDs, ready to hand to a spawner.
        """
        return self.prepare(self.select(num_objects))

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
        return ObjectSpec(url, body_path, mass, (size[0], size[1], size[2]), _latest_validation_passed(stage))


def _latest_validation_passed(stage) -> bool:
    """Return the newest-dated ``FET003_BASE_PHYSX`` verdict, or ``True`` when there is no history."""
    metadata = (stage.GetRootLayer().customLayerData or {}).get("SimReady_Metadata")
    if metadata is None:
        return True
    metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
    validated = (metadata.get("validation") or {}).get("validated_features") or {}
    if not validated:
        return True
    return validated[max(validated)].get("FET003_BASE_PHYSX", {}).get("passed") is not False


def _rejection_reason(spec: ObjectSpec | None, cfg: SimReadyObjectFilterCfg) -> str | None:
    """Return why an asset is unusable for the task, or ``None`` when it passes every filter."""
    if spec is None:
        return "could not be opened"
    if spec.body_path is None:
        return "no rigid body (multibody asset: colliders but nothing dynamic)"
    if cfg.require_latest_validation and not spec.validation_passed:
        return "latest FET003_BASE_PHYSX validation failed"
    if spec.max_dim > cfg.size_range[1]:
        return "too large for the workspace"
    if spec.min_dim < cfg.size_range[0]:
        return "too thin for stable contact"
    if spec.mass is None:
        return "no authored mass"
    if spec.mass > cfg.mass_range[1]:
        return "heavier than the gripper can hold"
    if spec.mass < cfg.mass_range[0]:
        return "too light for stable contact"
    return None
