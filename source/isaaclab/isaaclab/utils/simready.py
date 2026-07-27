# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capability-based object selection on top of the SimReady USD-Search service.

The search service answers *"which assets look like X?"* and can pre-filter server-side on a single
bounding-box axis and a validation feature. What it returns, however, is only
``asset_path`` / ``asset_rel_path`` / ``asset_version`` / ``relevance_score`` / ``source_id`` /
``tags`` — no dimensions, no mass, no per-asset validation verdict. Every property a manipulation
task needs to decide *"can this robot pick this up?"* therefore requires opening the asset itself.

This module is that missing layer:

* :meth:`SimReadyObjectLibrary.search` — cast a wide net (the index ranks by appearance, so one
  phrase never returns a geometry class; the union of many phrases is what covers the catalogue).
* :meth:`SimReadyObjectLibrary.audit` — open an asset once, ever, and cache the facts the service
  does not expose: rigid-body presence, authored mass, bounding box, and the *latest-dated*
  ``FET003_BASE_PHYSX`` verdict.
* :meth:`SimReadyObjectLibrary.select` — filter on **robot capability**, not on asset opinion, and
  stratify by mass so a deliberate minority of objects stay too heavy to lift.
* :meth:`SimReadyObjectLibrary.prepare` — emit task-ready USDs whose rigid body sits at a uniform
  ``/Object`` root, the one modification heterogeneous cloning actually requires.

Authored masses are treated as ground truth: a canned good genuinely is heavy, so an object is
dropped only when the robot could not lift it, never because the number looks surprising.

**The whole pipeline runs without Kit.** Search is plain HTTP; assets are mirrored over HTTPS with
their full layer closure and opened as local files. Stock OpenUSD cannot open an ``https://`` URL, and
importing ``omni.client`` does not help (its resolver only activates inside the Kit runtime) -- but
neither is required once the layers are local. That removes the container/Kit dependency from asset
discovery, auditing, and preparation entirely.

Opening assets is the expensive step (each is fetched from remote storage), so every audit is cached
to ``cache_path`` as JSON. The first resolve of a catalogue costs minutes; later ones are instant,
which is what makes run-time resolution practical inside a training script.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

# Catalogue areas that are not table-top manipulables (heavy machine parts, vehicle assemblies).
INDUSTRIAL_PATH_FRAGMENTS = (
    "Warehouse",
    "Machines",
    "Hardware",
    "engineComponent",
    "drivetrain",
    "airSystem",
    "coolingSystem",
    "electricalSystem",
    "fuelSystem",
    "brakeSystem",
    "exhaust",
    "transmission",
)

# A wide net of table-top phrases: the index is appearance-ranked, so coverage comes from breadth.
DEFAULT_PHRASES = (
    "food",
    "grocery",
    "can",
    "canned food",
    "box",
    "boxed food",
    "carton",
    "bottle",
    "jar",
    "cup",
    "fruit",
    "vegetable",
    "snack",
    "candy",
    "cereal",
    "spice",
    "tea box",
    "coffee",
    "milk",
    "juice",
    "package",
    "container",
    "bowl",
    "toy",
    "block",
    "soap",
    "medicine",
    "tube",
    "tin",
    "packet",
)


@dataclass
class ObjectSpec:
    """Physical facts about one asset, gathered by opening it (the service does not expose these)."""

    url: str
    body_path: str | None
    """Prim path of the ``PhysicsRigidBodyAPI`` body, or ``None`` for multibody assets that ship
    colliders but nothing dynamic (not simulatable as published)."""
    mass: float | None
    """Authored mass [kg], trusted as real data."""
    dims: tuple[float, float, float]
    """World-space bounding-box extents [m]."""
    fet003_passed: bool
    """Latest-dated ``FET003_BASE_PHYSX`` verdict. The service matches "passed on *some* date", so an
    asset that later regressed still comes back as a hit — this is the authoritative check."""

    @property
    def max_dim(self) -> float:
        return max(self.dims)

    @property
    def min_dim(self) -> float:
        return min(self.dims)


class SimReadyObjectLibrary:
    """Search, audit, filter, and prepare SimReady assets for a manipulation task.

    Args:
        cache_path: JSON file holding audit results, so each asset is opened at most once, ever.
        service_url: SimReady search endpoint. The production endpoint is public from corpnet; the
            dev endpoint is stale.
    """

    def __init__(
        self,
        cache_path: str = "logs/objset/audit_cache.json",
        service_url: str = "https://search.simready.omniverse.nvidia.com/",
        download_dir: str = "logs/objset/assets",
    ):
        self.cache_path = cache_path
        self.service_url = service_url
        self.download_dir = download_dir
        self._cache: dict[str, dict] = {}
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                self._cache = json.load(f)

    def search(self, phrases=DEFAULT_PHRASES, max_height: float = 0.15, per_phrase: int = 100) -> list[str]:
        """Return distinct candidate URLs, using the server-side filters the service does support.

        ``SearchFilterHeight`` maps to a real bounding-box filter and ``FET003_BASE_PHYSX`` restricts
        to single-body PhysX-ready assets (``FET004`` is the *multibody* feature — the wrong default
        for a task that needs one rigid body per object).
        """
        from simready.search import (
            AssetLibrary,
            SearchFilterFeature,
            SearchFilterHeight,
            SearchFilterPathContains,
            SearchFilterPhrase,
        )

        lib = AssetLibrary(raise_on_network_error=True)
        lib.add_service_source(self.service_url)
        # Push exclusions to the service via ``exclude_any`` rather than post-filtering locally: it
        # keeps industrial assets out of the result budget, so ``per_phrase`` is spent on candidates
        # that can actually be used.
        excludes = [SearchFilterPathContains(frag) for frag in INDUSTRIAL_PATH_FRAGMENTS]
        found: dict[str, str] = {}
        for phrase in phrases:
            try:
                matches = lib.search(
                    include_all=[
                        SearchFilterPhrase(phrase),
                        SearchFilterHeight(maximum=max_height),
                        SearchFilterFeature("FET003_BASE_PHYSX"),
                    ],
                    exclude_any=excludes,
                    max_count=per_phrase,
                )
            except Exception as exc:  # noqa: BLE001 — one bad phrase must not sink the sweep
                print(f"[WARN] search phrase {phrase!r} failed: {str(exc)[:70]}")
                continue
            for match in matches:
                path = match.asset_path
                if not any(frag in path for frag in INDUSTRIAL_PATH_FRAGMENTS):  # belt-and-braces
                    found[path.rsplit("/", 1)[-1]] = path
        return sorted(found.values())

    def fetch(self, url: str) -> str | None:
        """Mirror an asset and its whole layer closure locally, over plain HTTPS.

        SimReady assets are layered (``wrapper`` -> ``payloads/base.usda`` -> ``Physics/physics.usda``),
        so downloading only the top ``.usd`` yields a stage with no physics and a degenerate bounding
        box. Following the sublayer / reference / payload graph and mirroring relative paths under
        ``download_dir`` reproduces the asset exactly.

        This is what keeps the pipeline **Kit-free**: stock OpenUSD cannot open an ``https://`` URL and
        importing ``omni.client`` is not enough (its resolver only activates inside the Kit runtime),
        but neither is needed once the layers are local files.
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
            local = os.path.join(self.download_dir, relative)
            if root_local is None:
                root_local = local
            os.makedirs(os.path.dirname(local), exist_ok=True)
            if not os.path.exists(local):
                try:
                    urllib.request.urlretrieve(current, local)
                except Exception:  # noqa: BLE001 — a missing shared material must not sink the asset
                    continue
            try:
                sublayers, references, payloads = UsdUtils.ExtractExternalReferences(local)
            except Exception:  # noqa: BLE001 — non-USD payloads (textures) have no references
                continue
            base = current.rsplit("/", 1)[0] + "/"
            for dep in list(sublayers) + list(references) + list(payloads):
                if dep.startswith(("http://", "https://")):
                    pending.append(dep)
                elif not os.path.isabs(dep):
                    pending.append(urllib.parse.urljoin(base, dep))
        return root_local if root_local and os.path.exists(root_local) else None

    def audit(self, url: str) -> ObjectSpec | None:
        """Return the physical facts for one asset, downloading it only on a cache miss.

        Runs without the simulation app: :meth:`fetch` mirrors the layer closure over HTTPS and the
        stage is opened from local files.
        """
        if url in self._cache:
            cached = self._cache[url]
            return None if cached is None else ObjectSpec(**{**cached, "dims": tuple(cached["dims"])})
        spec = self._open_and_audit(url)
        self._cache[url] = None if spec is None else asdict(spec)
        return spec

    def _open_and_audit(self, url: str) -> ObjectSpec | None:
        from pxr import Usd, UsdGeom

        local = self.fetch(url)
        if local is None:
            return None
        try:
            stage = Usd.Stage.Open(local, Usd.Stage.LoadAll)
        except Exception:  # noqa: BLE001 — unreadable asset is simply not a candidate
            return None
        if stage is None or stage.GetDefaultPrim() is None:
            return None
        body = mass = None
        for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
            if "PhysicsRigidBodyAPI" in [str(s) for s in prim.GetAppliedSchemas()]:
                body = str(prim.GetPath())
            attr = prim.GetAttribute("physics:mass")
            if attr and attr.IsValid() and attr.Get():
                mass = float(attr.Get())
        size = (
            UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
            .ComputeWorldBound(stage.GetDefaultPrim())
            .ComputeAlignedRange()
            .GetSize()
        )
        return ObjectSpec(url, body, mass, (size[0], size[1], size[2]), _latest_fet003(stage))

    def save_cache(self) -> None:
        """Persist audits so a later resolve costs no asset opens."""
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self._cache, f)

    def select(
        self,
        target: int,
        candidates: list[str] | None = None,
        phrases=None,
        size_range: tuple[float, float] = (0.02, 0.15),
        mass_range: tuple[float, float] = (0.005, 3.0),
        heavy_from: float = 1.5,
        heavy_frac: float = 0.10,
        distinct_families: bool = False,
        verbose: bool = True,
    ) -> list[ObjectSpec]:
        """Audit candidates and keep ``target`` objects the robot can work with.

        ``mass_range`` and ``size_range`` express *robot capability*: the upper mass bound is what the
        gripper can hold, not a judgement about the asset. ``heavy_from``/``heavy_frac`` reserve a
        slice of the set for objects at the edge of that capability, so the policy meets things it
        cannot always lift instead of a world where everything succeeds.

        ``distinct_families`` additionally keeps at most one asset per product family. The catalogue
        ships many near-identical variants (``Golf_Ball``/``Golf_Ball_A01``..``A04``,
        ``Boxed_Drink_A01``..``E01``), so a set of unique *files* can still render as a grid of
        look-alikes -- file uniqueness is not visual variety. Use it for demos and for any set small
        enough that repetition would dominate.
        """
        candidates = (
            candidates
            if candidates is not None
            else self.search(phrases=phrases or DEFAULT_PHRASES, max_height=size_range[1])
        )
        kept: list[ObjectSpec] = []
        dropped: dict[str, int] = {}
        for i, url in enumerate(candidates):
            spec = self.audit(url)
            reason = _rejection_reason(spec, size_range, mass_range)
            if reason is None:
                kept.append(spec)
            else:
                dropped[reason] = dropped.get(reason, 0) + 1
            if verbose and (i + 1) % 25 == 0:
                print(f"[INFO] audited {i + 1}/{len(candidates)}, kept {len(kept)}")
        self.save_cache()

        if distinct_families:
            by_family: dict[str, ObjectSpec] = {}
            for spec in kept:
                by_family.setdefault(_family(spec.url), spec)
            if verbose:
                print(f"[INFO] {len(by_family)} distinct families among {len(kept)} usable assets")
            kept = list(by_family.values())

        heavy = sorted((k for k in kept if k.mass >= heavy_from), key=lambda k: k.mass)
        light = sorted((k for k in kept if k.mass < heavy_from), key=lambda k: k.mass)
        n_heavy = min(len(heavy), int(round(target * heavy_frac)))
        selected = light[: target - n_heavy] + heavy[:n_heavy]
        if verbose:
            print(
                f"[INFO] candidates={len(candidates)} kept={len(kept)} selected={len(selected)} "
                f"({len(selected) - n_heavy} light, {n_heavy} at/above {heavy_from} kg)"
            )
            for reason, count in sorted(dropped.items(), key=lambda kv: -kv[1]):
                print(f"[INFO]   dropped {count:4d}  {reason}")
        return selected

    def prepare(self, specs: list[ObjectSpec], out_dir: str) -> list[str]:
        """Write task-ready USDs: each asset's rigid body extracted to a uniform ``/Object`` root.

        This is the one modification heterogeneous cloning requires. Assets nest their body under a
        per-asset ``RootNode/Geometry/<name>`` path, and a per-env view cannot be built across
        varying paths. Collider, mass, and origin are left exactly as published.
        """
        from pxr import Sdf, Usd

        os.makedirs(out_dir, exist_ok=True)
        prepared: list[str] = []
        for spec in specs:
            out = os.path.join(out_dir, os.path.splitext(os.path.basename(spec.url))[0] + ".usda")
            if os.path.exists(out):
                prepared.append(out)
                continue
            stage = Usd.Stage.Open(self.fetch(spec.url), Usd.Stage.LoadAll)
            for _ in range(8):  # de-instance; instancing can nest, so repeat until stable
                changed = False
                for prim in stage.Traverse():
                    if prim.IsInstanceable():
                        prim.SetInstanceable(False)
                        changed = True
                if not changed:
                    break
            layer = Sdf.Layer.CreateNew(out)
            Sdf.CreatePrimInLayer(layer, "/Object")
            Sdf.CopySpec(stage.Flatten(), Sdf.Path(spec.body_path), layer, Sdf.Path("/Object"))
            layer.Save()
            obj_stage = Usd.Stage.Open(out)
            obj_stage.SetDefaultPrim(obj_stage.GetPrimAtPath("/Object"))
            obj_stage.GetRootLayer().Save()
            prepared.append(out)
        return prepared


def _family(url: str) -> str:
    """Product family for an asset URL, with the variant code stripped.

    ``Golf_Ball_A03`` and ``Golf_Ball`` are the same family; so are ``Boxed_Drink_A01`` and
    ``Boxed_Drink_E01``. Grouping on this is what turns "12 unique files" into "12 objects that look
    different", which is the property a viewer actually judges.
    """
    import re

    directory = url.rsplit("/", 2)[-2]
    directory = re.sub(r"_[A-Z]?\d+$", "", directory)
    directory = re.sub(r"_[A-Z]\d+$", "", directory)
    return directory.lower().replace("_", "")


def _latest_fet003(stage) -> bool:
    """Latest-dated ``FET003_BASE_PHYSX`` verdict; ``True`` when the asset carries no history."""
    meta = (stage.GetRootLayer().customLayerData or {}).get("SimReady_Metadata")
    if meta is None:
        return True
    meta = json.loads(meta) if isinstance(meta, str) else meta
    validated = (meta.get("validation") or {}).get("validated_features") or {}
    if not validated:
        return True
    return validated[max(validated)].get("FET003_BASE_PHYSX", {}).get("passed") is not False


def _rejection_reason(spec, size_range, mass_range) -> str | None:
    """Why this asset is unusable for the task, or ``None`` when it passes every filter."""
    if spec is None:
        return "open-failed"
    if spec.body_path is None:
        return "no-rigid-body (multibody: colliders but nothing dynamic)"
    if not spec.fet003_passed:
        return "latest FET003_BASE_PHYSX failed"
    if spec.max_dim > size_range[1]:
        return "too large for the workspace"
    if spec.min_dim < size_range[0]:
        return "too thin for stable contact"
    if spec.mass is None:
        return "no authored mass"
    if spec.mass > mass_range[1]:
        return "heavier than the gripper can hold"
    if spec.mass < mass_range[0]:
        return "too light for stable contact"
    return None
