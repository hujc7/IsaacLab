# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lab rules executed on an Asset Transformer working stage."""

from __future__ import annotations

import math

from isaacsim.asset.transformer import RuleConfigurationParam, RuleInterface
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from ...utils.prims import make_uninstanceable, select_usd_variants


class RelocateRigidBodyToRoot(RuleInterface):
    """Put a standalone rigid object's physics on the default prim and retain its collider hierarchy."""

    def get_configuration_parameters(self) -> list[RuleConfigurationParam]:
        return [
            RuleConfigurationParam(
                name="variants", display_name="Variant selections", param_type=dict, default_value={}
            ),
            RuleConfigurationParam(
                name="make_uninstanceable",
                display_name="Make the source subtree editable",
                param_type=bool,
                default_value=False,
            ),
        ]

    def process_rule(self) -> None:
        stage = self.source_stage
        root = stage.GetDefaultPrim()
        if not root:
            raise ValueError("Rigid body relocation requires an authored defaultPrim.")
        root_path = root.GetPath()
        params = self.args["params"]
        if params.get("variants"):
            select_usd_variants(str(root_path), params["variants"], stage=stage)
        if params.get("make_uninstanceable", False):
            make_uninstanceable(root_path, stage=stage)

        prims = list(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
        bodies = [prim for prim in prims if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(bodies) != 1:
            raise ValueError(f"Expected one rigid body at or below '{root_path}', found {len(bodies)}.")
        if any(prim.HasAPI(UsdPhysics.ArticulationRootAPI) or prim.IsA(UsdPhysics.Joint) for prim in prims):
            raise ValueError(f"Rigid body relocation does not support joints or articulations below '{root_path}'.")
        body = bodies[0]
        if body == root:
            return
        if body.IsInstanceProxy():
            raise ValueError(
                f"Rigid body '{body.GetPath()}' is inside an instance. Make the prim uninstanceable first."
            )

        # Moving the body would turn a static collider below the root into part of that body.
        static_colliders = [
            str(prim.GetPath())
            for prim in prims
            if prim.HasAPI(UsdPhysics.CollisionAPI) and not prim.GetPath().HasPrefix(body.GetPath())
        ]
        if static_colliders:
            raise ValueError(
                f"Colliders {static_colliders} are outside the rigid body '{body.GetPath()}' and would join it."
            )

        # The default prim is now the body frame. Re-express mass properties in that frame.
        offset, _ = UsdGeom.XformCache().ComputeRelativeTransform(body, root)
        rotation = Gf.Quatf(Gf.Transform(offset).GetRotation().GetQuat())
        to_root_frame = {
            "physics:centerOfMass": lambda com: Gf.Vec3f(offset.Transform(Gf.Vec3d(com)))
            if math.isfinite(com[0])
            else com,
            "physics:principalAxes": lambda axes: rotation * axes,
        }

        # The composed token list includes PhysX schemas even when their plugin is unavailable.
        applied = body.GetMetadata("apiSchemas").GetAppliedItems()
        moved = [
            schema for schema in ("PhysicsRigidBodyAPI", "PhysxRigidBodyAPI", "PhysicsMassAPI") if schema in applied
        ]
        # A mass schema on a collider describes that collider.
        if body.HasAPI(UsdPhysics.CollisionAPI) and "PhysicsMassAPI" in moved:
            moved.remove("PhysicsMassAPI")
        definitions = [Usd.SchemaRegistry().FindAppliedAPIPrimDefinition(schema) for schema in moved]
        names = {name for definition in definitions if definition for name in definition.GetPropertyNames()}
        for prop in body.GetAuthoredProperties():
            name = prop.GetName()
            if name not in names and not name.startswith("physxRigidBody:"):
                continue
            if isinstance(prop, Usd.Relationship):
                root.CreateRelationship(name).SetTargets(prop.GetTargets())
                continue
            attribute = root.CreateAttribute(name, prop.GetTypeName(), prop.IsCustom(), prop.GetVariability())
            convert = to_root_frame.get(name, lambda value: value)
            for time in [Usd.TimeCode.Default(), *map(Usd.TimeCode, prop.GetTimeSamples())]:
                value = prop.Get(time)
                if value is not None:
                    attribute.Set(convert(value), time)
        for schema in moved:
            root.AddAppliedSchema(schema)
            body.RemoveAppliedSchema(schema)
