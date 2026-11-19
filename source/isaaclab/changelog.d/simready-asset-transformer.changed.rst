* Replaced the SimReady branch's rigid-body relocation flag and stage utility with the
  ``relocate_rigid_body_to_root`` Asset Transformer profile. To opt in, set
  ``UsdFileCfg.transformer`` to ``TransformerCfg(profile="relocate_rigid_body_to_root")``.
