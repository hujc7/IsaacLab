Changed
^^^^^^^

* Changed the Shadow Hand configurations to spawn one asset and select the physics engine through
  its ``Physics`` USD variant, replacing the two separate PhysX and Newton assets whose joints were
  named differently. Both engines now spawn the hand at the same orientation; the previous assets
  needed two, because one baked a root orientation that the other did not.

Added
^^^^^

* Added :class:`~isaaclab_assets.robots.shadow_hand.ShadowHand`, which supplies everything needed to
  spawn or address the hand: the asset path, the sixteen motors that drive a joint each
  (``joint_names``), the four that drive a tendon (``tendon_names``) and their commandable range,
  the fingertip bodies, and :meth:`~isaaclab_assets.robots.shadow_hand.ShadowHand.cfg`, which
  returns the configuration for one engine. ``SHADOW_HAND_CFG`` and ``SHADOW_HAND_NEWTON_CFG``
  remain as aliases for the PhysX and Newton variants.

Removed
^^^^^^^

* Removed ``SHADOW_ACTUATED_JOINT_NAMES``, ``SHADOW_TENDON_JOINT_NAMES`` and
  ``SHADOW_PHYSX_TENDON_GEARING``. Use ``ShadowHand.joint_names`` and ``ShadowHand.tendon_names``
  instead. Note that ``SHADOW_ACTUATED_JOINT_NAMES`` listed all twenty motors, so code that fed it
  to ``find_joints`` was asking for four joints that do not exist; ``joint_names`` lists only the
  sixteen that drive a joint.

Fixed
^^^^^

* Fixed the Shadow Hand asset applying an articulation-root schema to two prims, which made any
  consumer that resolves the root by search fail with ``Expected 1 prims ... found 2`` once the
  asset was loaded in Kit. ``JointWrenchSensor`` hit this on every backend, so the manager-based
  reorientation environment could not start. The second schema carried one attribute, the Newton
  self-collision flag, which the configuration already supplies for both engines; removing both
  leaves a single articulation root.

* Removed configuration that restated the asset or the defaults: the joint drive type, which the
  asset authors on every joint; ``soft_joint_pos_limit_factor`` and ``activate_contact_sensors``,
  which repeated their defaults; and the PhysX solver iteration counts and sleep and stabilization
  thresholds, which only one engine reads. ``ShadowHand.cfg`` now returns the same configuration
  for both engines, differing only in the selected USD variant. The removed solver values are
  recorded in the Shadow Hand physics preset should a run show they are needed.
