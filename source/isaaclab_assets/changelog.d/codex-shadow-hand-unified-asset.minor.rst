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
