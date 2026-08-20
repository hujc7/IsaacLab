Added
^^^^^

* Added fixed-tendon actuation to the Shadow Hand tasks. The hand's twenty motors drive sixteen
  joints and four tendons, so each manager-based task pairs a joint action term with a fixed-tendon
  action term -- one pair per hand, so handover carries two -- and the direct tasks apply the joint
  and tendon halves of the action in turn. Without the tendon term the eight joints coupled by a
  tendon took no command at all.

Fixed
^^^^^

* Fixed the direct Shadow Hand reorientation and handover tasks rescaling their whole twenty-motor
  action against their sixteen actuated joints, which raised a shape error on the first step.

* Fixed the Shadow Hand reorientation task spawning the hand in an orientation that left the palm
  facing sideways on the current asset, so the object could not be held.

Fixed
^^^^^

* Fixed the Shadow Hand reorientation and handover tasks diverging on PhysX. The hand's
  configuration previously raised its own solver iteration count; with that override removed the
  default budget is not enough for twenty-four joints under finger-object contact, and training
  ended with non-finite observations. The physics preset now clamps the scene's minimum position
  iterations and enables stabilization, so the setting is visible alongside the other solver
  choices instead of being attached to the asset.
