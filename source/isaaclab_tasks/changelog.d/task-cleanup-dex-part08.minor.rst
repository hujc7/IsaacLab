Added
^^^^^

* Added manager-based ``Isaac-Reorient-Cube-Shadow``,
  ``Isaac-Reorient-Cube-Shadow-OpenAI-FF``, and
  ``Isaac-Reorient-Cube-Shadow-OpenAI-LSTM`` environments.
* Added manager terms and backend presets for matching the corresponding
  Shadow Hand Direct task observations, actions, resets, and terminations.

Fixed
^^^^^

* Fixed manager-based reorientation ``Metrics/success_rate`` to report the
  per-episode success bit, including a goal reached on the terminal step, as
  reported by the Direct tasks.
