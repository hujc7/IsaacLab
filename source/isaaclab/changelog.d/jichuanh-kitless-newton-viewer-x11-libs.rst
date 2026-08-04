Fixed
^^^^^

* Fixed ``--visualizer newton`` failing to start in the kit-less container with
  ``AttributeError: 'NoneType' object has no attribute 'XRenderFindVisualFormat'``,
  by installing the X11 libraries the Newton viewer loads at runtime.
