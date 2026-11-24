# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""What becomes of the collected windows: detection, reporting and display.

Cheap to change or replace, since the collection does not import it (a CPU test
checks); standard library, numpy, requests and ``core`` only.

* ``detector``: which rank is slow and why, from one gathered window.
* ``report``: ``WindowAnalyzer``, the collector's ``analyze``; the window's
  scalars, timeline bars and log lines.
* ``trace``: the sampled step as a Chrome trace file.
* ``control``: the level control page and the primary rank's poller.
"""
