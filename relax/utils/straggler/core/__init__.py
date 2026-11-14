# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Data collection, the part that runs inside every training step; standard
library only (a CPU test checks the imports).

``timers`` bracket the framework's timer call sites (``modules`` the blocks of
every layer) -> ``recorder`` folds the brackets into this rank's ``stats`` row
-> ``collector`` gathers one row per window and keeps every rank on the same
collectives -> the primary rank's ``worker`` thread hands each closed window to
the ``analyze`` passed in; ``health`` counts the profiler's own failures and
decides when it switches itself off. ``stats`` is everything the analysis
receives.

Device events, gathers, timer names, model structure and the analysis are
passed in; the Megatron adapter and the Relax integration one level up supply
them.
"""
