# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Shared ANSI helpers for colored log badges.

Used by role-tagged memory logs (relax/utils/memory_utils.py) and by agentic
event logs (relax/agentic/__init__.py). Keeping the palette in one place
ensures the same role has the same color across every log line, which is what
makes cross-role hangs (e.g. actor vs critic wake_up races) actually readable
in a shared log stream.
"""

from __future__ import annotations


_ANSI_RESET = "\033[0m"
_ANSI_BADGE_FG = "\033[1;38;5;231m"  # bright white, bold

# Fixed palette per role. Picked from xterm-256 so colors are distinct at a
# glance in a dark terminal. Add new roles here rather than hashing — a stable
# mapping matters more than automatic assignment.
_ROLE_BG = {
    "actor": 24,  # steel blue
    "actor_fwd": 25,  # steel blue, lighter
    "critic": 22,  # forest green
    "rollout": 54,  # deep purple
    "genrm": 130,  # burnt orange
    "advantages": 240,  # neutral grey
    "reference": 88,  # dark red
    "teacher": 94,  # olive
    "agentic": 54,  # deep purple (same as rollout — agentic is rollout-side)
}
_DEFAULT_BG = 60  # dim purple-grey fallback for unregistered labels


def _bg_for(label: str) -> int:
    return _ROLE_BG.get(label.lower(), _DEFAULT_BG)


def format_badge(label: str, *, bg: int | None = None) -> str:
    """Render ``label`` as a filled ANSI badge (bright white on bg color).

    Pass an explicit ``bg`` (xterm-256 code) to override the role palette —
    useful for one-off event badges that don't correspond to a role.
    """
    color = bg if bg is not None else _bg_for(label)
    return f"\033[48;5;{color}m{_ANSI_BADGE_FG} {label.upper()} {_ANSI_RESET}"


def format_role_tag(role: str) -> str:
    """Compact single-word role badge, e.g. ``[ACTOR]`` colored.

    Distinct from ``format_badge`` in that it uses ``[...]`` framing (no inner
    padding) — cheaper visually for prefixing every log line.
    """
    color = _bg_for(role)
    return f"\033[48;5;{color}m{_ANSI_BADGE_FG}[{role.upper()}]{_ANSI_RESET}"
