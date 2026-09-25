# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Plot the training-continuity timeline from train_continuity_20260925 raw
data.

Inputs : train_events.json (step/rollout log-tail timestamps, unix epoch),
         events.json (monitor events: scaling boundaries, engines snapshots)
Output : continuity-chart.png (committed next to the raw data it renders)

Time axis is relative to the monitor start (events.json ``monitor_start``).
Scaling windows are drawn from the monitor's own request lifetimes
(scale_out submitted -> ACTIVE, scale_in submitted -> COMPLETED); training
event times come from tailing the ray job driver log, an independent source
from the service-side engines snapshots.
"""

import json
import os

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = os.path.dirname(os.path.abspath(__file__))
events = json.load(open(os.path.join(HERE, "events.json")))
train_events = json.load(open(os.path.join(HERE, "train_events.json")))

T0 = next(e["t"] for e in events if e["event"] == "monitor_start")


def t(ev_name, **match):
    for e in events:
        if e["event"] == ev_name and all(e.get(k) == v for k, v in match.items()):
            return e["t"] - T0
    return None


t_base = t("baseline_engines")
t_first_step = t("first_step_seen")
t_so_submit = t("scale_out_submitted")
t_so_active = t("scale_out_final", status="ACTIVE")
t_si_submit = t("scale_in_submitted")
t_si_done = t("scale_in_final", status="COMPLETED")

steps = [(e["ts"] - T0, int(e["kind"].split()[1])) for e in train_events if e["kind"].startswith("step ")]
# Each rollout index logs two lines in the job driver log (batch start and
# result); dedupe by index and keep the first (batch-start) timestamp.
rollout_first = {}
for e in train_events:
    if e["kind"].startswith("rollout "):
        idx = int(e["kind"].split()[1])
        rollout_first.setdefault(idx, e["ts"] - T0)
rollouts = sorted(rollout_first.values())
t_done = next(e["ts"] - T0 for e in train_events if e["kind"] == "All training steps finished")

snaps = [e for e in events if e["event"] == "engines_snapshot"]
snap_ts = [e["t"] - T0 for e in snaps]
cap = [e["current"] for e in snaps]
INIT_PORT, ELASTIC_PORT = "node-0:16000", "node-0:16001"
served_init = [e.get("served", {}).get(INIT_PORT) for e in snaps]
served_el = [e.get("served", {}).get(ELASTIC_PORT) for e in snaps]

C_BLUE, C_ORANGE, C_GREEN, C_RED, C_GRAY = "#1f5fb4", "#e07b28", "#1a8f5c", "#c23a3a", "#8a8f98"

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10.4, 6.3), sharex=True, gridspec_kw={"height_ratios": [1.15, 1], "hspace": 0.13}
)

XMAX = max(t_done, snap_ts[-1]) + 12

# ---- panel 1: capacity + scaling windows + training-event ticks ----
for x0, x1, label, color, a in (
    (t_so_submit, t_so_active, "scale-out window (submitted -> ACTIVE)", "#dce9f8", 1.0),
    (t_si_submit, t_si_done, "scale-in window", "#fdeee0", 1.0),
):
    ax1.axvspan(x0, x1, color=color, alpha=a, lw=0)
    ax1.annotate(
        label,
        xy=((x0 + x1) / 2, 2.36),
        ha="center",
        va="center",
        fontsize=8.2,
        color="#333333",
        rotation=0,
    )
# scale-in window is ~1 s wide; annotate it with a leader line.
ax1.annotate(
    "scale-in window\n(1 s drain)",
    xy=(t_si_done, 1.78),
    xytext=(t_si_done + 38, 2.18),
    fontsize=8.2,
    color="#333333",
    arrowprops=dict(arrowstyle="-", color=C_GRAY, lw=0.8),
)

ax1.step(snap_ts, cap, where="post", color=C_BLUE, lw=2.0, label="GenRM capacity (engines)")
ax1.set_ylim(0.6, 2.6)
ax1.set_yticks([1, 2])
ax1.set_ylabel("GenRM engine capacity", fontsize=10)

# Training events: steps as tall ticks up top, rollouts as short ticks.
ax1.vlines([s for s, _ in steps], 2.0, 2.32, color=C_GREEN, lw=1.6, label="train step (log-tailed)")
ax1.vlines(rollouts, 2.0, 2.16, color=C_ORANGE, lw=1.4, label="rollout (log-tailed)")
ax1.annotate(
    f"first step t={t_first_step:.0f}s",
    xy=(steps[0][0], 2.32),
    xytext=(steps[0][0] - 14, 2.5),
    fontsize=8,
    color=C_GREEN,
    ha="left",
)
ax1.annotate(
    f"training finished t={t_done:.0f}s",
    xy=(t_done, 2.32),
    xytext=(t_done - 46, 2.5),
    fontsize=8,
    color=C_GREEN,
    ha="left",
)
ax1.legend(loc="lower right", fontsize=8.2, framealpha=0.95)
ax1.set_title(
    "Training continuity through live GenRM scaling — steps and rollouts keep landing\n"
    "(real DAPO + GenRM recipe, actor TP1xDP1 + rollout + GenRM, Qwen3-0.6B, 4x RTX 4090)",
    fontsize=10.5,
    pad=10,
)

# ---- panel 2: per-engine served counters ----
ax2.step(snap_ts, served_init, where="post", color=C_BLUE, lw=1.8, label="initial engine :16000 (cumulative served)")
ax2.step(snap_ts, served_el, where="post", color=C_RED, lw=1.8, label="elastic engine :16001 (cumulative served)")
ax2.axvline(t_si_done + 40, color=C_GRAY, lw=0)
# The elastic engine's real reward traffic is proven by the monitor's direct
# /engines observation (snapshot sampling can miss the tail).
t_el_served = t("elastic_served_check")
ax2.plot([t_el_served], [1], "o", ms=7, color=C_RED, zorder=5)
ax2.annotate(
    "elastic engine observed serving\na reward request (served=1)",
    xy=(t_el_served, 1),
    xytext=(t_el_served + 26, 2.6),
    fontsize=8.2,
    color=C_RED,
    arrowprops=dict(arrowstyle="->", color=C_RED, lw=0.9),
)
ax2.set_ylabel("cumulative served requests", fontsize=10)
ax2.set_xlabel("seconds since monitor start (run raysubmit_HeLEsxfc5fpGNfLA)", fontsize=10)
ax2.set_xlim(0, XMAX)
ax2.set_ylim(-0.4, max(10, (served_init[-1] or 0) + 2))
ax2.legend(loc="upper left", fontsize=8.2, framealpha=0.95)
ax2.grid(axis="y", color="#dddddd", lw=0.5)

for ax in (ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.savefig(os.path.join(HERE, "continuity-chart.png"), dpi=150, bbox_inches="tight")
print("wrote continuity-chart.png")
