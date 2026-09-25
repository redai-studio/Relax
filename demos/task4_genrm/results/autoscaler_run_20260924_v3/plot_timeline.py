# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Plot the autoscaler full-cycle timeline from autoscaler_run_20260924_v3 raw
data.

Inputs : timeline.json (capacity snapshots, target ~1 Hz), events.json (phase
         boundaries), scale_history.json (decision/completion unix timestamps)
Output : timeline-chart.png (committed next to the raw data it renders)

Time axes are relative to the load start (events.json ``load_started``).
Decision/completion times come from /scale_history (unix epoch) and are mapped
to the run clock by aligning the scale-in completion with the first timeline
poll that observed capacity back at 1 (poll interval ~1 s, so +/-1 s).
Load-phase boundaries come from run events, NOT from poll discovery times.
"""

import json
import os

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = os.path.dirname(os.path.abspath(__file__))
timeline = json.load(open(os.path.join(HERE, "timeline.json")))
events = json.load(open(os.path.join(HERE, "events.json")))
scale_history = json.load(open(os.path.join(HERE, "scale_history.json")))

T0 = next(e["t"] for e in events if e["event"] == "load_started")
so = next(h for h in scale_history if h["action"] == "scale_out")
si = next(h for h in scale_history if h["action"] == "scale_in")


def t(ev_name):
    for e in events:
        if e["event"] == ev_name:
            return e["t"] - T0
    return None


# Load-phase boundaries (run events).
t_low_done = t("phase_low_done")  # end of LOW
t_high_done = t("phase_high_result")  # end of HIGH (confirmed scale-out)
t_steady_done = t("phase_steady_done")  # end of STEADY
t_load_stop = t("load_stopped")

# Decision/completion times from /scale_history, unix -> run clock via the
# scale-in-completion <-> first-poll-observing-1 anchor (+/-1 s).
seen2 = False
t_poll_back_to_1 = None
for row in timeline:
    if row["current"] == 2:
        seen2 = True
    elif seen2 and row["current"] == 1:
        t_poll_back_to_1 = row["t"]
        break
offset = t_poll_back_to_1 - si["completed_at"]
so_decision, so_completion = so["triggered_at"] + offset - T0, so["completed_at"] + offset - T0
si_decision, si_completion = si["triggered_at"] + offset - T0, si["completed_at"] + offset - T0

# Capacity + per-engine series.
ts, cap = [], []
served_init, served_elastic = [], []
for row in timeline:
    ts.append(row["t"] - T0)
    cap.append(row["current"])
    per = {p: s for _, p, s in row["engines"]}
    served_init.append(per.get(16000, None))
    served_elastic.append(per.get(16001, None))

# Elastic engine served count: events.json is authoritative (timeline snapshots
# at ~1 Hz can miss the last request).
elastic_total = next(e["elastic_served"] for e in events if e["event"] == "phase_steady_done")

C_BLUE, C_ORANGE, C_GREEN, C_RED, C_GRAY = "#1f5fb4", "#e07b28", "#1a8f5c", "#c23a3a", "#8a8f98"

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10.4, 6.3), sharex=True, gridspec_kw={"height_ratios": [1.2, 1], "hspace": 0.12}
)

# ---- panel 1: capacity, load phases, decisions, completions ----
phases = [
    (0, t_low_done, "LOW", "#eef3fb"),
    (t_low_done, t_high_done, "HIGH (saturated)", "#fdeee0"),
    (t_high_done, t_steady_done, "STEADY", "#e9f5ef"),
    (t_steady_done, t_load_stop, "LOW'", "#eef3fb"),
]
for x0, x1, name, color in phases:
    ax1.axvspan(x0, x1, color=color, zorder=0)
    ax2.axvspan(x0, x1, color=color, zorder=0)
    ax1.text((x0 + x1) / 2, 2.86, name, ha="center", va="top", fontsize=8.5, color="#444a54")

ax1.step(ts, cap, where="post", color=C_BLUE, lw=2.2, zorder=3)
ax1.fill_between(ts, cap, step="post", color=C_BLUE, alpha=0.10, zorder=2)
ax1.set_ylim(0.6, 3.05)
ax1.set_yticks([1, 2])
ax1.set_ylabel("GenRM engine count (current)", fontsize=9.5)
ax1.tick_params(labelsize=9)

# Decision times: dashed vlines. Completion times: thin solid vlines.
ax1.axvline(so_decision, color=C_GREEN, ls="--", lw=1.4, zorder=2)
ax1.axvline(si_decision, color=C_RED, ls="--", lw=1.4, zorder=2)
ax1.axvline(so_completion, color=C_GREEN, ls="-", lw=0.9, alpha=0.55, zorder=2)
ax1.axvline(si_completion, color=C_RED, ls="-", lw=0.9, alpha=0.55, zorder=2)
ax1.text(so_completion, 0.78, "op ACTIVE", rotation=90, fontsize=7, color=C_GREEN, ha="right", va="bottom")
ax1.text(si_completion, 0.78, "op COMPLETED", rotation=90, fontsize=7, color=C_RED, ha="right", va="bottom")

ax1.annotate(
    "auto scale-out 1\u21922 decided (t\u224836.5 s, in HIGH, ~16 s after onset)\ntrigger: token_usage_high",
    xy=(so_decision, 2),
    xytext=(so_decision + 6, 2.52),
    fontsize=8.2,
    color=C_GREEN,
    ha="left",
    arrowprops=dict(arrowstyle="-|>", color=C_GREEN, lw=1.1),
)
ax1.annotate(
    "auto scale-in 2\u21921 decided (t\u2248132.5 s, in STEADY):\nload diluted across 2 engines, avg token usage\n~1.8% < 5%; triggers: token_usage_low + no_queue\n+ throughput_stable",
    xy=(si_decision, 1),
    xytext=(si_decision - 52, 1.62),
    fontsize=8.2,
    color=C_RED,
    ha="left",
    arrowprops=dict(arrowstyle="-|>", color=C_RED, lw=1.1),
)
ax1.text(2, 1.13, "initial engine only \u2014 no false scale-out under LOW", fontsize=8, color="#444a54", va="bottom")
ax1.set_title(
    "GenRM autoscaler full cycle on 4\u00d7RTX 4090 (Qwen3-0.6B judge): auto 1\u21922\u21921, 3,202 requests, 0 failures",
    fontsize=10.5,
    pad=8,
)

# ---- panel 2: per-engine cumulative served ----
ax2.plot(ts, served_init, color=C_BLUE, lw=1.9, label="initial engine (port 16000)")
el_ts = [x for x, s in zip(ts, served_elastic) if s is not None]
el_s = [s for s in served_elastic if s is not None]
ax2.plot(el_ts, el_s, color=C_ORANGE, lw=1.9, label=f"elastic engine (port 16001, served {elastic_total})")
ax2.set_ylabel("Cumulative requests served", fontsize=9.5)
ax2.set_xlabel(
    f"Elapsed time since load start (s) \u2014 snapshots target ~1 Hz ({len(ts)} rows over ~{ts[-1] - ts[0]:.0f} s)",
    fontsize=9.5,
)
ax2.set_xlim(-2, t_load_stop + 3)
ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
ax2.tick_params(labelsize=9)
ax2.grid(axis="y", color="#d8dce2", lw=0.6, alpha=0.7)

ax2.annotate(
    "elastic engine drained\n(516 served; scale-in COMPLETED)",
    xy=(el_ts[-1], el_s[-1]),
    xytext=(el_ts[-1] + 9, el_s[-1] - 260),
    fontsize=8,
    color=C_ORANGE,
    arrowprops=dict(arrowstyle="-|>", color=C_ORANGE, lw=1.0),
)

for ax in (ax1, ax2):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

fig.text(
    0.012,
    0.005,
    "Shaded bands: load phases (run events). Dashed lines: autoscaler decisions; thin solid lines: operation completion\n"
    "(from /scale_history, unix\u2192run-clock aligned via scale-in completion \u2194 first poll observing capacity 1, \u00b11 s).",
    fontsize=7.2,
    color="#666b74",
)

fig.savefig(os.path.join(HERE, "timeline-chart.png"), dpi=200, bbox_inches="tight", facecolor="white")
print("written", os.path.join(HERE, "timeline-chart.png"))
print(
    f"phases end at: LOW {t_low_done:.1f}, HIGH {t_high_done:.1f}, STEADY {t_steady_done:.1f}, stop {t_load_stop:.1f}"
)
print(
    f"scale_out: decision {so_decision:.3f} completion {so_completion:.3f} (dur {so['completed_at'] - so['triggered_at']:.1f}s)"
)
print(
    f"scale_in:  decision {si_decision:.3f} completion {si_completion:.3f} (dur {si['completed_at'] - si['triggered_at']:.1f}s)"
)
print(f"scale-in inside STEADY: {t_high_done < si_decision and si_completion < t_steady_done}")
