"""Figures. Every bar carries its interval; a bare bar is a claim without evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
PALETTE = ["#2d6a9f", "#c0663a", "#4e8f6d", "#8a6bab", "#a8a029"]


def _err(points, los, his):
    lower = [max(0.0, p - lo) for p, lo in zip(points, los)]
    upper = [max(0.0, hi - p) for p, hi in zip(points, his)]
    return [lower, upper]


def plot_by_hint(summary: Dict, out: Path = None) -> Path:
    out = out or FIG_DIR / "{r}_by_hint.png".format(r=summary["run"])
    out.parent.mkdir(parents=True, exist_ok=True)
    hints = list(summary["by_hint"])
    det = summary["primary_detector"]

    flips = [summary["by_hint"][h]["flip_rate"] for h in hints]
    acks = [summary["by_hint"][h]["acknowledgement"][det] for h in hints]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, data, title, colour in (
            (axes[0], flips, "Flip rate to hinted option", PALETTE[0]),
            (axes[1], acks, "Hint acknowledged in CoT | flipped", PALETTE[1])):
        pts = [d["point"] for d in data]
        ax.bar(range(len(hints)), pts, color=colour, width=0.6)
        ax.errorbar(range(len(hints)), pts,
                    yerr=_err(pts, [d["lo"] for d in data], [d["hi"] for d in data]),
                    fmt="none", ecolor="#222", capsize=5, lw=1.4)
        ax.set_xticks(range(len(hints)))
        ax.set_xticklabels(hints, rotation=18, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("rate")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        for i, d in enumerate(data):
            ax.text(i, 0.02, "n={n}".format(n=d["n"]), ha="center", fontsize=8,
                    color="white" if d["point"] > 0.12 else "#333")
    fig.suptitle("{m} on {d}  (bars: 95% clustered bootstrap CI)".format(
        m=summary["model"] or summary["provider"], d=summary["dataset"]), fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_detector_spread(summary: Dict, out: Path = None) -> Path:
    out = out or FIG_DIR / "{r}_detector_spread.png".format(r=summary["run"])
    spread = summary.get("detector_spread")
    if not spread:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    names = list(spread)
    pts = [spread[n]["point"] for n in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(names)), pts, color=PALETTE[2], width=0.5)
    ax.errorbar(range(len(names)), pts,
                yerr=_err(pts, [spread[n]["lo"] for n in names],
                          [spread[n]["hi"] for n in names]),
                fmt="none", ecolor="#222", capsize=5, lw=1.4)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("acknowledgement rate")
    ax.set_title("Same flips, three definitions of 'mentions the hint'", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_monitors(results: List[Dict], run_name: str, out: Path = None) -> Path:
    out = out or FIG_DIR / "{r}_monitors.png".format(r=run_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    names = [r["name"] for r in results]
    pts = [r["f1"] if r["f1"] == r["f1"] else 0.0 for r in results]
    colours = [PALETTE[3] if r["regime"] == "oracle" else PALETTE[0] for r in results]
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(names) + 2))
    ax.barh(range(len(names)), pts, color=colours, height=0.55)
    ax.errorbar(pts, range(len(names)),
                xerr=_err(pts, [r["f1_lo"] for r in results],
                          [r["f1_hi"] for r in results]),
                fmt="none", ecolor="#222", capsize=4, lw=1.3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 on held-out items")
    ax.set_title("Monitor vs baselines (purple = oracle regime, sees the hint)",
                 fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
