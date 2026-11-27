"""Static PNG charts for the README."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = {"P0_status_quo": "Status quo", "P1_rbi_flat_10k": "Flat 1h ≥₹10k",
          "P2_flat_25k": "Flat 1h ≥₹25k", "P3_smart_cooling_off": "Smart Cooling-Off"}
COLORS = {"P0_status_quo": "#9aa0a6", "P1_rbi_flat_10k": "#e8a33d",
          "P2_flat_25k": "#d9774b", "P3_smart_cooling_off": "#2a7de1"}


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def make_charts(out: dict, folder: Path):
    pol = out["policies"]
    keys = list(LABELS)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar([LABELS[k] for k in keys], [pol[k]["prevention_rate"] * 100 for k in keys],
                color=[COLORS[k] for k in keys])
    axes[0].set_title("Scam value prevented (%)")
    axes[1].bar([LABELS[k] for k in keys], [pol[k]["legit_delayed_pct"] * 100 for k in keys],
                color=[COLORS[k] for k in keys])
    axes[1].set_title("Legitimate payments delayed (%)")
    for ax in axes:
        _style(ax)
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(folder / "policy_comparison.png", dpi=150)
    plt.close(fig)

    typos = list(pol["P3_smart_cooling_off"]["prevention_by_typology"])
    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.27
    for i, k in enumerate(["P1_rbi_flat_10k", "P2_flat_25k", "P3_smart_cooling_off"]):
        ax.bar([j + (i - 1) * w for j in range(len(typos))],
               [pol[k]["prevention_by_typology"].get(t, 0) * 100 for t in typos],
               width=w, label=LABELS[k], color=COLORS[k])
    ax.set_xticks(range(len(typos)), [t.replace("_", " ") for t in typos])
    ax.set_title("Scam value prevented by typology (%)")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    _style(ax)
    fig.tight_layout()
    fig.savefig(folder / "prevention_by_typology.png", dpi=150)
    plt.close(fig)

    fi = list(out["model_metrics"]["feature_importance_gain"].items())[:12][::-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh([f for f, _ in fi], [v * 100 for _, v in fi], color="#2a7de1")
    ax.set_title("Top risk signals (share of model gain, %)")
    _style(ax)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(folder / "feature_importance.png", dpi=150)
    plt.close(fig)
