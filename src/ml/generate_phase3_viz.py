"""Generate Phase 3 visualizations from real search-log JSON files.

Reads Phase 2 baseline metrics and Phase 3 results (search logs + metrics)
from data/models/property_buyer/phase3/<model>/ and produces:
  1. Per-model HP search trace (trial vs CV macro F1 + best-so-far line)
  2. Phase 2 vs Phase 3 overall comparison bar chart
  3. Macro F1 uplift bar chart
  4. Feature count shrinkage chart
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PHASE3 = ROOT / "data" / "models" / "property_buyer" / "phase3"
OUT = ROOT / "visualizations" / "ml" / "phase3"
OUT.mkdir(parents=True, exist_ok=True)

# ── Phase 2 baselines (test set, from Phase 2 metrics.json files) ──
P2 = {
    "CatBoost":            {"acc": 0.9563, "bacc": 0.9237, "f1m": 0.9112, "f1w": 0.9584},
    "Neural Net":          {"acc": 0.8859, "bacc": 0.8300, "f1m": 0.8271, "f1w": 0.8839},
    "Random Forest":       {"acc": 0.8232, "bacc": 0.6762, "f1m": 0.6870, "f1w": 0.8179},
    "Logistic Regression": {"acc": 0.8327, "bacc": 0.7923, "f1m": 0.7286, "f1w": 0.8394},
}

METRIC_LABELS = {"acc": "Accuracy", "bacc": "Balanced Acc.", "f1m": "Macro F1", "f1w": "Weighted F1"}

# ── helpers ──

def _load_json(path: Path) -> dict | list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_phase3_test(model_dir: str) -> dict:
    mdir = PHASE3 / model_dir
    # RF uses full-refit (no early stopping needed), others use metrics.json
    metrics_file = mdir / "metrics_full_refit.json"
    if not metrics_file.exists():
        metrics_file = mdir / "metrics.json"
    m = _load_json(metrics_file)["test"]
    fl = _load_json(mdir / "feature_list.json")
    return {
        "acc": m["accuracy"], "bacc": m["balanced_accuracy"],
        "f1m": m["macro_f1"], "f1w": m["weighted_f1"],
        "n_feat": fl["n_used"],
    }


def _load_search_log(model_dir: str) -> list[float]:
    log = _load_json(PHASE3 / model_dir / "search_log.json")
    return [entry["cv_macro_f1"] for entry in log]


# ── load real Phase 3 numbers ──
P3 = {
    "CatBoost":            _load_phase3_test("catboost"),
    "Neural Net":          _load_phase3_test("neural_net"),
    "Random Forest":       _load_phase3_test("random_forest"),
    "Logistic Regression": _load_phase3_test("logistic_regression"),
}

SEARCH_LOGS = {
    "CatBoost":            _load_search_log("catboost"),
    "Random Forest":       _load_search_log("random_forest"),
    "Logistic Regression": _load_search_log("logistic_regression"),
    "Neural Net":          _load_search_log("neural_net"),
}


# ══════════════════════════════════════════════════════════
# 1.  Per-model HP search trace
# ══════════════════════════════════════════════════════════

def hp_search_traces() -> None:
    for name, scores in SEARCH_LOGS.items():
        slug = name.lower().replace(" ", "_")
        trials = np.arange(len(scores))
        best_so_far = np.maximum.accumulate(scores)
        baseline = scores[0]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(trials, scores, alpha=0.55, s=30, color="#4c72b0",
                   label="Trial CV macro F1", zorder=3)
        ax.plot(trials, best_so_far, color="#c0392b", lw=2,
                label="Best so far", zorder=4)
        ax.axhline(baseline, ls="--", color="grey", lw=1,
                   label=f"Baseline ({baseline:.4f})")
        ax.set_xlabel("Trial #")
        ax.set_ylabel("5-fold CV macro F1")
        ax.set_title(f"{name} — hyperparameter search progress")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / f"{slug}_hp_search.png", dpi=140)
        plt.close(fig)


# ══════════════════════════════════════════════════════════
# 2.  Overall comparison (grouped bars)
# ══════════════════════════════════════════════════════════

def overall_comparison() -> None:
    models = list(P2.keys())
    metrics = list(METRIC_LABELS.keys())
    x = np.arange(len(models))
    width = 0.1

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, [P2[k][m] for k in models],
               width, label=f"Phase 2 — {METRIC_LABELS[m]}", alpha=0.55,
               color=plt.cm.Blues(0.4 + 0.15 * i))
        ax.bar(x + (i - 1.5) * width + 4 * width + 0.04,
               [P3[k][m] for k in models], width,
               label=f"Phase 3 — {METRIC_LABELS[m]}",
               color=plt.cm.Oranges(0.4 + 0.15 * i))

    ax.set_xticks(x + 2 * width)
    ax.set_xticklabels(models)
    ax.set_ylabel("Score")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Phase 2 (baseline) vs Phase 3 (tuned + pruned) — test-set metrics")
    ax.legend(ncol=2, fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "phase2_vs_phase3_metrics.png", dpi=140)
    plt.close(fig)


# ══════════════════════════════════════════════════════════
# 3.  Macro F1 uplift
# ══════════════════════════════════════════════════════════

def macro_f1_uplift() -> None:
    models = list(P2.keys())
    base = [P2[m]["f1m"] for m in models]
    tuned = [P3[m]["f1m"] for m in models]
    delta = [t - b for b, t in zip(base, tuned)]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    ax.bar(x - 0.2, base, 0.4, label="Phase 2", color="#4c72b0")
    ax.bar(x + 0.2, tuned, 0.4, label="Phase 3", color="#dd8452")
    for i, d in enumerate(delta):
        ax.annotate(f"+{d*100:.2f} pp", xy=(i, max(base[i], tuned[i]) + 0.01),
                    ha="center", fontsize=9, color="#2a9d8f")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Macro F1 (test)")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Macro F1 uplift after Phase 3 pruning + hyperparameter search")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "macro_f1_uplift.png", dpi=140)
    plt.close(fig)


# ══════════════════════════════════════════════════════════
# 4.  Feature count shrinkage
# ══════════════════════════════════════════════════════════

def feature_count_shrinkage() -> None:
    models = list(P3.keys())
    before = [52] * len(models)
    after = [P3[m]["n_feat"] for m in models]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(models))
    ax.bar(x - 0.2, before, 0.4, label="Phase 2 (full features)", color="#4c72b0")
    ax.bar(x + 0.2, after, 0.4, label="Phase 3 (pruned)", color="#dd8452")
    for i, (b, a) in enumerate(zip(before, after)):
        ax.annotate(f"−{b - a}", xy=(i + 0.2, a + 1.0), ha="center",
                    fontsize=9, color="#c0392b")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Number of features used")
    ax.set_title("Feature count: before vs. after Phase 3 pruning")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "feature_count_shrinkage.png", dpi=140)
    plt.close(fig)


# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    hp_search_traces()
    overall_comparison()
    macro_f1_uplift()
    feature_count_shrinkage()
    print(f"Wrote Phase 3 figures to {OUT}")