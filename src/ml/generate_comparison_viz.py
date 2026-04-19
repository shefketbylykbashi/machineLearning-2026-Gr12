"""Generate cross-model comparison visualizations for the README Discussion section.

Outputs to visualizations/ml/comparison/:
    - overall_metrics_comparison.png
    - per_class_f1_comparison.png
    - cv_stability_comparison.png
    - accuracy_vs_balanced_accuracy.png
    - supervised_vs_unsupervised.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "visualizations" / "ml" / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AUTOENCODER_LABEL = "Autoencoder+kNN"

MODELS = {
    "Random Forest": ROOT / "data/models/property_buyer/random_forest_property/metrics.json",
    "Logistic Reg.": ROOT / "data/models/property_buyer/logistic_regression_full_v1/metrics.json",
    "CatBoost": ROOT / "data/models/property_buyer/catboost_full/metrics.json",
    "Neural Net": ROOT / "data/models/neural_net/metrics.json",
    AUTOENCODER_LABEL: ROOT / "data/models/autoencoder/metrics.json",
}

# Models that are unsupervised (trained without labels). Used for styling
# and for the explicit "supervised vs. unsupervised" chart.
UNSUPERVISED_MODELS = {AUTOENCODER_LABEL}

COLORS = {
    "Random Forest": "#4C72B0",
    "Logistic Reg.": "#55A868",
    "CatBoost": "#C44E52",
    "Neural Net": "#8172B2",
    AUTOENCODER_LABEL: "#DD8452",
}


def load_metrics():
    out = {}
    for name, path in MODELS.items():
        if not path.exists():
            print(f"[warn] metrics file missing for {name}: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            out[name] = json.load(f)
    return out


def _get_test_block(data: dict) -> dict:
    """Return the test-set metrics block regardless of schema variant."""
    for key in ("test", "test_metrics", "holdout"):
        if key in data and isinstance(data[key], dict):
            return data[key]
    return data


def _load_cv_summary(model_name: str, data: dict) -> dict | None:
    """Return a {metric: {'mean': .., 'std': ..}} dict for CV scores, or None.

    - RF/LR/CB store this directly under `cv_summary` in metrics.json.
    - The neural net stores it in a sibling `cv_results.json` file.
    """
    # Primary location
    if "cv_summary" in data and isinstance(data["cv_summary"], dict):
        return data["cv_summary"]

    # Fallback: sibling cv_results.json
    metrics_path = MODELS.get(model_name)
    if metrics_path is not None:
        sibling = metrics_path.parent / "cv_results.json"
        if sibling.exists():
            try:
                with open(sibling, "r", encoding="utf-8") as f:
                    cv_data = json.load(f)
                summary = cv_data.get("summary")
                if isinstance(summary, dict):
                    return summary
            except Exception:
                return None
    return None


def _get_metric(block: dict, key: str) -> float:
    """Pull a metric value that may be a float or a nested {mean, std, ...}."""
    if not isinstance(block, dict):
        return 0.0
    v = block.get(key)
    if isinstance(v, dict):
        return float(v.get("mean", 0.0))
    if v is None:
        return 0.0
    return float(v)


def plot_overall_metrics(metrics: dict) -> None:
    metric_keys = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
    labels = ["Accuracy", "Balanced Acc.", "Macro F1", "Weighted F1"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    n_models = len(metrics)
    width = 0.8 / max(n_models, 1)

    for i, (name, data) in enumerate(metrics.items()):
        test = _get_test_block(data)
        vals = []
        for mk in metric_keys:
            v = test.get(mk)
            if v is None and mk == "weighted_f1":
                cv = _load_cv_summary(name, data) or {}
                v = _get_metric(cv, "weighted_f1")
            vals.append(float(v or 0.0))
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=name, color=COLORS[name])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Test-set metrics across all models (supervised + unsupervised)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "overall_metrics_comparison.png", dpi=120)
    plt.close(fig)


def plot_per_class_f1(metrics: dict) -> None:
    # Collect classes from the first model's classification report
    first = next(iter(metrics.values()))
    test = _get_test_block(first)
    report = test.get("classification_report", {})
    class_names = [c for c in report.keys()
                   if c not in ("accuracy", "macro avg", "weighted avg")]

    fig, ax = plt.subplots(figsize=(14, 6.5))
    x = np.arange(len(class_names))
    n_models = len(metrics)
    width = 0.8 / max(n_models, 1)

    for i, (name, data) in enumerate(metrics.items()):
        test = _get_test_block(data)
        report = test.get("classification_report", {})
        vals = [float(report.get(c, {}).get("f1-score", 0.0)) for c in class_names]
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=name, color=COLORS[name])

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("__", "\n") for c in class_names], fontsize=9)
    ax.set_ylabel("F1 score (test set)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class F1 by model (test set)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "per_class_f1_comparison.png", dpi=120)
    plt.close(fig)


def plot_cv_stability(metrics: dict) -> None:
    means, stds, names = [], [], []
    for name, data in metrics.items():
        cv = _load_cv_summary(name, data)
        if not cv:
            continue
        f1_block = cv.get("macro_f1")
        if isinstance(f1_block, dict):
            mean = f1_block.get("mean")
            std = f1_block.get("std")
        else:
            mean = cv.get("macro_f1_mean") or f1_block
            std = cv.get("macro_f1_std", 0.0)
        if mean is None:
            continue
        names.append(name)
        means.append(float(mean))
        stds.append(float(std or 0.0))

    if not names:
        print("[warn] no CV data found, skipping cv_stability plot")
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=10,
                  color=[COLORS[n] for n in names], alpha=0.85,
                  error_kw={"elinewidth": 2, "ecolor": "#333"})
    for b, m, s in zip(bars, means, stds):
        ax.text(b.get_x() + b.get_width() / 2, m + s + 0.01,
                f"{m:.3f} ± {s:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Macro F1 (5-fold CV)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Cross-validation stability: macro F1 mean ± std")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cv_stability_comparison.png", dpi=120)
    plt.close(fig)


def plot_accuracy_vs_balanced(metrics: dict) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    for name, data in metrics.items():
        test = _get_test_block(data)
        acc = float(test.get("accuracy", 0.0))
        bal = float(test.get("balanced_accuracy", 0.0))
        ax.scatter(acc, bal, s=240, color=COLORS[name], edgecolor="black",
                   linewidth=1.2, zorder=3, label=name)
        ax.annotate(name, (acc, bal), textcoords="offset points",
                    xytext=(10, 8), fontsize=10)

    lims = [0.3, 1.0]
    ax.plot(lims, lims, "--", color="gray", alpha=0.6,
            label="Accuracy = Balanced Accuracy")
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xlabel("Accuracy (test)")
    ax.set_ylabel("Balanced accuracy (test)")
    ax.set_title("Overall accuracy vs. balanced accuracy\n"
                 "(distance below the dashed line = minority-class weakness)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "accuracy_vs_balanced_accuracy.png", dpi=120)
    plt.close(fig)


def plot_supervised_vs_unsupervised(metrics: dict) -> None:
    """Bar chart framing the autoencoder's latent k-NN as an explicitly
    unsupervised baseline against the four supervised models."""
    labels = ["Accuracy", "Balanced Acc.", "Macro F1"]
    mkeys = ["accuracy", "balanced_accuracy", "macro_f1"]

    rows = []
    for name, data in metrics.items():
        test = _get_test_block(data)
        kind = "unsupervised" if name in UNSUPERVISED_MODELS else "supervised"
        rows.append((name, [float(test.get(k, 0.0)) for k in mkeys], kind))

    if not any(kind == "unsupervised" for _, _, kind in rows):
        print("[warn] no unsupervised model present, skipping supervised-vs-unsupervised plot")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    n_models = len(rows)
    width = 0.8 / max(n_models, 1)

    for i, (name, vals, kind) in enumerate(rows):
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width,
                      label=f"{name} ({kind})",
                      color=COLORS.get(name, "#888"),
                      edgecolor="black", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (test set)")
    ax.set_title(
        "Supervised models vs. unsupervised autoencoder + kNN\n"
        "(autoencoder trained without labels; kNN uses labels only at eval)"
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "supervised_vs_unsupervised.png", dpi=120)
    plt.close(fig)


def main() -> None:
    metrics = load_metrics()
    plot_overall_metrics(metrics)
    plot_per_class_f1(metrics)
    plot_cv_stability(metrics)
    plot_accuracy_vs_balanced(metrics)
    plot_supervised_vs_unsupervised(metrics)
    print(f"Done: wrote comparison plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
