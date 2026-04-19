"""Generate visualizations for all trained models that lack them."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]

def save_json_metrics_plots(model_name, model_dir, out_dir):
    """Generate standard classification plots from metrics.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)

    # Determine where test metrics live
    if "test" in metrics:
        test_m = metrics["test"]
        val_m = metrics.get("validation", {})
    else:
        test_m = metrics
        val_m = metrics.get("validation", metrics)

    # --- 1. Confusion matrix ---
    if "confusion_matrix" in test_m:
        cm = np.array(test_m["confusion_matrix"])
        report = test_m.get("classification_report", {})
        labels = [k for k in report if k not in ("accuracy", "macro avg", "weighted avg")]
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{model_name} — Test Confusion Matrix")
        plt.xticks(rotation=35, ha="right", fontsize=8)
        plt.yticks(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "confusion_matrix_test.png", dpi=150)
        plt.close(fig)
        print(f"  ✓ {model_name}/confusion_matrix_test.png")

    # --- 2. Per-class F1 ---
    if "classification_report" in test_m:
        report = test_m["classification_report"]
        classes = [k for k in report if k not in ("accuracy", "macro avg", "weighted avg")]
        f1s = [report[c]["f1-score"] for c in classes]
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(classes, f1s, color=sns.color_palette("muted", len(classes)))
        ax.set_xlabel("F1-score")
        ax.set_title(f"{model_name} — Per-Class F1 (Test)")
        ax.set_xlim(0, 1.05)
        for bar, v in zip(bars, f1s):
            ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / "per_class_f1_test.png", dpi=150)
        plt.close(fig)
        print(f"  ✓ {model_name}/per_class_f1_test.png")

    # --- 3. Precision & Recall ---
    if "classification_report" in test_m:
        report = test_m["classification_report"]
        classes = [k for k in report if k not in ("accuracy", "macro avg", "weighted avg")]
        prec = [report[c]["precision"] for c in classes]
        rec = [report[c]["recall"] for c in classes]
        x = np.arange(len(classes))
        w = 0.35
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - w/2, prec, w, label="Precision", color="#5b9bd5")
        ax.bar(x + w/2, rec, w, label="Recall", color="#ed7d31")
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.1)
        ax.set_title(f"{model_name} — Precision & Recall (Test)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "precision_recall_test.png", dpi=150)
        plt.close(fig)
        print(f"  ✓ {model_name}/precision_recall_test.png")

    # --- 4. CV fold comparison ---
    cv_path = model_dir / "cv_results.json"
    if cv_path.exists():
        with open(cv_path, encoding="utf-8") as f:
            cv = json.load(f)
        folds = cv.get("folds", cv.get("fold_results", []))
        if folds:
            fold_df = pd.DataFrame(folds)
            metric_cols = [c for c in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"] if c in fold_df.columns]
            if metric_cols:
                fig, ax = plt.subplots(figsize=(9, 5))
                fold_df[metric_cols].plot(kind="bar", ax=ax, width=0.7)
                ax.set_xlabel("Fold")
                ax.set_ylabel("Score")
                ax.set_title(f"{model_name} — 5-Fold CV Metrics")
                fold_labels = fold_df.get("fold", range(1, len(fold_df)+1))
                ax.set_xticklabels([f"Fold {i}" for i in fold_labels], rotation=0)
                ax.set_ylim(0.5, 1.0)
                ax.legend(loc="lower right", fontsize=8)
                fig.tight_layout()
                fig.savefig(out_dir / "cv_fold_comparison.png", dpi=150)
                plt.close(fig)
                print(f"  ✓ {model_name}/cv_fold_comparison.png")

    # --- 5. Validation vs Test summary ---
    summary_metrics = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
    if val_m and all(m in val_m for m in summary_metrics) and all(m in test_m for m in summary_metrics):
        val_vals = [val_m[m] for m in summary_metrics]
        test_vals = [test_m[m] for m in summary_metrics]
        x = np.arange(len(summary_metrics))
        w = 0.35
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - w/2, val_vals, w, label="Validation", color="#70ad47")
        ax.bar(x + w/2, test_vals, w, label="Test", color="#4472c4")
        ax.set_xticks(x)
        ax.set_xticklabels(summary_metrics, fontsize=9)
        ax.set_ylim(0.5, 1.0)
        ax.set_ylabel("Score")
        ax.set_title(f"{model_name} — Validation vs Test")
        ax.legend()
        for i, (v, t) in enumerate(zip(val_vals, test_vals)):
            ax.text(i - w/2, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
            ax.text(i + w/2, t + 0.005, f"{t:.3f}", ha="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "validation_vs_test.png", dpi=150)
        plt.close(fig)
        print(f"  ✓ {model_name}/validation_vs_test.png")


def save_kmeans_plots(model_dir, out_dir):
    """Copy/regenerate KMeans-specific plots."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)

    # Cluster selection metrics (elbow + silhouette)
    candidates = metrics.get("candidate_results", [])
    if candidates:
        ks = [c["n_clusters"] for c in candidates]
        inertias = [c["inertia"] for c in candidates]
        silhouettes = [c["silhouette_score"] for c in candidates]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(ks, inertias, "bo-")
        ax1.set_xlabel("Number of Clusters (k)")
        ax1.set_ylabel("Inertia")
        ax1.set_title("Elbow Method")
        ax2.plot(ks, silhouettes, "ro-")
        ax2.set_xlabel("Number of Clusters (k)")
        ax2.set_ylabel("Silhouette Score")
        ax2.set_title("Silhouette Score vs k")
        fig.tight_layout()
        fig.savefig(out_dir / "cluster_selection_metrics.png", dpi=150)
        plt.close(fig)
        print(f"  ✓ kmeans/cluster_selection_metrics.png")

    # Cluster sizes
    summary_path = model_dir / "cluster_summary.json"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        clusters = summary if isinstance(summary, list) else summary.get("clusters", [])
        if clusters:
            labels = [f"Cluster {c.get('cluster', i)}" for i, c in enumerate(clusters)]
            sizes = [c.get("size", c.get("count", 0)) for c in clusters]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(labels, sizes, color=sns.color_palette("Set2", len(labels)))
            ax.set_ylabel("Number of Samples")
            ax.set_title("KMeans — Cluster Sizes")
            for i, v in enumerate(sizes):
                ax.text(i, v + 5, str(v), ha="center", fontsize=9)
            fig.tight_layout()
            fig.savefig(out_dir / "cluster_sizes.png", dpi=150)
            plt.close(fig)
            print(f"  ✓ kmeans/cluster_sizes.png")


# ── Run for each model ────────────────────────────────────────
if __name__ == "__main__":
    VIZ_ROOT = ROOT / "visualizations" / "ml"

    # Random Forest
    rf_dir = ROOT / "data" / "models" / "property_buyer" / "random_forest_property"
    if (rf_dir / "metrics.json").exists():
        print("Random Forest:")
        save_json_metrics_plots("Random Forest", rf_dir, VIZ_ROOT / "random_forest")

    # CatBoost
    cb_dir = ROOT / "data" / "models" / "property_buyer" / "catboost_full"
    if (cb_dir / "metrics.json").exists():
        print("CatBoost:")
        save_json_metrics_plots("CatBoost", cb_dir, VIZ_ROOT / "catboost")

    # Logistic Regression
    lr_dir = ROOT / "data" / "models" / "property_buyer" / "logistic_regression_full_v1"
    if (lr_dir / "metrics.json").exists():
        print("Logistic Regression:")
        save_json_metrics_plots("Logistic Regression", lr_dir, VIZ_ROOT / "logistic_regression")

    # KMeans
    km_dir = ROOT / "data" / "models" / "property_buyer" / "kmeans_full_v1"
    if (km_dir / "metrics.json").exists():
        print("KMeans:")
        save_kmeans_plots(km_dir, VIZ_ROOT / "kmeans")

    print("\nDone.")
