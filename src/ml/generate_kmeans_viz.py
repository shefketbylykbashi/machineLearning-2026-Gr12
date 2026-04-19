import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
km_dir = ROOT / "data" / "models" / "property_buyer" / "kmeans_full_v1"
out_dir = ROOT / "visualizations" / "ml" / "kmeans"
out_dir.mkdir(parents=True, exist_ok=True)

with open(km_dir / "metrics.json", encoding="utf-8") as f:
    metrics = json.load(f)
with open(km_dir / "cluster_summary.json", encoding="utf-8") as f:
    summary = json.load(f)

evals = metrics["all_cluster_evaluations"]
ks = [e["n_clusters"] for e in evals]
inertias = [e["inertia"] for e in evals]
silhouettes = [e["silhouette_score"] for e in evals]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(ks, inertias, "bo-")
ax1.axvline(5, color="red", linestyle="--", alpha=0.6, label="Selected k=5")
ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("Inertia")
ax1.set_title("Elbow Method")
ax1.legend()
ax2.plot(ks, silhouettes, "ro-")
ax2.axvline(5, color="blue", linestyle="--", alpha=0.6, label="Selected k=5")
ax2.set_xlabel("Number of Clusters (k)")
ax2.set_ylabel("Silhouette Score")
ax2.set_title("Silhouette Score vs k")
ax2.legend()
fig.tight_layout()
fig.savefig(out_dir / "cluster_selection_metrics.png", dpi=150)
plt.close(fig)
print("Done: cluster_selection_metrics.png")

cluster_names = sorted(summary.keys())
sizes = [summary[c]["size"] for c in cluster_names]
labels = [c.replace("cluster_", "Cluster ") for c in cluster_names]

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(labels, sizes, color=sns.color_palette("Set2", len(labels)))
ax.set_ylabel("Number of Samples")
ax.set_title("KMeans - Cluster Sizes (k=5)")
for i, v in enumerate(sizes):
    ax.text(i, v + 10, str(v), ha="center", fontsize=10)
fig.tight_layout()
fig.savefig(out_dir / "cluster_sizes.png", dpi=150)
plt.close(fig)
print("Done: cluster_sizes.png")

all_labels_set = set()
for c in cluster_names:
    all_labels_set.update(summary[c].get("top_reference_labels", {}).keys())
all_labels_list = sorted(all_labels_set)

matrix = []
for c in cluster_names:
    row = [summary[c].get("top_reference_labels", {}).get(l, 0) for l in all_labels_list]
    matrix.append(row)
matrix = np.array(matrix, dtype=float)
row_sums = matrix.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
matrix_norm = matrix / row_sums

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(matrix_norm, annot=True, fmt=".2f", cmap="YlOrRd",
            xticklabels=all_labels_list, yticklabels=labels, ax=ax)
ax.set_xlabel("Buyer Profile")
ax.set_ylabel("Cluster")
ax.set_title("KMeans - Buyer Profile Distribution per Cluster (Normalized)")
plt.xticks(rotation=35, ha="right", fontsize=8)
fig.tight_layout()
fig.savefig(out_dir / "cluster_profile_heatmap.png", dpi=150)
plt.close(fig)
print("Done: cluster_profile_heatmap.png")
