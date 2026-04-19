"""Visualize the autoencoder's learned latent space and training dynamics.

Reads artifacts from data/models/autoencoder/ and writes plots to
visualizations/ml/autoencoder/:
    - loss_curve.png               training + validation reconstruction loss
    - latent_pca_by_profile.png    2-D PCA of latent space colored by buyer_profile
    - latent_tsne_by_profile.png   2-D t-SNE of latent space colored by buyer_profile
    - latent_pca_by_cluster.png    same PCA colored by KMeans cluster
    - cluster_profile_heatmap.png  KMeans cluster x buyer_profile contingency
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "data" / "models" / "autoencoder"
OUT_DIR = ROOT / "visualizations" / "ml" / "autoencoder"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_artifacts():
    with open(MODEL_DIR / "metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    npz = np.load(MODEL_DIR / "latent_representations.npz")
    return metrics, npz


def plot_loss_curve(metrics: dict) -> None:
    history = metrics["training"]["history"]
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    best_epoch = metrics["training"]["best_epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(epochs, history["train_loss"], label="Train", color="#4C72B0")
    axes[0].plot(epochs, history["val_loss"], label="Validation", color="#C44E52")
    axes[0].axvline(best_epoch, color="black", linestyle="--", alpha=0.5,
                    label=f"Best epoch ({best_epoch})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total reconstruction loss")
    axes[0].set_title("Autoencoder training: total reconstruction loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_mse"], label="Train MSE (numeric)", color="#4C72B0")
    axes[1].plot(epochs, history["val_mse"], label="Val MSE", color="#C44E52")
    axes[1].plot(epochs, history["train_ce"], label="Train CE (categorical)",
                 color="#4C72B0", linestyle="--")
    axes[1].plot(epochs, history["val_ce"], label="Val CE",
                 color="#C44E52", linestyle="--")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Reconstruction loss components")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "loss_curve.png", dpi=120)
    plt.close(fig)


def _scatter_2d(
    coords: np.ndarray, labels: np.ndarray, label_names: list[str],
    title: str, outfile: str, cmap: str = "tab10",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    unique = np.unique(labels)
    palette = plt.get_cmap(cmap, max(len(unique), 3))
    for i, u in enumerate(unique):
        mask = labels == u
        name = label_names[int(u)] if int(u) < len(label_names) else str(u)
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=18, alpha=0.65, color=palette(i), label=name,
                   edgecolor="none")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / outfile, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_latent_projections(npz, metrics: dict) -> None:
    z = npz["z_train"]
    y = npz["y_train"]
    clusters = npz["kmeans_train_clusters"]
    class_names = metrics["class_names"]

    # PCA (fast, deterministic)
    pca = PCA(n_components=2, random_state=42)
    z_pca = pca.fit_transform(z)
    evr = pca.explained_variance_ratio_

    _scatter_2d(
        z_pca, y, class_names,
        title=(f"Autoencoder latent space (PCA)\n"
               f"colored by buyer_profile  |  explained variance: "
               f"{evr[0]:.1%} + {evr[1]:.1%} = {evr.sum():.1%}"),
        outfile="latent_pca_by_profile.png",
    )

    # PCA colored by KMeans cluster
    cluster_names = [f"Cluster {i}" for i in range(int(clusters.max()) + 1)]
    _scatter_2d(
        z_pca, clusters, cluster_names,
        title=("Autoencoder latent space (PCA)\n"
               "colored by KMeans cluster (k=8)"),
        outfile="latent_pca_by_cluster.png",
        cmap="tab10",
    )

    # t-SNE (slower, non-linear). Use a modest perplexity given small n.
    tsne = TSNE(n_components=2, perplexity=30, init="pca",
                learning_rate="auto", random_state=42)
    z_tsne = tsne.fit_transform(z)
    _scatter_2d(
        z_tsne, y, class_names,
        title="Autoencoder latent space (t-SNE)\ncolored by buyer_profile",
        outfile="latent_tsne_by_profile.png",
    )


def plot_cluster_profile_heatmap(npz, metrics: dict) -> None:
    y = npz["y_train"]
    clusters = npz["kmeans_train_clusters"]
    class_names = metrics["class_names"]

    k = int(clusters.max()) + 1
    n_classes = len(class_names)
    mat = np.zeros((k, n_classes), dtype=float)
    for c, profile in zip(clusters, y):
        mat[int(c), int(profile)] += 1
    # Normalize each cluster row to proportions
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    mat_norm = mat / row_sums

    sizes = mat.sum(axis=1).astype(int)
    cluster_labels = [f"Cluster {i}\n(n={sizes[i]})" for i in range(k)]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        mat_norm, annot=True, fmt=".2f", cmap="YlGnBu",
        xticklabels=class_names, yticklabels=cluster_labels,
        cbar_kws={"label": "Proportion within cluster"}, ax=ax,
    )
    ari = metrics["clustering_on_latent"]["ari_vs_buyer_profile"]
    nmi = metrics["clustering_on_latent"]["nmi_vs_buyer_profile"]
    ax.set_title(
        "Buyer-profile distribution per latent-space KMeans cluster\n"
        f"ARI = {ari:.3f}  |  NMI = {nmi:.3f}"
    )
    ax.set_xlabel("buyer_profile")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cluster_profile_heatmap.png", dpi=120)
    plt.close(fig)


def main() -> None:
    metrics, npz = load_artifacts()
    plot_loss_curve(metrics)
    plot_latent_projections(npz, metrics)
    plot_cluster_profile_heatmap(npz, metrics)
    print(f"Done: wrote autoencoder plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
