from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype, is_object_dtype, is_string_dtype
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class TrainingConfig:
    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root / "data" / "processed" / "property_buyer"
    output_root: Path = project_root / "data" / "models" / "property_buyer"

    dataset_filename: str = "cleaned_dataset_no_missing_v1.0.xlsx"
    target_column: str = "buyer_profile"
    feature_mode: str = "property_only"

    random_state: int = 42
    fill_missing_categorical_with: str = "__missing__"

    min_clusters: int = 3
    max_clusters: int = 10
    n_init: int = 20
    max_iter: int = 500

    @property
    def output_dir(self) -> Path:
        return self.output_root / f"kmeans_{self.feature_mode}_v1"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


logger = logging.getLogger(__name__)

PROPERTY_CORE_FEATURES = [
    "rajoni_akp",
    "kategoria_e_asetit",
    "menyra_e_shitjes",
    "komuna_lokacioni_i_asetit_te_shitur",
    "qytet_apo_fshat_lokacioni_i_asetit_te_shitur",
    "zona_kadastrale",
    "eshte_shitur_si_ndermarrje_e_re_apo_aset_ne_likuidim",
]

PROPERTY_AREA_FEATURES = [
    "siperfaqja_e_objektit_nese_ka_objekt_te_toka_dhe_dihet_siperfaqja_m2",
    "siperfaqja_e_tokes_ne_metra_katror",
    "siperfaqja_e_objektit_nese_ka_objekt_te_toka_dhe_dihet_siperfaqja_m2_was_missing",
    "siperfaqja_e_tokes_ne_metra_katror_was_missing",
    "raw_has_object_area",
    "raw_has_land_area",
    "total_area_m2",
    "object_to_land_ratio",
    "is_land_only",
    "has_object",
    "is_large_land",
    "area_data_completeness_score",
    "asset_structure_type",
]

PROPERTY_PRICE_FEATURES = [
    "cmimi_i_shitjes_se_asetit",
    "cmimi_i_shitjes_se_asetit_was_missing",
    "price_per_total_m2",
    "price_per_land_m2",
    "price_per_object_m2",
    "log_sale_price",
    "is_high_value_asset",
    "area_price_interaction",
]

TIME_FEATURES = [
    "contract_year",
    "contract_month",
    "contract_quarter",
]

QUALITY_FLAGS = [
    "siperfaqja_e_objektit_nese_ka_objekt_te_toka_dhe_dihet_siperfaqja_m2_is_outlier_iqr",
    "siperfaqja_e_tokes_ne_metra_katror_is_outlier_iqr",
    "total_area_m2_is_outlier_iqr",
    "price_per_land_m2_is_outlier_iqr",
    "data_e_kontrates_was_missing",
]

LIGHT_BUSINESS_FEATURES = [
    "arbk_numripunetoreve",
    "arbk_kapitali",
    "business_age_days_at_sale",
    "business_age_years_at_sale",
    "capital_to_sale_price_ratio",
    "arbk_kapitali_was_missing",
    "arbk_dataregjistrimit_was_missing",
]

FULL_BUSINESS_FEATURES = [
    "arbk_statusiarbk",
    "arbk_aktiviteti_1_pershkrimi",
    "arbk_aktiviteti_2_pershkrimi",
    "arbk_pronari_1_kapitali",
    "arbk_pronari_1_kapitali_is_outlier_iqr",
    "capital_to_sale_price_ratio_is_outlier_iqr",
    "arbk_aktiviteti_1_kodinace_is_outlier_iqr",
    "arbk_aktiviteti_2_kodinace_is_outlier_iqr",
    "arbk_aktiviteti_3_kodinace_is_outlier_iqr",
]


def ensure_output_dirs(cfg: TrainingConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)


def load_excel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    logger.info("Loading file: %s", path)
    return pd.read_excel(path)


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    except Exception:
        return None


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def get_feature_columns(df: pd.DataFrame, cfg: TrainingConfig) -> list[str]:
    base = (
        PROPERTY_CORE_FEATURES
        + PROPERTY_AREA_FEATURES
        + PROPERTY_PRICE_FEATURES
        + TIME_FEATURES
        + QUALITY_FLAGS
    )

    if cfg.feature_mode == "property_only":
        candidates = base
    elif cfg.feature_mode == "property_plus_light_business":
        candidates = base + LIGHT_BUSINESS_FEATURES
    elif cfg.feature_mode == "full":
        candidates = base + LIGHT_BUSINESS_FEATURES + FULL_BUSINESS_FEATURES
    else:
        raise ValueError(
            f"Unsupported feature_mode='{cfg.feature_mode}'. "
            f"Use property_only, property_plus_light_business, or full."
        )

    selected: list[str] = []
    seen: set[str] = set()
    for col in candidates:
        if col in df.columns and col not in seen:
            selected.append(col)
            seen.add(col)

    if not selected:
        raise ValueError("No usable feature columns were selected.")

    return selected


def detect_categorical_columns(df: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    categorical_columns: list[str] = []
    for col in feature_columns:
        dtype = df[col].dtype
        if is_object_dtype(df[col]) or isinstance(dtype, CategoricalDtype) or is_string_dtype(dtype):
            categorical_columns.append(col)
    return categorical_columns


def preprocess_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    fill_missing_categorical_with: str,
) -> pd.DataFrame:
    X = df[feature_columns].copy()

    for col in categorical_columns:
        X[col] = X[col].astype("string").fillna(fill_missing_categorical_with)

    numeric_columns = [col for col in feature_columns if col not in categorical_columns]
    for col in numeric_columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    return X


def build_preprocessor(
    categorical_columns: list[str],
    numeric_columns: list[str],
    cfg: TrainingConfig,
) -> ColumnTransformer:
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=cfg.fill_missing_categorical_with)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False)),
        ]
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_columns),
            ("num", numeric_transformer, numeric_columns),
        ],
        remainder="drop",
    )


def evaluate_kmeans(X_matrix: np.ndarray, labels: np.ndarray, kmeans: KMeans) -> dict[str, Any]:
    return {
        "n_clusters": int(kmeans.n_clusters),
        "inertia": safe_float(kmeans.inertia_),
        "silhouette_score": safe_float(silhouette_score(X_matrix, labels)),
        "calinski_harabasz_score": safe_float(calinski_harabasz_score(X_matrix, labels)),
        "davies_bouldin_score": safe_float(davies_bouldin_score(X_matrix, labels)),
    }


def select_best_cluster_count(
    X_matrix: np.ndarray,
    cfg: TrainingConfig,
) -> tuple[KMeans, list[dict[str, Any]]]:
    evaluations: list[dict[str, Any]] = []
    best_model: KMeans | None = None
    best_silhouette = -np.inf

    for n_clusters in range(cfg.min_clusters, cfg.max_clusters + 1):
        logger.info("Evaluating KMeans with n_clusters=%d", n_clusters)
        model = KMeans(
            n_clusters=n_clusters,
            n_init=cfg.n_init,
            max_iter=cfg.max_iter,
            random_state=cfg.random_state,
        )
        labels = model.fit_predict(X_matrix)
        metrics = evaluate_kmeans(X_matrix, labels, model)
        evaluations.append(metrics)

        silhouette = metrics["silhouette_score"]
        if silhouette is not None and silhouette > best_silhouette:
            best_model = model
            best_silhouette = silhouette

    if best_model is None:
        raise ValueError("Failed to select a KMeans model.")

    return best_model, evaluations


def build_cluster_profile_summary(
    df: pd.DataFrame,
    cluster_column: str,
    reference_target_column: str,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for cluster_id, cluster_df in df.groupby(cluster_column):
        cluster_key = f"cluster_{int(cluster_id)}"
        item: dict[str, Any] = {
            "size": int(len(cluster_df)),
        }
        if reference_target_column in cluster_df.columns:
            counts = cluster_df[reference_target_column].value_counts(dropna=False)
            item["top_reference_labels"] = counts.head(5).to_dict()
            item["dominant_reference_label"] = str(counts.index[0]) if not counts.empty else None
            item["dominant_reference_label_ratio"] = (
                safe_float(counts.iloc[0] / len(cluster_df)) if not counts.empty else None
            )
        summary[cluster_key] = item
    return summary


def plot_cluster_selection_metrics(
    evaluations: list[dict[str, Any]],
    output_path: Path,
) -> None:
    metrics_df = pd.DataFrame(evaluations).sort_values("n_clusters")
    x = metrics_df["n_clusters"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_specs = [
        ("silhouette_score", "Silhouette Score", "#1f77b4"),
        ("inertia", "Inertia", "#ff7f0e"),
        ("calinski_harabasz_score", "Calinski-Harabasz", "#2ca02c"),
        ("davies_bouldin_score", "Davies-Bouldin", "#d62728"),
    ]

    for ax, (column, title, color) in zip(axes.ravel(), plot_specs, strict=False):
        ax.plot(x, metrics_df[column], marker="o", color=color)
        ax.set_title(title)
        ax.set_xlabel("Number of Clusters")
        ax.set_xticks(x)
        ax.grid(alpha=0.25)

    fig.suptitle("KMeans Cluster Selection Metrics", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_size_distribution(labels: np.ndarray, output_path: Path) -> None:
    counts = pd.Series(labels).value_counts().sort_index()
    positions = np.arange(len(counts))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(positions, counts.values, color="#4c956c")
    ax.set_title("Cluster Size Distribution")
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Rows")
    ax.set_xticks(positions)
    ax.set_xticklabels(counts.index.astype(str))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_pca_scatter(
    X_matrix: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
) -> None:
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_matrix)

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        components[:, 0],
        components[:, 1],
        c=labels,
        cmap="tab10",
        s=22,
        alpha=0.75,
        edgecolors="none",
    )
    ax.set_title("KMeans Clusters in PCA Space")
    ax.set_xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)")
    legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
    ax.add_artist(legend)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_profile_heatmap(
    assignments_df: pd.DataFrame,
    cluster_column: str,
    reference_target_column: str,
    output_path: Path,
) -> None:
    if reference_target_column not in assignments_df.columns:
        return

    heatmap_df = pd.crosstab(
        assignments_df[cluster_column],
        assignments_df[reference_target_column],
        normalize="index",
    ).sort_index()

    fig_width = max(9, len(heatmap_df.columns) * 0.9)
    fig_height = max(4.8, len(heatmap_df.index) * 0.9)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(heatmap_df.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)

    ax.set_title("Cluster Composition by Buyer Profile")
    ax.set_xlabel("Buyer Profile")
    ax.set_ylabel("Cluster ID")
    ax.set_xticks(np.arange(len(heatmap_df.columns)))
    ax.set_xticklabels(heatmap_df.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(heatmap_df.index)))
    ax.set_yticklabels(heatmap_df.index.astype(str))

    for i in range(heatmap_df.shape[0]):
        for j in range(heatmap_df.shape[1]):
            value = heatmap_df.iat[i, j]
            if value >= 0.08:
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.45 else "black",
                    fontsize=8,
                )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Share within cluster")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-mode",
        type=str,
        default="property_only",
        choices=["property_only", "property_plus_light_business", "full"],
    )
    parser.add_argument("--min-clusters", type=int, default=3)
    parser.add_argument("--max-clusters", type=int, default=10)
    args = parser.parse_args()

    cfg = TrainingConfig(
        feature_mode=args.feature_mode,
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters,
    )
    ensure_output_dirs(cfg)

    df = load_excel(cfg.data_dir / cfg.dataset_filename)
    logger.info("Loaded cleaned dataset shape: %s", df.shape)

    feature_columns = get_feature_columns(df, cfg)
    categorical_columns = detect_categorical_columns(df, feature_columns)
    X = preprocess_features(
        df,
        feature_columns,
        categorical_columns,
        cfg.fill_missing_categorical_with,
    )
    final_categorical_columns = [
        col
        for col in X.columns
        if is_object_dtype(X[col]) or isinstance(X[col].dtype, CategoricalDtype) or is_string_dtype(X[col].dtype)
    ]
    final_numeric_columns = [col for col in X.columns if col not in final_categorical_columns]

    logger.info("Feature mode: %s", cfg.feature_mode)
    logger.info("Selected %d base features.", len(feature_columns))
    logger.info("Final categorical columns: %d", len(final_categorical_columns))
    logger.info("Final numeric columns: %d", len(final_numeric_columns))

    preprocessor = build_preprocessor(final_categorical_columns, final_numeric_columns, cfg)
    X_matrix = preprocessor.fit_transform(X)
    X_matrix = np.asarray(X_matrix)

    best_model, evaluations = select_best_cluster_count(X_matrix, cfg)
    cluster_labels = best_model.labels_

    model_path = cfg.output_dir / "property_buyer_kmeans_pipeline.joblib"
    bundle_path = cfg.output_dir / "property_buyer_kmeans_bundle.joblib"
    metrics_path = cfg.output_dir / "metrics.json"
    metadata_path = cfg.output_dir / "training_metadata.json"
    assignments_path = cfg.output_dir / "cluster_assignments.csv"
    cluster_summary_path = cfg.output_dir / "cluster_summary.json"

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", best_model),
        ]
    )
    joblib.dump(pipeline, model_path)

    assignments_df = df.copy()
    assignments_df["cluster_id"] = cluster_labels
    assignments_df.to_csv(assignments_path, index=False, encoding="utf-8")

    selected_metrics = next(
        metric for metric in evaluations if metric["n_clusters"] == int(best_model.n_clusters)
    )
    cluster_summary = build_cluster_profile_summary(
        assignments_df,
        cluster_column="cluster_id",
        reference_target_column=cfg.target_column,
    )

    plot_cluster_selection_metrics(
        evaluations,
        cfg.output_dir / "cluster_selection_metrics.png",
    )
    plot_cluster_size_distribution(
        cluster_labels,
        cfg.output_dir / "cluster_sizes.png",
    )
    plot_cluster_pca_scatter(
        X_matrix,
        cluster_labels,
        cfg.output_dir / "cluster_pca_scatter.png",
    )
    plot_cluster_profile_heatmap(
        assignments_df,
        cluster_column="cluster_id",
        reference_target_column=cfg.target_column,
        output_path=cfg.output_dir / "cluster_profile_heatmap.png",
    )

    save_json(
        metrics_path,
        {
            "selection_metric": "silhouette_score",
            "selected_model_metrics": selected_metrics,
            "all_cluster_evaluations": evaluations,
        },
    )

    save_json(
        metadata_path,
        {
            "config": {
                **asdict(cfg),
                "project_root": str(cfg.project_root),
                "data_dir": str(cfg.data_dir),
                "output_root": str(cfg.output_root),
                "output_dir": str(cfg.output_dir),
            },
            "dataset_shape": [int(df.shape[0]), int(df.shape[1])],
            "selected_feature_columns": feature_columns,
            "categorical_columns": final_categorical_columns,
            "numeric_columns": final_numeric_columns,
            "transformed_feature_count": int(X_matrix.shape[1]),
        },
    )

    save_json(cluster_summary_path, cluster_summary)

    joblib.dump(
        {
            "model_path": str(model_path),
            "feature_mode": cfg.feature_mode,
            "feature_columns": feature_columns,
            "categorical_columns": final_categorical_columns,
            "numeric_columns": final_numeric_columns,
            "selected_n_clusters": int(best_model.n_clusters),
        },
        bundle_path,
    )

    logger.info("Artifacts saved to: %s", cfg.output_dir)
    logger.info(
        "Selected n_clusters=%d with silhouette_score=%.4f",
        int(best_model.n_clusters),
        selected_metrics["silhouette_score"],
    )


if __name__ == "__main__":
    main()
