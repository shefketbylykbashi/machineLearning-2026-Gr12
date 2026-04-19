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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class TrainingConfig:
    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root / "data" / "processed" / "property_buyer"
    output_root: Path = project_root / "data" / "models" / "property_buyer"

    train_filename: str = "train_dataset_v1.0.xlsx"
    test_filename: str = "test_dataset_v1.0.xlsx"

    target_column: str = "buyer_profile"
    feature_mode: str = "property_only"

    random_state: int = 42
    validation_size: float = 0.15
    n_splits_cv: int = 5
    min_class_count: int = 3

    fill_missing_categorical_with: str = "__missing__"
    rare_category_threshold: int = 15
    add_numeric_bins: bool = False

    penalty: str = "l2"
    C: float = 1.0
    max_iter: int = 4000
    solver: str = "lbfgs"

    @property
    def output_dir(self) -> Path:
        return self.output_root / f"logistic_regression_{self.feature_mode}_v1"


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


def filter_rare_target_classes(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    min_class_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    class_counts = train_df[target_column].value_counts(dropna=False)
    valid_classes = class_counts[class_counts >= min_class_count].index

    filtered_train = train_df[train_df[target_column].isin(valid_classes)].copy()
    filtered_test = test_df[test_df[target_column].isin(valid_classes)].copy()

    logger.info(
        "Rare-class filtering applied with min_class_count=%d. Train: %d -> %d | Test: %d -> %d",
        min_class_count,
        len(train_df),
        len(filtered_train),
        len(test_df),
        len(filtered_test),
    )
    return filtered_train, filtered_test


def collapse_rare_categories(
    train_df: pd.DataFrame,
    other_dfs: list[pd.DataFrame],
    categorical_columns: list[str],
    threshold: int,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    train_df = train_df.copy()
    transformed_others = [df.copy() for df in other_dfs]

    for col in categorical_columns:
        counts = train_df[col].astype("string").fillna("__missing__").value_counts()
        keep_values = set(counts[counts >= threshold].index)

        train_df[col] = train_df[col].astype("string").fillna("__missing__")
        train_df[col] = train_df[col].where(train_df[col].isin(keep_values), "__rare__")

        for other_df in transformed_others:
            other_df[col] = other_df[col].astype("string").fillna("__missing__")
            other_df[col] = other_df[col].where(other_df[col].isin(keep_values), "__rare__")

    return train_df, transformed_others


def add_binned_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    bin_candidates = [
        "cmimi_i_shitjes_se_asetit",
        "log_sale_price",
        "total_area_m2",
        "price_per_total_m2",
        "price_per_land_m2",
        "price_per_object_m2",
        "siperfaqja_e_tokes_ne_metra_katror",
        "siperfaqja_e_objektit_nese_ka_objekt_te_toka_dhe_dihet_siperfaqja_m2",
    ]

    for col in bin_candidates:
        if col not in df.columns:
            continue

        series = pd.to_numeric(df[col], errors="coerce")
        if series.dropna().nunique() < 8:
            continue

        try:
            binned = pd.qcut(series, q=6, duplicates="drop")
            df[f"{col}_bin"] = binned.astype("string").fillna("__missing__")
        except Exception:
            continue

    return df


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, Any]:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": safe_float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": safe_float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": safe_float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": safe_float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def summarize_cv_results(cv_results: list[dict[str, Any]]) -> dict[str, Any]:
    def metric_values(metric_name: str) -> list[float]:
        return [float(result[metric_name]) for result in cv_results if result[metric_name] is not None]

    summary: dict[str, Any] = {}
    for metric_name in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]:
        values = metric_values(metric_name)
        summary[metric_name] = {
            "mean": safe_float(np.mean(values)) if values else None,
            "std": safe_float(np.std(values)) if values else None,
            "min": safe_float(np.min(values)) if values else None,
            "max": safe_float(np.max(values)) if values else None,
        }
    return summary


def build_pipeline(
    categorical_columns: list[str],
    numeric_columns: list[str],
    cfg: TrainingConfig,
) -> Pipeline:
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=cfg.fill_missing_categorical_with)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
        ]
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_columns),
            ("num", numeric_transformer, numeric_columns),
        ],
        remainder="drop",
    )

    model_kwargs: dict[str, Any] = {
        "C": cfg.C,
        "class_weight": "balanced",
        "max_iter": cfg.max_iter,
        "random_state": cfg.random_state,
        "solver": cfg.solver,
    }
    if cfg.penalty != "l2":
        model_kwargs["penalty"] = cfg.penalty

    model = LogisticRegression(**model_kwargs)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def extract_feature_importance(pipeline: Pipeline, output_path: Path) -> pd.DataFrame:
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    model: LogisticRegression = pipeline.named_steps["model"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = np.array([f"feature_{idx}" for idx in range(model.coef_.shape[1])])

    coef_abs = np.abs(model.coef_)
    feature_importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_coefficient": coef_abs.mean(axis=0),
            "max_abs_coefficient": coef_abs.max(axis=0),
        }
    ).sort_values("mean_abs_coefficient", ascending=False)

    feature_importance_df.to_csv(output_path, index=False, encoding="utf-8")
    return feature_importance_df


def plot_confusion_matrix_figure(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: list[str],
    output_path: Path,
    title: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig_width = max(8, len(labels) * 1.15)
    fig_height = max(6, len(labels) * 0.95)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)

    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = int(cm[i, j])
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=9,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_class_metrics_figure(
    classification_report_dict: dict[str, Any],
    labels: list[str],
    output_path: Path,
    title: str,
) -> None:
    metrics_df = pd.DataFrame(
        {
            "label": labels,
            "precision": [classification_report_dict.get(label, {}).get("precision", 0.0) for label in labels],
            "recall": [classification_report_dict.get(label, {}).get("recall", 0.0) for label in labels],
            "f1-score": [classification_report_dict.get(label, {}).get("f1-score", 0.0) for label in labels],
        }
    )

    x = np.arange(len(labels))
    width = 0.24
    fig_width = max(9, len(labels) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, 6.5))
    ax.bar(x - width, metrics_df["precision"], width=width, label="Precision", color="#1f77b4")
    ax.bar(x, metrics_df["recall"], width=width, label="Recall", color="#ff7f0e")
    ax.bar(x + width, metrics_df["f1-score"], width=width, label="F1", color="#2ca02c")

    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance_figure(
    feature_importance_df: pd.DataFrame,
    output_path: Path,
    title: str,
    top_n: int = 20,
) -> None:
    plot_df = feature_importance_df.head(top_n).iloc[::-1].copy()

    fig_height = max(6, top_n * 0.36)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    ax.barh(plot_df["feature"], plot_df["mean_abs_coefficient"], color="#2a6f97")
    ax.set_title(title)
    ax.set_xlabel("Mean Absolute Coefficient")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_columns: list[str],
    cfg: TrainingConfig,
) -> list[dict[str, Any]]:
    skf = StratifiedKFold(
        n_splits=cfg.n_splits_cv,
        shuffle=True,
        random_state=cfg.random_state,
    )

    cv_results: list[dict[str, Any]] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
        logger.info("Starting CV fold %d/%d", fold_idx, cfg.n_splits_cv)

        X_train_fold = X.iloc[train_idx].copy()
        y_train_fold = y.iloc[train_idx].copy()
        X_valid_fold = X.iloc[valid_idx].copy()
        y_valid_fold = y.iloc[valid_idx].copy()

        fold_categorical_columns = [
            col
            for col in X_train_fold.columns
            if is_object_dtype(X_train_fold[col])
            or isinstance(X_train_fold[col].dtype, CategoricalDtype)
            or is_string_dtype(X_train_fold[col].dtype)
        ]
        fold_numeric_columns = [col for col in X_train_fold.columns if col not in fold_categorical_columns]

        pipeline = build_pipeline(fold_categorical_columns, fold_numeric_columns, cfg)
        pipeline.fit(X_train_fold, y_train_fold)

        y_valid_pred = pipeline.predict(X_valid_fold)
        fold_metrics = evaluate_predictions(y_valid_fold, y_valid_pred)

        cv_results.append(
            {
                "fold": fold_idx,
                "accuracy": fold_metrics["accuracy"],
                "balanced_accuracy": fold_metrics["balanced_accuracy"],
                "macro_f1": fold_metrics["macro_f1"],
                "weighted_f1": fold_metrics["weighted_f1"],
                "n_iter_max": int(np.max(pipeline.named_steps["model"].n_iter_)),
            }
        )

    return cv_results


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-mode",
        type=str,
        default="property_only",
        choices=["property_only", "property_plus_light_business", "full"],
    )
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=8000)
    args = parser.parse_args()

    cfg = TrainingConfig(
        feature_mode=args.feature_mode,
        C=args.C,
        max_iter=args.max_iter,
    )
    ensure_output_dirs(cfg)

    train_df = load_excel(cfg.data_dir / cfg.train_filename)
    test_df = load_excel(cfg.data_dir / cfg.test_filename)

    logger.info("Initial train shape: %s", train_df.shape)
    logger.info("Initial test shape: %s", test_df.shape)

    if cfg.target_column not in train_df.columns:
        raise ValueError(f"Target column '{cfg.target_column}' not found in train dataset.")
    if cfg.target_column not in test_df.columns:
        raise ValueError(f"Target column '{cfg.target_column}' not found in test dataset.")

    train_df = train_df.dropna(subset=[cfg.target_column]).copy()
    test_df = test_df.dropna(subset=[cfg.target_column]).copy()

    train_df, test_df = filter_rare_target_classes(
        train_df=train_df,
        test_df=test_df,
        target_column=cfg.target_column,
        min_class_count=cfg.min_class_count,
    )

    feature_columns = get_feature_columns(train_df, cfg)
    categorical_columns = detect_categorical_columns(train_df, feature_columns)

    logger.info("Feature mode: %s", cfg.feature_mode)
    logger.info("Selected %d features.", len(feature_columns))
    logger.info("Detected %d categorical features.", len(categorical_columns))

    X_train_all = preprocess_features(
        train_df,
        feature_columns,
        categorical_columns,
        cfg.fill_missing_categorical_with,
    )
    y_train_all = train_df[cfg.target_column].astype("string")

    X_test = preprocess_features(
        test_df,
        feature_columns,
        categorical_columns,
        cfg.fill_missing_categorical_with,
    )
    y_test = test_df[cfg.target_column].astype("string")

    logger.info("Running %d-fold cross-validation...", cfg.n_splits_cv)
    cv_results = run_cross_validation(
        X=X_train_all,
        y=y_train_all,
        categorical_columns=categorical_columns,
        cfg=cfg,
    )
    cv_summary = summarize_cv_results(cv_results)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_all,
        y_train_all,
        test_size=cfg.validation_size,
        random_state=cfg.random_state,
        stratify=y_train_all,
    )

    final_categorical_columns = [
        col
        for col in X_train.columns
        if is_object_dtype(X_train[col])
        or isinstance(X_train[col].dtype, CategoricalDtype)
        or is_string_dtype(X_train[col].dtype)
    ]
    final_numeric_columns = [col for col in X_train.columns if col not in final_categorical_columns]

    logger.info("Training multinomial logistic regression...")
    pipeline = build_pipeline(final_categorical_columns, final_numeric_columns, cfg)
    pipeline.fit(X_train, y_train)

    y_valid_pred = pipeline.predict(X_valid)
    y_test_pred = pipeline.predict(X_test)
    y_test_proba = pipeline.predict_proba(X_test)

    valid_metrics = evaluate_predictions(y_valid, y_valid_pred)
    test_metrics = evaluate_predictions(y_test, y_test_pred)
    feature_importance_df = extract_feature_importance(
        pipeline,
        cfg.output_dir / "feature_importance.csv",
    )
    class_labels = sorted(y_train_all.unique().tolist())

    plot_confusion_matrix_figure(
        y_test,
        y_test_pred,
        class_labels,
        cfg.output_dir / "test_confusion_matrix.png",
        "Logistic Regression Test Confusion Matrix",
    )
    plot_class_metrics_figure(
        test_metrics["classification_report"],
        class_labels,
        cfg.output_dir / "test_class_metrics.png",
        "Logistic Regression Test Class Metrics",
    )
    plot_feature_importance_figure(
        feature_importance_df,
        cfg.output_dir / "feature_importance_top20.png",
        "Top Logistic Regression Features",
    )

    model_path = cfg.output_dir / "buyer_profile_logistic_regression_pipeline.joblib"
    bundle_path = cfg.output_dir / "buyer_profile_logistic_regression_bundle.joblib"
    metrics_path = cfg.output_dir / "metrics.json"
    cv_results_path = cfg.output_dir / "cv_results.json"
    metadata_path = cfg.output_dir / "training_metadata.json"
    valid_predictions_path = cfg.output_dir / "validation_predictions.csv"
    test_predictions_path = cfg.output_dir / "test_predictions.csv"

    joblib.dump(pipeline, model_path)

    valid_predictions_df = X_valid.copy()
    valid_predictions_df[cfg.target_column] = y_valid.values
    valid_predictions_df["predicted_buyer_profile"] = y_valid_pred
    valid_predictions_df["is_correct"] = (
        valid_predictions_df[cfg.target_column] == valid_predictions_df["predicted_buyer_profile"]
    )
    valid_predictions_df.to_csv(valid_predictions_path, index=False, encoding="utf-8")

    test_predictions_df = X_test.copy()
    test_predictions_df[cfg.target_column] = y_test.values
    test_predictions_df["predicted_buyer_profile"] = y_test_pred
    test_predictions_df["prediction_confidence"] = y_test_proba.max(axis=1)
    test_predictions_df["is_correct"] = (
        test_predictions_df[cfg.target_column] == test_predictions_df["predicted_buyer_profile"]
    )
    test_predictions_df.to_csv(test_predictions_path, index=False, encoding="utf-8")

    save_json(
        metrics_path,
        {
            "validation": valid_metrics,
            "test": test_metrics,
            "cv_summary": cv_summary,
            "top_25_feature_importance": feature_importance_df.head(25).to_dict(orient="records"),
        },
    )

    save_json(
        cv_results_path,
        {
            "feature_mode": cfg.feature_mode,
            "fold_results": cv_results,
            "summary": cv_summary,
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
            "selected_feature_columns": feature_columns,
            "categorical_columns": final_categorical_columns,
            "numeric_columns": final_numeric_columns,
            "n_train_rows": int(len(train_df)),
            "n_test_rows": int(len(test_df)),
            "n_classes_train": int(y_train_all.nunique()),
            "classes_train": sorted(y_train_all.unique().tolist()),
            "n_iter_final_model": int(np.max(pipeline.named_steps["model"].n_iter_)),
        },
    )

    joblib.dump(
        {
            "model_path": str(model_path),
            "feature_mode": cfg.feature_mode,
            "feature_columns": feature_columns,
            "categorical_columns": final_categorical_columns,
            "numeric_columns": final_numeric_columns,
            "target_column": cfg.target_column,
            "add_numeric_bins": cfg.add_numeric_bins,
            "rare_category_threshold": cfg.rare_category_threshold,
        },
        bundle_path,
    )

    logger.info("Artifacts saved to: %s", cfg.output_dir)
    logger.info("Validation macro-F1: %.4f", valid_metrics["macro_f1"])
    logger.info("Test macro-F1: %.4f", test_metrics["macro_f1"])


if __name__ == "__main__":
    main()
