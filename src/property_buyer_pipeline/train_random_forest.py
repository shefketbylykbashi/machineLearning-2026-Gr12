from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype, is_object_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# Configuration
# ============================================================

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

    # Property-only optimization switches
    drop_zona_kadastrale: bool = False
    add_numeric_bins: bool = True

    # Tuned RF params for property_only
    n_estimators: int = 1200
    max_depth: int | None = 24
    min_samples_split: int = 6
    min_samples_leaf: int = 1
    max_features: float = 0.35
    bootstrap: bool = True
    n_jobs: int = -1

    @property
    def output_dir(self) -> Path:
        suffix = "no_zona" if self.drop_zona_kadastrale else "with_zona"
        return self.output_root / f"random_forest_{self.feature_mode}_optimized_{suffix}_v1"


# ============================================================
# Logging
# ============================================================

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


logger = logging.getLogger(__name__)


# ============================================================
# Feature groups
# ============================================================

TARGET_LEAKAGE_COLUMNS = {
    "buyer_profile",
    "bleresi",
    "arbk_pronari_1",
    "emri_i_ndermarrjes_se_re_apo_asetit_ne_likuidim",
}

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


# ============================================================
# Utilities
# ============================================================

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
    if cfg.feature_mode != "property_only":
        raise ValueError("This script is optimized for feature_mode='property_only'.")

    candidates = (
        PROPERTY_CORE_FEATURES
        + PROPERTY_AREA_FEATURES
        + PROPERTY_PRICE_FEATURES
        + TIME_FEATURES
        + QUALITY_FLAGS
    )

    selected: list[str] = []
    seen: set[str] = set()

    for col in candidates:
        if cfg.drop_zona_kadastrale and col == "zona_kadastrale":
            continue
        if col in df.columns and col not in TARGET_LEAKAGE_COLUMNS and col not in seen:
            selected.append(col)
            seen.add(col)

    if not selected:
        raise ValueError("No usable feature columns were selected.")

    return selected


def detect_categorical_columns(df: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    categorical_cols: list[str] = []
    for col in feature_columns:
        dtype = df[col].dtype
        if is_object_dtype(df[col]) or isinstance(dtype, CategoricalDtype):
            categorical_cols.append(col)
    return categorical_cols


def preprocess_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    fill_missing_categorical_with: str,
) -> pd.DataFrame:
    X = df[feature_columns].copy()

    for col in categorical_columns:
        X[col] = X[col].astype("string").fillna(fill_missing_categorical_with)

    numeric_columns = [c for c in feature_columns if c not in categorical_columns]
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

        for df in transformed_others:
            df[col] = df[col].astype("string").fillna("__missing__")
            df[col] = df[col].where(df[col].isin(keep_values), "__rare__")

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
        non_null = series.dropna()

        if non_null.nunique() < 8:
            continue

        try:
            binned = pd.qcut(series, q=6, duplicates="drop")
            df[f"{col}_bin"] = binned.astype("string").fillna("__missing__")
        except Exception:
            pass

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
    def metric_values(name: str) -> list[float]:
        return [float(r[name]) for r in cv_results if r[name] is not None]

    summary: dict[str, Any] = {}
    for metric in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]:
        values = metric_values(metric)
        summary[metric] = {
            "mean": safe_float(np.mean(values)) if values else None,
            "std": safe_float(np.std(values)) if values else None,
            "min": safe_float(np.min(values)) if values else None,
            "max": safe_float(np.max(values)) if values else None,
        }

    return summary


def build_pipeline(categorical_columns: list[str], numeric_columns: list[str], cfg: TrainingConfig) -> Pipeline:
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=cfg.fill_missing_categorical_with)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
        ]
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_columns),
            ("num", numeric_transformer, numeric_columns),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_split=cfg.min_samples_split,
        min_samples_leaf=cfg.min_samples_leaf,
        max_features=cfg.max_features,
        bootstrap=cfg.bootstrap,
        class_weight="balanced_subsample",
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def extract_feature_importance(pipeline: Pipeline, output_path: Path) -> pd.DataFrame:
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    model: RandomForestClassifier = pipeline.named_steps["model"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = np.array([f"feature_{i}" for i in range(len(model.feature_importances_))])

    df_imp = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    df_imp.to_csv(output_path, index=False, encoding="utf-8")
    return df_imp


def run_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_columns: list[str],
    numeric_columns: list[str],
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

        X_train_fold, [X_valid_fold] = collapse_rare_categories(
            train_df=X_train_fold,
            other_dfs=[X_valid_fold],
            categorical_columns=categorical_columns,
            threshold=cfg.rare_category_threshold,
        )

        if cfg.add_numeric_bins:
            X_train_fold = add_binned_numeric_features(X_train_fold)
            X_valid_fold = add_binned_numeric_features(X_valid_fold)

        fold_categorical_columns = [
            c for c in X_train_fold.columns
            if is_object_dtype(X_train_fold[c]) or isinstance(X_train_fold[c].dtype, CategoricalDtype) or str(X_train_fold[c].dtype) == "string"
        ]
        fold_numeric_columns = [c for c in X_train_fold.columns if c not in fold_categorical_columns]

        pipeline = build_pipeline(fold_categorical_columns, fold_numeric_columns, cfg)
        pipeline.fit(X_train_fold, y_train_fold)

        y_valid_pred = pipeline.predict(X_valid_fold)
        fold_metrics = evaluate_predictions(y_valid_fold, y_valid_pred)

        fold_result = {
            "fold": fold_idx,
            "accuracy": fold_metrics["accuracy"],
            "balanced_accuracy": fold_metrics["balanced_accuracy"],
            "macro_f1": fold_metrics["macro_f1"],
            "weighted_f1": fold_metrics["weighted_f1"],
        }
        cv_results.append(fold_result)

        logger.info(
            "Fold %d | macro_f1=%.4f | balanced_accuracy=%.4f",
            fold_idx,
            fold_result["macro_f1"],
            fold_result["balanced_accuracy"],
        )

    return cv_results


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--drop-zona-kadastrale", action="store_true")
    args = parser.parse_args()

    cfg = TrainingConfig(drop_zona_kadastrale=args.drop_zona_kadastrale)
    ensure_output_dirs(cfg)

    train_path = cfg.data_dir / cfg.train_filename
    test_path = cfg.data_dir / cfg.test_filename

    train_df = load_excel(train_path)
    test_df = load_excel(test_path)

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
        numeric_columns=[c for c in X_train_all.columns if c not in categorical_columns],
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

    X_train, [X_valid, X_test] = collapse_rare_categories(
        train_df=X_train,
        other_dfs=[X_valid, X_test],
        categorical_columns=categorical_columns,
        threshold=cfg.rare_category_threshold,
    )

    if cfg.add_numeric_bins:
        X_train = add_binned_numeric_features(X_train)
        X_valid = add_binned_numeric_features(X_valid)
        X_test = add_binned_numeric_features(X_test)

    final_categorical_columns = [
        c for c in X_train.columns
        if is_object_dtype(X_train[c]) or isinstance(X_train[c].dtype, CategoricalDtype) or str(X_train[c].dtype) == "string"
    ]
    final_numeric_columns = [c for c in X_train.columns if c not in final_categorical_columns]

    logger.info("Training optimized Random Forest model...")
    pipeline = build_pipeline(final_categorical_columns, final_numeric_columns, cfg)
    pipeline.fit(X_train, y_train)

    y_valid_pred = pipeline.predict(X_valid)
    y_test_pred = pipeline.predict(X_test)
    y_test_proba = pipeline.predict_proba(X_test)
    prediction_confidence = y_test_proba.max(axis=1)

    valid_metrics = evaluate_predictions(y_valid, y_valid_pred)
    test_metrics = evaluate_predictions(y_test, y_test_pred)

    model_path = cfg.output_dir / "buyer_profile_random_forest_pipeline.joblib"
    bundle_path = cfg.output_dir / "buyer_profile_random_forest_bundle.joblib"
    metrics_path = cfg.output_dir / "metrics.json"
    cv_results_path = cfg.output_dir / "cv_results.json"
    metadata_path = cfg.output_dir / "training_metadata.json"
    feature_importance_path = cfg.output_dir / "feature_importance.csv"
    valid_predictions_path = cfg.output_dir / "validation_predictions.csv"
    test_predictions_path = cfg.output_dir / "test_predictions.csv"

    joblib.dump(pipeline, model_path)
    feature_importance_df = extract_feature_importance(pipeline, feature_importance_path)

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
    test_predictions_df["prediction_confidence"] = prediction_confidence
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
            "final_categorical_columns": final_categorical_columns,
            "final_numeric_columns": final_numeric_columns,
            "n_train_rows": int(len(train_df)),
            "n_test_rows": int(len(test_df)),
            "n_classes_train": int(y_train_all.nunique()),
            "classes_train": sorted(y_train_all.unique().tolist()),
        },
    )

    joblib.dump(
        {
            "model_path": str(model_path),
            "feature_mode": cfg.feature_mode,
            "feature_columns": feature_columns,
            "final_categorical_columns": final_categorical_columns,
            "final_numeric_columns": final_numeric_columns,
            "target_column": cfg.target_column,
            "drop_zona_kadastrale": cfg.drop_zona_kadastrale,
            "add_numeric_bins": cfg.add_numeric_bins,
            "rare_category_threshold": cfg.rare_category_threshold,
        },
        bundle_path,
    )

    logger.info("Artifacts saved to: %s", cfg.output_dir)
    logger.info("Training completed successfully.")


if __name__ == "__main__":
    main()