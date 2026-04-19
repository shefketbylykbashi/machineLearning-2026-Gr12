from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from pandas.api.types import CategoricalDtype, is_object_dtype
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
import argparse

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

    # Modes:
    # - property_only
    # - property_plus_light_business
    # - full
    feature_mode: str = "property_only"

    random_state: int = 42
    validation_size: float = 0.15
    n_splits_cv: int = 5
    min_class_count: int = 3

    fill_missing_categorical_with: str = "__missing__"

    # CatBoost params
    iterations: int = 2500
    learning_rate: float = 0.03
    depth: int = 7
    l2_leaf_reg: float = 9.0
    random_strength: float = 1.5
    bagging_temperature: float = 1.0
    border_count: int = 254
    early_stopping_rounds: int = 150
    loss_function: str = "MultiClass"
    eval_metric: str = "TotalF1:average=Macro"

    verbose_eval: int = 100

    @property
    def output_dir(self) -> Path:
        return self.output_root / f"catboost_{self.feature_mode}_v4"


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

    selected = []
    seen = set()

    for col in candidates:
        if col in df.columns and col not in TARGET_LEAKAGE_COLUMNS and col not in seen:
            selected.append(col)
            seen.add(col)

    if not selected:
        raise ValueError("No usable feature columns were selected.")

    return selected


def detect_categorical_columns(df: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    categorical_cols = []
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
        "Rare-class filtering applied with min_class_count=%d. "
        "Train: %d -> %d | Test: %d -> %d",
        min_class_count,
        len(train_df),
        len(filtered_train),
        len(test_df),
        len(filtered_test),
    )
    return filtered_train, filtered_test


def build_class_weights(y: pd.Series) -> dict[str, float]:
    counts = y.value_counts()
    total = len(y)
    n_classes = len(counts)

    weights: dict[str, float] = {}
    for cls, count in counts.items():
        weights[str(cls)] = total / (n_classes * count)

    return weights


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


def create_model(cfg: TrainingConfig, class_weights: dict[str, float]) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function=cfg.loss_function,
        eval_metric=cfg.eval_metric,
        iterations=cfg.iterations,
        learning_rate=cfg.learning_rate,
        depth=cfg.depth,
        l2_leaf_reg=cfg.l2_leaf_reg,
        random_strength=cfg.random_strength,
        bagging_temperature=cfg.bagging_temperature,
        border_count=cfg.border_count,
        random_seed=cfg.random_state,
        class_weights=class_weights,
        early_stopping_rounds=cfg.early_stopping_rounds,
        verbose=cfg.verbose_eval,
        allow_writing_files=False,
        auto_class_weights=None,
    )


def save_feature_importance(
    model: CatBoostClassifier,
    feature_columns: list[str],
    output_path: Path,
) -> pd.DataFrame:
    importance = model.get_feature_importance(prettified=False)
    df_imp = pd.DataFrame(
        {"feature": feature_columns, "importance": importance}
    ).sort_values("importance", ascending=False)
    df_imp.to_csv(output_path, index=False, encoding="utf-8")
    return df_imp


def run_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    feature_columns: list[str],
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

        class_weights = build_class_weights(y_train_fold)

        train_pool = Pool(
            data=X_train_fold,
            label=y_train_fold,
            cat_features=categorical_columns,
            feature_names=feature_columns,
        )
        valid_pool = Pool(
            data=X_valid_fold,
            label=y_valid_fold,
            cat_features=categorical_columns,
            feature_names=feature_columns,
        )

        model = create_model(cfg, class_weights)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

        y_valid_pred = model.predict(valid_pool).reshape(-1)
        fold_metrics = evaluate_predictions(y_valid_fold, y_valid_pred)

        fold_result = {
            "fold": fold_idx,
            "best_iteration": int(model.get_best_iteration()) if model.get_best_iteration() is not None else None,
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


def summarize_cv_results(cv_results: list[dict[str, Any]]) -> dict[str, Any]:
    def metric_values(name: str) -> list[float]:
        return [float(r[name]) for r in cv_results if r[name] is not None]

    summary = {}
    for metric in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]:
        values = metric_values(metric)
        summary[metric] = {
            "mean": safe_float(np.mean(values)) if values else None,
            "std": safe_float(np.std(values)) if values else None,
            "min": safe_float(np.min(values)) if values else None,
            "max": safe_float(np.max(values)) if values else None,
        }

    return summary


# ============================================================
# Main pipeline
# ============================================================

def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-mode",
        type=str,
        default="property_only",
        choices=["property_only", "property_plus_light_business", "full"],
    )
    args = parser.parse_args()

    cfg = TrainingConfig(feature_mode=args.feature_mode)
    ensure_output_dirs(cfg)

    train_path = cfg.data_dir / cfg.train_filename
    test_path = cfg.data_dir / cfg.test_filename

    train_df = load_excel(train_path)
    test_df = load_excel(test_path)

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

    # ------------------------------------------------------------
    # Cross-validation on train set
    # ------------------------------------------------------------
    logger.info("Running %d-fold cross-validation...", cfg.n_splits_cv)
    cv_results = run_cross_validation(
        X=X_train_all,
        y=y_train_all,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        cfg=cfg,
    )
    cv_summary = summarize_cv_results(cv_results)

    # ------------------------------------------------------------
    # Final train/validation split for final model selection
    # ------------------------------------------------------------
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_all,
        y_train_all,
        test_size=cfg.validation_size,
        random_state=cfg.random_state,
        stratify=y_train_all,
    )

    class_weights = build_class_weights(y_train)

    train_pool = Pool(
        data=X_train,
        label=y_train,
        cat_features=categorical_columns,
        feature_names=feature_columns,
    )
    valid_pool = Pool(
        data=X_valid,
        label=y_valid,
        cat_features=categorical_columns,
        feature_names=feature_columns,
    )
    test_pool = Pool(
        data=X_test,
        label=y_test,
        cat_features=categorical_columns,
        feature_names=feature_columns,
    )

    logger.info("Training final model...")
    final_model = create_model(cfg, class_weights)
    final_model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    y_valid_pred = final_model.predict(valid_pool).reshape(-1)
    y_test_pred = final_model.predict(test_pool).reshape(-1)
    y_test_proba = final_model.predict_proba(test_pool)

    valid_metrics = evaluate_predictions(y_valid, y_valid_pred)
    test_metrics = evaluate_predictions(y_test, y_test_pred)

    logger.info(
        "Final validation | macro_f1=%.4f | balanced_accuracy=%.4f",
        valid_metrics["macro_f1"],
        valid_metrics["balanced_accuracy"],
    )
    logger.info(
        "Final test       | macro_f1=%.4f | balanced_accuracy=%.4f",
        test_metrics["macro_f1"],
        test_metrics["balanced_accuracy"],
    )

    # ------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------
    model_path = cfg.output_dir / "buyer_profile_catboost_model.cbm"
    bundle_path = cfg.output_dir / "buyer_profile_catboost_bundle.joblib"
    metrics_path = cfg.output_dir / "metrics.json"
    cv_results_path = cfg.output_dir / "cv_results.json"
    metadata_path = cfg.output_dir / "training_metadata.json"
    feature_importance_path = cfg.output_dir / "feature_importance.csv"
    valid_predictions_path = cfg.output_dir / "validation_predictions.csv"
    test_predictions_path = cfg.output_dir / "test_predictions.csv"

    final_model.save_model(model_path)

    feature_importance_df = save_feature_importance(
        final_model,
        feature_columns,
        feature_importance_path,
    )

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

    metrics_payload = {
        "validation": valid_metrics,
        "test": test_metrics,
        "cv_summary": cv_summary,
        "top_25_feature_importance": feature_importance_df.head(25).to_dict(orient="records"),
    }
    save_json(metrics_path, metrics_payload)

    save_json(
        cv_results_path,
        {
            "feature_mode": cfg.feature_mode,
            "fold_results": cv_results,
            "summary": cv_summary,
        },
    )

    metadata_payload = {
        "config": {
            **asdict(cfg),
            "project_root": str(cfg.project_root),
            "data_dir": str(cfg.data_dir),
            "output_root": str(cfg.output_root),
            "output_dir": str(cfg.output_dir),
        },
        "selected_feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
        "n_classes_train": int(y_train_all.nunique()),
        "classes_train": sorted(y_train_all.unique().tolist()),
        "best_iteration": int(final_model.get_best_iteration()) if final_model.get_best_iteration() is not None else None,
        "best_score": final_model.get_best_score(),
    }
    save_json(metadata_path, metadata_payload)

    joblib.dump(
        {
            "model_path": str(model_path),
            "feature_mode": cfg.feature_mode,
            "feature_columns": feature_columns,
            "categorical_columns": categorical_columns,
            "target_column": cfg.target_column,
        },
        bundle_path,
    )

    logger.info("Artifacts saved to: %s", cfg.output_dir)
    logger.info("Training completed successfully.")


if __name__ == "__main__":
    main()