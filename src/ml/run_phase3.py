"""
Phase 3 — Feature Pruning & Hyperparameter Tuning.

This script:
1. Loads the same train/test splits used in Phase 2
2. Prunes zero/low-importance features per algorithm
3. Runs Optuna hyperparameter search (objective = 5-fold CV macro F1)
4. Refits with best params on full train set
5. Evaluates on held-out test set
6. Saves all artefacts to data/models/property_buyer/phase3/<model>/
"""
from __future__ import annotations

import json
import logging
import math
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from pandas.api.types import CategoricalDtype, is_object_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / "property_buyer"
PHASE3_DIR = ROOT / "data" / "models" / "property_buyer" / "phase3"
RANDOM_STATE = 42

# ── All 50 features used in Phase 2 ──────────────────────────

ALL_FEATURES = [
    "rajoni_akp", "kategoria_e_asetit", "menyra_e_shitjes",
    "komuna_lokacioni_i_asetit_te_shitur",
    "qytet_apo_fshat_lokacioni_i_asetit_te_shitur", "zona_kadastrale",
    "eshte_shitur_si_ndermarrje_e_re_apo_aset_ne_likuidim",
    "siperfaqja_e_objektit_nese_ka_objekt_te_toka_dhe_dihet_siperfaqja_m2",
    "siperfaqja_e_tokes_ne_metra_katror",
    "siperfaqja_e_objektit_nese_ka_objekt_te_toka_dhe_dihet_siperfaqja_m2_was_missing",
    "siperfaqja_e_tokes_ne_metra_katror_was_missing",
    "raw_has_object_area", "raw_has_land_area", "total_area_m2",
    "object_to_land_ratio", "is_land_only", "has_object", "is_large_land",
    "area_data_completeness_score", "asset_structure_type",
    "cmimi_i_shitjes_se_asetit", "cmimi_i_shitjes_se_asetit_was_missing",
    "price_per_total_m2", "price_per_land_m2", "price_per_object_m2",
    "log_sale_price", "is_high_value_asset", "area_price_interaction",
    "contract_year", "contract_month", "contract_quarter",
    "siperfaqja_e_objektit_nese_ka_objekt_te_toka_dhe_dihet_siperfaqja_m2_is_outlier_iqr",
    "siperfaqja_e_tokes_ne_metra_katror_is_outlier_iqr",
    "total_area_m2_is_outlier_iqr", "price_per_land_m2_is_outlier_iqr",
    "data_e_kontrates_was_missing",
    "arbk_numripunetoreve", "arbk_kapitali",
    "business_age_days_at_sale", "business_age_years_at_sale",
    "capital_to_sale_price_ratio",
    "arbk_kapitali_was_missing", "arbk_dataregjistrimit_was_missing",
    "arbk_statusiarbk", "arbk_aktiviteti_1_pershkrimi",
    "arbk_aktiviteti_2_pershkrimi", "arbk_pronari_1_kapitali",
    "arbk_pronari_1_kapitali_is_outlier_iqr",
    "capital_to_sale_price_ratio_is_outlier_iqr",
    "arbk_aktiviteti_1_kodinace_is_outlier_iqr",
    "arbk_aktiviteti_2_kodinace_is_outlier_iqr",
    "arbk_aktiviteti_3_kodinace_is_outlier_iqr",
]

# Features to DROP — zero importance in CatBoost Phase 2 + redundant flags
ZERO_IMPORTANCE_FEATURES = {
    "raw_has_land_area", "zona_kadastrale",
    "siperfaqja_e_objektit_nese_ka_objekt_te_toka_dhe_dihet_siperfaqja_m2_was_missing",
    "siperfaqja_e_tokes_ne_metra_katror_was_missing",
    "raw_has_object_area", "has_object", "is_large_land",
    "cmimi_i_shitjes_se_asetit_was_missing",
    "total_area_m2_is_outlier_iqr", "price_per_land_m2_is_outlier_iqr",
    "arbk_kapitali_was_missing", "arbk_dataregjistrimit_was_missing",
    "arbk_aktiviteti_2_kodinace_is_outlier_iqr",
}

# Additional low-value features (< 0.05% importance) + collinear
LOW_VALUE_FEATURES = {
    "is_high_value_asset", "area_price_interaction",
    "area_data_completeness_score", "data_e_kontrates_was_missing",
    "siperfaqja_e_tokes_ne_metra_katror_is_outlier_iqr",
    "siperfaqja_e_objektit_nese_ka_objekt_te_toka_dhe_dihet_siperfaqja_m2_is_outlier_iqr",
}

DROP_FEATURES = ZERO_IMPORTANCE_FEATURES | LOW_VALUE_FEATURES
PRUNED_FEATURES = [f for f in ALL_FEATURES if f not in DROP_FEATURES]


# ── helpers ────────────────────────────────────────────────

def safe_float(v: Any) -> float | None:
    try:
        v = float(v)
        return None if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return None


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_excel(DATA_DIR / "train_dataset_v1.0.xlsx")
    test_df = pd.read_excel(DATA_DIR / "test_dataset_v1.0.xlsx")
    target = "buyer_profile"
    train_df = train_df.dropna(subset=[target])
    test_df = test_df.dropna(subset=[target])
    # filter classes with < 3 samples
    counts = train_df[target].value_counts()
    keep = counts[counts >= 3].index
    train_df = train_df[train_df[target].isin(keep)].copy()
    test_df = test_df[test_df[target].isin(keep)].copy()
    return train_df, test_df


def detect_cat_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    from pandas.api.types import is_string_dtype
    return [c for c in cols if c in df.columns and
            (is_object_dtype(df[c]) or is_string_dtype(df[c]) or isinstance(df[c].dtype, CategoricalDtype))]


def prep_xy(df: pd.DataFrame, features: list[str], cat_cols: list[str]):
    feats = [f for f in features if f in df.columns]
    X = df[feats].copy()
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype(object).fillna("__missing__")
    num_cols = [c for c in feats if c not in cat_cols]
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    y = df["buyer_profile"].astype(str)
    return X, y, feats


def evaluate(y_true, y_pred) -> dict:
    return {
        "accuracy": safe_float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": safe_float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": safe_float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": safe_float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def cv_macro_f1(make_model_fn, X, y, cat_cols, n_splits=5, is_catboost=False):
    """Return mean macro F1 over stratified folds."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    feat_names = list(X.columns)
    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        Xt, yt = X.iloc[train_idx], y.iloc[train_idx]
        Xv, yv = X.iloc[val_idx], y.iloc[val_idx]
        model = make_model_fn()
        if is_catboost:
            tp = Pool(Xt, yt, cat_features=cat_cols, feature_names=feat_names)
            vp = Pool(Xv, yv, cat_features=cat_cols, feature_names=feat_names)
            model.fit(tp, eval_set=vp, use_best_model=True)
            preds = model.predict(vp).reshape(-1)
        else:
            model.fit(Xt, yt)
            preds = model.predict(Xv)
        f = f1_score(yv, preds, average="macro", zero_division=0)
        scores.append(f)
        logger.info("  Fold %d/%d  macro_f1=%.4f", fold_i, n_splits, f)
    return float(np.mean(scores))


def save_json(path: Path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ================================================================
# 1. CatBoost Phase 3
# ================================================================

def run_catboost(train_df, test_df, n_trials=40):
    logger.info("=" * 60)
    logger.info("CATBOOST — Phase 3 feature pruning + HP search (%d trials)", n_trials)
    out = PHASE3_DIR / "catboost"; out.mkdir(parents=True, exist_ok=True)

    features = PRUNED_FEATURES
    cat_cols = detect_cat_cols(train_df, features)
    X_train, y_train, used_feats = prep_xy(train_df, features, cat_cols)
    X_test, y_test, _ = prep_xy(test_df, features, cat_cols)
    cat_indices = [used_feats.index(c) for c in cat_cols if c in used_feats]

    logger.info("Features used: %d (pruned from %d)", len(used_feats), len(ALL_FEATURES))
    logger.info("Removed features: %s", sorted(DROP_FEATURES & set(ALL_FEATURES)))

    # Class weights
    counts = y_train.value_counts()
    total = len(y_train); n_cls = len(counts)
    cw = {str(c): total / (n_cls * cnt) for c, cnt in counts.items()}

    # Baseline (Phase 2 params on pruned features)
    def make_baseline():
        return CatBoostClassifier(
            iterations=2500, learning_rate=0.03, depth=7,
            l2_leaf_reg=9.0, random_strength=1.5,
            bagging_temperature=1.0, border_count=254,
            class_weights=cw, early_stopping_rounds=150,
            random_seed=RANDOM_STATE, verbose=0, allow_writing_files=False,
            loss_function="MultiClass", eval_metric="TotalF1:average=Macro",
        )

    baseline_f1 = cv_macro_f1(make_baseline, X_train, y_train, cat_cols, is_catboost=True)
    logger.info("CatBoost baseline (pruned, Phase 2 params) CV macro F1: %.4f", baseline_f1)

    search_log = [{"trial": 0, "params": "baseline", "cv_macro_f1": baseline_f1}]

    def objective(trial):
        lr = trial.suggest_float("learning_rate", 0.01, 0.10, log=True)
        depth = trial.suggest_int("depth", 4, 10)
        l2 = trial.suggest_float("l2_leaf_reg", 1.0, 15.0, log=True)
        rs = trial.suggest_float("random_strength", 0.0, 10.0)
        bt = trial.suggest_float("bagging_temperature", 0.0, 1.0)
        bc = trial.suggest_int("border_count", 32, 254)
        acw = trial.suggest_categorical("auto_class_weights", ["None", "Balanced", "SqrtBalanced"])

        def make_model():
            return CatBoostClassifier(
                iterations=3000, learning_rate=lr, depth=depth,
                l2_leaf_reg=l2, random_strength=rs,
                bagging_temperature=bt, border_count=bc,
                auto_class_weights=acw if acw != "None" else None,
                class_weights=cw if acw == "None" else None,
                early_stopping_rounds=150,
                random_seed=RANDOM_STATE, verbose=0, allow_writing_files=False,
                loss_function="MultiClass", eval_metric="TotalF1:average=Macro",
            )

        score = cv_macro_f1(make_model, X_train, y_train, cat_cols, is_catboost=True)
        search_log.append({"trial": trial.number + 1, "params": trial.params, "cv_macro_f1": score})
        return score

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best_f1 = study.best_value
    logger.info("CatBoost best trial CV macro F1: %.4f  params: %s", best_f1, best)

    # Refit on full train
    acw = best.pop("auto_class_weights")
    final_model = CatBoostClassifier(
        iterations=3000, **best,
        auto_class_weights=acw if acw != "None" else None,
        class_weights=cw if acw == "None" else None,
        early_stopping_rounds=150,
        random_seed=RANDOM_STATE, verbose=0, allow_writing_files=False,
        loss_function="MultiClass", eval_metric="TotalF1:average=Macro",
    )
    Xtr, Xval, ytr, yval = train_test_split(X_train, y_train, test_size=0.15,
                                              random_state=RANDOM_STATE, stratify=y_train)
    feat_names = list(X_train.columns)
    tp = Pool(Xtr, ytr, cat_features=cat_cols, feature_names=feat_names)
    vp = Pool(Xval, yval, cat_features=cat_cols, feature_names=feat_names)
    final_model.fit(tp, eval_set=vp, use_best_model=True)

    # Test eval
    test_pool = Pool(X_test, cat_features=cat_cols, feature_names=feat_names)
    test_preds = final_model.predict(test_pool).reshape(-1)
    test_metrics = evaluate(y_test, test_preds)

    val_preds = final_model.predict(vp).reshape(-1)
    val_metrics = evaluate(yval, val_preds)

    logger.info("CatBoost Phase 3 TEST — acc=%.4f  bacc=%.4f  f1m=%.4f  f1w=%.4f",
                test_metrics["accuracy"], test_metrics["balanced_accuracy"],
                test_metrics["macro_f1"], test_metrics["weighted_f1"])

    # Save
    save_json(out / "metrics.json", {"validation": val_metrics, "test": test_metrics})
    save_json(out / "best_params.json", {"best_trial_cv_f1": best_f1, "params": {**best, "auto_class_weights": acw}})
    save_json(out / "search_log.json", search_log)
    save_json(out / "feature_list.json", {"features_used": used_feats, "features_removed": sorted(DROP_FEATURES & set(ALL_FEATURES)), "n_used": len(used_feats), "n_removed": len(DROP_FEATURES & set(ALL_FEATURES))})

    # feature importance
    imp = final_model.get_feature_importance(prettified=False)
    imp_df = pd.DataFrame({"feature": used_feats, "importance": imp}).sort_values("importance", ascending=False)
    imp_df.to_csv(out / "feature_importance.csv", index=False)

    return test_metrics


# ================================================================
# 2. Random Forest Phase 3
# ================================================================

def run_random_forest(train_df, test_df, n_trials=40):
    logger.info("=" * 60)
    logger.info("RANDOM FOREST — Phase 3 feature pruning + HP search (%d trials)", n_trials)
    out = PHASE3_DIR / "random_forest"; out.mkdir(parents=True, exist_ok=True)

    features = PRUNED_FEATURES
    cat_cols = detect_cat_cols(train_df, features)
    X_train_raw, y_train, used_feats = prep_xy(train_df, features, cat_cols)
    X_test_raw, y_test, _ = prep_xy(test_df, features, cat_cols)
    num_cols = [c for c in used_feats if c not in cat_cols]

    logger.info("Features used: %d", len(used_feats))

    def build_pipe(n_est, max_d, mss, msl, mf, cw):
        cat_tf = Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
        ])
        num_tf = Pipeline([("imp", SimpleImputer(strategy="median"))])
        pre = ColumnTransformer([
            ("cat", cat_tf, cat_cols),
            ("num", num_tf, num_cols),
        ], remainder="drop")
        clf = RandomForestClassifier(
            n_estimators=n_est, max_depth=max_d,
            min_samples_split=mss, min_samples_leaf=msl,
            max_features=mf, class_weight=cw,
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        return Pipeline([("pre", pre), ("clf", clf)])

    # Baseline
    def make_baseline():
        return build_pipe(1200, 24, 6, 1, 0.35, "balanced_subsample")

    baseline_f1 = cv_macro_f1(make_baseline, X_train_raw, y_train, cat_cols)
    logger.info("RF baseline (pruned, Phase 2 params) CV macro F1: %.4f", baseline_f1)
    search_log = [{"trial": 0, "params": "baseline", "cv_macro_f1": baseline_f1}]

    def objective(trial):
        n_est = trial.suggest_int("n_estimators", 400, 2000, step=200)
        max_d = trial.suggest_int("max_depth", 8, 32)
        mss = trial.suggest_int("min_samples_split", 2, 20)
        msl = trial.suggest_int("min_samples_leaf", 1, 8)
        mf = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.4, 0.5])
        cw = trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"])

        def make_model():
            return build_pipe(n_est, max_d, mss, msl, mf, cw)

        score = cv_macro_f1(make_model, X_train_raw, y_train, cat_cols)
        search_log.append({"trial": trial.number + 1, "params": trial.params, "cv_macro_f1": score})
        return score

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_params
    best_f1 = study.best_value
    logger.info("RF best trial CV macro F1: %.4f  params: %s", best_f1, best)

    # Refit
    final_pipe = build_pipe(
        best["n_estimators"], best["max_depth"], best["min_samples_split"],
        best["min_samples_leaf"], best["max_features"], best["class_weight"],
    )

    Xtr, Xval, ytr, yval = train_test_split(X_train_raw, y_train, test_size=0.15,
                                              random_state=RANDOM_STATE, stratify=y_train)
    final_pipe.fit(Xtr, ytr)
    val_preds = final_pipe.predict(Xval)
    test_preds = final_pipe.predict(X_test_raw)

    val_metrics = evaluate(yval, val_preds)
    test_metrics = evaluate(y_test, test_preds)

    logger.info("RF Phase 3 TEST — acc=%.4f  bacc=%.4f  f1m=%.4f  f1w=%.4f",
                test_metrics["accuracy"], test_metrics["balanced_accuracy"],
                test_metrics["macro_f1"], test_metrics["weighted_f1"])

    save_json(out / "metrics.json", {"validation": val_metrics, "test": test_metrics})
    save_json(out / "best_params.json", {"best_trial_cv_f1": best_f1, "params": best})
    save_json(out / "search_log.json", search_log)
    save_json(out / "feature_list.json", {"features_used": used_feats, "features_removed": sorted(DROP_FEATURES & set(ALL_FEATURES)), "n_used": len(used_feats), "n_removed": len(DROP_FEATURES & set(ALL_FEATURES))})

    # Refit on ALL train for final test
    final_pipe.fit(X_train_raw, y_train)
    test_preds_full = final_pipe.predict(X_test_raw)
    test_metrics_full = evaluate(y_test, test_preds_full)
    save_json(out / "metrics_full_refit.json", {"test": test_metrics_full})

    logger.info("RF Phase 3 full-refit TEST — acc=%.4f  bacc=%.4f  f1m=%.4f  f1w=%.4f",
                test_metrics_full["accuracy"], test_metrics_full["balanced_accuracy"],
                test_metrics_full["macro_f1"], test_metrics_full["weighted_f1"])

    return test_metrics_full


# ================================================================
# 3. Logistic Regression Phase 3
# ================================================================

def run_logistic_regression(train_df, test_df):
    logger.info("=" * 60)
    logger.info("LOGISTIC REGRESSION — Phase 3 feature pruning + grid search")
    out = PHASE3_DIR / "logistic_regression"; out.mkdir(parents=True, exist_ok=True)

    features = PRUNED_FEATURES
    cat_cols = detect_cat_cols(train_df, features)
    X_train_raw, y_train, used_feats = prep_xy(train_df, features, cat_cols)
    X_test_raw, y_test, _ = prep_xy(test_df, features, cat_cols)
    num_cols = [c for c in used_feats if c not in cat_cols]

    logger.info("Features used: %d", len(used_feats))

    def build_pipe(C, penalty, solver, class_weight):
        cat_tf = Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
        ])
        num_tf = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
        ])
        pre = ColumnTransformer([
            ("cat", cat_tf, cat_cols),
            ("num", num_tf, num_cols),
        ], remainder="drop")
        clf = LogisticRegression(
            C=C, penalty=penalty, solver=solver,
            class_weight=class_weight, max_iter=8000,
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        return Pipeline([("pre", pre), ("clf", clf)])

    # Grid search
    grid = [
        {"C": 1.0,  "penalty": "l2", "solver": "lbfgs", "class_weight": None},
        {"C": 1.0,  "penalty": "l2", "solver": "lbfgs", "class_weight": "balanced"},
        {"C": 0.3,  "penalty": "l2", "solver": "lbfgs", "class_weight": "balanced"},
        {"C": 0.1,  "penalty": "l2", "solver": "lbfgs", "class_weight": "balanced"},
        {"C": 3.0,  "penalty": "l2", "solver": "lbfgs", "class_weight": "balanced"},
        {"C": 10.0, "penalty": "l2", "solver": "lbfgs", "class_weight": "balanced"},
        {"C": 0.3,  "penalty": "l1", "solver": "saga",  "class_weight": "balanced"},
        {"C": 0.1,  "penalty": "l1", "solver": "saga",  "class_weight": "balanced"},
        {"C": 0.03, "penalty": "l1", "solver": "saga",  "class_weight": "balanced"},
        {"C": 1.0,  "penalty": "l1", "solver": "saga",  "class_weight": "balanced"},
    ]

    search_log = []
    best_score = -1; best_params = None
    for cfg in grid:
        def make_model(c=cfg):
            return build_pipe(**c)
        score = cv_macro_f1(make_model, X_train_raw, y_train, cat_cols)
        search_log.append({"params": {k: str(v) for k, v in cfg.items()}, "cv_macro_f1": score})
        logger.info("  LR C=%.3f pen=%s cw=%s → CV macro F1=%.4f", cfg["C"], cfg["penalty"], cfg["class_weight"], score)
        if score > best_score:
            best_score = score
            best_params = cfg

    logger.info("LR best: CV macro F1=%.4f  params=%s", best_score, best_params)

    # Refit on full train
    final_pipe = build_pipe(**best_params)
    final_pipe.fit(X_train_raw, y_train)
    test_preds = final_pipe.predict(X_test_raw)
    test_metrics = evaluate(y_test, test_preds)

    Xtr, Xval, ytr, yval = train_test_split(X_train_raw, y_train, test_size=0.15,
                                              random_state=RANDOM_STATE, stratify=y_train)
    val_pipe = build_pipe(**best_params)
    val_pipe.fit(Xtr, ytr)
    val_preds = val_pipe.predict(Xval)
    val_metrics = evaluate(yval, val_preds)

    logger.info("LR Phase 3 TEST — acc=%.4f  bacc=%.4f  f1m=%.4f  f1w=%.4f",
                test_metrics["accuracy"], test_metrics["balanced_accuracy"],
                test_metrics["macro_f1"], test_metrics["weighted_f1"])

    save_json(out / "metrics.json", {"validation": val_metrics, "test": test_metrics})
    save_json(out / "best_params.json", {"best_cv_f1": best_score, "params": {k: str(v) for k, v in best_params.items()}})
    save_json(out / "search_log.json", search_log)
    save_json(out / "feature_list.json", {"features_used": used_feats, "features_removed": sorted(DROP_FEATURES & set(ALL_FEATURES)), "n_used": len(used_feats), "n_removed": len(DROP_FEATURES & set(ALL_FEATURES))})

    return test_metrics


# ================================================================
# 4. Neural Network Phase 3
# ================================================================

# --- NN feature lists (pruned) ---
NN_CATEGORICAL_FEATURES = [
    "rajoni_akp", "kategoria_e_asetit", "menyra_e_shitjes",
    "komuna_lokacioni_i_asetit_te_shitur",
    "qytet_apo_fshat_lokacioni_i_asetit_te_shitur",
    "eshte_shitur_si_ndermarrje_e_re_apo_aset_ne_likuidim",
    "arbk_statusiarbk", "arbk_aktiviteti_1_pershkrimi",
    "arbk_aktiviteti_2_pershkrimi", "asset_structure_type",
]

NN_NUMERIC_FEATURES = [f for f in PRUNED_FEATURES if f not in NN_CATEGORICAL_FEATURES]


def _emb_dim(n_cat: int) -> int:
    return min(50, 1 + n_cat // 2)


class _NNPreprocessor:
    def __init__(self):
        self.ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value",
                                      unknown_value=-1, encoded_missing_value=-1)
        self.scaler = StandardScaler()
        self.label_enc = LabelEncoder()
        self.cardinalities: list[int] = []

    def fit(self, df, target_col):
        X_cat = df[NN_CATEGORICAL_FEATURES].astype(str).fillna("__missing__")
        X_num = df[NN_NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        self.ord_enc.fit(X_cat)
        self.scaler.fit(X_num)
        self.label_enc.fit(df[target_col].astype(str))
        self.cardinalities = [len(c) + 1 for c in self.ord_enc.categories_]

    def transform(self, df, target_col=None):
        X_cat = df[NN_CATEGORICAL_FEATURES].astype(str).fillna("__missing__")
        X_num = df[NN_NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        cat_enc = self.ord_enc.transform(X_cat).astype(np.int64)
        for i, n in enumerate(self.cardinalities):
            cat_enc[:, i] = np.where(cat_enc[:, i] < 0, n - 1, cat_enc[:, i])
        num_sc = self.scaler.transform(X_num).astype(np.float32)
        y = None
        if target_col and target_col in df.columns:
            y = self.label_enc.transform(df[target_col].astype(str))
        return cat_enc, num_sc, y


class _BuyerNet(nn.Module):
    def __init__(self, cardinalities, n_numeric, hidden_dims, n_classes, dropout, emb_dropout):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(n, _emb_dim(n)) for n in cardinalities]
        )
        self.emb_drop = nn.Dropout(emb_dropout)
        total_emb = sum(_emb_dim(n) for n in cardinalities)
        layers: list[nn.Module] = []
        prev = total_emb + n_numeric
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        embs = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
        x = torch.cat(embs, 1)
        x = self.emb_drop(x)
        x = torch.cat([x, x_num], 1)
        return self.classifier(x)


def _make_loader(cat, num, y, bs, shuffle):
    ts = [torch.tensor(cat, dtype=torch.long), torch.tensor(num, dtype=torch.float32)]
    if y is not None:
        ts.append(torch.tensor(y, dtype=torch.long))
    return DataLoader(TensorDataset(*ts), batch_size=bs, shuffle=shuffle)


def _train_nn(model, tr_loader, va_loader, criterion, optimizer, epochs, patience, device):
    best_loss = float("inf")
    best_state = None
    wait = 0
    for ep in range(1, epochs + 1):
        model.train()
        for batch in tr_loader:
            xc, xn, yt = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            loss = criterion(model(xc, xn), yt)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        vl = 0.0; n = 0
        with torch.no_grad():
            for batch in va_loader:
                xc, xn, yt = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                vl += criterion(model(xc, xn), yt).item() * len(yt); n += len(yt)
        vl /= n
        if vl < best_loss:
            best_loss = vl; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model


def _nn_predict(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            xc, xn = batch[0].to(device), batch[1].to(device)
            preds.append(model(xc, xn).argmax(1).cpu().numpy())
            if len(batch) > 2:
                trues.append(batch[2].numpy())
    return np.concatenate(preds), (np.concatenate(trues) if trues else None)


def _nn_cv_f1(pp, train_df, hidden_dims, dropout, emb_dropout, lr, wd, bs, epochs, patience, n_splits=5):
    """5-fold CV returning mean macro F1 for neural net."""
    device = torch.device("cpu")
    y_all = train_df["buyer_profile"].astype(str)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    n_classes = len(pp.label_enc.classes_)
    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(train_df, y_all), 1):
        fold_pp = _NNPreprocessor()
        fold_pp.fit(train_df.iloc[tr_idx], "buyer_profile")
        cat_tr, num_tr, y_tr = fold_pp.transform(train_df.iloc[tr_idx], "buyer_profile")
        cat_va, num_va, y_va = fold_pp.transform(train_df.iloc[va_idx], "buyer_profile")
        tr_ld = _make_loader(cat_tr, num_tr, y_tr, bs, True)
        va_ld = _make_loader(cat_va, num_va, y_va, bs, False)
        model = _BuyerNet(fold_pp.cardinalities, len(NN_NUMERIC_FEATURES),
                          hidden_dims, n_classes, dropout, emb_dropout).to(device)
        cw = torch.tensor(len(y_tr) / (n_classes * np.maximum(np.bincount(y_tr, minlength=n_classes), 1).astype(np.float64)),
                          dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=cw)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        model = _train_nn(model, tr_ld, va_ld, criterion, optimizer, epochs, patience, device)
        y_pred, y_true = _nn_predict(model, va_ld, device)
        f = f1_score(y_true, y_pred, average="macro", zero_division=0)
        scores.append(f)
        logger.info("  Fold %d/%d  macro_f1=%.4f", fold_i, n_splits, f)
    return float(np.mean(scores))


def run_neural_net(train_df, test_df, n_trials=20):
    logger.info("=" * 60)
    logger.info("NEURAL NET — Phase 3 feature pruning + HP search (%d trials)", n_trials)
    out = PHASE3_DIR / "neural_net"; out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)

    pp = _NNPreprocessor()
    pp.fit(train_df, "buyer_profile")
    n_classes = len(pp.label_enc.classes_)

    logger.info("Features used: %d cat + %d num = %d",
                len(NN_CATEGORICAL_FEATURES), len(NN_NUMERIC_FEATURES),
                len(NN_CATEGORICAL_FEATURES) + len(NN_NUMERIC_FEATURES))

    # Baseline (Phase 2 params)
    baseline_f1 = _nn_cv_f1(pp, train_df,
                            hidden_dims=(256, 128, 64), dropout=0.3, emb_dropout=0.15,
                            lr=1e-3, wd=1e-4, bs=64, epochs=200, patience=20)
    logger.info("NN baseline (pruned, Phase 2 params) CV macro F1: %.4f", baseline_f1)
    search_log = [{"trial": 0, "params": "baseline", "cv_macro_f1": baseline_f1}]

    def objective(trial):
        n_layers = trial.suggest_int("n_layers", 2, 4)
        dims = tuple(trial.suggest_categorical(f"dim_{i}", [64, 128, 256, 512]) for i in range(n_layers))
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        emb_dropout = trial.suggest_float("emb_dropout", 0.0, 0.3)
        lr = trial.suggest_float("learning_rate", 5e-4, 5e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        bs = trial.suggest_categorical("batch_size", [32, 64, 128])

        score = _nn_cv_f1(pp, train_df, hidden_dims=dims, dropout=dropout,
                          emb_dropout=emb_dropout, lr=lr, wd=wd, bs=bs,
                          epochs=200, patience=20)
        search_log.append({"trial": trial.number + 1,
                           "params": {**trial.params, "hidden_dims": list(dims)},
                           "cv_macro_f1": score})
        return score

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_params
    best_f1 = study.best_value
    logger.info("NN best trial CV macro F1: %.4f  params: %s", best_f1, best)

    # Refit with best params
    n_layers = best["n_layers"]
    best_dims = tuple(best[f"dim_{i}"] for i in range(n_layers))
    best_dropout = best["dropout"]
    best_emb_dropout = best["emb_dropout"]
    best_lr = best["learning_rate"]
    best_wd = best["weight_decay"]
    best_bs = best["batch_size"]

    pp_final = _NNPreprocessor()
    pp_final.fit(train_df, "buyer_profile")
    train_sub, val_sub = train_test_split(train_df, test_size=0.15,
                                           stratify=train_df["buyer_profile"],
                                           random_state=RANDOM_STATE)
    cat_tr, num_tr, y_tr = pp_final.transform(train_sub, "buyer_profile")
    cat_va, num_va, y_va = pp_final.transform(val_sub, "buyer_profile")
    cat_te, num_te, y_te = pp_final.transform(test_df, "buyer_profile")

    tr_ld = _make_loader(cat_tr, num_tr, y_tr, best_bs, True)
    va_ld = _make_loader(cat_va, num_va, y_va, best_bs, False)
    te_ld = _make_loader(cat_te, num_te, y_te, best_bs, False)

    model = _BuyerNet(pp_final.cardinalities, len(NN_NUMERIC_FEATURES),
                      best_dims, n_classes, best_dropout, best_emb_dropout).to(device)
    cw = torch.tensor(len(y_tr) / (n_classes * np.maximum(np.bincount(y_tr, minlength=n_classes), 1).astype(np.float64)),
                      dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr, weight_decay=best_wd)
    model = _train_nn(model, tr_ld, va_ld, criterion, optimizer, 200, 20, device)

    y_pred_val, y_true_val = _nn_predict(model, va_ld, device)
    val_labels_pred = pp_final.label_enc.inverse_transform(y_pred_val)
    val_labels_true = pp_final.label_enc.inverse_transform(y_true_val)
    val_metrics = evaluate(val_labels_true, val_labels_pred)

    y_pred_te, y_true_te = _nn_predict(model, te_ld, device)
    te_labels_pred = pp_final.label_enc.inverse_transform(y_pred_te)
    te_labels_true = pp_final.label_enc.inverse_transform(y_true_te)
    test_metrics = evaluate(te_labels_true, te_labels_pred)

    logger.info("NN Phase 3 TEST — acc=%.4f  bacc=%.4f  f1m=%.4f  f1w=%.4f",
                test_metrics["accuracy"], test_metrics["balanced_accuracy"],
                test_metrics["macro_f1"], test_metrics["weighted_f1"])

    save_json(out / "metrics.json", {"validation": val_metrics, "test": test_metrics})
    save_json(out / "best_params.json", {"best_trial_cv_f1": best_f1,
              "params": {**best, "hidden_dims": list(best_dims)}})
    save_json(out / "search_log.json", search_log)
    save_json(out / "feature_list.json", {
        "features_used": NN_CATEGORICAL_FEATURES + NN_NUMERIC_FEATURES,
        "features_removed": sorted(DROP_FEATURES & set(ALL_FEATURES)),
        "n_used": len(NN_CATEGORICAL_FEATURES) + len(NN_NUMERIC_FEATURES),
        "n_removed": len(DROP_FEATURES & set(ALL_FEATURES)),
    })

    return test_metrics


# ================================================================
# Main
# ================================================================

def main():
    train_df, test_df = load_data()
    logger.info("Loaded train=%d  test=%d", len(train_df), len(test_df))
    logger.info("Pruned feature set: %d features (removed %d)", len(PRUNED_FEATURES), len(DROP_FEATURES))
    logger.info("Removed: %s", sorted(DROP_FEATURES))

    results = {}

    # CatBoost
    cb_metrics = run_catboost(train_df, test_df, n_trials=40)
    results["catboost"] = {k: cb_metrics[k] for k in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]}

    # Random Forest
    rf_metrics = run_random_forest(train_df, test_df, n_trials=40)
    results["random_forest"] = {k: rf_metrics[k] for k in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]}

    # Logistic Regression
    lr_metrics = run_logistic_regression(train_df, test_df)
    results["logistic_regression"] = {k: lr_metrics[k] for k in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]}

    # Neural Network
    nn_metrics = run_neural_net(train_df, test_df, n_trials=20)
    results["neural_net"] = {k: nn_metrics[k] for k in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]}

    save_json(PHASE3_DIR / "summary.json", results)

    logger.info("=" * 60)
    logger.info("Phase 3 complete. Summary:")
    for model, m in results.items():
        logger.info("  %-25s acc=%.4f  bacc=%.4f  f1m=%.4f  f1w=%.4f",
                     model, m["accuracy"], m["balanced_accuracy"], m["macro_f1"], m["weighted_f1"])


if __name__ == "__main__":
    main()
