"""
Feedforward neural network with entity embeddings for buyer-profile
multiclass classification.

Uses learned embeddings for high-cardinality categorical features and
standard scaling for numeric features.  Class-weighted cross-entropy
handles the imbalanced target distribution.

Outputs are saved to  data/models/neural_net/.
"""

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
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class NeuralNetConfig:
    project_root: Path = Path(__file__).resolve().parents[3]
    data_dir: Path = project_root / "data" / "processed" / "property_buyer"
    output_root: Path = project_root / "data" / "models"

    train_filename: str = "train_dataset_v1.0.xlsx"
    test_filename: str = "test_dataset_v1.0.xlsx"

    target_column: str = "buyer_profile"

    # Architecture
    hidden_dims: tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.3
    embedding_dropout: float = 0.15

    # Training
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 20          # early-stopping patience

    # Validation split carved from training data
    validation_size: float = 0.15

    # Cross-validation
    n_splits_cv: int = 5

    random_state: int = 42

    fill_missing_categorical_with: str = "__missing__"

    @property
    def output_dir(self) -> Path:
        return self.output_root / "neural_net"


# ============================================================
# Feature list
# ============================================================

FEATURES = [
    "rajoni_akp",
    "kategoria_e_asetit",
    "menyra_e_shitjes",
    "komuna_lokacioni_i_asetit_te_shitur",
    "qytet_apo_fshat_lokacioni_i_asetit_te_shitur",
    "zona_kadastrale",
    "eshte_shitur_si_ndermarrje_e_re_apo_aset_ne_likuidim",
    "arbk_numripunetoreve",
    "arbk_kapitali",
    "business_age_days_at_sale",
    "business_age_years_at_sale",
    "capital_to_sale_price_ratio",
    "arbk_kapitali_was_missing",
    "arbk_dataregjistrimit_was_missing",
    "arbk_statusiarbk",
    "arbk_aktiviteti_1_pershkrimi",
    "arbk_aktiviteti_2_pershkrimi",
    "arbk_pronari_1_kapitali",
    "arbk_pronari_1_kapitali_is_outlier_iqr",
    "capital_to_sale_price_ratio_is_outlier_iqr",
    "arbk_aktiviteti_1_kodinace_is_outlier_iqr",
    "arbk_aktiviteti_2_kodinace_is_outlier_iqr",
    "arbk_aktiviteti_3_kodinace_is_outlier_iqr",
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
    "cmimi_i_shitjes_se_asetit",
    "cmimi_i_shitjes_se_asetit_was_missing",
    "price_per_total_m2",
    "price_per_land_m2",
    "price_per_object_m2",
    "log_sale_price",
    "is_high_value_asset",
    "area_price_interaction",
    "contract_year",
    "contract_month",
    "contract_quarter",
    "siperfaqja_e_objektit_nese_ka_objekt_te_toka_dhe_dihet_siperfaqja_m2_is_outlier_iqr",
    "siperfaqja_e_tokes_ne_metra_katror_is_outlier_iqr",
    "total_area_m2_is_outlier_iqr",
    "price_per_land_m2_is_outlier_iqr",
    "data_e_kontrates_was_missing",
]

CATEGORICAL_FEATURES = [
    "rajoni_akp",
    "kategoria_e_asetit",
    "menyra_e_shitjes",
    "komuna_lokacioni_i_asetit_te_shitur",
    "qytet_apo_fshat_lokacioni_i_asetit_te_shitur",
    "zona_kadastrale",
    "eshte_shitur_si_ndermarrje_e_re_apo_aset_ne_likuidim",
    "arbk_statusiarbk",
    "arbk_aktiviteti_1_pershkrimi",
    "arbk_aktiviteti_2_pershkrimi",
    "asset_structure_type",
]

NUMERIC_FEATURES = [f for f in FEATURES if f not in CATEGORICAL_FEATURES]

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
# Helpers
# ============================================================

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


def embedding_dim_rule(n_categories: int) -> int:
    """Heuristic: min(50, 1 + n_categories // 2)."""
    return min(50, 1 + n_categories // 2)


# ============================================================
# Preprocessing
# ============================================================

class FeaturePreprocessor:
    """Fit on training data, transform any split consistently."""

    def __init__(self, fill_missing: str = "__missing__"):
        self.fill_missing = fill_missing
        self.ordinal_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1,
            encoded_missing_value=-1,
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.category_cardinalities: list[int] = []

    def fit(self, df: pd.DataFrame, target_col: str) -> None:
        X_cat = self._prep_cat(df)
        X_num = self._prep_num(df)

        self.ordinal_encoder.fit(X_cat)
        self.scaler.fit(X_num)
        self.label_encoder.fit(df[target_col].astype(str))

        # +1 for the unknown / missing sentinel mapped to index n_cats
        self.category_cardinalities = [
            len(cats) + 1 for cats in self.ordinal_encoder.categories_
        ]

    def transform(
        self, df: pd.DataFrame, target_col: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        X_cat = self._prep_cat(df)
        X_num = self._prep_num(df)

        cat_encoded = self.ordinal_encoder.transform(X_cat).astype(np.int64)
        # Map -1 (unknown/missing) to last index for embedding lookup
        for i, n in enumerate(self.category_cardinalities):
            cat_encoded[:, i] = np.where(cat_encoded[:, i] < 0, n - 1, cat_encoded[:, i])

        num_scaled = self.scaler.transform(X_num).astype(np.float32)

        y = None
        if target_col is not None and target_col in df.columns:
            y = self.label_encoder.transform(df[target_col].astype(str))

        return cat_encoded, num_scaled, y

    # ------ internal ------
    def _prep_cat(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[CATEGORICAL_FEATURES].copy()
        for col in CATEGORICAL_FEATURES:
            X[col] = X[col].astype(str).fillna(self.fill_missing)
        return X

    def _prep_num(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[NUMERIC_FEATURES].copy()
        for col in NUMERIC_FEATURES:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
        return X


# ============================================================
# Model
# ============================================================

class BuyerProfileNet(nn.Module):
    """Entity-embedding feedforward classifier."""

    def __init__(
        self,
        category_cardinalities: list[int],
        n_numeric: int,
        hidden_dims: tuple[int, ...],
        n_classes: int,
        dropout: float,
        embedding_dropout: float,
    ):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(n_cat, embedding_dim_rule(n_cat))
            for n_cat in category_cardinalities
        ])
        self.emb_dropout = nn.Dropout(embedding_dropout)

        total_emb_dim = sum(embedding_dim_rule(n) for n in category_cardinalities)
        input_dim = total_emb_dim + n_numeric

        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_emb = torch.cat(embs, dim=1)
        x_emb = self.emb_dropout(x_emb)
        x = torch.cat([x_emb, x_num], dim=1)
        return self.classifier(x)


# ============================================================
# Metrics
# ============================================================

def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str],
) -> dict[str, Any]:
    report = classification_report(
        y_true, y_pred, target_names=label_names,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": safe_float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": safe_float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": safe_float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": safe_float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


# ============================================================
# Training helpers
# ============================================================

def compute_class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1)
    weights = len(y) / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def _make_loader(
    cat: np.ndarray, num: np.ndarray, y: np.ndarray | None,
    batch_size: int, shuffle: bool,
) -> DataLoader:
    tensors = [torch.tensor(cat, dtype=torch.long), torch.tensor(num, dtype=torch.float32)]
    if y is not None:
        tensors.append(torch.tensor(y, dtype=torch.long))
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(
    model: BuyerProfileNet, loader: DataLoader,
    criterion: nn.Module, optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        x_cat, x_num, targets = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        logits = model(x_cat, x_num)
        loss = criterion(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(targets)
        n += len(targets)
    return total_loss / n


@torch.no_grad()
def evaluate_epoch(
    model: BuyerProfileNet, loader: DataLoader,
    criterion: nn.Module, device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    n = 0
    all_preds, all_true = [], []
    for batch in loader:
        x_cat, x_num, targets = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        logits = model(x_cat, x_num)
        loss = criterion(logits, targets)
        total_loss += loss.item() * len(targets)
        n += len(targets)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_true.append(targets.cpu().numpy())
    return total_loss / n, np.concatenate(all_preds), np.concatenate(all_true)


# ============================================================
# Cross-validation
# ============================================================

def run_cross_validation(
    train_df: pd.DataFrame,
    preprocessor: FeaturePreprocessor,
    cfg: NeuralNetConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Stratified k-fold CV on training data."""

    y_all = train_df[cfg.target_column].astype(str)
    skf = StratifiedKFold(n_splits=cfg.n_splits_cv, shuffle=True, random_state=cfg.random_state)
    n_classes = len(preprocessor.label_encoder.classes_)
    label_names = list(preprocessor.label_encoder.classes_)

    fold_results = []
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(train_df, y_all), 1):
        logger.info("CV fold %d/%d", fold_idx, cfg.n_splits_cv)
        fold_train = train_df.iloc[tr_idx]
        fold_val = train_df.iloc[val_idx]

        # Fit a fresh preprocessor per fold
        fold_pp = FeaturePreprocessor(fill_missing=cfg.fill_missing_categorical_with)
        fold_pp.fit(fold_train, cfg.target_column)

        cat_tr, num_tr, y_tr = fold_pp.transform(fold_train, cfg.target_column)
        cat_va, num_va, y_va = fold_pp.transform(fold_val, cfg.target_column)

        tr_loader = _make_loader(cat_tr, num_tr, y_tr, cfg.batch_size, shuffle=True)
        va_loader = _make_loader(cat_va, num_va, y_va, cfg.batch_size, shuffle=False)

        model = BuyerProfileNet(
            category_cardinalities=fold_pp.category_cardinalities,
            n_numeric=len(NUMERIC_FEATURES),
            hidden_dims=cfg.hidden_dims,
            n_classes=n_classes,
            dropout=cfg.dropout,
            embedding_dropout=cfg.embedding_dropout,
        ).to(device)

        class_w = compute_class_weights(y_tr, n_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_w)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        best_val_loss = float("inf")
        wait = 0
        for epoch in range(1, cfg.epochs + 1):
            train_one_epoch(model, tr_loader, criterion, optimizer, device)
            val_loss, _, _ = evaluate_epoch(model, va_loader, criterion, device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
            else:
                wait += 1
                if wait >= cfg.patience:
                    break

        _, y_pred_va, y_true_va = evaluate_epoch(model, va_loader, criterion, device)
        fold_metrics = evaluate_predictions(y_true_va, y_pred_va, label_names)
        fold_results.append({
            "fold": fold_idx,
            "stopped_epoch": epoch,
            **{k: v for k, v in fold_metrics.items() if k != "classification_report" and k != "confusion_matrix"},
        })

    summary = {}
    for key in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]:
        vals = [f[key] for f in fold_results if f[key] is not None]
        summary[key] = {
            "mean": safe_float(np.mean(vals)),
            "std": safe_float(np.std(vals)),
            "min": safe_float(np.min(vals)),
            "max": safe_float(np.max(vals)),
        }

    return {"folds": fold_results, "summary": summary}


# ============================================================
# Main training
# ============================================================

def train(cfg: NeuralNetConfig | None = None) -> None:
    setup_logging()

    if cfg is None:
        cfg = NeuralNetConfig()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.random_state)
    np.random.seed(cfg.random_state)

    device = torch.device("cpu")

    # --- Load data ---
    train_path = cfg.data_dir / cfg.train_filename
    test_path = cfg.data_dir / cfg.test_filename
    logger.info("Loading train: %s", train_path)
    logger.info("Loading test:  %s", test_path)
    train_df = pd.read_excel(train_path)
    test_df = pd.read_excel(test_path)
    logger.info("Train shape: %s  |  Test shape: %s", train_df.shape, test_df.shape)

    # --- Fit preprocessor on full training set ---
    preprocessor = FeaturePreprocessor(fill_missing=cfg.fill_missing_categorical_with)
    preprocessor.fit(train_df, cfg.target_column)
    n_classes = len(preprocessor.label_encoder.classes_)
    label_names = list(preprocessor.label_encoder.classes_)
    logger.info("Classes (%d): %s", n_classes, label_names)

    # --- Cross-validation ---
    logger.info("Running %d-fold stratified cross-validation", cfg.n_splits_cv)
    cv_results = run_cross_validation(train_df, preprocessor, cfg, device)
    logger.info(
        "CV summary — macro_f1: %.4f ± %.4f",
        cv_results["summary"]["macro_f1"]["mean"],
        cv_results["summary"]["macro_f1"]["std"],
    )

    # --- Train / validation split for final model ---
    train_sub, val_sub = train_test_split(
        train_df, test_size=cfg.validation_size,
        stratify=train_df[cfg.target_column], random_state=cfg.random_state,
    )

    cat_tr, num_tr, y_tr = preprocessor.transform(train_sub, cfg.target_column)
    cat_va, num_va, y_va = preprocessor.transform(val_sub, cfg.target_column)
    cat_te, num_te, y_te = preprocessor.transform(test_df, cfg.target_column)

    tr_loader = _make_loader(cat_tr, num_tr, y_tr, cfg.batch_size, shuffle=True)
    va_loader = _make_loader(cat_va, num_va, y_va, cfg.batch_size, shuffle=False)
    te_loader = _make_loader(cat_te, num_te, y_te, cfg.batch_size, shuffle=False)

    # --- Build model ---
    model = BuyerProfileNet(
        category_cardinalities=preprocessor.category_cardinalities,
        n_numeric=len(NUMERIC_FEATURES),
        hidden_dims=cfg.hidden_dims,
        n_classes=n_classes,
        dropout=cfg.dropout,
        embedding_dropout=cfg.embedding_dropout,
    ).to(device)

    class_weights = compute_class_weights(y_tr, n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    # --- Training loop with early stopping ---
    best_val_loss = float("inf")
    best_state = None
    wait = 0
    train_losses, val_losses = [], []

    for epoch in range(1, cfg.epochs + 1):
        tr_loss = train_one_epoch(model, tr_loader, criterion, optimizer, device)
        va_loss, _, _ = evaluate_epoch(model, va_loader, criterion, device)
        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch % 20 == 0 or epoch == 1:
            logger.info(
                "Epoch %d/%d — train_loss: %.4f  val_loss: %.4f  patience: %d/%d",
                epoch, cfg.epochs, tr_loss, va_loss, wait, cfg.patience,
            )

        if wait >= cfg.patience:
            logger.info("Early stopping at epoch %d", epoch)
            break

    # Restore best weights
    model.load_state_dict(best_state)

    # --- Evaluate on validation ---
    _, y_pred_val, y_true_val = evaluate_epoch(model, va_loader, criterion, device)
    val_metrics = evaluate_predictions(y_true_val, y_pred_val, label_names)
    logger.info("Validation — macro_f1: %.4f  accuracy: %.4f", val_metrics["macro_f1"], val_metrics["accuracy"])

    # --- Evaluate on test ---
    _, y_pred_test, y_true_test = evaluate_epoch(model, te_loader, criterion, device)
    test_metrics = evaluate_predictions(y_true_test, y_pred_test, label_names)
    logger.info("Test — macro_f1: %.4f  accuracy: %.4f", test_metrics["macro_f1"], test_metrics["accuracy"])

    # --- Save artefacts ---
    # 1. Model weights
    torch.save(best_state, cfg.output_dir / "buyer_profile_neural_net_weights.pt")

    # 2. Bundle (preprocessor + model config for reconstruction)
    bundle = {
        "preprocessor": preprocessor,
        "model_params": {
            "category_cardinalities": preprocessor.category_cardinalities,
            "n_numeric": len(NUMERIC_FEATURES),
            "hidden_dims": cfg.hidden_dims,
            "n_classes": n_classes,
            "dropout": cfg.dropout,
            "embedding_dropout": cfg.embedding_dropout,
        },
        "label_names": label_names,
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
    }
    joblib.dump(bundle, cfg.output_dir / "buyer_profile_neural_net_bundle.joblib")

    # 3. Metrics
    save_json(cfg.output_dir / "metrics.json", {
        "validation": val_metrics,
        "test": test_metrics,
    })

    # 4. CV results
    save_json(cfg.output_dir / "cv_results.json", cv_results)

    # 5. Test predictions
    pred_labels_test = preprocessor.label_encoder.inverse_transform(y_pred_test)
    true_labels_test = preprocessor.label_encoder.inverse_transform(y_true_test)
    test_pred_df = pd.DataFrame({
        "true_label": true_labels_test,
        "predicted_label": pred_labels_test,
    })
    test_pred_df.to_csv(cfg.output_dir / "test_predictions.csv", index=False, encoding="utf-8")

    # 6. Validation predictions
    pred_labels_val = preprocessor.label_encoder.inverse_transform(y_pred_val)
    true_labels_val = preprocessor.label_encoder.inverse_transform(y_true_val)
    val_pred_df = pd.DataFrame({
        "true_label": true_labels_val,
        "predicted_label": pred_labels_val,
    })
    val_pred_df.to_csv(cfg.output_dir / "validation_predictions.csv", index=False, encoding="utf-8")

    # 7. Training loss history
    save_json(cfg.output_dir / "training_loss_history.json", {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "stopped_epoch": epoch,
    })

    # 8. Training metadata
    metadata = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape),
        "n_features": len(FEATURES),
        "n_categorical": len(CATEGORICAL_FEATURES),
        "n_numeric": len(NUMERIC_FEATURES),
        "n_classes": n_classes,
        "label_names": label_names,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "feature_list": FEATURES,
    }
    save_json(cfg.output_dir / "training_metadata.json", metadata)

    logger.info("All artefacts saved to %s", cfg.output_dir)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train neural-net buyer-profile classifier")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()

    cfg = NeuralNetConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        dropout=args.dropout,
    )
    train(cfg)
