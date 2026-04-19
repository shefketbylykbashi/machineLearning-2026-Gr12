"""
Deep autoencoder with entity embeddings for unsupervised representation
learning on property-buyer data.

This is an **unsupervised** neural network: it never sees the buyer_profile
label during training. Instead it learns to compress each row into a small
latent vector (8 dims by default) and reconstruct the original features from
that vector. Rows that are semantically similar end up close together in the
latent space.

After training we perform two post-hoc evaluations (labels used only for
scoring, never for training):
    1. KMeans clustering on the latent space - do natural clusters align
       with the known buyer profiles?
    2. k-NN classification in the latent space - can a simple k-NN vote in
       latent space predict buyer_profile?

Outputs are saved to data/models/autoencoder/.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    calinski_harabasz_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Reuse the feature lists from the supervised module so both models see
# exactly the same columns.
from src.ml.neural_net.train_neural_net import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    embedding_dim_rule,
)

# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class AutoencoderConfig:
    project_root: Path = Path(__file__).resolve().parents[3]
    data_dir: Path = project_root / "data" / "processed" / "property_buyer"
    output_root: Path = project_root / "data" / "models"

    train_filename: str = "train_dataset_v1.0.xlsx"
    test_filename: str = "test_dataset_v1.0.xlsx"

    target_column: str = "buyer_profile"  # used only for evaluation

    # Architecture
    hidden_dims: tuple[int, ...] = (128, 64)
    latent_dim: int = 8
    dropout: float = 0.1
    embedding_dropout: float = 0.1

    # Training
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 25  # early stopping on validation reconstruction loss
    validation_size: float = 0.15

    # Post-hoc evaluation
    kmeans_k: int = 8     # match the number of buyer profiles
    knn_k: int = 5

    # Cross-validation on the kNN-in-latent-space pipeline (labels used
    # only at scoring time, never to train the autoencoder).
    n_splits_cv: int = 3
    cv_epochs: int = 80       # reduced epochs to keep CV tractable
    cv_patience: int = 15

    random_state: int = 42

    fill_missing_categorical_with: str = "__missing__"

    @property
    def output_dir(self) -> Path:
        return self.output_root / "autoencoder"


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
# Preprocessing (same shape as supervised preprocessor, but no class labels
# during fit - only to allow later evaluation).
# ============================================================

class FeaturePreprocessor:
    def __init__(self, fill_missing: str = "__missing__"):
        self.fill_missing = fill_missing
        self.ordinal_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1,
            encoded_missing_value=-1,
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()  # fitted only for evaluation
        self.category_cardinalities: list[int] = []

    def fit(self, df: pd.DataFrame, target_col: str | None = None) -> None:
        X_cat = self._prep_cat(df)
        X_num = self._prep_num(df)
        self.ordinal_encoder.fit(X_cat)
        self.scaler.fit(X_num)
        self.category_cardinalities = [
            len(cats) + 1 for cats in self.ordinal_encoder.categories_
        ]
        if target_col is not None and target_col in df.columns:
            self.label_encoder.fit(df[target_col].astype(str))

    def transform(
        self, df: pd.DataFrame, target_col: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        X_cat = self._prep_cat(df)
        X_num = self._prep_num(df)

        cat_encoded = self.ordinal_encoder.transform(X_cat).astype(np.int64)
        for i, n in enumerate(self.category_cardinalities):
            cat_encoded[:, i] = np.where(cat_encoded[:, i] < 0, n - 1, cat_encoded[:, i])

        num_scaled = self.scaler.transform(X_num).astype(np.float32)

        y = None
        if target_col is not None and target_col in df.columns:
            y = self.label_encoder.transform(df[target_col].astype(str))
        return cat_encoded, num_scaled, y

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
# Autoencoder model
# ============================================================

class AutoEncoder(nn.Module):
    """Symmetric autoencoder with entity embeddings on the input side.

    Encoder:
        embeddings(cat) + numeric  ->  MLP  ->  latent z

    Decoder:
        z  ->  MLP  ->  reconstructed [numeric_hat, cat_logits_0, cat_logits_1, ...]

    Losses:
        - MSE on the numeric block
        - Cross-entropy per categorical feature (multinomial reconstruction)
    """

    def __init__(
        self,
        category_cardinalities: list[int],
        n_numeric: int,
        hidden_dims: tuple[int, ...],
        latent_dim: int,
        dropout: float,
        embedding_dropout: float,
    ):
        super().__init__()

        self.category_cardinalities = list(category_cardinalities)
        self.n_numeric = n_numeric

        self.embeddings = nn.ModuleList([
            nn.Embedding(n_cat, embedding_dim_rule(n_cat))
            for n_cat in category_cardinalities
        ])
        self.emb_dropout = nn.Dropout(embedding_dropout)
        total_emb_dim = sum(embedding_dim_rule(n) for n in category_cardinalities)

        encoder_layers: list[nn.Module] = []
        prev = total_emb_dim + n_numeric
        for h in hidden_dims:
            encoder_layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = h
        encoder_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder mirrors encoder
        decoder_layers: list[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = h
        self.decoder_shared = nn.Sequential(*decoder_layers)

        # Two reconstruction heads
        self.numeric_head = nn.Linear(prev, n_numeric)
        self.cat_heads = nn.ModuleList([
            nn.Linear(prev, n_cat) for n_cat in category_cardinalities
        ])

    # ---- encode only (used at eval time) ----
    def encode(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        emb = [layer(x_cat[:, i]) for i, layer in enumerate(self.embeddings)]
        h = torch.cat(emb + [x_num], dim=1)
        h = self.emb_dropout(h)
        return self.encoder(h)

    def forward(
        self, x_cat: torch.Tensor, x_num: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        z = self.encode(x_cat, x_num)
        h = self.decoder_shared(z)
        num_hat = self.numeric_head(h)
        cat_logits = [head(h) for head in self.cat_heads]
        return z, num_hat, cat_logits


# ============================================================
# Training
# ============================================================

def reconstruction_loss(
    num_hat: torch.Tensor,
    cat_logits: list[torch.Tensor],
    x_num: torch.Tensor,
    x_cat: torch.Tensor,
    cat_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mse = nn.functional.mse_loss(num_hat, x_num)
    ce_total = torch.zeros((), device=x_num.device)
    for i, logits in enumerate(cat_logits):
        ce_total = ce_total + nn.functional.cross_entropy(logits, x_cat[:, i])
    ce_total = ce_total / len(cat_logits)
    total = mse + cat_weight * ce_total
    return total, mse.detach(), ce_total.detach()


def train_autoencoder(
    model: AutoEncoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: AutoencoderConfig,
    device: torch.device,
) -> dict[str, list[float]]:
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [],
        "train_mse": [], "val_mse": [],
        "train_ce": [], "val_ce": [],
    }

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        t_loss = t_mse = t_ce = 0.0
        n_batches = 0
        for x_cat, x_num in train_loader:
            x_cat = x_cat.to(device)
            x_num = x_num.to(device)
            optimizer.zero_grad()
            _, num_hat, cat_logits = model(x_cat, x_num)
            loss, mse, ce = reconstruction_loss(num_hat, cat_logits, x_num, x_cat)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            t_mse += mse.item()
            t_ce += ce.item()
            n_batches += 1

        model.eval()
        v_loss = v_mse = v_ce = 0.0
        v_batches = 0
        with torch.no_grad():
            for x_cat, x_num in val_loader:
                x_cat = x_cat.to(device)
                x_num = x_num.to(device)
                _, num_hat, cat_logits = model(x_cat, x_num)
                loss, mse, ce = reconstruction_loss(num_hat, cat_logits, x_num, x_cat)
                v_loss += loss.item()
                v_mse += mse.item()
                v_ce += ce.item()
                v_batches += 1

        tl = t_loss / max(n_batches, 1)
        vl = v_loss / max(v_batches, 1)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_mse"].append(t_mse / max(n_batches, 1))
        history["val_mse"].append(v_mse / max(v_batches, 1))
        history["train_ce"].append(t_ce / max(n_batches, 1))
        history["val_ce"].append(v_ce / max(v_batches, 1))

        if vl < best_val - 1e-5:
            best_val = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "Epoch %3d | train_loss=%.4f val_loss=%.4f | mse=%.4f ce=%.4f | patience=%d",
                epoch, tl, vl, history["val_mse"][-1], history["val_ce"][-1], patience_counter,
            )

        if patience_counter >= config.patience:
            logger.info("Early stopping at epoch %d (best val_loss=%.4f)", epoch, best_val)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


# ============================================================
# Evaluation helpers
# ============================================================

def encode_dataset(
    model: AutoEncoder,
    x_cat: np.ndarray,
    x_num: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    latents: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(x_num), batch_size):
            c = torch.from_numpy(x_cat[i:i + batch_size]).to(device)
            n = torch.from_numpy(x_num[i:i + batch_size]).to(device)
            z = model.encode(c, n)
            latents.append(z.cpu().numpy())
    return np.concatenate(latents, axis=0)


def evaluate_clustering(
    z: np.ndarray, y: np.ndarray, k: int, random_state: int,
) -> dict[str, Any]:
    km = KMeans(n_clusters=k, n_init=20, max_iter=500, random_state=random_state)
    clusters = km.fit_predict(z)

    metrics = {
        "silhouette": float(silhouette_score(z, clusters)),
        "calinski_harabasz": float(calinski_harabasz_score(z, clusters)),
        "davies_bouldin": float(davies_bouldin_score(z, clusters)),
        "inertia": float(km.inertia_),
        "ari_vs_buyer_profile": float(adjusted_rand_score(y, clusters)),
        "nmi_vs_buyer_profile": float(normalized_mutual_info_score(y, clusters)),
    }
    return {"metrics": metrics, "cluster_labels": clusters.tolist()}


def evaluate_knn_in_latent(
    z_train: np.ndarray, y_train: np.ndarray,
    z_eval: np.ndarray, y_eval: np.ndarray,
    k: int,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """k-NN vote in the latent space. Returns metrics in the same schema
    the supervised models use (accuracy, balanced_accuracy, macro_f1,
    weighted_f1, classification_report, confusion_matrix).

    Labels are only used at evaluation time; the autoencoder itself has
    never seen them.
    """
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn.fit(z_train, y_train)
    y_pred = knn.predict(z_eval)

    if class_names is not None:
        labels = list(range(len(class_names)))
        report = classification_report(
            y_eval, y_pred, labels=labels, target_names=class_names,
            output_dict=True, zero_division=0,
        )
        cm = confusion_matrix(y_eval, y_pred, labels=labels).tolist()
    else:
        report = classification_report(
            y_eval, y_pred, output_dict=True, zero_division=0,
        )
        cm = confusion_matrix(y_eval, y_pred).tolist()

    return {
        "accuracy": float(accuracy_score(y_eval, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_eval, y_pred)),
        "macro_f1": float(f1_score(y_eval, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_eval, y_pred, average="weighted")),
        "classification_report": report,
        "confusion_matrix": cm,
        "predictions": y_pred.tolist(),
    }


def _train_encoder_only(
    df_train: pd.DataFrame,
    config: AutoencoderConfig,
    device: torch.device,
    epochs: int,
    patience: int,
    log_prefix: str = "",
) -> tuple[AutoEncoder, "FeaturePreprocessor"]:
    """Fit preprocessor + autoencoder on the given frame (no labels used)
    and return both. Used by the CV loop."""
    pre = FeaturePreprocessor(fill_missing=config.fill_missing_categorical_with)
    pre.fit(df_train, target_col=config.target_column)
    xc, xn, _ = pre.transform(df_train, target_col=config.target_column)

    # carve a small val slice from this training subset for early stopping
    n = len(xc)
    val_n = max(16, int(n * config.validation_size))
    perm = np.random.default_rng(config.random_state).permutation(n)
    val_idx, tr_idx = perm[:val_n], perm[val_n:]

    tr_ds = TensorDataset(torch.from_numpy(xc[tr_idx]), torch.from_numpy(xn[tr_idx]))
    va_ds = TensorDataset(torch.from_numpy(xc[val_idx]), torch.from_numpy(xn[val_idx]))
    tr_loader = DataLoader(tr_ds, batch_size=config.batch_size, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=config.batch_size, shuffle=False)

    model = AutoEncoder(
        category_cardinalities=pre.category_cardinalities,
        n_numeric=len(NUMERIC_FEATURES),
        hidden_dims=config.hidden_dims,
        latent_dim=config.latent_dim,
        dropout=config.dropout,
        embedding_dropout=config.embedding_dropout,
    ).to(device)

    # Ephemeral config for the CV sub-training (shorter epochs/patience).
    sub_cfg = AutoencoderConfig(
        epochs=epochs, patience=patience,
        random_state=config.random_state,
    )
    logger.info("%sFold training: %d rows (%d train / %d val)",
                log_prefix, n, len(tr_idx), len(val_idx))
    train_autoencoder(model, tr_loader, va_loader, sub_cfg, device)
    return model, pre


def cross_validate_latent_knn(
    df_train_full: pd.DataFrame,
    config: AutoencoderConfig,
    device: torch.device,
    class_names: list[str],
) -> dict[str, Any]:
    """Run stratified k-fold CV: for each fold, (1) train an autoencoder on
    the fold's training partition (unsupervised), (2) encode both partitions,
    (3) kNN-classify the held-out partition. Labels are only used at step 3.
    Returns mean/std/min/max across folds for accuracy/balanced/macro_f1/weighted_f1.
    """
    y_full = df_train_full[config.target_column].astype(str).values
    skf = StratifiedKFold(
        n_splits=config.n_splits_cv, shuffle=True, random_state=config.random_state,
    )
    fold_results: list[dict[str, float]] = []
    for fold_idx, (tr_i, va_i) in enumerate(skf.split(np.zeros(len(y_full)), y_full), start=1):
        logger.info("[CV fold %d/%d] training autoencoder on %d rows",
                    fold_idx, config.n_splits_cv, len(tr_i))
        df_fold_train = df_train_full.iloc[tr_i].reset_index(drop=True)
        df_fold_val = df_train_full.iloc[va_i].reset_index(drop=True)

        model, pre = _train_encoder_only(
            df_fold_train, config, device,
            epochs=config.cv_epochs, patience=config.cv_patience,
            log_prefix=f"[CV fold {fold_idx}] ",
        )

        xc_tr, xn_tr, y_tr = pre.transform(df_fold_train, target_col=config.target_column)
        xc_va, xn_va, y_va = pre.transform(df_fold_val, target_col=config.target_column)
        z_tr = encode_dataset(model, xc_tr, xn_tr, device)
        z_va = encode_dataset(model, xc_va, xn_va, device)

        knn = KNeighborsClassifier(n_neighbors=config.knn_k, weights="distance")
        knn.fit(z_tr, y_tr)
        y_pred = knn.predict(z_va)
        fr = {
            "fold": fold_idx,
            "accuracy": float(accuracy_score(y_va, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_va, y_pred)),
            "macro_f1": float(f1_score(y_va, y_pred, average="macro")),
            "weighted_f1": float(f1_score(y_va, y_pred, average="weighted")),
        }
        logger.info("[CV fold %d] acc=%.4f bal=%.4f macro_f1=%.4f",
                    fold_idx, fr["accuracy"], fr["balanced_accuracy"], fr["macro_f1"])
        fold_results.append(fr)

    # aggregate
    def agg(key: str) -> dict[str, float]:
        vals = np.array([f[key] for f in fold_results], dtype=float)
        return {
            "mean": float(vals.mean()), "std": float(vals.std()),
            "min": float(vals.min()), "max": float(vals.max()),
        }
    summary = {k: agg(k) for k in
               ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")}
    return {"folds": fold_results, "summary": summary}


# ============================================================
# Main training entry point
# ============================================================

def train(config: AutoencoderConfig) -> None:
    setup_logging()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    torch.manual_seed(config.random_state)
    np.random.seed(config.random_state)

    train_path = config.data_dir / config.train_filename
    test_path = config.data_dir / config.test_filename
    logger.info("Loading data from %s and %s", train_path, test_path)
    df_train_full = pd.read_excel(train_path)
    df_test = pd.read_excel(test_path)
    logger.info("Train rows: %d | Test rows: %d", len(df_train_full), len(df_test))

    # Split a validation slice for early-stopping (purely on reconstruction)
    val_n = max(1, int(len(df_train_full) * config.validation_size))
    shuffled = df_train_full.sample(frac=1.0, random_state=config.random_state).reset_index(drop=True)
    df_val = shuffled.iloc[:val_n].reset_index(drop=True)
    df_train = shuffled.iloc[val_n:].reset_index(drop=True)
    logger.info("Train/val split: %d / %d", len(df_train), len(df_val))

    # Fit preprocessor on training-only slice; labels kept for later eval
    pre = FeaturePreprocessor(fill_missing=config.fill_missing_categorical_with)
    pre.fit(df_train, target_col=config.target_column)

    xc_tr, xn_tr, y_tr = pre.transform(df_train, target_col=config.target_column)
    xc_va, xn_va, y_va = pre.transform(df_val, target_col=config.target_column)
    xc_te, xn_te, y_te = pre.transform(df_test, target_col=config.target_column)

    logger.info(
        "Feature shapes | cat=%s num=%s | cardinalities=%s",
        xc_tr.shape, xn_tr.shape, pre.category_cardinalities,
    )

    # DataLoaders (no labels used during training)
    train_ds = TensorDataset(torch.from_numpy(xc_tr), torch.from_numpy(xn_tr))
    val_ds = TensorDataset(torch.from_numpy(xc_va), torch.from_numpy(xn_va))
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    # Model
    model = AutoEncoder(
        category_cardinalities=pre.category_cardinalities,
        n_numeric=len(NUMERIC_FEATURES),
        hidden_dims=config.hidden_dims,
        latent_dim=config.latent_dim,
        dropout=config.dropout,
        embedding_dropout=config.embedding_dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Autoencoder parameters: %d", n_params)

    # Train
    history = train_autoencoder(model, train_loader, val_loader, config, device)

    # Encode every split in the latent space
    z_train = encode_dataset(model, xc_tr, xn_tr, device)
    z_val = encode_dataset(model, xc_va, xn_va, device)
    z_test = encode_dataset(model, xc_te, xn_te, device)

    # --- Evaluation 1: KMeans clustering on latent space (train set) ---
    logger.info("Evaluating KMeans on latent space (k=%d)", config.kmeans_k)
    cluster_eval = evaluate_clustering(
        z_train, y_tr, k=config.kmeans_k, random_state=config.random_state,
    )
    logger.info("Clustering metrics: %s", cluster_eval["metrics"])

    # --- Evaluation 2: k-NN in latent space on validation and test sets ---
    class_names = list(pre.label_encoder.classes_)
    logger.info("Evaluating k-NN in latent space on validation set (k=%d)", config.knn_k)
    val_eval = evaluate_knn_in_latent(
        z_train, y_tr, z_val, y_va, k=config.knn_k, class_names=class_names,
    )
    logger.info(
        "Validation latent k-NN  acc=%.4f  balanced_acc=%.4f  macro_f1=%.4f",
        val_eval["accuracy"], val_eval["balanced_accuracy"], val_eval["macro_f1"],
    )

    logger.info("Evaluating k-NN in latent space on test set (k=%d)", config.knn_k)
    test_eval = evaluate_knn_in_latent(
        z_train, y_tr, z_test, y_te, k=config.knn_k, class_names=class_names,
    )
    logger.info(
        "Test latent k-NN  acc=%.4f  balanced_acc=%.4f  macro_f1=%.4f",
        test_eval["accuracy"], test_eval["balanced_accuracy"], test_eval["macro_f1"],
    )

    # --- Evaluation 3: stratified k-fold CV on the full unsupervised pipeline ---
    logger.info("Running %d-fold CV on the autoencoder+kNN pipeline "
                "(this retrains the autoencoder once per fold)",
                config.n_splits_cv)
    cv = cross_validate_latent_knn(df_train_full, config, device, class_names)
    logger.info("CV summary: %s", cv["summary"])

    # --- Save artifacts ---
    def _clean(d: dict[str, Any]) -> dict[str, Any]:
        """Drop the per-row predictions before writing the supervised-schema blocks."""
        return {k: v for k, v in d.items() if k != "predictions"}

    metrics_payload: dict[str, Any] = {
        "model": "autoencoder_knn",
        "config": {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in asdict(config).items()},
        "training": {
            "n_parameters": int(n_params),
            "epochs_trained": len(history["val_loss"]),
            "best_val_loss": float(min(history["val_loss"])),
            "best_epoch": int(np.argmin(history["val_loss"]) + 1),
            "history": history,
        },
        "validation": _clean(val_eval),
        "test": _clean(test_eval),
        "cv_summary": cv["summary"],
        "clustering_on_latent": cluster_eval["metrics"],
        "class_names": class_names,
    }
    metrics_path = config.output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)
    logger.info("Wrote metrics to %s", metrics_path)

    # Sibling cv_results.json so the comparison viz can pick it up like the
    # supervised models.
    cv_path = config.output_dir / "cv_results.json"
    with cv_path.open("w", encoding="utf-8") as f:
        json.dump({"folds": cv["folds"], "summary": cv["summary"]},
                  f, ensure_ascii=False, indent=2)
    logger.info("Wrote CV results to %s", cv_path)

    # Save the latent representations - useful for downstream visualization
    np.savez_compressed(
        config.output_dir / "latent_representations.npz",
        z_train=z_train, y_train=y_tr,
        z_val=z_val,     y_val=y_va,
        z_test=z_test,   y_test=y_te,
        kmeans_train_clusters=np.asarray(cluster_eval["cluster_labels"], dtype=np.int32),
    )

    # Save the torch weights + preprocessor bundle
    torch.save(model.state_dict(), config.output_dir / "autoencoder_weights.pt")
    joblib.dump(
        {"preprocessor": pre,
         "category_cardinalities": pre.category_cardinalities,
         "latent_dim": config.latent_dim,
         "hidden_dims": config.hidden_dims,
         "numeric_features": NUMERIC_FEATURES,
         "categorical_features": CATEGORICAL_FEATURES,
         "class_names": list(pre.label_encoder.classes_)},
        config.output_dir / "autoencoder_bundle.joblib",
    )

    logger.info("Done. Artifacts written to %s", config.output_dir)


if __name__ == "__main__":
    train(AutoencoderConfig())
