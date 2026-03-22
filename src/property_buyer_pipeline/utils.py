"""
Utility functions for data cleaning and validation.
"""
import re
import numpy as np
import pandas as pd


# =========================================================
# Text Normalization
# =========================================================
def normalize_column_name(name: str) -> str:
    """Normalize column names by removing special characters and standardizing format."""
    name = str(name).strip()
    replacements = {
        "ë": "e", "Ë": "E", "ç": "c", "Ç": "C",
        "'": "", "'": "", '"': "", "(": " ", ")": " ",
        "[": " ", "]": " ", "/": " ", "-": " ", ":": " ",
        ",": " ", ".": " ", "%": " perqind ", "&": " dhe ",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^a-zA-Z0-9_]", "", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name.lower()


def clean_numeric_series(series: pd.Series) -> pd.Series:
    """Clean numeric series by removing currency symbols and converting to numeric."""
    s = series.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "<NA>": np.nan})
    s = s.str.replace("€", "", regex=False)
    s = s.str.replace("%", "", regex=False)
    s = s.str.replace(",", "", regex=False)
    s = s.str.replace(r"[^0-9.\-]", "", regex=True)
    s = s.replace({"": np.nan})
    return pd.to_numeric(s, errors="coerce")


def try_parse_datetime(series: pd.Series) -> pd.Series:
    """Attempt to parse series as datetime."""
    return pd.to_datetime(series, errors="coerce")


def mode_or_default(series: pd.Series, default):
    """Get mode of series or return default if empty."""
    non_null = series.dropna()
    if non_null.empty:
        return default
    mode = non_null.mode(dropna=True)
    if mode.empty:
        return default
    return mode.iloc[0]


def normalize_text_for_rules(series: pd.Series) -> pd.Series:
    """Normalize text for rule-based matching."""
    return (
        series.astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


# =========================================================
# Asset Classification
# =========================================================
def derive_asset_text(df: pd.DataFrame) -> pd.Series:
    """Combine asset-related columns into single text for classification."""
    candidates = [
        "kategoria_e_asetit",
        "kategoria",
        "lloji_i_prones",
        "pershkrimi",
        "pronesia",
        "destinimi",
        "tipi_i_asetit",
        "emri_i_ndermarrjes_se_re_apo_asetit_ne_likuidim",
    ]
    existing = [c for c in candidates if c in df.columns]
    if not existing:
        return pd.Series("", index=df.index)
    combined = df[existing].fillna("").astype(str).agg(" ".join, axis=1)
    return normalize_text_for_rules(combined)


def classify_asset_masks(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Classify assets into land-only, object-only, or mixed."""
    text = derive_asset_text(df)

    land_tokens = r"\btoke\b|tok[eë]\b|tok[eë]\s+bujq[eë]sore|truall|parcel|parcel[eë]"
    object_tokens = r"ndert|objekt|banes|lokal|shtepi|depo|fabrik|hotel|ndertese|zyr|magazin|qender|mulli|warehouse|administrative"

    explicit_mixed_tokens = (
        r"dhe\s+toke|dhe\s+tok[eë]|me\s+toke|me\s+tok[eë]|"
        r"dhe\s+truall|me\s+truall|depo\s+dhe\s+toke|"
        r"ndert(es|e)[a-z]*\s+administrative\s+dhe\s+toke"
    )

    has_land = text.str.contains(land_tokens, regex=True, na=False)
    has_object = text.str.contains(object_tokens, regex=True, na=False)
    explicit_mixed = text.str.contains(explicit_mixed_tokens, regex=True, na=False)

    land_only = has_land & ~has_object & ~explicit_mixed
    object_only = has_object & ~has_land & ~explicit_mixed
    mixed = explicit_mixed | (has_land & has_object)

    return land_only, object_only, mixed


# =========================================================
# Group-Based Imputation
# =========================================================
def safe_group_median_fill(
    df: pd.DataFrame,
    col: str,
    group_cols: list[str],
    min_group_size: int,
    allowed_mask: pd.Series | None = None
) -> tuple[pd.DataFrame, int]:
    """Fill missing values using group median with minimum group size constraints."""
    usable_groups = [g for g in group_cols if g in df.columns]
    if col not in df.columns or not usable_groups:
        return df, 0

    target_mask = df[col].isna()
    if allowed_mask is not None:
        target_mask = target_mask & allowed_mask.fillna(False)

    if int(target_mask.sum()) == 0:
        return df, 0

    stats = (
        df.groupby(usable_groups, dropna=False)[col]
        .agg(["median", "count"])
        .reset_index()
    )
    stats = stats[stats["count"] >= min_group_size]
    if stats.empty:
        return df, 0

    stats = stats.rename(columns={"median": f"{col}__group_median", "count": f"{col}__group_count"})
    df = df.merge(stats, on=usable_groups, how="left")

    fill_mask = target_mask & df[f"{col}__group_median"].notna()
    filled = int(fill_mask.sum())
    df.loc[fill_mask, col] = df.loc[fill_mask, f"{col}__group_median"]

    df = df.drop(columns=[f"{col}__group_median", f"{col}__group_count"], errors="ignore")
    return df, filled


# =========================================================
# Outlier Detection
# =========================================================
def identify_outliers_iqr(
    df: pd.DataFrame,
    numeric_cols: list[str],
    exclude_cols: set[str] | None = None
) -> tuple[pd.DataFrame, dict]:
    """Identify outliers using IQR method without modifying values."""
    stats = {}
    exclude_cols = exclude_cols or set()

    for col in numeric_cols:
        if col in exclude_cols:
            continue

        s = df[col].dropna()
        if s.nunique() < 5:
            continue

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        non_negative_signals = [
            "cmimi", "kapital", "siperfaq", "area", "price", "punetoreve",
            "mosha", "days", "years", "ratio", "m2", "total"
        ]
        if any(token in col for token in non_negative_signals):
            lower = max(lower, 0)

        mask = (df[col] < lower) | (df[col] > upper)
        outlier_count = int(mask.sum())
        outlier_ratio = float(outlier_count / len(df)) if len(df) else 0.0

        if outlier_count > 0:
            df[f"{col}_is_outlier_iqr"] = mask.astype(int)

        stats[col] = {
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "lower": float(lower),
            "upper": float(upper),
            "outlier_count": outlier_count,
            "outlier_ratio": outlier_ratio,
        }

    return df, stats
