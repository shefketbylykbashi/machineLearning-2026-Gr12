"""
Transform Layer - Data transformation and feature engineering.
"""
import numpy as np
import pandas as pd
from .config import (
    TARGET_COL, MANDATORY_BUSINESS_COL, LEAKAGE_COLUMNS, NUMERIC_LIKE_COLUMNS,
    DATE_COLUMNS, AREA_COLUMNS, PRICE_COLUMN, SPARSE_COL_THRESHOLD,
    DOMINANT_VALUE_THRESHOLD, PROTECTED_COLUMNS, CATEGORICAL_DEFAULTS,
    ZERO_FILL_COLUMNS, AREA_GROUP_MIN_SIZE, GENERIC_GROUP_MIN_SIZE, MIN_TARGET_FREQUENCY
)
from .utils import (
    clean_numeric_series, try_parse_datetime, classify_asset_masks, mode_or_default,
    safe_group_median_fill, identify_outliers_iqr
)


class DataValidator:
    """Validates data quality and business rules."""
    
    @staticmethod
    def validate_required_columns(df: pd.DataFrame) -> None:
        """Check for mandatory business and target columns."""
        if MANDATORY_BUSINESS_COL not in df.columns:
            raise KeyError(f"Column '{MANDATORY_BUSINESS_COL}' not found after normalization.")
        if TARGET_COL not in df.columns:
            raise KeyError(f"Column '{TARGET_COL}' not found after normalization.")
    
    @staticmethod
    def remove_null_business_types(df: pd.DataFrame, report: dict) -> pd.DataFrame:
        """Remove rows with null business type."""
        before = len(df)
        df = df[df[MANDATORY_BUSINESS_COL].notna()].copy()
        report["rows_after_remove_null_business_type"] = int(len(df))
        print(f"Removed {before - len(df)} rows with null business type")
        return df
    
    @staticmethod
    def remove_null_targets(df: pd.DataFrame, report: dict) -> pd.DataFrame:
        """Remove rows with null target variable."""
        before = len(df)
        df = df[df[TARGET_COL].notna()].copy()
        report["rows_after_remove_null_target"] = int(len(df))
        print(f"Removed {before - len(df)} rows with null target")
        return df


class DuplicateHandler:
    """Handles duplicate records."""
    
    @staticmethod
    def remove_full_duplicates(df: pd.DataFrame, report: dict) -> pd.DataFrame:
        """Remove exact row duplicates."""
        before = len(df)
        df = df.drop_duplicates().copy()
        duplicates_removed = int(before - len(df))
        report["duplicates_removed_full_row"] = duplicates_removed
        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} full-row duplicates")
        return df
    
    @staticmethod
    def remove_business_key_duplicates(df: pd.DataFrame, report: dict) -> pd.DataFrame:
        """Remove duplicates based on business key."""
        duplicate_key_candidates = [
            "pak_id",
            "emri_i_ndermarrjes_se_re_apo_asetit_ne_likuidim",
            "komuna_lokacioni_i_asetit_te_shitur",
            PRICE_COLUMN,
            "data_e_kontrates",
            TARGET_COL,
        ]
        duplicate_keys = [c for c in duplicate_key_candidates if c in df.columns]
        
        if not duplicate_keys:
            report["duplicates_removed_business_key"] = 0
            return df
        
        before = len(df)
        df = df.drop_duplicates(subset=duplicate_keys, keep="first").copy()
        duplicates_removed = int(before - len(df))
        report["duplicates_removed_business_key"] = duplicates_removed
        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} business-key duplicates")
        return df


class TypeConverter:
    """Converts data types."""
    
    @staticmethod
    def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Convert numeric-like columns to numeric type."""
        numeric_cols = [c for c in NUMERIC_LIKE_COLUMNS if c in df.columns]
        for col in numeric_cols:
            df[col] = clean_numeric_series(df[col])
        print(f"Converted {len(numeric_cols)} numeric columns")
        return df
    
    @staticmethod
    def convert_date_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Convert date columns to datetime type."""
        date_cols = [c for c in DATE_COLUMNS if c in df.columns]
        for col in date_cols:
            df[col] = try_parse_datetime(df[col])
        print(f"Converted {len(date_cols)} date columns")
        return df


class AreaInferencer:
    """Infers and fills area fields based on asset type."""
    
    @staticmethod
    def infer_areas(df: pd.DataFrame, report: dict) -> pd.DataFrame:
        """Infer missing area values based on business rules."""
        obj_col = AREA_COLUMNS["object"]
        land_col = AREA_COLUMNS["land"]
        
        if obj_col not in df.columns or land_col not in df.columns:
            return df
        
        # Create flags before filling
        df[f"{obj_col}_was_missing"] = df[obj_col].isna().astype(int)
        df[f"{land_col}_was_missing"] = df[land_col].isna().astype(int)
        df["raw_has_object_area"] = df[obj_col].notna().astype(int)
        df["raw_has_land_area"] = df[land_col].notna().astype(int)
        
        land_only, object_only, mixed = classify_asset_masks(df)
        
        report.setdefault("area_inference", {})
        report["area_inference"]["rules"] = {}
        
        # Apply business rules
        mask = land_only & df[obj_col].isna()
        report["area_inference"]["rules"]["land_only_object_area_to_zero"] = int(mask.sum())
        df.loc[mask, obj_col] = 0
        
        mask = object_only & df[land_col].isna()
        report["area_inference"]["rules"]["object_only_land_area_to_zero"] = int(mask.sum())
        df.loc[mask, land_col] = 0
        
        # Group fill for mixed assets only
        group_sets = [
            ["kategoria_e_asetit", "komuna_lokacioni_i_asetit_te_shitur", MANDATORY_BUSINESS_COL],
            ["kategoria_e_asetit", "komuna_lokacioni_i_asetit_te_shitur"],
            ["komuna_lokacioni_i_asetit_te_shitur", MANDATORY_BUSINESS_COL],
        ]
        
        obj_filled = 0
        land_filled = 0
        for groups in group_sets:
            df, n1 = safe_group_median_fill(df, obj_col, groups, AREA_GROUP_MIN_SIZE, allowed_mask=mixed)
            obj_filled += n1
            df, n2 = safe_group_median_fill(df, land_col, groups, AREA_GROUP_MIN_SIZE, allowed_mask=mixed)
            land_filled += n2
        
        report["area_inference"]["rules"]["object_area_group_fill_mixed_only"] = obj_filled
        report["area_inference"]["rules"]["land_area_group_fill_mixed_only"] = land_filled
        report["area_inference"]["rules"]["mixed_asset_rows_detected"] = int(mixed.sum())
        
        # Final fallback: zero fill
        remaining_obj = int(df[obj_col].isna().sum())
        remaining_land = int(df[land_col].isna().sum())
        df[obj_col] = df[obj_col].fillna(0)
        df[land_col] = df[land_col].fillna(0)
        
        report["area_inference"]["rules"]["remaining_object_area_filled_with_zero"] = remaining_obj
        report["area_inference"]["rules"]["remaining_land_area_filled_with_zero"] = remaining_land
        
        print(f"Area inference completed: {obj_filled + land_filled} values filled")
        return df


class MissingValueHandler:
    """Handles missing values with business-aware strategies."""
    
    @staticmethod
    def fill_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Fill missing values using hierarchical strategies."""
        fill_report = {"rule_based_fills": {}, "group_fills": {}, "fallback_fills": {}}
        
        # Create missing flags
        important_flags = [
            "arbk_kapitali", "arbk_numpunetoreve", PRICE_COLUMN,
            AREA_COLUMNS["object"], AREA_COLUMNS["land"],
            "arbk_dataregjistrimit", "data_e_kontrates",
        ]
        for col in important_flags:
            if col in df.columns and f"{col}_was_missing" not in df.columns:
                df[f"{col}_was_missing"] = df[col].isna().astype(int)
        
        # Zero-fill for safe numeric columns
        for col in ZERO_FILL_COLUMNS:
            if col in df.columns and df[col].isna().sum() > 0:
                n = int(df[col].isna().sum())
                df[col] = df[col].fillna(0)
                fill_report["rule_based_fills"][col] = {"strategy": "fill_zero", "filled": n}
        
        # Activity text fields
        for col in df.columns:
            if "aktiviteti" in col and not pd.api.types.is_numeric_dtype(df[col]):
                n = int(df[col].isna().sum())
                if n > 0:
                    df[col] = df[col].fillna("Aktivitete tjera")
                    fill_report["rule_based_fills"][col] = {
                        "strategy": "fill_aktivitete_tjera",
                        "filled": n,
                    }
        
        # Categorical defaults
        for col, default in CATEGORICAL_DEFAULTS.items():
            if col in df.columns and df[col].isna().sum() > 0:
                n = int(df[col].isna().sum())
                df[col] = df[col].fillna(default)
                fill_report["rule_based_fills"][col] = {
                    "strategy": f"fill_{default}",
                    "filled": n,
                }
        
        # Date fields
        if "data_e_kontrates" in df.columns and df["data_e_kontrates"].isna().sum() > 0:
            n = int(df["data_e_kontrates"].isna().sum())
            fill_value = df["data_e_kontrates"].dropna().median()
            if pd.isna(fill_value):
                fill_value = pd.Timestamp("2000-01-01")
            df["data_e_kontrates"] = df["data_e_kontrates"].fillna(fill_value)
            fill_report["rule_based_fills"]["data_e_kontrates"] = {
                "strategy": "fill_median_date",
                "filled": n, "value": str(fill_value),
            }
        
        if "arbk_dataregjistrimit" in df.columns and df["arbk_dataregjistrimit"].isna().sum() > 0:
            n = int(df["arbk_dataregjistrimit"].isna().sum())
            fill_value = df["arbk_dataregjistrimit"].dropna().median()
            if pd.isna(fill_value):
                contract_median = (df["data_e_kontrates"].dropna().median() 
                                  if "data_e_kontrates" in df.columns else pd.NaT)
                fill_value = (contract_median - pd.Timedelta(days=365) 
                            if pd.notna(contract_median) else pd.Timestamp("1999-01-01"))
            df["arbk_dataregjistrimit"] = df["arbk_dataregjistrimit"].fillna(fill_value)
            fill_report["rule_based_fills"]["arbk_dataregjistrimit"] = {
                "strategy": "fill_median_or_derived_date",
                "filled": n, "value": str(fill_value),
            }
        
        # Hierarchical group fill for business fields
        group_sets = [
            [MANDATORY_BUSINESS_COL, "komuna_lokacioni_i_asetit_te_shitur", "kategoria_e_asetit"],
            [MANDATORY_BUSINESS_COL, "komuna_lokacioni_i_asetit_te_shitur"],
            [MANDATORY_BUSINESS_COL],
        ]
        group_candidates = [
            "arbk_kapitali", "arbk_numpunetoreve", "arbk_pronari_1_kapitali",
            "arbk_pronari_1_kapitaliperqindje",
        ]
        for col in group_candidates:
            if col not in df.columns:
                continue
            total_filled = 0
            for groups in group_sets:
                if int(df[col].isna().sum()) == 0:
                    break
                df, filled = safe_group_median_fill(df, col, groups, GENERIC_GROUP_MIN_SIZE)
                total_filled += filled
            if total_filled > 0:
                fill_report["group_fills"][col] = {
                    "strategy": "hierarchical_group_median",
                    "filled": total_filled,
                }
        
        # Final fallback fills
        for col in df.columns:
            if int(df[col].isna().sum()) == 0:
                continue
            
            n = int(df[col].isna().sum())
            
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(0)
                fill_report["fallback_fills"][col] = {
                    "strategy": "fill_zero_numeric_no_artificial_bias",
                    "filled": n, "value": 0,
                }
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                fill_value = df[col].dropna().median()
                if pd.isna(fill_value):
                    fill_value = pd.Timestamp("2000-01-01")
                df[col] = df[col].fillna(fill_value)
                fill_report["fallback_fills"][col] = {
                    "strategy": "median_datetime",
                    "filled": n, "value": str(fill_value),
                }
            else:
                default = "Aktivitete tjera" if "aktiviteti" in col else "Te panjohura"
                fill_value = mode_or_default(df[col], default)
                df[col] = df[col].fillna(fill_value)
                fill_report["fallback_fills"][col] = {
                    "strategy": "mode_or_default_text",
                    "filled": n, "value": str(fill_value),
                }
        
        print(f"Missing value handling completed")
        return df, fill_report


class FeatureEngineer:
    """Creates derived features."""
    
    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create domain-relevant features."""
        obj_col = AREA_COLUMNS["object"]
        land_col = AREA_COLUMNS["land"]
        price_col = PRICE_COLUMN
        
        land_only, object_only, mixed = classify_asset_masks(df)
        
        # Area features
        if obj_col in df.columns and land_col in df.columns:
            df["total_area_m2"] = df[[obj_col, land_col]].fillna(0).sum(axis=1)
            df["object_to_land_ratio"] = np.where(
                df[land_col].fillna(0) > 0,
                df[obj_col].fillna(0) / df[land_col].replace(0, np.nan), 0,
            )
            df["is_land_only"] = land_only.astype(int)
            df["has_object"] = np.where(
                object_only | mixed | (df[obj_col].fillna(0) > 0), 1, 0
            )
            df.loc[land_only, "has_object"] = 0
            df["is_large_land"] = np.where(
                df[land_col].fillna(0) > df[land_col].median(skipna=True), 1, 0
            )
            if {"raw_has_object_area", "raw_has_land_area"}.issubset(df.columns):
                df["area_data_completeness_score"] = (
                    df[["raw_has_object_area", "raw_has_land_area"]].sum(axis=1)
                )
            else:
                df["area_data_completeness_score"] = 0
            
            df["asset_structure_type"] = np.select(
                [land_only, object_only, mixed],
                ["land_only", "object_only", "mixed_asset"],
                default="unknown_asset_type",
            )
        
        # Price features
        if price_col in df.columns:
            if "total_area_m2" in df.columns:
                df["price_per_total_m2"] = np.where(
                    df["total_area_m2"].fillna(0) > 0,
                    df[price_col] / df["total_area_m2"].replace(0, np.nan), 0,
                )
            if land_col in df.columns:
                df["price_per_land_m2"] = np.where(
                    df[land_col].fillna(0) > 0,
                    df[price_col] / df[land_col].replace(0, np.nan), 0,
                )
            if obj_col in df.columns:
                df["price_per_object_m2"] = np.where(
                    df[obj_col].fillna(0) > 0,
                    df[price_col] / df[obj_col].replace(0, np.nan), 0,
                )
            
            df["log_sale_price"] = np.log1p(df[price_col].clip(lower=0).fillna(0))
            df["is_high_value_asset"] = np.where(
                df[price_col].fillna(0) > df[price_col].median(skipna=True), 1, 0
            )
        
        # Interaction features
        if {"total_area_m2", "price_per_total_m2"}.issubset(df.columns):
            df["area_price_interaction"] = (
                df["total_area_m2"].fillna(0) * df["price_per_total_m2"].fillna(0)
            )
        
        # Temporal features
        if "data_e_kontrates" in df.columns:
            df["contract_year"] = df["data_e_kontrates"].dt.year
            df["contract_month"] = df["data_e_kontrates"].dt.month
            df["contract_quarter"] = df["data_e_kontrates"].dt.quarter
        
        # Business age features
        if "arbk_dataregjistrimit" in df.columns and "data_e_kontrates" in df.columns:
            business_age_days = (df["data_e_kontrates"] - df["arbk_dataregjistrimit"]).dt.days
            business_age_days = business_age_days.clip(lower=0)
            df["business_age_days_at_sale"] = business_age_days
            df["business_age_years_at_sale"] = business_age_days / 365.25
        
        # Capital/price ratio
        if "arbk_kapitali" in df.columns and price_col in df.columns:
            df["capital_to_sale_price_ratio"] = np.where(
                df[price_col].fillna(0) > 0,
                df["arbk_kapitali"].fillna(0) / df[price_col].replace(0, np.nan), 0,
            )
        
        # Price per employee
        if "arbk_numpunetoreve" in df.columns and price_col in df.columns:
            df["sale_price_per_employee"] = np.where(
                df["arbk_numpunetoreve"].fillna(0) > 0,
                df[price_col] / df["arbk_numpunetoreve"].replace(0, np.nan), 0,
            )
        
        print(f"Feature engineering completed")
        return df


class ColumnCleaner:
    """Removes low-quality columns."""
    
    @staticmethod
    def remove_leakage_columns(df: pd.DataFrame, report: dict) -> pd.DataFrame:
        """Remove data leakage columns."""
        drop_leakage = [c for c in LEAKAGE_COLUMNS if c in df.columns]
        if drop_leakage:
            df = df.drop(columns=drop_leakage)
        report["leakage_columns_removed"] = drop_leakage
        if drop_leakage:
            print(f"Removed {len(drop_leakage)} leakage columns")
        return df
    
    @staticmethod
    def remove_sparse_columns(df: pd.DataFrame, report: dict) -> pd.DataFrame:
        """Remove columns with high missing value ratios."""
        missing_ratio = df.isna().mean()
        sparse_cols = [
            c for c in missing_ratio.index
            if missing_ratio[c] > SPARSE_COL_THRESHOLD and c != TARGET_COL
        ]
        if sparse_cols:
            df = df.drop(columns=sparse_cols)
        report["sparse_columns_removed"] = sparse_cols
        if sparse_cols:
            print(f"Removed {len(sparse_cols)} sparse columns")
        return df
    
    @staticmethod
    def remove_constant_columns(df: pd.DataFrame, report: dict) -> pd.DataFrame:
        """Remove columns with only one unique value."""
        nunique = df.nunique(dropna=False)
        constant_cols = [
            c for c in nunique.index if nunique[c] <= 1 and c != TARGET_COL
        ]
        if constant_cols:
            df = df.drop(columns=constant_cols, errors="ignore")
        report["constant_columns_removed"] = constant_cols
        if constant_cols:
            print(f"Removed {len(constant_cols)} constant columns")
        return df
    
    @staticmethod
    def remove_dominant_noise_columns(df: pd.DataFrame, report: dict) -> pd.DataFrame:
        """Remove columns where one value dominates."""
        removed = []
        for col in list(df.columns):
            if col in PROTECTED_COLUMNS:
                continue
            
            vc = df[col].value_counts(dropna=False, normalize=True)
            if vc.empty:
                continue
            dominant_ratio = float(vc.iloc[0])
            if (dominant_ratio >= DOMINANT_VALUE_THRESHOLD and 
                df[col].nunique(dropna=False) <= 3):
                removed.append({
                    "column": col,
                    "dominant_ratio": dominant_ratio,
                    "nunique": int(df[col].nunique(dropna=False)),
                })
        
        cols_to_drop = [x["column"] for x in removed]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop, errors="ignore")
        
        report["dominant_low_information_columns_removed"] = removed
        if removed:
            print(f"Removed {len(removed)} dominant-value columns")
        return df
    
    @staticmethod
    def remove_identifier_columns(df: pd.DataFrame, report: dict) -> pd.DataFrame:
        """Remove identifier-like columns."""
        from .config import IDENTIFIER_COLUMNS
        identifier_cols = [c for c in IDENTIFIER_COLUMNS if c in df.columns]
        if identifier_cols:
            df = df.drop(columns=identifier_cols, errors="ignore")
        report["identifier_columns_removed"] = identifier_cols
        if identifier_cols:
            print(f"Removed {len(identifier_cols)} identifier columns")
        return df
    
    @staticmethod
    def remove_raw_date_columns(df: pd.DataFrame, report: dict) -> pd.DataFrame:
        """Remove raw date columns after feature extraction."""
        raw_date_cols = [c for c in DATE_COLUMNS if c in df.columns]
        if raw_date_cols:
            df = df.drop(columns=raw_date_cols, errors="ignore")
        report["raw_date_columns_removed"] = raw_date_cols
        if raw_date_cols:
            print(f"Removed {len(raw_date_cols)} raw date columns")
        return df


class TargetFilter:
    """Filters target variable."""
    
    @staticmethod
    def remove_rare_targets(df: pd.DataFrame, report: dict) -> pd.DataFrame:
        """Remove rows with rare target values."""
        buyer_counts = df[TARGET_COL].value_counts(dropna=True)
        valid_buyers = buyer_counts[buyer_counts >= MIN_TARGET_FREQUENCY].index
        before = len(df)
        df = df[df[TARGET_COL].isin(valid_buyers)].copy()
        report["rows_removed_rare_targets"] = int(before - len(df))
        report["buyers_remaining"] = int(df[TARGET_COL].nunique())
        if before > len(df):
            print(f"Removed {before - len(df)} rows with rare targets")
        return df
