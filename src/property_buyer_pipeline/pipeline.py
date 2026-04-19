"""
ETL Pipeline Orchestration - Coordinates Extract, Transform, Load operations.
"""
from pathlib import Path
import sys
import warnings
from sklearn.model_selection import train_test_split
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from property_buyer_pipeline.config import (
        INPUT_FILE, OUTPUT_DIR, MODELS_DIR, TARGET_COL, MANDATORY_BUSINESS_COL
    )
    from property_buyer_pipeline.extractor import DataExtractor
    from property_buyer_pipeline.transformer import (
        DataValidator, DuplicateHandler, TypeConverter, AreaInferencer,
        MissingValueHandler, FeatureEngineer, BuyerProfileEngineer, ColumnCleaner, TargetFilter
    )
    from property_buyer_pipeline.loader import DataLoader
    from property_buyer_pipeline.utils import identify_outliers_iqr
else:
    from .config import (
        INPUT_FILE, OUTPUT_DIR, MODELS_DIR, TARGET_COL, MANDATORY_BUSINESS_COL
    )
    from .extractor import DataExtractor
    from .transformer import (
        DataValidator, DuplicateHandler, TypeConverter, AreaInferencer,
        MissingValueHandler, FeatureEngineer, BuyerProfileEngineer, ColumnCleaner, TargetFilter
    )
    from .loader import DataLoader
    from .utils import identify_outliers_iqr

warnings.filterwarnings("ignore", category=FutureWarning)


class ETLPipeline:
    """Main ETL pipeline orchestrator."""
    
    def __init__(self):
        self.report = {
            "pipeline_version": "v3.2",
            "description": "Clean architecture ETL for property buyer classification",
        }
        self.loader = DataLoader(OUTPUT_DIR, MODELS_DIR)
    
    def run(self) -> None:
        """Execute complete ETL pipeline."""
        print("\n" + "=" * 70)
        print("STARTING ETL PIPELINE - PROPERTY BUYER CLASSIFICATION")
        print("=" * 70 + "\n")
        
        # EXTRACT
        df = self._extract()
        
        # TRANSFORM
        df_prepared = self._transform(df)
        
        # LOAD
        self._load(df_prepared)
        
        print("\n" + "=" * 70)
        print("ETL PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 70 + "\n")
    
    def _extract(self) -> np.ndarray:
        """Extract phase - Load raw data."""
        print("\n[PHASE 1: EXTRACT]")
        print("-" * 70)
        
        extractor = DataExtractor(INPUT_FILE)
        df = extractor.extract()
        
        self.report["rows_initial"] = int(len(df))
        self.report["columns_initial"] = int(len(df.columns))
        
        return df
    
    def _transform(self, df) -> np.ndarray:
        """Transform phase - Data cleaning and feature engineering."""
        print("\n[PHASE 2: TRANSFORM]")
        print("-" * 70)
        
        # Validation
        DataValidator.validate_required_columns(df)
        df = DataValidator.remove_null_business_types(df, self.report)
        df = DataValidator.remove_null_targets(df, self.report)
        
        # Duplicates
        df = DuplicateHandler.remove_full_duplicates(df, self.report)
        df = DuplicateHandler.remove_business_key_duplicates(df, self.report)
        
        # Type conversion
        df = TypeConverter.convert_numeric_columns(df)
        df = TypeConverter.convert_date_columns(df)
        
        # Area inference
        df = AreaInferencer.infer_areas(df, self.report)
        
        # Feature engineering
        df = FeatureEngineer.engineer_features(df)
        
        # Outlier detection (no modification)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_from_outlier = {TARGET_COL, MANDATORY_BUSINESS_COL}
        df, outlier_stats = identify_outliers_iqr(df, numeric_cols, exclude_cols=exclude_from_outlier)
        self.report["outlier_identification"] = outlier_stats
        
        # Column cleaning
        df = ColumnCleaner.remove_leakage_columns(df, self.report)
        df = ColumnCleaner.remove_sparse_columns(df, self.report)
        df = ColumnCleaner.remove_constant_columns(df, self.report)
        df = ColumnCleaner.remove_dominant_noise_columns(df, self.report)
        df = ColumnCleaner.remove_identifier_columns(df, self.report)
        
        # Missing values
        df, fill_report = MissingValueHandler.fill_missing_values(df)
        self.report["missing_value_treatment"] = fill_report

        # Create the derived profile target once in the preprocessing pipeline.
        df = BuyerProfileEngineer.add_buyer_profile(df, self.report)
        df = TargetFilter.keep_only_allowed_buyer_profiles(df, self.report)
        # Verify no missing values
        remaining_missing = int(df.isna().sum().sum())
        if remaining_missing != 0:
            raise ValueError(f"Found {remaining_missing} missing values after filling.")
        self.report["remaining_missing_values_after_fill"] = remaining_missing
        
        # Remove rare targets
        df = TargetFilter.remove_rare_targets(df, self.report)
        
        # Remove raw dates after feature extraction
        df = ColumnCleaner.remove_raw_date_columns(df, self.report)
        
        self.report["rows_final_after_all_transformations"] = int(len(df))
        self.report["columns_final_after_all_transformations"] = int(len(df.columns))
        
        print("\nTransform phase completed")
        return df
    
    def _load(self, df_prepared) -> None:
        """Load phase - Save cleaned dataset, train/test split, and report."""
        print("\n[PHASE 3: LOAD]")
        print("-" * 70)
        
        # Save cleaned dataset
        self.loader.save_cleaned_data(
            df_cleaned=df_prepared,
            report=self.report
        )
        
        # Train-test split (80-20)
        df_train, df_test = train_test_split(
            df_prepared,
            test_size=0.2,
            random_state=42
        )
        
        # Save train and test datasets
        self.loader.save_train_test_data(
            df_train=df_train,
            df_test=df_test
        )
        
        self.report["train_size"] = int(len(df_train))
        self.report["test_size"] = int(len(df_test))


def main():
    """Entry point for ETL pipeline."""
    pipeline = ETLPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
