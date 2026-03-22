"""
Load Layer - Data loading and persistence.
"""
from pathlib import Path
import json
import pandas as pd
from .config import OUTPUT_FILES


class DataLoader:
    """Loads and persists data artifacts."""
    
    def __init__(self, output_dir: Path, models_dir: Path):
        self.output_dir = Path(output_dir)
        self.models_dir = Path(models_dir)
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Create output directories if they don't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def save_cleaned_data(self, df_cleaned: pd.DataFrame, report: dict) -> None:
        """Save cleaned dataset and transformation report."""
        self._save_cleaned_data(df_cleaned)
        self._save_report(report)
        self._print_save_summary()
    
    def _save_cleaned_data(self, df: pd.DataFrame) -> None:
        """Save cleaned dataset."""
        path = self.output_dir / OUTPUT_FILES["cleaned_dataset"]
        df.to_excel(path, index=False)
        print(f"✓ Saved: {path}")
    
    def _save_report(self, report: dict) -> None:
        """Save transformation report."""
        path = self.output_dir / OUTPUT_FILES["report"]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f"✓ Saved: {path}")
    
    @staticmethod
    def _print_save_summary() -> None:
        """Print summary of saved artifacts."""
        print("\n" + "=" * 60)
        print("ALL ARTIFACTS SAVED SUCCESSFULLY")
        print("=" * 60)
