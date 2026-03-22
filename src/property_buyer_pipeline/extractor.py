"""
Extract Layer - Data extraction from source systems.
"""
from pathlib import Path
import pandas as pd
from .utils import normalize_column_name


class DataExtractor:
    """Extracts data from source files."""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
    
    def extract(self) -> pd.DataFrame:
        """Load raw data from Excel file and normalize column names."""
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Input file '{self.file_path}' was not found. "
                f"Put the Excel file next to this script or update INPUT_FILE."
            )
        
        df = pd.read_excel(self.file_path)
        original_columns = df.columns.tolist()
        df.columns = [normalize_column_name(c) for c in df.columns]
        
        self._print_extraction_summary(df, original_columns)
        return df
    
    @staticmethod
    def _print_extraction_summary(df: pd.DataFrame, original_columns: list) -> None:
        """Print summary of extracted data."""
        print("=" * 60)
        print("DATA EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"Loaded rows: {len(df)}")
        print(f"Loaded columns: {len(df.columns)}")
        print("\nSample column normalization (Original -> Normalized):")
        for old, new in list(zip(original_columns, df.columns))[:12]:
            print(f"  {old} -> {new}")
        print("=" * 60)
