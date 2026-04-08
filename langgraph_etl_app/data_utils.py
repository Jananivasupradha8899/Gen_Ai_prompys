import pandas as pd
import numpy as np
from typing import Dict, Any

def clean_data(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Cleans the dataframe based on user configuration.
    """
    new_df = df.copy()
    
    # 1. Handle Missing Values
    if config.get("drop_na", False):
        new_df = new_df.dropna()
    else:
        # Fill numeric NAs with mean/median
        fill_strategy = config.get("fill_na_strategy", "mean")
        numeric_cols = new_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_strategy == "mean":
                new_df[col] = new_df[col].fillna(new_df[col].mean())
            elif fill_strategy == "median":
                new_df[col] = new_df[col].fillna(new_df[col].median())
                
        # Fill object NAs with 'Unknown'
        object_cols = new_df.select_dtypes(include=['object']).columns
        for col in object_cols:
            new_df[col] = new_df[col].fillna("Unknown")

    # 2. Normalize Numeric Columns
    if config.get("normalize", False):
        numeric_cols = new_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            min_val = new_df[col].min()
            max_val = new_df[col].max()
            if max_val != min_val:
                new_df[col] = (new_df[col] - min_val) / (max_val - min_val)

    # 3. Filtering (Example: Filter by Age if column exists)
    age_threshold = config.get("age_threshold", 0)
    if "age" in new_df.columns:
        new_df = new_df[new_df["age"] >= age_threshold]

    return new_df

def validate_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Basic validation checks for data quality.
    """
    metrics = {
        "rows": len(df),
        "cols": len(df.columns),
        "missing_values": df.isnull().sum().sum(),
        "is_empty": df.empty
    }
    return metrics
