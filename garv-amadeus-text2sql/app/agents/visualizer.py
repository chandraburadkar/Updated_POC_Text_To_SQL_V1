from __future__ import annotations

from typing import Dict, Any, Optional
import pandas as pd


def suggest_plot(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns a suggested plot type + columns.
    Does NOT plot. (Notebook can plot separately)
    """
    if df is None or df.empty:
        return {"plot": None, "reason": "Empty dataframe"}

    cols = list(df.columns)

    # Detect datetime column
    datetime_cols = [c for c in cols if pd.api.types.is_datetime64_any_dtype(df[c])]
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in cols if c not in numeric_cols and c not in datetime_cols]

    # Time series
    if len(datetime_cols) >= 1 and len(numeric_cols) >= 1:
        return {
            "plot": "line",
            "x": datetime_cols[0],
            "y": numeric_cols[0],
            "reason": "Datetime + numeric detected",
        }

    # Ranking chart
    if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
        return {
            "plot": "bar",
            "x": categorical_cols[0],
            "y": numeric_cols[0],
            "reason": "Category + numeric detected",
        }

    return {"plot": None, "reason": "No clear chart mapping"}