"""
A module containing all data quality checks for LintData.

Each check is implemented as a function that takes a pandas DataFrame
and returns a list of issues found. If no issues are found, an empty list is returned.
"""

from typing import List

import pandas as pd


def check_missing_values(df: pd.DataFrame) -> List[str]:
    warnings: List[str] = []
    missing_info = df.isna().sum()
    missing_cols = missing_info[missing_info > 0]

    if not missing_cols.empty:
        total_rows = len(df)
        for col, count in missing_cols.items():
            percent = (count / total_rows) * 100
            warnings.append(
                f"[Missing Values] Column '{col}': {count} missing values "
                f"({percent:.1f}%)"
            )

    return warnings
