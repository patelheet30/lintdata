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


def check_duplicate_rows(df: pd.DataFrame) -> List[str]:
    """Check for duplicate rows in the DataFrame

    A row is considered a duplicate if all its values match another row in the
    DataFrame. The first occurrence is not counted as a duplicate. Indices start at 0.


    Args:
        df (pd.DataFrame): The pandas DataFrame to check.

    Returns:
        List[str]: A list of warning messages for duplicate rows with specific indices.

    Example:
    >>> df = pd.DataFrame({'a': [1, 2, 2], 'b': ['x', 'y', 'y']})
    >>> warnings = check_duplicate_rows(df)
    >>> print(warnings[0])
    ["[Duplicate Rows] Found 1 duplicate row(s) at index: 2"]
    """
    warnings: List[str] = []

    if df.empty:
        return warnings

    duplicate_mask = df.duplicated()
    duplicate_indices = df.index[duplicate_mask].tolist()

    if len(duplicate_indices) > 0:
        indices_str = ", ".join(map(str, duplicate_indices))
        warnings.append(
            f"[Duplicate Rows] Found {len(duplicate_indices)} duplicate row(s) "
            f"at index: {indices_str}"
        )

    return warnings
