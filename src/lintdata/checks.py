from typing import List, Optional

import numpy as np
import pandas as pd


def check_missing_values(df: pd.DataFrame) -> List[str]:
    """Check for missing values in the DataFrame

    Args:
        df (pd.DataFrame): The pandas DataFrame to check

    Returns:
        List[str]: A list of warning messages describing missing values found.

    Example:
    >>> df = pd.DataFrame({'a': [1, 2, None], 'b': ['x', 'y', 'z']})
    >>> warnings = check_missing_values(df)
    >>> print(warnings[0])
    [Missing Values] Column 'a': 1 missing values (33.3%)
    """
    warnings: List[str] = []
    missing_info = df.isna().sum()
    missing_cols = missing_info[missing_info > 0]

    if not missing_cols.empty:
        total_rows = len(df)
        for col, count in missing_cols.items():
            percent = (count / total_rows) * 100
            warnings.append(
                f"[Missing Values] Column '{col}': {count} missing values ({percent:.1f}%)"
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
    [Duplicate Rows] Found 1 duplicate row(s) at index: 2
    """
    warnings: List[str] = []

    if df.empty:
        return warnings

    duplicate_mask = df.duplicated()
    duplicate_indices = df.index[duplicate_mask].tolist()

    if len(duplicate_indices) > 0:
        indices_str = ", ".join(map(str, duplicate_indices))
        warnings.append(
            f"[Duplicate Rows] Found {len(duplicate_indices)} duplicate row(s) at index: {indices_str}"
        )

    return warnings


def check_mixed_types(df: pd.DataFrame) -> List[str]:
    """Check for columns containing mixed data types.

    Detects columns where values have different Python types (e.g., integers
    mixed with strings). Reports the specific types found and their proportions.

    Args:
        df (pd.DataFrame): The pandas DataFrame to check.

    Returns:
        List[str]: A list of warning messages for mixed data types found with specific
                    type breakdowns.

    Example:
    >>> df = pd.DataFrame({'a': [1, 'two', 3], 'b': [1.0, 2.0, 3.0]})
    >>> warnings = check_mixed_types(df)
    >>> print(warnings[0])
    [Mixed Types] Column 'a' has mixed types: int (66%), str (33%)
    """
    warnings: List[str] = []
    if df.empty:
        return warnings

    for col in df.columns:
        if df[col].isna().all():
            continue

        non_null_values = df[col].dropna()

        if len(non_null_values) == 0:
            continue

        type_counts = {}
        for value in non_null_values:
            value_type = type(value).__name__
            type_counts[value_type] = type_counts.get(value_type, 0) + 1

        if len(type_counts) > 1:
            total = len(non_null_values)

            sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)

            type_breakdown = ", ".join(
                [
                    f"{type_name} ({count / total * 100:.0f}%)"
                    for type_name, count in sorted_types
                ]
            )

            warnings.append(
                f"[Mixed Types] Column '{col}' has mixed types: {type_breakdown}"
            )

    return warnings


def check_whitespace(df: pd.DataFrame) -> List[str]:
    """Detects string values that have leading or trailing spaces,
    which can cause issues in data analysis and matching operations.

    Args:
        df (pd.DataFrame): The pandas DataFrame to check.

    Returns:
        List[str]: A list of warning messages for columns with leading or
                    trailing whitespace.

    Example:
    >>> df = pd.DataFrame({'a': [' x', 'y ', ' z ']})
    >>> warnings = check_whitespace(df)
    >>> print(warnings[0])
    [Whitespace] Column 'a' has 3 value(s) with leading or trailing whitespace.
    """
    warnings: List[str] = []

    if df.empty:
        return warnings

    string_columns = df.select_dtypes(include=["object"]).columns

    for col in string_columns:
        if df[col].isna().all():
            continue

        non_null_values = df[col].dropna()

        has_whitespace = (
            non_null_values.astype(str) != non_null_values.astype(str).str.strip()
        )
        whitespace_count = has_whitespace.sum()

        if whitespace_count > 0:
            warnings.append(
                f"[Whitespace] Column '{col}' has {whitespace_count} value(s) with leading or trailing whitespace."
            )
    return warnings


def check_constant_columns(df: pd.DataFrame) -> List[str]:
    """Check for columns where all values are the same (zero variance).

    Identifies columns with only one unique value (excluding NaN), which are often
    redundant for analysin/modelling.

    Args:
        df (pd.DataFrame): The pandas DataFrame to check.

    Returns:
        List[str]: A list of warning messages for constant columns found.

    Example:
    >>> df = pd.DataFrame({'a': [1, 1, 1]})
    >>> warnings = check_constant_columns(df)
    >>> print(warnings[0])
    [Constant Column] Column 'a' has only one unique value: 1.
    """
    warnings: List[str] = []

    if df.empty:
        return warnings

    for col in df.columns:
        unique_values = df[col].dropna().unique()

        if len(unique_values) == 0:
            warnings.append(
                f"[Constant Column] Column '{col}' contains only missing values."
            )

        elif len(unique_values) == 1:
            constant_value = unique_values[0]

            display_value = (
                f"'{constant_value}'"
                if isinstance(constant_value, str)
                else str(constant_value)
            )

            warnings.append(
                f"[Constant Column] Column '{col}' has only one unique value: {display_value}."
            )
    return warnings


def check_unique_columns(
    df: pd.DataFrame, threshold: Optional[float] = 0.95
) -> List[str]:
    """Check for columns with a high proportion of unique values.

    Args:
        df (pd.DataFrame): The pandas DataFrame to check.
        threshold (float, optional): The unique value proportion threshold.
                                        Defaults to 0.95.

    Returns:
        List[str]: A list of warning messages for columns exceeding
                    the unique value threshold.

    Example:
    >>> df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
    >>> warnings = check_unique_columns(df, threshold=0.8)
    >>> print(warnings[0])
    [Unique Column] Column 'a' is 100.0% unique
    """
    warnings: List[str] = []

    if threshold:
        if not (0 < threshold <= 1):
            raise ValueError("Uniqueness threshold must be between 0 and 1.")
    else:
        threshold = 0.95

    if df.empty:
        return warnings

    for col in df.columns:
        non_null_values = df[col].dropna()

        if len(non_null_values) == 0:
            continue

        num_unique = non_null_values.nunique()
        total_non_null = len(non_null_values)
        unique_ratio = num_unique / total_non_null

        if unique_ratio >= threshold:
            percent = unique_ratio * 100
            warnings.append(f"[Unique Column] Column '{col}' is {percent:.1f}% unique")

    return warnings


def check_outliers(
    df: pd.DataFrame, method: Optional[str] = "iqr", threshold: Optional[float] = 1.5
) -> List[str]:
    """Check for outliers using different methods.

    Provides a threshold value that sets trigger for outliers. Currently
    supports IQR method.

    Args:
        df (pd.DataFrame): The pandas DataFrame to check.
        method (Optional[str], optional): The outlier detection method to use.
                                            Defaults to "iqr".
        threshold (Optional[float], optional): The threshold for detecting outliers.
                                                Defaults to 1.5.

    Raises:
        ValueError (Method): If the method is not supported.
        ValueError (Threshold): If the threshold is not a positive number.

    Returns:
        List[str]: A list of warning messages for columns with detected outliers.
    Example:
    >>> df = pd.DataFrame({'a': [1, 2, 3, 100, 5]})
    >>> warnings = check_outliers(df)
    >>> print(warnings[0])
    [Outliers] Column 'a': 1 potential outlier(s) detected (iqr method).
    """
    warnings: List[str] = []

    if method and method.lower() != "iqr":
        raise ValueError(
            "Currently, only 'iqr' method is supported for outlier detection."
        )

    if threshold:
        if threshold <= 0:
            raise ValueError("Threshold must be a positive number.")
    else:
        threshold = 1.5

    if df.empty:
        return warnings

    numeric_columns = df.select_dtypes(include=[np.number]).columns

    for col in numeric_columns:
        non_null_values = df[col].dropna()

        if len(non_null_values) == 0:
            continue

        q1 = non_null_values.quantile(0.25)
        q3 = non_null_values.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)

        if outlier_count > 0:
            warnings.append(
                f"[Outliers] Column '{col}': {outlier_count} potential outlier(s) detected ({method} method)."
            )

    return warnings


def check_missing_patterns(df: pd.DataFrame, threshold: float = 0.9) -> List[str]:
    """Detects patterns in missing data across columns.

    Identifies when multiple columns have missing values in the same rows,
    which may indicate systematic missingness (e.g., related fields that are
    filled out together or not at all.)

    Args:
        df (pd.DataFrame): The pandas DataFrame to check.
        threshold (float, optional): The threshold for detecting missing patterns.
                                        Defaults to 0.9.

    Returns:
        List[str]: A list of warning messages for columns with detected missing patterns.
    """
    warnings: List[str] = []

    if df.empty:
        return warnings

    cols_with_missing = df.columns[df.isna().any()].tolist()

    if len(cols_with_missing) < 2:
        return warnings

    checked_pairs = set()

    for i, col1 in enumerate(cols_with_missing):
        for col2 in cols_with_missing[i + 1 :]:
            pair_key = tuple(sorted([col1, col2]))
            if pair_key in checked_pairs:
                continue
            checked_pairs.add(pair_key)

            missing_col1 = df[col1].isna()
            missing_col2 = df[col2].isna()

            both_missing = (missing_col1 & missing_col2).sum()

            total_missing_col1 = missing_col1.sum()
            total_missing_col2 = missing_col2.sum()

            if total_missing_col1 == 0 or total_missing_col2 == 0:
                continue

            overlap_ratio1 = both_missing / total_missing_col1
            overlap_ratio2 = both_missing / total_missing_col2

            if overlap_ratio1 >= threshold and overlap_ratio2 >= threshold:
                warnings.append(
                    f"[Missing Patterns] Columns '{col1}' and '{col2}' identical missing rows (likely related)"
                )
    return warnings
