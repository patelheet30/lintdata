"""
Tests for the individual check functions in checks.py
"""

import numpy as np
import pandas as pd

from lintdata import checks


def test_check_missing_values_clean():
    """Test that a DataFrame with no missing values returns an empty list."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    warnings = checks.check_missing_values(df)
    assert warnings == []


def test_check_missing_values_one_column_missing():
    """Test detection and correct reporting for one missing value."""
    df = pd.DataFrame({"a": [1, 2, np.nan], "b": ["x", "y", "z"]})
    warnings = checks.check_missing_values(df)

    assert len(warnings) == 1
    # Check the content of the warning string
    assert "Column 'a'" in warnings[0]
    assert "1 missing" in warnings[0]
    assert "(33.3%)" in warnings[0]


def test_check_missing_values_multiple_columns_missing():
    """Test detection and reporting for multiple columns."""
    df = pd.DataFrame(
        {"a": [1, np.nan, np.nan, 4], "b": ["w", "x", "y", "z"], "c": [np.nan, 2, 3, 4]}
    )
    warnings = checks.check_missing_values(df)

    assert len(warnings) == 2
    # Check warning for 'a'
    assert "Column 'a'" in warnings[0]
    assert "2 missing" in warnings[0]
    assert "(50.0%)" in warnings[0]
    # Check warning for 'c'
    assert "Column 'c'" in warnings[1]
    assert "1 missing" in warnings[1]
    assert "(25.0%)" in warnings[1]


def test_check_missing_values_all_missing():
    """Test a column that is entirely missing values."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [np.nan, np.nan, np.nan]})
    warnings = checks.check_missing_values(df)

    assert len(warnings) == 1
    assert "Column 'b'" in warnings[0]
    assert "3 missing" in warnings[0]
    assert "(100.0%)" in warnings[0]


# ==== Tests for check_duplicate_rows ====


def test_check_duplicate_rows_no_duplicates():
    """No duplicate rows present."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    warnings = checks.check_duplicate_rows(df)
    assert warnings == []


def test_check_duplicate_rows_detects_duplicates():
    """Core functionality: detects duplicate rows."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 2],
            "name": ["Alice", "Bob", "Bob"],
        }
    )
    warnings = checks.check_duplicate_rows(df)
    assert len(warnings) == 1
    assert "1 duplicate row(s)" in warnings[0] or "1 duplicate" in warnings[0]
    assert "index: 2" in warnings[0]


def test_check_duplicate_rows_empty_dataframe():
    """Edge case: empty DataFrame should return no warnings."""
    df = pd.DataFrame(columns=["a", "b"])
    warnings = checks.check_duplicate_rows(df)
    assert warnings == []


def test_check_duplicate_rows_all_duplicates():
    """All rows are duplicates except the first one."""
    df = pd.DataFrame(
        {
            "id": [1, 1, 1],
            "name": ["Alice", "Alice", "Alice"],
        }
    )
    warnings = checks.check_duplicate_rows(df)
    assert len(warnings) == 1
    assert "2 duplicate row(s)" in warnings[0] or "2 duplicates" in warnings[0]
    assert "index: 1, 2" in warnings[0]


def test_check_duplicate_rows_multiple_duplicates_sets():
    """Multiple sets of duplicate rows."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 2, 3, 3, 3],
            "name": ["Alice", "Bob", "Bob", "Charlie", "Charlie", "Charlie"],
        }
    )
    warnings = checks.check_duplicate_rows(df)
    assert len(warnings) == 1
    assert "3 duplicate row(s)" in warnings[0] or "3 duplicates" in warnings[0]
    assert "index: 2, 4, 5" in warnings[0]
