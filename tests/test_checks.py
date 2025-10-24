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
