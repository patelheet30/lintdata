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


# ==== Mixed Type Tests ====


def test_check_mixed_types_no_mixed_types():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    warnings = checks.check_mixed_types(df)
    assert warnings == []


def test_check_mixed_types_detects_mixed():
    df = pd.DataFrame(
        {
            "price": [10, "20", 30],
        }
    )
    warnings = checks.check_mixed_types(df)
    assert len(warnings) == 1
    assert "Column 'price'" in warnings[0]
    assert "int" in warnings[0] or "int64" in warnings[0]
    assert "str" in warnings[0] or "object" in warnings[0]


def test_check_mixed_types_empty_dataframe():
    df = pd.DataFrame()
    warnings = checks.check_mixed_types(df)
    assert warnings == []


def test_check_mixed_types_with_nan():
    df = pd.DataFrame(
        {
            "value": [1, 2, np.nan, "text"],
        }
    )
    warnings = checks.check_mixed_types(df)
    assert len(warnings) == 1
    assert "Column 'value'" in warnings[0]
    assert "int" in warnings[0] or "int64" in warnings[0]
    assert "str" in warnings[0] or "object" in warnings[0]


def test_check_mixed_types_multiple_columns():
    df = pd.DataFrame(
        {
            "col1": [1, 2, 3],
            "col2": [1.0, "2.0", 3.0],
            "col3": ["a", "b", "c"],
            "col4": [True, False, "True"],
        }
    )
    warnings = checks.check_mixed_types(df)
    assert len(warnings) == 2
    assert any("Column 'col2'" in warning for warning in warnings)
    assert any("Column 'col4'" in warning for warning in warnings)


# ==== Whitespace Tests ====


def test_check_whitespace_no_whitespace():
    df = pd.DataFrame({"a": ["x", "y", "z"], "b": ["foo", "bar", "baz"]})
    warnings = checks.check_whitespace(df)
    assert warnings == []


def test_check_whitespace_detects_leading():
    df = pd.DataFrame({"a": [" x", "y", "z"]})
    warnings = checks.check_whitespace(df)
    assert len(warnings) == 1
    assert "Column 'a'" in warnings[0]
    assert "1 value(s)" in warnings[0]


def test_check_whitespace_detects_trailing():
    df = pd.DataFrame({"a": ["x ", "y", "z"]})
    warnings = checks.check_whitespace(df)
    assert len(warnings) == 1
    assert "Column 'a'" in warnings[0]
    assert "1 value(s)" in warnings[0]


def test_check_whitespace_detects_both():
    df = pd.DataFrame({"a": [" x ", "y", "z"]})
    warnings = checks.check_whitespace(df)
    assert len(warnings) == 1
    assert "Column 'a'" in warnings[0]
    assert "1 value(s)" in warnings[0]


def test_check_whitespace_multiple_columns():
    df = pd.DataFrame(
        {
            "a": [" x", "y", "z"],
            "b": ["foo", " bar", "baz "],
        }
    )
    warnings = checks.check_whitespace(df)
    assert len(warnings) == 2
    assert any("Column 'a'" in warning for warning in warnings)
    assert any("Column 'b'" in warning for warning in warnings)


def test_check_whitespace_empty_dataframe():
    df = pd.DataFrame()
    warnings = checks.check_whitespace(df)
    assert warnings == []


def test_check_whitespace_non_string_column():
    df = pd.DataFrame({"a": [1, 2, 3]})
    warnings = checks.check_whitespace(df)
    assert warnings == []


def test_check_whitespace_nan_values():
    df = pd.DataFrame({"a": [" x", np.nan, "z "]})
    warnings = checks.check_whitespace(df)
    assert len(warnings) == 1
    assert "Column 'a'" in warnings[0]
    assert "2 value(s)" in warnings[0]


# === Check Constant Columns Tests ====


def test_check_constant_columns_no_constants():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    warnings = checks.check_constant_columns(df)
    assert warnings == []


def test_check_constant_columns_detects_constants():
    df = pd.DataFrame({"a": ["x", "x", "x"]})
    warnings = checks.check_constant_columns(df)
    assert len(warnings) == 1
    assert "Column 'a'" in warnings[0]
    assert "only one unique value: 'x'" in warnings[0]


def test_check_constant_columns_numeric_constants():
    df = pd.DataFrame({"a": [3.14, 3.14, 3.14], "b": [1, 2, 3]})
    warnings = checks.check_constant_columns(df)
    assert len(warnings) == 1
    assert "Column 'a'" in warnings[0]
    assert "only one unique value: 3.14" in warnings[0]


def test_check_constant_columns_empty_dataframe():
    df = pd.DataFrame()
    warnings = checks.check_constant_columns(df)
    assert warnings == []


def test_check_constant_columns_single_row():
    df = pd.DataFrame({"a": [42]})
    warnings = checks.check_constant_columns(df)
    assert len(warnings) == 1


def test_check_constant_columns_with_nan():
    df = pd.DataFrame({"a": [np.nan, np.nan, np.nan]})
    warnings = checks.check_constant_columns(df)
    assert len(warnings) == 1
    assert "Column 'a'" in warnings[0]


def test_check_constant_columns_mixed_with_nan_and_constant():
    df = pd.DataFrame({"a": [5, 5, np.nan, 5]})
    warnings = checks.check_constant_columns(df)
    assert len(warnings) == 1
    assert "Column 'a'" in warnings[0]
    assert "only one unique value: 5" in warnings[0]


def test_check_constant_columns_multiple_constants():
    df = pd.DataFrame(
        {"a": ["constant", "constant", "constant"], "b": [42, 42, 42], "c": [1, 2, 3]}
    )
    warnings = checks.check_constant_columns(df)
    assert len(warnings) == 2
    assert any("Column 'a'" in warning for warning in warnings)
    assert any("Column 'b'" in warning for warning in warnings)


def test_check_constant_columns_boolean_constant():
    df = pd.DataFrame({"a": [True, True, True, True]})
    warnings = checks.check_constant_columns(df)
    assert len(warnings) == 1
    assert "Column 'a'" in warnings[0]
    assert "only one unique value: True" in warnings[0]


# ==== Unique Columns Test ====


def test_check_unique_columns_detects_uniques():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    warnings = checks.check_unique_columns(df)
    assert len(warnings) == 1
    assert "Column 'a'" in warnings[0]
    assert "100.0% unique" in warnings[0]


def test_check_unique_columns_custom_threshold():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 5, 5, 5, 5, 5],
            "b": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        }
    )
    warnings = checks.check_unique_columns(df)
    assert len(warnings) == 1
    assert "Column 'b'" in warnings[0]

    warnings_low = checks.check_unique_columns(df, threshold=0.4)
    assert len(warnings_low) == 2


def test_check_unique_columns_empty_dataframe():
    df = pd.DataFrame()
    warnings = checks.check_unique_columns(df)
    assert warnings == []


def test_check_unique_columns_with_nan():
    df = pd.DataFrame({"a": [1, 2, 3, np.nan, np.nan]})
    warnings = checks.check_unique_columns(df)
    assert len(warnings) == 1
    assert "Column 'a'" in warnings[0]
    assert "100.0% unique" in warnings[0]


def test_check_unique_columns_single_row():
    df = pd.DataFrame({"a": [42]})
    warnings = checks.check_unique_columns(df)
    assert len(warnings) == 1
    assert "Column 'a'" in warnings[0]


def test_check_unique_columns_all_nan():
    df = pd.DataFrame({"a": [np.nan, np.nan, np.nan], "b": [1, 2, 3]})
    warnings = checks.check_unique_columns(df)
    assert len(warnings) == 1
    assert "Column 'b'" in warnings[0]


def test_check_unique_columns_multiple_unique_columns():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["x", "y", "z", "w"], "c": [1, 1, 1, 1]})
    warnings = checks.check_unique_columns(df)
    assert len(warnings) == 2
    assert any("Column 'a'" in warning for warning in warnings)
    assert any("Column 'b'" in warning for warning in warnings)


# ==== Outliers Tests ====


def test_check_outliers_no_outliers():
    df = pd.DataFrame({"a": [10, 12, 11, 13, 12], "b": [20, 22, 21, 19, 20]})
    warnings = checks.check_outliers(df)
    assert warnings == []


def test_check_outliers_with_outliers():
    df = pd.DataFrame({"a": [10, 12, 11, 13, 100], "b": [20, 22, 21, 19, 20]})
    warnings = checks.check_outliers(df)
    assert len(warnings) == 1
    assert "Column 'a'" in warnings[0]
    assert "potential outlier(s)" in warnings[0]


def test_check_outliers_empty_dataframe():
    df = pd.DataFrame()
    warnings = checks.check_outliers(df)
    assert warnings == []


def test_check_outliers_custom_threshold():
    df = pd.DataFrame(
        {"a": [10, 15, 20, 25, 30, 35, 150], "b": [10, 20, 30, 40, 50, 80, 110]}
    )
    warnings = checks.check_outliers(df)
    assert len(warnings) == 1
    assert "Column 'a'" in warnings[0]
    warnings_low = checks.check_outliers(df, threshold=0.9)
    assert len(warnings_low) == 2
    assert "Column 'a'" in warnings_low[0]
    assert "potential outlier(s)" in warnings_low[0]
    assert "Column 'b'" in warnings_low[1]
    assert "potential outlier(s)" in warnings_low[1]
