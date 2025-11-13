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
    df = pd.DataFrame({"a": [1, np.nan, np.nan, 4], "b": ["w", "x", "y", "z"], "c": [np.nan, 2, 3, 4]})
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
    df = pd.DataFrame({"a": ["constant", "constant", "constant"], "b": [42, 42, 42], "c": [1, 2, 3]})
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
    df = pd.DataFrame({"a": [10, 15, 20, 25, 30, 35, 150], "b": [10, 20, 30, 40, 50, 80, 110]})
    warnings = checks.check_outliers(df)
    assert len(warnings) == 1
    assert "Column 'a'" in warnings[0]
    warnings_low = checks.check_outliers(df, threshold=0.9)
    assert len(warnings_low) == 2
    assert "Column 'a'" in warnings_low[0]
    assert "potential outlier(s)" in warnings_low[0]
    assert "Column 'b'" in warnings_low[1]
    assert "potential outlier(s)" in warnings_low[1]


# ==== Missing Patterns Tests ====


def test_check_missing_patterns_no_pattern():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    warnings = checks.check_missing_patterns(df)
    assert warnings == []


def test_check_missing_patterns_detects_pattern():
    df = pd.DataFrame(
        {
            "income": [50000, np.nan, 60000, np.nan],
            "job": ["Engineer", np.nan, "Doctor", np.nan],
            "age": [25, 26, 27, 28],
        }
    )

    warnings = checks.check_missing_patterns(df)
    assert len(warnings) == 1
    assert "income" in warnings[0]
    assert "job" in warnings[0]
    assert "identical missing rows" in warnings[0].lower()


def test_check_missing_patterns_empty_dataframe():
    df = pd.DataFrame()
    warnings = checks.check_missing_patterns(df)
    assert warnings == []


# ==== Case Consistency Tests ====


def test_check_case_consistency_no_issues():
    df = pd.DataFrame({"a": ["Apple", "Banana", "Cherry"], "b": ["Dog", "Elephant", "Frog"]})
    warnings = checks.check_case_consistency(df)
    assert warnings == []


def test_check_case_consistency_detects_issues():
    df = pd.DataFrame({"category": ["apple", "APPLE", "Apple", "Banana"]})
    warnings = checks.check_case_consistency(df)
    assert len(warnings) == 1
    assert "category" in warnings[0]
    assert "mixed case" in warnings[0].lower()


def test_check_case_consistency_empty_dataframe():
    df = pd.DataFrame()
    warnings = checks.check_case_consistency(df)
    assert warnings == []


def test_check_case_consistency_multiple_columns():
    df = pd.DataFrame({"fruit": ["apple", "APPLE", "Apple", "Banana"], "animal": ["dog", "Dog", "DOG", "cat"]})
    warnings = checks.check_case_consistency(df)
    assert len(warnings) == 2
    assert any("fruit" in warning for warning in warnings)
    assert any("animal" in warning for warning in warnings)


# ==== Cardinality Tests ====


def test_check_cardinality_no_issues():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "w", "v"]})
    warnings = checks.check_cardinality(df)
    assert warnings == []


def test_check_cardinality_detects_high_cardinality():
    df = pd.DataFrame({"a": list(range(100)), "b": ["x", "y", "z", "w", "v"] * 20})
    warnings = checks.check_cardinality(df)
    assert len(warnings) == 1
    assert "Column 'a'" in warnings[0]
    assert "High Cardinality" in warnings[0]
    assert "100.0% unique" in warnings[0]
    assert "100 unique values" in warnings[0]


def test_check_cardinality_low_cardinality():
    df = pd.DataFrame({"status": ["active"] * 100})
    warnings = checks.check_cardinality(df)
    assert len(warnings) == 1
    assert "Low Cardinality" in warnings[0]
    assert "1 unique value" in warnings[0]
    assert "status" in warnings[0]


def test_check_cardinality_empty_dataframe():
    df = pd.DataFrame()
    warnings = checks.check_cardinality(df)
    assert warnings == []


def test_check_cardinality_multiple_columns():
    df = pd.DataFrame({"high_card": list(range(100)), "low_card": ["A"] * 100, "medium_card": ["A", "B"] * 50})
    warnings = checks.check_cardinality(df)
    assert len(warnings) == 2
    assert any("high_card" in warning for warning in warnings if "High Cardinality" in warning)
    assert any("low_card" in warning for warning in warnings if "Low Cardinality" in warning)


def test_check_cardinality_custom_thresholds():
    df = pd.DataFrame({"a": list(range(30)), "b": ["x", "y", "z"] * 10})
    warnings = checks.check_cardinality(df, high_threshold=25, low_threshold=2)
    assert len(warnings) == 1
    assert "Column 'a'" in warnings[0]
    assert "High Cardinality" in warnings[0]

    warnings_low = checks.check_cardinality(df, high_threshold=40, low_threshold=4)
    assert len(warnings_low) == 1
    assert "Column 'b'" in warnings_low[0]
    assert "Low Cardinality" in warnings_low[0]


def test_check_cardinality_all_nan_values():
    df = pd.DataFrame({"a": [np.nan] * 10})
    warnings = checks.check_cardinality(df)
    assert warnings == []


# ==== Skewness Tests ====


def test_check_skewness_no_skewness():
    np.random.seed(42)
    df = pd.DataFrame({"a": np.random.normal(50, 10, 1000)})
    warnings = checks.check_skewness(df)
    assert warnings == []


def test_check_skewness_detects_right_skewness():
    df = pd.DataFrame({"a": [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 100]})
    warnings = checks.check_skewness(df)
    assert len(warnings) == 1
    assert "Column 'a'" in warnings[0]
    assert "right-skewed" in warnings[0].lower()


def test_check_skewness_detects_left_skewness():
    df = pd.DataFrame({"a": [100, 95, 90, 85, 80, 75, 70, 10]})
    warnings = checks.check_skewness(df)
    assert len(warnings) == 1
    assert "Column 'a'" in warnings[0]
    assert "left-skewed" in warnings[0].lower()


def test_check_skewness_empty_dataframe():
    df = pd.DataFrame()
    warnings = checks.check_skewness(df)
    assert warnings == []


def test_check_skewness_with_nan():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5, np.nan, np.nan, 100]})
    warnings = checks.check_skewness(df)
    assert len(warnings) == 1
    assert "Column 'a'" in warnings[0]
    assert "right-skewed" in warnings[0].lower()


def test_check_skewness_with_custom_thresholds():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 20]})

    warnings_high = checks.check_skewness(df, threshold=2.0)
    assert warnings_high == []

    warnings_low = checks.check_skewness(df, threshold=0.5)
    assert len(warnings_low) == 1
    assert "Column 'a'" in warnings_low[0]
    assert "right-skewed" in warnings_low[0].lower()


# ==== Duplicate columns tests ====


def test_check_duplicate_columns_no_duplicates():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    warnings = checks.check_duplicate_columns(df)
    assert warnings == []


def test_check_duplicate_columns_detects_duplicates():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "name_dup": ["Alice", "Bob", "Charlie"],
        }
    )
    warnings = checks.check_duplicate_columns(df)
    assert len(warnings) == 1
    assert "Columns 'name' and 'name_dup'" in warnings[0]
    assert "are identical" in warnings[0]


def test_check_duplicate_columns_empty_dataframe():
    df = pd.DataFrame()
    warnings = checks.check_duplicate_columns(df)
    assert warnings == []


def test_check_duplicate_columns_multiple_duplicates():
    df = pd.DataFrame(
        {
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"],
            "col1_dup": [1, 2, 3],
            "col2_dup": ["a", "b", "c"],
        }
    )
    warnings = checks.check_duplicate_columns(df)
    assert len(warnings) == 2
    assert any("Columns 'col1' and 'col1_dup'" in warning for warning in warnings)
    assert any("Columns 'col2' and 'col2_dup'" in warning for warning in warnings)


def test_check_duplicate_columns_duplicates_with_different_dtypes():
    df = pd.DataFrame(
        {
            "col1": [1, 2, 3],
            "col2": [1.0, 2.0, 3.0],
        }
    )
    warnings = checks.check_duplicate_columns(df)
    assert warnings == []


# ==== Data Type Consistency Tests ====


def test_check_data_type_consistency_no_issues():
    df = pd.DataFrame(
        {
            "age": [25, 30, 22],
            "salary": [50000.0, 60000.0, 55000.0],
        }
    )
    warnings = checks.check_data_type_consistency(df)
    assert warnings == []


def test_check_data_type_consistency_detects_numeric_issues():
    df = pd.DataFrame(
        {
            "price": [10, "20", 30],
        }
    )
    warnings = checks.check_data_type_consistency(df)
    assert len(warnings) == 1
    assert "Column 'price'" in warnings[0]
    assert "numeric type" in warnings[0].lower()


def test_check_data_type_consistency_detects_datetime_issues():
    df = pd.DataFrame(
        {
            "start_date": ["2020-01-01", "2020-02-01", "not_a_date"],
        }
    )
    warnings = checks.check_data_type_consistency(df)
    assert len(warnings) == 1
    assert "Column 'start_date'" in warnings[0]
    assert "datetime type" in warnings[0].lower()


def test_check_data_type_consistency_detects_boolean_issues():
    df = pd.DataFrame(
        {
            "is_active": ["yes", "no", "no", "yes"],
        }
    )
    warnings = checks.check_data_type_consistency(df)
    assert len(warnings) == 1
    assert "Column 'is_active'" in warnings[0]
    assert "boolean type" in warnings[0].lower()


def test_check_data_type_consistency_empty_dataframe():
    df = pd.DataFrame()
    warnings = checks.check_data_type_consistency(df)
    assert warnings == []


def test_check_data_type_consistency_multiple_issues():
    df = pd.DataFrame(
        {
            "price": [10, "20", 30],
            "start_date": ["2020-01-01", "2020-02-01", "not_a_date"],
            "is_active": ["yes", "no", "no"],
        }
    )
    warnings = checks.check_data_type_consistency(df)
    assert len(warnings) == 3
    assert any("Column 'price'" in warning for warning in warnings)
    assert any("Column 'start_date'" in warning for warning in warnings)
    assert any("Column 'is_active'" in warning for warning in warnings)


def test_check_data_type_consistency_with_nan():
    df = pd.DataFrame(
        {
            "price": [10, np.nan, 30],
            "start_date": ["2020-01-01", np.nan, "not_a_date"],
            "is_active": [True, False, np.nan],
        }
    )
    warnings = checks.check_data_type_consistency(df)
    assert len(warnings) == 2
    assert any("Column 'start_date'" in warning for warning in warnings)
    assert any("Column 'is_active'" in warning for warning in warnings)


# ==== Negative Value Check ====


def test_check_negative_values_no_negatives():
    df = pd.DataFrame({"age": [25, 30, 35], "price": [10.0, 20.0, 30.0]})
    warnings = checks.check_negative_values(df)
    assert warnings == []


def test_check_negative_values_detects_negatives():
    df = pd.DataFrame({"age": [25, -5, 30], "balance": [100, 200, 300]})
    warnings = checks.check_negative_values(df)
    assert len(warnings) == 1
    assert "Column 'age'" in warnings[0]
    assert "1 negative value(s)" in warnings[0]


def test_check_negative_values_specific_columns():
    df = pd.DataFrame({"age": [25, -5, 30], "balance": [100, -50, 200]})
    warnings = checks.check_negative_values(df, columns=["age"])
    assert len(warnings) == 1
    assert "Column 'age'" in warnings[0]
    assert "balance" not in str(warnings)


def test_check_negative_values_empty_dataframe():
    df = pd.DataFrame()
    warnings = checks.check_negative_values(df)
    assert warnings == []


def test_check_negative_values_with_nan():
    df = pd.DataFrame({"age": [25, -5, np.nan, 30]})
    warnings = checks.check_negative_values(df)
    assert len(warnings) == 1
    assert "Column 'age'" in warnings[0]
    assert "1 negative value(s)" in warnings[0]


def test_check_negative_values_multiple_columns():
    df = pd.DataFrame({"age": [25, -5, 30], "balance": [100, -50, 200], "score": [-10, -20, -30]})
    warnings = checks.check_negative_values(df)
    assert len(warnings) == 3
    assert any("age" in warning for warning in warnings)
    assert any("balance" in warning for warning in warnings)
    assert any("score" in warning for warning in warnings)


def test_check_negative_values_non_numeric_columns():
    df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]})
    warnings = checks.check_negative_values(df)
    assert warnings == []


# ==== Rare Categories Tests ====


def test_check_rare_categories_no_rare():
    df = pd.DataFrame({"category": ["A", "B", "C"] * 30})
    warnings = checks.check_rare_categories(df)
    assert warnings == []


def test_check_rare_categories_detects_rare():
    df = pd.DataFrame({"category": ["A"] * 98 + ["B", "C"]})
    warnings = checks.check_rare_categories(df, threshold=0.02)
    assert len(warnings) == 1
    assert "Column 'category'" in warnings[0]
    assert "2 categories" in warnings[0]
    assert "<2.0%" in warnings[0]


def test_check_rare_categories_empty_dataframe():
    df = pd.DataFrame()
    warnings = checks.check_rare_categories(df)
    assert warnings == []


def test_check_rare_categories_custom_threshold():
    df = pd.DataFrame({"category": ["A"] * 90 + ["B"] * 5 + ["C"] * 5})
    warnings_1 = checks.check_rare_categories(df, threshold=0.01)
    assert warnings_1 == []

    warnings_2 = checks.check_rare_categories(df, threshold=0.06)
    assert len(warnings_2) == 1
    assert "2 categories" in warnings_2[0]


def test_check_rare_categories_with_nan():
    df = pd.DataFrame({"category": ["A"] * 95 + ["B"] * 3 + [np.nan, np.nan]})
    warnings = checks.check_rare_categories(df, threshold=0.05)
    assert len(warnings) == 1
    assert "1 categories" in warnings[0]


def test_check_rare_categories_multiple_columns():
    df = pd.DataFrame({"cat1": ["A"] * 98 + ["B", "C"], "cat2": ["X"] * 99 + ["Y"]})
    warnings = checks.check_rare_categories(df, threshold=0.015)
    assert len(warnings) == 2
    assert any("cat1" in warning for warning in warnings)
    assert any("cat2" in warning for warning in warnings)


def test_check_rare_categories_invalid_threshold():
    df = pd.DataFrame({"category": ["A", "B", "C"]})
    try:
        checks.check_rare_categories(df, threshold=1.5)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ==== Date Format Consistency Tests ====


def test_check_date_format_consistency_no_dates():
    df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"]})
    warnings = checks.check_date_format_consistency(df)
    assert warnings == []


def test_check_date_format_consistency_consistent_format():
    df = pd.DataFrame({"date": ["2020-01-01", "2020-02-01", "2020-03-01"]})
    warnings = checks.check_date_format_consistency(df)
    assert warnings == []


def test_check_date_format_consistency_detects_mixed():
    df = pd.DataFrame({"date": ["2020-01-01", "01/02/2020", "2020-03-01"]})
    warnings = checks.check_date_format_consistency(df)
    assert len(warnings) == 1
    assert "Column 'date'" in warnings[0]
    assert "inconsistent date formats" in warnings[0]


def test_check_date_format_consistency_empty_dataframe():
    df = pd.DataFrame()
    warnings = checks.check_date_format_consistency(df)
    assert warnings == []


def test_check_date_format_consistency_no_date_column_names():
    df = pd.DataFrame({"value": ["2020-01-01", "01/02/2020", "2020-03-01"]})
    warnings = checks.check_date_format_consistency(df)
    assert warnings == []


def test_check_date_format_consistency_with_nan():
    df = pd.DataFrame({"date": ["2020-01-01", np.nan, "01/02/2020", "2020-03-01"]})
    warnings = checks.check_date_format_consistency(df)
    assert len(warnings) == 1
    assert "Column 'date'" in warnings[0]


def test_check_date_format_consistency_multiple_date_columns():
    df = pd.DataFrame(
        {"start_date": ["2020-01-01", "01/02/2020"], "end_date": ["2020-03-01", "2020-04-01"], "value": [1, 2]}
    )
    warnings = checks.check_date_format_consistency(df)
    assert len(warnings) == 1
    assert "start_date" in warnings[0]


def test_check_date_format_consistency_with_time():
    df = pd.DataFrame({"timestamp": ["2020-01-01", "01/02/2020 10:30", "2020-03-01"]})
    warnings = checks.check_date_format_consistency(df)
    assert len(warnings) == 1
    assert "timestamp" in warnings[0]


# ==== String Length Outliers Tests ====


def test_check_string_length_outliers_no_outliers():
    df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie", "David"]})
    warnings = checks.check_string_length_outliers(df)
    assert warnings == []


def test_check_string_length_outliers_detects_outliers():
    df = pd.DataFrame({"email": ["a@b.com", "test@example.com", "x" * 100 + "@example.com"]})
    warnings = checks.check_string_length_outliers(df)
    assert len(warnings) == 1
    assert "Column 'email'" in warnings[0]
    assert "unusual length" in warnings[0]


def test_check_string_length_outliers_empty_dataframe():
    df = pd.DataFrame()
    warnings = checks.check_string_length_outliers(df)
    assert warnings == []


def test_check_string_length_outliers_too_few_values():
    df = pd.DataFrame({"name": ["A", "B"]})
    warnings = checks.check_string_length_outliers(df)
    assert warnings == []


def test_check_string_length_outliers_custom_threshold():
    df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie", "x" * 20]})
    warnings_high = checks.check_string_length_outliers(df, threshold=5.0)
    assert warnings_high == []

    warnings_low = checks.check_string_length_outliers(df, threshold=2.0)
    assert len(warnings_low) == 1


def test_check_string_length_outliers_with_nan():
    df = pd.DataFrame({"email": ["a@b.com", np.nan, "test@example.com", "x" * 100 + "@example.com"]})
    warnings = checks.check_string_length_outliers(df)
    assert len(warnings) == 1
    assert "Column 'email'" in warnings[0]


def test_check_string_length_outliers_constant_length():
    df = pd.DataFrame({"code": ["ABC", "DEF", "GHI", "JKL"]})
    warnings = checks.check_string_length_outliers(df)
    assert warnings == []


def test_check_string_length_outliers_multiple_columns():
    df = pd.DataFrame(
        {"email": ["a@b.com", "test@example.com", "x" * 100 + "@example.com"], "name": ["A", "Bob", "x" * 50]}
    )
    warnings = checks.check_string_length_outliers(df)
    assert len(warnings) == 2
    assert any("email" in warning for warning in warnings)
    assert any("name" in warning for warning in warnings)


def test_check_string_length_outliers_invalid_threshold():
    df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"]})
    try:
        checks.check_string_length_outliers(df, threshold=-1.0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
