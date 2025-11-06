# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.5.0] - 2025-11-03 - 2025-11-06

### Added

- `check_cardinality`: New check to identify columns with high cardinality.
- `check_skewness`: New check to assess skewness in numerical columns.
- `check_duplicate_columns`: New check to identify duplicate columns in DataFrames.
- `check_data_type_consistency`: New check to identify columns with inconsistent data types.

## [0.4.0] - 2025-10-31 - 2025-11-02

### Added

- `check_outliers`: New check to identify outliers using the IQR method.
- `check_missing_patterns`: New check to detect common missing value patterns in DataFrames.
- `check_case_consistency`: New check to identify inconsistencies in string casing within columns.

## [0.3.0] - 2025-10-31

### Added

- `check_constant_columns`: New check to identify columns with constant values.
- `check_unique_columns`: New check to identify columns where all values are unique.

## [0.2.0] - 2025-10-27 - 2025-10-31

### Added

- `check_duplicate_rows`: New check to identify duplicate rows in DataFrames.
- `check_mixed_types`: New check to detect columns with mixed data types.
- `check_whitespace`: New check to identify leading or trailing whitespace in string columns.

## [0.1.0] - 2025-10-24

### Added

- Initial release of LintData.
- Basic check for missing values in DataFrames.
