# Checks Module

The `checks` module contains all data quality validation functions.

## Overview

Each check function:

- Accepts a pandas DataFrame
- Returns a list of warning strings
- Returns empty list `[]` if no issues found
- Can be called directly or via the `.lint.report()` accessor

```python
from lintdata import checks
import pandas as pd

df = pd.DataFrame({"age": [25, 30, None]})
warnings = checks.check_missing_values(df)
# Returns: ["[Missing Values] Column 'age': 1 missing values (33.3%)"]
```

## Check Categories

### Missing Data

::: lintdata.checks.check_missing_values
options:
show_root_heading: false
show_source: false
heading_level: 3

::: lintdata.checks.check_missing_patterns
options:
show_root_heading: false
show_source: false
heading_level: 3

### Duplicate Detection

::: lintdata.checks.check_duplicate_rows
options:
show_root_heading: false
show_source: false
heading_level: 3

::: lintdata.checks.check_duplicate_columns
options:
show_root_heading: false
show_source: false
heading_level: 3

### Type Consistency

::: lintdata.checks.check_mixed_types
options:
show_root_heading: false
show_source: false
heading_level: 3

::: lintdata.checks.check_data_type_consistency
options:
show_root_heading: false
show_source: false
heading_level: 3

### String Validation

::: lintdata.checks.check_whitespace
options:
show_root_heading: false
show_source: false
heading_level: 3

::: lintdata.checks.check_case_consistency
options:
show_root_heading: false
show_source: false
heading_level: 3

::: lintdata.checks.check_string_length_outliers
options:
show_root_heading: false
show_source: false
heading_level: 3

::: lintdata.checks.check_special_characters
options:
show_root_heading: false
show_source: false
heading_level: 3

### Column Profiling

::: lintdata.checks.check_constant_columns
options:
show_root_heading: false
show_source: false
heading_level: 3

::: lintdata.checks.check_unique_columns
options:
show_root_heading: false
show_source: false
heading_level: 3

::: lintdata.checks.check_cardinality
options:
show_root_heading: false
show_source: false
heading_level: 3

### Statistical Analysis

::: lintdata.checks.check_outliers
options:
show_root_heading: false
show_source: false
heading_level: 3

::: lintdata.checks.check_skewness
options:
show_root_heading: false
show_source: false
heading_level: 3

::: lintdata.checks.check_zero_inflation
options:
show_root_heading: false
show_source: false
heading_level: 3

::: lintdata.checks.check_correlation_warnings
options:
show_root_heading: false
show_source: false
heading_level: 3

### Numerical Validation

::: lintdata.checks.check_negative_values
options:
show_root_heading: false
show_source: false
heading_level: 3

### Categorical Data

::: lintdata.checks.check_rare_categories
options:
show_root_heading: false
show_source: false
heading_level: 3

### Date & Time

::: lintdata.checks.check_date_format_consistency
options:
show_root_heading: false
show_source: false
heading_level: 3

::: lintdata.checks.check_future_dates
options:
show_root_heading: false
show_source: false
heading_level: 3

::: lintdata.checks.check_date_range_anomalies
options:
show_root_heading: false
show_source: false
heading_level: 3
