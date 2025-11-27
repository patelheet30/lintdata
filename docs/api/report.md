# report() Method

Generate a comprehensive quality report for your DataFrame.

## Basic Usage

```python
import pandas as pd
import lintdata

df = pd.read_csv("data.csv")
report = df.lint.report()
print(report)
```

## Method Signature

```python
def report(
    checks_to_run: Optional[Union[List[str], str]] = None,
    outlier_threshold: float = 1.5,
    skewness_threshold: float = 1.0,
    rare_category_threshold: float = 0.01,
    unique_column_threshold: float = 0.95,
    cardinality_high_threshold: int = 50,
    cardinality_low_threshold: int = 2,
    string_length_threshold: float = 3.0,
    negative_value_columns: Optional[List[str]] = None,
    zero_inflation_threshold: float = 0.5,
    future_date_columns: Optional[List[str]] = None,
    future_date_reference: Optional[str] = None,
    special_chars_threshold: float = 0.1,
    threshold_years: float = 50,
    correlation_threshold: float = 0.95,
    foreign_key_mappings: Optional[Dict[str, Union[pd.DataFrame, Tuple[pd.DataFrame, str]]]] = None,
    report_format: str = "text",
    output: Optional[str] = None,
    return_dict: bool = False,
) -> Union[str, Dict[str, Any]]
```

## Parameters

### Check Selection

| Parameter       | Type                 | Default      | Description                                                 |
| --------------- | -------------------- | ------------ | ----------------------------------------------------------- |
| `checks_to_run` | `List[str]` or `str` | `None` (all) | Specific checks to run. Use `"all"` or list of check names. |

**Available check names:**

`missing`, `duplicates`, `mixed_types`, `whitespace`, `constant`, `unique`, `outliers`, `missing_patterns`, `case`, `cardinality`, `skewness`, `duplicate_columns`, `type_consistency`, `negative`, `rare_categories`, `date_format`, `string_length`, `zero_inflation`, `future_dates`, `special_chars`, `date_anomalies`, `correlation`, `foreign_keys`.

### Threshold Parameters

| Parameter                    | Type    | Default | Description                                            |
| ---------------------------- | ------- | ------- | ------------------------------------------------------ |
| `outlier_threshold`          | `float` | `1.5`   | IQR multiplier for outlier detection                   |
| `skewness_threshold`         | `float` | `1.0`   | Threshold for skewness detection                       |
| `rare_category_threshold`    | `float` | `0.01`  | Minimum proportion for rare categories (1%)            |
| `unique_column_threshold`    | `float` | `0.95`  | Uniqueness threshold for ID detection (95%)            |
| `cardinality_high_threshold` | `int`   | `50`    | Maximum unique values before flagging high cardinality |
| `cardinality_low_threshold`  | `int`   | `2`     | Minimum unique values before flagging low cardinality  |
| `string_length_threshold`    | `float` | `3.0`   | Standard deviations for string outlier detection       |
| `zero_inflation_threshold`   | `float` | `0.5`   | Minimum proportion of zeros to flag (50%)              |
| `special_chars_threshold`    | `float` | `0.1`   | Minimum proportion of special characters (10%)         |
| `threshold_years`            | `float` | `50`    | Maximum acceptable date range in years                 |
| `correlation_threshold`      | `float` | `0.95`  | Minimum correlation to flag (95%)                      |
| `foreign_key_mappings`       | `Dict`  | `None`  | Referential integrity mappings for foreign key checks  |

### Column-Specific Parameters

| Parameter                | Type        | Default              | Description                                   |
| ------------------------ | ----------- | -------------------- | --------------------------------------------- |
| `negative_value_columns` | `List[str]` | `None` (all numeric) | Specific columns to check for negative values |
| `future_date_columns`    | `List[str]` | `None` (all date)    | Specific columns to check for future dates    |
| `future_date_reference`  | `str`       | `None` (today)       | Reference date in YYYY-MM-DD format           |

### Output Parameters

| Parameter       | Type   | Default  | Description                                              |
| --------------- | ------ | -------- | -------------------------------------------------------- |
| `report_format` | `str`  | `"text"` | Output format: `"text"`, `"html"`, `"json"`, `"csv"`     |
| `output`        | `str`  | `None`   | File path to save report (optional)                      |
| `return_dict`   | `bool` | `False`  | Return structured dictionary instead of formatted string |

## Return Value

Returns `str` (formatted report) or `Dict[str, Any]` (structured data) depending on parameters.

### Text Format (Default)

```python
report = df.lint.report()
# Returns: Multi-line string with formatted report
```

### Structured Dictionary

```python
data = df.lint.report(return_dict=True)
# Returns: {
#   "shape": [rows, cols],
#   "issue_count": int,
#   "issues": [
#     {
#       "check": str,
#       "column": str,
#       "severity": str,
#       "message": str
#     }
#   ]
# }
```

## Usage Examples

### 1. Run All Checks (Default)

```python
report = df.lint.report()
print(report)
```

### 2. Run Specific Checks

```python
# Data cleaning checks
report = df.lint.report(
    checks_to_run=["missing", "duplicates", "whitespace"]
)

# ML preprocessing checks
report = df.lint.report(
    checks_to_run=["missing", "outliers", "skewness", "correlation"]
)
```

### 3. Custom Thresholds

```python
# More sensitive outlier detection
report = df.lint.report(
    checks_to_run=["outliers"],
    outlier_threshold=2.0  # Default is 1.5
)

# Stricter cardinality limits
report = df.lint.report(
    checks_to_run=["cardinality"],
    cardinality_high_threshold=100  # Default is 50
)
```

### 4. Column-Specific Validation

```python
# Check specific columns for negative values
report = df.lint.report(
    checks_to_run=["negative"],
    negative_value_columns=["age", "price", "quantity"]
)

# Check specific columns for future dates
report = df.lint.report(
    checks_to_run=["future_dates"],
    future_date_columns=["birth_date", "created_at"],
    future_date_reference="2024-01-01"
)
```

### 5. Export Formats

#### HTML Report

```python
# Generate and save HTML report
df.lint.report(
    report_format="html",
    output="data_quality_report.html"
)
```

#### JSON Export

```python
# Get JSON string
json_report = df.lint.report(report_format="json")

# Or save to file
df.lint.report(
    report_format="json",
    output="report.json"
)
```

#### CSV Export

```python
# Export issues as CSV
df.lint.report(
    report_format="csv",
    output="issues.csv"
)
```

#### Python Dictionary

```python
# Get structured data
data = df.lint.report(return_dict=True)

# Access components
print(f"Shape: {data['shape']}")
print(f"Issues found: {data['issue_count']}")

for issue in data['issues']:
    print(f"{issue['severity']}: {issue['message']}")
```

### 6. Complete Configuration Example

```python
report = df.lint.report(
    # Select checks
    checks_to_run=["missing", "outliers", "cardinality", "negative"],

    # Configure thresholds
    outlier_threshold=2.0,
    cardinality_high_threshold=100,
    cardinality_low_threshold=5,

    # Specify columns
    negative_value_columns=["age", "price", "balance"],

    # Output options
    report_format="html",
    output="quality_report.html"
)
```

## Common Patterns

### Quick Data Quality Check

```python
# Fast check for common issues
report = df.lint.report(
    checks_to_run=["missing", "duplicates", "whitespace", "mixed_types"]
)
```

### ML Preprocessing Validation

```python
# Checks relevant for machine learning
report = df.lint.report(
    checks_to_run=[
        "missing",
        "outliers",
        "skewness",
        "cardinality",
        "constant",
        "correlation"
    ],
    outlier_threshold=2.0,
    skewness_threshold=0.75
)
```

### Production Data Validation

```python
# Strict validation for production pipelines
def validate_production_data(df):
    data = df.lint.report(
        checks_to_run=[
            "missing",
            "duplicates",
            "type_consistency",
            "negative",
            "future_dates"
        ],
        return_dict=True
    )

    if data["issue_count"] > 0:
        raise ValueError(f"Data quality check failed: {data['issue_count']} issues")

    return True

# Use in pipeline
validate_production_data(df)
```

### Reusable Configuration

```python
# Define standard configuration
STANDARD_CONFIG = {
    "checks_to_run": ["missing", "duplicates", "outliers"],
    "outlier_threshold": 2.0,
    "report_format": "html"
}

# Apply to multiple DataFrames
for name, df in dataframes.items():
    df.lint.report(**STANDARD_CONFIG, output=f"report_{name}.html")
```

## Tips

### Performance

For very large DataFrames (100M+ rows), run checks selectively:

```python
# Fast checks first
quick_report = df.lint.report(
    checks_to_run=["missing", "duplicates", "constant"]
)

# Then run slower checks if needed
detailed_report = df.lint.report(
    checks_to_run=["correlation", "missing_patterns"]
)
```

### Adjusting Sensitivity

Different use cases need different sensitivity:

```python
# Exploratory analysis (permissive)
df.lint.report(outlier_threshold=3.0, rare_category_threshold=0.001)

# Production validation (strict)
df.lint.report(outlier_threshold=1.5, rare_category_threshold=0.05)
```

### Integration with pandas

```python
# Method chaining
result = (
    pd.read_csv("data.csv")
    .assign(clean_name=lambda x: x["name"].str.strip())
    .pipe(lambda x: x if x.lint.report(return_dict=True)["issue_count"] == 0 else None)
)
```

## See Also

- [Available Checks](../user-guide/available_checks.md) - Details on all 22+ checks
- [Custom Checks](../user-guide/custom_checks.md) - Writing custom validation logic
