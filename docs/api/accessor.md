# LintAccessor

The `LintAccessor` class provides the `.lint` accessor for pandas DataFrames.

::: lintdata.accessor.LintAccessor
options:
show_root_heading: true
show_source: false
members: - report - register_check - unregister_check - list_custom_checks
group_by_category: true
show_signature_annotations: true
separate_signature: true

## Usage Examples

### Basic Report

```python
import pandas as pd
import lintdata

df = pd.read_csv("data.csv")
report = df.lint.report()
print(report)
```

### Custom Configuration

```python
# Run specific checks with custom thresholds
report = df.lint.report(
    checks_to_run=["missing", "outliers", "duplicates"],
    outlier_threshold=2.0,
    skewness_threshold=0.5
)
```

### Export Formats

```python
# HTML report
df.lint.report(
    report_format="html",
    output="report.html"
)

# JSON export
json_data = df.lint.report(report_format="json")

# CSV export
df.lint.report(
    report_format="csv",
    output="issues.csv"
)

# Python dictionary
data = df.lint.report(return_dict=True)
```

### Custom Checks

```python
# Define custom check
def check_email_format(df):
    warnings = []
    for col in df.select_dtypes(include="object").columns:
        if "email" in col.lower():
            invalid = df[~df[col].str.contains("@", na=False)]
            if len(invalid) > 0:
                warnings.append(
                    f"[Email] Column '{col}': {len(invalid)} invalid email(s)"
                )
    return warnings

# Register and use
df.lint.register_check(check_email_format)
report = df.lint.report()
```

### Validation Pipeline

```python
def validate_dataset(df, config):
    """Validate DataFrame against quality standards."""
    data = df.lint.report(
        checks_to_run=config["checks"],
        outlier_threshold=config["outlier_threshold"],
        return_dict=True
    )

    if data["issue_count"] > config["max_issues"]:
        raise ValueError(
            f"Quality check failed: {data['issue_count']} issues found"
        )

    return data

# Use in pipeline
config = {
    "checks": ["missing", "duplicates", "outliers"],
    "outlier_threshold": 2.0,
    "max_issues": 5
}

result = validate_dataset(df, config)
```

## Parameter Reference

### Check Selection

- `checks_to_run`: List of check names or `"all"`
- Available checks: `missing`, `duplicates`, `mixed_types`, `whitespace`, `constant`, `unique`, `outliers`, `missing_patterns`, `case`, `cardinality`, `skewness`, `duplicate_columns`, `type_consistency`, `negative`, `rare_categories`, `date_format`, `string_length`, `zero_inflation`, `future_dates`, `special_chars`, `date_anomalies`, `correlation`

### Threshold Parameters

- `outlier_threshold` (float): IQR multiplier for outliers (default: 1.5)
- `skewness_threshold` (float): Skewness detection threshold (default: 1.0)
- `rare_category_threshold` (float): Proportion for rare categories (default: 0.01)
- `unique_column_threshold` (float): Uniqueness threshold (default: 0.95)
- `cardinality_high_threshold` (int): High cardinality limit (default: 50)
- `cardinality_low_threshold` (int): Low cardinality limit (default: 2)
- `string_length_threshold` (float): String outlier threshold (default: 3.0)
- `zero_inflation_threshold` (float): Zero proportion threshold (default: 0.5)
- `special_chars_threshold` (float): Special char proportion (default: 0.1)
- `threshold_years` (float): Date range threshold in years (default: 50)
- `correlation_threshold` (float): Correlation threshold (default: 0.95)

### Column-Specific Parameters

- `negative_value_columns` (List[str]): Columns to check for negatives
- `future_date_columns` (List[str]): Columns to check for future dates
- `future_date_reference` (str): Reference date (YYYY-MM-DD)

### Output Parameters

- `report_format` (str): Output format - `"text"`, `"html"`, `"json"`, `"csv"`
- `output` (str): File path to save report
- `return_dict` (bool): Return structured dictionary

## Design Patterns

### Reusable Configuration

```python
# Define standard config
STANDARD_CONFIG = {
    "checks_to_run": ["missing", "duplicates", "outliers"],
    "outlier_threshold": 2.0,
    "report_format": "html"
}

# Apply to multiple DataFrames
for df in dataframes:
    df.lint.report(**STANDARD_CONFIG, output=f"report_{df.name}.html")
```

### Conditional Validation

```python
# Validate only if conditions met
if df.shape[0] > 1000:  # Large dataset
    report = df.lint.report(
        checks_to_run=["missing", "duplicates"],
        cardinality_high_threshold=100
    )
```

### Integration with pandas Method Chaining

```python
result = (
    pd.read_csv("data.csv")
    .assign(clean_name=lambda x: x["name"].str.strip())
    .pipe(lambda x: x if x.lint.report(return_dict=True)["issue_count"] == 0 else None)
)
```
