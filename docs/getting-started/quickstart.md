# Quick Start

Get up and running with LintData in 5 minutes.

## Basic Usage

```python
import pandas as pd
import lintdata

# Load your DataFrame
df = pd.read_csv("data.csv")

# Run all quality checks
report = df.lint.report()
print(report)
```

That's it! LintData will analyse your DataFrame and provide a comprehensive quality report.

## Understanding the Report

The text report shows:

1. **DataFrame shape** - Number of rows and columns
2. **Issue count** - Total number of problems found
3. **Detailed warnings** - Specific issues with actionable information

Example output:

```
--- LintData Quality Report ---
Shape: (1000, 8)

Running checks...
Found 4 issue(s):
  1. [Missing Values] Column 'age': 45 missing values (4.5%)
  2. [Duplicate Rows] Found 3 duplicate row(s) at index: 23, 45, 67
  3. [Outliers] Column 'salary': 12 potential outlier(s) detected (iqr method)
  4. [Whitespace] Column 'name' has 8 value(s) with leading or trailing whitespace

--- End of Report ---
```

## Running Specific Checks

Run only the checks you need:

```python
# Check only for missing values and duplicates
report = df.lint.report(checks_to_run=["missing", "duplicates"])
```

Available check names:

- `missing` - Missing values
- `duplicates` - Duplicate rows
- `mixed_types` - Mixed data types
- `whitespace` - Leading/trailing whitespace
- `constant` - Constant columns
- `unique` - High uniqueness columns
- `outliers` - Statistical outliers
- `case` - Case consistency
- And [many more](../user-guide/available_checks.md)...

## Customising Thresholds

Adjust sensitivity for your use case:

```python
# More sensitive outlier detection
report = df.lint.report(
    checks_to_run=["outliers"],
    outlier_threshold=2.0  # Default is 1.5
)

# Higher cardinality threshold
report = df.lint.report(
    checks_to_run=["cardinality"],
    cardinality_high_threshold=100  # Default is 50
)
```

## Export Formats

### HTML Report

Generate an interactive HTML report:

```python
# Save to file
df.lint.report(
    report_format="html",
    output="data_quality_report.html"
)
```

Opens a beautifully styled report with:

- Colour-coded severity levels
- Responsive design
- Summary statistics

### JSON Export

Get structured data for programmatic use:

```python
# Get as string
json_report = df.lint.report(report_format="json")

# Save to file
df.lint.report(
    report_format="json",
    output="report.json"
)
```

### CSV Export

Export issues as a CSV for spreadsheet analysis:

```python
df.lint.report(
    report_format="csv",
    output="issues.csv"
)
```

### Python Dictionary

Get structured data directly:

```python
data = df.lint.report(return_dict=True)

print(data["shape"])        # [1000, 8]
print(data["issue_count"])  # 4
print(data["issues"])       # List of issue dicts
```

## Common Workflows

### Before Analysis

```python
import pandas as pd
import lintdata

df = pd.read_csv("survey_data.csv")

# Quick quality check
report = df.lint.report()
print(report)

# Fix issues if needed
# ... data cleaning ...
```

### Before ML Training

```python
# Check for issues that affect models
report = df.lint.report(
    checks_to_run=["missing", "outliers", "skewness", "cardinality"]
)

# Address issues
# ... preprocessing ...
```

### Automated Validation

```python
def validate_data(df, max_issues=5):
    """Validate DataFrame meets quality standards."""
    data = df.lint.report(return_dict=True)

    if data["issue_count"] > max_issues:
        raise ValueError(f"Data quality check failed: {data['issue_count']} issues found")

    return True

# Use in pipeline
validate_data(df)
```

## Next Steps

- [Learn about all available checks](../user-guide/available_checks.md)
