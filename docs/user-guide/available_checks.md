# Available Checks

LintData provides 22+ comprehensive data quality checks organized into categories.

## Missing Data (2 checks)

### Missing Values

Detects NaN/None values in columns.

```python
df.lint.report(checks_to_run=["missing"])
```

**Example Output:**

```
[Missing Values] Column 'age': 45 missing values (15.3%)
```

### Missing Patterns

Identifies systematic missing data across columns.

```python
df.lint.report(checks_to_run=["missing_patterns"])
```

**Example Output:**

```
[Missing Patterns] Columns 'income' and 'job' identical missing rows (likely related)
```

## Duplicate Detection (2 checks)

### Duplicate Rows

Finds exact row duplicates with specific indices.

```python
df.lint.report(checks_to_run=["duplicates"])
```

**Example Output:**

```
[Duplicate Rows] Found 23 duplicate row(s) at index: 45, 67, 89
```

### Duplicate Columns

Identifies columns with identical values.

```python
df.lint.report(checks_to_run=["duplicate_columns"])
```

**Example Output:**

```
[Duplicate Columns] Columns 'user_id' and 'customer_id' are identical
```

## Type Consistency (2 checks)

### Mixed Types

Detects columns with multiple Python types.

```python
df.lint.report(checks_to_run=["mixed_types"])
```

**Example Output:**

```
[Mixed Types] Column 'price' has mixed types: int (66%), str (33%)
```

### Data Type Consistency

Validates types match column name patterns.

```python
df.lint.report(checks_to_run=["type_consistency"])
```

**Example Output:**

```
[Type Warning] Column 'age' is stored as object, consider numeric type
```

## String Validation (4 checks)

### Whitespace

Finds leading/trailing spaces.

```python
df.lint.report(checks_to_run=["whitespace"])
```

**Example Output:**

```
[Whitespace] Column 'name' has 12 value(s) with leading or trailing whitespace
```

### Case Consistency

Detects inconsistent casing in categorical data.

```python
df.lint.report(checks_to_run=["case"])
```

**Example Output:**

```
[Case Consistency] Column 'status': mixed case detected (e.g., 'Active', 'active', 'ACTIVE')
```

### String Length Outliers

Identifies unusually long/short strings.

```python
df.lint.report(checks_to_run=["string_length"])
```

**Example Output:**

```
[String Length] Column 'email' has 3 value(s) with unusual length
```

### Special Characters

Detects encoding issues and unusual characters.

```python
df.lint.report(checks_to_run=["special_chars"])
```

**Example Output:**

```
[Special Characters] Column 'name': 15.0% of values contain special or non-standard characters
```

## Column Profiling (3 checks)

### Constant Columns

Finds zero-variance columns.

```python
df.lint.report(checks_to_run=["constant"])
```

**Example Output:**

```
[Constant Column] Column 'country' has only one unique value: 'UK'
```

### Unique Columns

Identifies potential ID columns.

```python
df.lint.report(checks_to_run=["unique"])
```

**Example Output:**

```
[Unique Column] Column 'user_id' is 99.8% unique
```

### Cardinality

Detects columns with too many/few unique values.

```python
df.lint.report(checks_to_run=["cardinality"])
```

**Example Output:**

```
[High Cardinality] Column 'user_id' has 9,847 unique values (98% unique)
[Low Cardinality] Column 'gender' has only 1 unique value
```

## Statistical Analysis (4 checks)

### Outliers (IQR Method)

Detects statistical outliers using IQR.

```python
df.lint.report(checks_to_run=["outliers"], outlier_threshold=1.5)
```

**Example Output:**

```
[Outliers] Column 'age': 7 potential outlier(s) detected (iqr method)
```

### Skewness

Identifies highly skewed distributions.

```python
df.lint.report(checks_to_run=["skewness"], skewness_threshold=1.0)
```

**Example Output:**

```
[Skewness] Column 'income' is highly right-skewed (skewness=3.2). Consider log transformation.
```

### Zero Inflation

Detects excessive zero values.

```python
df.lint.report(checks_to_run=["zero_inflation"])
```

**Example Output:**

```
[Zero Inflation] Column 'purchases': 78.0% of values are zero
```

### Correlation Warnings

Identifies highly correlated columns.

```python
df.lint.report(checks_to_run=["correlation"])
```

**Example Output:**

```
[High Correlation] Columns 'height_cm' and 'height_inches' are 99.0% correlated
```

## Numerical Validation (1 check)

### Negative Values

Detects negative values in specified columns.

```python
df.lint.report(
    checks_to_run=["negative"],
    negative_value_columns=["age", "price"]
)
```

**Example Output:**

```
[Negative Values] Column 'age' has 3 negative value(s)
```

## Categorical Data (1 check)

### Rare Categories

Finds infrequent categorical values.

```python
df.lint.report(
    checks_to_run=["rare_categories"],
    rare_category_threshold=0.01
)
```

**Example Output:**

```
[Rare Categories] Column 'country': 5 categories appear <1.0% of the time
```

## Date & Time (3 checks)

### Date Format Consistency

Detects mixed date formats.

```python
df.lint.report(checks_to_run=["date_format"])
```

**Example Output:**

```
[Date Format Consistency] Column 'date' has inconsistent date formats
```

### Future Dates

Identifies dates in the future.

```python
df.lint.report(checks_to_run=["future_dates"])
```

**Example Output:**

```
[Future Dates] Column 'birth_date' has 3 date(s) in the future compared to 2024-11-27
```

### Date Range Anomalies

Detects suspiciously wide date ranges.

```python
df.lint.report(
    checks_to_run=["date_anomalies"],
    threshold_years=50
)
```

**Example Output:**

```
[Date Range Anomalies] Column 'event_date': date range spans 129.0 years (1900-01-01 to 2029-12-31)
```

## Referential Integrity (1 check)

### Foreign Key Validation

Ensures foreign key relationships are valid.

```python
df.lint.report(
    checks_to_run=["foreign_keys"],
    foreign_key_mappings={
        "customer_id": customers_df,
        "product_id": (products_df, "product_id")
    }
)
```

**Example Output:**

```
[Foreign Key Violation] Column 'customer_id': 15 value(s) not found in referenced table

```

## Check Categories Summary

| Category     | Checks | Use Case                |
| ------------ | ------ | ----------------------- |
| Missing Data | 2      | Completeness validation |
| Duplicates   | 2      | Data deduplication      |
| Types        | 2      | Type consistency        |
| Strings      | 4      | Text quality            |
| Profiling    | 3      | Column analysis         |
| Statistical  | 4      | Numerical analysis      |
| Numerical    | 1      | Value validation        |
| Categorical  | 1      | Category analysis       |
| Dates        | 3      | Temporal validation     |
| Referential  | 1      | Foreign key integrity   |

## Running Multiple Checks

```python
# Data cleaning workflow
df.lint.report(
    checks_to_run=[
        "missing",
        "duplicates",
        "whitespace",
        "mixed_types"
    ]
)

# ML preprocessing workflow
df.lint.report(
    checks_to_run=[
        "missing",
        "outliers",
        "skewness",
        "cardinality",
        "correlation"
    ]
)

# Production validation workflow
df.lint.report(
    checks_to_run=[
        "missing",
        "duplicates",
        "type_consistency",
        "negative",
        "future_dates"
    ]
)
```

## Performance Considerations

Most checks are optimized for large datasets (millions of rows):

- **Fast:** missing, duplicates, whitespace, constant, unique
- **Medium:** outliers, skewness, cardinality, mixed_types
- **Slower:** correlation (O(n²) for n columns), missing_patterns

For very large datasets (100M+ rows), consider running checks selectively.
