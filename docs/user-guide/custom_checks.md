# Custom Checks Guide

LintData allows you to extend its validation capabilities with your own custom checks. This guide shows you how to create, register, and use custom validation logic.

## Quick Start

```python
import pandas as pd
import lintdata

# Define your custom check
def check_email_format(df):
    """Validate email addresses."""
    warnings = []

    for col in df.select_dtypes(include="object").columns:
        if "email" in col.lower():
            # Check for @ symbol
            invalid = df[~df[col].str.contains("@", na=False)]
            if len(invalid) > 0:
                warnings.append(
                    f"[Email] Column '{col}': {len(invalid)} invalid email(s)"
                )

    return warnings

# Register the check
df = pd.read_csv("users.csv")
df.lint.register_check(check_email_format)

# Run all checks (including your custom one)
report = df.lint.report()
print(report)
```

## Custom Check Basics

### Function Signature

Custom checks must follow this pattern:

```python
def check_name(df: pd.DataFrame) -> List[str]:
    """
    Brief description of what this check does.

    Args:
        df: The pandas DataFrame to validate

    Returns:
        List of warning strings (empty if no issues)
    """
    warnings: List[str] = []

    # Your validation logic here

    return warnings
```

### Key Requirements

1. **Accept a DataFrame** - First parameter must be `pd.DataFrame`
2. **Return List[str]** - Return list of warning strings
3. **Return empty list** - If no issues found, return `[]`
4. **Format warnings** - Use format: `"[Check Name] Column 'name': description"`

## Registration API

### register_check()

Register a custom check function.

```python
df.lint.register_check(
    check_func,           # Your function
    name="custom_name"    # Optional: custom name
)
```

**Parameters:**

- `check_func` - Callable function that accepts DataFrame
- `name` - Optional name for the check (defaults to function name)

**Example:**

```python
def my_validation(df):
    return []

# Register with default name
df.lint.register_check(my_validation)
# Registered as "my_validation"

# Register with custom name
df.lint.register_check(my_validation, name="special_validation")
# Registered as "special_validation"
```

### unregister_check()

Remove a registered custom check.

```python
df.lint.unregister_check("check_name")
```

### list_custom_checks()

List all registered custom checks.

```python
checks = df.lint.list_custom_checks()
print(checks)  # ['check_email_format', 'check_phone_number']
```

## Writing Effective Custom Checks

### Pattern 1: Column Name Detection

Validate specific columns based on their names.

```python
def check_email_format(df):
    """Validate email format in columns with 'email' in the name."""
    warnings = []

    # Find email columns
    email_cols = [col for col in df.columns if "email" in col.lower()]

    for col in email_cols:
        # Simple regex for email validation
        pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        invalid = df[~df[col].str.match(pattern, na=False)]

        if len(invalid) > 0:
            warnings.append(
                f"[Email Format] Column '{col}': {len(invalid)} invalid email(s)"
            )

    return warnings
```

### Pattern 2: Value Range Validation

Check if values fall within acceptable ranges.

```python
def check_age_range(df):
    """Validate age values are between 0 and 120."""
    warnings = []

    if "age" in df.columns:
        invalid = df[(df["age"] < 0) | (df["age"] > 120)]

        if len(invalid) > 0:
            warnings.append(
                f"[Age Range] Column 'age': {len(invalid)} value(s) outside valid range (0-120)"
            )

    return warnings
```

### Pattern 3: Cross-Column Validation

Validate relationships between columns.

```python
def check_date_consistency(df):
    """Ensure start_date is before end_date."""
    warnings = []

    if "start_date" in df.columns and "end_date" in df.columns:
        # Convert to datetime if needed
        start = pd.to_datetime(df["start_date"], errors="coerce")
        end = pd.to_datetime(df["end_date"], errors="coerce")

        # Find violations
        invalid = df[start > end]

        if len(invalid) > 0:
            indices = invalid.index.tolist()
            warnings.append(
                f"[Date Consistency] {len(invalid)} row(s) where start_date > end_date at index: {', '.join(map(str, indices[:5]))}"
            )

    return warnings
```

### Pattern 4: Business Logic Validation

Implement domain-specific rules.

```python
def check_order_total(df):
    """Validate order total equals sum of line items."""
    warnings = []

    required_cols = ["quantity", "unit_price", "total"]
    if all(col in df.columns for col in required_cols):
        # Calculate expected total
        expected = df["quantity"] * df["unit_price"]

        # Allow small floating point differences
        mismatch = df[abs(df["total"] - expected) > 0.01]

        if len(mismatch) > 0:
            warnings.append(
                f"[Order Total] {len(mismatch)} row(s) with incorrect total calculation"
            )

    return warnings
```

### Pattern 5: Data Type Validation

Ensure columns contain expected types.

```python
def check_numeric_strings(df):
    """Check if string columns contain numeric-looking data."""
    warnings = []

    for col in df.select_dtypes(include="object").columns:
        # Skip obviously non-numeric columns
        if any(keyword in col.lower() for keyword in ["name", "text", "description"]):
            continue

        # Try converting to numeric
        numeric_count = pd.to_numeric(df[col], errors="coerce").notna().sum()
        total = df[col].notna().sum()

        if total > 0 and numeric_count / total > 0.9:
            warnings.append(
                f"[Numeric Strings] Column '{col}': {numeric_count}/{total} values are numeric but stored as string"
            )

    return warnings
```

## Advanced Examples

### Example 1: Parameterised Checks

Create checks that accept configuration parameters.

```python
def check_price_range(df, min_price=0, max_price=10000):
    """Validate prices fall within expected range."""
    warnings = []

    price_cols = [col for col in df.columns if "price" in col.lower()]

    for col in price_cols:
        invalid = df[(df[col] < min_price) | (df[col] > max_price)]

        if len(invalid) > 0:
            warnings.append(
                f"[Price Range] Column '{col}': {len(invalid)} value(s) outside range [{min_price}, {max_price}]"
            )

    return warnings

# Register with parameters using lambda
df.lint.register_check(
    lambda d: check_price_range(d, min_price=10, max_price=5000),
    name="check_price_range"
)
```

### Example 2: Multiple Check Conditions

Combine multiple validation rules in one check.

```python
def check_customer_data(df):
    """Comprehensive customer data validation."""
    warnings = []

    # Check 1: Email format
    if "email" in df.columns:
        invalid_email = df[~df["email"].str.contains("@", na=False)]
        if len(invalid_email) > 0:
            warnings.append(
                f"[Customer Data] {len(invalid_email)} invalid email(s)"
            )

    # Check 2: Phone format (US)
    if "phone" in df.columns:
        pattern = r'^\d{3}-\d{3}-\d{4}$'
        invalid_phone = df[~df["phone"].str.match(pattern, na=False)]
        if len(invalid_phone) > 0:
            warnings.append(
                f"[Customer Data] {len(invalid_phone)} invalid phone number(s)"
            )

    # Check 3: Age range
    if "age" in df.columns:
        invalid_age = df[(df["age"] < 18) | (df["age"] > 120)]
        if len(invalid_age) > 0:
            warnings.append(
                f"[Customer Data] {len(invalid_age)} age value(s) outside valid range"
            )

    return warnings
```

### Example 3: Statistical Validation

Custom statistical checks beyond built-ins.

```python
def check_distribution_shift(df, reference_col="category", expected_dist=None):
    """Check if categorical distribution matches expected."""
    warnings = []

    if reference_col not in df.columns or expected_dist is None:
        return warnings

    # Calculate actual distribution
    actual_dist = df[reference_col].value_counts(normalize=True).to_dict()

    # Compare to expected
    for category, expected_prop in expected_dist.items():
        actual_prop = actual_dist.get(category, 0)
        diff = abs(actual_prop - expected_prop)

        # Flag if difference > 10%
        if diff > 0.10:
            warnings.append(
                f"[Distribution Shift] Category '{category}': {actual_prop:.1%} (expected {expected_prop:.1%})"
            )

    return warnings

# Use with expected distribution
expected = {"A": 0.5, "B": 0.3, "C": 0.2}
df.lint.register_check(
    lambda d: check_distribution_shift(d, expected_dist=expected),
    name="check_distribution"
)
```

### Example 4: External Data Validation

Validate against external reference data.

```python
def check_country_codes(df, valid_codes_file="valid_countries.txt"):
    """Validate country codes against official list."""
    warnings = []

    # Load valid codes
    with open(valid_codes_file, "r") as f:
        valid_codes = set(line.strip() for line in f)

    # Find country columns
    country_cols = [col for col in df.columns if "country" in col.lower()]

    for col in country_cols:
        invalid = df[~df[col].isin(valid_codes)]

        if len(invalid) > 0:
            invalid_codes = invalid[col].unique()
            warnings.append(
                f"[Country Code] Column '{col}': {len(invalid)} invalid code(s) - {', '.join(invalid_codes[:5])}"
            )

    return warnings
```

## Error Handling

LintData handles errors in custom checks gracefully.

### Catching Errors

```python
def risky_check(df):
    """A check that might fail."""
    warnings = []

    try:
        # Risky operation
        result = df["column_that_might_not_exist"].sum()
        if result < 0:
            warnings.append("[Risky] Negative sum detected")
    except KeyError:
        # Handle missing column
        warnings.append("[Risky] Required column not found")
    except Exception as e:
        # Handle other errors
        warnings.append(f"[Risky] Error during check: {str(e)}")

    return warnings
```

### Built-in Error Handling

If your check raises an exception, LintData catches it:

```python
def broken_check(df):
    raise ValueError("Something went wrong!")

df.lint.register_check(broken_check)
report = df.lint.report()

# Output includes:
# [Custom Check Error] 'broken_check' raised an exception: Something went wrong!
```

### Validation of Return Type

If your check returns something other than a list:

```python
def bad_check(df):
    return "not a list"  # Wrong!

df.lint.register_check(bad_check)
report = df.lint.report()

# Output includes:
# [Custom Check Error] 'bad_check' did not return a list of warnings
```

## Best Practices

### 1. Handle Edge Cases

```python
def robust_check(df):
    """A check that handles edge cases."""
    warnings = []

    # Check if DataFrame is empty
    if df.empty:
        return warnings

    # Check if column exists
    if "target_col" not in df.columns:
        return warnings

    # Check if column has any non-null values
    if df["target_col"].isna().all():
        return warnings

    # Now safe to validate
    # ... validation logic ...

    return warnings
```

### 2. Provide Specific Information

```python
# ❌ Bad: Vague message
def bad_check(df):
    if some_condition:
        return ["Data quality issue"]

# ✅ Good: Specific and actionable
def good_check(df):
    warnings = []
    if "email" in df.columns:
        invalid = df[~df["email"].str.contains("@", na=False)]
        if len(invalid) > 0:
            indices = invalid.index.tolist()[:5]
            warnings.append(
                f"[Email] Column 'email': {len(invalid)} invalid email(s) at index: {', '.join(map(str, indices))}"
            )
    return warnings
```

### 3. Use Consistent Formatting

```python
def well_formatted_check(df):
    """Follow the standard warning format."""
    warnings = []

    # Format: [Check Type] Column 'name': description with details
    warnings.append(
        f"[Custom Check] Column '{col}': {count} issue(s) found"
    )

    return warnings
```

### 4. Document Your Checks

```python
def check_with_docs(df):
    """
    Validate customer loyalty points.

    Checks:
    - Points must be non-negative
    - Points must be less than 1,000,000
    - Points must be integer values

    Args:
        df: DataFrame containing 'loyalty_points' column

    Returns:
        List of warnings for violations

    Example:
        >>> df = pd.DataFrame({'loyalty_points': [100, -50, 1500000]})
        >>> warnings = check_with_docs(df)
        >>> print(warnings[0])
        [Loyalty Points] Column 'loyalty_points': 1 negative value(s)
    """
    warnings = []

    if "loyalty_points" not in df.columns:
        return warnings

    # Validation logic...

    return warnings
```

### 5. Performance Considerations

```python
def optimised_check(df):
    """Use vectorised operations for performance."""
    warnings = []

    if "price" not in df.columns:
        return warnings

    # ✅ Good: Vectorised operation
    invalid = df[(df["price"] < 0) | (df["price"] > 10000)]

    # ❌ Bad: Row-by-row iteration
    # for idx, row in df.iterrows():
    #     if row["price"] < 0 or row["price"] > 10000:
    #         invalid_indices.append(idx)

    if len(invalid) > 0:
        warnings.append(f"[Price] {len(invalid)} invalid price(s)")

    return warnings
```

## Integration Patterns

### Pattern 1: Conditional Registration

Register checks based on DataFrame contents.

```python
def register_relevant_checks(df):
    """Register checks based on columns present."""

    if "email" in df.columns:
        df.lint.register_check(check_email_format)

    if "phone" in df.columns:
        df.lint.register_check(check_phone_format)

    if all(col in df.columns for col in ["start_date", "end_date"]):
        df.lint.register_check(check_date_consistency)

# Use in pipeline
df = pd.read_csv("data.csv")
register_relevant_checks(df)
report = df.lint.report()
```

### Pattern 2: Check Library

Create a reusable library of checks.

```python
# checks_library.py
def check_email_format(df):
    """Validate email format."""
    # ... implementation ...
    return warnings

def check_phone_format(df):
    """Validate phone format."""
    # ... implementation ...
    return warnings

def check_postal_code(df):
    """Validate postal codes."""
    # ... implementation ...
    return warnings

# Usage
from checks_library import check_email_format, check_phone_format

df.lint.register_check(check_email_format)
df.lint.register_check(check_phone_format)
```

## See Also

- [Available Checks](available_checks.md) - Built-in checks you can use
- [API Reference](../api/accessor.md) - Complete API documentation
