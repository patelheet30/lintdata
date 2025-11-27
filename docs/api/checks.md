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

## Available Checks

::: lintdata.checks
    options:
        show_root_heading: false
        show_source: false
        heading_level: 3
