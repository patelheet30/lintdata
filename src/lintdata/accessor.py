"""
Implements the core LintData accessor for pandas Dataframes
"""

from typing import List

import pandas as pd

from . import checks


@pd.api.extensions.register_dataframe_accessor("lint")
class LintAccessor:
    def __init__(self, pandas_obj: pd.DataFrame) -> None:
        self._validate(pandas_obj)
        self._df = pandas_obj

    @staticmethod
    def _validate(obj: pd.DataFrame) -> None:
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError(
                "LintData accessor can only be used with pandas DataFrames."
            )

    def report(self) -> str:
        report_lines = ["--- LintData Quality Report ---"]

        if self._df.empty:
            report_lines.append("The DataFrame is empty. No checks run.")
            return "\n".join(report_lines)

        report_lines.append(f"Shape: {self._df.shape}")
        report_lines.append("\nRunning checks...")

        all_warnings: List[str] = []
        all_warnings.extend(checks.check_missing_values(self._df))
        all_warnings.extend(checks.check_duplicate_rows(self._df))
        all_warnings.extend(checks.check_mixed_types(self._df))
        all_warnings.extend(checks.check_whitespace(self._df))
        all_warnings.extend(checks.check_constant_columns(self._df))
        all_warnings.extend(checks.check_unique_columns(self._df))
        all_warnings.extend(checks.check_outliers(self._df))

        if not all_warnings:
            report_lines.append("No issues found. DataFrame looks good!")
        else:
            report_lines.append(f"Found {len(all_warnings)} issue(s):")
            for i, warning in enumerate(all_warnings, 1):
                report_lines.append(f"  {i}. {warning}")

        report_lines.append("\n--- End of Report ---")
        return "\n".join(report_lines)
