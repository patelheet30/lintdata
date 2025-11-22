"""
Implements the core LintData accessor for pandas Dataframes
"""

from typing import List, Optional, Union

import pandas as pd

from . import checks
from .report_formatter import HTMLReportFormatter

__all__ = ["LintAccessor"]


@pd.api.extensions.register_dataframe_accessor("lint")
class LintAccessor:
    """An Accessor for pandas DataFrames to run data quality checks."""

    def __init__(self, pandas_obj: pd.DataFrame) -> None:
        self._validate(pandas_obj)
        self._df = pandas_obj

    @staticmethod
    def _validate(obj: pd.DataFrame) -> None:
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("LintData accessor can only be used with pandas DataFrames.")

    def report(
        self,
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
        format: str = "text",
        output: Optional[str] = None,
    ) -> str:
        """Generate a comprehensive quality report for the DataFrame.

        Args:
            checks_to_run (Optional[Union[List[str], str]], optional): Specific checks to run.
                Options: 'missing', 'duplicates', 'mixed_types', 'whitespace', 'constant',
                'unique', 'outliers', 'missing_patterns', 'case', 'cardinality', 'skewness',
                'duplicate_columns', 'type_consistency', 'negative', 'rare_categories',
                'date_format', 'string_length', 'zero_inflation', 'future_dates',
                'special_chars'. Use 'all' to run all checks. Defaults to None.
            outlier_threshold (float, optional): Outlier detection threshold using the IQR method. Defaults to 1.5.
            skewness_threshold (float, optional): Threshold for skewness detection. Defaults to 1.0.
            rare_category_threshold (float, optional): Minimum proportion for rare categories. Defaults to 0.01.
            unique_column_threshold (float, optional): Threshold for identifying unique columns. Defaults to 0.95.
            cardinality_high_threshold (int, optional): High cardinality threshold. Defaults to 50.
            cardinality_low_threshold (int, optional): Low cardinality threshold. Defaults to 2.
            string_length_threshold (float, optional): Threshold for identifying string length outliers. Defaults to 3.0.
            negative_value_columns (Optional[List[str]], optional): Specific columns to check for negative values. Defaults to None.
            zero_inflation_threshold (float, optional): Minimum proportion of zeros to flag. Defaults to 0.5.
            future_date_columns (Optional[List[str]], optional): Specific columns to check for future dates. Defaults to None.
            future_date_reference (Optional[str], optional): Reference date for future date check (YYYY-MM-DD). Defaults to None (today).
            special_chars_threshold (float, optional): Minimum proportion of values with special characters. Defaults to 0.1.
            format (str, optional): Output format. Options: 'text', 'html'. Defaults to 'text'.
            output (Optional[str], optional): File path to save the report. If None, returns as string. Defaults to None.

        Raises:
            ValueError: If invalid check names are provided or invalid format specified.

        Returns:
            str: A comprehensive quality report for the DataFrame in the specified format.

        Example:
            >>> # Text report
            >>> report = df.lint.report()
            >>> print(report)

            >>> # HTML report saved to file
            >>> df.lint.report(format='html', output='report.html')

            >>> # HTML report as string
            >>> html_report = df.lint.report(format='html')
        """
        valid_formats = ["text", "html"]
        if format not in valid_formats:
            raise ValueError(f"Invalid format '{format}'. Valid options: {valid_formats}")

        if self._df.empty:
            empty_message = "The DataFrame is empty. No checks run."
            if format == "text":
                result = f"--- LintData Quality Report ---\n{empty_message}"
            else:
                result = HTMLReportFormatter.generate((0, 0), [])

            if output:
                with open(output, "w", encoding="utf-8") as f:
                    f.write(result)

            return result

        if checks_to_run == "all":
            checks_to_run = None
        elif isinstance(checks_to_run, str):
            checks_to_run = [checks_to_run]

        available_checks = {
            "missing": lambda: checks.check_missing_values(self._df),
            "duplicates": lambda: checks.check_duplicate_rows(self._df),
            "mixed_types": lambda: checks.check_mixed_types(self._df),
            "whitespace": lambda: checks.check_whitespace(self._df),
            "constant": lambda: checks.check_constant_columns(self._df),
            "unique": lambda: checks.check_unique_columns(self._df, threshold=unique_column_threshold),
            "outliers": lambda: checks.check_outliers(self._df, threshold=outlier_threshold),
            "missing_patterns": lambda: checks.check_missing_patterns(self._df),
            "case": lambda: checks.check_case_consistency(self._df),
            "cardinality": lambda: checks.check_cardinality(
                self._df, high_threshold=cardinality_high_threshold, low_threshold=cardinality_low_threshold
            ),
            "skewness": lambda: checks.check_skewness(self._df, threshold=skewness_threshold),
            "duplicate_columns": lambda: checks.check_duplicate_columns(self._df),
            "type_consistency": lambda: checks.check_data_type_consistency(self._df),
            "negative": lambda: checks.check_negative_values(self._df, columns=negative_value_columns),
            "rare_categories": lambda: checks.check_rare_categories(self._df, threshold=rare_category_threshold),
            "date_format": lambda: checks.check_date_format_consistency(self._df),
            "string_length": lambda: checks.check_string_length_outliers(self._df, threshold=string_length_threshold),
            "zero_inflation": lambda: checks.check_zero_inflation(self._df, threshold=zero_inflation_threshold),
            "future_dates": lambda: checks.check_future_dates(
                self._df, columns=future_date_columns, reference_date=future_date_reference
            ),
            "special_chars": lambda: checks.check_special_characters(self._df, threshold=special_chars_threshold),
        }

        if checks_to_run is None:
            checks_to_execute = available_checks.keys()
        else:
            invalid_checks = [c for c in checks_to_run if c not in available_checks]
            if invalid_checks:
                raise ValueError(f"Invalid check(s): {invalid_checks}. Valid options: {list(available_checks.keys())}")
            checks_to_execute = checks_to_run

        all_warnings: List[str] = []
        for check_name in checks_to_execute:
            all_warnings.extend(available_checks[check_name]())

        if format == "text":
            report_lines = ["--- LintData Quality Report ---"]
            report_lines.append(f"Shape: {self._df.shape}")
            report_lines.append("\nRunning checks...")

            if not all_warnings:
                report_lines.append("No issues found. DataFrame looks good!")
            else:
                report_lines.append(f"Found {len(all_warnings)} issue(s):")
                for i, warning in enumerate(all_warnings, 1):
                    report_lines.append(f"  {i}. {warning}")

            report_lines.append("\n--- End of Report ---")
            result = "\n".join(report_lines)

        elif format == "html":
            result = HTMLReportFormatter.generate(self._df.shape, all_warnings)

        # Save to file if output path provided
        if output:
            with open(output, "w", encoding="utf-8") as f:
                f.write(result)  # type: ignore

        return result  # type: ignore
