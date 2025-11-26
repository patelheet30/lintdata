"""
Implements the core LintData accessor for pandas Dataframes
"""

import csv
import json
import re
from typing import Any, Dict, List, Optional, Union

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
        threshold_years: float = 50,
        report_format: str = "text",
        output: Optional[str] = None,
        return_dict: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        """Generate a comprehensive quality report for the DataFrame.

        Args:
            checks_to_run (Optional[Union[List[str], str]], optional): Specific checks to run.
                Options: 'missing', 'duplicates', 'mixed_types', 'whitespace', 'constant',
                'unique', 'outliers', 'missing_patterns', 'case', 'cardinality', 'skewness',
                'duplicate_columns', 'type_consistency', 'negative', 'rare_categories',
                'date_format', 'string_length', 'zero_inflation', 'future_dates',
                'special_chars', 'date_anomalies'. Use 'all' to run all checks. Defaults to None.
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
            threshold_years (float, optional): Maximum acceptable date range in years. Columns with date ranges exceeding will be flagged. Defaults to 50.
            report_format (str, optional): Output format. Options: 'text', 'html', 'json', 'csv'. Defaults to 'text'.
            output (Optional[str], optional): File path to save the report. If None, returns as string. Defaults to None.
            return_dict (bool, optional): If True, returns structured dictionary instead of formatted string. Defaults to False.

        Raises:
            ValueError: If invalid check names are provided or invalid format specified.

        Returns:
            Union[str, Dict[str, Any]]: A comprehensive quality report in the specified format,
                or a structured dictionary if return_dict=True.

        Example:
            >>> # Text report
            >>> report = df.lint.report()

            >>> # HTML report
            >>> df.lint.report(report_format='html', output='report.html')

            >>> # JSON export
            >>> df.lint.report(report_format='json', output='report.json')

            >>> # CSV export
            >>> df.lint.report(report_format='csv', output='issues.csv')

            >>> # Get structured data
            >>> data = df.lint.report(return_dict=True)
        """
        valid_formats = ["text", "html", "json", "csv"]
        if report_format not in valid_formats:
            raise ValueError(f"Invalid format '{report_format}'. Valid options: {valid_formats}")

        if self._df.empty:
            if return_dict:
                return {"shape": [0, 0], "issues": [], "issue_count": 0}

            empty_message = "The DataFrame is empty. No checks run."
            if report_format == "text":
                result = f"--- LintData Quality Report ---\n{empty_message}"
            elif report_format == "html":
                result = HTMLReportFormatter.generate((0, 0), [])
            elif report_format == "json":
                result = json.dumps({"shape": [0, 0], "issues": [], "issue_count": 0}, indent=2)
            else:  # csv
                result = "check,column,severity,message\n"

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
            "date_anomalies": lambda: checks.check_date_range_anomalies(
                self._df, columns=future_date_columns, threshold_years=threshold_years
            ),
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

        if return_dict:
            structured_data = self._format_as_dict(all_warnings)
            return structured_data

        if report_format == "text":
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

        elif report_format == "html":
            result = HTMLReportFormatter.generate(self._df.shape, all_warnings)

        elif report_format == "json":
            structured_data = self._format_as_dict(all_warnings)
            result = json.dumps(structured_data, indent=2)

        elif report_format == "csv":
            result = self._format_as_csv(all_warnings)

        # Save to file if output path provided
        if output:
            with open(output, "w", encoding="utf-8") as f:
                f.write(result)  # pyright: ignore[reportPossiblyUnboundVariable]

        return result  # pyright: ignore[reportPossiblyUnboundVariable]

    def _format_as_dict(self, warnings: List[str]) -> Dict[str, Any]:
        """Convert warnings to structured dictionary.

        Args:
            warnings: List of warning strings

        Returns:
            Dict with shape, issues list, and metadata
        """
        issues = []
        for warning in warnings:
            parsed = self._parse_warning(warning)
            issues.append(parsed)

        return {"shape": list(self._df.shape), "issue_count": len(warnings), "issues": issues}

    def _parse_warning(self, warning: str) -> Dict[str, Any]:
        """Parse a warning string into structured data.

        Args:
            warning: Warning string like "[Missing Values] Column 'age': 5 missing values"

        Returns:
            Dict with check, column, severity, and message
        """
        # Extract check type from [brackets]
        if "]" in warning:
            check_type = warning.split("]")[0].replace("[", "").strip()
            message = warning.split("]", 1)[1].strip()
        else:
            check_type = "Unknown"
            message = warning

        # Extract column name if present
        column = None
        column_match = re.search(r"Column (['\"])(.*?)\1", message)
        column = column_match.group(2) if column_match else None

        # Determine severity
        severity = self._get_severity(warning)

        return {"check": check_type, "column": column, "severity": severity, "message": message}

    def _get_severity(self, warning: str) -> str:
        """Determine severity level from warning text.

        Args:
            warning: Warning string

        Returns:
            'high', 'medium', or 'low'
        """
        warning_lower = warning.lower()

        high_indicators = [
            "missing values",
            "duplicate rows",
            "mixed types",
            "future dates",
            "negative values",
        ]

        medium_indicators = [
            "outliers",
            "whitespace",
            "case consistency",
            "special characters",
            "date format",
        ]

        if any(indicator in warning_lower for indicator in high_indicators):
            return "high"
        elif any(indicator in warning_lower for indicator in medium_indicators):
            return "medium"
        else:
            return "low"

    def _format_as_csv(self, warnings: List[str]) -> str:
        """Format warnings as CSV string.

        Args:
            warnings: List of warning strings

        Returns:
            CSV formatted string
        """
        from io import StringIO

        output = StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow(["check", "column", "severity", "message"])

        # Write data
        for warning in warnings:
            parsed = self._parse_warning(warning)
            writer.writerow([parsed["check"], parsed["column"] or "N/A", parsed["severity"], parsed["message"]])

        return output.getvalue()
