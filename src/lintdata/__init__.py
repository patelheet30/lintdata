"""
LintData: A 'linter' for pandas DataFrames to automate data quality audits.
"""

__version__ = "0.4.0"

from .accessor import LintAccessor

__all__ = ["LintAccessor"]
