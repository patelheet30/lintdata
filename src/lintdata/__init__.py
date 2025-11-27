"""
LintData: A 'linter' for pandas DataFrames to automate data quality audits.
"""

__version__ = "1.0.0"

from .accessor import LintAccessor

__all__ = ["LintAccessor"]
