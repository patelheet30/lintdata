"""
LintData: A 'linter' for pandas DataFrames to automate data quality audits.
"""

__version__ = "0.5.0"

from .accessor import LintAccessor

__all__ = ["LintAccessor"]
