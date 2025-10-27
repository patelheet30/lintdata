"""
LintData: A 'linter' for pandas DataFrames to automate data quality audits.
"""

__version__ = "0.2.0"

try:
    from . import accessor  # noqa: F401
except ImportError:
    pass
