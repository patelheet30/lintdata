---
title: Home
description: LintData documentation
hide:
  - toc
  - navigation
---

# LintData

**A "linter" for pandas DataFrames to automate data quality audits.**

LintData helps data scientists, analysts, and ML engineers identify data quality issues quickly and systematically. Run one command and get actionable insights about your data.

## Features

✅ **20+ Data Quality Checks** - Missing values, duplicates, outliers, type consistency, and more  
✅ **Zero Configuration** - Works out of the box with sensible defaults  
✅ **Highly Configurable** - Customise thresholds and select specific checks  
✅ **Multiple Export Formats** - Text, HTML, JSON, and CSV reports  
✅ **Custom Checks API** - Extend with your own validation logic  
✅ **Pandas Native** - Integrates seamlessly via `.lint` accessor  
✅ **Parallel Execution** - Speed up checks with multiprocessing, threading, or joblib backends

## Why LintData?

Manual data quality checks are:

- ⏰ **Time-consuming** - Hours spent writing repetitive validation code
- 🐛 **Error-prone** - Easy to miss edge cases and subtle issues
- 🔄 **Not reusable** - Custom scripts need rewriting for each dataset

LintData automates this process, saving you considerable time during data preparation and catching common data quality issues automatically.

## Installation

```bash
pip install lintdata
```

## Next Steps

- [Quick Start Guide](getting-started/quickstart.md) - Get up and running in 5 minutes
- [Available Checks](user-guide/available_checks.md) - See all 22+ quality checks
- [Custom Checks](user-guide/custom_checks.md) - Extend LintData with your own validations

## Project Status

**Current Version:** v0.9.0 (Beta)  
**Target Release:** v1.0.0 (Production Ready)

We're approaching our stable 1.0 release! The API is stabilising, and we're focusing on documentation, examples, and community feedback.

## Licence

MIT Licence - see [LICENCE](https://github.com/patelheet30/lintdata/blob/main/LICENSE)
