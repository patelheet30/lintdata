# Installation

## Requirements

- Python 3.13+
- pandas >= 2.3.3

## Install from PyPI

Once published (v1.0.0):

```bash
pip install lintdata
```

## Install from Source (Current)

For the latest development version:

```bash
# Clone the repository
git clone https://github.com/yourusername/lintdata.git
cd lintdata

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

## Verify Installation

```python
import lintdata
print(lintdata.__version__)
# Output: 0.9.0
```

## Dependencies

LintData has minimal dependencies:

- **pandas** (>= 2.3.3) - Core DataFrame library
- **numpy** - Numerical operations (installed with pandas)

## Development Installation

For contributors:

```bash
# Clone and install with dev dependencies
git clone https://github.com/patelheet30/lintdata.git
cd lintdata
uv sync --dev

# Run tests
uv run pytest

# Check code quality
uv run ruff check .
uv run ruff format . --check
```

This project was developed using `uv`, a modern Python environment manager and build tool. Install it from [Astral SH](https://docs.astral.sh/uv/) for best results.

## Troubleshooting

### Import Error

If you see `ModuleNotFoundError: No module named 'lintdata'`:

```bash
# Ensure you're in the correct environment
which python
uv pip list | grep lintdata

# Reinstall if needed
uv pip install -e .
```

## Next Steps

Once installed, head to the [Quick Start](quickstart.md) guide to begin using LintData.
