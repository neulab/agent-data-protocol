# Agent Data Collection Repository

## Repository Structure
- `datasets/`: Contains all datasets with their raw, standardized, and SFT formats
- `schema/`: Contains schema definitions for data formats
- `scripts/`: Contains utility scripts for data processing
- `tests/`: Contains unit tests for the repository

## Commands
- **Run tests**: `pytest`
- **Lint code**: `ruff check .`
- **Format code**: `ruff format .`
- **Type check**: `mypy .`

## Development Guidelines

### Testing
- All unit tests should be implemented using pytest
- Test files should be named with the prefix `test_` and placed in the `tests/` directory
- Use pytest fixtures for test setup when appropriate
- Use pytest's built-in assertion system rather than unittest assertions

### Code Style
- Follow PEP 8 guidelines
- Use double quotes for strings
- Maximum line length is 100 characters
- Use ruff for linting and formatting

### Type Checking
- Use mypy for type checking
- Type annotations are encouraged but not required for all functions

### Data Processing
- Each dataset should have:
  - `sample_raw.json`: Raw data format
  - `raw_to_standardized.py`: Script to convert raw to standardized format
  - `sample_std.json`: Standardized data format
  - `sample_sft.json`: SFT data format
