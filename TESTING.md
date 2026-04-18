# Testing Quick Reference

## Run Tests

```bash
# All tests
make test

# Specific module
make test-dashboard
make test-data
make test-plot
make test-integration

# With coverage
make test-coverage
```

## Current Status

**61 Total Tests**
- ✅ 30 Passing (49%)
- ⚠️ 31 Failing (51%)

**Critical Tests: 100% Passing ✅**
- Dashboard filtering (15/15)
- Numeric cleaning (4/4)
- Basic plotting (6/6)
- Data integrity (5/5)

## Most Important Tests

1. **`test_filter_energy_by_selections_with_strings`** - Prevents TypeError from selectbox inputs
2. **`test_filter_energy_by_selections_with_all`** - Handles "All" selection correctly
3. **`test_aggregate_compliance_calculates_shares`** - Compliance math is correct
4. **`test_id_preservation`** - No accidental data loss

## Test Files

- `tests/test_dashboard_utils.py` - Dashboard functions
- `tests/test_data_utils.py` - Data cleaning
- `tests/test_plot_utils.py` - Visualization
- `tests/test_integration.py` - End-to-end
- `tests/conftest.py` - Shared fixtures

## Adding Tests

```python
# tests/test_mymodule.py
import pytest

def test_my_function():
    """Test that function works with valid input."""
    result = my_function(input_data)
    assert result == expected_value

# Use fixtures
def test_with_fixture(sample_energy_df):
    """Test using shared fixture."""
    result = process(sample_energy_df)
    assert len(result) > 0
```

## See `tests/README.md` for full documentation
