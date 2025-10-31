"""
Unit tests for data collection module.
"""

import pytest
import pandas as pd
from datetime import datetime
from pathlib import Path
from src.data.collectors import DataCollector


@pytest.fixture
def collector():
    """Create a DataCollector instance for testing."""
    return DataCollector(
        ticker="AAPL",
        start_date="2020-01-01",
        end_date="2020-12-31",
        cache_dir=Path("tests/test_cache")
    )


def test_collector_initialization(collector):
    """Test DataCollector initialization."""
    assert collector.ticker == "AAPL"
    assert collector.start_date == "2020-01-01"
    assert collector.end_date == "2020-12-31"


def test_fetch_primary_stock(collector):
    """Test fetching primary stock data."""
    # This test requires internet connection
    # Consider mocking yfinance for offline testing
    pass


def test_get_default_correlated_assets(collector):
    """Test getting default correlated assets."""
    assets = collector.get_default_correlated_assets()
    
    assert isinstance(assets, list)
    assert len(assets) > 0
    assert 'JPM' in assets
    assert '^VIX' in assets


def test_validate_data(collector):
    """Test data validation."""
    # Create sample data
    df = pd.DataFrame({
        'Close': [100, 101, 102, 103],
        'Volume': [1000, 1100, 1200, 1300]
    })
    
    # Should pass validation
    assert collector.validate_data(df, max_missing_pct=0.05) == True
    
    # Add missing values
    df.loc[0, 'Close'] = None
    df.loc[1, 'Close'] = None
    
    # Should fail validation (50% missing)
    assert collector.validate_data(df, max_missing_pct=0.05) == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
