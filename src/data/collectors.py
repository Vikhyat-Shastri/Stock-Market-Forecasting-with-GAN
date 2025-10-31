"""
Data collection module for fetching stock market data from various sources.
Supports multiple data providers and implements retry logic for reliability.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class DataCollector:
    """
    Collects stock market data from multiple sources including:
    - Historical price data (OHLCV)
    - Correlated assets
    - Market indices
    - Currency pairs
    - Volatility index (VIX)
    """
    
    def __init__(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize DataCollector.
        
        Args:
            ticker: Primary stock ticker symbol (e.g., 'GS')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            cache_dir: Directory to cache downloaded data
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.cache_dir = cache_dir or Path("data/raw")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup retry strategy for API calls
        self.session = self._setup_session()
        
        logger.info(
            f"Initialized DataCollector for {ticker} "
            f"from {start_date} to {end_date}"
        )
    
    def _setup_session(self) -> requests.Session:
        """Setup requests session with retry strategy."""
        session = requests.Session()
        retry = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
    def fetch_primary_stock(self) -> pd.DataFrame:
        """
        Fetch primary stock data.
        
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching primary stock data for {self.ticker}")
        
        cache_file = self.cache_dir / f"{self.ticker}_primary.csv"
        
        # Check cache
        if cache_file.exists():
            logger.info(f"Loading cached data from {cache_file}")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        
        # Fetch from API
        try:
            stock = yf.Ticker(self.ticker)
            df = stock.history(start=self.start_date, end=self.end_date)
            
            if df.empty:
                raise ValueError(f"No data found for {self.ticker}")
            
            # Save to cache
            df.to_csv(cache_file)
            logger.info(f"Cached data to {cache_file}")
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching data for {self.ticker}: {e}")
            raise
    
    def fetch_correlated_assets(
        self,
        tickers: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for correlated assets.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        logger.info(f"Fetching {len(tickers)} correlated assets")
        
        assets_data = {}
        
        for ticker in tickers:
            try:
                cache_file = self.cache_dir / f"{ticker}_correlated.csv"
                
                if cache_file.exists():
                    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                else:
                    stock = yf.Ticker(ticker)
                    df = stock.history(start=self.start_date, end=self.end_date)
                    df.to_csv(cache_file)
                
                assets_data[ticker] = df
                logger.info(f"Fetched data for {ticker}")
                
            except Exception as e:
                logger.warning(f"Failed to fetch {ticker}: {e}")
                continue
        
        return assets_data
    
    def get_default_correlated_assets(self) -> List[str]:
        """
        Get default list of correlated assets for a financial stock.
        
        Returns:
            List of ticker symbols
        """
        # Similar financial institutions
        financials = ['JPM', 'MS', 'BAC', 'C', 'WFC']
        
        # Market indices
        indices = ['^GSPC', '^DJI', '^IXIC', '^FTSE', '^N225']
        
        # Volatility
        volatility = ['^VIX']
        
        # Currencies (if available through yfinance)
        currencies = ['EURUSD=X', 'GBPUSD=X', 'JPYUSD=X']
        
        # Commodities
        commodities = ['GC=F', 'CL=F']  # Gold, Crude Oil
        
        return financials + indices + volatility + currencies + commodities
    
    def fetch_all_data(
        self,
        correlated_tickers: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Fetch all required data.
        
        Args:
            correlated_tickers: Optional list of correlated asset tickers
            
        Returns:
            Tuple of (primary_data, correlated_data)
        """
        # Fetch primary stock
        primary_data = self.fetch_primary_stock()
        
        # Fetch correlated assets
        if correlated_tickers is None:
            correlated_tickers = self.get_default_correlated_assets()
        
        correlated_data = self.fetch_correlated_assets(correlated_tickers)
        
        logger.info(
            f"Data collection complete: "
            f"{len(primary_data)} days, "
            f"{len(correlated_data)} correlated assets"
        )
        
        return primary_data, correlated_data
    
    def validate_data(
        self,
        df: pd.DataFrame,
        max_missing_pct: float = 0.05
    ) -> bool:
        """
        Validate data quality.
        
        Args:
            df: DataFrame to validate
            max_missing_pct: Maximum allowed missing data percentage
            
        Returns:
            True if data passes validation
        """
        missing_pct = df.isna().sum().sum() / (df.shape[0] * df.shape[1])
        
        if missing_pct > max_missing_pct:
            logger.warning(
                f"Data has {missing_pct:.2%} missing values "
                f"(threshold: {max_missing_pct:.2%})"
            )
            return False
        
        if len(df) < 100:
            logger.warning(f"Insufficient data: only {len(df)} samples")
            return False
        
        logger.info("Data validation passed")
        return True


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    collector = DataCollector(
        ticker="GS",
        start_date="2010-01-01",
        end_date="2023-12-31"
    )
    
    primary, correlated = collector.fetch_all_data()
    print(f"Primary data shape: {primary.shape}")
    print(f"Correlated assets: {len(correlated)}")
