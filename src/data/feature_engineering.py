"""
Feature engineering module for creating advanced features from stock market data.
Includes technical indicators, Fourier transforms, ARIMA features, and more.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from scipy import fft
from statsmodels.tsa.arima.model import ARIMA
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Creates features for stock prediction including:
    - Technical indicators (MA, MACD, RSI, Bollinger Bands)
    - Fourier transforms for trend extraction
    - ARIMA predictions as features
    - Momentum and volatility features
    """
    
    def __init__(self):
        """Initialize FeatureEngineer."""
        logger.info("Initialized FeatureEngineer")
    
    def add_technical_indicators(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close'
    ) -> pd.DataFrame:
        """
        Add technical indicators to the DataFrame.
        
        Args:
            df: Input DataFrame with price data
            price_col: Column name containing price
        
        Returns:
            DataFrame with technical indicators added
        """
        logger.info("Adding technical indicators")
        
        df_features = df.copy()
        
        # Moving Averages
        df_features['MA7'] = df_features[price_col].rolling(window=7).mean()
        df_features['MA21'] = df_features[price_col].rolling(window=21).mean()
        df_features['MA50'] = df_features[price_col].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df_features['EMA12'] = df_features[price_col].ewm(span=12, adjust=False).mean()
        df_features['EMA26'] = df_features[price_col].ewm(span=26, adjust=False).mean()
        
        # MACD (Moving Average Convergence Divergence)
        df_features['MACD'] = df_features['EMA12'] - df_features['EMA26']
        df_features['MACD_signal'] = df_features['MACD'].ewm(span=9, adjust=False).mean()
        df_features['MACD_hist'] = df_features['MACD'] - df_features['MACD_signal']
        
        # RSI (Relative Strength Index)
        df_features['RSI'] = self._calculate_rsi(df_features[price_col], period=14)
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        df_features['BB_middle'] = df_features[price_col].rolling(window=bb_period).mean()
        bb_std_dev = df_features[price_col].rolling(window=bb_period).std()
        df_features['BB_upper'] = df_features['BB_middle'] + (bb_std * bb_std_dev)
        df_features['BB_lower'] = df_features['BB_middle'] - (bb_std * bb_std_dev)
        df_features['BB_width'] = df_features['BB_upper'] - df_features['BB_lower']
        
        # Price position in Bollinger Bands
        df_features['BB_position'] = (
            (df_features[price_col] - df_features['BB_lower']) / 
            (df_features['BB_upper'] - df_features['BB_lower'])
        )
        
        # Momentum
        df_features['Momentum_1'] = df_features[price_col].diff(1)
        df_features['Momentum_5'] = df_features[price_col].diff(5)
        df_features['Momentum_10'] = df_features[price_col].diff(10)
        
        # Rate of Change (ROC)
        df_features['ROC'] = df_features[price_col].pct_change(periods=10) * 100
        
        # Volatility (Standard Deviation)
        df_features['Volatility_10'] = df_features[price_col].rolling(window=10).std()
        df_features['Volatility_30'] = df_features[price_col].rolling(window=30).std()
        
        # Average True Range (ATR) - if High/Low available
        if 'High' in df.columns and 'Low' in df.columns:
            df_features['ATR'] = self._calculate_atr(
                df_features['High'],
                df_features['Low'],
                df_features[price_col],
                period=14
            )
        
        # On-Balance Volume (OBV) - if Volume available
        if 'Volume' in df.columns:
            df_features['OBV'] = self._calculate_obv(
                df_features[price_col],
                df_features['Volume']
            )
        
        logger.info(f"Added {len(df_features.columns) - len(df.columns)} technical indicators")
        
        return df_features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            prices: Price series
            period: RSI period
        
        Returns:
            RSI series
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Average True Range.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: ATR period
        
        Returns:
            ATR series
        """
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate On-Balance Volume.
        
        Args:
            close: Close price series
            volume: Volume series
        
        Returns:
            OBV series
        """
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv
    
    def add_fourier_transforms(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        n_components: List[int] = [3, 6, 9, 50]
    ) -> pd.DataFrame:
        """
        Add Fourier transform features for trend extraction.
        
        Args:
            df: Input DataFrame
            price_col: Column containing price data
            n_components: List of number of Fourier components to use
        
        Returns:
            DataFrame with Fourier features added
        """
        logger.info(f"Adding Fourier transforms with {n_components} components")
        
        df_features = df.copy()
        prices = df_features[price_col].values
        
        # Compute FFT
        fft_values = fft.fft(prices)
        fft_abs = np.abs(fft_values)
        
        # Add Fourier transforms with different components
        for n in n_components:
            # Create a copy of FFT values
            fft_filtered = fft_values.copy()
            
            # Zero out all but first n components
            fft_filtered[n:-n] = 0
            
            # Inverse FFT
            prices_filtered = np.real(fft.ifft(fft_filtered))
            
            df_features[f'Fourier_{n}'] = prices_filtered
        
        logger.info(f"Added {len(n_components)} Fourier transform features")
        
        return df_features
    
    def add_arima_features(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        order: Tuple[int, int, int] = (5, 1, 0),
        forecast_steps: int = 1
    ) -> pd.DataFrame:
        """
        Add ARIMA predictions as features.
        
        Args:
            df: Input DataFrame
            price_col: Column containing price data
            order: ARIMA order (p, d, q)
            forecast_steps: Number of steps to forecast
        
        Returns:
            DataFrame with ARIMA features added
        """
        logger.info(f"Adding ARIMA features with order {order}")
        
        df_features = df.copy()
        prices = df_features[price_col].values
        
        # We'll use a rolling window to create ARIMA predictions
        arima_predictions = []
        window_size = 100  # Use last 100 days for fitting
        
        for i in range(len(prices)):
            if i < window_size:
                # Not enough data yet
                arima_predictions.append(np.nan)
            else:
                try:
                    # Fit ARIMA on historical data
                    train_data = prices[max(0, i-window_size):i]
                    model = ARIMA(train_data, order=order)
                    model_fit = model.fit()
                    
                    # Make prediction
                    forecast = model_fit.forecast(steps=forecast_steps)
                    arima_predictions.append(forecast[0])
                
                except Exception as e:
                    # If ARIMA fails, use last value
                    arima_predictions.append(prices[i-1] if i > 0 else np.nan)
        
        df_features['ARIMA_prediction'] = arima_predictions
        
        # Add difference between actual and ARIMA prediction
        df_features['ARIMA_error'] = (
            df_features[price_col] - df_features['ARIMA_prediction']
        )
        
        logger.info("Added ARIMA prediction features")
        
        return df_features
    
    def add_lag_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        lags: List[int] = [1, 2, 3, 5, 10]
    ) -> pd.DataFrame:
        """
        Add lagged features.
        
        Args:
            df: Input DataFrame
            columns: Columns to create lags for
            lags: List of lag periods
        
        Returns:
            DataFrame with lag features added
        """
        logger.info(f"Adding lag features for {len(columns)} columns")
        
        df_features = df.copy()
        
        for col in columns:
            if col not in df_features.columns:
                logger.warning(f"Column {col} not found, skipping")
                continue
            
            for lag in lags:
                df_features[f'{col}_lag_{lag}'] = df_features[col].shift(lag)
        
        logger.info(f"Added {len(columns) * len(lags)} lag features")
        
        return df_features
    
    def add_rolling_statistics(
        self,
        df: pd.DataFrame,
        columns: List[str],
        windows: List[int] = [5, 10, 20, 50]
    ) -> pd.DataFrame:
        """
        Add rolling statistical features.
        
        Args:
            df: Input DataFrame
            columns: Columns to compute statistics for
            windows: List of window sizes
        
        Returns:
            DataFrame with rolling statistics added
        """
        logger.info(f"Adding rolling statistics for {len(columns)} columns")
        
        df_features = df.copy()
        
        for col in columns:
            if col not in df_features.columns:
                continue
            
            for window in windows:
                # Rolling mean
                df_features[f'{col}_rolling_mean_{window}'] = (
                    df_features[col].rolling(window=window).mean()
                )
                
                # Rolling std
                df_features[f'{col}_rolling_std_{window}'] = (
                    df_features[col].rolling(window=window).std()
                )
                
                # Rolling min/max
                df_features[f'{col}_rolling_min_{window}'] = (
                    df_features[col].rolling(window=window).min()
                )
                df_features[f'{col}_rolling_max_{window}'] = (
                    df_features[col].rolling(window=window).max()
                )
        
        logger.info("Added rolling statistics features")
        
        return df_features
    
    def create_all_features(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        add_technical: bool = True,
        add_fourier: bool = True,
        add_arima: bool = False,  # ARIMA is slow, make it optional
        add_lags: bool = True,
        add_rolling: bool = True
    ) -> pd.DataFrame:
        """
        Create all features in a single pipeline.
        
        Args:
            df: Input DataFrame
            price_col: Column containing price data
            add_technical: Whether to add technical indicators
            add_fourier: Whether to add Fourier transforms
            add_arima: Whether to add ARIMA features
            add_lags: Whether to add lag features
            add_rolling: Whether to add rolling statistics
        
        Returns:
            DataFrame with all features
        """
        logger.info("Creating all features")
        
        df_features = df.copy()
        
        if add_technical:
            df_features = self.add_technical_indicators(df_features, price_col)
        
        if add_fourier:
            df_features = self.add_fourier_transforms(df_features, price_col)
        
        if add_arima:
            df_features = self.add_arima_features(df_features, price_col)
        
        if add_lags:
            lag_columns = [price_col]
            if 'Volume' in df_features.columns:
                lag_columns.append('Volume')
            df_features = self.add_lag_features(df_features, lag_columns)
        
        if add_rolling:
            rolling_columns = [price_col]
            if 'Volume' in df_features.columns:
                rolling_columns.append('Volume')
            df_features = self.add_rolling_statistics(df_features, rolling_columns)
        
        # Drop NaN values created by rolling/lagging
        initial_rows = len(df_features)
        df_features = df_features.dropna()
        dropped_rows = initial_rows - len(df_features)
        
        logger.info(
            f"Feature engineering complete: "
            f"{len(df_features.columns)} total features, "
            f"{dropped_rows} rows dropped due to NaN"
        )
        
        return df_features


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=500)
    df = pd.DataFrame({
        'Open': np.random.randn(500).cumsum() + 100,
        'High': np.random.randn(500).cumsum() + 102,
        'Low': np.random.randn(500).cumsum() + 98,
        'Close': np.random.randn(500).cumsum() + 100,
        'Volume': np.random.randint(1000000, 10000000, 500)
    }, index=dates)
    
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(
        df,
        add_technical=True,
        add_fourier=True,
        add_arima=False,
        add_lags=True,
        add_rolling=True
    )
    
    print(f"Original features: {len(df.columns)}")
    print(f"Total features: {len(df_features.columns)}")
    print(f"New features: {len(df_features.columns) - len(df.columns)}")
