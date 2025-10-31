"""
Data preprocessing module for cleaning and normalizing stock market data.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocesses stock market data including:
    - Handling missing values
    - Outlier detection and treatment
    - Normalization/Standardization
    - Data alignment across assets
    """
    
    def __init__(
        self,
        scaling_method: str = "standard",
        handle_outliers: bool = True,
        outlier_std: float = 3.0
    ):
        """
        Initialize DataPreprocessor.
        
        Args:
            scaling_method: Scaling method ('standard', 'minmax', 'robust')
            handle_outliers: Whether to handle outliers
            outlier_std: Number of standard deviations for outlier detection
        """
        self.scaling_method = scaling_method
        self.handle_outliers = handle_outliers
        self.outlier_std = outlier_std
        
        # Initialize scalers
        self.scalers: Dict[str, object] = {}
        self._init_scaler()
        
        logger.info(f"Initialized DataPreprocessor with {scaling_method} scaling")
    
    def _init_scaler(self):
        """Initialize the appropriate scaler based on method."""
        if self.scaling_method == "standard":
            self.primary_scaler = StandardScaler()
        elif self.scaling_method == "minmax":
            self.primary_scaler = MinMaxScaler()
        elif self.scaling_method == "robust":
            self.primary_scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")
    
    def handle_missing_values(
        self,
        df: pd.DataFrame,
        method: str = "forward_fill"
    ) -> pd.DataFrame:
        """
        Handle missing values in the data.
        
        Args:
            df: Input DataFrame
            method: Method to handle missing values
                   ('forward_fill', 'backward_fill', 'interpolate', 'drop')
        
        Returns:
            DataFrame with missing values handled
        """
        initial_missing = df.isna().sum().sum()
        
        if initial_missing == 0:
            return df
        
        logger.info(f"Handling {initial_missing} missing values using {method}")
        
        df_clean = df.copy()
        
        if method == "forward_fill":
            df_clean = df_clean.ffill()
        elif method == "backward_fill":
            df_clean = df_clean.bfill()
        elif method == "interpolate":
            df_clean = df_clean.interpolate(method='linear')
        elif method == "drop":
            df_clean = df_clean.dropna()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # If any missing values remain, forward fill then backward fill
        if df_clean.isna().any().any():
            df_clean = df_clean.ffill().bfill()
        
        final_missing = df_clean.isna().sum().sum()
        logger.info(f"Missing values reduced from {initial_missing} to {final_missing}")
        
        return df_clean
    
    def detect_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Detect outliers using z-score method.
        
        Args:
            df: Input DataFrame
            columns: Columns to check for outliers (None = all numeric)
        
        Returns:
            Boolean DataFrame indicating outliers
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers = pd.DataFrame(False, index=df.index, columns=df.columns)
        
        for col in columns:
            if col not in df.columns:
                continue
            
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers[col] = z_scores > self.outlier_std
        
        total_outliers = outliers.sum().sum()
        logger.info(f"Detected {total_outliers} outliers")
        
        return outliers
    
    def treat_outliers(
        self,
        df: pd.DataFrame,
        method: str = "winsorize"
    ) -> pd.DataFrame:
        """
        Treat outliers in the data.
        
        Args:
            df: Input DataFrame
            method: Treatment method ('winsorize', 'clip', 'remove')
        
        Returns:
            DataFrame with outliers treated
        """
        if not self.handle_outliers:
            return df
        
        df_clean = df.copy()
        outliers = self.detect_outliers(df_clean)
        
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == "winsorize":
                # Cap outliers at mean ± 3*std
                mean = df_clean[col].mean()
                std = df_clean[col].std()
                lower_bound = mean - self.outlier_std * std
                upper_bound = mean + self.outlier_std * std
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
            
            elif method == "clip":
                # Clip to percentiles
                lower = df_clean[col].quantile(0.01)
                upper = df_clean[col].quantile(0.99)
                df_clean[col] = df_clean[col].clip(lower, upper)
            
            elif method == "remove":
                # Replace outliers with NaN then interpolate
                df_clean.loc[outliers[col], col] = np.nan
                df_clean[col] = df_clean[col].interpolate(method='linear')
        
        logger.info(f"Outliers treated using {method} method")
        return df_clean
    
    def align_data(
        self,
        primary_df: pd.DataFrame,
        correlated_dfs: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Align primary and correlated data to same date range.
        
        Args:
            primary_df: Primary stock DataFrame
            correlated_dfs: Dictionary of correlated asset DataFrames
        
        Returns:
            Tuple of aligned (primary, correlated) data
        """
        # Find common date range
        start_date = primary_df.index.min()
        end_date = primary_df.index.max()
        
        for ticker, df in correlated_dfs.items():
            start_date = max(start_date, df.index.min())
            end_date = min(end_date, df.index.max())
        
        # Align primary data
        primary_aligned = primary_df.loc[start_date:end_date]
        
        # Align correlated data
        correlated_aligned = {}
        for ticker, df in correlated_dfs.items():
            aligned = df.loc[start_date:end_date]
            # Reindex to match primary data dates
            aligned = aligned.reindex(primary_aligned.index, method='ffill')
            correlated_aligned[ticker] = aligned
        
        logger.info(
            f"Data aligned to {len(primary_aligned)} days "
            f"({start_date.date()} to {end_date.date()})"
        )
        
        return primary_aligned, correlated_aligned
    
    def scale_data(
        self,
        df: pd.DataFrame,
        fit: bool = True,
        scaler_key: str = "default"
    ) -> pd.DataFrame:
        """
        Scale/normalize the data.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the scaler (True for training data)
            scaler_key: Key to store/retrieve scaler
        
        Returns:
            Scaled DataFrame
        """
        if fit:
            # Create new scaler
            if self.scaling_method == "standard":
                scaler = StandardScaler()
            elif self.scaling_method == "minmax":
                scaler = MinMaxScaler()
            else:
                scaler = RobustScaler()
            
            scaled_values = scaler.fit_transform(df.values)
            self.scalers[scaler_key] = scaler
            logger.info(f"Fitted and transformed data with {self.scaling_method} scaler")
        else:
            # Use existing scaler
            if scaler_key not in self.scalers:
                raise ValueError(f"Scaler '{scaler_key}' not found. Fit first.")
            
            scaler = self.scalers[scaler_key]
            scaled_values = scaler.transform(df.values)
            logger.info(f"Transformed data with existing {self.scaling_method} scaler")
        
        return pd.DataFrame(
            scaled_values,
            index=df.index,
            columns=df.columns
        )
    
    def inverse_scale(
        self,
        df: pd.DataFrame,
        scaler_key: str = "default"
    ) -> pd.DataFrame:
        """
        Inverse transform scaled data back to original scale.
        
        Args:
            df: Scaled DataFrame
            scaler_key: Key to retrieve scaler
        
        Returns:
            Original scale DataFrame
        """
        if scaler_key not in self.scalers:
            raise ValueError(f"Scaler '{scaler_key}' not found")
        
        scaler = self.scalers[scaler_key]
        original_values = scaler.inverse_transform(df.values)
        
        return pd.DataFrame(
            original_values,
            index=df.index,
            columns=df.columns
        )
    
    def preprocess_pipeline(
        self,
        primary_df: pd.DataFrame,
        correlated_dfs: Optional[Dict[str, pd.DataFrame]] = None,
        fit: bool = True
    ) -> Tuple[pd.DataFrame, Optional[Dict[str, pd.DataFrame]]]:
        """
        Complete preprocessing pipeline.
        
        Args:
            primary_df: Primary stock DataFrame
            correlated_dfs: Optional correlated assets DataFrames
            fit: Whether to fit scalers
        
        Returns:
            Tuple of processed (primary, correlated) data
        """
        logger.info("Starting preprocessing pipeline")
        
        # Handle missing values
        primary_clean = self.handle_missing_values(primary_df)
        
        # Treat outliers
        primary_clean = self.treat_outliers(primary_clean)
        
        # Process correlated data if provided
        correlated_clean = None
        if correlated_dfs:
            correlated_clean = {}
            for ticker, df in correlated_dfs.items():
                df_clean = self.handle_missing_values(df)
                df_clean = self.treat_outliers(df_clean)
                correlated_clean[ticker] = df_clean
            
            # Align all data
            primary_clean, correlated_clean = self.align_data(
                primary_clean, correlated_clean
            )
        
        # Scale data
        primary_scaled = self.scale_data(primary_clean, fit=fit, scaler_key="primary")
        
        if correlated_clean:
            correlated_scaled = {}
            for ticker, df in correlated_clean.items():
                df_scaled = self.scale_data(
                    df,
                    fit=fit,
                    scaler_key=f"correlated_{ticker}"
                )
                correlated_scaled[ticker] = df_scaled
            correlated_clean = correlated_scaled
        
        logger.info("Preprocessing pipeline complete")
        
        return primary_clean, correlated_clean
    
    def save_scalers(self, path: Path):
        """Save fitted scalers to disk."""
        import pickle
        
        path.mkdir(parents=True, exist_ok=True)
        
        for key, scaler in self.scalers.items():
            scaler_file = path / f"scaler_{key}.pkl"
            with open(scaler_file, 'wb') as f:
                pickle.dump(scaler, f)
        
        logger.info(f"Saved {len(self.scalers)} scalers to {path}")
    
    def load_scalers(self, path: Path):
        """Load fitted scalers from disk."""
        import pickle
        
        scaler_files = list(path.glob("scaler_*.pkl"))
        
        for scaler_file in scaler_files:
            key = scaler_file.stem.replace("scaler_", "")
            with open(scaler_file, 'rb') as f:
                self.scalers[key] = pickle.load(f)
        
        logger.info(f"Loaded {len(self.scalers)} scalers from {path}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=100)
    df = pd.DataFrame({
        'Close': np.random.randn(100).cumsum() + 100,
        'Volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Add some missing values and outliers
    df.iloc[10:15, 0] = np.nan
    df.iloc[50, 0] = df['Close'].mean() + 10 * df['Close'].std()
    
    preprocessor = DataPreprocessor(scaling_method="standard")
    df_clean, _ = preprocessor.preprocess_pipeline(df, fit=True)
    
    print("Original data shape:", df.shape)
    print("Cleaned data shape:", df_clean.shape)
