"""
Multi-Timeframe Analysis for Stock Market Forecasting.
Supports daily, hourly, and intraday predictions with regime detection and ensemble methods.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from sklearn.cluster import KMeans
from hmmlearn import hmm
import joblib

logger = logging.getLogger(__name__)


class Timeframe(Enum):
    """Supported timeframes."""
    MINUTE_15 = "15min"
    MINUTE_30 = "30min"
    HOURLY = "1h"
    DAILY = "1d"
    WEEKLY = "1w"


class MarketRegime(Enum):
    """Market regime types."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class TimeframeConfig:
    """Configuration for a specific timeframe."""
    timeframe: Timeframe
    sequence_length: int
    forecast_horizon: int
    model_path: Optional[str] = None
    weight: float = 1.0  # Weight in ensemble
    features: Optional[List[str]] = None


class RegimeDetector:
    """
    Detects market regimes using clustering or HMM.
    """
    
    def __init__(
        self,
        method: str = "hmm",
        n_regimes: int = 5,
        lookback_period: int = 20
    ):
        """
        Initialize regime detector.
        
        Args:
            method: Detection method ('hmm', 'kmeans')
            n_regimes: Number of regimes
            lookback_period: Lookback period for regime features
        """
        self.method = method
        self.n_regimes = n_regimes
        self.lookback_period = lookback_period
        self.model = None
        
        logger.info(
            f"Initialized RegimeDetector: method={method}, n_regimes={n_regimes}"
        )
    
    def fit(self, price_data: pd.DataFrame):
        """
        Fit regime detection model.
        
        Args:
            price_data: DataFrame with OHLCV data
        """
        features = self._calculate_regime_features(price_data)
        
        if self.method == "hmm":
            self.model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=100,
                random_state=42
            )
            self.model.fit(features)
            
        elif self.method == "kmeans":
            self.model = KMeans(
                n_clusters=self.n_regimes,
                random_state=42,
                n_init=10
            )
            self.model.fit(features)
        
        logger.info(f"Fitted regime detector on {len(features)} samples")
    
    def predict(self, price_data: pd.DataFrame) -> np.ndarray:
        """
        Predict market regimes.
        
        Args:
            price_data: DataFrame with OHLCV data
        
        Returns:
            Array of regime predictions
        """
        if self.model is None:
            raise ValueError("Must fit model before predicting")
        
        features = self._calculate_regime_features(price_data)
        
        if self.method == "hmm":
            regimes = self.model.predict(features)
        else:  # kmeans
            regimes = self.model.predict(features)
        
        return regimes
    
    def _calculate_regime_features(self, price_data: pd.DataFrame) -> np.ndarray:
        """
        Calculate features for regime detection.
        
        Args:
            price_data: DataFrame with OHLCV data
        
        Returns:
            Feature array
        """
        df = price_data.copy()
        
        # Returns
        df['returns'] = df['Close'].pct_change()
        
        # Volatility
        df['volatility'] = df['returns'].rolling(self.lookback_period).std()
        
        # Trend strength (ADX-like)
        df['price_ma'] = df['Close'].rolling(self.lookback_period).mean()
        df['trend_strength'] = abs(df['Close'] - df['price_ma']) / df['price_ma']
        
        # Volume
        if 'Volume' in df.columns:
            df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(self.lookback_period).mean()
        else:
            df['volume_ratio'] = 1.0
        
        # High-Low range
        if 'High' in df.columns and 'Low' in df.columns:
            df['hl_range'] = (df['High'] - df['Low']) / df['Close']
        else:
            df['hl_range'] = 0.01
        
        feature_cols = ['returns', 'volatility', 'trend_strength', 'volume_ratio', 'hl_range']
        features = df[feature_cols].fillna(0).values
        
        return features
    
    def interpret_regime(self, regime_id: int, features: np.ndarray) -> MarketRegime:
        """
        Interpret numerical regime as MarketRegime enum.
        
        Args:
            regime_id: Numerical regime ID
            features: Current feature values
        
        Returns:
            MarketRegime interpretation
        """
        # Extract current features
        returns = features[0] if len(features) > 0 else 0
        volatility = features[1] if len(features) > 1 else 0.01
        trend_strength = features[2] if len(features) > 2 else 0
        
        # Simple heuristics
        if volatility > 0.03:
            return MarketRegime.VOLATILE
        elif volatility < 0.01:
            return MarketRegime.LOW_VOLATILITY
        elif trend_strength > 0.05:
            if returns > 0:
                return MarketRegime.TRENDING_UP
            else:
                return MarketRegime.TRENDING_DOWN
        else:
            return MarketRegime.RANGING
    
    def save(self, filepath: str):
        """Save regime detector to disk."""
        joblib.dump({
            'model': self.model,
            'method': self.method,
            'n_regimes': self.n_regimes,
            'lookback_period': self.lookback_period
        }, filepath)
        logger.info(f"Saved regime detector to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'RegimeDetector':
        """Load regime detector from disk."""
        data = joblib.load(filepath)
        detector = cls(
            method=data['method'],
            n_regimes=data['n_regimes'],
            lookback_period=data['lookback_period']
        )
        detector.model = data['model']
        logger.info(f"Loaded regime detector from {filepath}")
        return detector


class MultiTimeframePredictor:
    """
    Manages predictions across multiple timeframes.
    """
    
    def __init__(
        self,
        timeframe_configs: List[TimeframeConfig],
        regime_detector: Optional[RegimeDetector] = None,
        ensemble_method: str = "weighted_average"
    ):
        """
        Initialize multi-timeframe predictor.
        
        Args:
            timeframe_configs: List of timeframe configurations
            regime_detector: Optional regime detector
            ensemble_method: Ensemble method ('weighted_average', 'regime_adaptive', 'stacking')
        """
        self.timeframe_configs = timeframe_configs
        self.regime_detector = regime_detector
        self.ensemble_method = ensemble_method
        
        self.models: Dict[Timeframe, torch.nn.Module] = {}
        self.predictions_cache: Dict[Timeframe, np.ndarray] = {}
        
        logger.info(
            f"Initialized MultiTimeframePredictor: "
            f"timeframes={[tf.timeframe.value for tf in timeframe_configs]}, "
            f"ensemble={ensemble_method}"
        )
    
    def load_models(self):
        """Load models for all timeframes."""
        for config in self.timeframe_configs:
            if config.model_path and Path(config.model_path).exists():
                try:
                    model = torch.load(config.model_path)
                    self.models[config.timeframe] = model
                    logger.info(f"Loaded model for {config.timeframe.value}")
                except Exception as e:
                    logger.error(f"Error loading model for {config.timeframe.value}: {e}")
    
    def predict(
        self,
        data: Dict[Timeframe, pd.DataFrame],
        current_regime: Optional[MarketRegime] = None
    ) -> Dict:
        """
        Generate predictions across all timeframes.
        
        Args:
            data: Dictionary mapping timeframes to data
            current_regime: Current market regime (optional)
        
        Returns:
            Dictionary with predictions and metadata
        """
        timeframe_predictions = {}
        
        # Generate predictions for each timeframe
        for config in self.timeframe_configs:
            if config.timeframe not in data:
                logger.warning(f"No data for timeframe {config.timeframe.value}")
                continue
            
            if config.timeframe not in self.models:
                logger.warning(f"No model loaded for {config.timeframe.value}")
                continue
            
            # Get model and data
            model = self.models[config.timeframe]
            tf_data = data[config.timeframe]
            
            # Prepare input
            # Placeholder: actual implementation depends on model architecture
            input_tensor = self._prepare_input(tf_data, config)
            
            # Generate prediction
            with torch.no_grad():
                model.eval()
                prediction = model(input_tensor)
            
            # Store prediction
            timeframe_predictions[config.timeframe] = {
                'prediction': prediction.cpu().numpy(),
                'weight': config.weight,
                'horizon': config.forecast_horizon
            }
        
        # Ensemble predictions
        ensemble_prediction = self._ensemble_predictions(
            timeframe_predictions,
            current_regime
        )
        
        return {
            'ensemble_prediction': ensemble_prediction,
            'timeframe_predictions': timeframe_predictions,
            'regime': current_regime
        }
    
    def _prepare_input(
        self,
        data: pd.DataFrame,
        config: TimeframeConfig
    ) -> torch.Tensor:
        """
        Prepare input tensor for model.
        
        Args:
            data: DataFrame with features
            config: Timeframe configuration
        
        Returns:
            Input tensor
        """
        # Get last sequence
        sequence = data.iloc[-config.sequence_length:].values
        
        # Convert to tensor
        tensor = torch.FloatTensor(sequence).unsqueeze(0)  # Add batch dimension
        
        return tensor
    
    def _ensemble_predictions(
        self,
        timeframe_predictions: Dict[Timeframe, Dict],
        current_regime: Optional[MarketRegime]
    ) -> np.ndarray:
        """
        Ensemble predictions from multiple timeframes.
        
        Args:
            timeframe_predictions: Predictions from each timeframe
            current_regime: Current market regime
        
        Returns:
            Ensembled prediction
        """
        if len(timeframe_predictions) == 0:
            return np.array([])
        
        if self.ensemble_method == "weighted_average":
            return self._weighted_average_ensemble(timeframe_predictions)
        
        elif self.ensemble_method == "regime_adaptive":
            return self._regime_adaptive_ensemble(
                timeframe_predictions,
                current_regime
            )
        
        else:
            logger.warning(f"Unknown ensemble method: {self.ensemble_method}")
            return self._weighted_average_ensemble(timeframe_predictions)
    
    def _weighted_average_ensemble(
        self,
        timeframe_predictions: Dict[Timeframe, Dict]
    ) -> np.ndarray:
        """
        Simple weighted average ensemble.
        
        Args:
            timeframe_predictions: Predictions from each timeframe
        
        Returns:
            Ensembled prediction
        """
        predictions = []
        weights = []
        
        for tf_pred in timeframe_predictions.values():
            predictions.append(tf_pred['prediction'])
            weights.append(tf_pred['weight'])
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Weighted average
        ensemble = sum(pred * w for pred, w in zip(predictions, weights))
        
        return ensemble
    
    def _regime_adaptive_ensemble(
        self,
        timeframe_predictions: Dict[Timeframe, Dict],
        current_regime: Optional[MarketRegime]
    ) -> np.ndarray:
        """
        Regime-adaptive ensemble weighting.
        
        Args:
            timeframe_predictions: Predictions from each timeframe
            current_regime: Current market regime
        
        Returns:
            Ensembled prediction
        """
        # Adjust weights based on regime
        regime_weights = {
            MarketRegime.TRENDING_UP: {
                Timeframe.DAILY: 1.5,
                Timeframe.HOURLY: 1.0,
                Timeframe.MINUTE_15: 0.5
            },
            MarketRegime.TRENDING_DOWN: {
                Timeframe.DAILY: 1.5,
                Timeframe.HOURLY: 1.0,
                Timeframe.MINUTE_15: 0.5
            },
            MarketRegime.VOLATILE: {
                Timeframe.DAILY: 0.5,
                Timeframe.HOURLY: 1.0,
                Timeframe.MINUTE_15: 1.5
            },
            MarketRegime.RANGING: {
                Timeframe.DAILY: 1.0,
                Timeframe.HOURLY: 1.2,
                Timeframe.MINUTE_15: 0.8
            }
        }
        
        # Get regime-specific weights
        if current_regime and current_regime in regime_weights:
            regime_weight_map = regime_weights[current_regime]
        else:
            # Default: equal weighting
            regime_weight_map = {tf: 1.0 for tf in Timeframe}
        
        # Apply regime weights
        predictions = []
        weights = []
        
        for tf, tf_pred in timeframe_predictions.items():
            predictions.append(tf_pred['prediction'])
            base_weight = tf_pred['weight']
            regime_weight = regime_weight_map.get(tf, 1.0)
            weights.append(base_weight * regime_weight)
        
        # Normalize and ensemble
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        ensemble = sum(pred * w for pred, w in zip(predictions, weights))
        
        return ensemble


class HierarchicalForecaster:
    """
    Hierarchical forecasting: aggregate short-term predictions to long-term.
    """
    
    def __init__(self):
        """Initialize hierarchical forecaster."""
        logger.info("Initialized HierarchicalForecaster")
    
    def aggregate_predictions(
        self,
        short_term_predictions: np.ndarray,
        aggregation_method: str = "sum"
    ) -> np.ndarray:
        """
        Aggregate short-term predictions to long-term.
        
        Args:
            short_term_predictions: Array of short-term predictions
            aggregation_method: Aggregation method ('sum', 'mean', 'last')
        
        Returns:
            Aggregated prediction
        """
        if aggregation_method == "sum":
            return np.sum(short_term_predictions, axis=0)
        elif aggregation_method == "mean":
            return np.mean(short_term_predictions, axis=0)
        elif aggregation_method == "last":
            return short_term_predictions[-1]
        else:
            logger.warning(f"Unknown aggregation method: {aggregation_method}")
            return np.mean(short_term_predictions, axis=0)
    
    def reconcile_forecasts(
        self,
        short_term: np.ndarray,
        long_term: np.ndarray,
        method: str = "proportional"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconcile short-term and long-term forecasts to be consistent.
        
        Args:
            short_term: Short-term predictions
            long_term: Long-term prediction
            method: Reconciliation method
        
        Returns:
            Tuple of reconciled (short_term, long_term)
        """
        if method == "proportional":
            # Adjust short-term to match long-term aggregate
            short_term_sum = np.sum(short_term)
            
            if short_term_sum != 0:
                adjustment_factor = long_term / short_term_sum
                reconciled_short = short_term * adjustment_factor
            else:
                reconciled_short = short_term
            
            return reconciled_short, long_term
        
        elif method == "top_down":
            # Distribute long-term proportionally to short-term
            proportions = short_term / np.sum(short_term) if np.sum(short_term) != 0 else np.ones_like(short_term) / len(short_term)
            reconciled_short = long_term * proportions
            
            return reconciled_short, long_term
        
        else:
            return short_term, long_term


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    
    price_data = pd.DataFrame({
        'Close': 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, len(dates)))),
        'High': 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, len(dates)))) * 1.01,
        'Low': 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, len(dates)))) * 0.99,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    # Test regime detector
    print("Testing Regime Detector...")
    regime_detector = RegimeDetector(method="kmeans", n_regimes=4)
    regime_detector.fit(price_data)
    
    regimes = regime_detector.predict(price_data)
    print(f"Detected {len(np.unique(regimes))} unique regimes")
    print(f"Regime distribution: {np.bincount(regimes)}")
    
    # Test multi-timeframe predictor
    print("\nTesting Multi-Timeframe Predictor...")
    configs = [
        TimeframeConfig(Timeframe.DAILY, sequence_length=30, forecast_horizon=5, weight=1.5),
        TimeframeConfig(Timeframe.HOURLY, sequence_length=50, forecast_horizon=10, weight=1.0)
    ]
    
    predictor = MultiTimeframePredictor(
        timeframe_configs=configs,
        regime_detector=regime_detector,
        ensemble_method="regime_adaptive"
    )
    
    print(f"Initialized predictor with {len(configs)} timeframes")
    
    # Test hierarchical forecaster
    print("\nTesting Hierarchical Forecaster...")
    forecaster = HierarchicalForecaster()
    
    short_term = np.array([0.01, 0.02, -0.01, 0.03, 0.01])
    long_term = 0.05
    
    reconciled_short, reconciled_long = forecaster.reconcile_forecasts(
        short_term,
        long_term,
        method="proportional"
    )
    
    print(f"Original short-term sum: {np.sum(short_term):.4f}")
    print(f"Long-term target: {long_term:.4f}")
    print(f"Reconciled short-term sum: {np.sum(reconciled_short):.4f}")
