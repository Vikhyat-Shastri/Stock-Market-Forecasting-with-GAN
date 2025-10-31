"""
Evaluation metrics for stock market prediction and GAN training.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Squared Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        MSE value
    """
    return mean_squared_error(y_true, y_pred)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        MAE value
    """
    return mean_absolute_error(y_true, y_pred)


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero
    
    Returns:
        MAPE value (as percentage)
    """
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R-squared (coefficient of determination).
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        R² value
    """
    return r2_score(y_true, y_pred)


def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate directional accuracy (percentage of correct direction predictions).
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Directional accuracy (0-100)
    """
    # Calculate direction of change
    true_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0
    
    # Calculate accuracy
    accuracy = np.mean(true_direction == pred_direction) * 100
    
    return accuracy


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio for returns.
    
    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year (252 for daily)
    
    Returns:
        Sharpe ratio
    """
    excess_returns = returns - (risk_free_rate / periods_per_year)
    
    if excess_returns.std() == 0:
        return 0.0
    
    sharpe = np.sqrt(periods_per_year) * (excess_returns.mean() / excess_returns.std())
    
    return sharpe


def calculate_max_drawdown(prices: np.ndarray) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        prices: Array of prices
    
    Returns:
        Maximum drawdown (as percentage)
    """
    # Calculate cumulative maximum
    cummax = np.maximum.accumulate(prices)
    
    # Calculate drawdown
    drawdown = (prices - cummax) / cummax
    
    # Return maximum drawdown
    return drawdown.min() * 100


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prices: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        prices: Optional price data for financial metrics
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'mse': calculate_mse(y_true, y_pred),
        'rmse': calculate_rmse(y_true, y_pred),
        'mae': calculate_mae(y_true, y_pred),
        'mape': calculate_mape(y_true, y_pred),
        'r2': calculate_r2(y_true, y_pred),
        'directional_accuracy': calculate_directional_accuracy(y_true, y_pred)
    }
    
    # Add financial metrics if prices provided
    if prices is not None:
        returns = np.diff(prices) / prices[:-1]
        metrics['sharpe_ratio'] = calculate_sharpe_ratio(returns)
        metrics['max_drawdown'] = calculate_max_drawdown(prices)
    
    return metrics


def wasserstein_distance(
    real_scores: torch.Tensor,
    fake_scores: torch.Tensor
) -> float:
    """
    Calculate Wasserstein distance (Earth Mover's Distance).
    
    Args:
        real_scores: Discriminator scores for real data
        fake_scores: Discriminator scores for fake data
    
    Returns:
        Wasserstein distance
    """
    return (real_scores.mean() - fake_scores.mean()).item()


def inception_score(
    predictions: torch.Tensor,
    splits: int = 10,
    eps: float = 1e-16
) -> Tuple[float, float]:
    """
    Calculate Inception Score for generated samples.
    
    Args:
        predictions: Predicted probabilities for generated samples
        splits: Number of splits for calculating std
        eps: Small value to avoid log(0)
    
    Returns:
        Tuple of (mean IS, std IS)
    """
    # Split predictions
    split_scores = []
    
    N = predictions.shape[0]
    split_size = N // splits
    
    for i in range(splits):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i < splits - 1 else N
        
        part = predictions[start_idx:end_idx]
        
        # Calculate p(y)
        py = part.mean(dim=0)
        
        # Calculate KL divergence
        kl = part * (torch.log(part + eps) - torch.log(py + eps))
        kl = kl.sum(dim=1).mean()
        
        split_scores.append(torch.exp(kl).item())
    
    return np.mean(split_scores), np.std(split_scores)


class MetricsTracker:
    """
    Track and aggregate metrics over time.
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics_history = []
    
    def update(self, metrics: Dict[str, float], epoch: int):
        """
        Update metrics for an epoch.
        
        Args:
            metrics: Dictionary of metrics
            epoch: Epoch number
        """
        entry = {'epoch': epoch, **metrics}
        self.metrics_history.append(entry)
    
    def get_history(self) -> pd.DataFrame:
        """
        Get metrics history as DataFrame.
        
        Returns:
            DataFrame with metrics history
        """
        return pd.DataFrame(self.metrics_history)
    
    def get_best_epoch(self, metric: str, mode: str = 'min') -> int:
        """
        Get epoch with best metric value.
        
        Args:
            metric: Metric name
            mode: 'min' or 'max'
        
        Returns:
            Best epoch number
        """
        df = self.get_history()
        
        if mode == 'min':
            best_idx = df[metric].idxmin()
        else:
            best_idx = df[metric].idxmax()
        
        return df.loc[best_idx, 'epoch']
    
    def summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all metrics.
        
        Returns:
            Dictionary with summary stats
        """
        df = self.get_history()
        
        summary = {}
        for col in df.columns:
            if col != 'epoch':
                summary[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'final': df[col].iloc[-1]
                }
        
        return summary


if __name__ == "__main__":
    # Test metrics
    np.random.seed(42)
    
    y_true = np.random.randn(100).cumsum() + 100
    y_pred = y_true + np.random.randn(100) * 2
    
    metrics = calculate_metrics(y_true, y_pred, prices=y_true)
    
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
