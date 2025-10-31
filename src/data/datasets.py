"""
PyTorch Dataset classes for stock market time-series data.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class StockDataset(Dataset):
    """
    PyTorch Dataset for stock market time-series data.
    Creates sequences for LSTM training.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        sequence_length: int = 20,
        target_col: str = 'Close',
        prediction_horizon: int = 1
    ):
        """
        Initialize StockDataset.
        
        Args:
            data: DataFrame containing features
            sequence_length: Length of input sequences
            target_col: Column name for target variable
            prediction_horizon: Number of steps ahead to predict
        """
        self.data = data
        self.sequence_length = sequence_length
        self.target_col = target_col
        self.prediction_horizon = prediction_horizon
        
        # Separate features and target
        if target_col in data.columns:
            self.target_idx = data.columns.get_loc(target_col)
        else:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Convert to numpy arrays
        self.features = data.values.astype(np.float32)
        self.targets = data[target_col].values.astype(np.float32)
        
        # Calculate number of valid sequences
        self.n_samples = len(self.features) - sequence_length - prediction_horizon + 1
        
        logger.info(
            f"Initialized StockDataset: "
            f"{self.n_samples} samples, "
            f"{self.features.shape[1]} features, "
            f"sequence_length={sequence_length}"
        )
    
    def __len__(self) -> int:
        """Return number of samples."""
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
        
        Returns:
            Tuple of (features, target)
        """
        # Extract sequence
        seq_start = idx
        seq_end = idx + self.sequence_length
        
        # Features: [sequence_length, n_features]
        features = self.features[seq_start:seq_end]
        
        # Target: value at prediction_horizon steps ahead
        target_idx = seq_end + self.prediction_horizon - 1
        target = self.targets[target_idx]
        
        return (
            torch.from_numpy(features),
            torch.tensor(target, dtype=torch.float32)
        )
    
    def get_feature_names(self) -> list:
        """Return list of feature names."""
        return self.data.columns.tolist()
    
    def get_num_features(self) -> int:
        """Return number of features."""
        return self.features.shape[1]


class GANDataset(Dataset):
    """
    Special dataset for GAN training.
    Returns sequences without explicit targets (for unsupervised learning).
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        sequence_length: int = 20
    ):
        """
        Initialize GANDataset.
        
        Args:
            data: DataFrame containing features
            sequence_length: Length of sequences
        """
        self.data = data
        self.sequence_length = sequence_length
        
        # Convert to numpy arrays
        self.features = data.values.astype(np.float32)
        
        # Calculate number of valid sequences
        self.n_samples = len(self.features) - sequence_length + 1
        
        logger.info(
            f"Initialized GANDataset: "
            f"{self.n_samples} samples, "
            f"{self.features.shape[1]} features"
        )
    
    def __len__(self) -> int:
        """Return number of samples."""
        return self.n_samples
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a single sequence.
        
        Args:
            idx: Sample index
        
        Returns:
            Feature tensor of shape [sequence_length, n_features]
        """
        seq_start = idx
        seq_end = idx + self.sequence_length
        
        features = self.features[seq_start:seq_end]
        
        return torch.from_numpy(features)
    
    def get_num_features(self) -> int:
        """Return number of features."""
        return self.features.shape[1]


def create_dataloaders(
    train_data: pd.DataFrame,
    val_data: Optional[pd.DataFrame] = None,
    batch_size: int = 32,
    sequence_length: int = 20,
    target_col: str = 'Close',
    num_workers: int = 0,
    shuffle_train: bool = True,
    dataset_type: str = 'stock'
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and validation dataloaders.
    
    Args:
        train_data: Training DataFrame
        val_data: Optional validation DataFrame
        batch_size: Batch size
        sequence_length: Length of sequences
        target_col: Target column name
        num_workers: Number of workers for data loading
        shuffle_train: Whether to shuffle training data
        dataset_type: Type of dataset ('stock' or 'gan')
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    if dataset_type == 'stock':
        train_dataset = StockDataset(
            train_data,
            sequence_length=sequence_length,
            target_col=target_col
        )
        val_dataset = (
            StockDataset(val_data, sequence_length=sequence_length, target_col=target_col)
            if val_data is not None else None
        )
    elif dataset_type == 'gan':
        train_dataset = GANDataset(
            train_data,
            sequence_length=sequence_length
        )
        val_dataset = (
            GANDataset(val_data, sequence_length=sequence_length)
            if val_data is not None else None
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True  # Faster data transfer to GPU
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    logger.info(
        f"Created dataloaders: "
        f"train_size={len(train_dataset)}, "
        f"val_size={len(val_dataset) if val_dataset else 0}"
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=500)
    df = pd.DataFrame({
        'Close': np.random.randn(500).cumsum() + 100,
        'Volume': np.random.randint(1000000, 10000000, 500),
        'Feature1': np.random.randn(500),
        'Feature2': np.random.randn(500)
    }, index=dates)
    
    # Split into train/val
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_df,
        val_df,
        batch_size=32,
        sequence_length=20
    )
    
    # Test
    for batch_x, batch_y in train_loader:
        print(f"Batch X shape: {batch_x.shape}")  # [batch_size, seq_len, n_features]
        print(f"Batch Y shape: {batch_y.shape}")  # [batch_size]
        break
