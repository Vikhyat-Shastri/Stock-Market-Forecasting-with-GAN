"""
CNN-based Discriminator for GAN architecture.
Distinguishes between real and generated stock sequences.
"""

import torch
import torch.nn as nn
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class CNNDiscriminator(nn.Module):
    """
    1D CNN-based Discriminator for time-series data.
    
    Architecture:
    - Input: Sequence [batch, sequence_length, n_features]
    - Conv1D layers with LeakyReLU and BatchNorm
    - Fully connected layers
    - Output: Real/Fake probability [batch, 1]
    """
    
    def __init__(
        self,
        n_features: int = 1,
        sequence_length: int = 20,
        base_filters: int = 64,
        num_conv_layers: int = 3,
        fc_hidden_dim: int = 128,
        dropout: float = 0.3,
        use_spectral_norm: bool = False
    ):
        """
        Initialize CNN Discriminator.
        
        Args:
            n_features: Number of input features per timestep
            sequence_length: Length of input sequences
            base_filters: Number of filters in first conv layer
            num_conv_layers: Number of convolutional layers
            fc_hidden_dim: Hidden dimension in fully connected layers
            dropout: Dropout probability
            use_spectral_norm: Whether to use spectral normalization
        """
        super(CNNDiscriminator, self).__init__()
        
        self.n_features = n_features
        self.sequence_length = sequence_length
        self.base_filters = base_filters
        
        # Build convolutional layers
        conv_layers = []
        in_channels = n_features
        out_channels = base_filters
        
        for i in range(num_conv_layers):
            # Convolutional layer
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=5,
                stride=2,
                padding=2
            )
            
            # Apply spectral normalization if requested
            if use_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            
            conv_layers.append(conv)
            conv_layers.append(nn.LeakyReLU(0.2))
            
            # Add batch normalization for all but first layer
            if i > 0:
                conv_layers.append(nn.BatchNorm1d(out_channels))
            
            conv_layers.append(nn.Dropout(dropout))
            
            # Update channels for next layer
            in_channels = out_channels
            out_channels = min(out_channels * 2, 512)  # Cap at 512
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Calculate size after convolutions
        # Each conv layer with stride=2 reduces length by ~half
        conv_out_length = sequence_length
        for _ in range(num_conv_layers):
            conv_out_length = (conv_out_length + 2 * 2 - 5) // 2 + 1
        
        self.conv_out_dim = in_channels * conv_out_length
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.conv_out_dim, fc_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, fc_hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim // 2, 1)
            # No sigmoid - will use BCEWithLogitsLoss or Wasserstein loss
        )
        
        # Initialize weights
        self._init_weights()
        
        logger.info(
            f"Initialized CNNDiscriminator: "
            f"n_features={n_features}, sequence_length={sequence_length}, "
            f"base_filters={base_filters}, num_conv_layers={num_conv_layers}, "
            f"conv_out_dim={self.conv_out_dim}"
        )
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of discriminator.
        
        Args:
            x: Input sequence [batch_size, sequence_length, n_features]
        
        Returns:
            Discriminator output [batch_size, 1]
        """
        # Transpose for Conv1d: [batch, n_features, sequence_length]
        x = x.transpose(1, 2)
        
        # Pass through conv layers
        conv_out = self.conv_layers(x)
        
        # Flatten
        flattened = conv_out.view(conv_out.size(0), -1)
        
        # Pass through FC layers
        output = self.fc_layers(flattened)
        
        return output


class ResidualBlock(nn.Module):
    """Residual block for discriminator."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1
    ):
        """
        Initialize residual block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size
            stride: Stride
        """
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU(0.2)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResidualCNNDiscriminator(nn.Module):
    """
    CNN Discriminator with residual connections for better gradient flow.
    """
    
    def __init__(
        self,
        n_features: int = 1,
        sequence_length: int = 20,
        base_filters: int = 64,
        num_blocks: int = 3,
        fc_hidden_dim: int = 128,
        dropout: float = 0.3
    ):
        """
        Initialize Residual CNN Discriminator.
        
        Args:
            n_features: Number of input features
            sequence_length: Sequence length
            base_filters: Base number of filters
            num_blocks: Number of residual blocks
            fc_hidden_dim: FC hidden dimension
            dropout: Dropout rate
        """
        super(ResidualCNNDiscriminator, self).__init__()
        
        self.n_features = n_features
        self.sequence_length = sequence_length
        
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv1d(n_features, base_filters, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(base_filters),
            nn.LeakyReLU(0.2)
        )
        
        # Residual blocks
        blocks = []
        in_channels = base_filters
        out_channels = base_filters
        
        for i in range(num_blocks):
            stride = 2 if i < num_blocks - 1 else 1
            blocks.append(ResidualBlock(in_channels, out_channels, stride=stride))
            in_channels = out_channels
            out_channels = min(out_channels * 2, 512)
        
        self.residual_blocks = nn.Sequential(*blocks)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(in_channels, fc_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, 1)
        )
        
        self._init_weights()
        
        logger.info(
            f"Initialized ResidualCNNDiscriminator with {num_blocks} residual blocks"
        )
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input [batch_size, sequence_length, n_features]
        
        Returns:
            Output [batch_size, 1]
        """
        # Transpose for Conv1d
        x = x.transpose(1, 2)
        
        # Initial conv
        x = self.initial_conv(x)
        
        # Residual blocks
        x = self.residual_blocks(x)
        
        # Global average pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # FC layers
        output = self.fc_layers(x)
        
        return output


if __name__ == "__main__":
    # Test discriminator
    logging.basicConfig(level=logging.INFO)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test basic discriminator
    discriminator = CNNDiscriminator(
        n_features=10,
        sequence_length=20,
        base_filters=64,
        num_conv_layers=3
    ).to(device)
    
    # Test with sample input
    sample_input = torch.randn(32, 20, 10, device=device)
    output = discriminator(sample_input)
    
    print(f"Discriminator output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test residual discriminator
    res_discriminator = ResidualCNNDiscriminator(
        n_features=10,
        sequence_length=20,
        base_filters=64,
        num_blocks=3
    ).to(device)
    
    output_res = res_discriminator(sample_input)
    print(f"Residual discriminator output shape: {output_res.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in discriminator.parameters())
    trainable_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
