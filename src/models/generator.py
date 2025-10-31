"""
LSTM-based Generator for GAN architecture.
Generates realistic stock price sequences from random noise.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LSTMGenerator(nn.Module):
    """
    LSTM-based Generator for time-series generation.
    
    Architecture:
    - Input: Random noise [batch, noise_dim]
    - LSTM layers with dropout
    - Output: Sequence of stock features [batch, sequence_length, n_features]
    """
    
    def __init__(
        self,
        noise_dim: int = 100,
        hidden_dim: int = 256,
        num_layers: int = 2,
        n_features: int = 1,
        sequence_length: int = 20,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        """
        Initialize LSTM Generator.
        
        Args:
            noise_dim: Dimension of input noise vector
            hidden_dim: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            n_features: Number of output features per timestep
            sequence_length: Length of generated sequences
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMGenerator, self).__init__()
        
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_features = n_features
        self.sequence_length = sequence_length
        self.bidirectional = bidirectional
        
        # Calculate direction multiplier for bidirectional LSTM
        self.num_directions = 2 if bidirectional else 1
        
        # Input projection: noise -> initial sequence
        self.input_projection = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output projection: LSTM hidden states -> features
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_features),
            nn.Tanh()  # Normalize output to [-1, 1]
        )
        
        # Initialize weights
        self._init_weights()
        
        logger.info(
            f"Initialized LSTMGenerator: "
            f"noise_dim={noise_dim}, hidden_dim={hidden_dim}, "
            f"num_layers={num_layers}, n_features={n_features}, "
            f"sequence_length={sequence_length}"
        )
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(
        self,
        noise: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass of generator.
        
        Args:
            noise: Random noise tensor [batch_size, noise_dim]
            hidden: Optional hidden state for LSTM
        
        Returns:
            Generated sequence [batch_size, sequence_length, n_features]
        """
        batch_size = noise.size(0)
        
        # Project noise to hidden dimension
        # [batch_size, noise_dim] -> [batch_size, hidden_dim]
        projected = self.input_projection(noise)
        
        # Repeat projected noise for each timestep
        # [batch_size, hidden_dim] -> [batch_size, sequence_length, hidden_dim]
        lstm_input = projected.unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        # Pass through LSTM
        # [batch_size, sequence_length, hidden_dim * num_directions]
        lstm_out, hidden = self.lstm(lstm_input, hidden)
        
        # Project to output features
        # [batch_size, sequence_length, hidden_dim * num_directions] 
        # -> [batch_size, sequence_length, n_features]
        output = self.output_projection(lstm_out)
        
        return output
    
    def generate(
        self,
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Generate random sequences.
        
        Args:
            batch_size: Number of sequences to generate
            device: Device to generate on
        
        Returns:
            Generated sequences
        """
        with torch.no_grad():
            noise = torch.randn(batch_size, self.noise_dim, device=device)
            return self.forward(noise)


class AttentionLSTMGenerator(nn.Module):
    """
    LSTM Generator with attention mechanism for better sequence generation.
    """
    
    def __init__(
        self,
        noise_dim: int = 100,
        hidden_dim: int = 256,
        num_layers: int = 2,
        n_features: int = 1,
        sequence_length: int = 20,
        dropout: float = 0.2,
        num_heads: int = 4
    ):
        """
        Initialize Attention LSTM Generator.
        
        Args:
            noise_dim: Dimension of input noise
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            n_features: Number of output features
            sequence_length: Length of sequences
            dropout: Dropout rate
            num_heads: Number of attention heads
        """
        super(AttentionLSTMGenerator, self).__init__()
        
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_features = n_features
        self.sequence_length = sequence_length
        
        # Input projection
        self.input_projection = nn.Linear(noise_dim, hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_features),
            nn.Tanh()
        )
        
        self._init_weights()
        
        logger.info(
            f"Initialized AttentionLSTMGenerator with {num_heads} attention heads"
        )
    
    def _init_weights(self):
        """Initialize weights."""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention.
        
        Args:
            noise: Input noise [batch_size, noise_dim]
        
        Returns:
            Generated sequence [batch_size, sequence_length, n_features]
        """
        batch_size = noise.size(0)
        
        # Project and repeat noise
        projected = self.input_projection(noise)
        lstm_input = projected.unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        # LSTM
        lstm_out, _ = self.lstm(lstm_input)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Add residual connection
        combined = lstm_out + attn_out
        
        # Project to output
        output = self.output_projection(combined)
        
        return output


if __name__ == "__main__":
    # Test generator
    logging.basicConfig(level=logging.INFO)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test basic generator
    generator = LSTMGenerator(
        noise_dim=100,
        hidden_dim=256,
        num_layers=2,
        n_features=10,
        sequence_length=20,
        dropout=0.2
    ).to(device)
    
    # Generate sample
    noise = torch.randn(32, 100, device=device)
    output = generator(noise)
    
    print(f"Generator output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test attention generator
    attn_generator = AttentionLSTMGenerator(
        noise_dim=100,
        hidden_dim=256,
        n_features=10,
        sequence_length=20
    ).to(device)
    
    output_attn = attn_generator(noise)
    print(f"Attention generator output shape: {output_attn.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in generator.parameters())
    trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
