"""
Transformer-based generator for time-series forecasting.
Alternative to LSTM with multi-head self-attention mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np
import math
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer.
    Injects information about position in sequence.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Dimension of model embeddings
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Use sine and cosine functions of different frequencies
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    Allows model to attend to different representation subspaces.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        """
        Initialize multi-head attention.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear transformations for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
    
    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.
        
        Args:
            Q: Query tensor [batch_size, num_heads, seq_len, d_k]
            K: Key tensor [batch_size, num_heads, seq_len, d_k]
            V: Value tensor [batch_size, num_heads, seq_len, d_k]
            mask: Attention mask
        
        Returns:
            Tuple of (attention output, attention weights)
        """
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute weighted sum of values
        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through multi-head attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Attention mask
        
        Returns:
            Tuple of (output, attention weights)
        """
        batch_size = query.size(0)
        
        # Linear transformations and reshape for multi-head attention
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        output = self.W_o(attn_output)
        
        return output, attn_weights


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    Applied to each position independently and identically.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        """
        Initialize feed-forward network.
        
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension (usually 4 * d_model)
            dropout: Dropout rate
            activation: Activation function ('gelu' or 'relu')
        """
        super(FeedForward, self).__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        """
        Initialize transformer encoder layer.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            activation: Activation function
        """
        super(TransformerEncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through encoder layer.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask
        
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerGenerator(nn.Module):
    """
    Transformer-based generator for time-series forecasting.
    Uses multi-head self-attention instead of recurrent layers.
    """
    
    def __init__(
        self,
        noise_dim: int,
        feature_dim: int,
        seq_length: int,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        """
        Initialize transformer generator.
        
        Args:
            noise_dim: Dimension of input noise vector
            feature_dim: Dimension of output features
            seq_length: Length of output sequence
            d_model: Model dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            activation: Activation function
        """
        super(TransformerGenerator, self).__init__()
        
        self.noise_dim = noise_dim
        self.feature_dim = feature_dim
        self.seq_length = seq_length
        self.d_model = d_model
        
        # Project noise to model dimension
        self.noise_projection = nn.Linear(noise_dim, d_model * seq_length)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, seq_length, dropout)
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, activation)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, feature_dim),
            nn.Tanh()
        )
        
        self._init_weights()
        
        logger.info(
            f"Created TransformerGenerator: noise_dim={noise_dim}, "
            f"seq_length={seq_length}, d_model={d_model}, "
            f"num_layers={num_layers}, num_heads={num_heads}"
        )
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.noise_projection.weight)
        for layer in self.output_projection:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
    
    def forward(
        self,
        noise: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate time-series from noise.
        
        Args:
            noise: Input noise [batch_size, noise_dim]
            mask: Attention mask
        
        Returns:
            Generated sequence [batch_size, seq_length, feature_dim]
        """
        batch_size = noise.size(0)
        
        # Project noise and reshape to sequence
        x = self.noise_projection(noise)
        x = x.view(batch_size, self.seq_length, self.d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, mask)
        
        # Project to output dimension
        output = self.output_projection(x)
        
        return output


class TransformerDiscriminator(nn.Module):
    """
    Transformer-based discriminator for sequence classification.
    Determines if a sequence is real or fake.
    """
    
    def __init__(
        self,
        feature_dim: int,
        seq_length: int,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        """
        Initialize transformer discriminator.
        
        Args:
            feature_dim: Dimension of input features
            seq_length: Length of input sequence
            d_model: Model dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            activation: Activation function
        """
        super(TransformerDiscriminator, self).__init__()
        
        self.feature_dim = feature_dim
        self.seq_length = seq_length
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(feature_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, seq_length, dropout)
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, activation)
            for _ in range(num_layers)
        ])
        
        # Classification head (global average pooling + FC)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        self._init_weights()
        
        logger.info(
            f"Created TransformerDiscriminator: feature_dim={feature_dim}, "
            f"seq_length={seq_length}, d_model={d_model}, "
            f"num_layers={num_layers}, num_heads={num_heads}"
        )
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.input_projection.weight)
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Classify sequence as real or fake.
        
        Args:
            x: Input sequence [batch_size, seq_length, feature_dim]
            mask: Attention mask
        
        Returns:
            Classification scores [batch_size, 1]
        """
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, mask)
        
        # Global average pooling across sequence
        x = x.mean(dim=1)
        
        # Classify
        output = self.classifier(x)
        
        return output


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test TransformerGenerator
    batch_size = 32
    noise_dim = 100
    feature_dim = 10
    seq_length = 50
    
    generator = TransformerGenerator(
        noise_dim=noise_dim,
        feature_dim=feature_dim,
        seq_length=seq_length,
        d_model=256,
        num_layers=4,
        num_heads=8
    )
    
    noise = torch.randn(batch_size, noise_dim)
    generated = generator(noise)
    
    print(f"Generated sequence shape: {generated.shape}")
    print(f"Expected shape: [{batch_size}, {seq_length}, {feature_dim}]")
    
    # Test TransformerDiscriminator
    discriminator = TransformerDiscriminator(
        feature_dim=feature_dim,
        seq_length=seq_length,
        d_model=256,
        num_layers=4,
        num_heads=8
    )
    
    scores = discriminator(generated)
    print(f"Discriminator scores shape: {scores.shape}")
    print(f"Expected shape: [{batch_size}, 1]")
    
    # Count parameters
    gen_params = sum(p.numel() for p in generator.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())
    
    print(f"\nGenerator parameters: {gen_params:,}")
    print(f"Discriminator parameters: {disc_params:,}")
