"""
Stacked Autoencoder for dimensionality reduction and feature extraction.
Uses custom GELU activation for improved gradient flow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) activation function.
    More sophisticated than ReLU, provides smooth gradients.
    """
    
    def __init__(self, approximate: bool = False):
        """
        Initialize GELU activation.
        
        Args:
            approximate: Use tanh approximation for speed
        """
        super(GELU, self).__init__()
        self.approximate = approximate
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GELU activation."""
        if self.approximate:
            # Faster approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            return 0.5 * x * (1 + torch.tanh(
                np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))
            ))
        else:
            # Exact GELU: x * Φ(x) where Φ is standard Gaussian CDF
            return x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))


class AutoencoderLayer(nn.Module):
    """Single autoencoder layer with encoder and decoder."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        activation: str = 'gelu',
        dropout: float = 0.2,
        use_batchnorm: bool = True
    ):
        """
        Initialize autoencoder layer.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden (latent) dimension
            activation: Activation function ('gelu', 'relu', 'tanh')
            dropout: Dropout rate
            use_batchnorm: Whether to use batch normalization
        """
        super(AutoencoderLayer, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('linear_enc', nn.Linear(input_dim, hidden_dim))
        
        if use_batchnorm:
            self.encoder.add_module('batchnorm_enc', nn.BatchNorm1d(hidden_dim))
        
        if activation == 'gelu':
            self.encoder.add_module('activation_enc', GELU())
        elif activation == 'relu':
            self.encoder.add_module('activation_enc', nn.ReLU())
        elif activation == 'tanh':
            self.encoder.add_module('activation_enc', nn.Tanh())
        
        if dropout > 0:
            self.encoder.add_module('dropout_enc', nn.Dropout(dropout))
        
        # Decoder
        self.decoder = nn.Sequential()
        self.decoder.add_module('linear_dec', nn.Linear(hidden_dim, input_dim))
        
        if use_batchnorm:
            self.decoder.add_module('batchnorm_dec', nn.BatchNorm1d(input_dim))
        
        # Output activation (usually linear or sigmoid depending on data normalization)
        self.decoder.add_module('activation_dec', nn.Tanh())
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.
        
        Args:
            x: Input tensor
        
        Returns:
            Tuple of (latent representation, reconstruction)
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return z, x_recon


class StackedAutoencoder(nn.Module):
    """
    Stacked Autoencoder for hierarchical feature learning.
    Multiple layers progressively reduce dimensionality.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        activation: str = 'gelu',
        dropout: float = 0.2,
        use_batchnorm: bool = True,
        tie_weights: bool = False
    ):
        """
        Initialize stacked autoencoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden dimensions for each layer
            activation: Activation function
            dropout: Dropout rate
            use_batchnorm: Whether to use batch normalization
            tie_weights: Whether to tie encoder/decoder weights (transpose)
        """
        super(StackedAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.tie_weights = tie_weights
        
        # Build layers
        self.layers = nn.ModuleList()
        
        dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layer = AutoencoderLayer(
                input_dim=dims[i],
                hidden_dim=dims[i + 1],
                activation=activation,
                dropout=dropout if i < len(hidden_dims) - 1 else 0,  # No dropout in last layer
                use_batchnorm=use_batchnorm
            )
            self.layers.append(layer)
        
        logger.info(
            f"Created StackedAutoencoder: {input_dim} -> "
            f"{' -> '.join(map(str, hidden_dims))}"
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input through all layers to get compressed representation.
        
        Args:
            x: Input tensor [batch_size, input_dim]
        
        Returns:
            Encoded representation [batch_size, hidden_dims[-1]]
        """
        h = x
        for layer in self.layers:
            h = layer.encode(h)
        return h
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation back to input space.
        
        Args:
            z: Latent representation [batch_size, hidden_dims[-1]]
        
        Returns:
            Reconstructed input [batch_size, input_dim]
        """
        h = z
        for layer in reversed(self.layers):
            h = layer.decode(h)
        return h
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode then decode.
        
        Args:
            x: Input tensor [batch_size, input_dim]
        
        Returns:
            Tuple of (latent representation, reconstruction)
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return z, x_recon
    
    def get_layer_reconstructions(
        self,
        x: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get intermediate representations and reconstructions for each layer.
        Useful for layer-wise pretraining.
        
        Args:
            x: Input tensor
        
        Returns:
            List of (latent, reconstruction) tuples for each layer
        """
        results = []
        h = x
        
        for layer in self.layers:
            z, recon = layer(h)
            results.append((z, recon))
            h = z  # Next layer takes encoded representation as input
        
        return results


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder (VAE) for probabilistic latent representations.
    Useful for generating synthetic samples and uncertainty estimation.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        activation: str = 'gelu',
        dropout: float = 0.2
    ):
        """
        Initialize VAE.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden dimensions
            latent_dim: Latent space dimension
            activation: Activation function
            dropout: Dropout rate
        """
        super(VariationalAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            encoder_layers.append(nn.Linear(dims[i], dims[i + 1]))
            encoder_layers.append(nn.BatchNorm1d(dims[i + 1]))
            
            if activation == 'gelu':
                encoder_layers.append(GELU())
            elif activation == 'relu':
                encoder_layers.append(nn.ReLU())
            
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        decoder_layers.append(nn.Linear(latent_dim, hidden_dims[-1]))
        
        dims_reversed = list(reversed(hidden_dims)) + [input_dim]
        
        for i in range(len(dims_reversed) - 1):
            decoder_layers.append(nn.BatchNorm1d(dims_reversed[i]))
            
            if activation == 'gelu':
                decoder_layers.append(GELU())
            elif activation == 'relu':
                decoder_layers.append(nn.ReLU())
            
            if dropout > 0 and i < len(dims_reversed) - 2:
                decoder_layers.append(nn.Dropout(dropout))
            
            decoder_layers.append(nn.Linear(dims_reversed[i], dims_reversed[i + 1]))
        
        # Output activation
        decoder_layers.append(nn.Tanh())
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        logger.info(
            f"Created VAE: {input_dim} -> "
            f"{' -> '.join(map(str, hidden_dims))} -> {latent_dim}"
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.
        
        Args:
            x: Input tensor [batch_size, input_dim]
        
        Returns:
            Tuple of (mean, log_variance)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + sigma * epsilon.
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to reconstruction.
        
        Args:
            z: Latent vector [batch_size, latent_dim]
        
        Returns:
            Reconstructed input [batch_size, input_dim]
        """
        return self.decoder(z)
    
    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Args:
            x: Input tensor [batch_size, input_dim]
        
        Returns:
            Tuple of (reconstruction, mean, log_variance)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Generate samples from the latent space.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate on
        
        Returns:
            Generated samples [num_samples, input_dim]
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z)


def vae_loss_function(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE loss function = Reconstruction loss + KL divergence.
    
    Args:
        recon_x: Reconstructed input
        x: Original input
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence (beta-VAE)
    
    Returns:
        Tuple of (total loss, reconstruction loss, KL divergence)
    """
    # Reconstruction loss (MSE or BCE depending on data)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = recon_loss + beta * kl_div
    
    return total_loss, recon_loss, kl_div


def train_autoencoder_layerwise(
    autoencoder: StackedAutoencoder,
    data_loader: torch.utils.data.DataLoader,
    num_epochs: int = 50,
    lr: float = 0.001,
    device: torch.device = torch.device('cpu')
) -> StackedAutoencoder:
    """
    Train stacked autoencoder using greedy layer-wise pretraining.
    Each layer is trained independently before moving to the next.
    
    Args:
        autoencoder: Stacked autoencoder to train
        data_loader: Training data loader
        num_epochs: Number of epochs per layer
        lr: Learning rate
        device: Device to train on
    
    Returns:
        Trained autoencoder
    """
    logger.info("Starting layer-wise pretraining")
    
    autoencoder.to(device)
    
    for layer_idx, layer in enumerate(autoencoder.layers):
        logger.info(f"Training layer {layer_idx + 1}/{len(autoencoder.layers)}")
        
        optimizer = torch.optim.Adam(layer.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        layer.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            
            for batch_idx, data in enumerate(data_loader):
                if isinstance(data, (list, tuple)):
                    data = data[0]
                
                data = data.to(device)
                
                # If not first layer, encode with previous layers
                if layer_idx > 0:
                    with torch.no_grad():
                        for prev_layer in autoencoder.layers[:layer_idx]:
                            data = prev_layer.encode(data)
                
                # Train current layer
                optimizer.zero_grad()
                z, recon = layer(data)
                loss = criterion(recon, data)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(data_loader)
            
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Layer {layer_idx + 1}, Epoch {epoch + 1}/{num_epochs}, "
                    f"Loss: {avg_loss:.6f}"
                )
    
    logger.info("Layer-wise pretraining complete")
    return autoencoder


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test StackedAutoencoder
    input_dim = 100
    hidden_dims = [64, 32, 16]
    
    autoencoder = StackedAutoencoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        activation='gelu',
        dropout=0.2
    )
    
    # Test forward pass
    x = torch.randn(32, input_dim)
    z, x_recon = autoencoder(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Latent shape: {z.shape}")
    print(f"Reconstruction shape: {x_recon.shape}")
    print(f"Reconstruction error: {F.mse_loss(x_recon, x).item():.6f}")
    
    # Test VAE
    vae = VariationalAutoencoder(
        input_dim=input_dim,
        hidden_dims=[64, 32],
        latent_dim=16,
        activation='gelu'
    )
    
    x_recon, mu, logvar = vae(x)
    total_loss, recon_loss, kl_div = vae_loss_function(x_recon, x, mu, logvar)
    
    print(f"\nVAE reconstruction error: {recon_loss.item():.6f}")
    print(f"VAE KL divergence: {kl_div.item():.6f}")
