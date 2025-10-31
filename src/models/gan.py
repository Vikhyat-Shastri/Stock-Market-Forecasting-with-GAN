"""
Wasserstein GAN with Gradient Penalty (WGAN-GP) for stock market forecasting.
Combines LSTM Generator and CNN Discriminator with Wasserstein loss.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, Callable
import logging
from pathlib import Path
import numpy as np

from src.models.generator import LSTMGenerator, AttentionLSTMGenerator
from src.models.discriminator import CNNDiscriminator, ResidualCNNDiscriminator

logger = logging.getLogger(__name__)


class WGAN_GP:
    """
    Wasserstein GAN with Gradient Penalty implementation.
    
    Key features:
    - Wasserstein loss for stable training
    - Gradient penalty for Lipschitz constraint
    - Support for different generator/discriminator architectures
    - GPU acceleration
    """
    
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        noise_dim: int = 100,
        lambda_gp: float = 10.0,
        n_critic: int = 5,
        device: torch.device = None
    ):
        """
        Initialize WGAN-GP.
        
        Args:
            generator: Generator network
            discriminator: Discriminator (critic) network
            noise_dim: Dimension of noise vector
            lambda_gp: Gradient penalty coefficient
            n_critic: Number of critic updates per generator update
            device: Device to train on
        """
        self.generator = generator
        self.discriminator = discriminator
        self.noise_dim = noise_dim
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Move models to device
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        
        # Initialize optimizers (will be set in configure_optimizers)
        self.g_optimizer = None
        self.d_optimizer = None
        
        # Training history
        self.history = {
            'd_loss': [],
            'g_loss': [],
            'gp': [],
            'wasserstein_distance': []
        }
        
        logger.info(
            f"Initialized WGAN-GP on {self.device}\n"
            f"  Generator parameters: {self._count_parameters(generator):,}\n"
            f"  Discriminator parameters: {self._count_parameters(discriminator):,}\n"
            f"  lambda_gp={lambda_gp}, n_critic={n_critic}"
        )
    
    def _count_parameters(self, model: nn.Module) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def configure_optimizers(
        self,
        g_lr: float = 0.0001,
        d_lr: float = 0.0004,
        betas: Tuple[float, float] = (0.0, 0.9)
    ):
        """
        Configure optimizers for generator and discriminator.
        
        Args:
            g_lr: Generator learning rate
            d_lr: Discriminator learning rate
            betas: Adam optimizer betas
        """
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=g_lr,
            betas=betas
        )
        
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=d_lr,
            betas=betas
        )
        
        logger.info(
            f"Configured optimizers: g_lr={g_lr}, d_lr={d_lr}, betas={betas}"
        )
    
    def gradient_penalty(
        self,
        real_data: torch.Tensor,
        fake_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate gradient penalty for WGAN-GP.
        
        Args:
            real_data: Real data samples
            fake_data: Generated fake samples
        
        Returns:
            Gradient penalty term
        """
        batch_size = real_data.size(0)
        
        # Random interpolation coefficient
        alpha = torch.rand(batch_size, 1, 1, device=self.device)
        
        # Create interpolated samples
        interpolated = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
        
        # Get discriminator output for interpolated samples
        d_interpolated = self.discriminator(interpolated)
        
        # Calculate gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Flatten gradients
        gradients = gradients.view(batch_size, -1)
        
        # Calculate gradient penalty
        gradient_norm = gradients.norm(2, dim=1)
        gp = ((gradient_norm - 1) ** 2).mean()
        
        return gp
    
    def train_discriminator(
        self,
        real_data: torch.Tensor
    ) -> Dict[str, float]:
        """
        Train discriminator for one step.
        
        Args:
            real_data: Real data batch
        
        Returns:
            Dictionary with loss metrics
        """
        self.discriminator.train()
        self.generator.eval()
        
        batch_size = real_data.size(0)
        
        # Zero gradients
        self.d_optimizer.zero_grad()
        
        # Get discriminator output for real data
        real_data = real_data.to(self.device)
        d_real = self.discriminator(real_data)
        
        # Generate fake data
        noise = torch.randn(batch_size, self.noise_dim, device=self.device)
        with torch.no_grad():
            fake_data = self.generator(noise)
        
        # Get discriminator output for fake data
        d_fake = self.discriminator(fake_data)
        
        # Calculate gradient penalty
        gp = self.gradient_penalty(real_data, fake_data)
        
        # Wasserstein loss: maximize d(real) - d(fake)
        # Equivalently, minimize d(fake) - d(real)
        d_loss = d_fake.mean() - d_real.mean() + self.lambda_gp * gp
        
        # Backward pass
        d_loss.backward()
        self.d_optimizer.step()
        
        # Calculate Wasserstein distance (without gradient penalty)
        wasserstein_distance = d_real.mean().item() - d_fake.mean().item()
        
        return {
            'd_loss': d_loss.item(),
            'gp': gp.item(),
            'wasserstein_distance': wasserstein_distance
        }
    
    def train_generator(self) -> Dict[str, float]:
        """
        Train generator for one step.
        
        Returns:
            Dictionary with loss metrics
        """
        self.generator.train()
        self.discriminator.eval()
        
        # Zero gradients
        self.g_optimizer.zero_grad()
        
        # Generate fake data
        noise = torch.randn(self.generator.batch_size, self.noise_dim, device=self.device)
        fake_data = self.generator(noise)
        
        # Get discriminator output
        d_fake = self.discriminator(fake_data)
        
        # Generator loss: maximize d(fake)
        # Equivalently, minimize -d(fake)
        g_loss = -d_fake.mean()
        
        # Backward pass
        g_loss.backward()
        self.g_optimizer.step()
        
        return {'g_loss': g_loss.item()}
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
        
        Returns:
            Dictionary with average epoch metrics
        """
        epoch_metrics = {
            'd_loss': [],
            'g_loss': [],
            'gp': [],
            'wasserstein_distance': []
        }
        
        for batch_idx, real_data in enumerate(dataloader):
            # Train discriminator
            d_metrics = self.train_discriminator(real_data)
            
            epoch_metrics['d_loss'].append(d_metrics['d_loss'])
            epoch_metrics['gp'].append(d_metrics['gp'])
            epoch_metrics['wasserstein_distance'].append(d_metrics['wasserstein_distance'])
            
            # Train generator every n_critic steps
            if (batch_idx + 1) % self.n_critic == 0:
                g_metrics = self.train_generator()
                epoch_metrics['g_loss'].append(g_metrics['g_loss'])
        
        # Calculate averages
        avg_metrics = {
            key: np.mean(values) if values else 0.0
            for key, values in epoch_metrics.items()
        }
        
        # Update history
        for key, value in avg_metrics.items():
            self.history[key].append(value)
        
        logger.info(
            f"Epoch {epoch:3d} | "
            f"D Loss: {avg_metrics['d_loss']:.4f} | "
            f"G Loss: {avg_metrics['g_loss']:.4f} | "
            f"GP: {avg_metrics['gp']:.4f} | "
            f"W-dist: {avg_metrics['wasserstein_distance']:.4f}"
        )
        
        return avg_metrics
    
    def fit(
        self,
        dataloader: DataLoader,
        epochs: int,
        save_dir: Optional[Path] = None,
        save_interval: int = 10,
        callback: Optional[Callable] = None
    ):
        """
        Train the WGAN-GP.
        
        Args:
            dataloader: Training data loader
            epochs: Number of training epochs
            save_dir: Directory to save checkpoints
            save_interval: Save model every N epochs
            callback: Optional callback function called after each epoch
        """
        if self.g_optimizer is None or self.d_optimizer is None:
            raise ValueError("Optimizers not configured. Call configure_optimizers() first.")
        
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(1, epochs + 1):
            # Train for one epoch
            metrics = self.train_epoch(dataloader, epoch)
            
            # Call callback if provided
            if callback is not None:
                callback(epoch, metrics, self)
            
            # Save checkpoint
            if save_dir and epoch % save_interval == 0:
                self.save_checkpoint(save_dir / f"checkpoint_epoch_{epoch}.pt", epoch)
        
        logger.info("Training complete")
    
    def generate_samples(
        self,
        n_samples: int = 100
    ) -> torch.Tensor:
        """
        Generate samples using the generator.
        
        Args:
            n_samples: Number of samples to generate
        
        Returns:
            Generated samples
        """
        self.generator.eval()
        
        with torch.no_grad():
            noise = torch.randn(n_samples, self.noise_dim, device=self.device)
            samples = self.generator(noise)
        
        return samples
    
    def save_checkpoint(self, path: Path, epoch: int):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'history': self.history
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: Path):
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        if self.g_optimizer and self.d_optimizer:
            self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
            self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        
        self.history = checkpoint['history']
        
        logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
        
        return checkpoint['epoch']


if __name__ == "__main__":
    # Test WGAN-GP
    logging.basicConfig(level=logging.INFO)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create generator and discriminator
    generator = LSTMGenerator(
        noise_dim=100,
        hidden_dim=256,
        num_layers=2,
        n_features=10,
        sequence_length=20
    )
    
    discriminator = CNNDiscriminator(
        n_features=10,
        sequence_length=20,
        base_filters=64,
        num_conv_layers=3
    )
    
    # Create WGAN-GP
    wgan = WGAN_GP(
        generator=generator,
        discriminator=discriminator,
        noise_dim=100,
        lambda_gp=10.0,
        n_critic=5,
        device=device
    )
    
    # Configure optimizers
    wgan.configure_optimizers(g_lr=0.0001, d_lr=0.0004)
    
    # Generate some samples
    samples = wgan.generate_samples(n_samples=10)
    print(f"Generated samples shape: {samples.shape}")
