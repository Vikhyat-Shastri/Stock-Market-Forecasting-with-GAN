"""
Metropolis-Hastings GAN (MH-GAN) implementation.
Uses Metropolis-Hastings algorithm to improve sample quality through rejection sampling.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Callable
import logging
import numpy as np

logger = logging.getLogger(__name__)


class MetropolisHastingsGAN(nn.Module):
    """
    MH-GAN: Metropolis-Hastings sampling applied to GANs.
    
    Key idea: Generated samples are accepted/rejected based on discriminator scores,
    creating a Markov chain that converges to the true data distribution.
    
    Reference: "Metropolis-Hastings Generative Adversarial Networks" (ICML 2018)
    """
    
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        num_mh_steps: int = 5,
        temperature: float = 1.0,
        use_adaptive_temp: bool = True
    ):
        """
        Initialize MH-GAN.
        
        Args:
            generator: Generator network
            discriminator: Discriminator network
            num_mh_steps: Number of MH sampling steps
            temperature: Temperature for acceptance probability
            use_adaptive_temp: Whether to use adaptive temperature scheduling
        """
        super(MetropolisHastingsGAN, self).__init__()
        
        self.generator = generator
        self.discriminator = discriminator
        self.num_mh_steps = num_mh_steps
        self.temperature = temperature
        self.use_adaptive_temp = use_adaptive_temp
        
        # Statistics for adaptive temperature
        self.acceptance_rate_history = []
        self.target_acceptance_rate = 0.234  # Optimal for high dimensions
        
        logger.info(
            f"Initialized MH-GAN with {num_mh_steps} MH steps, "
            f"temperature={temperature}"
        )
    
    def acceptance_probability(
        self,
        current_score: torch.Tensor,
        proposed_score: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute acceptance probability using Metropolis-Hastings criterion.
        
        Args:
            current_score: Discriminator score for current sample
            proposed_score: Discriminator score for proposed sample
        
        Returns:
            Acceptance probability [batch_size]
        """
        # MH acceptance: min(1, exp((D(x') - D(x)) / T))
        # Higher discriminator score = more realistic
        delta = proposed_score - current_score
        prob = torch.exp(delta / self.temperature)
        prob = torch.min(prob, torch.ones_like(prob))
        
        return prob
    
    def mh_sample(
        self,
        initial_noise: torch.Tensor,
        return_chain: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Generate samples using Metropolis-Hastings algorithm.
        
        Args:
            initial_noise: Initial noise vectors [batch_size, noise_dim]
            return_chain: Whether to return full Markov chain
        
        Returns:
            Tuple of (final samples, final noise, acceptance rate)
        """
        batch_size = initial_noise.size(0)
        device = initial_noise.device
        
        # Generate initial samples
        current_noise = initial_noise.clone()
        current_samples = self.generator(current_noise)
        
        with torch.no_grad():
            current_scores = self.discriminator(current_samples)
        
        # Track chain and acceptance
        chain = [current_samples] if return_chain else []
        accepted = 0
        total = 0
        
        # MH sampling loop
        for step in range(self.num_mh_steps):
            # Propose new noise by adding Gaussian perturbation
            noise_std = 0.1  # Standard deviation for noise perturbation
            noise_perturbation = torch.randn_like(current_noise) * noise_std
            proposed_noise = current_noise + noise_perturbation
            
            # Generate proposed samples
            proposed_samples = self.generator(proposed_noise)
            
            with torch.no_grad():
                proposed_scores = self.discriminator(proposed_samples)
            
            # Compute acceptance probability
            accept_prob = self.acceptance_probability(current_scores, proposed_scores)
            
            # Accept/reject
            uniform_samples = torch.rand(batch_size, 1, 1).to(device)
            accept_mask = (uniform_samples < accept_prob.view(batch_size, 1, 1)).float()
            
            # Update samples and scores
            current_samples = (
                accept_mask * proposed_samples + 
                (1 - accept_mask) * current_samples
            )
            current_scores = (
                accept_mask.squeeze() * proposed_scores + 
                (1 - accept_mask.squeeze()) * current_scores
            )
            current_noise = (
                accept_mask.squeeze(-1) * proposed_noise + 
                (1 - accept_mask.squeeze(-1)) * current_noise
            )
            
            # Track acceptance rate
            accepted += accept_mask.sum().item()
            total += batch_size
            
            if return_chain:
                chain.append(current_samples.clone())
        
        acceptance_rate = accepted / total if total > 0 else 0.0
        self.acceptance_rate_history.append(acceptance_rate)
        
        # Adaptive temperature
        if self.use_adaptive_temp and len(self.acceptance_rate_history) > 10:
            self._update_temperature()
        
        if return_chain:
            return torch.stack(chain), current_noise, acceptance_rate
        else:
            return current_samples, current_noise, acceptance_rate
    
    def _update_temperature(self):
        """Update temperature based on acceptance rate."""
        recent_rate = np.mean(self.acceptance_rate_history[-10:])
        
        if recent_rate < self.target_acceptance_rate - 0.05:
            # Too many rejections, increase temperature
            self.temperature *= 1.05
        elif recent_rate > self.target_acceptance_rate + 0.05:
            # Too many acceptances, decrease temperature
            self.temperature *= 0.95
        
        # Clamp temperature
        self.temperature = np.clip(self.temperature, 0.1, 10.0)
    
    def generate(
        self,
        batch_size: int,
        noise_dim: int,
        device: torch.device,
        use_mh: bool = True
    ) -> Tuple[torch.Tensor, float]:
        """
        Generate samples with optional MH sampling.
        
        Args:
            batch_size: Number of samples to generate
            noise_dim: Dimension of noise vector
            device: Device to generate on
            use_mh: Whether to use MH sampling
        
        Returns:
            Tuple of (generated samples, acceptance rate)
        """
        noise = torch.randn(batch_size, noise_dim).to(device)
        
        if use_mh:
            samples, _, acceptance_rate = self.mh_sample(noise)
            return samples, acceptance_rate
        else:
            samples = self.generator(noise)
            return samples, 1.0
    
    def forward(
        self,
        noise: torch.Tensor,
        use_mh: bool = True
    ) -> Tuple[torch.Tensor, float]:
        """
        Forward pass through MH-GAN.
        
        Args:
            noise: Input noise vectors
            use_mh: Whether to use MH sampling
        
        Returns:
            Tuple of (generated samples, acceptance rate)
        """
        if use_mh:
            samples, _, acceptance_rate = self.mh_sample(noise)
            return samples, acceptance_rate
        else:
            samples = self.generator(noise)
            return samples, 1.0


class EnergyBasedGAN(nn.Module):
    """
    Energy-Based GAN (EBGAN) using energy function instead of probability.
    Discriminator is treated as an energy function.
    
    This provides an alternative formulation that can be combined with MH sampling.
    """
    
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        margin: float = 1.0,
        pull_away_weight: float = 0.1
    ):
        """
        Initialize EBGAN.
        
        Args:
            generator: Generator network
            discriminator: Discriminator (energy function)
            margin: Margin for hinge loss
            pull_away_weight: Weight for pull-away term
        """
        super(EnergyBasedGAN, self).__init__()
        
        self.generator = generator
        self.discriminator = discriminator
        self.margin = margin
        self.pull_away_weight = pull_away_weight
        
        logger.info(f"Initialized EBGAN with margin={margin}")
    
    def discriminator_loss(
        self,
        real_data: torch.Tensor,
        fake_data: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute discriminator loss (energy-based).
        
        Args:
            real_data: Real samples
            fake_data: Generated samples
        
        Returns:
            Tuple of (loss, metrics dict)
        """
        # Energy for real data should be low
        energy_real = self.discriminator(real_data)
        
        # Energy for fake data should be high (at least margin)
        energy_fake = self.discriminator(fake_data.detach())
        
        # Hinge loss
        loss_real = energy_real.mean()
        loss_fake = torch.clamp(self.margin - energy_fake, min=0).mean()
        
        loss = loss_real + loss_fake
        
        metrics = {
            'energy_real': energy_real.mean().item(),
            'energy_fake': energy_fake.mean().item(),
            'loss_real': loss_real.item(),
            'loss_fake': loss_fake.item()
        }
        
        return loss, metrics
    
    def generator_loss(
        self,
        fake_data: torch.Tensor,
        use_pull_away: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute generator loss.
        
        Args:
            fake_data: Generated samples
            use_pull_away: Whether to use pull-away term
        
        Returns:
            Tuple of (loss, metrics dict)
        """
        # Generator wants to minimize energy of fake data
        energy_fake = self.discriminator(fake_data)
        loss = energy_fake.mean()
        
        metrics = {
            'gen_energy': energy_fake.mean().item()
        }
        
        # Pull-away term: encourage diversity
        if use_pull_away:
            batch_size = fake_data.size(0)
            fake_flat = fake_data.view(batch_size, -1)
            
            # Normalize
            fake_norm = fake_flat / (fake_flat.norm(dim=1, keepdim=True) + 1e-8)
            
            # Compute cosine similarity matrix
            similarity = torch.matmul(fake_norm, fake_norm.t())
            
            # Pull-away term: minimize off-diagonal elements
            mask = (1 - torch.eye(batch_size)).to(fake_data.device)
            pull_away = (similarity * mask).pow(2).sum() / (batch_size * (batch_size - 1))
            
            loss = loss + self.pull_away_weight * pull_away
            metrics['pull_away'] = pull_away.item()
        
        return loss, metrics
    
    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """Generate samples."""
        return self.generator(noise)


class LangevinSampler:
    """
    Langevin dynamics sampler for energy-based models.
    Can be used with EBGAN for improved sample quality.
    """
    
    def __init__(
        self,
        energy_function: Callable,
        step_size: float = 0.01,
        num_steps: int = 10,
        noise_scale: float = 0.01
    ):
        """
        Initialize Langevin sampler.
        
        Args:
            energy_function: Function that computes energy
            step_size: Step size for gradient descent
            num_steps: Number of sampling steps
            noise_scale: Scale of noise to add
        """
        self.energy_function = energy_function
        self.step_size = step_size
        self.num_steps = num_steps
        self.noise_scale = noise_scale
    
    def sample(
        self,
        initial_samples: torch.Tensor,
        return_trajectory: bool = False
    ) -> torch.Tensor:
        """
        Sample using Langevin dynamics.
        
        Args:
            initial_samples: Initial samples
            return_trajectory: Whether to return full trajectory
        
        Returns:
            Refined samples
        """
        samples = initial_samples.clone()
        samples.requires_grad_(True)
        
        trajectory = [samples.detach().clone()] if return_trajectory else []
        
        for step in range(self.num_steps):
            # Compute energy and gradient
            energy = self.energy_function(samples)
            grad = torch.autograd.grad(
                energy.sum(),
                samples,
                create_graph=False
            )[0]
            
            # Langevin update: x = x - ε∇E(x) + √(2ε)z
            noise = torch.randn_like(samples) * self.noise_scale
            samples = samples - self.step_size * grad + noise
            samples = samples.detach()
            samples.requires_grad_(True)
            
            if return_trajectory:
                trajectory.append(samples.detach().clone())
        
        samples = samples.detach()
        
        if return_trajectory:
            return torch.stack(trajectory)
        else:
            return samples


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy generator and discriminator
    from src.models.generator import LSTMGenerator
    from src.models.discriminator import CNNDiscriminator
    
    noise_dim = 100
    feature_dim = 10
    seq_length = 50
    hidden_dim = 128
    
    generator = LSTMGenerator(
        noise_dim=noise_dim,
        hidden_dim=hidden_dim,
        feature_dim=feature_dim,
        seq_length=seq_length,
        num_layers=2
    )
    
    discriminator = CNNDiscriminator(
        feature_dim=feature_dim,
        seq_length=seq_length,
        base_filters=64,
        num_conv_layers=3
    )
    
    # Test MH-GAN
    mh_gan = MetropolisHastingsGAN(
        generator=generator,
        discriminator=discriminator,
        num_mh_steps=5
    )
    
    batch_size = 32
    noise = torch.randn(batch_size, noise_dim)
    
    samples, acceptance_rate = mh_gan(noise, use_mh=True)
    print(f"Generated samples shape: {samples.shape}")
    print(f"Acceptance rate: {acceptance_rate:.3f}")
    
    # Test EBGAN
    ebgan = EnergyBasedGAN(
        generator=generator,
        discriminator=discriminator,
        margin=1.0
    )
    
    fake_data = ebgan(noise)
    real_data = torch.randn_like(fake_data)
    
    d_loss, d_metrics = ebgan.discriminator_loss(real_data, fake_data)
    g_loss, g_metrics = ebgan.generator_loss(fake_data)
    
    print(f"\nDiscriminator loss: {d_loss.item():.4f}")
    print(f"Generator loss: {g_loss.item():.4f}")
    print(f"Metrics: {d_metrics}")
