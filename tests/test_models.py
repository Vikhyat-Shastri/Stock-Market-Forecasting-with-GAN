"""
Unit tests for GAN models.
"""

import pytest
import torch
from src.models.generator import LSTMGenerator, AttentionLSTMGenerator
from src.models.discriminator import CNNDiscriminator, ResidualCNNDiscriminator
from src.models.gan import WGAN_GP


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device('cpu')  # Use CPU for tests


@pytest.fixture
def generator(device):
    """Create a test generator."""
    return LSTMGenerator(
        noise_dim=10,
        hidden_dim=32,
        num_layers=1,
        n_features=5,
        sequence_length=10,
        dropout=0.2
    ).to(device)


@pytest.fixture
def discriminator(device):
    """Create a test discriminator."""
    return CNNDiscriminator(
        n_features=5,
        sequence_length=10,
        base_filters=16,
        num_conv_layers=2
    ).to(device)


def test_generator_forward(generator, device):
    """Test generator forward pass."""
    batch_size = 4
    noise = torch.randn(batch_size, 10, device=device)
    
    output = generator(noise)
    
    assert output.shape == (batch_size, 10, 5)  # [batch, seq_len, features]
    assert output.min() >= -1 and output.max() <= 1  # tanh output


def test_discriminator_forward(discriminator, device):
    """Test discriminator forward pass."""
    batch_size = 4
    seq_length = 10
    n_features = 5
    
    input_seq = torch.randn(batch_size, seq_length, n_features, device=device)
    
    output = discriminator(input_seq)
    
    assert output.shape == (batch_size, 1)


def test_attention_generator(device):
    """Test attention generator."""
    gen = AttentionLSTMGenerator(
        noise_dim=10,
        hidden_dim=32,
        num_layers=1,
        n_features=5,
        sequence_length=10
    ).to(device)
    
    noise = torch.randn(4, 10, device=device)
    output = gen(noise)
    
    assert output.shape == (4, 10, 5)


def test_residual_discriminator(device):
    """Test residual discriminator."""
    disc = ResidualCNNDiscriminator(
        n_features=5,
        sequence_length=10,
        base_filters=16,
        num_blocks=2
    ).to(device)
    
    input_seq = torch.randn(4, 10, 5, device=device)
    output = disc(input_seq)
    
    assert output.shape == (4, 1)


def test_wgan_gp_initialization(generator, discriminator, device):
    """Test WGAN-GP initialization."""
    wgan = WGAN_GP(
        generator=generator,
        discriminator=discriminator,
        noise_dim=10,
        lambda_gp=10.0,
        n_critic=5,
        device=device
    )
    
    assert wgan.device == device
    assert wgan.lambda_gp == 10.0
    assert wgan.n_critic == 5


def test_gradient_penalty(generator, discriminator, device):
    """Test gradient penalty calculation."""
    wgan = WGAN_GP(
        generator=generator,
        discriminator=discriminator,
        noise_dim=10,
        device=device
    )
    
    wgan.configure_optimizers()
    
    real_data = torch.randn(4, 10, 5, device=device)
    fake_data = torch.randn(4, 10, 5, device=device)
    
    gp = wgan.gradient_penalty(real_data, fake_data)
    
    assert isinstance(gp, torch.Tensor)
    assert gp.item() >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
