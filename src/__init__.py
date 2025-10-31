"""
Stock Market Forecasting with GAN
A production-ready implementation of Generative Adversarial Networks for time-series prediction.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from src.models.generator import LSTMGenerator
from src.models.discriminator import CNNDiscriminator
from src.models.gan import WGAN_GP

__all__ = [
    "LSTMGenerator",
    "CNNDiscriminator",
    "WGAN_GP",
]
