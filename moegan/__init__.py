"""
MoE-GAN Model Package

This package contains the implementation of the Mixture of Experts Generative Adversarial Network.
"""

# Import key components to make them available from the package
from .t2i_moe_gan import AuroraGenerator, AuroraDiscriminator, train_aurora_gan

__all__ = ['AuroraGenerator', 'AuroraDiscriminator', 'train_aurora_gan']