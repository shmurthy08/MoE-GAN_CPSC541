# aurora_gan_moe.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import clip
import math
from torch.distributions import Normal, kl_divergence

# Constants
LATENT_DIM = 512
TEXT_EMBEDDING_DIM = 512
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_EXPERTS = 8  # Number of experts in MoE layers

class ModulatedConv(nn.Module):
    """
    Modulated Convolution as described in StyleGAN2/Aurora.
    Applies modulation from the latent code to the convolutional weights.
    """
    def __init__(self, in_channels, out_channels, kernel_size, 
                 latent_dim=LATENT_DIM, stride=1, padding=0, 
                 demodulate=True, upsample=False):
        super(ModulatedConv, self).__init__()
        
        # Convolutional parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.upsample = upsample
        self.demodulate = demodulate
        
        # Convolutional weight
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        
        # Modulation network (w -> style)
        self.modulation = nn.Linear(latent_dim, in_channels)
        
        # Initialize weights
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.ones_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)
    
    def forward(self, x, w):
        batch_size, in_channels, height, width = x.shape
        
        # Get modulation scaling factors from latent code
        style = self.modulation(w).view(batch_size, 1, in_channels, 1, 1)
        
        # Modulate weights
        weight = self.weight.unsqueeze(0) * style
        
        # Demodulate weights if required
        if self.demodulate:
            d = torch.rsqrt(
                (weight ** 2).sum(dim=(2, 3, 4), keepdim=True) + 1e-8
            )
            weight = weight * d
        
        # Reshape for batch matrix multiplication
        weight = weight.view(
            batch_size * self.out_channels, in_channels, 
            self.kernel_size, self.kernel_size
        )
        
        # Reshape input
        x = x.reshape(1, batch_size * in_channels, height, width)
        
        # Perform convolution
        if self.upsample:
            # Upsample then convolve
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = F.conv2d(x, weight, padding=self.padding, stride=self.stride, groups=batch_size)
        else:
            # Regular convolution
            x = F.conv2d(x, weight, padding=self.padding, stride=self.stride, groups=batch_size)
        
        # Reshape output
        _, _, height, width = x.shape
        x = x.view(batch_size, self.out_channels, height, width)
        
        return x

class ModulatedTransformationModule(nn.Module):
    """
    MTM (Modulated Transformation Module) as described in the Aurora paper.
    Combines modulated convolution with learnable feature deformation.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 latent_dim=LATENT_DIM, use_offset=False, resolution=None):
        super(ModulatedTransformationModule, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_offset = use_offset and resolution is not None and resolution <= 16
        
        # Modulated convolution
        self.modulated_conv = ModulatedConv(
            in_channels, out_channels, kernel_size,
            latent_dim=latent_dim, stride=1, padding=kernel_size//2
        )
        
        # Offset prediction network (for feature deformation)
        if self.use_offset:
            self.offset_net = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(32, 2, kernel_size=3, padding=1)  # 2 channels for x, y offsets
            )
        
        # Activation function
        self.activation = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x, w):
        batch_size, _, height, width = x.shape
        
        # Predict offsets if enabled
        if self.use_offset:
            offsets = self.offset_net(x)
            
            # Create sampling grid with offsets
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(-1, 1, height, device=x.device),
                torch.linspace(-1, 1, width, device=x.device)
            )
            grid = torch.stack((grid_x, grid_y), dim=2).unsqueeze(0)
            grid = grid.repeat(batch_size, 1, 1, 1)
            
            # Add predicted offsets (scaled to be in range [-1, 1])
            offsets = offsets.permute(0, 2, 3, 1) * 0.1  # Scale factor for stability
            grid = grid + offsets
            
            # Sample input features with the deformed grid
            x = F.grid_sample(x, grid, mode='bilinear', align_corners=False)
        
        # Apply modulated convolution
        x = self.modulated_conv(x, w)
        
        # Apply activation
        x = self.activation(x)
        
        return x

class SparseExpertFFN(nn.Module):
    """
    A single expert as a Feed-Forward Network (FFN).
    """
    def __init__(self, dim):
        super(SparseExpertFFN, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, x):
        return self.net(x)

class BayesianRouter(nn.Module):
    """
    Bayesian Router for selecting experts based on features and text embedding.
    Implemented with weight uncertainty for better generalization.
    """
    def __init__(self, feature_dim, text_dim, num_experts=NUM_EXPERTS):
        super(BayesianRouter, self).__init__()
        
        # Dimensions
        self.feature_dim = feature_dim
        self.text_dim = text_dim
        self.num_experts = num_experts
        
        # Feature projection
        self.feature_mu = nn.Parameter(torch.Tensor(feature_dim, 128).normal_(0, 0.1))
        self.feature_rho = nn.Parameter(torch.Tensor(feature_dim, 128).normal_(-3, 0.1))
        
        # Text projection
        self.text_mu = nn.Parameter(torch.Tensor(text_dim, 128).normal_(0, 0.1))
        self.text_rho = nn.Parameter(torch.Tensor(text_dim, 128).normal_(-3, 0.1))
        
        # Combined projection to expert logits
        self.combined_mu = nn.Parameter(torch.Tensor(256, num_experts).normal_(0, 0.1))
        self.combined_rho = nn.Parameter(torch.Tensor(256, num_experts).normal_(-3, 0.1))
        
        # Noise for sampling
        self.register_buffer('epsilon_f', torch.zeros(feature_dim, 128))
        self.register_buffer('epsilon_t', torch.zeros(text_dim, 128))
        self.register_buffer('epsilon_c', torch.zeros(256, num_experts))
        
        # Temperature parameter for sharpening distribution
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
    
    def reparameterize(self, mu, rho, epsilon=None):
        """
        Reparameterization trick for sampling from a Gaussian posterior.
        """
        sigma = torch.log(1 + torch.exp(rho))
        if epsilon is None:
            epsilon = torch.randn_like(sigma)
        return mu + sigma * epsilon
    
    def forward(self, feature, text_embedding, sampling=True):
        batch_size = feature.size(0)
        
        # Sample weights if in training mode
        if sampling and self.training:
            # Resample noise
            self.epsilon_f.normal_()
            self.epsilon_t.normal_()
            self.epsilon_c.normal_()
            
            # Sample weights
            feature_weight = self.reparameterize(self.feature_mu, self.feature_rho, self.epsilon_f)
            text_weight = self.reparameterize(self.text_mu, self.text_rho, self.epsilon_t)
            combined_weight = self.reparameterize(self.combined_mu, self.combined_rho, self.epsilon_c)
        else:
            # Use mean weights for inference
            feature_weight = self.feature_mu
            text_weight = self.text_mu
            combined_weight = self.combined_mu
        
        # Project feature and text embedding
        feature_proj = torch.matmul(feature, feature_weight)
        text_proj = torch.matmul(text_embedding, text_weight)
        
        # Combine projections
        combined = torch.cat([feature_proj, text_proj], dim=1)
        
        # Get expert logits
        logits = torch.matmul(combined, combined_weight)
        
        # Apply temperature for sharper distribution
        logits = logits / torch.clamp(self.temperature, min=0.1)
        
        # Get probabilities using softmax
        probs = F.softmax(logits, dim=1)
        
        # For inference, make routing sparse by taking the top-1 expert
        if not self.training:
            top_probs, top_indices = probs.topk(1, dim=1)
            
            # Create one-hot encoding
            one_hot = torch.zeros_like(probs)
            one_hot.scatter_(1, top_indices, 1.0)
            
            # Use one-hot encoding as probabilities
            probs = one_hot
        
        return probs, logits
    
    def kl_divergence(self):
        """
        Calculate KL divergence for Bayesian weights.
        """
        # Prior distribution (standard normal)
        prior_mu = torch.zeros_like(self.feature_mu)
        prior_sigma = torch.ones_like(self.feature_rho)
        
        # Posterior distributions
        feature_sigma = torch.log(1 + torch.exp(self.feature_rho))
        text_sigma = torch.log(1 + torch.exp(self.text_rho))
        combined_sigma = torch.log(1 + torch.exp(self.combined_rho))
        
        # KL divergence for each set of weights
        kl_feature = self._kl_normal(self.feature_mu, feature_sigma, prior_mu[:, :128], prior_sigma[:, :128])
        kl_text = self._kl_normal(self.text_mu, text_sigma, prior_mu[:, :128], prior_sigma[:, :128])
        kl_combined = self._kl_normal(self.combined_mu, combined_sigma, 
                                      prior_mu[:128, :self.num_experts], 
                                      prior_sigma[:128, :self.num_experts])
        
        return kl_feature + kl_text + kl_combined
    
    def _kl_normal(self, mu_q, sigma_q, mu_p, sigma_p):
        """
        KL divergence between two normal distributions.
        """
        var_q = sigma_q ** 2
        var_p = sigma_p ** 2
        
        kl = 0.5 * torch.sum(
            torch.log(var_p / var_q) + (var_q + (mu_q - mu_p) ** 2) / var_p - 1
        )
        
        return kl

class SparseMoE(nn.Module):
    """
    Sparse Mixture of Experts layer with a Bayesian router.
    """
    def __init__(self, dim, text_dim, num_experts=NUM_EXPERTS):
        super(SparseMoE, self).__init__()
        
        self.dim = dim
        self.num_experts = num_experts
        
        # Create experts
        self.experts = nn.ModuleList([
            SparseExpertFFN(dim) for _ in range(num_experts)
        ])
        
        # Create Bayesian router
        self.router = BayesianRouter(dim, text_dim, num_experts)
    
    def forward(self, x, w):
        batch_size, channels, height, width = x.shape
        
        # Reshape input for experts
        x_reshaped = x.permute(0, 2, 3, 1).reshape(batch_size * height * width, channels)
        
        # Expand w to match feature points
        w_expanded = w.unsqueeze(1).unsqueeze(1).repeat(1, height, width, 1)
        w_reshaped = w_expanded.reshape(batch_size * height * width, -1)
        
        # Get routing probabilities
        routing_probs, routing_logits = self.router(x_reshaped, w_reshaped)
        
        # Initialize output with zeros
        combined_output = torch.zeros_like(x_reshaped)
        
        # Route to experts
        if self.training:
            # If training, use soft routing with all experts
            for i, expert in enumerate(self.experts):
                expert_output = expert(x_reshaped)
                # Weight expert output by router probability
                combined_output += routing_probs[:, i:i+1] * expert_output
        else:
            # If inference, use hard routing (top-1 expert)
            expert_indices = torch.argmax(routing_probs, dim=1)
            
            # Group inputs by assigned expert to process in batches
            for i, expert in enumerate(self.experts):
                # Find which inputs are routed to this expert
                mask = (expert_indices == i)
                if torch.any(mask):
                    # Process only the inputs routed to this expert
                    expert_inputs = x_reshaped[mask]
                    expert_output = expert(expert_inputs)
                    combined_output[mask] = expert_output
        
        # Reshape output back to original dimensions
        output = combined_output.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        
        # Calculate KL divergence
        kl_div = self.router.kl_divergence() if self.training else 0.0
        
        return output, kl_div, routing_probs

class AttentionBlock(nn.Module):
    """
    Attention Block with self-attention, cross-attention, and FFN (MoE).
    """
    def __init__(self, dim, text_dim=768, heads=8):
        super(AttentionBlock, self).__init__()
        
        self.dim = dim
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        
        # MoE FFN
        self.moe = SparseMoE(dim, text_dim)
        
        # Projection convolutions (for channel adjustment)
        self.proj_in = ModulatedConv(dim, dim, kernel_size=1)
        self.proj_out = ModulatedConv(dim, dim, kernel_size=1)
    
    def forward(self, x, w, text_seq, kl_losses=None):
        batch_size, channels, height, width = x.shape
        
        # Project input
        x_in = self.proj_in(x, w)
        
        # Reshape for attention
        x_flat = x_in.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        
        # Self-attention
        x_norm = self.norm1(x_flat)
        sa_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + sa_out
        
        # Cross-attention
        x_norm = self.norm2(x_flat)
        ca_out, _ = self.cross_attn(x_norm, text_seq, text_seq)
        x_flat = x_flat + ca_out
        
        # Back to spatial dimensions
        x_spatial = x_flat.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        
        # Sparse MoE FFN
        x_norm_spatial = self.norm3(x_flat).reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        moe_out, moe_kl, routing_probs = self.moe(x_norm_spatial, w)
        
        # Store KL losses if tracking
        if kl_losses is not None:
            kl_losses.append(moe_kl)
        
        # Skip connection
        x_spatial = x_spatial + moe_out
        
        # Project output
        x_out = self.proj_out(x_spatial, w)
        
        return x_out, routing_probs

class ConvolutionBlock(nn.Module):
    """
    Convolution Block with MTMs (Modulated Transformation Modules).
    """
    def __init__(self, in_channels, out_channels, latent_dim=LATENT_DIM, resolution=None):
        super(ConvolutionBlock, self).__init__()
        
        self.mtm1 = ModulatedTransformationModule(
            in_channels, out_channels, kernel_size=3,
            latent_dim=latent_dim, use_offset=True, resolution=resolution
        )
        
        self.mtm2 = ModulatedTransformationModule(
            out_channels, out_channels, kernel_size=3,
            latent_dim=latent_dim, use_offset=True, resolution=resolution
        )
    
    def forward(self, x, w):
        # First MTM
        out = self.mtm1(x, w)
        
        # Second MTM
        out = self.mtm2(out, w)
        
        # Skip connection
        out = out + x
        
        return out

class GenerativeBlock(nn.Module):
    """
    Unit Generative Block combining Convolution and Attention blocks.
    """
    def __init__(self, in_channels, out_channels, text_dim=768, 
                 upsample=False, resolution=None):
        super(GenerativeBlock, self).__init__()
        
        self.upsample = upsample
        
        # Upsampling if needed
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else nn.Identity()
        
        # Convolution block
        self.conv_block = ConvolutionBlock(
            in_channels, out_channels, resolution=resolution
        )
        
        # Attention block
        self.attn_block = AttentionBlock(
            out_channels, text_dim=text_dim
        )
    
    def forward(self, x, w, text_seq, kl_losses=None):
        # Upsample if needed
        if self.upsample:
            x = self.up(x)
        
        # Convolution block
        x = self.conv_block(x, w)
        
        # Attention block
        x, routing_probs = self.attn_block(x, w, text_seq, kl_losses)
        
        return x, routing_probs

class AuroraGenerator(nn.Module):
    """
    Aurora GAN Generator with MoE and dynamic routing.
    """
    def __init__(self, latent_dim=LATENT_DIM, text_embedding_dim=512, 
                 clip_model_type="ViT-B/32"):
        super(AuroraGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        self.text_embedding_dim = text_embedding_dim
        
        # Load CLIP model for text encoding
        self.clip_model, _ = clip.load(clip_model_type, device=DEVICE)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Additional learnable layers for text encoder fine-tuning
        self.text_projection = nn.Sequential(
            nn.Linear(text_embedding_dim, text_embedding_dim),
            nn.LayerNorm(text_embedding_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(text_embedding_dim, text_embedding_dim)
        )
        
        # Mapping network (z, text_global -> w)
        self.mapping = nn.Sequential(
            nn.Linear(latent_dim + text_embedding_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512)
        )
        
        # Initial generation (4x4 constant)
        self.constant = nn.Parameter(torch.randn(1, 512, 4, 4))
        
        # Generative blocks with increasing resolution
        self.gen_block_4 = GenerativeBlock(512, 512, text_dim=text_embedding_dim, upsample=False, resolution=4)
        self.gen_block_8 = GenerativeBlock(512, 256, text_dim=text_embedding_dim, upsample=True, resolution=8)
        self.gen_block_16 = GenerativeBlock(256, 128, text_dim=text_embedding_dim, upsample=True, resolution=16)
        self.gen_block_32 = GenerativeBlock(128, 64, text_dim=text_embedding_dim, upsample=True, resolution=32)
        self.gen_block_64 = GenerativeBlock(64, 32, text_dim=text_embedding_dim, upsample=True, resolution=64)
        
        # To RGB layers for different resolutions
        self.to_rgb_32 = ModulatedConv(64, 3, kernel_size=1)
        self.to_rgb_64 = ModulatedConv(32, 3, kernel_size=1)
    
    def encode_text(self, text):
        # Tokenize text
        tokens = clip.tokenize(text).to(DEVICE)
        
        # Extract text features using CLIP
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokens)
            text_features = text_features.float()
        
        # Get global text feature (for w)
        text_global = text_features
        
        # Process text sequence for attention layers
        text_seq = self.text_projection(text_features).unsqueeze(1)  # [B, 1, D]
        
        return text_global, text_seq
    
    def forward(self, z, text, truncation_psi=0.7, return_routing=False):
        batch_size = z.size(0)
        
        # Encode text
        text_global, text_seq = self.encode_text(text)
        
        # Concatenate z and text global feature
        z_text = torch.cat([z, text_global], dim=1)
        
        # Map to W space
        w = self.mapping(z_text)
        
        # Apply truncation trick (tradeoff between quality and diversity)
        if truncation_psi < 1.0:
            # Create mean latent vector
            with torch.no_grad():
                mean_latent = self.mapping(
                    torch.cat([
                        torch.zeros(1, self.latent_dim, device=DEVICE),
                        torch.zeros(1, self.text_embedding_dim, device=DEVICE)
                    ], dim=1)
                )
            w = mean_latent + truncation_psi * (w - mean_latent)
        
        # Track KL losses
        kl_losses = []
        routing_probs = []
        
        # Generate 4x4 image
        x = self.constant.repeat(batch_size, 1, 1, 1)
        x, r_probs = self.gen_block_4(x, w, text_seq, kl_losses)
        routing_probs.append(r_probs)
        
        # Generate 8x8 image
        x, r_probs = self.gen_block_8(x, w, text_seq, kl_losses)
        routing_probs.append(r_probs)
        
        # Generate 16x16 image
        x, r_probs = self.gen_block_16(x, w, text_seq, kl_losses)
        routing_probs.append(r_probs)
        
        # Generate 32x32 image
        x, r_probs = self.gen_block_32(x, w, text_seq, kl_losses)
        routing_probs.append(r_probs)
        x_32 = self.to_rgb_32(x, w)
        
        # Generate 64x64 image (final resolution)
        x, r_probs = self.gen_block_64(x, w, text_seq, kl_losses)
        routing_probs.append(r_probs)
        x_64 = self.to_rgb_64(x, w)
        
        # Combine outputs from different resolutions
        x_64 = x_64 + F.interpolate(x_32, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Calculate total KL loss
        kl_loss = sum(kl_losses) if kl_losses else torch.tensor(0.0, device=x.device)
        
        if return_routing:
            return x_64, kl_loss, routing_probs
        return x_64, kl_loss

class AuroraDiscriminator(nn.Module):
    """
    Discriminator for Aurora GAN with text conditioning.
    """
    def __init__(self, text_embedding_dim=512):
        super(AuroraDiscriminator, self).__init__()
        
        # Text projection
        self.text_projection = nn.Sequential(
            nn.Linear(text_embedding_dim, 128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 -> 4x4
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Conv2d(512 + 128, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, img, text_embedding):
        # Process image
        features = self.conv_layers(img)
        
        # Process text embedding
        text_features = self.text_projection(text_embedding)
        
        # Replicate text features to match feature map size
        text_features = text_features.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 4, 4)
        
        # Concatenate image features and text features
        combined = torch.cat([features, text_features], dim=1)
        
        # Final output
        output = self.output_layer(combined)
        
        return output.view(-1)

class AuroraGANLoss:
    """
    Loss functions for Aurora GAN training.
    """
    def __init__(self, device=DEVICE):
        self.device = device
    
    def generator_loss(self, fake_pred, kl_loss=None, kl_weight=0.001):
        # Non-saturating GAN loss
        g_loss = F.softplus(-fake_pred).mean()
        
        # Add KL divergence loss if provided
        if kl_loss is not None:
            g_loss = g_loss + kl_weight * kl_loss
        
        return g_loss
    
    def discriminator_loss(self, real_pred, fake_pred):
        # Logistic loss
        d_loss_real = F.softplus(-real_pred).mean()
        d_loss_fake = F.softplus(fake_pred).mean()
        
        return d_loss_real + d_loss_fake
    
    def moe_balance_loss(self, routing_probs, balance_weight=0.01):
        """
        Load balancing loss for MoE following Switch Transformer paper.
        """
        # Sum over batch dimension to get expert usage
        expert_usage = routing_probs.sum(dim=0)
        
        # Normalize to get probability distribution
        expert_usage = expert_usage / torch.sum(expert_usage)
        
        # Target uniform distribution
        num_experts = expert_usage.size(0)
        target_dist = torch.ones(num_experts, device=self.device) / num_experts
        
        # KL divergence between actual and target distribution
        balance_loss = F.kl_div(
            torch.log(expert_usage + 1e-10), 
            target_dist, 
            reduction='batchmean'
        )
        
        return balance_weight * balance_loss

def train_aurora_gan(
    dataloader, val_dataloader=None, 
    num_epochs=50, lr=0.0002, beta1=0.5, beta2=0.999,
    device=DEVICE, save_dir='./aurora_checkpoints',
    log_interval=10, save_interval=1000
):
    """
    Train the Aurora GAN model.
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize models
    generator = AuroraGenerator().to(device)
    discriminator = AuroraDiscriminator().to(device)
    
    # Initialize optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))
    
    # Initialize loss function
    gan_loss = AuroraGANLoss(device)
    
    # Training loop
    step = 0
    for epoch in range(num_epochs):
        for batch_idx, (real_images, text_embeddings) in enumerate(dataloader):
            batch_size = real_images.size(0)
            
            # Move data to device
            real_images = real_images.to(device)
            text_embeddings = text_embeddings.to(device)
            
            # Sample random noise
            z = torch.randn(batch_size, LATENT_DIM, device=device)
            
            # ------------------------
            # Train Discriminator
            # ------------------------
            optimizer_d.zero_grad()
            
            # Real images
            real_pred = discriminator(real_images, text_embeddings)
            
            # Generate fake images
            with torch.no_grad():
                fake_images, _ = generator(z, text_embeddings)
            
            # Fake images
            fake_pred = discriminator(fake_images.detach(), text_embeddings)
            
            # Discriminator loss
            d_loss = gan_loss.discriminator_loss(real_pred, fake_pred)
            d_loss.backward()
            optimizer_d.step()
            
            # ------------------------
            # Train Generator
            # ------------------------
            optimizer_g.zero_grad()
            
            # Generate fake images
            fake_images, kl_loss, routing_probs = generator(z, text_embeddings, return_routing=True)
            
            # Discriminator prediction on fake images
            fake_pred = discriminator(fake_images, text_embeddings)
            
            # Calculate load balancing loss
            balance_loss = gan_loss.moe_balance_loss(routing_probs[-1])
            
            # Generator loss
            g_loss = gan_loss.generator_loss(fake_pred, kl_loss, kl_weight=0.001)
            g_loss = g_loss + balance_loss
            g_loss.backward()
            optimizer_g.step()
            
            # Logging
            if step % log_interval == 0:
                print(f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(dataloader)}] "
                      f"D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}, "
                      f"KL_loss: {kl_loss.item():.4f}, Balance_loss: {balance_loss.item():.4f}")
            
            # Save models
            if step % save_interval == 0:
                torch.save({
                    'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'optimizer_g': optimizer_g.state_dict(),
                    'optimizer_d': optimizer_d.state_dict(),
                    'epoch': epoch,
                    'step': step
                }, os.path.join(save_dir, f"aurora_checkpoint_{step}.pt"))
            
            step += 1
        
        # Validation (if provided)
        if val_dataloader is not None:
            generator.eval()
            discriminator.eval()
            
            val_g_loss = 0
            val_d_loss = 0
            
            with torch.no_grad():
                for val_real_images, val_text_embeddings in val_dataloader:
                    val_batch_size = val_real_images.size(0)
                    
                    # Move data to device
                    val_real_images = val_real_images.to(device)
                    val_text_embeddings = val_text_embeddings.to(device)
                    
                    # Sample random noise
                    val_z = torch.randn(val_batch_size, LATENT_DIM, device=device)
                    
                    # Real images
                    val_real_pred = discriminator(val_real_images, val_text_embeddings)
                    
                    # Generate fake images
                    val_fake_images, val_kl_loss = generator(val_z, val_text_embeddings)
                    
                    # Fake images
                    val_fake_pred = discriminator(val_fake_images, val_text_embeddings)
                    
                    # Losses
                    val_d_loss += gan_loss.discriminator_loss(val_real_pred, val_fake_pred).item()
                    val_g_loss += gan_loss.generator_loss(val_fake_pred, val_kl_loss).item()
            
            # Average losses
            val_d_loss /= len(val_dataloader)
            val_g_loss /= len(val_dataloader)
            
            print(f"Validation Epoch [{epoch}/{num_epochs}] "
                  f"D_loss: {val_d_loss:.4f}, G_loss: {val_g_loss:.4f}")
            
            # Set models back to training mode
            generator.train()
            discriminator.train()
    
    # Save final models
    torch.save({
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'optimizer_g': optimizer_g.state_dict(),
        'optimizer_d': optimizer_d.state_dict(),
        'epoch': num_epochs,
        'step': step
    }, os.path.join(save_dir, "aurora_final.pt"))
    
    return generator, discriminator

def sample_aurora_gan(generator, text_prompt, num_samples=1, truncation_psi=0.7, device=DEVICE):
    """
    Generate images from text prompt using the Aurora GAN.
    """
    generator.eval()
    
    # Sample random noise
    z = torch.randn(num_samples, LATENT_DIM, device=device)
    
    # Generate images
    with torch.no_grad():
        fake_images, _ = generator(z, [text_prompt] * num_samples, truncation_psi=truncation_psi)
    
    return fake_images

