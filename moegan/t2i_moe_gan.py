# aurora_gan_moe.py
import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import clip
import math
import logging
from torch.distributions import Normal, kl_divergence
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.utils.weight_norm as weight_norm



# Constants
LATENT_DIM = 512
TEXT_EMBEDDING_DIM = 512
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_EXPERTS = 4  # Number of experts in MoE layers
CLIP_MODEL_TYPE = "ViT-B/32"  # Vision Transformer model

_clip_model = None
_clip_preprocess = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_clip_model():
    """
    Lazy loading of CLIP model.
    
    Returns:
       clip model and preprocess
    """
    global _clip_model
    global _clip_preprocess
    
    if (_clip_model is None) or (_clip_preprocess is None):
        logger.info(f"\nLoading CLIP model {CLIP_MODEL_TYPE}")
        _clip_model, _clip_preprocess = clip.load(CLIP_MODEL_TYPE, device=DEVICE, jit=False)
        _clip_model.eval()
    
    return _clip_model, _clip_preprocess

def encode_text_with_clip(text):
    """
    Encode using CLIP model. Accepts single string or list.
    
    Args:
        text: text string or list of strings
    Returns:
        text embeddings
    """
    model , _ = get_clip_model()
    if isinstance(text, str):
        text = [text]
    with torch.no_grad():
        text_tokens = clip.tokenize(text).to(model.device)
        text_embeddings = model.encode_text(text_tokens)
    return text_embeddings

class CLIPLoss(nn.Module):
    """
    CLIP-based perceptual loss for text-to-image alignment
    """
    def __init__(self, device=DEVICE):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = get_clip_model()
        self.device = device

    def forward(self, images, text_embeddings):
        """
        Calculate CLIP loss between generated images and text embeddings

        Args:
            images: Tensor of generated images [batch_size, 3, height, width]
            text_embeddings: Tensor of text embeddings from CLIP [batch_size, embedding_dim]

        Returns:
            loss: CLIP-based loss value
        """
        if images is None: # Handle cases where an intermediate output might not exist
             return torch.tensor(0.0, device=self.device)

        # Ensure images are in the correct range for CLIP [-1, 1]
        normed_images = torch.clamp(images, -1, 1)

        # Resize images to match CLIP's expected input size (224x224)
        if normed_images.shape[-1] != 224 or normed_images.shape[-2] != 224:
            normed_images = F.interpolate(normed_images, size=(224, 224), mode='bilinear', align_corners=False)

        # Extract image features using CLIP
        # Use try-except for potential mixed precision issues if CLIP model uses it
        try:
            with torch.no_grad():
                image_features = self.model.encode_image(normed_images).float()
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # Ensure text embeddings are normalized
                text_features = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        except RuntimeError as e:
             # If CLIP uses mixed precision, might need explicit casting
             print(f"CLIPLoss warning: Runtime error {e}. Trying with explicit cast.")
             image_features = self.model.encode_image(normed_images.to(self.model.dtype)).float()
             image_features = image_features / image_features.norm(dim=-1, keepdim=True)
             text_features = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)


        # Cosine similarity as the loss (higher similarity = lower loss)
        # Handle potential NaN from normalization if norm is zero
        similarity = torch.sum(image_features * text_features, dim=1)
        similarity = torch.nan_to_num(similarity) # Replace NaN with 0
        loss = 1.0 - similarity.mean()

        return loss


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
        nn.init.normal_(self.modulation.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.modulation.bias)
    
    def forward(self, x, w):
        batch_size, in_channels, height, width = x.shape
        
        # Get modulation scaling factors from latent code
        style = self.modulation(w).view(batch_size, 1, in_channels, 1, 1)
        
        # Modulate weights
        weight = self.weight.unsqueeze(0) * style
        
        # Demodulate weights if required
        if self.demodulate:
            d = torch.rsqrt((weight ** 2).sum(dim=(2, 3, 4), keepdim=True) + 1e-8)
            weight = weight * d
        
        # Reshape for efficient batch matrix multiplication
        weight = weight.view(batch_size * self.out_channels, in_channels, 
                            self.kernel_size, self.kernel_size)
        
        # Reshape input for grouped convolution
        x = x.view(1, batch_size * in_channels, height, width)
        
        # Perform convolution
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = F.conv2d(x, weight, padding=self.padding, stride=self.stride, groups=batch_size)
        else:
            x = F.conv2d(x, weight, padding=self.padding, stride=self.stride, groups=batch_size)
        
        # Reshape output back to batch form
        _, _, new_height, new_width = x.shape
        x = x.view(batch_size, self.out_channels, new_height, new_width)
        
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
            offsets = offsets.permute(0, 2, 3, 1) * 0.05  # Scale factor for stability
            grid = grid + offsets
            grid = grid.clamp(-1, 1)  # Clamp to valid range for grid_sample
            
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
        
        # Feature projection with much smaller initialization
        self.feature_mu = nn.Parameter(torch.Tensor(feature_dim, 128))
        # Use much smaller standard deviation for initialization
        nn.init.normal_(self.feature_mu, mean=0.0, std=0.01)  # Changed from 0.01 to 0.001
        # Initialize rho to a value corresponding to even lower initial variance
        self.feature_rho = nn.Parameter(torch.Tensor(feature_dim, 128).fill_(-5.0))  # Changed from -4.0 to -6.0
        
        # Text projection with much smaller initialization
        self.text_mu = nn.Parameter(torch.Tensor(text_dim, 128))
        nn.init.normal_(self.text_mu, mean=0.0, std=0.01)  # Changed from 0.01 to 0.001
        self.text_rho = nn.Parameter(torch.Tensor(text_dim, 128).fill_(-5.0))  # Changed from -4.0 to -6.0
        
        # Combined projection to expert logits with much smaller initialization
        self.combined_mu = nn.Parameter(torch.Tensor(256, num_experts))
        nn.init.normal_(self.combined_mu, mean=0.0, std=0.01)  # Changed from 0.01 to 0.001
        self.combined_rho = nn.Parameter(torch.Tensor(256, num_experts).fill_(-5.0))  # Changed from -4.0 to -6.0

        # Noise for sampling
        self.register_buffer('epsilon_f', torch.zeros(feature_dim, 128))
        self.register_buffer('epsilon_t', torch.zeros(text_dim, 128))
        self.register_buffer('epsilon_c', torch.zeros(256, num_experts))
        
        # Temperature parameter - start with higher value for less sharp distributions
        self.temperature = nn.Parameter(torch.ones(1) * 4.0)
    def reparameterize(self, mu, rho, epsilon=None):
        """Numerically stable reparameterization with debugging"""
        # Debug input values
        if torch.isnan(mu).any().item() or torch.isinf(mu).any().item():
            print(f"⚠️ NaN/Inf detected in mu before clamp: min={mu.min().item()}, max={mu.max().item()}")
        
        if torch.isnan(rho).any().item() or torch.isinf(rho).any().item():
            print(f"⚠️ NaN/Inf detected in rho before clamp: min={rho.min().item()}, max={rho.max().item()}")
            
        # Clamp mu and rho to prevent extreme values
        mu = torch.clamp(mu, min=-10.0, max=10.0)
        rho = torch.clamp(rho, min=-8.0, max=4.0)
        
        # More stable sigma calculation with minimum bound
        sigma = torch.clamp(torch.log1p(torch.exp(rho)), min=1e-6, max=10.0)
        
        if torch.isnan(sigma).any().item() or torch.isinf(sigma).any().item():
            print(f"⚠️ NaN/Inf detected in sigma: min={sigma.min().item()}, max={sigma.max().item()}")
        
        if epsilon is None:
            epsilon = torch.randn_like(sigma)
        
        # Clamp epsilon too for more stability
        epsilon = torch.clamp(epsilon, min=-2.0, max=2.0)
        
        result = mu + sigma * epsilon
        
        # Check for NaN/Inf in final result
        if torch.isnan(result).any().item() or torch.isinf(result).any().item():
            print(f"⚠️ NaN/Inf detected in reparameterize result: min={result.min().item()}, max={result.max().item()}")
        
        return result
       
    def forward(self, feature, text_embedding, sampling=True, annealing_factor=1.0):
        batch_size = feature.size(0)
        # Check inputs for NaN/Inf
        if torch.isnan(feature).any().item() or torch.isinf(feature).any().item():
            print(f"⚠️ NaN/Inf detected in feature input to BayesianRouter")
            feature = torch.nan_to_num(feature, nan=0.0, posinf=1.0, neginf=-1.0)
            
        if torch.isnan(text_embedding).any().item() or torch.isinf(text_embedding).any().item():
            print(f"⚠️ NaN/Inf detected in text_embedding input to BayesianRouter")
            text_embedding = torch.nan_to_num(text_embedding, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Sample weights if in training mode
        if sampling and self.training:
            # Resample noise
            self.epsilon_f.normal_()
            self.epsilon_t.normal_()
            self.epsilon_c.normal_()
            
            # Sample weights with improved reparameterization
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
        
        # Apply temperature with annealing for smoother training
        # Start with higher temperature (larger annealing_factor) and reduce over time
        effective_temp = torch.clamp(self.temperature * annealing_factor, min=0.5, max=5.0)
        
        # Temperature-scaled logits with safeguards
        logits = logits / effective_temp
        
        # Prevent extreme logits that could cause NaN in softmax
        logits = torch.clamp(logits, min=-20.0, max=20.0)
        
        # Get probabilities using softmax
        probs = F.softmax(logits, dim=1)
        
        # Prevent numerical instability from near-zero probabilities
        probs = torch.clamp(probs, min=1e-6, max=1.0)
        # Re-normalize to ensure sum is exactly 1.0
        probs = probs / probs.sum(dim=1, keepdim=True)
        
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
        """Numerically stable KL divergence"""
        # Convert to log-variance for stability
        log_var_f = 2 * torch.log(torch.log1p(torch.exp(self.feature_rho)))
        log_var_t = 2 * torch.log(torch.log1p(torch.exp(self.text_rho)))
        log_var_c = 2 * torch.log(torch.log1p(torch.exp(self.combined_rho)))
        
        # Stable KL calculation
        kl_f = 0.5 * torch.sum(torch.exp(log_var_f) + self.feature_mu.pow(2) - 1 - log_var_f)
        kl_t = 0.5 * torch.sum(torch.exp(log_var_t) + self.text_mu.pow(2) - 1 - log_var_t)
        kl_c = 0.5 * torch.sum(torch.exp(log_var_c) + self.combined_mu.pow(2) - 1 - log_var_c)
        
        # Combine and handle numerical issues
        kl_div = kl_f + kl_t + kl_c
        kl_div = torch.nan_to_num(kl_div, nan=0.0, posinf=200.0, neginf=0.0)
        kl_div = torch.clamp(kl_div, min=0.0, max=120.0)
        
        # The scalar multiplier will be applied by the annealing
        return kl_div
    
    
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
    
    def forward(self, x, w, annealing_factor=1.0):
        """
        Forward pass with temperature annealing support.
        
        Args:
            x: Input features
            w: Style vector/text conditioning
            annealing_factor: Temperature annealing factor for Bayesian router
        """
        batch_size, channels, height, width = x.shape
        
        x_reshaped = x.permute(0, 2, 3, 1).contiguous().view(-1, channels)
        w_reshaped = w.unsqueeze(1).unsqueeze(1).expand(batch_size, height, width, -1).contiguous().view(-1, w.size(1))
        
        # Get routing probabilities with annealing factor
        routing_probs, routing_logits = self.router(x_reshaped, w_reshaped, sampling=self.training, annealing_factor=annealing_factor)
        
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
        kl_div = self.router.kl_divergence() if self.training else torch.tensor(0.0, device=x.device)
        
        return output, kl_div, routing_probs
    
class AttentionBlock(nn.Module):
    """
    Attention Block with self-attention, cross-attention, and FFN (MoE).
    """
    def __init__(self, dim, text_dim=512, heads=8):
        super(AttentionBlock, self).__init__()
        
        self.dim = dim
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        # Text projection to match spatial feature dimension
        self.text_proj = nn.Linear(text_dim, dim)
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        
        # MoE FFN
        self.moe = SparseMoE(dim, text_dim)
        
        # Projection convolutions (for channel adjustment)
        self.proj_in = ModulatedConv(dim, dim, kernel_size=1)
        self.proj_out = ModulatedConv(dim, dim, kernel_size=1)
    
    def forward(self, x, w, text_seq, kl_losses=None, annealing_factor=1.0):
        """
        Forward pass with temperature annealing support.
        
        Args:
            x: Input tensor
            w: Style vector
            text_seq: Text embedding sequence
            kl_losses: List to store KL losses
            annealing_factor: Temperature annealing factor for Bayesian router
        """
        batch_size, channels, height, width = x.shape
        
        # Project input
        x_in = self.proj_in(x, w)
        
        # Reshape for attention
        x_flat = x_in.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        
        # Self-attention
        x_norm = self.norm1(x_flat)
        sa_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + sa_out
        
        # Project text sequence to match spatial feature dimension
        text_proj = self.text_proj(text_seq)
        
        # Cross-attention
        x_norm = self.norm2(x_flat)
        ca_out, _ = self.cross_attn(x_norm, text_proj, text_proj)
        x_flat = x_flat + ca_out
        
        # Back to spatial dimensions
        x_spatial = x_flat.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        
        # Sparse MoE FFN with annealing factor
        x_norm_spatial = self.norm3(x_flat).reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        
        # Pass annealing_factor to MoE layer
        moe_out, moe_kl, routing_probs = self.moe(x_norm_spatial, w, annealing_factor)
        
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
        
        # Add a projection layer for skip connection when dimensions don't match
        self.skip_proj = None
        if in_channels != out_channels:
            self.skip_proj = ModulatedConv(
                in_channels, out_channels, kernel_size=1,
                latent_dim=latent_dim
            )
    
    def forward(self, x, w):
        # Save input for skip connection
        identity = x
        
        # First MTM
        out = self.mtm1(x, w)
        
        # Second MTM
        out = self.mtm2(out, w)
        
        # Project skip connection if needed
        if self.skip_proj is not None:
            identity = self.skip_proj(identity, w)
        
        # Skip connection
        out = out + identity
        
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
    
    def forward(self, x, w, text_seq, kl_losses=None, annealing_factor=1.0):
        """
        Forward pass with temperature annealing support.
        
        Args:
            x: Input tensor
            w: Style vector
            text_seq: Text embedding sequence
            kl_losses: List to store KL losses
            annealing_factor: Temperature annealing factor for Bayesian router
        """
        # ======== MODIFY FORWARD METHOD ========
        # Upsample if needed
        if self.upsample:
            x = self.up(x)
        
        # Convolution block
        x = self.conv_block(x, w)
        
        # Attention block with annealing factor
        x, routing_probs = self.attn_block(x, w, text_seq, kl_losses, annealing_factor)
        
        return x, routing_probs

class AuroraGenerator(nn.Module):
    """
    Aurora GAN Generator with MoE and dynamic routing.
    Limited to 16x16 resolution - no 32x32 or 64x64.
    """
    def __init__(self, latent_dim=LATENT_DIM, text_embedding_dim=512, max_resolution=16):
        super(AuroraGenerator, self).__init__()

        self.latent_dim = latent_dim
        self.text_embedding_dim = text_embedding_dim
        self._use_checkpointing = False
        self.max_resolution = max_resolution  # Always 16 in this version

        # Text projection (using pre-computed CLIP embeddings)
        self.text_projection = nn.Sequential(
            nn.Linear(text_embedding_dim, text_embedding_dim),
            nn.LayerNorm(text_embedding_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(text_embedding_dim, text_embedding_dim)
        )

        # Mapping network (z, text_embeddings -> w)
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

        # Generative blocks with increasing resolution - only up to 16x16
        self.gen_block_4 = GenerativeBlock(512, 512, text_dim=text_embedding_dim, upsample=False, resolution=4)
        self.gen_block_8 = GenerativeBlock(512, 256, text_dim=text_embedding_dim, upsample=True, resolution=8)
        self.gen_block_16 = GenerativeBlock(256, 128, text_dim=text_embedding_dim, upsample=True, resolution=16)
        
        # To RGB layers for each resolution
        self.to_rgb_8 = ModulatedConv(256, 3, kernel_size=1)
        self.to_rgb_16 = ModulatedConv(128, 3, kernel_size=1)
        
    def enable_checkpointing(self):
        """Enable gradient checkpointing to save memory"""
        self._use_checkpointing = True
        logger.info("Gradient checkpointing enabled on generator")
        return self
    
    def disable_checkpointing(self):
        """Disable gradient checkpointing"""
        self._use_checkpointing = False
        logger.info("Gradient checkpointing disabled on generator")
        return self

    def encode_text(self, text_or_embeddings):
        """
        Encode text using CLIP model if text is a string.
        """
        if isinstance(text_or_embeddings, str) or (isinstance(text_or_embeddings, list) and isinstance(text_or_embeddings[0], str)):
            text_embeddings = encode_text_with_clip(text_or_embeddings)
        else:
            text_embeddings = text_or_embeddings
        return text_embeddings
    
    def _run_block_with_checkpoint(self, block_fn, x, w, text_seq, kl_losses, annealing_factor=1.0):
        """Run a generative block with optional gradient checkpointing"""
        should_checkpoint = self._use_checkpointing and self.training
            
        if should_checkpoint:
            try:
                # Define a wrapper that matches the original signature expected by checkpoint
                def _checkpoint_wrapper(_x, _w, _text_seq, _kl_losses, _annealing_factor):
                    output, routing_probs = block_fn(_x, _w, _text_seq, _kl_losses, _annealing_factor)
                    return output, routing_probs

                # Run checkpoint with the wrapper
                output, routing_probs = torch.utils.checkpoint.checkpoint(
                    _checkpoint_wrapper,
                    x, w, text_seq, kl_losses, annealing_factor,
                    use_reentrant=False
                )
                return output, routing_probs

            except Exception as e:
                logger.error(f"Checkpointing failed: {e}")
                logger.error("Falling back to non-checkpointed execution")
                # Fallback to direct execution if checkpointing fails
                return block_fn(x, w, text_seq, kl_losses, annealing_factor)
        else:
            # Execute directly if not checkpointing
            return block_fn(x, w, text_seq, kl_losses, annealing_factor)
    
    def forward(self, z, text_input, truncation_psi=0.7, return_routing=False, 
                return_intermediate=False, annealing_factor=1.0):
        """
        Forward pass with temperature annealing support for MoE layers.
        Limited to 16x16 resolution.
        
        Args:
            z: Input noise vector
            text_input: Text embeddings or text prompt
            truncation_psi: Truncation trick factor
            return_routing: Whether to return routing probabilities
            return_intermediate: Whether to return intermediate outputs
            annealing_factor: Temperature annealing factor for Bayesian routers
        """
        batch_size = z.size(0)

        text_embeddings = self.encode_text(text_input)

        # Process text sequence for attention layers - keep the full dimension
        text_seq = self.text_projection(text_embeddings).unsqueeze(1)  # [B, 1, D]

        # Concatenate z and text global feature
        z_text = torch.cat([z, text_embeddings], dim=1)

        # Map to W space
        w = self.mapping(z_text)

        # Apply truncation trick (tradeoff between quality and diversity)
        if truncation_psi < 1.0:
            # Create mean latent vector
            with torch.no_grad():
                mean_latent = self.mapping(
                    torch.cat([
                        torch.zeros(1, self.latent_dim, device=z.device), 
                        torch.zeros(1, self.text_embedding_dim, device=z.device)
                    ], dim=1)
                )
            w = mean_latent + truncation_psi * (w - mean_latent)

        # Track KL losses
        kl_losses = []
        routing_probs = []

        # Generate 4x4 image
        x = self.constant.repeat(batch_size, 1, 1, 1)
        
        # Pass annealing_factor to each block (propagate to MoE layers)
        x, r_probs = self._run_block_with_checkpoint(
            lambda x, w, text_seq, kl_losses, annealing_factor: 
                self.gen_block_4(x, w, text_seq, kl_losses, annealing_factor),
            x, w, text_seq, kl_losses, annealing_factor
        )
        routing_probs.append(r_probs)
        
        x, r_probs = self._run_block_with_checkpoint(
            lambda x, w, text_seq, kl_losses, annealing_factor: 
                self.gen_block_8(x, w, text_seq, kl_losses, annealing_factor),
            x, w, text_seq, kl_losses, annealing_factor
        )
        routing_probs.append(r_probs)
        x_8 = self.to_rgb_8(x, w)
        
        x, r_probs = self._run_block_with_checkpoint(
            lambda x, w, text_seq, kl_losses, annealing_factor: 
                self.gen_block_16(x, w, text_seq, kl_losses, annealing_factor),
            x, w, text_seq, kl_losses, annealing_factor
        )
        routing_probs.append(r_probs)
        x_16 = self.to_rgb_16(x, w)
        
        # Final image is at 16x16 resolution
        final_image = x_16
        intermediate_image = x_8

        # Calculate total KL loss
        kl_loss = sum(kl_losses) if kl_losses else torch.tensor(0.0, device=x.device)

        if return_routing and return_intermediate:
            return final_image, intermediate_image, kl_loss, routing_probs
        elif return_routing:
            return final_image, None, kl_loss, routing_probs
        elif return_intermediate:
            return final_image, intermediate_image, kl_loss
        else:
            return final_image, kl_loss



class AuroraDiscriminator(nn.Module):
    """
    Discriminator for Aurora GAN with text conditioning.
    """
    def __init__(self, text_embedding_dim=512, max_resolution=16):
        super(AuroraDiscriminator, self).__init__()

        self.max_resolution = max_resolution
        
        # Text projection
        self.text_projection = nn.Sequential(
            weight_norm(nn.Linear(text_embedding_dim, 128)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Convolutional layers for 16x16 resolution - using weight_norm instead of BatchNorm
        self.conv_layers = nn.Sequential(
            # 16x16 -> 8x8
            weight_norm(nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 -> 4x4
            weight_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Output layer
        self.output_layer = nn.Sequential(
            weight_norm(nn.Conv2d(256 + 128, 1, kernel_size=4, stride=1, padding=0, bias=True))
        )


    def forward(self, img, text_embedding):
        # Process image
        features = self.conv_layers(img)  # Shape: [B, 256, 4, 4]

        # Process text embedding
        text_features = self.text_projection(text_embedding)  # Shape: [B, 128]

        # Replicate text features to match feature map size
        text_features = text_features.unsqueeze(-1).unsqueeze(-1)  # Shape: [B, 128, 1, 1]
        text_features = text_features.repeat(1, 1, features.shape[2], features.shape[3])  # Shape: [B, 128, 4, 4]

        # Concatenate image features and text features
        combined = torch.cat([features, text_features], dim=1)  # Shape: [B, 256+128, 4, 4]

        # Final output (logits)
        output = self.output_layer(combined)  # Shape: [B, 1, 1, 1]

        return output.view(-1)  # Return logits as a flat tensor [B]

class AuroraGANLoss:
    """
    Loss functions for Aurora GAN training.
    """
    def __init__(self, device=DEVICE):
        self.device = device
        self.clip_loss_fn = CLIPLoss(device) # Renamed to avoid conflict

    def generator_loss(self, fake_pred, kl_loss=None, kl_weight=0.001):
        # Non-saturating GAN loss for generator (wants D to output high for fake images)
        g_loss = F.softplus(-fake_pred).mean()

        # Note: CLIP loss is handled separately in the training loop now for multi-level
        g_loss = g_loss + kl_weight * kl_loss if kl_loss is not None else g_loss

        return g_loss

    def compute_clip_loss(self, images, text_input):
        """ Helper to compute CLIP loss, ensures text is encoded """
        if isinstance(text_input, str) or (isinstance(text_input, list) and isinstance(text_input[0], str)):
            # Use cached embeddings if possible, otherwise encode
            # Assuming text_input are embeddings during training
             text_embeddings = text_input
        else:
            text_embeddings = text_input # Assume they are already embeddings

        if text_embeddings.device != images.device:
            text_embeddings = text_embeddings.to(images.device)

        return self.clip_loss_fn(images, text_embeddings)

    # !! MODIFIED: Added mismatched_pred !!
    def discriminator_loss(self, real_pred, fake_pred, mismatched_pred):
        # Logistic loss using softplus (numerically more stable than sigmoid+log)
        # Wants real_pred high (minimize softplus(-real_pred))
        # Wants fake_pred low (minimize softplus(fake_pred))
        # Wants mismatched_pred low (minimize softplus(mismatched_pred))
        d_loss_real = F.softplus(-real_pred).mean()
        d_loss_fake = F.softplus(fake_pred).mean()
        d_loss_mismatched = F.softplus(mismatched_pred).mean() # Added mismatched loss term

        return d_loss_real + d_loss_fake + d_loss_mismatched # Combined loss

    def moe_balance_loss(self, routing_probs, balance_weight=0.01):
        """
        Load balancing loss for MoE following Switch Transformer paper.
        Assumes routing_probs is a list, takes the last one (final layer).
        """
        if not routing_probs:
            return torch.tensor(0.0, device=self.device)
            
        # Use probabilities from the last MoE layer (final layer)
        last_probs = routing_probs[-1]
        
        if last_probs is None or last_probs.numel() == 0:
            return torch.tensor(0.0, device=self.device)
            
        # Add small constant to prevent division by zero
        eps = 1e-6
        
        num_experts = last_probs.size(1)
        batch_items = last_probs.size(0)
        
        # Sum probabilities per expert across the batch
        # This represents how much each expert is used
        load = last_probs.sum(dim=0) + eps  # Shape: [num_experts]
        
        # Compute normalized expert usage (fraction of total routing)
        fraction_routed = load / batch_items  # Shape: [num_experts]
        
        # Compute per-item mean activation
        # First normalize the routing probs per item (should sum to 1 already, but ensure)
        prob_normalize = last_probs / (last_probs.sum(dim=1, keepdim=True) + eps)
        
        # Now compute mean probability per expert
        expert_prob_mean = prob_normalize.mean(dim=0)  # Shape: [num_experts]
        
        # Compute coefficient of variation (CV) - standard deviation / mean
        # This measures how uneven the expert utilization is
        mean_usage = torch.mean(fraction_routed)
        std_usage = torch.std(fraction_routed)
        cv = std_usage / (mean_usage + eps)
        
        # Scale CV by number of experts for normalized balance loss
        balance_loss = num_experts * cv
        
        # Apply safety clamping to prevent extreme values
        balance_loss = torch.clamp(balance_loss, min=0.0, max=10.0)
        
        # Handle NaN values (replace with 0.0)
        balance_loss = torch.nan_to_num(balance_loss, nan=0.0)
        
        return balance_weight * balance_loss


from tqdm import tqdm

def create_optimizer_for_active_blocks(generator, active_resolutions, lr, betas, weight_decay):
    """Create a fresh optimizer with only the active parameters"""
    # Always include these core parameters
    active_params = list(generator.text_projection.parameters()) + \
                   list(generator.mapping.parameters()) + \
                   [generator.constant]
    
    # Add resolution-specific parameters
    if 4 in active_resolutions:
        active_params.extend(generator.gen_block_4.parameters())
    if 8 in active_resolutions:
        active_params.extend(generator.gen_block_8.parameters())
    if 16 in active_resolutions:
        active_params.extend(generator.gen_block_16.parameters())
    if 32 in active_resolutions:
        active_params.extend(generator.gen_block_32.parameters())
        active_params.extend(generator.to_rgb_32.parameters())
    if 64 in active_resolutions:
        active_params.extend(generator.gen_block_64.parameters())
        active_params.extend(generator.to_rgb_64.parameters())
    
    return torch.optim.AdamW(active_params, lr=lr, betas=betas, weight_decay=weight_decay)


def train_aurora_gan(
    dataloader, val_dataloader=None,
    num_epochs=50, lr=0.0002, beta1=0.5, beta2=0.999,
    r1_gamma=10.0,
    clip_weight_16=0.1,  # Weight for 16x16 resolution
    clip_weight_8=0.05,  # Weight for 8x8 resolution
    kl_weight=0.001,
    kl_annealing_epochs=5,
    lr_warmup_epochs=3,
    balance_weight=0.01,
    device=DEVICE, save_dir='./aurora_checkpoints',
    log_interval=10, save_interval=1000,
    metric_callback=None,
    use_amp=True,
    gradient_accumulation_steps=8,
    checkpoint_activation=True,
    batch_memory_limit=20.0,
    max_resolution=16        # Fixed at 16x16 resolution
):
    """
    Train the Aurora GAN model with R1, Matching-Aware, Multi-level CLIP loss,
    fixed at 16x16 resolution, for 50 epochs.
    
    Args:
        dataloader: Training data loader
        val_dataloader: Optional validation data loader
        num_epochs: Number of training epochs (default 50)
        lr: Learning rate
        beta1: Beta1 for AdamW optimizer
        beta2: Beta2 for AdamW optimizer
        r1_gamma: R1 regularization weight
        clip_weight_16: CLIP loss weight for 16x16 resolution
        clip_weight_8: CLIP loss weight for 8x8 resolution
        kl_weight: KL divergence loss weight
        kl_annealing_epochs: Number of epochs for KL annealing
        lr_warmup_epochs: Number of epochs for learning rate warmup
        balance_weight: MoE balance loss weight
        device: Device to train on
        save_dir: Directory to save checkpoints
        log_interval: How often to log metrics
        save_interval: How often to save checkpoints
        metric_callback: Optional callback function for hyperparameter tuning
        use_amp: Whether to use automatic mixed precision training
        gradient_accumulation_steps: Number of steps to accumulate gradients
        checkpoint_activation: Whether to use gradient checkpointing
        batch_memory_limit: Maximum GPU memory per batch in GB
        max_resolution: Fixed at 16x16 resolution
        
    Returns:
        Trained generator and discriminator models
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    torch.autograd.set_detect_anomaly(True)
    print("Anomaly detection enabled - will show traceback for NaN values")
    # Memory cleanup before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Initialize models with max_resolution=16
    print(f"Initializing models with fixed 16x16 resolution")
    generator = AuroraGenerator(max_resolution=16).to(device)
    discriminator = AuroraDiscriminator(max_resolution=16).to(device)
    
    # Enable gradient checkpointing to save memory
    if checkpoint_activation and hasattr(generator, 'enable_checkpointing'):
        logger.info("Enabling gradient checkpointing for memory efficiency")
        generator.enable_checkpointing()
    
    # Initialize optimizers with weight decay
    weight_decay = 0.01
    optimizer_g = torch.optim.AdamW(generator.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)

    # Set up learning rate schedulers with cosine decay
    from torch.optim.lr_scheduler import CosineAnnealingLR
    
    # Create schedulers - will be activated after warmup
    lr_scheduler_g = CosineAnnealingLR(
        optimizer_g,
        T_max=num_epochs - lr_warmup_epochs,  # Exclude warmup epochs
        eta_min=lr * 0.05  # Minimum LR at 5% of initial
    )
    
    lr_scheduler_d = CosineAnnealingLR(
        optimizer_d,
        T_max=num_epochs - lr_warmup_epochs,
        eta_min=lr * 0.05
    )

    # Initialize AMP scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler() if use_amp and torch.cuda.is_available() else None
    
    # Initialize loss function
    gan_loss = AuroraGANLoss(device)

    # Log model parameters
    num_params_g = sum(p.numel() for p in generator.parameters())
    num_params_d = sum(p.numel() for p in discriminator.parameters())
    print(f"Generator parameters: {num_params_g:,}")
    print(f"Discriminator parameters: {num_params_d:,}")
    print(f"Total parameters: {num_params_g + num_params_d:,}")

    # Memory check after model initialization
    if torch.cuda.is_available():
        print(f"Memory after model init: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        torch.cuda.reset_peak_memory_stats()

    # Variables for tracking OOM errors
    total_oom_count = 0
    consecutive_oom_count = 0
    current_accumulation_steps = gradient_accumulation_steps

    # Training loop
    step = 0
    for epoch in range(num_epochs):
        print(f"\n{'='*20} Epoch {epoch+1}/{num_epochs} {'='*20}")
        
        # Calculate LR warmup factor
        if epoch < lr_warmup_epochs:
            # Linear warmup from 10% to 100% of base LR
            warmup_factor = 0.1 + 0.9 * (epoch / lr_warmup_epochs)
            current_lr_g = lr * warmup_factor
            current_lr_d = lr * warmup_factor
            
            # Update optimizer learning rates
            for param_group in optimizer_g.param_groups:
                param_group['lr'] = current_lr_g
            for param_group in optimizer_d.param_groups:
                param_group['lr'] = current_lr_d
                
            print(f"LR Warmup: {warmup_factor:.2f} - G: {current_lr_g:.6f}, D: {current_lr_d:.6f}")
        else:
            # After warmup, use schedulers
            current_lr_g = optimizer_g.param_groups[0]['lr']
            current_lr_d = optimizer_d.param_groups[0]['lr']
            print(f"Scheduled LR - G: {current_lr_g:.6f}, D: {current_lr_d:.6f}")
        
        # Calculate KL annealing factor 
        # Starts very low and increases over kl_annealing_epochs
        kl_warmup_factor = min(1.0, (epoch / kl_annealing_epochs)**2)
        # Start with an extremely small value (1e-5) of the configured weight
        initial_factor = 1e-5
        actual_factor = initial_factor + (1.0 - initial_factor) * kl_warmup_factor
        effective_kl_weight = kl_weight * actual_factor
        
        # Calculate MoE router temperature factor
        # Starts high (more uniform routing) and gradually decreases
        temperature_factor = max(1.0, 3.0 - epoch * 0.1)
        
        print(f"Epoch {epoch+1} settings:")
        print(f"  Temperature factor: {temperature_factor:.2f}")
        print(f"  KL warmup factor: {kl_warmup_factor:.3f}")
        print(f"  Effective KL weight: {effective_kl_weight:.8f}")
        print(f"  Gradient accumulation steps: {current_accumulation_steps}")

        # Create progress bar for this epoch
        epoch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        # Initialize running losses for this epoch
        running_d_loss = 0.0
        running_g_loss = 0.0
        running_r1_loss = 0.0
        running_kl_loss = 0.0
        running_balance_loss = 0.0
        running_clip_loss_16 = 0.0
        running_clip_loss_8 = 0.0
        
        # Reset gradients at start of epoch
        optimizer_d.zero_grad()
        optimizer_g.zero_grad()
        
        # Memory cleanup at start of epoch
        torch.cuda.empty_cache()
        gc.collect()
        
        # Reset batch counters for memory monitoring
        batch_count = 0
        memory_warning_count = 0
        epoch_oom_count = 0
        
        # Periodic memory cleanup interval
        memory_cleanup_interval = 10  # Check every 10 batches

        for batch_idx, (real_images, text_embeddings) in enumerate(epoch_pbar):
            batch_size = real_images.size(0)
            batch_count += 1
            
            # Periodic memory monitoring and cleanup
            if batch_count % memory_cleanup_interval == 0 and torch.cuda.is_available():
                current_mem = torch.cuda.memory_allocated() / 1e9
                peak_mem = torch.cuda.max_memory_allocated() / 1e9
                print(f"Batch {batch_idx} memory: {current_mem:.2f} GB, Peak: {peak_mem:.2f} GB")
                
                # Check if we're approaching memory limit (80% of limit)
                if peak_mem > batch_memory_limit * 0.8:
                    print("⚠️ Approaching memory limit, forcing cleanup")
                    # Force cleanup
                    torch.cuda.empty_cache()
                    gc.collect()
            
            # Check memory limit if specified
            if batch_memory_limit and torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated() / 1e9  # Convert to GB
                if current_memory > batch_memory_limit:
                    memory_warning_count += 1
                    epoch_oom_count += 1
                    total_oom_count += 1
                    consecutive_oom_count += 1
                    
                    logger.warning(f"Memory limit exceeded: {current_memory:.2f}GB > {batch_memory_limit}GB "
                                   f"(warning #{memory_warning_count}, consecutive #{consecutive_oom_count})")
                    logger.warning("Skipping batch to prevent OOM error")
                    
                    # Aggressive cleanup
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    # If we've had too many consecutive memory warnings, reduce batch size dynamically
                    if consecutive_oom_count >= 3:
                        old_steps = current_accumulation_steps
                        current_accumulation_steps = min(current_accumulation_steps * 2, 64)  # Double steps, max 64
                        logger.warning(f"Multiple consecutive OOM warnings! Increasing gradient accumulation "
                                      f"from {old_steps} to {current_accumulation_steps}.")
                        consecutive_oom_count = 0  # Reset counter
                    
                    continue
                else:
                    # Reset consecutive counter when we have a successful batch
                    consecutive_oom_count = 0

            # Move data to device
            real_images = real_images.to(device)
            text_embeddings = text_embeddings.to(device)

            # Sample random noise
            z = torch.randn(batch_size, LATENT_DIM, device=device)

            # ------------------------
            # Train Discriminator
            # ------------------------
            # Zero gradients based on accumulation steps
            if batch_idx % current_accumulation_steps == 0:
                optimizer_d.zero_grad()

            # Enable gradient computation for R1 penalty
            real_images.requires_grad = True
            
            with torch.cuda.amp.autocast() if use_amp else nullcontext():
                real_pred = discriminator(real_images, text_embeddings)

                # --- R1 Regularization ---
                grad_real, = torch.autograd.grad(
                    outputs=real_pred.sum(), inputs=real_images, create_graph=True
                )
                grad_penalty = grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                r1_loss = (r1_gamma / 2) * grad_penalty.mean() # Divide gamma by 2, common practice

                # --- Fake images ---
                with torch.no_grad():
                    # Request intermediate output for multi-level CLIP later
                    # Note: generator returns final_image, intermediate_image, kl_loss, routing_probs
                    fake_images_16, fake_images_8, _, _ = generator(
                        z, 
                        text_embeddings, 
                        return_intermediate=True, 
                        return_routing=True,
                        annealing_factor=temperature_factor  # Pass temperature factor
                    )
                fake_pred = discriminator(fake_images_16.detach(), text_embeddings) # Use final fake image (16x16)

                # --- Mismatched real images/text ---
                # Simple shuffle within the batch for mismatching
                shuffled_indices = torch.randperm(batch_size)
                mismatched_text_embeddings = text_embeddings[shuffled_indices]
                mismatched_pred = discriminator(real_images.detach(), mismatched_text_embeddings) # Use detached real images

                # --- Discriminator loss (includes matching-aware) ---
                # Loss now includes real, fake, and mismatched terms
                d_loss_gan = gan_loss.discriminator_loss(real_pred, fake_pred, mismatched_pred)

                # Total Discriminator Loss
                d_loss = d_loss_gan + r1_loss
            
            # Check for NaN/Inf in discriminator loss
            if torch.isnan(d_loss).item() or torch.isinf(d_loss).item():
                print(f"⚠️ NaN/Inf detected in discriminator loss! Skipping this batch.")
                # Set d_loss to zero to avoid NaN/Inf in optimizer step
                original_requires_grad = d_loss.requires_grad
                d_loss = torch.tensor(0.0, device=device, requires_grad=original_requires_grad)
                continue
            
            # Scale the loss if using AMP
            if scaler:
                scaler.scale(d_loss / current_accumulation_steps).backward(retain_graph=True)
            else:
                (d_loss / current_accumulation_steps).backward(retain_graph=True)
            
            # Only step optimizer after accumulating gradients
            if (batch_idx + 1) % current_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                if scaler:
                    # Clip before scaler step
                    scaler.unscale_(optimizer_d)
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=0.7)
                    scaler.step(optimizer_d)
                else:
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=0.7)
                    optimizer_d.step()
                
                if scaler:
                    scaler.update()
                
                # Clear discriminator part of computation graph from memory
                del grad_real, grad_penalty, real_pred, fake_pred, mismatched_pred
                
                # Periodic memory check after discriminator step
                if torch.cuda.is_available() and batch_idx % (current_accumulation_steps * 5) == 0:
                    current_mem = torch.cuda.memory_allocated() / 1e9
                    print(f"Memory after D step: {current_mem:.2f} GB")

            # ------------------------
            # Train Generator
            # ------------------------
            if batch_idx % current_accumulation_steps == 0:
                optimizer_g.zero_grad()

            with torch.cuda.amp.autocast() if use_amp else nullcontext():
                # Generate fake images, get intermediate, KL loss, and routing probs
                fake_images_16, fake_images_8, kl_loss, routing_probs = generator(
                    z, 
                    text_embeddings, 
                    return_intermediate=True, 
                    return_routing=True,
                    annealing_factor=temperature_factor  # Pass temperature factor
                )

                # Apply safe handling to KL loss - keep printing high KL values
                if kl_loss is not None:
                    # Print high KL values
                    if kl_loss > 50.0:
                        kl_loss = torch.clamp(kl_loss, max=50.0)
                    # Only reset for NaN/Inf
                    if torch.isnan(kl_loss).item() or torch.isinf(kl_loss).item():
                        print(f"⚠️ NaN/Inf detected in KL loss! Replacing with zero.")
                        # Ensure kl_loss requires grad if it's part of the computation graph before replacement
                        original_requires_grad = kl_loss.requires_grad
                        kl_loss = torch.tensor(0.0, device=device, requires_grad=original_requires_grad)
                            
                # Discriminator prediction on fake images (16x16 resolution)
                fake_pred_g = discriminator(fake_images_16, text_embeddings)

                # --- Generator Adversarial Loss ---
                g_loss_gan = gan_loss.generator_loss(fake_pred_g)

                # --- Multi-level CLIP Loss ---
                clip_loss_16 = gan_loss.compute_clip_loss(fake_images_16, text_embeddings)
                clip_loss_8 = gan_loss.compute_clip_loss(fake_images_8, text_embeddings)
                g_loss_clip = (clip_weight_16 * clip_loss_16) + (clip_weight_8 * clip_loss_8)

                # --- MoE Balance Loss ---
                balance_loss = gan_loss.moe_balance_loss(routing_probs, balance_weight=balance_weight)

                # --- Total Generator Loss ---
                g_loss = g_loss_gan + g_loss_clip + balance_loss
                
                # Check for NaN/Inf in generator loss
                if torch.isnan(g_loss).item() or torch.isinf(g_loss).item():
                    print("⚠️ NaN or Inf detected in generator loss! Resetting to zero.")
                    original_requires_grad = g_loss.requires_grad
                    g_loss = torch.tensor(0.0, device=device, requires_grad=original_requires_grad)
                
                # Add KL loss separately - apply effective KL weight with annealing
                if kl_loss is not None:
                    kl_component = effective_kl_weight * kl_loss
                    g_loss = g_loss + kl_component
            
            # Scale the loss if using AMP
            if scaler:
                scaler.scale(g_loss / current_accumulation_steps).backward()
            else:
                (g_loss / current_accumulation_steps).backward()
            
            # Only step optimizer after accumulating gradients
            if (batch_idx + 1) % current_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                if scaler:
                    # Clip before scaler step
                    scaler.unscale_(optimizer_g)
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=0.8)
                    scaler.step(optimizer_g)
                else:
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=0.8)
                    optimizer_g.step()
                
                if scaler:
                    scaler.update()
                
                # Periodic memory check and cleanup after generator step
                if torch.cuda.is_available() and batch_idx % (current_accumulation_steps * 5) == 0:
                    peak_mem = torch.cuda.max_memory_allocated() / 1e9
                    current_mem = torch.cuda.memory_allocated() / 1e9
                    print(f"Memory after G step: {current_mem:.2f} GB, Peak: {peak_mem:.2f} GB")
                    
                    if peak_mem > batch_memory_limit * 0.8:  # Over 80% of limit
                        print(f"⚠️ High memory usage detected: {peak_mem:.2f} GB")
                        # Force cleanup
                        torch.cuda.empty_cache()
                        gc.collect()
                        # Reset peak stats after cleanup
                        torch.cuda.reset_peak_memory_stats()

            # Update running losses for progress bar
            running_d_loss = 0.9 * running_d_loss + 0.1 * d_loss_gan.item()  # Track GAN part of D loss
            running_r1_loss = 0.9 * running_r1_loss + 0.1 * r1_loss.item()
            running_g_loss = 0.9 * running_g_loss + 0.1 * g_loss_gan.item()  # Track GAN part of G loss
            running_kl_loss = 0.9 * running_kl_loss + 0.1 * kl_loss.item() if kl_loss is not None else running_kl_loss
            running_balance_loss = 0.9 * running_balance_loss + 0.1 * balance_loss.item()
            running_clip_loss_16 = 0.9 * running_clip_loss_16 + 0.1 * clip_loss_16.item()
            running_clip_loss_8 = 0.9 * running_clip_loss_8 + 0.1 * clip_loss_8.item()

            # Update progress bar description with loss values
            epoch_pbar.set_postfix({
                'D_loss': f'{running_d_loss:.3f}',
                'R1': f'{running_r1_loss:.3f}',
                'G_loss': f'{running_g_loss:.3f}',
                'KL': f'{running_kl_loss:.4f}',
                'Balance': f'{running_balance_loss:.4f}',
                'Clip16': f'{running_clip_loss_16:.3f}',
                'Clip8': f'{running_clip_loss_8:.3f}',
            })

            # Clean up batch-specific tensors
            del real_images, text_embeddings, z
            del fake_images_16, fake_images_8, routing_probs
            if 'fake_pred_g' in locals(): del fake_pred_g

            # Logging
            if step % log_interval == 0:
                logger.info(f"\nStep [{step}] Epoch [{epoch+1}] Batch [{batch_idx}/{len(dataloader)}] "
                      f"D_loss: {d_loss.item():.4f} (GAN: {d_loss_gan.item():.4f}, R1: {r1_loss.item():.4f}), "
                      f"G_loss: {g_loss.item():.4f} (GAN: {g_loss_gan.item():.4f}, Clip16: {clip_loss_16.item():.4f}, "
                      f"Clip8: {clip_loss_8.item():.4f}, KL: {kl_loss.item() if kl_loss is not None else 0:.4f}, "
                      f"Balance: {balance_loss.item():.4f})")
                
                if torch.cuda.is_available():
                    current_mem = torch.cuda.memory_allocated() / 1e9
                    peak_mem = torch.cuda.max_memory_allocated() / 1e9
                    print(f"Current memory: {current_mem:.2f} GB, Peak: {peak_mem:.2f} GB")

            # Save models at step intervals if specified
            # if step % save_interval == 0 and step > 0:
            #     # Clean GPU memory before saving to avoid OOM
            #     torch.cuda.empty_cache()
                
            #     checkpoint_path = os.path.join(save_dir, f"aurora_checkpoint_{step}.pt")
            #     torch.save({
            #         'generator': generator.state_dict(),
            #         'discriminator': discriminator.state_dict(),
            #         'optimizer_g': optimizer_g.state_dict(),
            #         'optimizer_d': optimizer_d.state_dict(),
            #         'epoch': epoch,
            #         'step': step
            #     }, checkpoint_path)
                
            #     print(f"Checkpoint saved to {checkpoint_path}")
                
            step += 1

        # Close the progress bar for this epoch
        epoch_pbar.close()
        
        # End of epoch memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
        
        # Log memory stats at end of epoch
        if torch.cuda.is_available():
            print(f"Memory after epoch {epoch+1}: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            torch.cuda.reset_peak_memory_stats()
                
        # Report epoch OOM count
        if epoch_oom_count > 0:
            print(f"⚠️ Epoch had {epoch_oom_count} OOM warnings")
        
        # Step learning rate schedulers after warmup period
        if epoch >= lr_warmup_epochs:
            lr_scheduler_g.step()
            lr_scheduler_d.step()

        # ===== VALIDATION SECTION =====
        if val_dataloader is not None:
            generator.eval()
            discriminator.eval()

            val_g_loss_gan = 0
            val_d_loss_gan = 0
            val_clip_loss_16 = 0
            val_clip_loss_8 = 0
            val_samples = 0
            
            # Memory cleanup before validation
            torch.cuda.empty_cache()
            gc.collect()

            print("Running validation...")
            val_pbar = tqdm(val_dataloader, desc=f"Validation Epoch {epoch+1}")

            with torch.no_grad():
                for val_real_images, val_text_embeddings in val_pbar:
                    val_batch_size = val_real_images.size(0)
                    val_samples += val_batch_size

                    val_real_images = val_real_images.to(device)
                    val_text_embeddings = val_text_embeddings.to(device)
                    val_z = torch.randn(val_batch_size, LATENT_DIM, device=device)

                    # Memory check before validation step
                    if torch.cuda.is_available() and batch_memory_limit:
                        current_memory = torch.cuda.memory_allocated() / 1e9
                        if current_memory > batch_memory_limit * 0.9:  # 90% of limit
                            print(f"⚠️ High memory during validation: {current_memory:.2f} GB")
                            torch.cuda.empty_cache()
                            gc.collect()
                    
                    # Real images
                    val_real_pred = discriminator(val_real_images, val_text_embeddings)

                    # Fake images - Pass temperature factor
                    val_fake_images_16, val_fake_images_8, val_kl_loss = generator(
                        val_z, 
                        val_text_embeddings, 
                        return_intermediate=True,
                        annealing_factor=temperature_factor
                    )

                    # Fake images prediction
                    val_fake_pred = discriminator(val_fake_images_16, val_text_embeddings)

                    # Mismatched prediction
                    val_shuffled_indices = torch.randperm(val_batch_size)
                    val_mismatched_text = val_text_embeddings[val_shuffled_indices]
                    val_mismatched_pred = discriminator(val_real_images, val_mismatched_text)

                    # Losses
                    batch_d_loss = gan_loss.discriminator_loss(
                        val_real_pred, val_fake_pred, val_mismatched_pred
                    ).item()
                    
                    batch_g_loss = gan_loss.generator_loss(
                        val_fake_pred, val_kl_loss, kl_weight=effective_kl_weight
                    ).item()
                    
                    batch_clip_loss_16 = gan_loss.compute_clip_loss(
                        val_fake_images_16, val_text_embeddings
                    ).item()
                    
                    batch_clip_loss_8 = gan_loss.compute_clip_loss(
                        val_fake_images_8, val_text_embeddings
                    ).item()

                    val_d_loss_gan += batch_d_loss * val_batch_size
                    val_g_loss_gan += batch_g_loss * val_batch_size
                    val_clip_loss_16 += batch_clip_loss_16 * val_batch_size
                    val_clip_loss_8 += batch_clip_loss_8 * val_batch_size

                    val_pbar.set_postfix({
                        'D_loss': f'{batch_d_loss:.4f}',
                        'G_loss': f'{batch_g_loss:.4f}',
                        'Clip16': f'{batch_clip_loss_16:.4f}',
                        'Clip8': f'{batch_clip_loss_8:.4f}'
                    })
                    
                    # Clean up validation batch tensors
                    del val_real_images, val_text_embeddings, val_z
                    del val_fake_images_16, val_fake_images_8
                    del val_real_pred, val_fake_pred, val_mismatched_pred

            val_pbar.close()
            
            # Memory cleanup after validation
            torch.cuda.empty_cache()
            gc.collect()
            
            # Average losses
            if val_samples > 0:
                val_d_loss_gan /= val_samples
                val_g_loss_gan /= val_samples
                val_clip_loss_16 /= val_samples
                val_clip_loss_8 /= val_samples

                # Collect metrics for reporting
                val_metrics = {
                    'val_d_loss': val_d_loss_gan,
                    'val_g_loss': val_g_loss_gan,
                    'val_clip_loss_16': val_clip_loss_16,
                    'val_clip_loss_8': val_clip_loss_8,
                    'val_clip_loss': val_clip_loss_16  # Primary metric for hyperparameter tuning
                }

                print(f"Validation Results - D_loss: {val_d_loss_gan:.4f}, G_loss: {val_g_loss_gan:.4f}, "
                     f"Clip_Loss_16: {val_clip_loss_16:.4f}, Clip_Loss_8: {val_clip_loss_8:.4f}")
                
                # Call the metric callback if provided
                if metric_callback:
                    if not metric_callback(epoch, val_metrics):
                        # If the callback returns False, stop training
                        print("Early stopping triggered by metric callback")
                        break

            generator.train()
            discriminator.train()

        # Save checkpoint at the end of each epoch
        # epoch_save_path = os.path.join(save_dir, f"aurora_checkpoint_epoch_{epoch+1}.pt")
        # torch.save({
        #     'generator': generator.state_dict(),
        #     'discriminator': discriminator.state_dict(),
        #     'optimizer_g': optimizer_g.state_dict(),
        #     'optimizer_d': optimizer_d.state_dict(),
        #     'epoch': epoch + 1,
        #     'step': step
        # }, epoch_save_path)
        
        # print(f"Epoch {epoch+1} checkpoint saved to {epoch_save_path}")
    
    # Final memory cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    # Save final model
    # final_save_path = os.path.join(save_dir, "aurora_final.pt")
    # torch.save({
    #     'generator': generator.state_dict(),
    #     'discriminator': discriminator.state_dict(),
    #     'epoch': num_epochs,
    #     'step': step
    # }, final_save_path)
    
    # print(f"Final model saved to {final_save_path}")
    
    return generator, discriminator


def sample_aurora_gan(generator, text_prompt, num_samples=1, truncation_psi=0.7, device=DEVICE):
    """
    Generate images from text prompt using the Aurora GAN.
    !! Needs adjustment for new generator return signature !!
    """
    generator.eval()

    # Sample random noise
    z = torch.randn(num_samples, LATENT_DIM, device=device)

    # Generate images
    with torch.no_grad():
        # Adjust the call to match the generator's forward method
        # We only need the final image for sampling
        # Assuming the non-intermediate, non-routing call returns (final_image, kl_loss)
        fake_images, _ = generator(z, text_prompt, truncation_psi=truncation_psi, return_intermediate=False, return_routing=False) # Adjusted call

    return fake_images
