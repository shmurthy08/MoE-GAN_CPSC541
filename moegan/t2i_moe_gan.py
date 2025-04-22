# aurora_gan_moe.py
import os
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

# Constants
LATENT_DIM = 512
TEXT_EMBEDDING_DIM = 512
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_EXPERTS = 8  # Number of experts in MoE layers
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
    
    if _clip_model is None:
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
        self.feature_mu = nn.Parameter(torch.Tensor(feature_dim, 128))
        nn.init.kaiming_normal_(self.feature_mu, mode='fan_in', nonlinearity='leaky_relu')
        self.feature_rho = nn.Parameter(torch.Tensor(feature_dim, 128).fill_(-3))
        
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
        Calculate KL divergence between posterior and prior distributions.
        Fixed to handle different tensor dimensions properly.
        """
        kl_div = 0.0
        
        # Prior: standard normal distribution
        prior_mean = 0.0
        prior_var = 1.0
        
        # Calculate KL for feature weights
        posterior_var_f = torch.log(1 + torch.exp(self.feature_rho)) ** 2
        kl_div += 0.5 * torch.sum(
            torch.log(prior_var / posterior_var_f) +
            (posterior_var_f + self.feature_mu ** 2) / prior_var - 1
        )
        
        # Calculate KL for text weights
        posterior_var_t = torch.log(1 + torch.exp(self.text_rho)) ** 2
        kl_div += 0.5 * torch.sum(
            torch.log(prior_var / posterior_var_t) +
            (posterior_var_t + self.text_mu ** 2) / prior_var - 1
        )
        
        # Calculate KL for combined weights
        posterior_var_c = torch.log(1 + torch.exp(self.combined_rho)) ** 2
        kl_div += 0.5 * torch.sum(
            torch.log(prior_var / posterior_var_c) +
            (posterior_var_c + self.combined_mu ** 2) / prior_var - 1
        )
        
        return kl_div * 0.0001
    
    
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
        
        x_reshaped = x.permute(0, 2, 3, 1).contiguous().view(-1, channels)
        w_reshaped = w.unsqueeze(1).unsqueeze(1).expand(batch_size, height, width, -1).contiguous().view(-1, w.size(1))
        
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
        
        # Project text sequence to match spatial feature dimension
        text_proj = self.text_proj(text_seq)
        
        # Cross-attention
        x_norm = self.norm2(x_flat)
        ca_out, _ = self.cross_attn(x_norm, text_proj, text_proj)
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
    def __init__(self, latent_dim=LATENT_DIM, text_embedding_dim=512):
        super(AuroraGenerator, self).__init__()

        self.latent_dim = latent_dim
        self.text_embedding_dim = text_embedding_dim

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

        # Generative blocks with increasing resolution
        self.gen_block_4 = GenerativeBlock(512, 512, text_dim=text_embedding_dim, upsample=False, resolution=4)
        self.gen_block_8 = GenerativeBlock(512, 256, text_dim=text_embedding_dim, upsample=True, resolution=8)
        self.gen_block_16 = GenerativeBlock(256, 128, text_dim=text_embedding_dim, upsample=True, resolution=16)
        self.gen_block_32 = GenerativeBlock(128, 64, text_dim=text_embedding_dim, upsample=True, resolution=32)
        self.gen_block_64 = GenerativeBlock(64, 32, text_dim=text_embedding_dim, upsample=True, resolution=64)

        # To RGB layers for different resolutions
        self.to_rgb_32 = ModulatedConv(64, 3, kernel_size=1)
        self.to_rgb_64 = ModulatedConv(32, 3, kernel_size=1)

    def encode_text(self, text_or_embeddings):
        """
        Encode text using CLIP model if text is a string.
        """
        if isinstance(text_or_embeddings, str) or (isinstance(text_or_embeddings, list) and isinstance(text_or_embeddings[0], str)):
            text_embeddings = encode_text_with_clip(text_or_embeddings)
        else:
            text_embeddings = text_or_embeddings
        return text_embeddings

    # !! MODIFIED: Added return_intermediate flag !!
    def forward(self, z, text_input, truncation_psi=0.7, return_routing=False, return_intermediate=False):
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
                        torch.zeros(1, self.latent_dim, device=z.device), # Use z.device
                        torch.zeros(1, self.text_embedding_dim, device=z.device) # Use z.device
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
        x_32 = self.to_rgb_32(x, w) # Keep 32x32 output

        # Generate 64x64 image (final resolution)
        x, r_probs = self.gen_block_64(x, w, text_seq, kl_losses)
        routing_probs.append(r_probs)
        x_64 = self.to_rgb_64(x, w) # Keep 64x64 output

        # Combine outputs from different resolutions (optional, but common in progressive GANs)
        # Ensure interpolation happens before adding
        x_32_upsampled = F.interpolate(x_32, scale_factor=2, mode='bilinear', align_corners=False)
        final_image = x_64 + x_32_upsampled # The paper might do this differently, check Fig 2/StyleGAN details if needed

        # Calculate total KL loss
        kl_loss = sum(kl_losses) if kl_losses else torch.tensor(0.0, device=x.device) # Use x.device

        # !! MODIFIED Return Value !!
        if return_routing and return_intermediate:
            return final_image, x_32, kl_loss, routing_probs
        elif return_routing:
             # Need to decide what intermediate value means here, returning None for x_32
            return final_image, None, kl_loss, routing_probs
        elif return_intermediate:
            return final_image, x_32, kl_loss
        else:
            # Original return signature for inference might need None for x_32
            return final_image, kl_loss


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
        # Input channels: 512 (image features) + 128 (text features)
        self.output_layer = nn.Sequential(
            nn.Conv2d(512 + 128, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # Removed Sigmoid - typically use raw logits for stability with BCEWithLogitsLoss or softplus
        )

    def forward(self, img, text_embedding):
        # Process image
        features = self.conv_layers(img) # Shape: [B, 512, 4, 4]

        # Process text embedding
        text_features = self.text_projection(text_embedding) # Shape: [B, 128]

        # Replicate text features to match feature map size
        text_features = text_features.unsqueeze(-1).unsqueeze(-1) # Shape: [B, 128, 1, 1]
        text_features = text_features.repeat(1, 1, features.shape[2], features.shape[3]) # Shape: [B, 128, 4, 4]

        # Concatenate image features and text features
        combined = torch.cat([features, text_features], dim=1) # Shape: [B, 512+128, 4, 4]

        # Final output (logits)
        output = self.output_layer(combined) # Shape: [B, 1, 1, 1]

        return output.view(-1) # Return logits as a flat tensor [B]

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

        # Add KL divergence loss if provided
        if kl_loss is not None:
            g_loss = g_loss + kl_weight * kl_loss

        # Note: CLIP loss is handled separately in the training loop now for multi-level

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

        # Use probabilities from the last MoE layer
        last_probs = routing_probs[-1] # Shape [batch*h*w, num_experts]

        if last_probs is None or last_probs.numel() == 0:
             return torch.tensor(0.0, device=self.device)

        num_experts = last_probs.size(1)
        batch_items = last_probs.size(0) # Total number of items routed

        # Sum probabilities per expert across all items routed
        load = last_probs.sum(dim=0) # Shape [num_experts]

        # Compute fraction of items routed to each expert (P_i in paper)
        fraction_routed = load / batch_items

        # Compute average probability per expert across items (f_i in paper)
        # Need sum of squares of probabilities per expert, then sum over items
        sum_probs_per_item = last_probs.sum(dim=1) # Should ideally be 1 if hard routing, may vary slightly
        avg_prob_per_expert = (last_probs / (sum_probs_per_item.unsqueeze(1) + 1e-10)).sum(dim=0) / batch_items

        # Balance loss = N * sum(fraction_routed * avg_prob_per_expert)
        # Paper uses alpha * N * sum(f_i * P_i), where alpha is the balance_weight
        balance_loss = num_experts * torch.sum(fraction_routed * avg_prob_per_expert)

        return balance_weight * balance_loss

from tqdm import tqdm

def train_aurora_gan(
    dataloader, val_dataloader=None,
    num_epochs=50, lr=0.0002, beta1=0.5, beta2=0.999,
    r1_gamma=10.0,
    clip_weight_64=0.1,
    clip_weight_32=0.05,
    kl_weight=0.001,
    balance_weight = 0.01,
    device=DEVICE, save_dir='./aurora_checkpoints',
    log_interval=10, save_interval=1000,
    metric_callback=None  # Add metric callback parameter
):
    """
    Train the Aurora GAN model with R1, Matching-Aware, Multi-level CLIP loss.
    
    Args:
        dataloader: Training data loader
        val_dataloader: Optional validation data loader
        num_epochs: Number of training epochs
        lr: Learning rate
        beta1: Beta1 for Adam optimizer
        beta2: Beta2 for Adam optimizer
        r1_gamma: R1 regularization weight
        clip_weight_64: CLIP loss weight for 64x64 resolution
        clip_weight_32: CLIP loss weight for 32x32 resolution
        kl_weight: KL divergence loss weight
        balance_weight: MoE balance loss weight
        device: Device to train on
        save_dir: Directory to save checkpoints
        log_interval: How often to log metrics
        save_interval: How often to save checkpoints
        metric_callback: Optional callback function for reporting metrics for hyperparameter tuning
        
    Returns:
        Trained generator and discriminator models
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
        print(f"Starting epoch {epoch+1}/{num_epochs}")

        epoch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        running_d_loss = 0.0
        running_g_loss = 0.0
        running_r1_loss = 0.0
        running_kl_loss = 0.0
        running_balance_loss = 0.0
        running_clip_loss_64 = 0.0
        running_clip_loss_32 = 0.0

        for batch_idx, (real_images, text_embeddings) in enumerate(epoch_pbar):
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

            # --- Real images (Matching) ---
            # Enable gradient computation for R1 penalty
            real_images.requires_grad = True
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
                fake_images_64, fake_images_32, _, _ = generator(z, text_embeddings, return_intermediate=True, return_routing=True)
            fake_pred = discriminator(fake_images_64.detach(), text_embeddings) # Use final fake image

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
            d_loss.backward()
            optimizer_d.step()

            # ------------------------
            # Train Generator
            # ------------------------
            optimizer_g.zero_grad()

            # Generate fake images, get intermediate, KL loss, and routing probs
            fake_images_64, fake_images_32, kl_loss, routing_probs = generator(z, text_embeddings, return_intermediate=True, return_routing=True)

            # Discriminator prediction on fake images (use final resolution)
            fake_pred_g = discriminator(fake_images_64, text_embeddings) # Use G's output directly

            # --- Generator Adversarial Loss ---
            g_loss_gan = gan_loss.generator_loss(fake_pred_g, kl_loss, kl_weight=kl_weight)

            # --- Multi-level CLIP Loss ---
            clip_loss_64 = gan_loss.compute_clip_loss(fake_images_64, text_embeddings)
            clip_loss_32 = gan_loss.compute_clip_loss(fake_images_32, text_embeddings) # CLIP loss on 32x32
            g_loss_clip = (clip_weight_64 * clip_loss_64) + (clip_weight_32 * clip_loss_32)

            # --- MoE Balance Loss ---
            balance_loss = gan_loss.moe_balance_loss(routing_probs, balance_weight=balance_weight)

            # --- Total Generator Loss ---
            g_loss = g_loss_gan + g_loss_clip + balance_loss
            g_loss.backward()
            optimizer_g.step()

            # Update running losses for progress bar
            running_d_loss = 0.9 * running_d_loss + 0.1 * d_loss_gan.item() # Track gan part of D loss
            running_r1_loss = 0.9 * running_r1_loss + 0.1 * r1_loss.item()
            running_g_loss = 0.9 * running_g_loss + 0.1 * g_loss_gan.item() # Track gan part of G loss
            running_kl_loss = 0.9 * running_kl_loss + 0.1 * kl_loss.item() if kl_loss is not None else running_kl_loss
            running_balance_loss = 0.9 * running_balance_loss + 0.1 * balance_loss.item()
            running_clip_loss_64 = 0.9 * running_clip_loss_64 + 0.1 * clip_loss_64.item()
            running_clip_loss_32 = 0.9 * running_clip_loss_32 + 0.1 * clip_loss_32.item()


            # Update progress bar description with loss values
            epoch_pbar.set_postfix({
                'D_loss': f'{running_d_loss:.3f}',
                'R1': f'{running_r1_loss:.3f}',
                'G_loss': f'{running_g_loss:.3f}',
                'KL': f'{running_kl_loss:.4f}',
                'Balance': f'{running_balance_loss:.4f}',
                'Clip64': f'{running_clip_loss_64:.3f}',
                'Clip32': f'{running_clip_loss_32:.3f}',
            })

            # Logging
            if step % log_interval == 0:
                logger.info(f"\nStep [{step}] Epoch [{epoch+1}] Batch [{batch_idx}/{len(dataloader)}] "
                      f"D_loss: {d_loss.item():.4f} (GAN: {d_loss_gan.item():.4f}, R1: {r1_loss.item():.4f}), "
                      f"G_loss: {g_loss.item():.4f} (GAN: {g_loss_gan.item():.4f}, Clip64: {clip_loss_64.item():.4f}, Clip32: {clip_loss_32.item():.4f}, KL: {kl_loss.item():.4f}, Balance: {balance_loss.item():.4f})")


            # Save models
            if step % save_interval == 0 and step > 0: # Avoid saving at step 0
                torch.save({
                    'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'optimizer_g': optimizer_g.state_dict(),
                    'optimizer_d': optimizer_d.state_dict(),
                    'epoch': epoch,
                    'step': step
                }, os.path.join(save_dir, f"aurora_checkpoint_{step}.pt"))

            step += 1

        # Close the progress bar for this epoch
        epoch_pbar.close()

        # Validation (if provided) - Needs adjustment for new generator return values
        if val_dataloader is not None:
            generator.eval()
            discriminator.eval()

            val_g_loss_gan = 0
            val_d_loss_gan = 0
            val_clip_loss_64 = 0
            val_clip_loss_32 = 0
            val_samples = 0

            print("Running validation...")
            val_pbar = tqdm(val_dataloader, desc=f"Validation Epoch {epoch+1}")

            with torch.no_grad():
                for val_real_images, val_text_embeddings in val_pbar:
                    val_batch_size = val_real_images.size(0)
                    val_samples += val_batch_size

                    val_real_images = val_real_images.to(device)
                    val_text_embeddings = val_text_embeddings.to(device)
                    val_z = torch.randn(val_batch_size, LATENT_DIM, device=device)

                    # Real images
                    val_real_pred = discriminator(val_real_images, val_text_embeddings)

                    # Fake images - Adjust call based on modified generator forward
                    # We only need final image and KL for validation loss calculation usually
                    val_fake_images_64, val_fake_images_32, val_kl_loss = generator(val_z, val_text_embeddings, return_intermediate=True) # Get KL loss

                    # Fake images prediction
                    val_fake_pred = discriminator(val_fake_images_64, val_text_embeddings)

                    # Mismatched prediction (optional but good for consistency)
                    val_shuffled_indices = torch.randperm(val_batch_size)
                    val_mismatched_text = val_text_embeddings[val_shuffled_indices]
                    val_mismatched_pred = discriminator(val_real_images, val_mismatched_text)

                    # Losses
                    batch_d_loss = gan_loss.discriminator_loss(val_real_pred, val_fake_pred, val_mismatched_pred).item()
                    batch_g_loss = gan_loss.generator_loss(val_fake_pred, val_kl_loss, kl_weight=kl_weight).item() # Only GAN + KL part for validation G loss
                    batch_clip_loss_64 = gan_loss.compute_clip_loss(val_fake_images_64, val_text_embeddings).item()
                    batch_clip_loss_32 = gan_loss.compute_clip_loss(val_fake_images_32, val_text_embeddings).item()

                    val_d_loss_gan += batch_d_loss * val_batch_size
                    val_g_loss_gan += batch_g_loss * val_batch_size
                    val_clip_loss_64 += batch_clip_loss_64 * val_batch_size
                    val_clip_loss_32 += batch_clip_loss_32 * val_batch_size

                    val_pbar.set_postfix({
                        'D_loss': f'{batch_d_loss:.4f}',
                        'G_loss': f'{batch_g_loss:.4f}',
                        'Clip64': f'{batch_clip_loss_64:.4f}',
                        'Clip32': f'{batch_clip_loss_32:.4f}'
                    })

            val_pbar.close()

            # Average losses
            if val_samples > 0:
                val_d_loss_gan /= val_samples
                val_g_loss_gan /= val_samples
                val_clip_loss_64 /= val_samples
                val_clip_loss_32 /= val_samples

                # Collect metrics for reporting
                val_metrics = {
                    'val_d_loss': val_d_loss_gan,
                    'val_g_loss': val_g_loss_gan,
                    'val_clip_loss_64': val_clip_loss_64,
                    'val_clip_loss_32': val_clip_loss_32,
                    'val_clip_loss': val_clip_loss_64  # Primary metric for HPO
                }

                print(f"Validation Results - D_loss: {val_d_loss_gan:.4f}, G_loss: {val_g_loss_gan:.4f}, "
                     f"Clip_Loss_64: {val_clip_loss_64:.4f}, Clip_Loss_32: {val_clip_loss_32:.4f}")
                
                # Call the metric callback if provided
                if metric_callback:
                    if not metric_callback(epoch, val_metrics):
                        # If the callback returns False, stop training
                        print("Early stopping triggered by metric callback")
                        break

            generator.train()
            discriminator.train()

        # Save model after each epoch
        epoch_save_path = os.path.join(save_dir, f"aurora_checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'optimizer_g': optimizer_g.state_dict(),
            'optimizer_d': optimizer_d.state_dict(),
            'epoch': epoch + 1,
            'step': step
        }, epoch_save_path)
        print(f"Checkpoint saved to {epoch_save_path}")

    # Save final models
    final_save_path = os.path.join(save_dir, "aurora_final.pt")
    torch.save({
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'optimizer_g': optimizer_g.state_dict(),
        'optimizer_d': optimizer_d.state_dict(),
        'epoch': num_epochs,
        'step': step
    }, final_save_path)
    print(f"Final model saved to {final_save_path}")

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
