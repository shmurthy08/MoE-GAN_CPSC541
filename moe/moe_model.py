import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal, kl_divergence

class BayesianLinear(nn.Module):
    """
    Bayesian Linear Layer that implements the reparameterization trick for
    sampling weights from a distribution during forward passes.
    
    This allows for uncertainty estimation in the gating network.
    """
    def __init__(self, in_features, out_features, prior_sigma1=1.0, prior_sigma2=0.0025, prior_pi=0.5):
        """
        Initialize Bayesian Linear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            prior_sigma1: Prior sigma for the first Gaussian component (default: 1.0)
            prior_sigma2: Prior sigma for the second Gaussian component (default: 0.0025)
            prior_pi: Mixing coefficient for the prior (default: 0.5)
        """
        super(BayesianLinear, self).__init__()
        
        # Layer dimensions
        self.in_features = in_features
        self.out_features = out_features
        
        # Prior distribution parameters
        self.prior_sigma1 = prior_sigma1
        self.prior_sigma2 = prior_sigma2
        self.prior_pi = prior_pi
        
        # Weight parameters (mean and rho for the posterior)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).normal_(-3, 0.1))
        
        # Bias parameters (mean and rho for the posterior)
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).normal_(-3, 0.1))
        
        # Initialize log prior and log posterior
        self.log_prior = 0
        self.log_posterior = 0

    def forward(self, x, sample=True):
        """
        Forward pass with optional sampling.
        
        Args:
            x: Input tensor
            sample: Whether to sample from the posterior (default: True)
        
        Returns:
            Output tensor and KL divergence
        """
        # If sampling, draw random weights from the posterior
        if sample:
            # Convert rho to sigma using softplus function
            weight_sigma = torch.log1p(torch.exp(self.weight_rho))
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            
            # Sample from the posterior using the reparameterization trick
            weight_epsilon = torch.randn_like(self.weight_mu)
            bias_epsilon = torch.randn_like(self.bias_mu)
            
            weight = self.weight_mu + weight_epsilon * weight_sigma
            bias = self.bias_mu + bias_epsilon * bias_sigma
            
            # Compute log posterior and log prior for KL divergence
            self.log_posterior = self._log_gaussian(weight, self.weight_mu, weight_sigma).sum() + \
                              self._log_gaussian(bias, self.bias_mu, bias_sigma).sum()
            
            self.log_prior = self._log_gaussian_mixture(weight, 0., self.prior_sigma1, self.prior_sigma2, self.prior_pi).sum() + \
                          self._log_gaussian_mixture(bias, 0., self.prior_sigma1, self.prior_sigma2, self.prior_pi).sum()
        else:
            # Use mean weights if not sampling
            weight = self.weight_mu
            bias = self.bias_mu
            self.log_prior = 0
            self.log_posterior = 0
        
        # Perform linear operation
        output = F.linear(x, weight, bias)
        
        # Return output and KL divergence (if sampling)
        kl_divergence = self.log_posterior - self.log_prior if sample else 0
        
        return output, kl_divergence

    def _log_gaussian(self, x, mu, sigma):
        """
        Compute log probability of Gaussian distribution.
        
        Args:
            x: Input tensor
            mu: Mean of the Gaussian
            sigma: Standard deviation of the Gaussian
        
        Returns:
            Log probability
        """
        pi_term = torch.tensor(2 * np.pi, device=x.device)
        return -0.5 * torch.log(pi_term * sigma**2) - (x - mu)**2 / (2 * sigma**2)
    def _log_gaussian_mixture(self, x, mu, sigma1, sigma2, pi):
        """
        Compute log probability of Gaussian mixture distribution.
        
        Args:
            x: Input tensor
            mu: Mean of the Gaussian
            sigma1: Standard deviation of the first Gaussian component
            sigma2: Standard deviation of the second Gaussian component
            pi: Mixing coefficient
        
        Returns:
            Log probability
        """
        log_gaussian1 = self._log_gaussian(x, mu, sigma1)
        log_gaussian2 = self._log_gaussian(x, mu, sigma2)
        return torch.log(pi * torch.exp(log_gaussian1) + (1 - pi) * torch.exp(log_gaussian2))


class BayesianMoEGatingNetwork(nn.Module):
    """
    Bayesian Mixture of Experts Gating Network that determines which expert to use
    for each input prompt, with uncertainty estimation through Bayesian techniques.
    """
    def __init__(self, input_dim, hidden_dim, num_experts, num_samples=10):
        """
        Initialize Bayesian MoE Gating Network.
        
        Args:
            input_dim: Dimension of the input (CLIP text embeddings)
            hidden_dim: Dimension of the hidden layer
            num_experts: Number of experts in the MoE model
            num_samples: Number of MC samples for uncertainty estimation (default: 10)
        """
        super(BayesianMoEGatingNetwork, self).__init__()
        
        # Model parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_samples = num_samples
        
        # Bayesian layers
        self.bayesian_layer1 = BayesianLinear(input_dim, hidden_dim)
        self.bayesian_layer2 = BayesianLinear(hidden_dim, hidden_dim)
        self.bayesian_layer3 = BayesianLinear(hidden_dim, num_experts)
        
        # Activation function
        self.activation = nn.ReLU()

    def monte_carlo_sample(self, x, num_samples=None):
        """
        Perform Monte Carlo sampling to estimate uncertainty without recursion.
        
        Args:
            x: Input tensor (original CLIP embeddings)
            num_samples: Number of samples (default: None, uses self.num_samples)
        
        Returns:
            Mean probabilities and uncertainty
        """
        if num_samples is None:
            num_samples = self.num_samples
            
        with torch.no_grad():
            mc_samples = []
            for _ in range(num_samples):
                # Run the forward pass with sample=True but calculate_log_probs=False
                # First layer
                h1, _ = self.bayesian_layer1(x, True)
                h1 = self.activation(h1)
                
                # Second layer
                h2, _ = self.bayesian_layer2(h1, True)
                h2 = self.activation(h2)
                
                # Output layer
                logits, _ = self.bayesian_layer3(h2, True)
                probs = F.softmax(logits, dim=1)
                mc_samples.append(probs)
            
            # Stack samples and compute uncertainty
            mc_samples = torch.stack(mc_samples, dim=0)
            mean_probs = torch.mean(mc_samples, dim=0)
            uncertainty = torch.std(mc_samples, dim=0)
            
        return mean_probs, uncertainty
    def forward(self, x, sample=True, calculate_log_probs=False):
        # Initialize KL divergence
        kl = 0
        
        # First Bayesian layer
        x, kl1 = self.bayesian_layer1(x, sample)
        x = self.activation(x)
        kl += kl1
        
        # Second Bayesian layer
        x, kl2 = self.bayesian_layer2(x, sample)
        x = self.activation(x)
        kl += kl2
        
        # Output Bayesian layer
        logits, kl3 = self.bayesian_layer3(x, sample)
        kl += kl3
        
        # Compute expert probabilities using softmax
        expert_probs = F.softmax(logits, dim=1)
        
        # If sampling for uncertainty estimation
        if sample and calculate_log_probs:
            # Use the separate Monte Carlo sampling method instead of recursive call
            _, uncertainty = self.monte_carlo_sample(x.new_zeros((x.size(0), self.input_dim)))
            return expert_probs, kl, logits, uncertainty
        
        return expert_probs, kl, logits

    def predict_expert(self, text_embedding, threshold=0.7, return_uncertainty=False):
        """
        Predict which expert to use for a given text embedding.
        
        Args:
            text_embedding: CLIP text embedding tensor
            threshold: Confidence threshold for expert selection (default: 0.7)
            return_uncertainty: Whether to return uncertainty (default: False)
        
        Returns:
            Selected expert index/indices, probabilities, and uncertainty (optional)
        """
        with torch.no_grad():
            # Make multiple forward passes for Monte Carlo sampling
            mc_samples = []
            for _ in range(self.num_samples):
                probs, _, _ = self.forward(text_embedding, sample=True)
                mc_samples.append(probs)
            
            # Stack samples and compute mean and uncertainty
            mc_samples = torch.stack(mc_samples, dim=0)
            mean_probs = torch.mean(mc_samples, dim=0)
            uncertainty = torch.std(mc_samples, dim=0)
            
            # Select experts based on threshold
            selected_experts = []
            for i in range(self.num_experts):
                if mean_probs[0, i] > threshold:
                    selected_experts.append(i)
            
            # If no expert meets the threshold, choose the one with highest probability
            if not selected_experts:
                selected_experts = [torch.argmax(mean_probs, dim=1).item()]
            
            if return_uncertainty:
                return selected_experts, mean_probs, uncertainty
            else:
                return selected_experts, mean_probs


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts model for image generation from text prompts.
    Currently implements the gating network only, as per requirements.
    """
    def __init__(self, input_dim, hidden_dim, num_experts, clip_model_type="ViT-B/32"):
        """
        Initialize Mixture of Experts model.
        
        Args:
            input_dim: Dimension of the input (CLIP text embeddings)
            hidden_dim: Dimension of the hidden layer in the gating network
            num_experts: Number of experts in the model
            clip_model_type: CLIP model type for text embedding (default: "ViT-B/32")
        """
        super(MixtureOfExperts, self).__init__()
        
        # Store parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.clip_model_type = clip_model_type
        
        # Initialize gating network
        self.gating_network = BayesianMoEGatingNetwork(input_dim, hidden_dim, num_experts)
        
        # Expert networks would be implemented here in the future
        # Each expert would specialize in generating different types of images
        # For now, we just use placeholder experts
        self.expert_descriptions = [
            "General Expert for all categories"
            "Expert for natural landscapes",
            "Expert for portraits and people",
            "Expert for urban environments",
            "Expert for animals and wildlife",
            "Expert for abstract concepts and styles",
            "Expert for indoor scenes and objects",
            "Expert for transportation and vehicles",
            "Expert for weather and atmospheric conditions"
        ][:num_experts]  # Truncate to actual number of experts

    def forward(self, text_embedding):
        """
        Forward pass through the MoE model.
        
        Args:
            text_embedding: CLIP text embedding tensor
        
        Returns:
            Expert probabilities, selected experts, and uncertainty
        """
        # Get expert probabilities from gating network
        expert_probs, kl, _, uncertainty = self.gating_network(
            text_embedding, 
            sample=True, 
            calculate_log_probs=True
        )
        
        # Select experts based on probabilities
        selected_experts, mean_probs, expert_uncertainty = self.gating_network.predict_expert(
            text_embedding,
            return_uncertainty=True
        )
        
        return expert_probs, selected_experts, uncertainty

    def describe_selection(self, text_embedding):
        """
        Describe the expert selection for a given text embedding.
        
        Args:
            text_embedding: CLIP text embedding tensor
        
        Returns:
            Description of expert selection with probabilities and uncertainty
        """
        with torch.no_grad():
            # Get expert probabilities and uncertainty
            _, selected_experts, mean_probs, uncertainty = self.forward(text_embedding)
            
            # Create description
            description = "Expert selection:\n"
            for i, prob in enumerate(mean_probs[0]):
                description += f"- {self.expert_descriptions[i]}: {prob.item():.4f} "
                description += f"(uncertainty: {uncertainty[0, i].item():.4f})\n"
            
            description += "\nSelected experts:\n"
            for expert_idx in selected_experts:
                description += f"- {self.expert_descriptions[expert_idx]}\n"
            
            return description


# Training function for the Bayesian MoE Gating Network
def train_bayesian_moe_gating(model, train_dataloader, val_dataloader, epochs=10, lr=0.001, kl_weight=0.1, device='cuda'):
    """
    Train the Bayesian MoE Gating Network.
    
    Args:
        model: MixtureOfExperts model
        train_dataloader: Training data dataloader
        val_dataloader: Validation data dataloader
        epochs: Number of training epochs (default: 10)
        lr: Learning rate (default: 0.001)
        kl_weight: Weight for KL divergence term in the loss (default: 0.1)
        device: Device to train on (default: 'cuda')
    
    Returns:
        Trained model and training statistics
    """
    # Move model to device
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Initialize loss function (Cross-Entropy for expert classification)
    criterion = nn.CrossEntropyLoss()
    
    # Training statistics
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        
        for batch_idx, (images, text_embeddings) in enumerate(train_dataloader):
            # Move data to device
            text_embeddings = text_embeddings.to(device)
            
            # For training, we need expert labels (not available yet, so we'll use dummy labels)
            # In a real implementation, we would have true expert labels based on image categories
            # For now, we assign random expert labels for demonstration
            expert_labels = torch.randint(0, model.num_experts, (text_embeddings.size(0),)).to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            expert_probs, kl, logits = model.gating_network(text_embeddings)
            
            # Compute loss (cross-entropy + KL divergence)
            ce_loss = criterion(logits, expert_labels)
            loss = ce_loss + kl_weight * kl
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Update epoch loss
            epoch_loss += loss.item()
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_dataloader)}, Loss: {loss.item():.4f}")
        
        # Calculate average epoch loss
        epoch_loss /= len(train_dataloader)
        train_losses.append(epoch_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch_idx, (images, text_embeddings) in enumerate(val_dataloader):
                # Move data to device
                text_embeddings = text_embeddings.to(device)
                
                # For validation, we use dummy labels (see above)
                expert_labels = torch.randint(0, model.num_experts, (text_embeddings.size(0),)).to(device)
                
                # Forward pass
                expert_probs, kl, logits = model.gating_network(text_embeddings, sample=False)
                
                # Compute loss (cross-entropy only, no KL for validation)
                ce_loss = criterion(logits, expert_labels)
                loss = ce_loss
                
                # Update validation loss
                val_loss += loss.item()
        
        # Calculate average validation loss
        val_loss /= len(val_dataloader)
        val_losses.append(val_loss)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return model, {'train_losses': train_losses, 'val_losses': val_losses}


# Example usage
def main():
    # Load processed data
    from torch.utils.data import DataLoader
    from data_processing.data_processing_pipeline import ProcessedMSCOCODataset
    
    # Define constants
    CLIP_EMBEDDING_DIM = 512  # Dimension of CLIP text embeddings for ViT-B/32
    HIDDEN_DIM = 256
    NUM_EXPERTS = 5
    BATCH_SIZE = 32
    EPOCHS = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataset and dataloaders
    train_dataset = ProcessedMSCOCODataset(
        "data_processing/processed_data/mscoco_train_images.npy",
        "data_processing/processed_data/mscoco_train_text_embeddings.npy"
    )
    
    val_dataset = ProcessedMSCOCODataset(
        "data_processing/processed_data/mscoco_validation_images.npy",
        "data_processing/processed_data/mscoco_validation_text_embeddings.npy"
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Create and train MoE model
    moe_model = MixtureOfExperts(CLIP_EMBEDDING_DIM, HIDDEN_DIM, NUM_EXPERTS)
    
    trained_model, stats = train_bayesian_moe_gating(
        moe_model,
        train_dataloader,
        val_dataloader,
        epochs=EPOCHS,
        device=DEVICE
    )
    
    # Save trained model
    torch.save(trained_model.state_dict(), "bayesian_moe_model.pth")
    
    # Example of using the model for expert selection
    example_text_embedding = torch.randn(1, CLIP_EMBEDDING_DIM).to(DEVICE)
    expert_description = trained_model.describe_selection(example_text_embedding)
    print("\nExample expert selection:")
    print(expert_description)


if __name__ == "__main__":
    main()
