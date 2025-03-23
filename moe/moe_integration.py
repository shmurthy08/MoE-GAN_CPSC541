import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import DataLoader, Dataset, random_split
import pickle
from sklearn.cluster import KMeans
import clip
from PIL import Image
from tqdm import tqdm

# Import the MoE model
from moe_model import MixtureOfExperts, BayesianMoEGatingNetwork

# Import data processing pipeline
import sys
import torch.nn.functional as F
sys.path.append('../')  # Go up one directory to the parent folder
from data_processing.data_processing_pipeline import ProcessedMSCOCODataset, clip_model as data_clip_model

# Constants
CLIP_MODEL_TYPE = "ViT-B/32"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIP_EMBEDDING_DIM = 512  # Dimension of CLIP text embeddings for ViT-B/32
HIDDEN_DIM = 256
NUM_EXPERTS = 5
BATCH_SIZE = 32
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moe_results")
OS_makedirs = os.makedirs(OUTPUT_DIR, exist_ok=True)

class EnhancedMSCOCODataset(Dataset):
    """
    Enhanced MS-COCO dataset with cluster labels for expert training.
    """
    def __init__(self, images_file, text_embeddings_file, metadata_file=None, cluster_labels=None):
        """
        Initialize Enhanced MS-COCO dataset.
        
        Args:
            images_file: Path to the numpy file with images
            text_embeddings_file: Path to the numpy file with text embeddings
            metadata_file: Path to the pickle file with metadata (default: None)
            cluster_labels: Pre-computed cluster labels (default: None)
        """
        # Load data
        self.images = np.load(images_file)
        self.text_embeddings = np.load(text_embeddings_file)
        
        # Load metadata if provided
        self.metadata = None
        if metadata_file and os.path.exists(metadata_file):
            with open(metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
        
        # Use provided cluster labels or initialize with zeros
        self.cluster_labels = cluster_labels if cluster_labels is not None else np.zeros(len(self.images), dtype=int)
        
        # Validate data
        assert len(self.images) == len(self.text_embeddings), "Images and text embeddings count mismatch"
        if self.metadata:
            assert len(self.images) == len(self.metadata), "Images and metadata count mismatch"
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx])
        text_embedding = torch.from_numpy(self.text_embeddings[idx])
        cluster_label = torch.tensor(self.cluster_labels[idx], dtype=torch.long)
        
        # Return metadata if available
        return image, text_embedding, cluster_label

def cluster_text_embeddings(text_embeddings, n_clusters=NUM_EXPERTS, seed=42):
    """
    Cluster text embeddings to create expert labels.
    
    Args:
        text_embeddings: Text embeddings
        n_clusters: Number of clusters/experts (default: NUM_EXPERTS)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Cluster labels and KMeans model
    """
    print(f"Clustering text embeddings into {n_clusters} clusters...")
    
    # Initialize KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    
    # Fit KMeans
    cluster_labels = kmeans.fit_predict(text_embeddings)
    
    # Print cluster distribution
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    print("Cluster distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  Cluster {label}: {count} samples ({count/len(cluster_labels)*100:.2f}%)")
    
    return cluster_labels, kmeans

def analyze_clusters(text_embeddings, cluster_labels, metadata=None, clip_model=None, n_samples_per_cluster=5):
    """
    Analyze clusters to understand what each expert might specialize in.
    
    Args:
        text_embeddings: Text embeddings
        cluster_labels: Cluster labels
        metadata: Metadata with captions (default: None)
        clip_model: CLIP model for text similarity (default: None)
        n_samples_per_cluster: Number of samples to analyze per cluster (default: 5)
    
    Returns:
        List of cluster descriptions
    """
    print("Analyzing clusters...")
    
    # Get number of clusters
    n_clusters = len(np.unique(cluster_labels))
    
    # Initialize cluster descriptions
    cluster_descriptions = []
    
    for cluster_idx in range(n_clusters):
        print(f"\nCluster {cluster_idx}:")
        
        # Get indices of samples in this cluster
        cluster_indices = np.where(cluster_labels == cluster_idx)[0]
        
        # Sample a few examples
        sample_indices = np.random.choice(cluster_indices, min(n_samples_per_cluster, len(cluster_indices)), replace=False)
        
        # If metadata is available, print captions
        if metadata:
            print("Sample captions:")
            for i, idx in enumerate(sample_indices):
                caption = metadata[idx]['caption']
                print(f"  {i+1}. {caption}")
        
        # If CLIP model is available, find common themes
        if clip_model is not None:
            # Get cluster centroid
            centroid = np.mean(text_embeddings[cluster_indices], axis=0)
            
            # Define potential themes
            themes = [
                "landscape", "nature", "mountains", "ocean", "beach",
                "person", "portrait", "people", "crowd", "family",
                "urban", "city", "street", "building", "architecture",
                "animal", "wildlife", "pet", "dog", "cat",
                "food", "meal", "cooking", "kitchen", "dining",
                "abstract", "concept", "idea", "emotion", "feeling",
                "sports", "activity", "game", "playing", "exercise",
                "technology", "gadget", "device", "computer", "phone",
                "art", "painting", "sculpture", "creative", "design",
                "transportation", "vehicle", "car", "train", "airplane"
            ]
            
            # Get theme embeddings
            with torch.no_grad():
                theme_tokens = clip.tokenize(themes).to(DEVICE)
                theme_embeddings = clip_model.encode_text(theme_tokens).float().cpu().numpy()
            
            # Calculate similarities
            similarities = np.dot(theme_embeddings, centroid) / (np.linalg.norm(theme_embeddings, axis=1) * np.linalg.norm(centroid))
            
            # Get top themes
            top_indices = np.argsort(similarities)[::-1][:5]
            top_themes = [themes[i] for i in top_indices]
            
            print("Top themes:")
            for i, (theme, similarity) in enumerate(zip(top_themes, similarities[top_indices])):
                print(f"  {i+1}. {theme} (similarity: {similarity:.4f})")
            
            # Create cluster description
            description = f"Expert for {', '.join(top_themes[:3])}"
        else:
            description = f"Expert {cluster_idx}"
        
        cluster_descriptions.append(description)
    
    return cluster_descriptions

def compute_moe_balance_loss(expert_probs):
    """
    Compute MoE load balancing loss following Aurora/Switch Transformer.
    
    Args:
        expert_probs: Expert probabilities from the gating network
    
    Returns:
        Load balancing loss
    """
    # Calculate expert usage per batch (sum over batch dimension)
    expert_usage = expert_probs.sum(dim=0)
    
    # Normalize to get probability distribution
    expert_usage = expert_usage / expert_usage.sum()
    
    # Target uniform distribution
    num_experts = expert_usage.size(0)
    target_prob = torch.ones_like(expert_usage) / num_experts
    
    # KL divergence between actual and target distribution
    loss = F.kl_div(expert_usage.log(), target_prob, reduction='sum')
    
    return loss



def train_moe_with_clusters(train_dataset, val_dataset, epochs=10, lr=0.001, kl_weight=0.01, save_path=None):
    """
    Train the Bayesian MoE Gating Network using clustered data.
    
    Args:
        train_dataset: Training dataset with cluster labels
        val_dataset: Validation dataset with cluster labels
        epochs: Number of training epochs (default: 10)
        lr: Learning rate (default: 0.001)
        kl_weight: Weight for KL divergence term in the loss (default: 0.1)
        save_path: Path to save the trained model (default: None)
    
    Returns:
        Trained model and training statistics
    """
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize MoE model with cluster descriptions
    num_clusters = len(np.unique(train_dataset.cluster_labels))
    model = MixtureOfExperts(CLIP_EMBEDDING_DIM, HIDDEN_DIM, num_clusters)
    
    # Update expert descriptions if available
    if hasattr(train_dataset, 'cluster_descriptions'):
        model.expert_descriptions = train_dataset.cluster_descriptions
    
    # Move model to device
    model = model.to(DEVICE)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Initialize loss function (Cross-Entropy for expert classification)
    criterion = torch.nn.CrossEntropyLoss()
    
    # ADDED: Balance loss weight
    balance_weight = 0.01
    
    # Training statistics
    train_losses = []
    val_losses = []
    balance_metrics = []  # ADDED: Track balance metrics
    
    print(f"Starting training for {epochs} epochs...")

    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        epoch_ce_loss = 0     # ADDED: Track CE loss separately
        epoch_kl_loss = 0     # ADDED: Track KL loss separately
        epoch_balance_loss = 0  # ADDED: Track balance loss separately
        
        for batch_idx, (images, text_embeddings, cluster_labels) in enumerate(train_loader):
            text_embeddings = text_embeddings.to(DEVICE)
            cluster_labels = cluster_labels.to(DEVICE)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            expert_probs, kl, logits = model.gating_network(text_embeddings)
            
            # CHANGED: Compute loss components separately
            ce_loss = criterion(logits, cluster_labels)
            kl_loss = kl_weight * kl
            
            # ADDED: Compute expert balance loss
            balance_loss = compute_moe_balance_loss(expert_probs)
            
            # CHANGED: Total loss with all components
            loss = ce_loss + kl_loss + balance_weight * balance_loss
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Update epoch losses
            epoch_loss += loss.item()
            epoch_ce_loss += ce_loss.item()  # ADDED
            epoch_kl_loss += kl_loss.item()  # ADDED
            epoch_balance_loss += balance_loss.item()  # ADDED
            
            # CHANGED: Print more detailed progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f} ("
                      f"CE: {ce_loss.item():.4f}, "
                      f"KL: {kl_loss.item():.4f}, "
                      f"Balance: {(balance_weight * balance_loss).item():.4f})")
        
        # Calculate average epoch losses
        epoch_loss /= len(train_loader)
        epoch_ce_loss /= len(train_loader)  # ADDED
        epoch_kl_loss /= len(train_loader)  # ADDED
        epoch_balance_loss /= len(train_loader)  # ADDED
        
        train_losses.append(epoch_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_balance_metric = 0  # ADDED: Track balance during validation
        
        with torch.no_grad():
            for batch_idx, (images, text_embeddings, cluster_labels) in enumerate(val_loader):  # FIXED: val_loader instead of train_loader
                # Move data to device
                text_embeddings = text_embeddings.to(DEVICE)
                cluster_labels = cluster_labels.to(DEVICE)
                
                # Forward pass
                expert_probs, kl, logits = model.gating_network(text_embeddings, sample=False)
                
                # Compute validation loss (cross-entropy only, no KL for validation)
                ce_loss = criterion(logits, cluster_labels)
                loss = ce_loss
                
                # ADDED: Monitor expert balance (but don't add to loss)
                batch_balance = compute_moe_balance_loss(expert_probs).item()
                val_balance_metric += batch_balance
                
                # Update validation loss
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(logits, 1)
                val_total += cluster_labels.size(0)
                val_correct += (predicted == cluster_labels).sum().item()
        
        # Calculate average validation metrics
        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total
        val_balance_metric /= len(val_loader)  # ADDED: Average balance metric
        
        val_losses.append(val_loss)
        balance_metrics.append(val_balance_metric)  # ADDED: Track balance metrics
        
        # CHANGED: Print more detailed epoch summary
        print(f"Epoch {epoch+1}/{epochs}, "
            f"Train Loss: {epoch_loss:.4f} ("
            f"CE: {epoch_ce_loss:.4f}, "
            f"KL: {epoch_kl_loss:.4f}, "
            f"Balance: {epoch_balance_loss:.4f}), "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_accuracy:.4f}, "
            f"Expert Balance: {val_balance_metric:.4f}")
    
    # Save model if path is provided
    if save_path:
        print(f"Saving model to {save_path}")
        torch.save(model.state_dict(), save_path)
    
    # CHANGED: Return additional metrics
    return model, {
        'train_losses': train_losses, 
        'val_losses': val_losses,
        'balance_metrics': balance_metrics  # ADDED
    }

def visualize_training_results(training_stats, save_path=None):
    """
    Visualize training results.
    
    Args:
        training_stats: Dictionary with training statistics
        save_path: Path to save visualization (default: None)
    """
    # Create figure
    plt.figure(figsize=(10, 5))
    
    # Plot training and validation loss
    train_losses = training_stats['train_losses']
    val_losses = training_stats['val_losses']
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    # Add labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save or show the figure
    if save_path:
        plt.savefig(save_path)
        print(f"Training visualization saved to {save_path}")
    else:
        plt.show()

def main():
    """
    Main function to integrate MoE with the data processing pipeline.
    """
    parser = argparse.ArgumentParser(description='Integrate MoE with Data Pipeline')
    
    parser.add_argument('--train_images', type=str, default='../data_processing/processed_data/mscoco_train_images.npy',
                        help='Path to training images')
    parser.add_argument('--train_embeddings', type=str, default='../data_processing/processed_data/mscoco_train_text_embeddings.npy',
                        help='Path to training text embeddings')
    parser.add_argument('--train_metadata', type=str, default='../data_processing/processed_data/mscoco_train_metadata.pkl',
                        help='Path to training metadata')
    
    parser.add_argument('--val_images', type=str, default='../data_processing/processed_data/mscoco_validation_images.npy',
                        help='Path to validation images')
    parser.add_argument('--val_embeddings', type=str, default='../data_processing/processed_data/mscoco_validation_text_embeddings.npy',
                        help='Path to validation text embeddings')
    parser.add_argument('--val_metadata', type=str, default='../data_processing/processed_data/mscoco_validation_metadata.pkl',
                        help='Path to validation metadata')
    
    parser.add_argument('--n_experts', type=int, default=NUM_EXPERTS,
                        help=f'Number of experts (default: {NUM_EXPERTS})')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    parser.add_argument('--save_model', type=str, default='moe_results/bayesian_moe_model.pth',
                        help='Path to save the trained model')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load CLIP model for text similarity
    print(f"Loading CLIP model: {CLIP_MODEL_TYPE}")
    clip_model, _ = clip.load(CLIP_MODEL_TYPE, device=DEVICE)
    clip_model.eval()
    
    # Load training data
    print("Loading training data...")
    train_images = np.load(args.train_images)
    train_embeddings = np.load(args.train_embeddings)
    
    with open(args.train_metadata, 'rb') as f:
        train_metadata = pickle.load(f)
    
    # Cluster text embeddings
    cluster_labels, kmeans_model = cluster_text_embeddings(train_embeddings, n_clusters=args.n_experts)
    
    # Analyze clusters
    cluster_descriptions = analyze_clusters(
        train_embeddings, 
        cluster_labels, 
        metadata=train_metadata, 
        clip_model=clip_model
    )
    
    # Create enhanced datasets
    train_dataset = EnhancedMSCOCODataset(
        args.train_images,
        args.train_embeddings,
        args.train_metadata,
        cluster_labels
    )
    train_dataset.cluster_descriptions = cluster_descriptions
    
    # Load validation data if available
    if os.path.exists(args.val_images) and os.path.exists(args.val_embeddings):
        print("Loading validation data...")
        val_embeddings = np.load(args.val_embeddings)
        
        # Use KMeans to predict clusters for validation data
        val_cluster_labels = kmeans_model.predict(val_embeddings)
        
        val_dataset = EnhancedMSCOCODataset(
            args.val_images,
            args.val_embeddings,
            args.val_metadata if os.path.exists(args.val_metadata) else None,
            val_cluster_labels
        )
        val_dataset.cluster_descriptions = cluster_descriptions
    else:
        print("Validation data not found. Splitting training data...")
        # Split training data for validation
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # Train MoE model
    model, training_stats = train_moe_with_clusters(
        train_dataset,
        val_dataset,
        epochs=args.epochs,
        lr=args.lr,
        kl_weight=0.00001,
        save_path=args.save_model
    )
    
    # Visualize training results
    visualize_training_results(training_stats, save_path=os.path.join(OUTPUT_DIR, 'training_results.png'))
    
    print("MoE integration complete!")
    
    # Save cluster descriptions
    cluster_desc_path = os.path.join(OUTPUT_DIR, 'cluster_descriptions.txt')
    with open(cluster_desc_path, 'w') as f:
        for i, desc in enumerate(cluster_descriptions):
            f.write(f"Cluster {i}: {desc}\n")
    
    print(f"Cluster descriptions saved to {cluster_desc_path}")

if __name__ == "__main__":
    main()
