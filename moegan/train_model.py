# train_aurora_gan.py
import os
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
# !! Make sure t2i_moe_gan has the updated train_aurora_gan !!
from t2i_moe_gan import train_aurora_gan, AuroraGenerator, AuroraDiscriminator

# Import data processing utilities
import sys
# Adjust path as needed for your project structure
script_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# Make sure the path is correct for your setup
try:
    from data_processing.data_processing_pipeline import ProcessedMSCOCODataset
except ImportError:
    print("Error: Could not import ProcessedMSCOCODataset.")
    print("Please ensure 'data_processing' directory is in the Python path or adjust sys.path.")
    sys.exit(1)


# Constants
BATCH_SIZE = 2 # Adjust based on GPU memory
NUM_EPOCHS = 10
LEARNING_RATE = 0.0002
BETA1 = 0.5
BETA2 = 0.999
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
R1_GAMMA = 10.0 # Default R1 gamma
CLIP_WEIGHT_64 = 0.1 # Default CLIP weight for 64x64
CLIP_WEIGHT_32 = 0.05 # Default CLIP weight for 32x32
KL_WEIGHT = 0.001 # Default Bayesian KL weight
BALANCE_WEIGHT = 0.01 # Default MoE Balance weight

def main():
    parser = argparse.ArgumentParser(description='Train Aurora GAN-MoE')

    # Data Args
    parser.add_argument('--data_dir', type=str, default='../data_processing/processed_data',
                        help='Directory containing processed data (train/val images and embeddings)')
    parser.add_argument('--train_images', type=str, default='mscoco_train_images.npy',
                        help='Filename for training images numpy array')
    parser.add_argument('--train_embeddings', type=str, default='mscoco_train_text_embeddings.npy',
                        help='Filename for training text embeddings numpy array')
    parser.add_argument('--val_images', type=str, default='mscoco_validation_images.npy',
                        help='Filename for validation images numpy array')
    parser.add_argument('--val_embeddings', type=str, default='mscoco_validation_text_embeddings.npy',
                        help='Filename for validation text embeddings numpy array')

    # Training Args
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--beta1', type=float, default=BETA1,
                        help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=BETA2,
                        help='Beta2 for Adam optimizer')
    parser.add_argument('--save_dir', type=str, default='./aurora_checkpoints_v2', # Use a new dir
                        help='Directory to save checkpoints')
    parser.add_argument('--log_interval', type=int, default=50, # Increase log interval maybe
                        help='Logging interval (steps)')
    parser.add_argument('--save_interval', type=int, default=1000,
                        help='Checkpoint saving interval (steps)')

    # Loss Weights / Hyperparameters
    parser.add_argument('--r1_gamma', type=float, default=R1_GAMMA, # <-- Added R1
                        help='Weight for R1 regularization')
    parser.add_argument('--clip_weight_64', type=float, default=CLIP_WEIGHT_64, # <-- Added Clip 64
                       help='Weight for CLIP perceptual loss (64x64 output)')
    parser.add_argument('--clip_weight_32', type=float, default=CLIP_WEIGHT_32, # <-- Added Clip 32
                       help='Weight for CLIP perceptual loss (32x32 output)')
    parser.add_argument('--kl_weight', type=float, default=KL_WEIGHT, # <-- Added KL weight
                        help='Weight for KL divergence loss of Bayesian router')
    parser.add_argument('--balance_weight', type=float, default=BALANCE_WEIGHT, # <-- Added Balance weight
                        help='Weight for MoE load balancing loss')

    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    print(f"Training args: {args}")

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # --- Load datasets ---
    train_img_path = os.path.join(args.data_dir, args.train_images)
    train_emb_path = os.path.join(args.data_dir, args.train_embeddings)
    val_img_path = os.path.join(args.data_dir, args.val_images)
    val_emb_path = os.path.join(args.data_dir, args.val_embeddings)

    # Check if files exist
    if not os.path.exists(train_img_path) or not os.path.exists(train_emb_path):
        print(f"Error: Training data not found at {train_img_path} or {train_emb_path}")
        sys.exit(1)
    if not os.path.exists(val_img_path) or not os.path.exists(val_emb_path):
        print(f"Warning: Validation data not found at {val_img_path} or {val_emb_path}. Skipping validation.")
        val_dataset = None
    else:
        val_dataset = ProcessedMSCOCODataset(val_img_path, val_emb_path)


    train_dataset = ProcessedMSCOCODataset(train_img_path, train_emb_path)

    print(f"Loaded training dataset with {len(train_dataset)} samples.")
    if val_dataset:
        print(f"Loaded validation dataset with {len(val_dataset)} samples.")

    # --- Create dataloaders ---
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=max(1, os.cpu_count() // 2), pin_memory=True, drop_last=True # drop_last can help stability
    )

    if val_dataset:
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=max(1, os.cpu_count() // 2), pin_memory=True
        )
    else:
        val_dataloader = None

    # Train Aurora GAN
    generator, discriminator = train_aurora_gan(
        train_dataloader, val_dataloader=val_dataloader,
        num_epochs=args.epochs, lr=args.lr, beta1=args.beta1, beta2=args.beta2,
        r1_gamma=args.r1_gamma,              # Pass R1 gamma
        clip_weight_64=args.clip_weight_64,  # Pass Clip 64 weight
        clip_weight_32=args.clip_weight_32,  # Pass Clip 32 weight
        kl_weight=args.kl_weight,            # Pass KL weight
        balance_weight=args.balance_weight,  # Pass Balance weight
        device=DEVICE, save_dir=args.save_dir,
        log_interval=args.log_interval, save_interval=args.save_interval
    )

    print("Training complete.")

if __name__ == '__main__':
    main()