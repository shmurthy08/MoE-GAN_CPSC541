# train_aurora_gan.py
import os
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from t2i_moe_gan import train_aurora_gan, AuroraGenerator, AuroraDiscriminator

# Import data processing utilities
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_processing.data_processing_pipeline import ProcessedMSCOCODataset

# Constants
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.0002
BETA1 = 0.5
BETA2 = 0.999
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser(description='Train Aurora GAN-MoE')
    
    parser.add_argument('--data_dir', type=str, default='../data_processing/processed_data',
                        help='Directory containing processed data')
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
    parser.add_argument('--save_dir', type=str, default='./aurora_checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Logging interval')
    parser.add_argument('--save_interval', type=int, default=1000,
                        help='Checkpoint saving interval')
    parser.add_argument('--clip_weight', type=float, default=0.1,
                   help='Weight for CLIP perceptual loss')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load datasets
    train_dataset = ProcessedMSCOCODataset(
        os.path.join(args.data_dir, 'mscoco_train_images.npy'),
        os.path.join(args.data_dir, 'mscoco_train_text_embeddings.npy')
    )
    
    val_dataset = ProcessedMSCOCODataset(
        os.path.join(args.data_dir, 'mscoco_validation_images.npy'),
        os.path.join(args.data_dir, 'mscoco_validation_text_embeddings.npy')
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Train Aurora GAN
    generator, discriminator = train_aurora_gan(
        train_dataloader, val_dataloader=val_dataloader,
        num_epochs=args.epochs, lr=args.lr, beta1=args.beta1, beta2=args.beta2,
        device=DEVICE, save_dir=args.save_dir,
        log_interval=args.log_interval, save_interval=args.save_interval
    )
    
    print("Training complete.")
if __name__ == '__main__':
    main()