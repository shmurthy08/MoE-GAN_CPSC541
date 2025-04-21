# sagemaker_train.py
import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import boto3
import sys

# Import GAN components
from t2i_moe_gan import train_aurora_gan, AuroraGenerator, AuroraDiscriminator

# SageMaker paths
TRAINING_PATH = '/opt/ml/input/data/training'
MODEL_PATH = '/opt/ml/model'
OUTPUT_PATH = '/opt/ml/output'
PARAM_PATH = '/opt/ml/input/config/hyperparameters.json'

# Import from data processing
try:
    from data_processing.data_processing_pipeline import ProcessedMSCOCODataset
except ImportError:
    print("Warning: Could not import ProcessedMSCOCODataset directly")
    # Try alternative import
    sys.path.append(os.path.abspath('..'))
    try:
        from data_processing.data_processing_pipeline import ProcessedMSCOCODataset
    except ImportError:
        print("Critical error: Cannot import data processing module")
        sys.exit(1)

def parse_sagemaker_parameters():
    """Parse hyperparameters from SageMaker-provided JSON file"""
    with open(PARAM_PATH, 'r') as f:
        hyperparameters = json.load(f)
        
    # Convert string parameters to appropriate types
    params = {}
    for key, value in hyperparameters.items():
        # Convert numerical parameters to the right type
        if key in ['batch_size', 'epochs']:
            params[key] = int(value)
        elif key in ['learning_rate', 'beta1', 'beta2', 'r1_gamma', 
                    'clip_weight_64', 'clip_weight_32', 'kl_weight', 'balance_weight']:
            params[key] = float(value)
        else:
            params[key] = value
            
    return params

def download_from_s3(bucket, prefix, local_dir):
    """Download preprocessed data from S3 to training instance"""
    os.makedirs(local_dir, exist_ok=True)
    s3 = boto3.client('s3')
    
    print(f"Downloading data from s3://{bucket}/{prefix} to {local_dir}")
    
    # List objects in bucket/prefix
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    
    if 'Contents' in response:
        for obj in response['Contents']:
            key = obj['Key']
            filename = os.path.basename(key)
            local_file = os.path.join(local_dir, filename)
            
            # Download file
            print(f"Downloading {key} to {local_file}")
            s3.download_file(bucket, key, local_file)
    else:
        print(f"No objects found in s3://{bucket}/{prefix}")

# Custom training function with metric reporting for SageMaker
def custom_train_aurora_gan(train_dataloader, val_dataloader=None, **kwargs):
    """
    Wrapper around train_aurora_gan that reports metrics for SageMaker hyperparameter tuning
    """
    # Start the training process
    generator, discriminator = train_aurora_gan(
        train_dataloader, 
        val_dataloader=val_dataloader,
        **kwargs
    )
    
    return generator, discriminator

def main():
    """Main training function for SageMaker environment"""
    print("Starting MoE-GAN training in SageMaker environment")
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Parse hyperparameters
    params = parse_sagemaker_parameters()
    print(f"Training with parameters: {params}")
    
    # Create save directory for checkpoints
    save_dir = os.path.join(MODEL_PATH, 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    
    # Get S3 data location from environment variables
    s3_bucket = os.environ.get('S3_DATA_BUCKET')
    s3_prefix = os.environ.get('S3_DATA_PREFIX', 'mscoco_processed')
    
    # Download data from S3 if needed
    if s3_bucket:
        data_dir = os.path.join(TRAINING_PATH, 'data')
        download_from_s3(s3_bucket, s3_prefix, data_dir)
    else:
        data_dir = TRAINING_PATH
    
    # Locate data files
    train_img_path = os.path.join(data_dir, 'mscoco_train_images.npy')
    train_emb_path = os.path.join(data_dir, 'mscoco_train_text_embeddings.npy')
    
    # Check for validation data
    val_img_path = os.path.join(data_dir, 'mscoco_validation_images.npy')
    val_emb_path = os.path.join(data_dir, 'mscoco_validation_text_embeddings.npy')
    
    has_validation = os.path.exists(val_img_path) and os.path.exists(val_emb_path)
    
    # Load training dataset
    print(f"Loading training data from {train_img_path} and {train_emb_path}")
    train_dataset = ProcessedMSCOCODataset(train_img_path, train_emb_path)
    print(f"Loaded {len(train_dataset)} training samples")
    
    # Setup CloudWatch metrics
    cloudwatch = boto3.client('cloudwatch', region_name=os.environ.get('AWS_REGION', 'us-west-2'))
    
    def log_metric(name, value, unit='None'):
        """Log metrics to CloudWatch and standard output for SageMaker to parse"""
        try:
            # Log to CloudWatch
            cloudwatch.put_metric_data(
                Namespace='MoEGAN',
                MetricData=[
                    {
                        'MetricName': name,
                        'Value': value,
                        'Unit': unit
                    }
                ]
            )
            
            # Log in a format SageMaker can parse for hyperparameter tuning
            # Format: "[METRIC] metric_name: value"
            print(f"[METRIC] {name}: {value}")
            
        except Exception as e:
            print(f"Failed to log metric {name}: {e}")
    
    # Create training dataloader
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=params.get('batch_size', 32),
        shuffle=True, 
        num_workers=os.cpu_count(), 
        pin_memory=True,
        drop_last=True
    )
    
    # Load validation dataset if available
    if has_validation:
        print(f"Loading validation data from {val_img_path} and {val_emb_path}")
        val_dataset = ProcessedMSCOCODataset(val_img_path, val_emb_path)
        print(f"Loaded {len(val_dataset)} validation samples")
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=params.get('batch_size', 32),
            shuffle=False,
            num_workers=max(1, os.cpu_count() // 2),
            pin_memory=True
        )
    else:
        val_dataloader = None
        print("No validation data found, skipping validation")
    
    # Define a custom callback function to report metrics during training
    def metric_callback(epoch, metrics):
        """Report metrics to SageMaker for hyperparameter tuning"""
        # Log key metrics for this epoch
        for metric_name, value in metrics.items():
            # Convert to string for consistent naming and prefix with 'val_' for validation metrics
            metric_name_str = f"val_{metric_name}" if 'val' in metric_name else metric_name
            log_metric(metric_name_str, value)
            
        # Return True to continue training
        return True
        
    # Train model with the metric reporting callback
    generator, discriminator = train_aurora_gan(
        train_dataloader, 
        val_dataloader=val_dataloader,
        num_epochs=params.get('epochs', 10),
        lr=params.get('learning_rate', 0.0002),
        beta1=params.get('beta1', 0.5),
        beta2=params.get('beta2', 0.999),
        r1_gamma=params.get('r1_gamma', 10.0),
        clip_weight_64=params.get('clip_weight_64', 0.1),
        clip_weight_32=params.get('clip_weight_32', 0.05),
        kl_weight=params.get('kl_weight', 0.001),
        balance_weight=params.get('balance_weight', 0.01),
        device=device,
        save_dir=save_dir,
        log_interval=50,
        save_interval=500,
        metric_callback=metric_callback  # Add the callback for metric reporting
    )
    
    # Save final model 
    final_model_path = os.path.join(MODEL_PATH, 'aurora_model_final.pt')
    torch.save({
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
    }, final_model_path)
    
    print(f"Training complete. Final model saved to {final_model_path}")

if __name__ == '__main__':
    main()