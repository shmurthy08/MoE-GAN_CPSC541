# sagemaker_train.py
import os
import json
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import boto3
import sys
import traceback

# Debug import issues by logging environment information
print("Current working directory: {}".format(os.getcwd()))
print("Python path at start: {}".format(sys.path))
print("Directory contents: {}".format(os.listdir('.')))
if os.path.exists('..'):
    print("Parent directory contents: {}".format(os.listdir('..')))
if os.path.exists('/app'):
    print("/app directory contents: {}".format(os.listdir('/app')))

# Simple dataset class - no dependency on data_processing
class SimpleDataset(Dataset):
    """Simple dataset class that just loads preprocessed .npy files"""
    
    # Add this function to SimpleDataset in sagemaker_train.py
    def __init__(self, images_file, text_embeddings_file, use_percentage=1.0):
        """
        Args:
            images_file (string): Path to the numpy file with images
            text_embeddings_file (string): Path to the numpy file with text embeddings
            use_percentage (float): Percentage of data to use (0.0-1.0)
        """
        print("Loading images from: {}".format(images_file))
        print("Loading text embeddings from: {}".format(text_embeddings_file))
        
        try:
            # Load full arrays
            all_images = np.load(images_file)
            all_text_embeddings = np.load(text_embeddings_file)
            
            # Calculate how many samples to keep
            total_samples = len(all_images)
            keep_samples = int(total_samples * use_percentage)
            
            # Use only a subset of the data
            self.images = all_images[:keep_samples]
            self.text_embeddings = all_text_embeddings[:keep_samples]
            
            print("Loaded images shape: {} (using {}% - {} samples)".format(
                self.images.shape, use_percentage*100, keep_samples))
            print("Loaded text embeddings shape: {}".format(self.text_embeddings.shape))
        except Exception as e:
            print("Error loading data files: {}".format(str(e)))
            traceback.print_exc()
            raise
        
        # Verify dimensions match
        assert len(self.images) == len(self.text_embeddings), "Images count ({}) and text embeddings count ({}) mismatch".format(
            len(self.images), len(self.text_embeddings))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx])
        text_embedding = torch.from_numpy(self.text_embeddings[idx])
        return image, text_embedding

# Import GAN components
try:
    from t2i_moe_gan import train_aurora_gan, AuroraGenerator, AuroraDiscriminator
    print("✓ Successfully imported GAN components")
except ImportError as e:
    print("Error importing GAN components: {}".format(str(e)))
    traceback.print_exc()
    sys.exit(1)

# SageMaker paths
TRAINING_PATH = '/opt/ml/input/data/training'
MODEL_PATH = '/opt/ml/model'
OUTPUT_PATH = '/opt/ml/output'
PARAM_PATH = '/opt/ml/input/config/hyperparameters.json'

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
    
    print("Downloading data from s3://{}/{} to {}".format(bucket, prefix, local_dir))
    
    # List objects in bucket/prefix
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    
    if 'Contents' in response:
        for obj in response['Contents']:
            key = obj['Key']
            filename = os.path.basename(key)
            local_file = os.path.join(local_dir, filename)
            
            # Download file
            print("Downloading {} to {}".format(key, local_file))
            s3.download_file(bucket, key, local_file)
    else:
        print("No objects found in s3://{}/{}".format(bucket, prefix))

def main():
    """Main training function for SageMaker environment"""
    print("Starting MoE-GAN training in SageMaker environment")
    
    try:
        # Memory optimization settings
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # Reduced from 512
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        # Additional memory optimizations
        torch.cuda.empty_cache()  # Clear CUDA cache before starting
        
        # Enable gradient checkpointing to save memory at cost of some speed
        os.environ['PYTORCH_ENABLE_GRAD_CHECKPOINT'] = '1'
        
        # Enable automatic mixed precision training
        use_amp = True
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        # Set memory efficient attention
        os.environ['PYTORCH_ATTENTION_IMPLEMENTATION'] = 'mem_efficient'
        
        # Use pinned memory for faster transfers but less host memory
        pin_memory = torch.cuda.is_available()
        
        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(device))
        
        # Set optimal tensor layout
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for faster training
            
        # Parse hyperparameters
        params = parse_sagemaker_parameters()
        print("Training with parameters: {}".format(params))
        
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
        
        # Check if files exist
        if not os.path.exists(train_img_path):
            raise FileNotFoundError("Training images file not found: {}".format(train_img_path))
        if not os.path.exists(train_emb_path):
            raise FileNotFoundError("Training embeddings file not found: {}".format(train_emb_path))
        
        # Check for validation data
        val_img_path = os.path.join(data_dir, 'mscoco_validation_images.npy')
        val_emb_path = os.path.join(data_dir, 'mscoco_validation_text_embeddings.npy')
        
        has_validation = os.path.exists(val_img_path) and os.path.exists(val_emb_path)
        
        # Load training dataset (USING ONLY 50% OF DATA FOR FASTER TRAINING)
        print("Loading training data from {} and {}".format(train_img_path, train_emb_path))
        train_dataset = SimpleDataset(train_img_path, train_emb_path, use_percentage=0.2)
        print("Loaded {} training samples (20% of total)".format(len(train_dataset)))
        
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
                print("[METRIC] {}: {}".format(name, value))
                
            except Exception as e:
                print("Failed to log metric {}: {}".format(name, e))
        
        # Create training dataloader with batch size 16
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=params.get('batch_size', 16),  # Changed to 16
            shuffle=True, 
            num_workers=max(1, os.cpu_count() // 2 if os.cpu_count() is not None else 1), 
            pin_memory=True,
            drop_last=True
        )

        # Leave validation dataset at 100%
        if has_validation:
            print("Loading validation data from {} and {}".format(val_img_path, val_emb_path))
            val_dataset = SimpleDataset(val_img_path, val_emb_path, use_percentage=1.0)
            print("Loaded {} validation samples".format(len(val_dataset)))
            
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=params.get('batch_size', 16),  # Changed to 16 for consistency
                shuffle=False,
                num_workers=max(1, os.cpu_count() // 2 if os.cpu_count() is not None else 1),
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
                metric_name_str = "val_{}".format(metric_name) if 'val' in metric_name else metric_name
                log_metric(metric_name_str, value)
                
            # Return True to continue training
            return True
            
        # Train model with the metric reporting callback
        generator, discriminator = train_aurora_gan(
            train_dataloader, 
            val_dataloader=val_dataloader,
            num_epochs=params.get('epochs', 5),
            lr=params.get('learning_rate', 0.0002),
            beta1=params.get('beta1', 0.5),
            beta2=params.get('beta2', 0.999),
            r1_gamma=params.get('r1_gamma', 10.0),
            clip_weight_64=params.get('clip_weight_64', 0.1),
            clip_weight_32=params.get('clip_weight_32', 0.05),
            kl_weight=params.get('kl_weight', 0.001),
            kl_annealing_epochs=params.get('kl_annealing_epochs', 1),
            balance_weight=params.get('balance_weight', 0.01),
            device=device,
            save_dir=save_dir,
            log_interval=100,
            save_interval=500,
            metric_callback=metric_callback,
            gradient_accumulation_steps=9,
            checkpoint_activation=True,
            batch_memory_limit=10.0
        )
        
        # Save final model 
        ### UNCOMMENT WHEN TRAINING FINAL MODEL ###
        # final_model_path = os.path.join(MODEL_PATH, 'aurora_model_final.pt')
        # torch.save({
        #     'generator': generator.state_dict(),
        #     'discriminator': discriminator.state_dict(),
        # }, final_model_path)
        
        print("Training complete")
    
    except Exception as e:
        print("\n" + "="*50)
        print("CRITICAL ERROR")
        print("="*50)
        print("Error type: {}".format(type(e).__name__))
        print("Error message: {}".format(str(e)))
        print("\nDetailed traceback:")
        traceback.print_exc()
        
        # Additional system information for debugging
        print("\nSystem information:")
        print("Python version: {}".format(sys.version))
        print("PyTorch version: {}".format(torch.__version__))
        print("NumPy version: {}".format(np.__version__))
        print("Available memory: {}".format(os.popen('free -h').read()))
        print("CPU info: {}".format(os.popen('cat /proc/cpuinfo | grep "model name" | head -1').read().strip()))
        if torch.cuda.is_available():
            print("GPU info: {}".format(torch.cuda.get_device_name(0)))
            print("GPU memory: {:.2f} GB".format(torch.cuda.get_device_properties(0).total_memory / 1e9))
        
        print("="*50)
        sys.exit(1)

if __name__ == '__main__':
    main()