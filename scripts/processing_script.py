# scripts/processing_script.py
import os
import argparse
import boto3
import json
import sys
from pathlib import Path

# Add parent directory to Python path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import data processing pipeline
from data_processing.data_processing_pipeline import run_pipeline

def parse_args():
    parser = argparse.ArgumentParser()
    
    # SageMaker Processing passes hyperparameters as command-line arguments
    parser.add_argument('--max-samples', type=int, default=-1, 
                        help='Maximum number of samples to process (default: all)')
    parser.add_argument('--no-augmentation', action='store_true', 
                        help='Disable data augmentation')
    parser.add_argument('--aug-factor', type=int, default=2, 
                        help='Augmentation factor (default: 2)')
    parser.add_argument('--s3-output-path', type=str, required=True,
                        help='S3 path to save processed data')
    
    # SageMaker Processing container directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/processing/output'))
    
    return parser.parse_args()

def upload_to_s3(local_dir, s3_output_path):
    """Upload processed data from local directory to S3 with batch processing for large datasets"""
    s3 = boto3.client('s3')
    
    # Parse S3 bucket and prefix from s3_output_path
    if s3_output_path.startswith('s3://'):
        s3_output_path = s3_output_path[5:]
    bucket, prefix = s3_output_path.split('/', 1)
    
    # Ensure prefix ends with '/'
    if not prefix.endswith('/'):
        prefix += '/'
    
    print(f"Uploading data from {local_dir} to s3://{bucket}/{prefix}")
    
    # First, upload all .npy and .pkl files
    npy_pkl_files = []
    for filename in os.listdir(local_dir):
        if filename.endswith('.npy') or filename.endswith('.pkl'):
            npy_pkl_files.append(filename)
    
    # Upload in batches
    batch_size = 10
    for i in range(0, len(npy_pkl_files), batch_size):
        batch = npy_pkl_files[i:i+batch_size]
        print(f"Uploading batch {i//batch_size + 1}/{(len(npy_pkl_files) + batch_size - 1)//batch_size} of .npy/.pkl files")
        
        for filename in batch:
            local_path = os.path.join(local_dir, filename)
            s3_key = f"{prefix}{filename}"
            
            print(f"Uploading {local_path} to s3://{bucket}/{s3_key}")
            try:
                s3.upload_file(local_path, bucket, s3_key)
                print(f"Successfully uploaded {filename}")
            except Exception as e:
                print(f"ERROR uploading {filename}: {str(e)}")
    
    # Upload metadata files (smaller files)
    print("Uploading metadata files")
    metadata_files = []
    for filename in os.listdir(local_dir):
        if filename.endswith('.json') or filename.endswith('.png'):
            metadata_files.append(filename)
    
    for filename in metadata_files:
        local_path = os.path.join(local_dir, filename)
        s3_key = f"{prefix}metadata/{filename}"
        
        print(f"Uploading metadata {local_path} to s3://{bucket}/{s3_key}")
        try:
            s3.upload_file(local_path, bucket, s3_key)
            print(f"Successfully uploaded metadata {filename}")
        except Exception as e:
            print(f"ERROR uploading metadata {filename}: {str(e)}")
    
    print("Upload complete")

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set max_samples to None if -1
    max_samples = None if args.max_samples == -1 else args.max_samples
    
    print(f"Starting data processing pipeline with settings:")
    print(f"  max_samples: {max_samples if max_samples is not None else 'ALL'}")
    print(f"  augmentation: {not args.no_augmentation}")
    print(f"  augmentation factor: {args.aug_factor}")
    print(f"  output directory: {args.output_data_dir}")
    
    # Install additional dependencies for data processing
    print("Installing required dependencies...")
    os.system("pip install torch==1.7.1 torchvision==0.8.2 clip-by-openai fiftyone matplotlib tqdm pandas pillow numpy ftfy regex")
    
    # Batch processing for large datasets
    if max_samples is None:
        print("Processing full MS-COCO dataset in batches")
        batch_size = 50000  # Process 50k images per batch
        
        # Process in batches
        processed_stats = {}
        batch_num = 1
        
        while True:
            start_idx = (batch_num - 1) * batch_size
            end_idx = batch_num * batch_size
            print(f"Processing batch {batch_num}: images {start_idx}-{end_idx}")
            
            # Run the pipeline on this batch
            batch_stats = run_pipeline(
                max_samples=batch_size,
                create_augmentations_flag=not args.no_augmentation,
                augmentation_factor=args.aug_factor,
            )
            
            # Upload this batch to S3
            print(f"Uploading to S3")
            batch_prefix = f"{args.s3_output_path}/batch_{batch_num}"
            upload_to_s3(os.path.dirname(os.path.abspath(__file__)) + "/processed_data", batch_prefix)
            
            # Accumulate stats
            for key, value in batch_stats.items():
                if key in processed_stats:
                    processed_stats[key] += value
                else:
                    processed_stats[key] = value
            
            # Check if we've processed all images
            if batch_stats.get('train_samples', 0) < batch_size:
                print(f"Reached end of dataset after processing {processed_stats.get('train_samples', 0)} images")
                break
                
            batch_num += 1
    else:
        # Run the pipeline for specified number of samples
        processed_stats = run_pipeline(
            max_samples=max_samples,
            create_augmentations_flag=not args.no_augmentation,
            augmentation_factor=args.aug_factor
        )
        
        # Upload to S3
        upload_to_s3(os.path.dirname(os.path.abspath(__file__)) + "/processed_data", args.s3_output_path)
    
    # Save processing stats
    with open(os.path.join(output_dir, 'processing_stats.json'), 'w') as f:
        json.dump(processed_stats, f, indent=2)
    
    print("Data processing complete!")

if __name__ == "__main__":
    main()