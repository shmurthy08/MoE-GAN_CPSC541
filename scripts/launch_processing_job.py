# scripts/launch_processing_job.py
import boto3
import argparse
import time
import os
import sys
from datetime import datetime

def launch_processing_job(
    role_arn,
    bucket,
    prefix='mscoco_processed',
    max_samples=-1,  # -1 processes all images
    no_augmentation=False,
    aug_factor=2,
    instance_type='ml.m5.4xlarge',
    volume_size=500,
    wait=False
):
    sagemaker = boto3.client('sagemaker')
    
    # Generate a unique job name
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    job_name = f"moe-gan-data-processing-{timestamp}"
    
    # Configure processing output
    s3_output_path = f"s3://{bucket}/{prefix}"
    outputs = [
        {
            "OutputName": "processed_data",
            "S3Output": {
                "S3Uri": s3_output_path,
                "LocalPath": "/opt/ml/processing/output",
                "S3UploadMode": "EndOfJob"
            }
        }
    ]
    
    # Create a processing script that will download and run your code
    processing_script = """
import os
import sys
import boto3
import subprocess
import glob

def main():
    print("Starting MS-COCO data processing pipeline...")
    
    # Install required dependencies
    print("Installing dependencies...")
    os.system("pip install torch==1.7.1 torchvision==0.8.2 clip-by-openai fiftyone matplotlib tqdm pandas pillow numpy ftfy regex")
    
    # Create output directory
    os.makedirs("/opt/ml/processing/output", exist_ok=True)
    
    # Clone the repository to access the processing code
    print("Cloning repository to access data processing code...")
    repo_url = "https://github.com/shmurthy08/MoE-GAN_CPSC541"
    if not repo_url:
        print("Error: Repository URL not provided")
        return
        
    os.system(f"git clone {repo_url} /tmp/repo")
    
    # Copy the data_processing directory to the current working directory
    os.system("cp -r /tmp/repo/data_processing .")
    
    # Run the data processing pipeline
    print("Running data processing pipeline...")
    max_samples = os.environ.get('MAX_SAMPLES', '-1')
    no_augmentation = os.environ.get('NO_AUGMENTATION', 'false')
    aug_factor = os.environ.get('AUG_FACTOR', '2')
    
    cmd = f"cd data_processing && python data_processing_pipeline.py --max_samples {max_samples}"
    if no_augmentation.lower() == 'true':
        cmd += " --no_augmentation"
    cmd += f" --aug_factor {aug_factor}"
    
    print(f"Running command: {cmd}")
    os.system(cmd)
    
    # Copy the processed files to the output directory
    print("Copying processed files to output directory...")
    os.system("cp -r data_processing/processed_data/* /opt/ml/processing/output/")
    
    # List the output files
    print("Output directory contents:")
    os.system("ls -la /opt/ml/processing/output/")
    
    # Count the number of .npy and .pkl files
    npy_files = glob.glob("/opt/ml/processing/output/*.npy")
    pkl_files = glob.glob("/opt/ml/processing/output/*.pkl")
    print(f"Generated {len(npy_files)} .npy files and {len(pkl_files)} .pkl files")
    
    print("Processing job completed successfully!")

if __name__ == "__main__":
    main()
"""
    
    # Write the script to a local file
    processing_script_path = "processing_script.py"
    with open(processing_script_path, "w") as f:
        f.write(processing_script)
    
    # Upload the script to S3
    script_s3_key = f"code/{job_name}/processing_script.py"
    s3 = boto3.client('s3')
    s3.upload_file(processing_script_path, bucket, script_s3_key)
    
    # Create processing job
    response = sagemaker.create_processing_job(
        ProcessingJobName=job_name,
        ProcessingResources={
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": instance_type,
                "VolumeSizeInGB": volume_size
            }
        },
        StoppingCondition={
            "MaxRuntimeInSeconds": 172800  # 48 hours
        },
        AppSpecification={
            "ImageUri": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.7.1-gpu-py3",
            "ContainerEntrypoint": ["python3", "/opt/ml/processing/input/code/processing_script.py"]
        },
        ProcessingInputs=[
            {
                "InputName": "code",
                "S3Input": {
                    "S3Uri": f"s3://{bucket}/{script_s3_key}",
                    "LocalPath": "/opt/ml/processing/input/code",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None"
                }
            }
        ],
        ProcessingOutputConfig={
            "Outputs": outputs
        },
        RoleArn=role_arn,
        Environment={
            "MAX_SAMPLES": str(max_samples),
            "NO_AUGMENTATION": "true" if no_augmentation else "false",
            "AUG_FACTOR": str(aug_factor),
            "REPOSITORY_URL": "https://github.com/shmurthy08/MoE-GAN_CPSC541"  # Replace with your GitHub repo URL
        }
    )
    
    print(f"Started processing job: {job_name}")
    print(f"Output will be stored at: {s3_output_path}")
    
    # Wait for job to complete if requested
    if wait:
        print("Waiting for processing job to complete...")
        waiter = sagemaker.get_waiter('processing_job_completed_or_stopped')
        waiter.wait(ProcessingJobName=job_name)
        
        # Check job status
        job_info = sagemaker.describe_processing_job(ProcessingJobName=job_name)
        job_status = job_info['ProcessingJobStatus']
        print(f"Processing job {job_name} finished with status: {job_status}")
        
        if job_status != 'Completed':
            print(f"Job failed: {job_info.get('FailureReason', 'No failure reason provided')}")
    
    return job_name


def main():
    """Parses command-line arguments and launches the SageMaker processing job."""
    parser = argparse.ArgumentParser(
        description="Launch a SageMaker Processing Job for MS-COCO data preprocessing using a script cloned from GitHub."
    )

    # Required arguments
    parser.add_argument('--role-arn', type=str, required=True,
                        help='The ARN of the SageMaker execution role.')
    parser.add_argument('--bucket', type=str, required=True,
                        help='The S3 bucket for storing the processing script and receiving output.')

    # Optional arguments with defaults matching the function signature
    parser.add_argument('--prefix', type=str, default='mscoco_processed',
                        help='S3 prefix within the bucket for the processed data output (default: mscoco_processed).')
    parser.add_argument('--max-samples', type=int, default=-1,
                        help='Maximum number of samples to process. -1 processes all images (default: -1).')
    parser.add_argument('--no-augmentation', action='store_true',
                        help='If set, disable data augmentation (default: augmentation is enabled).')
    parser.add_argument('--aug-factor', type=int, default=2,
                        help='Factor by which to augment the data if augmentation is enabled (default: 2).')
    parser.add_argument('--instance-type', type=str, default='ml.m5.4xlarge',
                        help='The EC2 instance type for the SageMaker Processing Job (default: ml.m5.24xlarge).')
    parser.add_argument('--volume-size', type=int, default=500,
                        help='The size in GB of the EBS volume attached to the processing instance (default: 500).')
    parser.add_argument('--wait', action='store_true',
                        help='If set, wait for the processing job to complete before exiting the script.')

    args = parser.parse_args()

    # Print the configuration being used - helpful for logs
    print("--- Launching SageMaker Processing Job ---")
    print(f"  Role ARN: {args.role_arn}")
    print(f"  S3 Bucket: {args.bucket}")
    print(f"  S3 Output Prefix: {args.prefix}")
    print(f"  Max Samples: {'All' if args.max_samples == -1 else args.max_samples}")
    print(f"  Data Augmentation: {not args.no_augmentation}")
    if not args.no_augmentation:
        print(f"  Augmentation Factor: {args.aug_factor}")
    print(f"  Instance Type: {args.instance_type}")
    print(f"  Volume Size (GB): {args.volume_size}")
    print(f"  Wait for completion: {args.wait}")
    print("-" * 40) # Separator

    try:
        # Call the main logic function with the parsed arguments
        job_name = launch_processing_job(
            role_arn=args.role_arn,
            bucket=args.bucket,
            prefix=args.prefix,
            max_samples=args.max_samples,
            no_augmentation=args.no_augmentation,
            aug_factor=args.aug_factor,
            instance_type=args.instance_type,
            volume_size=args.volume_size,
            wait=args.wait
        )
        # The success message is already printed inside launch_processing_job
        print(f"\nSuccessfully initiated SageMaker Processing Job: {job_name}")

    except Exception as e:
        print(f"\nERROR: Failed to launch SageMaker Processing Job.", file=sys.stderr)
        print(f"Error details: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc() # Print the full traceback for detailed debugging
        sys.exit(1) # Exit with a non-zero status code to indicate failure

# This ensures the main() function is called only when the script is executed directly
if __name__ == "__main__":
    main()
