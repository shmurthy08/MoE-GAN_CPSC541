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
    max_samples=-1,  # -1 will process all images
    no_augmentation=False,
    aug_factor=2,
    instance_type='ml.m5.24xlarge',  # Larger instance for full dataset
    volume_size=500,  # Increased EBS volume size for the full dataset
    wait=False
):
    sagemaker = boto3.client('sagemaker')
    
    # Generate a unique job name
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    job_name = f"moe-gan-data-processing-{timestamp}"
    
    # Configure processing inputs
    inputs = []
    
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
    
    # Prepare a tar.gz of code directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    code_path = os.path.join(current_dir, "code.tar.gz")
    
    # Create tar.gz with your code
    if os.system(f"tar -czf {code_path} -C {parent_dir} data_processing/ scripts/ *.py") != 0:
        print("Failed to create code archive")
        sys.exit(1)
    
    # Upload code to S3
    code_prefix = f"code/{job_name}/code.tar.gz"
    s3 = boto3.client('s3')
    s3.upload_file(code_path, bucket, code_prefix)
    
    # Build command
    command = [
        "python", "scripts/processing_script.py",
        f"--max-samples", str(max_samples),
        f"--aug-factor", str(aug_factor),
        f"--s3-output-path", s3_output_path
    ]
    
    if no_augmentation:
        command.append("--no-augmentation")
    
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
            "MaxRuntimeInSeconds": 172800  # 48 hours for the full dataset
        },
        AppSpecification={
            "ImageUri": f"{boto3.client('sts').get_caller_identity().get('Account')}.dkr.ecr.{boto3.session.Session().region_name}.amazonaws.com/sagemaker-pytorch:1.7.1-cpu-py3",
            "ContainerEntrypoint": command
        },
        ProcessingInputs=[
            {
                "InputName": "code",
                "S3Input": {
                    "S3Uri": f"s3://{bucket}/{code_prefix}",
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
            "PYTHONPATH": "/opt/ml/processing/input/code"
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
        job_status = sagemaker.describe_processing_job(ProcessingJobName=job_name)['ProcessingJobStatus']
        if job_status == 'Completed':
            print(f"Processing job {job_name} completed successfully!")
        else:
            print(f"Processing job {job_name} failed or stopped: {job_status}")
            
            # Get failure reason if available
            if 'FailureReason' in job_status:
                print(f"Failure reason: {job_status['FailureReason']}")
    
    return job_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Launch SageMaker Processing Job for MS-COCO dataset')
    parser.add_argument('--role-arn', required=True, help='SageMaker execution role ARN')
    parser.add_argument('--bucket', required=True, help='S3 bucket for processed data')
    parser.add_argument('--prefix', default='mscoco_processed', help='S3 prefix for processed data')
    parser.add_argument('--max-samples', type=int, default=-1, help='Max samples (-1 for all)')
    parser.add_argument('--no-augmentation', action='store_true', help='Disable augmentation')
    parser.add_argument('--aug-factor', type=int, default=2, help='Augmentation factor')
    parser.add_argument('--instance-type', default='ml.m5.24xlarge', help='Instance type')
    parser.add_argument('--volume-size', type=int, default=500, help='EBS volume size in GB')
    parser.add_argument('--wait', action='store_true', help='Wait for job to complete')
    
    args = parser.parse_args()
    
    launch_processing_job(
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