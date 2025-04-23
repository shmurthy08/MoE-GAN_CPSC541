#!/usr/bin/env python3
import boto3
import argparse
import time
import json
import os

def main():
    parser = argparse.ArgumentParser(description='Start a SageMaker training job')
    parser.add_argument('--job-name', required=True, help='Training job name')
    parser.add_argument('--role-arn', required=True, help='SageMaker execution role ARN')
    parser.add_argument('--image-uri', required=True, help='ECR image URI')
    parser.add_argument('--bucket', required=True, help='S3 bucket for model artifacts')
    parser.add_argument('--prefix', required=True, help='S3 prefix for model artifacts')
    parser.add_argument('--instance-type', default='ml.g5.xlarge', help='Instance type')
    parser.add_argument('--hyperparameters', help='Path to hyperparameters JSON file')
    parser.add_argument('--data-bucket', help='S3 bucket containing training data')
    parser.add_argument('--data-prefix', default='mscoco_processed', help='S3 prefix for training data')
    parser.add_argument('--wait', action='store_true',
                        help='If set, wait for the training job to complete before exiting the script.')
    
    args = parser.parse_args()
    
    sagemaker = boto3.client('sagemaker')
    
    print(f"Starting training job: {args.job_name}")
    
    # Load hyperparameters from file if provided
    if args.hyperparameters and os.path.exists(args.hyperparameters):
        with open(args.hyperparameters, 'r') as f:
            hyperparameters = json.load(f)
    else:
        # Default hyperparameters
        hyperparameters = {
            "epochs": "200",
            "batch_size": "8",
            "learning_rate": "0.0002",
            "beta1": "0.5",
            "beta2": "0.999",
            "r1_gamma": "10.0",
            "clip_weight_64": "0.1",
            "clip_weight_32": "0.05",
            "kl_weight": "0.001",
            "balance_weight": "0.01"
        }
    
    # Convert all values to strings (SageMaker requirement)
    for k, v in hyperparameters.items():
        hyperparameters[k] = str(v)
    
    # Add data location environment variables
    environment = {}
    if args.data_bucket:
        environment['S3_DATA_BUCKET'] = args.data_bucket
        environment['S3_DATA_PREFIX'] = args.data_prefix
    
    # Start the training job
    response = sagemaker.create_training_job(
        TrainingJobName=args.job_name,
        AlgorithmSpecification={
            'TrainingImage': args.image_uri,
            'TrainingInputMode': 'File'
        },
        RoleArn=args.role_arn,
        OutputDataConfig={
            'S3OutputPath': f's3://{args.bucket}/{args.prefix}'
        },
        ResourceConfig={
            'InstanceType': args.instance_type,
            'InstanceCount': 1,
            'VolumeSizeInGB': 30  # Increased size for ML datasets
        },
        StoppingCondition={
            'MaxRuntimeInSeconds': 864000  
        },
        HyperParameters=hyperparameters,
        Environment=environment,
        Tags=[
            {
                'Key': 'Project',
                'Value': 'MoE-GAN'
            }
        ]
    )
    
    print(f"Training job started: {response['TrainingJobArn']}")
    
    # Wait for job to complete if requested
    if args.wait:
        print("Waiting for training job to complete...")
        waiter = sagemaker.get_waiter('training_job_completed_or_stopped')
        waiter.wait(TrainingJobName=args.job_name)
        
        # Check job status
        job_info = sagemaker.describe_training_job(TrainingJobName=args.job_name)
        job_status = job_info['TrainingJobStatus']
        print(f"Training job {args.job_name} finished with status: {job_status}")
        
        if job_status != 'Completed':
            print(f"Job failed or stopped: {job_info.get('FailureReason', 'No failure reason provided')}")
    
    return response

if __name__ == '__main__':
    main()