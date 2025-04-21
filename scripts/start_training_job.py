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
    parser.add_argument('--instance-type', default='ml.g4dn.xlarge', help='Instance type')
    parser.add_argument('--hyperparameters', help='Path to hyperparameters JSON file')
    parser.add_argument('--data-bucket', help='S3 bucket containing training data')
    parser.add_argument('--data-prefix', default='mscoco_processed', help='S3 prefix for training data')
    
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
            "epochs": "10000",
            "batch_size": "32",
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
            'MaxRuntimeInSeconds': 86400  # 24 hours max runtime
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
    
    # Wait for job to complete if desired
    wait_for_completion = os.environ.get('WAIT_FOR_COMPLETION', 'false').lower() == 'true'
    if wait_for_completion:
        print("Waiting for training job to complete...")
        
        while True:
            status = sagemaker.describe_training_job(TrainingJobName=args.job_name)['TrainingJobStatus']
            print(f"Current status: {status}")
            
            if status in ['Completed', 'Failed', 'Stopped']:
                break
                
            time.sleep(60)  # Check every minute
            
        print(f"Training job finished with status: {status}")
    
    return response

if __name__ == '__main__':
    main()