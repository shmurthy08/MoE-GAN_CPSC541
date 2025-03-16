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
    parser.add_argument('--instance-type', default='ml.t2.medium', help='Instance type')
    
    args = parser.parse_args()
    
    sagemaker = boto3.client('sagemaker')
    
    print(f"Starting training job: {args.job_name}")
    
    # Define hyperparameters
    hyperparameters = {
        "epochs": "10",
        "batch_size": "32",
        "learning_rate": "0.0002"
    }
    
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
            'VolumeSizeInGB': 10
        },
        StoppingCondition={
            'MaxRuntimeInSeconds': 3600  # 1 hour
        },
        HyperParameters=hyperparameters
    )
    
    print(f"Training job started: {response['TrainingJobArn']}")
    return response

if __name__ == '__main__':
    main()