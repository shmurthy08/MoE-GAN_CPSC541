#!/usr/bin/env python3
import boto3
import argparse
import time
import json
import os
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Start a SageMaker hyperparameter tuning job')
    parser.add_argument('--job-name', help='Tuning job name (default: auto-generated)')
    parser.add_argument('--role-arn', required=True, help='SageMaker execution role ARN')
    parser.add_argument('--image-uri', required=True, help='ECR image URI')
    parser.add_argument('--bucket', required=True, help='S3 bucket for model artifacts')
    parser.add_argument('--prefix', required=True, help='S3 prefix for model artifacts')
    parser.add_argument('--data-bucket', help='S3 bucket containing training data')
    parser.add_argument('--data-prefix', default='mscoco_processed', help='S3 prefix for training data')
    parser.add_argument('--instance-type', default='ml.g4dn.xlarge', help='Instance type')
    parser.add_argument('--max-jobs', type=int, default=10, help='Maximum number of training jobs')
    parser.add_argument('--max-parallel-jobs', type=int, default=2, help='Maximum parallel training jobs')
    parser.add_argument('--config', help='Path to hyperparameter configuration JSON file')
    
    args = parser.parse_args()
    
    # Generate job name if not provided
    if not args.job_name:
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        args.job_name = f"moe-gan-hpo-{timestamp}"
    
    # Load hyperparameter configuration from file if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            hp_config = json.load(f)
    else:
        # Default hyperparameter ranges
        hp_config = {
            "hyperparameter_ranges": {
                "learning_rate": {
                    "min_value": 0.0001,
                    "max_value": 0.001,
                    "scaling_type": "Logarithmic"
                },
                "beta1": {
                    "min_value": 0.5,
                    "max_value": 0.9,
                    "scaling_type": "Linear"
                },
                "beta2": {
                    "min_value": 0.9,
                    "max_value": 0.999,
                    "scaling_type": "Linear"
                },
                "r1_gamma": {
                    "min_value": 1.0,
                    "max_value": 20.0,
                    "scaling_type": "Linear"
                },
                "clip_weight_64": {
                    "min_value": 0.05,
                    "max_value": 0.2,
                    "scaling_type": "Linear"
                },
                "clip_weight_32": {
                    "min_value": 0.01,
                    "max_value": 0.1,
                    "scaling_type": "Linear"
                },
                "kl_weight": {
                    "min_value": 0.0001,
                    "max_value": 0.01,
                    "scaling_type": "Logarithmic"
                },
                "balance_weight": {
                    "min_value": 0.001,
                    "max_value": 0.1,
                    "scaling_type": "Logarithmic"
                }
            },
            "static_hyperparameters": {
                "epochs": "5",  # Reduce epochs for faster tuning
                "batch_size": "32"
            },
            "objective_metric": {
                "name": "val_clip_loss",  # Metric to optimize
                "type": "Minimize"  # Minimize loss
            }
        }

    # Create SageMaker client
    sagemaker = boto3.client('sagemaker')
    
    print(f"Starting hyperparameter tuning job: {args.job_name}")
    
    # Environment variables for data location
    environment = {}
    if args.data_bucket:
        environment['S3_DATA_BUCKET'] = args.data_bucket
        environment['S3_DATA_PREFIX'] = args.data_prefix
    
    # Construct the HyperParameterTuningJobConfig
    tuning_job_config = {
        'Strategy': 'Bayesian',
        'HyperParameterTuningJobObjective': hp_config['objective_metric'],
        'ResourceLimits': {
            'MaxNumberOfTrainingJobs': args.max_jobs,
            'MaxParallelTrainingJobs': args.max_parallel_jobs
        },
        'ParameterRanges': {
            'ContinuousParameterRanges': []
        },
        'TrainingJobEarlyStoppingType': 'Auto'
    }
    
    # Add hyperparameter ranges
    for param_name, param_config in hp_config['hyperparameter_ranges'].items():
        tuning_job_config['ParameterRanges']['ContinuousParameterRanges'].append({
            'Name': param_name,
            'MinValue': str(param_config['min_value']),
            'MaxValue': str(param_config['max_value']),
            'ScalingType': param_config['scaling_type']
        })
    
    # Create the training job definition
    training_job_definition = {
        'StaticHyperParameters': hp_config['static_hyperparameters'],
        'AlgorithmSpecification': {
            'TrainingImage': args.image_uri,
            'TrainingInputMode': 'File',
            'MetricDefinitions': [
                {
                    'Name': hp_config['objective_metric']['name'],
                    # Adjust the regex pattern based on your model's output format
                    'Regex': f".*{hp_config['objective_metric']['name']}: ([0-9\\.]+).*"
                }
            ]
        },
        'RoleArn': args.role_arn,
        'OutputDataConfig': {
            'S3OutputPath': f's3://{args.bucket}/{args.prefix}'
        },
        'ResourceConfig': {
            'InstanceType': args.instance_type,
            'InstanceCount': 1,
            'VolumeSizeInGB': 30
        },
        'StoppingCondition': {
            'MaxRuntimeInSeconds': 86400  # 24 hours max runtime
        },
        'Environment': environment
    }
    
    # Start the hyperparameter tuning job
    response = sagemaker.create_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=args.job_name,
        HyperParameterTuningJobConfig=tuning_job_config,
        TrainingJobDefinition=training_job_definition,
        Tags=[
            {
                'Key': 'Project',
                'Value': 'MoE-GAN'
            }
        ]
    )
    
    print(f"Hyperparameter tuning job started: {response['HyperParameterTuningJobArn']}")
    print(f"Job will run up to {args.max_jobs} training jobs, with {args.max_parallel_jobs} in parallel")
    print(f"Job status can be monitored in the SageMaker console or via the AWS CLI/SDK")
    
    return response

if __name__ == '__main__':
    main()