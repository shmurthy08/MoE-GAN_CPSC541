#!/usr/bin/env python3
import boto3
import argparse
import time
import json
import os
from datetime import datetime

def wait_for_hyperparameter_tuning_job(sagemaker_client, job_name, timeout_minutes=120):
    """
    Wait for a hyperparameter tuning job to complete.
    
    Args:
        sagemaker_client: boto3 SageMaker client
        job_name: Name of the hyperparameter tuning job
        timeout_minutes: Maximum time to wait in minutes
    
    Returns:
        Final status of the job
    """
    start_time = time.time()
    status = None
    
    while True:
        # Check if we've exceeded the timeout
        elapsed_minutes = (time.time() - start_time) / 60
        if elapsed_minutes > timeout_minutes:
            print(f"Timeout: Waited {timeout_minutes} minutes for job to complete")
            break
        
        try:
            response = sagemaker_client.describe_hyper_parameter_tuning_job(
                HyperParameterTuningJobName=job_name
            )
            
            status = response['HyperParameterTuningJobStatus']
            
            # Print progress information
            if 'TrainingJobStatusCounters' in response:
                counters = response['TrainingJobStatusCounters']
                completed = counters.get('Completed', 0)
                in_progress = counters.get('InProgress', 0)
                failed = counters.get('Failed', 0)
                stopped = counters.get('Stopped', 0)
                
                total = completed + in_progress + failed + stopped
                print(f"Job: {job_name} | Status: {status} | Completed: {completed}, InProgress: {in_progress}, Failed: {failed}, Stopped: {stopped}, Total: {total}")
            else:
                print(f"Job: {job_name} | Status: {status}")
            
            # Check if job has reached a terminal state
            if status in ['Completed', 'Failed', 'Stopped']:
                break
            
            # If job has been running for a while, check if it's stuck
            if elapsed_minutes > 30 and 'TrainingJobStatusCounters' in response:
                # If no progress has been made for a while, it might be worth checking
                counters = response['TrainingJobStatusCounters']
                if counters.get('InProgress', 0) == 0 and counters.get('Completed', 0) == 0:
                    print("Warning: No training jobs have completed or are in progress after 30 minutes")
            
            # Wait before checking again
            time.sleep(30)
            
        except Exception as e:
            print(f"Error checking job status: {str(e)}")
            time.sleep(30)
    
    return status

def main():
    parser = argparse.ArgumentParser(description='Start a SageMaker hyperparameter tuning job')
    parser.add_argument('--job-name', help='Tuning job name (default: auto-generated)')
    parser.add_argument('--role-arn', required=True, help='SageMaker execution role ARN')
    parser.add_argument('--image-uri', required=True, help='ECR image URI')
    parser.add_argument('--bucket', required=True, help='S3 bucket for model artifacts')
    parser.add_argument('--prefix', required=True, help='S3 prefix for model artifacts')
    parser.add_argument('--data-bucket', help='S3 bucket containing training data')
    parser.add_argument('--data-prefix', default='mscoco_processed', help='S3 prefix for training data')
    parser.add_argument('--instance-type', default='ml.g5.xlarge', help='Instance type')
    parser.add_argument('--max-jobs', type=int, default=10, help='Maximum number of training jobs')
    parser.add_argument('--max-parallel-jobs', type=int, default=2, help='Maximum parallel training jobs')
    parser.add_argument('--config', help='Path to hyperparameter configuration JSON file')
    parser.add_argument('--wait', action='store_true',
                        help='If set, wait for the tuning job to complete before exiting the script.')
    parser.add_argument('--timeout-minutes', type=int, default=120,
                        help='Maximum time to wait for job completion in minutes')
    
    args = parser.parse_args()
    
    # Generate job name if not provided
    if not args.job_name:
        timestamp = datetime.now().strftime('%y%m%d%H%M')  
        # Ensure the job name is 32 characters or less
        args.job_name = f"gan-hpo-{timestamp}"[:32]
    
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
                "epochs": "10",  
                "batch_size": "8"
            },
            "objective_metric": {
                "name": "val_clip_loss",  
                "type": "Minimize"  
            }
        }

    # Create SageMaker client
    sagemaker = boto3.client('sagemaker')
    
    print(f"Starting hyperparameter tuning job: {args.job_name}")
    
    # Environment variables for data location
    # Define the desired PYTHONPATH
    desired_pythonpath = "/app:/app/data_processing:/app/scripts:/app/moegan"

    # Environment variables for data location AND PYTHONPATH
    environment = {
        'PYTHONPATH': desired_pythonpath # Explicitly set PYTHONPATH here
    }
    if args.data_bucket:
        environment['S3_DATA_BUCKET'] = args.data_bucket
        environment['S3_DATA_PREFIX'] = args.data_prefix
    # Construct the HyperParameterTuningJobConfig
    tuning_job_config = {
        'Strategy': 'Bayesian',
        'HyperParameterTuningJobObjective': {
            'Type': hp_config['objective_metric']['type'],
            'MetricName': hp_config['objective_metric']['name']
        },
        'ResourceLimits': {
            'MaxNumberOfTrainingJobs': args.max_jobs,
            'MaxParallelTrainingJobs': args.max_parallel_jobs
        },
        'ParameterRanges': {
            'ContinuousParameterRanges': [],
            'IntegerParameterRanges': []
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
    # Add integer hyperparameter ranges if any
    for param_name, param_config in hp_config['integer_parameter_ranges'].items():
        tuning_job_config['ParameterRanges']['IntegerParameterRanges'].append({
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
            'MaxRuntimeInSeconds': 432000 # 5 days  
        },
        'Environment': environment
    }
    
    try:
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
        
        # Wait for job to complete if requested
        if args.wait:
            print(f"Waiting for hyperparameter tuning job to complete (timeout: {args.timeout_minutes} minutes)...")
            final_status = wait_for_hyperparameter_tuning_job(sagemaker, args.job_name, args.timeout_minutes)
            
            # Get final job details
            final_job_info = sagemaker.describe_hyper_parameter_tuning_job(
                HyperParameterTuningJobName=args.job_name
            )
            
            print(f"\nHyperparameter tuning job {args.job_name} finished with status: {final_status}")
            
            # Print summary of job results
            if final_status == 'Completed':
                if 'BestTrainingJob' in final_job_info:
                    best_job = final_job_info['BestTrainingJob']
                    print(f"Best training job: {best_job['TrainingJobName']}")
                    print(f"Best objective metric value: {best_job['FinalHyperParameterTuningJobObjectiveMetric']['Value']}")
                else:
                    print("No best training job information available")
                    
                # Print overall statistics
                if 'TrainingJobStatusCounters' in final_job_info:
                    counters = final_job_info['TrainingJobStatusCounters']
                    print(f"\nJob statistics:")
                    print(f"  Completed: {counters.get('Completed', 0)}")
                    print(f"  Failed: {counters.get('Failed', 0)}")
                    print(f"  Stopped: {counters.get('Stopped', 0)}")
                    print(f"  Total: {counters.get('Completed', 0) + counters.get('Failed', 0) + counters.get('Stopped', 0)}")
                    
            elif final_status == 'Failed':
                print(f"Job failed: {final_job_info.get('FailureReason', 'No failure reason provided')}")
                exit(1)
            elif final_status == 'Stopped':
                print("Job was stopped")
                exit(1)
            else:
                print(f"Job ended with status: {final_status}")
                exit(1)
                
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)
    
    return response

if __name__ == '__main__':
    main()