#!/usr/bin/env python3
import boto3
import argparse
import json
import os

def get_best_tuning_job(tuning_job_name):
    """
    Get the best training job from a completed hyperparameter tuning job
    """
    sagemaker = boto3.client('sagemaker')
    
    # Get tuning job info
    tuning_job = sagemaker.describe_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=tuning_job_name
    )
    
    # Check if tuning job is complete
    status = tuning_job['HyperParameterTuningJobStatus']
    if status != 'Completed':
        print(f"Tuning job {tuning_job_name} is not complete. Current status: {status}")
        return None
    
    # Get best training job
    best_training_job = tuning_job['BestTrainingJob']
    if not best_training_job:
        print(f"No best training job found for tuning job {tuning_job_name}")
        return None
    
    print(f"Best training job: {best_training_job['TrainingJobName']}")
    print(f"Objective metric: {best_training_job['FinalHyperParameterTuningJobObjectiveMetric']['Name']} = {best_training_job['FinalHyperParameterTuningJobObjectiveMetric']['Value']}")
    
    # Get best hyperparameters
    training_job = sagemaker.describe_training_job(
        TrainingJobName=best_training_job['TrainingJobName']
    )
    
    hyperparameters = training_job['HyperParameters']
    print("\nBest hyperparameters:")
    for key, value in hyperparameters.items():
        print(f"  {key}: {value}")
    
    # Get model artifacts
    model_artifacts = training_job['ModelArtifacts']['S3ModelArtifacts']
    print(f"\nModel artifacts: {model_artifacts}")
    
    return {
        'training_job_name': best_training_job['TrainingJobName'],
        'model_artifacts': model_artifacts,
        'hyperparameters': hyperparameters,
        'objective_metric': {
            'name': best_training_job['FinalHyperParameterTuningJobObjectiveMetric']['Name'],
            'value': best_training_job['FinalHyperParameterTuningJobObjectiveMetric']['Value']
        }
    }

def deploy_best_model(best_model_info, role_arn, image_uri, model_name=None, endpoint_config_name=None, endpoint_name=None, instance_type='ml.g5.xlarge'):
    """
    Deploy the best model from a hyperparameter tuning job
    """
    sagemaker = boto3.client('sagemaker')
    
    # Set default names if not provided
    if not model_name:
        model_name = f"{best_model_info['training_job_name']}-model"
    if not endpoint_config_name:
        endpoint_config_name = f"{model_name}-config"
    if not endpoint_name:
        endpoint_name = f"{model_name}-endpoint"
    
    # Create model
    print(f"Creating model: {model_name}")
    try:
        sagemaker.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': image_uri,
                'ModelDataUrl': best_model_info['model_artifacts']
            },
            ExecutionRoleArn=role_arn
        )
    except sagemaker.exceptions.ClientError as e:
        if 'ResourceInUse' in str(e):
            print(f"Model {model_name} already exists. Deleting and recreating...")
            sagemaker.delete_model(ModelName=model_name)
            sagemaker.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    'Image': image_uri,
                    'ModelDataUrl': best_model_info['model_artifacts']
                },
                ExecutionRoleArn=role_arn
            )
        else:
            raise e
    
    # Create endpoint config
    print(f"Creating endpoint config: {endpoint_config_name}")
    try:
        sagemaker.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': instance_type
                }
            ]
        )
    except sagemaker.exceptions.ClientError as e:
        if 'ResourceInUse' in str(e):
            print(f"Endpoint config {endpoint_config_name} already exists. Deleting and recreating...")
            sagemaker.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
            sagemaker.create_endpoint_config(
                EndpointConfigName=endpoint_config_name,
                ProductionVariants=[
                    {
                        'VariantName': 'AllTraffic',
                        'ModelName': model_name,
                        'InitialInstanceCount': 1,
                        'InstanceType': instance_type
                    }
                ]
            )
        else:
            raise e
    
    # Check if endpoint exists
    try:
        sagemaker.describe_endpoint(EndpointName=endpoint_name)
        endpoint_exists = True
    except sagemaker.exceptions.ClientError:
        endpoint_exists = False
    
    # Create or update endpoint
    if endpoint_exists:
        print(f"Updating endpoint: {endpoint_name}")
        sagemaker.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
    else:
        print(f"Creating endpoint: {endpoint_name}")
        sagemaker.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
    
    print(f"Endpoint deployment initiated. This may take several minutes.")
    print(f"Endpoint URL: https://{sagemaker.meta.region_name}.sagemaker.amazonaws.com/endpoints/{endpoint_name}/invocations")
    
    return {
        'model_name': model_name,
        'endpoint_config_name': endpoint_config_name,
        'endpoint_name': endpoint_name
    }

def main():
    parser = argparse.ArgumentParser(description='Get the best model from a SageMaker hyperparameter tuning job')
    parser.add_argument('--tuning-job-name', required=True, help='Name of the hyperparameter tuning job')
    parser.add_argument('--role-arn', required=True, help='SageMaker execution role ARN')
    parser.add_argument('--image-uri', required=True, help='ECR image URI for inference')
    parser.add_argument('--deploy', action='store_true', help='Deploy the best model after retrieving it')
    parser.add_argument('--model-name', help='Name for the deployed model (default: auto-generated)')
    parser.add_argument('--endpoint-name', help='Name for the deployed endpoint (default: auto-generated)')
    parser.add_argument('--instance-type', default='ml.g5.xlarge', help='Instance type for the endpoint')
    parser.add_argument('--output-file', help='Save best model info to this JSON file')
    
    args = parser.parse_args()
    
    # Get best model from tuning job
    best_model_info = get_best_tuning_job(args.tuning_job_name)
    
    if best_model_info and args.deploy:
        # Deploy best model
        deployment_info = deploy_best_model(
            best_model_info,
            args.role_arn,
            args.image_uri,
            model_name=args.model_name,
            endpoint_name=args.endpoint_name,
            instance_type=args.instance_type
        )
        
        # Combine info
        output_info = {
            **best_model_info,
            **deployment_info
        }
    else:
        output_info = best_model_info
    
    # Save to file if requested
    if output_info and args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(output_info, f, indent=2)
        print(f"Best model info saved to {args.output_file}")
    
    return output_info

if __name__ == '__main__':
    main()