# Multi-stage Dockerfile for MoE-GAN

# Base stage with common dependencies
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        libgl1-mesa-glx \
        libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir tensorboard

# Install CLIP without dependencies to avoid downgrading PyTorch
RUN pip install --no-deps git+https://github.com/openai/CLIP.git && \
    pip install --no-deps ftfy regex tqdm

# Copy common code files
COPY moegan/*.py /app/moegan/
COPY data_processing/*.py /app/data_processing/

# Training stage
FROM base as training

# Install training-specific dependencies
RUN pip install --no-cache-dir \
    boto3 sagemaker

# Copy training-specific files
COPY scripts/*.py /app/scripts/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/app/data_processing:/app/scripts::/app/moegan



# Set entry point
ENTRYPOINT ["python", "/app/moegan/sagemaker_train.py"]

# Inference stage
FROM base as inference

# Install inference-specific dependencies
RUN pip install --no-cache-dir \
    flask \
    gunicorn \
    pillow \
    boto3 \
    sagemaker-inference

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/app/data_processing:/app/scripts:/app/moegan

# Create serving script
RUN echo '#!/usr/bin/env python3\n\
import sys\n\
import os\n\
from sagemaker_inference import model_server\n\
model_server.start_model_server(handler_service="/app/moegan/inference.py")' > /app/serve && \
chmod +x /app/serve

ENTRYPOINT ["/app/serve"]