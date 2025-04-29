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

# Create proper Python package structure
RUN mkdir -p /app/moegan /app/data_processing

# Create __init__.py files to make directories proper Python packages
RUN echo "from .t2i_moe_gan import AuroraGenerator, sample_aurora_gan" > /app/moegan/__init__.py
RUN touch /app/data_processing/__init__.py

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
RUN mkdir -p /app/scripts && touch /app/scripts/__init__.py

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/app/data_processing:/app/scripts:/app/moegan

# Set entry point
ENTRYPOINT ["python", "/app/moegan/sagemaker_train.py"]

# Inference stage
FROM base as inference

# Keep inference.py within package structure
# DO NOT copy to /app/inference.py as in original
# This ensures imports work correctly

RUN pip install --no-cache-dir \
    scipy \
    torchvision

# Install Java and inference-specific dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openjdk-11-jre-headless && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install inference-specific dependencies
RUN pip install --no-cache-dir \
    flask \
    gunicorn \
    pillow \
    boto3 \
    multi-model-server \
    sagemaker-inference

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/app/data_processing:/app/scripts:/app/moegan
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# IMPORTANT: Update handler service name to use correct module path
RUN echo '#!/usr/bin/env python3\n\
from sagemaker_inference import model_server\n\
\n\
# Use moegan.inference instead of inference\n\
model_server.start_model_server(handler_service="moegan.inference")' \
> /app/serve && chmod +x /app/serve

ENTRYPOINT ["/app/serve"]