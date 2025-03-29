# Multi-stage Dockerfile for MoE-GAN

# Base stage with common dependencies
FROM python:3.8-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install basic Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install CLIP and other ML dependencies
RUN pip install --no-cache-dir \
    torch==1.10.1 \
    torchvision==0.11.2 \
    clip-by-openai \
    fiftyone \
    tensorboard

# Copy common code files
COPY moegan/*.py /app/moegan/
COPY data_processing/*.py /app/data_processing/

# Training stage
FROM base as training

# Install training-specific dependencies
RUN pip install --no-cache-dir \
    ftfy regex tqdm boto3 sagemaker

# Copy training-specific files
COPY scripts/*.py /app/scripts/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Copy SageMaker training entry point
COPY moegan/sagemaker_train.py /app/
ENTRYPOINT ["python", "/app/sagemaker_train.py"]

# Inference stage
FROM base as inference

# Install inference-specific dependencies
RUN pip install --no-cache-dir \
    flask \
    gunicorn \
    pillow \
    boto3 \
    sagemaker-inference

# Copy inference code
COPY moegan/inference.py /app/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Create serving script
RUN echo '#!/usr/bin/env python3\n\
import sys\n\
import os\n\
from sagemaker_inference import model_server\n\
model_server.start_model_server(handler_service="/app/inference.py")' > /app/serve && \
chmod +x /app/serve

ENTRYPOINT ["/app/serve"]