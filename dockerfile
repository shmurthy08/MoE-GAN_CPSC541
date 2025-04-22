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

# Create __init__.py files to make directories proper Python packages
RUN touch /app/moegan/__init__.py
RUN touch /app/data_processing/__init__.py

# Create a debug script to help diagnose import issues
RUN echo '#!/usr/bin/env python3\n\
import sys\n\
import os\n\
import importlib\n\
\n\
print("Debug: Checking Python environment")\n\
print(f"Python version: {sys.version}")\n\
print(f"Current directory: {os.getcwd()}")\n\
print(f"Directory contents: {os.listdir()}")\n\
print(f"PYTHONPATH: {os.environ.get(\"PYTHONPATH\", \"Not set\")}")\n\
print(f"sys.path: {sys.path}")\n\
\n\
# Check data_processing directory\n\
print(f"data_processing directory contents: {os.listdir(\"/app/data_processing\") if os.path.exists(\"/app/data_processing\") else \"Directory not found\"}")\n\
\n\
# Try to find the ProcessedMSCOCODataset class\n\
for root, dirs, files in os.walk(\"/app\"):\n\
    for file in files:\n\
        if file.endswith(\".py\"):\n\
            with open(os.path.join(root, file), \"r\") as f:\n\
                content = f.read()\n\
                if \"ProcessedMSCOCODataset\" in content:\n\
                    print(f"Found ProcessedMSCOCODataset in {os.path.join(root, file)}")\n\
\n\
print("End of debug information")\n\
' > /app/debug_imports.py
RUN chmod +x /app/debug_imports.py

# Training stage
FROM base as training

# Install training-specific dependencies
RUN pip install --no-cache-dir \
    boto3 sagemaker

# Copy training-specific files
COPY scripts/*.py /app/scripts/

# Set environment variables
ENV PYTHONUNBUFFERED=1
# Ensure Python can find all modules by setting a comprehensive PYTHONPATH
ENV PYTHONPATH=/app:/app/moegan:/app/data_processing

# Modify the entrypoint to run our debug script first
ENTRYPOINT ["sh", "-c", "python /app/debug_imports.py && python /app/moegan/sagemaker_train.py"]

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
ENV PYTHONPATH=/app

# Create serving script
RUN echo '#!/usr/bin/env python3\n\
import sys\n\
import os\n\
from sagemaker_inference import model_server\n\
model_server.start_model_server(handler_service="/app/moegan/inference.py")' > /app/serve && \
chmod +x /app/serve

ENTRYPOINT ["/app/serve"]