# Use a lightweight Python base image
FROM python:3.8-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/ml/code:${PATH}"

# Set the working directory
WORKDIR /opt/ml/code

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies
RUN pip install --no-cache-dir \
    torch==1.10.0 \
    pyro-ppl==1.8.0 \
    transformers==4.18.0 \
    sagemaker-training==4.0.0 \
    sagemaker-inference==1.5.0

# Create SageMaker directories
RUN mkdir -p /opt/ml/model \
    /opt/ml/input/data \
    /opt/ml/input/config \
    /opt/ml/output

# Copy the code
COPY . .

# Ensure scripts are executable
RUN chmod +x /opt/ml/code/train.py || echo "train.py not found or not executable"
RUN chmod +x /opt/ml/code/inference.py || echo "inference.py not found or not executable"

# SageMaker environment variable
ENV SAGEMAKER_PROGRAM train.py

# Default command
CMD ["python", "/opt/ml/code/train.py"]