# Base image for PyTorch (change CUDA version as needed)
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/ml/code:${PATH}"

# Set the working directory
WORKDIR /opt/ml/code

# Copy requirements.txt first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for Bayesian networks and SageMaker
RUN pip install --no-cache-dir \
    pyro-ppl==1.8.0 \
    transformers==4.18.0 \
    sagemaker-training==4.0.0 \
    sagemaker-inference==1.5.0

# Copy the code
COPY . .

# Create directories expected by SageMaker
RUN mkdir -p /opt/ml/model \
    /opt/ml/input/data \
    /opt/ml/input/config \
    /opt/ml/output

# Set up entrypoints for training and serving
RUN chmod +x /opt/ml/code/train.py
RUN chmod +x /opt/ml/code/inference.py

# SageMaker uses the following entrypoints
ENV SAGEMAKER_PROGRAM train.py

# Create minimal train and inference scripts if they don't exist
RUN if [ ! -f /opt/ml/code/train.py ]; then \
    echo 'import sys; print("Training script placeholder - replace with actual training code"); sys.exit(0)' > /opt/ml/code/train.py; \
    fi

RUN if [ ! -f /opt/ml/code/inference.py ]; then \
    echo 'def input_fn(input_data, content_type): return input_data\ndef predict_fn(input_data, model): return {"response": "Inference script placeholder - replace with actual inference code"}\ndef output_fn(prediction, accept): return prediction\ndef model_fn(model_dir): return None' > /opt/ml/code/inference.py; \
    fi