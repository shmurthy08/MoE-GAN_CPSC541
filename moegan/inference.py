# moegan/inference.py
import os
import json
import torch
import base64
from io import BytesIO
import numpy as np
from PIL import Image
from t2i_moe_gan import AuroraGenerator, sample_aurora_gan

# Constants
LATENT_DIM = 512
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')

# Load model once when container starts
model = None

def load_model():
    """Load the model from the saved checkpoint"""
    global model
    if model is None:
        print(f"Loading model from: {MODEL_PATH}")
        checkpoint_path = os.path.join(MODEL_PATH, 'aurora_model_final.pt')
        
        # Check if model file exists
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            model = AuroraGenerator().to(DEVICE)
            
            # Load model weights
            if 'generator' in checkpoint:
                model.load_state_dict(checkpoint['generator'])
            else:
                model.load_state_dict(checkpoint)
                
            model.eval()
            print("Model loaded successfully")
        else:
            # Try to find any .pt file
            for file in os.listdir(MODEL_PATH):
                if file.endswith('.pt'):
                    alt_path = os.path.join(MODEL_PATH, file)
                    print(f"Loading alternative model: {alt_path}")
                    checkpoint = torch.load(alt_path, map_location=DEVICE)
                    model = AuroraGenerator().to(DEVICE)
                    
                    # Load model weights
                    if 'generator' in checkpoint:
                        model.load_state_dict(checkpoint['generator'])
                    else:
                        model.load_state_dict(checkpoint)
                    
                    model.eval()
                    break
            
            if model is None:
                raise FileNotFoundError(f"No model file found in {MODEL_PATH}")
                
    return model

# SageMaker inference handlers
def model_fn(model_dir):
    """
    Load model for SageMaker inference
    """
    global MODEL_PATH
    MODEL_PATH = model_dir
    return load_model()

def transform_fn(model, request_body, content_type, accept_type):
    """
    Transform function for SageMaker inference
    """
    if content_type != 'application/json':
        return json.dumps({"error": "Expected application/json content type"}), 'application/json', 415
    
    # Parse request
    request = json.loads(request_body)
    text_prompt = request.get('text', '')
    
    if not text_prompt:
        return json.dumps({"error": "Text prompt is required"}), 'application/json', 400
    
    # Number of images to generate (limit to 4 max)
    num_samples = min(request.get('num_samples', 1), 4)
    truncation_psi = request.get('truncation_psi', 0.7)
    
    try:
        # Generate images
        images = sample_aurora_gan(
            model, 
            text_prompt, 
            num_samples=num_samples,
            truncation_psi=truncation_psi, 
            device=DEVICE
        )
        
        # Convert to base64 encoded PNGs
        encoded_images = []
        for i in range(images.size(0)):
            # Convert to numpy array in correct format
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
            img = (img + 1) / 2  # [-1, 1] -> [0, 1]
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            
            # Save as PNG in memory
            pil_img = Image.fromarray(img)
            buffer = BytesIO()
            pil_img.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            encoded_images.append(img_str)
        
        # Create response
        response = {
            "images": encoded_images,
            "prompt": text_prompt
        }
        
        return json.dumps(response), 'application/json', 200
    
    except Exception as e:
        return json.dumps({"error": str(e)}), 'application/json', 500