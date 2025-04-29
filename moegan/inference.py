# moegan/inference.py
import os
import json
import torch
import base64
from io import BytesIO
import numpy as np
from PIL import Image
import traceback
import sys
from moegan.t2i_moe_gan import AuroraGenerator, sample_aurora_gan
from sagemaker_inference.default_handler_service import DefaultHandlerService
from sagemaker_inference.transformer import Transformer
from sagemaker_inference import default_inference_handler

# Constants
LATENT_DIM = 512
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')

# Load model once when container starts
model = None

def load_model():
    """Load the model from the saved checkpoint"""
    try:
        global model
        if model is None:
            print(f"[LOAD_MODEL] Loading model from: {MODEL_PATH}")
            
            # List directory contents for debugging
            print(f"[LOAD_MODEL] MODEL_PATH contents: {os.listdir(MODEL_PATH)}")
            
            checkpoint_path = os.path.join(MODEL_PATH, 'aurora_model_final.pt')
            
            # Check if model file exists
            if os.path.exists(checkpoint_path):
                print(f"[LOAD_MODEL] Loading checkpoint from default path: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
                # Initialize with max_resolution=16 explicitly
                model = AuroraGenerator(max_resolution=16).to(DEVICE)
                
                # Load model weights
                if 'generator' in checkpoint:
                    print("[LOAD_MODEL] Loading weights from 'generator' key")
                    model.load_state_dict(checkpoint['generator'])
                else:
                    print("[LOAD_MODEL] Loading weights directly")
                    model.load_state_dict(checkpoint)
                    
                model.eval()
                print("[LOAD_MODEL] Model loaded successfully from default path")
            else:
                # Try to find any .pt file
                print("[LOAD_MODEL] Default model not found, searching for alternatives...")
                found_model = False
                for file in os.listdir(MODEL_PATH):
                    if file.endswith('.pt'):
                        alt_path = os.path.join(MODEL_PATH, file)
                        print(f"[LOAD_MODEL] Loading alternative model: {alt_path}")
                        checkpoint = torch.load(alt_path, map_location=DEVICE)
                        model = AuroraGenerator(max_resolution=16).to(DEVICE)
                        
                        # Load model weights
                        if 'generator' in checkpoint:
                            print("[LOAD_MODEL] Loading weights from 'generator' key")
                            model.load_state_dict(checkpoint['generator'])
                        else:
                            print("[LOAD_MODEL] Loading weights directly")
                            model.load_state_dict(checkpoint)
                        
                        model.eval()
                        found_model = True
                        print(f"[LOAD_MODEL] Alternative model loaded successfully from {alt_path}")
                        break
                
                if not found_model:
                    # Search subdirectories
                    print("[LOAD_MODEL] Searching subdirectories for model files...")
                    for root, dirs, files in os.walk(MODEL_PATH):
                        for file in files:
                            if file.endswith('.pt'):
                                alt_path = os.path.join(root, file)
                                print(f"[LOAD_MODEL] Found model in subdirectory: {alt_path}")
                                checkpoint = torch.load(alt_path, map_location=DEVICE)
                                model = AuroraGenerator(max_resolution=16).to(DEVICE)
                                
                                # Load model weights
                                if 'generator' in checkpoint:
                                    print("[LOAD_MODEL] Loading weights from 'generator' key")
                                    model.load_state_dict(checkpoint['generator'])
                                else:
                                    print("[LOAD_MODEL] Loading weights directly")
                                    model.load_state_dict(checkpoint)
                                
                                model.eval()
                                found_model = True
                                print(f"[LOAD_MODEL] Model from subdirectory loaded successfully")
                                break
                        if found_model:
                            break
                    
                    if not found_model:
                        print(f"[LOAD_MODEL] ERROR: No model file found in {MODEL_PATH} or subdirectories")
                        raise FileNotFoundError(f"No model file found in {MODEL_PATH}")
        
        # Verify model is not None
        if model is None:
            print("[LOAD_MODEL] ERROR: Model is still None after loading attempt")
            raise ValueError("Model is still None after loading attempt")
            
        print(f"[LOAD_MODEL] Returning model of type: {type(model)}")
        return model  # CRITICAL: Always return the model!
            
    except Exception as e:
        print(f"[LOAD_MODEL] ERROR loading model: {e}")
        traceback.print_exc(file=sys.stdout)
        raise e

# SageMaker inference handlers
def model_fn(model_dir):
    """
    Load model for SageMaker inference
    """
    global MODEL_PATH
    MODEL_PATH = model_dir
    print(f"[MODEL_FN] Called with model_dir: {model_dir}")
    
    # Load and return the model
    loaded_model = load_model()
    
    # Verify loaded model
    if loaded_model is None:
        print("[MODEL_FN] ERROR: model_fn failed - model is None")
        raise ValueError("Model loading failed - returned None")
        
    print(f"[MODEL_FN] Returning loaded model of type: {type(loaded_model)}")
    return loaded_model

def calculate_fid_for_inference(generated_images, device=DEVICE):
    """
    Calculate FID score for inference images against a reference dataset
    
    Args:
        generated_images: Tensor of generated images [batch_size, 3, height, width]
        
    Returns:
        fid_score: Fréchet Inception Distance score
    """
    import torch
    import numpy as np
    from scipy import linalg
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.models import inception_v3, Inception_V3_Weights
    
    # Load or create a reference statistics file if it doesn't exist
    reference_stats_path = os.path.join(MODEL_PATH, 'reference_stats.npz')
    
    # Define Inception model for feature extraction
    class InceptionExtractor(nn.Module):
        def __init__(self):
            super(InceptionExtractor, self).__init__()
            # Load pretrained model
            self.inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
            # Remove final classifier layer
            self.inception.fc = nn.Identity()
            # Set to evaluation mode
            self.inception.eval()
            
        def forward(self, x):
            # Resize images to inception input size
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
            # Extract features
            with torch.no_grad():
                features = self.inception(x)
            return features
    
    # Create and load model
    inception_model = InceptionExtractor().to(device)
    
    # Get activations for generated images
    def get_activations(images):
        # Ensure images are in [0, 1] range for Inception
        images = (images + 1) / 2  # Convert from [-1, 1] to [0, 1]
        images = torch.clamp(images, 0, 1)
        
        activations = []
        # Process in batches to avoid OOM
        batch_size = 8  # Adjust based on GPU memory
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            with torch.no_grad():
                batch_activations = inception_model(batch)
            activations.append(batch_activations.cpu().numpy())
        
        return np.vstack(activations)
    
    # Calculate statistics (mean and covariance)
    def calculate_statistics(activations):
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma
    
    # Calculate FID between two sets of statistics
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        diff = mu1 - mu2
        
        # Calculate sqrt of product between covariances
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            print("FID calculation produced singular product; adding %s to diagonal of cov estimates" % eps)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # Numerical error might give complex numbers
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    
    # Either load precomputed reference statistics or use defaults
    try:
        if os.path.exists(reference_stats_path):
            # Load precomputed statistics
            ref_data = np.load(reference_stats_path)
            real_mu = ref_data['mu']
            real_sigma = ref_data['sigma']
            print(f"Loaded reference statistics from {reference_stats_path}")
        else:
            # Use standard ImageNet statistics as fallback
            print("Reference statistics not found, using default values")
            # These are placeholder values - ideally you'd precompute these from your validation set
            real_mu = np.zeros(2048) 
            real_sigma = np.eye(2048)
    except Exception as e:
        print(f"Error loading reference statistics: {e}")
        # Fallback values
        real_mu = np.zeros(2048)
        real_sigma = np.eye(2048)
    
    # Get statistics for generated images
    gen_activations = get_activations(generated_images)
    gen_mu, gen_sigma = calculate_statistics(gen_activations)
    
    # Calculate FID
    fid_score = calculate_frechet_distance(real_mu, real_sigma, gen_mu, gen_sigma)
    
    return fid_score


def transform_fn(model, request_body, content_type, accept_type):
    """
    Transform function for SageMaker inference with FID score calculation
    """
    print(f"[TRANSFORM_FN] Called with content_type: {content_type}, accept_type: {accept_type}")
    
    # Verify model is not None
    if model is None:
        error_msg = "Model is None in transform_fn! Model loading failed."
        print(f"[TRANSFORM_FN] ERROR: {error_msg}")
        return json.dumps({"error": error_msg}), 'application/json', 500
    
    # Check model has expected attributes
    if not hasattr(model, 'eval') or not callable(getattr(model, 'eval')):
        error_msg = f"Model lacks eval() method. Model type: {type(model)}"
        print(f"[TRANSFORM_FN] ERROR: {error_msg}")
        return json.dumps({"error": error_msg}), 'application/json', 500
    
    # Ensure model is in eval mode
    try:
        model.eval()
        print("[TRANSFORM_FN] Model successfully set to eval mode")
    except Exception as e:
        error_msg = f"Error calling model.eval(): {str(e)}"
        print(f"[TRANSFORM_FN] ERROR: {error_msg}")
        traceback.print_exc(file=sys.stdout)
        return json.dumps({"error": error_msg}), 'application/json', 500
    
    if content_type != 'application/json':
        return json.dumps({"error": "Expected application/json content type"}), 'application/json', 415
    
    # Parse request
    try:
        request = json.loads(request_body)
        text_prompt = request.get('text', '')
        
        print(f"[TRANSFORM_FN] Request parsed successfully. Text prompt: {text_prompt[:50]}...")
        
        if not text_prompt:
            return json.dumps({"error": "Text prompt is required"}), 'application/json', 400
        
        # Number of images to generate (limit to 4 max)
        num_samples = min(request.get('num_samples', 1), 4)
        truncation_psi = request.get('truncation_psi', 0.7)
        
        # Option to calculate FID
        calculate_fid = request.get('calculate_fid', False)
        
        # Generate images
        print(f"[TRANSFORM_FN] Generating {num_samples} images with truncation_psi={truncation_psi}")
        images = sample_aurora_gan(
            model, 
            text_prompt, 
            num_samples=num_samples,
            truncation_psi=truncation_psi, 
            device=DEVICE
        )
        
        print(f"[TRANSFORM_FN] Images generated successfully. Shape: {images.shape}")
        
        # Calculate FID if requested
        fid_score = None
        if calculate_fid and num_samples >= 2:  # Need at least a few samples for meaningful FID
            try:
                print("[TRANSFORM_FN] Calculating FID score...")
                fid_score = calculate_fid_for_inference(images)
                print(f"[TRANSFORM_FN] FID Score: {fid_score}")
            except Exception as fid_error:
                print(f"[TRANSFORM_FN] Failed to calculate FID: {fid_error}")
                traceback.print_exc(file=sys.stdout)
        
        # Convert to base64 encoded PNGs
        print("[TRANSFORM_FN] Converting images to base64...")
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
        
        # Add FID score if calculated
        if fid_score is not None:
            response["fid_score"] = float(fid_score)
        
        print("[TRANSFORM_FN] Response created successfully")
        return json.dumps(response), 'application/json', 200
    
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        print(f"[TRANSFORM_FN] ERROR: {error_msg}")
        traceback.print_exc(file=sys.stdout)
        return json.dumps({"error": error_msg}), 'application/json', 500
    


# Define a simple Custom Handler using your functions
class CustomInferenceHandler(default_inference_handler.DefaultInferenceHandler):
    def model_fn(self, model_dir):
        print(f"[CUSTOM_HANDLER] model_fn called with model_dir: {model_dir}")
        loaded_model = model_fn(model_dir)
        print(f"[CUSTOM_HANDLER] model_fn returned model of type: {type(loaded_model)}")
        return loaded_model
    
    def input_fn(self, input_data, content_type):
        print(f"[CUSTOM_HANDLER] input_fn called with content_type: {content_type}")
        return input_data  # input is already JSON
    
    def predict_fn(self, input_object, model):
        print(f"[CUSTOM_HANDLER] predict_fn called with model of type: {type(model)}")
        
        # Verify model is not None
        if model is None:
            error_msg = "Model is None in predict_fn! Model loading failed."
            print(f"[CUSTOM_HANDLER] ERROR: {error_msg}")
            return json.dumps({"error": error_msg})
            
        # Need to call eval again here - important for MMS handling
        try:
            model.eval()
            print("[CUSTOM_HANDLER] Model successfully set to eval mode in predict_fn")
        except Exception as e:
            error_msg = f"Error calling model.eval() in predict_fn: {str(e)}"
            print(f"[CUSTOM_HANDLER] ERROR: {error_msg}")
            traceback.print_exc(file=sys.stdout)
            return json.dumps({"error": error_msg})
        
        return transform_fn(model, input_object, 'application/json', 'application/json')[0]

    def output_fn(self, prediction, accept):
        print(f"[CUSTOM_HANDLER] output_fn called with accept: {accept}")
        return prediction

# Create a transformer with our custom handler
transformer = Transformer(default_inference_handler=CustomInferenceHandler())

# Create a service that MMS expects
_service = DefaultHandlerService(transformer=transformer)

def handle(data, context):
    print(f"[HANDLE] Called with data type: {type(data)}")
    if data is None:
        print("[HANDLE] WARNING: Input data is None")
        return None
    if isinstance(data, (bytes, bytearray)):
        data = data.decode('utf-8')
        print(f"[HANDLE] Decoded data from bytes, length: {len(data)}")
    
    try:
        result = _service.handle(data, context)
        print(f"[HANDLE] Request handled successfully, result length: {len(result) if result else 0}")
        return result
    except Exception as e:
        error_msg = f"Error in handle function: {str(e)}"
        print(f"[HANDLE] ERROR: {error_msg}")
        traceback.print_exc(file=sys.stdout)
        return [json.dumps({"error": error_msg})], content_types.JSON, 500
