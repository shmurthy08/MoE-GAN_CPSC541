import os
import torch
import clip
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Import the MoE model
from moe_model import MixtureOfExperts, BayesianMoEGatingNetwork

# Constants
CLIP_MODEL_TYPE = "ViT-B/32"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIP_EMBEDDING_DIM = 512  # Dimension of CLIP text embeddings for ViT-B/32
HIDDEN_DIM = 256  # Hidden dimension for the MoE model
NUM_EXPERTS = 8  # Number of experts in the MoE model 

def load_clip_model():
    """
    Load the CLIP model for text embedding extraction.
    
    Returns:
        CLIP model and preprocessing function
    """
    print(f"Loading CLIP model: {CLIP_MODEL_TYPE}")
    model, preprocess = clip.load(CLIP_MODEL_TYPE, device=DEVICE)
    model.eval()
    return model, preprocess


def get_text_embedding(clip_model, text):
    """
    Extract CLIP text embedding for a given text prompt.
    
    Args:
        clip_model: CLIP model
        text: Text prompt
    
    Returns:
        CLIP text embedding tensor
    """
    with torch.no_grad():
        tokens = clip.tokenize([text]).to(DEVICE)
        embedding = clip_model.encode_text(tokens).float()
        print(f"Embedding shape: {embedding.shape}")
        
        # Resize if dimensions don't match
        if embedding.shape[1] != CLIP_EMBEDDING_DIM:
            print(f"Resizing embedding from {embedding.shape[1]} to {CLIP_EMBEDDING_DIM}")
            # Option 1: Use zero padding
            new_embedding = torch.zeros((1, CLIP_EMBEDDING_DIM), device=embedding.device)
            new_embedding[:, :embedding.shape[1]] = embedding
            return new_embedding
            
    return embedding

def create_moe_model(load_path=None):
    """
    Create and optionally load a MoE model.
    
    Args:
        load_path: Path to load model weights from (default: None)
    
    Returns:
        MoE model
    """
    model = MixtureOfExperts(CLIP_EMBEDDING_DIM, HIDDEN_DIM, NUM_EXPERTS)
    
    if load_path and os.path.exists(load_path):
        print(f"Loading model weights from {load_path}")
        model.load_state_dict(torch.load(load_path))
    
    model.to(DEVICE)
    model.eval()
    return model

def test_expert_selection(model, clip_model, text_prompts):
    """
    Test expert selection for a list of text prompts.
    
    Args:
        model: MoE model
        clip_model: CLIP model
        text_prompts: List of text prompts
    
    Returns:
        Dictionary of results
    """
    results = {}
    
    for prompt in text_prompts:
        print(f"\nProcessing prompt: \"{prompt}\"")
        
        # Get text embedding
        text_embedding = get_text_embedding(clip_model, prompt)
        
        # Get expert selection
        with torch.no_grad():
            expert_probs, selected_experts, uncertainty = model.forward(text_embedding)
            
            # Print results
            expert_probs = expert_probs[0].cpu().numpy()
            uncertainty = uncertainty[0].cpu().numpy()
            
            # Store results
            results[prompt] = {
                'expert_probs': expert_probs,
                'selected_experts': selected_experts,
                'uncertainty': uncertainty
            }
            
            # Print expert probabilities
            print("Expert probabilities:")
            for i, (prob, unc) in enumerate(zip(expert_probs, uncertainty)):
                expert_desc = model.expert_descriptions[i]
                print(f"  Expert {i} ({expert_desc}): {prob:.4f} (uncertainty: {unc:.4f})")
            
            # Print selected experts
            print("Selected experts:")
            for expert_idx in selected_experts:
                print(f"  Expert {expert_idx} ({model.expert_descriptions[expert_idx]})")
    
    return results

def visualize_expert_probabilities(results, save_path=None):
    """
    Visualize expert probabilities for each prompt.
    
    Args:
        results: Dictionary of results from test_expert_selection
        save_path: Path to save the visualization (default: None)
    """
    num_prompts = len(results)
    num_experts = len(next(iter(results.values()))['expert_probs'])
    
    # Create figure
    fig, axs = plt.subplots(num_prompts, 1, figsize=(10, 4 * num_prompts))
    if num_prompts == 1:
        axs = [axs]
    
    # Plot expert probabilities for each prompt
    for i, (prompt, result) in enumerate(results.items()):
        expert_probs = result['expert_probs']
        uncertainty = result['uncertainty']
        
        # Create x-axis labels
        x_labels = [f"Expert {j}\n({model.expert_descriptions[j]})" for j in range(num_experts)]
        
        # Plot probabilities with error bars
        axs[i].bar(range(num_experts), expert_probs, yerr=uncertainty, capsize=10, 
                   color='skyblue', edgecolor='darkblue', alpha=0.7)
        
        # Highlight selected experts
        for expert_idx in result['selected_experts']:
            axs[i].bar(expert_idx, expert_probs[expert_idx], color='green', edgecolor='darkgreen', alpha=0.7)
        
        # Add labels and title
        axs[i].set_xticks(range(num_experts))
        axs[i].set_xticklabels(x_labels, rotation=45, ha='right')
        axs[i].set_ylabel('Probability')
        axs[i].set_title(f'Expert Probabilities for: "{prompt}"')
        axs[i].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add probability values on top of bars
        for j, prob in enumerate(expert_probs):
            axs[i].text(j, prob + 0.02, f'{prob:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save or show the figure
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

def analyze_uncertainties(results):
    """
    Analyze uncertainties across different prompts.
    
    Args:
        results: Dictionary of results from test_expert_selection
    """
    print("\nUncertainty Analysis:")
    
    # Calculate average uncertainty for each expert
    expert_uncertainties = {}
    for expert_idx in range(NUM_EXPERTS):
        uncertainties = [result['uncertainty'][expert_idx] for result in results.values()]
        avg_uncertainty = np.mean(uncertainties)
        expert_uncertainties[expert_idx] = avg_uncertainty
        
        print(f"Expert {expert_idx} ({model.expert_descriptions[expert_idx]}):")
        print(f"  Average uncertainty: {avg_uncertainty:.4f}")
    
    # Find most and least uncertain prompts
    prompt_uncertainties = {}
    for prompt, result in results.items():
        avg_prompt_uncertainty = np.mean(result['uncertainty'])
        prompt_uncertainties[prompt] = avg_prompt_uncertainty
    
    most_uncertain_prompt = max(prompt_uncertainties.items(), key=lambda x: x[1])
    least_uncertain_prompt = min(prompt_uncertainties.items(), key=lambda x: x[1])
    
    print("\nPrompt Uncertainty:")
    print(f"Most uncertain prompt: \"{most_uncertain_prompt[0]}\" (uncertainty: {most_uncertain_prompt[1]:.4f})")
    print(f"Least uncertain prompt: \"{least_uncertain_prompt[0]}\" (uncertainty: {least_uncertain_prompt[1]:.4f})")

def main():
    """
    Main function to test the Bayesian MoE Gating Network.
    """
    parser = argparse.ArgumentParser(description='Test Bayesian MoE Gating Network')
    parser.add_argument('--model_path', type=str, default=None, 
                        help='Path to load model weights from')
    parser.add_argument('--save_vis', type=str, default='expert_probabilities.png',
                        help='Path to save visualization')
    args = parser.parse_args()
    
    # Load CLIP model
    clip_model, _ = load_clip_model()
    
    # Create MoE model
    global model  # Make model accessible to all functions
    model = create_moe_model(args.model_path)
    
    # Test prompts
    test_prompts = [
        "A beautiful mountain landscape with a lake",
        "A portrait of a young woman with blue eyes",
        "An abstract representation of happiness",
        "A busy city street at night with cars and neon signs",
        "A dog playing in a garden",
        "The concept of artificial intelligence",
        "A tropical beach at sunset"
    ]
    
    # Test expert selection
    results = test_expert_selection(model, clip_model, test_prompts)
    
    # Visualize expert probabilities
    visualize_expert_probabilities(results, args.save_vis)
    
    # Analyze uncertainties
    analyze_uncertainties(results)

if __name__ == "__main__":
    main()
