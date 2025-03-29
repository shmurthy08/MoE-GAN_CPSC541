# generate_images.py
import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from t2i_moe_gan import AuroraGenerator, sample_aurora_gan

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser(description='Generate images with Aurora GAN-MoE')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--prompt', type=str, required=True,
                        help='Text prompt for image generation')
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='./generated_images',
                        help='Directory to save generated images')
    parser.add_argument('--truncation_psi', type=float, default=0.7,
                        help='Truncation psi parameter (lower = better quality, less diversity)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    
    # Initialize generator
    generator = AuroraGenerator().to(DEVICE)
    
    # Load generator state
    if 'generator' in checkpoint:
        generator.load_state_dict(checkpoint['generator'])
    else:
        generator.load_state_dict(checkpoint)
    
    # Generate images
    images = sample_aurora_gan(
        generator, args.prompt, num_samples=args.num_samples,
        truncation_psi=args.truncation_psi, device=DEVICE
    )
    
    # Convert images to numpy for visualization
    images_np = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
    images_np = (images_np + 1) / 2  # [-1, 1] -> [0, 1]
    images_np = np.clip(images_np, 0, 1)
    
    # Create figure
    fig, axes = plt.subplots(1, args.num_samples, figsize=(4*args.num_samples, 4))
    if args.num_samples == 1:
        axes = [axes]
    
    # Plot images
    for i, (ax, img) in enumerate(zip(axes, images_np)):
        ax.imshow(img)
        ax.set_title(f"Sample {i+1}")
        ax.axis('off')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"{args.prompt.replace(' ', '_')}.png"))
    plt.close()
    
    print(f"Generated {args.num_samples} images for prompt: '{args.prompt}'")
    print(f"Saved to {os.path.join(args.output_dir, f'{args.prompt.replace(' ', '_')}.png')}")

if __name__ == '__main__':
    main()