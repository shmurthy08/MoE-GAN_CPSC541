# MoE-GAN: Text-to-Image Generation with Mixture of Experts

A Mixture of Experts Generative Adversarial Network (MoE-GAN) for text-to-image generation using Bayesian Neural Networks. This project implements a novel architecture that combines Bayesian routing with GANs to create high-quality, text-conditional image generation.

## Table of Contents
- [Contributors](#contributors)
- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Directory Structure](#directory-structure)


## Contributors

- Shree Murthy
- Dylan Inafuku

## Project Overview

MoE-GAN uses a specialized architecture that integrates:
- Mixture of Experts (MoE) with Bayesian routing for adaptive feature generation
- Adversarial training with stability-enhancing regularization (R1 penalty)
- CLIP-based perceptual alignment for text-image matching
- Progressive training approach for multi-resolution image synthesis

The model is trained and evaluated on the MS-COCO dataset, generating 64×64 images from text descriptions.

## Architecture

The model consists of several key components:

- **Generator**: Aurora-style generator with MoE blocks that use Bayesian routing
- **Discriminator**: Text-conditional discriminator with matching awareness
- **Bayesian Router**: Probabilistic expert assignment using the reparameterization trick
- **Loss Functions**: Adversarial loss, CLIP loss, R1 gradient penalty, KL divergence, and MoE balance loss

## Directory Structure

- `moegan/`: Core model implementation
  - `t2i_moe_gan.py`: Model architecture definition
  - `train_model.py`: Local training script
  - `inference.py`: Inference code for generating images
  - `sagemaker_train.py`: SageMaker training script
- `data_processing/`: Data processing pipeline for MS-COCO
- `configs/`: Hyperparameter configurations
- `scripts/`: Utility scripts for training and evaluation
- `frontend/`: Web interface for image generation

## Installation

```bash
# Clone the repository
git clone https://github.com/username/MoE-GAN_CPSC541.git
cd MoE-GAN_CPSC541

# Install dependencies
pip install -r requirements.txt

# Optional: Install CUDA for GPU acceleration
# Follow instructions at https://developer.nvidia.com/cuda-downloads
```

## Usage

### Local Training

```bash
python moegan/train_model.py --config configs/hyperparameter_config.json
```

### AWS SageMaker Training

Configure AWS credentials and run:

```bash
python scripts/start_training_job.py
```

### Generating Images

```bash
python moegan/generate_images.py --prompt "A beautiful sunset over the ocean" --num-images 4
```

### Web Interface

Deploy the web interface:

```bash
aws s3 cp frontend/index.html s3://your-bucket-name/index.html --content-type "text/html"
```

## Results

The model achieves impressive results on text-to-image generation tasks, with:
- Diverse visual styles based on text prompts
- Expert specialization for different visual aspects
- Stable training through Bayesian uncertainty modeling

Sample generated images can be found in the `gan_model/gan_output/generated_images/` directory.

## Evaluation Metrics

- **FID Score**: Measures similarity between generated and real image distributions
- **CLIP Score**: Quantifies text-image alignment accuracy
- **Expert Utilization**: Measures load balancing across MoE experts

## References

- [MS-COCO Dataset](https://cocodataset.org/#download)
- [CLIP: Connecting Text and Images](https://openai.com/research/clip)
- [Bayesian Mixture of Experts](https://papers.nips.cc/paper_files/paper/1995/file/9da187a7a191431db943a9a5a6fec6f4-Paper.pdf)
- [Aurora: A Mixture of Experts GAN](https://arxiv.org/abs/2309.03904)
## License

This project is licensed under the MIT License - see the LICENSE file for details.

