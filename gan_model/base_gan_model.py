# imports
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from tqdm import tqdm
import clip

# constants
BATCH_SIZE = 64
IMAGE_SIZE = 64
LATENT_DIM = 100
TEXT_EMBEDDING_DIM = 512
NUM_EPOCHS = 50
LEARNING_RATE = 0.0002
BETA1 = 0.5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# path for model output
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gan_output")

# path to store modelprocessed images
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_processing", "processed_data")

# output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# dataset class for preprocessed MS-COCO data
class ProcessedMSCOCODataset(Dataset):
    """MS-COCO dataset with preprocessed images and CLIP text embeddings."""
    
    def __init__(self, images_file, text_embeddings_file, transform=None):
        """
        Args:
            images_file (string): Path to the numpy file with images
            text_embeddings_file (string): Path to the numpy file with text embeddings
            transform (callable, optional): Optional transform to be applied on images
        """
        self.images = np.load(images_file)
        self.text_embeddings = np.load(text_embeddings_file)
        self.transform = transform
        
        assert len(self.images) == len(self.text_embeddings), "Images and text embeddings count mismatch"
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx])
        text_embedding = torch.from_numpy(self.text_embeddings[idx])
        
        if self.transform:
            image = self.transform(image)
        
        return image, text_embedding

# generator network
class Generator(nn.Module):
    def __init__(self, latent_dim, text_embedding_dim):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.text_embedding_dim = text_embedding_dim
        
        # text embedding projection
        self.text_projection = nn.Sequential(
            nn.Linear(text_embedding_dim, 128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # input (noise + text embedding)
        self.combined_dim = latent_dim + 128
        
        # generator architecture
        self.main = nn.Sequential(
            # input: Z + text embedding
            nn.ConvTranspose2d(self.combined_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 512 x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 256 x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 128 x 16 x 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 64 x 32 x 32
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # 3 x 64 x 64
        )
        
    def forward(self, z, text_embedding):
        # project text embedding
        text_projected = self.text_projection(text_embedding)
        
        # noise and text embedding
        combined = torch.cat([z, text_projected], 1)
        
        # reshape
        combined = combined.view(-1, self.combined_dim, 1, 1)
        
        # generate image
        return self.main(combined)

# discriminator network
class Discriminator(nn.Module):
    def __init__(self, text_embedding_dim):
        super(Discriminator, self).__init__()
        
        self.text_embedding_dim = text_embedding_dim
        
        # text embedding projection
        self.text_projection = nn.Sequential(
            nn.Linear(text_embedding_dim, 128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # image processing layers
        self.image_conv = nn.Sequential(
            # input: 3 x 64 x 64
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 4 x 32 x 32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 16 x 16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 x 8 x 8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 512 x 4 x 4
        )
        
        # output layers after combining image features + text embedding
        self.output = nn.Sequential(
            nn.Conv2d(512 + 128, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, image, text_embedding):
        # image features
        image_features = self.image_conv(image)
        
        # project text embedding
        text_projected = self.text_projection(text_embedding)
        
        # replicate text embedding to match image feature map size
        text_projected = text_projected.view(-1, 128, 1, 1).repeat(1, 1, 4, 4)
        
        # concat image features and text embedding
        combined = torch.cat([image_features, text_projected], 1)
        
        # generate output
        return self.output(combined).view(-1, 1).squeeze(1)

# initialize weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# load a pretrained CLIP model for inference
def load_clip_model(model_type="ViT-B/32"):
    """
    Load the CLIP model for text encoding during inference
    """
    model, preprocess = clip.load(model_type, device=DEVICE)
    model.eval()
    return model, preprocess

# generate images given a text description
def generate_image_from_text(text_description, generator, clip_model=None, num_images=1):
    """
    Generate images based on a text description
    
    Args:
        text_description: Text prompt to generate images from
        generator: Trained generator model
        clip_model: CLIP model for text encoding (if None, one will be loaded)
        num_images: Number of images to generate
    """
    # load CLIP model if not provided
    if clip_model is None:
        clip_model, _ = load_clip_model()
    
    # encode text using CLIP
    with torch.no_grad():
        text = clip.tokenize([text_description]).to(DEVICE)
        text_features = clip_model.encode_text(text).float()
    
    # replicate text embedding for multiple images
    text_embeddings = text_features.repeat(num_images, 1)
    
    # generate random noise
    noise = torch.randn(num_images, LATENT_DIM, device=DEVICE)
    
    # generate images
    with torch.no_grad():
        generated_images = generator(noise, text_embeddings).detach().cpu()
    
    # convert to grid
    grid = vutils.make_grid(generated_images, padding=2, normalize=True)
    
    # output directory for generated images
    os.makedirs(os.path.join(OUTPUT_DIR, "generated_images"), exist_ok=True)
    
    # generate a filename based
    filename = text_description.replace(" ", "_").replace(".", "").lower()[:50]
    output_path = os.path.join(OUTPUT_DIR, "generated_images", f"{filename}.png")
    
    # save grid image
    vutils.save_image(grid, output_path, normalize=True)
    print(f"Saved image to {output_path}")
    
    # display image
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(f"Generated: {text_description}")
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.savefig(os.path.join(OUTPUT_DIR, "generated_images", f"{filename}_display.png"))
    plt.close()
    
    return generated_images

# training, allowing for subsetted data and epoch/batch adjustments (hardware limitations)
def train_with_limited_resources(batch_size=32, num_epochs=10, subset_size=1000):
    """
    Train the GAN model with limited resources
    
    Args:
        batch_size: Batch size for training
        num_epochs: Number of epochs to train
        subset_size: Number of samples to use from the dataset
    """
    # init models
    netG = Generator(LATENT_DIM, TEXT_EMBEDDING_DIM).to(DEVICE)
    netD = Discriminator(TEXT_EMBEDDING_DIM).to(DEVICE)
    
    # weight initialization
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    # optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    
    # loss function
    criterion = nn.BCELoss()
    
    # dataset and dataloader (uses augmented data)
    use_augmented_data = True
    
    if use_augmented_data:
        train_images_file = os.path.join(DATA_DIR, "mscoco_train_augmented_images.npy")
        train_embeddings_file = os.path.join(DATA_DIR, "mscoco_train_augmented_text_embeddings.npy")
    else:
        train_images_file = os.path.join(DATA_DIR, "mscoco_train_images.npy")
        train_embeddings_file = os.path.join(DATA_DIR, "mscoco_train_text_embeddings.npy")
    
    # create dataset
    dataset_full = ProcessedMSCOCODataset(
        images_file=train_images_file,
        text_embeddings_file=train_embeddings_file,
        transform=None
    )
    
    # smaller subset of the dataset for training
    indices = torch.randperm(len(dataset_full))[:subset_size]
    subset_dataset = torch.utils.data.Subset(dataset_full, indices)
    
    # dataloader
    dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # print training config
    print(f"Training with reduced settings:")
    print(f"- Epochs: {num_epochs}")
    print(f"- Batch size: {batch_size}")
    print(f"- Dataset size: {subset_size} samples")
    print(f"- Workers: 2")
    
    # lists to keep track of progress
    G_losses = []
    D_losses = []
    img_list = []
    fixed_noise = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
    
    # logging
    print("Starting Training Loop...")
    
    for epoch in range(num_epochs):
        for i, data in enumerate(tqdm(dataloader)):
            # (1) D network: maximize log(D(x)) + log(1 - D(G(z)))
            # Train with real batch
            netD.zero_grad()
            real_images = data[0].to(DEVICE)
            text_embeddings = data[1].to(DEVICE)
            batch_size_actual = real_images.size(0)
            
            label = torch.full((batch_size_actual,), 1, dtype=torch.float, device=DEVICE)
            output = netD(real_images, text_embeddings)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
            
            # train with fake batch
            noise = torch.randn(batch_size_actual, LATENT_DIM, device=DEVICE)
            fake = netG(noise, text_embeddings)
            label.fill_(0)
            output = netD(fake.detach(), text_embeddings)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()
            
            # (2)  G network: maximize log(D(G(z)))
            netG.zero_grad()
            # fake labels are real for generator cost
            label.fill_(1) 
            output = netD(fake, text_embeddings)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            
            # save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            
            # check how the generator is doing by saving G's output on fixed_noise
            if (i % 50 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    # use the current batch text embeddings
                    sample_text_embeddings = text_embeddings[:min(batch_size, text_embeddings.size(0))]
                    # match fixed_noise size to sample_text_embeddings
                    sample_fixed_noise = fixed_noise[:sample_text_embeddings.size(0)]
                    fake = netG(sample_fixed_noise, sample_text_embeddings).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                
                # print statistics
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                      f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                      f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}')
        
        # save models after each epoch
        torch.save(netG.state_dict(), os.path.join(OUTPUT_DIR, f'generator_epoch_{epoch}.pth'))
        torch.save(netD.state_dict(), os.path.join(OUTPUT_DIR, f'discriminator_epoch_{epoch}.pth'))
                
    # plot training results
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'loss_plot.png'))
    
    # plot generated images
    if len(img_list) > 0:
        fig = plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Generated Images")
        plt.imshow(np.transpose(img_list[-1], (1,2,0)))
        plt.savefig(os.path.join(OUTPUT_DIR, 'generated_images.png'))
    
    return netG, netD, G_losses, D_losses, img_list

if __name__ == "__main__":
    # train the model with reduced resources
    netG, netD, G_losses, D_losses, img_list = train_with_limited_resources(
        batch_size=32,
        num_epochs=10,
        subset_size=10000
    )
    
    # generate sample images after training
    clip_model, _ = load_clip_model()
    
    test_prompts = [
        "a cat sitting on a couch",
        "a beautiful sunset over the ocean",
        "a plate of delicious food",
        "a person riding a bicycle"
    ]
    
    print("Generating sample images from text prompts...")
    for prompt in test_prompts:
        print(f"Generating image for: '{prompt}'")
        # generate and save multiple images for each prompt
        generate_image_from_text(prompt, netG, clip_model, num_images=2)