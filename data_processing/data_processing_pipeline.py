# imports
import os
import json
import numpy as np
import torch
import argparse
from torch.utils.data import Dataset
from torchvision import transforms
import clip
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import pickle
import fiftyone as fo
import fiftyone.core.expressions as foe
import fiftyone.zoo as foz
import torchvision.utils as vutils

# constants
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_data")
BATCH_SIZE = 64
IMAGE_SIZE = 64
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CLIP_MODEL_TYPE = "ViT-B/32"
MAX_SAMPLES = 5000

# creates output directory if doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# load CLIP model
print(f"Loading CLIP model: {CLIP_MODEL_TYPE}")
clip_model, clip_preprocess = clip.load(CLIP_MODEL_TYPE, device=DEVICE)
clip_model.eval() 

# transformation to preprocess images
image_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def download_coco_with_fiftyone(split="train", max_samples=MAX_SAMPLES):
    """
    Downloads MS-COCO dataset using FiftyOne and returns a dataset view 
    containing images with captions
    """
    if max_samples is None:
        print(f"Downloading ALL MS-COCO {split} split with FiftyOne (no sample limit)")
    else:
        print(f"Downloading MS-COCO {split} split with FiftyOne (max_samples={max_samples})")
    
    # generate a unique dataset name to avoid conflicts
    import uuid
    dataset_name = f"coco-{split}-subset-{uuid.uuid4().hex[:8]}"
    
    # download COCO dataset with proper label types
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split=split,
        label_types=["detections"],  
        max_samples=max_samples,  
        dataset_name=dataset_name
    )
    
    print(f"Dataset contains {len(dataset)} images")
    return dataset

def load_coco_captions(split="train"):
    """
    Load COCO captions from the annotations file, downloading if necessary
    """
    import urllib.request
    import zipfile
    import os
    import shutil
    
    # adjusted file paths for the correct file names
    split_name = "val" if split == "validation" else split
    annotations_dir = os.path.join(os.path.expanduser("~"), "fiftyone", "coco-2017", "raw")
    captions_file = os.path.join(annotations_dir, f"captions_{split_name}2017.json")
    
    print(f"Looking for captions at {captions_file}")
    if not os.path.exists(captions_file):
        print(f"Captions file not found: {captions_file}")
        
        # create annotations directory
        os.makedirs(annotations_dir, exist_ok=True)
        
        # download annotations zip
        annotations_zip = os.path.join(annotations_dir, "annotations_trainval2017.zip")
        if not os.path.exists(annotations_zip):
            print("Downloading COCO annotations...")
            annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
            urllib.request.urlretrieve(annotations_url, annotations_zip)
        
        # extract annotations
        print("Extracting COCO annotations...")
        with zipfile.ZipFile(annotations_zip, 'r') as zip_ref:
            # extract the specific caption file
            extract_name = f"annotations/captions_{split_name}2017.json"
            for filename in zip_ref.namelist():
                if filename == extract_name:
                    # extract to a temporary file
                    zip_ref.extract(filename, annotations_dir)
                    # move to the target location
                    src_file = os.path.join(annotations_dir, filename)
                    os.rename(src_file, captions_file)
                    print(f"Extracted {filename} to {captions_file}")
                    break
            
            # clean up the annotations directory
            annotations_extract_dir = os.path.join(annotations_dir, "annotations")
            if os.path.exists(annotations_extract_dir):
                try:
                    os.rmdir(annotations_extract_dir)  
                except:
                    pass  
    
    # check if file is present
    if not os.path.exists(captions_file):
        raise FileNotFoundError(f"Could not find or download captions file: {captions_file}")
    
    print(f"Loading captions from {captions_file}")
    # load captions file
    with open(captions_file, 'r') as f:
        annotations = json.load(f)
    
    # create a mapping from image ID to captions
    image_id_to_captions = {}
    for annotation in annotations['annotations']:
        img_id = annotation['image_id']
        caption = annotation['caption']
        
        if img_id not in image_id_to_captions:
            image_id_to_captions[img_id] = []
        image_id_to_captions[img_id].append(caption)
    
    # create image ID to filename mapping
    image_id_to_filename = {}
    for img in annotations['images']:
        image_id_to_filename[img['id']] = img['file_name']
    
    return image_id_to_captions, image_id_to_filename

def extract_clip_text_embeddings(captions_list):
    """
    Extract CLIP text embeddings for a list of captions
    """
    embeddings = []
    
    # processed in batches
    batch_size = 256
    num_batches = (len(captions_list) + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc="Extracting text embeddings"):
        batch_captions = captions_list[i * batch_size:(i + 1) * batch_size]
        
        with torch.no_grad():
            text_tokens = clip.tokenize(batch_captions).to(DEVICE)
            batch_embeddings = clip_model.encode_text(text_tokens).float().cpu().numpy()
            embeddings.append(batch_embeddings)
    
    return np.vstack(embeddings)

def process_dataset_with_fiftyone(split="train", max_samples=MAX_SAMPLES):
    """
    Process the MS-COCO dataset from FiftyOne and save images with CLIP embeddings
    """
    # download dataset for images
    dataset_view = download_coco_with_fiftyone(split, max_samples)
    
    # load captions separately
    image_id_to_captions, image_id_to_filename = load_coco_captions(split)
    
    # output files for images, text embeddings, and metadata
    images_output_file = os.path.join(OUTPUT_DIR, f"mscoco_{split}_images.npy")
    text_embeddings_output_file = os.path.join(OUTPUT_DIR, f"mscoco_{split}_text_embeddings.npy")
    metadata_output_file = os.path.join(OUTPUT_DIR, f"mscoco_{split}_metadata.pkl")
    
    # procress images/captions
    all_images = []
    all_captions = []
    metadata = []
    
    print(f"Processing {len(dataset_view)} images from {split} split")
    
    for sample in tqdm(dataset_view, desc=f"Processing {split} images"):
        try:
            # extract the image filename and ID
            filename = os.path.basename(sample.filepath)
            
            # find the corresponding image ID
            img_id = None
            for id, fname in image_id_to_filename.items():
                if fname == filename:
                    img_id = id
                    break
            
            # skip if no image ID or it doesn't have captions
            if img_id is None or img_id not in image_id_to_captions:
                continue
                
            # get captions for this image
            captions = image_id_to_captions[img_id]
            if len(captions) == 0:
                continue
                
            # load and transform image
            img_path = sample.filepath
            image = Image.open(img_path).convert('RGB')
            transformed_image = image_transform(image).unsqueeze(0)
            
            # use the first caption
            caption = captions[0]
            
            # store image information
            all_images.append(transformed_image.numpy())
            all_captions.append(caption)
            metadata.append({
                'image_id': img_id,
                'file_name': filename,
                'caption': caption,
                'all_captions': captions
            })
        except Exception as e:
            print(f"Error processing image {sample.id}: {e}")
    
    # check if valid images with captions found
    if len(all_images) == 0:
        print(f"WARNING: No images with matching captions found for {split} split")
        if split == "validation":
            print("Returning empty validation set")
            return np.array([]), np.array([]), []
        else:
            raise ValueError(f"No images with captions found for {split} split. Cannot continue.")
    
    # list to array
    print(f"Found {len(all_images)} images with captions")
    all_images = np.vstack(all_images)
    
    # extract CLIP text embeddings
    print("Extracting CLIP text embeddings...")
    all_embeddings = extract_clip_text_embeddings(all_captions)
    
    # save processed data
    print(f"Saving {len(all_images)} processed images to {images_output_file}")
    np.save(images_output_file, all_images)
    
    print(f"Saving {len(all_embeddings)} text embeddings to {text_embeddings_output_file}")
    np.save(text_embeddings_output_file, all_embeddings)
    
    print(f"Saving metadata to {metadata_output_file}")
    with open(metadata_output_file, 'wb') as f:
        pickle.dump(metadata, f)
        
    captions_output_file = os.path.join(OUTPUT_DIR, f"mscoco_{split}_captions.npy")
    np.save(captions_output_file, np.array(all_captions))
    np.save(images_output_file, all_images)
    with open(metadata_output_file, 'wb') as f:
        pickle.dump(metadata, f)
    
    return all_images, all_embeddings, metadata

def create_augmentations(images, text_embeddings, metadata, augmentation_factor=2):
    """
    Create augmented versions of the dataset
    """
    augment_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # initialize arrays for augmented data
    num_samples = len(images)
    aug_images = []
    aug_text_embeddings = []
    aug_metadata = []
    
    print(f"Creating {augmentation_factor}x augmentations for {num_samples} samples")
    for i in tqdm(range(num_samples), desc="Creating augmentations"):
        # original image
        image_tensor = torch.from_numpy(images[i])
        
        # create augmentations for each image
        for j in range(augmentation_factor):
            # apply augmentation
            aug_image = augment_transform(image_tensor).unsqueeze(0).numpy()
            
            # store augmented data
            aug_images.append(aug_image)
            aug_text_embeddings.append(text_embeddings[i])
            
            # update metadata
            new_metadata = metadata[i].copy()
            new_metadata['augmentation_id'] = j
            aug_metadata.append(new_metadata)
    
    # combine original and augmented
    combined_images = np.vstack([images] + [np.vstack(aug_images)])
    combined_text_embeddings = np.vstack([text_embeddings] + [np.vstack(aug_text_embeddings)])
    combined_metadata = metadata + aug_metadata
    
    print(f"Total samples after augmentation: {len(combined_images)}")
    
    # save augmented data
    aug_images_output_file = os.path.join(OUTPUT_DIR, "mscoco_train_augmented_images.npy")
    aug_embeddings_output_file = os.path.join(OUTPUT_DIR, "mscoco_train_augmented_text_embeddings.npy")
    aug_metadata_output_file = os.path.join(OUTPUT_DIR, "mscoco_train_augmented_metadata.pkl")
    
    print(f"Saving augmented images to {aug_images_output_file}")
    np.save(aug_images_output_file, combined_images)
    
    print(f"Saving augmented text embeddings to {aug_embeddings_output_file}")
    np.save(aug_embeddings_output_file, combined_text_embeddings)
    
    print(f"Saving augmented metadata to {aug_metadata_output_file}")
    with open(aug_metadata_output_file, 'wb') as f:
        pickle.dump(combined_metadata, f)
    
    return combined_images, combined_text_embeddings, combined_metadata

def visualize_dataset(images, metadata, num_samples=5):
    """
    Visualize a few samples from the dataset
    """
    # randomly select samples
    indices = np.random.choice(len(images), num_samples, replace=False)
    
    # figure
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    for i, idx in enumerate(indices):
        # get image + caption
        img = images[idx]
        # converts from (C,H,W) to (H,W,C)
        img = np.transpose(img, (1, 2, 0))  
        # denormalizez
        img = img * 0.5 + 0.5 
        caption = metadata[idx]['caption']
        
        # display
        axes[i].imshow(img)
        axes[i].set_title(caption[:30] + '...' if len(caption) > 30 else caption, fontsize=8)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dataset_visualization.png'))
    plt.close()
    
    # grid visualization
    grid_image_path = os.path.join(OUTPUT_DIR, 'dataset_grid.png')
    print(f"Saving grid visualization to {grid_image_path}")
    
    sample_images = torch.from_numpy(images[indices])
    grid = vutils.make_grid(sample_images, nrow=num_samples, padding=2, normalize=True)
    vutils.save_image(grid, grid_image_path)

def analyze_dataset(metadata):
    """
    Perform analysis on the dataset to understand its distribution
    """
    # caption lengths
    caption_lengths = [len(m['caption'].split()) for m in metadata]
    
    # histogram of caption lengths
    plt.figure(figsize=(10, 5))
    plt.hist(caption_lengths, bins=30)
    plt.title('Distribution of Caption Lengths')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(OUTPUT_DIR, 'caption_length_distribution.png'))
    plt.close()
    
    # frequent words in captions
    all_words = []
    for m in metadata:
        all_words.extend(m['caption'].lower().split())
    
    # word frequency counts
    word_freq = {}
    for word in all_words:
        if word not in word_freq:
            word_freq[word] = 0
        word_freq[word] += 1
    
    # top 20 words
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
    
    # plot
    plt.figure(figsize=(12, 6))
    plt.bar([w[0] for w in top_words], [w[1] for w in top_words])
    plt.title('Top 20 Words in Captions')
    plt.xlabel('Word')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'top_words_distribution.png'))
    plt.close()
    
    # statistics
    print("\nDataset Statistics:")
    print(f"Total number of samples: {len(metadata)}")
    print(f"Average caption length: {np.mean(caption_lengths):.2f} words")
    print(f"Min caption length: {min(caption_lengths)} words")
    print(f"Max caption length: {max(caption_lengths)} words")
    
    # statistics to file
    stats = {
        'total_samples': len(metadata),
        'avg_caption_length': float(np.mean(caption_lengths)),
        'min_caption_length': min(caption_lengths),
        'max_caption_length': max(caption_lengths),
        'top_words': top_words
    }
    
    with open(os.path.join(OUTPUT_DIR, 'dataset_statistics.json'), 'w') as f:
        json.dump(stats, f, indent=4)

class ProcessedMSCOCODataset(Dataset):
    """MS-COCO dataset with preprocessed images, CLIP text embeddings, and original captions."""
    
    def __init__(self, images_file, text_embeddings_file, captions_file=None, metadata_file=None, transform=None):
        """
        Args:
            images_file (string): Path to the numpy file with images
            text_embeddings_file (string): Path to the numpy file with text embeddings
            captions_file (string, optional): Path to the numpy file with captions
            metadata_file (string, optional): Path to the pickle file with metadata
            transform (callable, optional): Optional transform to be applied on images
        """
        self.images = np.load(images_file)
        self.text_embeddings = np.load(text_embeddings_file)
        self.transform = transform
        
        # Load captions if available
        self.captions = None
        if captions_file and os.path.exists(captions_file):
            self.captions = np.load(captions_file, allow_pickle=True)
        
        # Load metadata if available
        self.metadata = None
        if metadata_file and os.path.exists(metadata_file):
            with open(metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
        
        assert len(self.images) == len(self.text_embeddings), "Images and text embeddings count mismatch"
        if self.captions is not None:
            assert len(self.images) == len(self.captions), "Images and captions count mismatch"
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx])
        text_embedding = torch.from_numpy(self.text_embeddings[idx])
        
        if self.transform:
            image = self.transform(image)
        
        # Return captions if available, otherwise just images and embeddings
        if self.captions is not None:
            caption = self.captions[idx]
            return image, text_embedding, caption
        else:
            return image, text_embedding
        
# Main function to run the entire pipeline
def run_pipeline(max_samples=MAX_SAMPLES, create_augmentations_flag=True, augmentation_factor=2):
    """
    Run the complete data processing pipeline
    
    Args:
        max_samples: Maximum number of samples to process (None for all)
        create_augmentations_flag: Whether to create augmented versions of the dataset
        augmentation_factor: Number of augmented versions to create for each sample
    """
    if max_samples is None:
        print("Starting MS-COCO data processing pipeline with FiftyOne (processing ALL samples)")
    else:
        print(f"Starting MS-COCO data processing pipeline with FiftyOne (max_samples={max_samples})")
    
    # process training data
    train_images, train_embeddings, train_metadata = process_dataset_with_fiftyone(
        split="train", 
        max_samples=max_samples
    )
    
    # visualize training data
    visualize_dataset(train_images, train_metadata, num_samples=5)
    
    # analyze training data
    analyze_dataset(train_metadata)
    
    # create augmentations if needed
    if create_augmentations_flag:
        train_augmented_images, train_augmented_embeddings, train_augmented_metadata = create_augmentations(
            train_images, 
            train_embeddings, 
            train_metadata,
            augmentation_factor=augmentation_factor
        )
    
    # if max_samples is None, use None for validation too
    # else use 1/5th or at least 1000
    val_max_samples = None if max_samples is None else max(1000, max_samples // 5)
    
    # process validation data
    val_images, val_embeddings, val_metadata = process_dataset_with_fiftyone(
        split="validation", 
        max_samples=val_max_samples
    )
    
    print("Data processing pipeline completed successfully!")
    
    # return statistics about the processed data
    stats = {
        'train_samples': len(train_images),
        'augmented_samples': len(train_augmented_images) if create_augmentations_flag else 0,
        'val_samples': val_max_samples,
        'image_size': IMAGE_SIZE,
        'embedding_dim': train_embeddings.shape[1]
    }
    
    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process MS-COCO dataset for GAN training')
    
    # add argument for maximum samples
    parser.add_argument('--max_samples', type=int, default=5000, 
                        help='Maximum number of samples to process (default: 5000, use -1 for all)')
    
    # add argument for augmentation
    parser.add_argument('--no_augmentation', action='store_true', 
                        help='Disable data augmentation')
    
    # add argument for augmentation factor
    parser.add_argument('--aug_factor', type=int, default=2, 
                        help='Augmentation factor (default: 2)')
    
    # parse arguments
    args = parser.parse_args()
    
    # convert -1 to None for processing all samples
    max_samples = None if args.max_samples == -1 else args.max_samples
    
    # run the pipeline
    stats = run_pipeline(
        max_samples=max_samples,
        create_augmentations_flag=not args.no_augmentation,
        augmentation_factor=args.aug_factor
    )
    
    print("\nPipeline completed with the following statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")