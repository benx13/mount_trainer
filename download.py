import os
from datasets import load_dataset, config, Dataset
from huggingface_hub import get_token
from torch.utils import data
import shutil
from PIL import Image
import requests
from io import BytesIO
import json

def configure_fast_downloads():
    """Configure Hugging Face for faster direct downloads."""
    # Set higher download speeds and concurrent downloads
    config.MAX_SHARD_SIZE = "1000GB"

    # Enable HF native transfer for better speed
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # Enable HF's fast transfer
    os.environ["USE_TORCH_DOWNLOAD"] = "0"  # Disable PyTorch backend
    os.environ["HF_HUB_DOWNLOAD_WORKERS"] = "8"  # Match CPU cores
    os.environ["DATASETS_MAX_WORKER_SIZE"] = "8"  # Match CPU cores
    os.environ["HF_DATASETS_OFFLINE"] = "0"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"  # Enable progress bars to monitor speed
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    # Optimize for high bandwidth
    os.environ["HF_HUB_DOWNLOAD_CHUNK_SIZE"] = "50MB"  # Smaller chunks for better parallelization
    os.environ["HF_HUB_DOWNLOAD_RETRY_TIMES"] = "5"
    os.environ["HF_DATASETS_IN_MEMORY_MAX_SIZE"] = "0"  # Disable memory caching
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "100"  # Increase timeout for large chunks

    # Set cache directory to local SSD for faster access
    os.environ["HF_HOME"] = "/tmp/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface/transformers"
    os.environ["HF_DATASETS_CACHE"] = "/tmp/huggingface/datasets"

def calculate_box_area_percentage(box, image_size):
    """
    Calculate the area of a bounding box as a percentage of the image area.
    
    Args:
        box: List of [x1, y1, x2, y2] coordinates
        image_size: Tuple of (width, height)
    
    Returns:
        float: Area percentage (0-100)
    """
    box_width = box[2] - box[0]
    box_height = box[3] - box[1]
    box_area = box_width * box_height
    
    image_area = image_size[0] * image_size[1]
    return (box_area / image_area) * 100

def load_and_save_shard(
    dataset_name="Harvard-Edge/Wake-Vision-Train-Large",
    split="train_large",
    target_images_per_shard=100,
    shard_id=0,
    relabel_json_path=None,
    confidence_threshold=0.55,
    min_box_area_percentage=5.0
):
    """
    Load and save a specific shard of the Wake Vision dataset to disk with optional relabeling.
    
    Args:
        dataset_name (str): Name of the dataset on HuggingFace
        split (str): Dataset split to use
        target_images_per_shard (int): Target number of images per shard
        shard_id (int): Which shard to process
        relabel_json_path (str, optional): Path to relabeling JSON file. If None, uses original labels
        confidence_threshold (float): Confidence threshold for person detection
        min_box_area_percentage (float): Minimum box area as percentage of image size
    """
    # Load relabeling data if path is provided
    relabel_data = {}
    use_relabeling = relabel_json_path is not None
    if use_relabeling:
        try:
            with open(relabel_json_path, 'r') as f:
                relabel_data = json.load(f)
            print(f"Loaded relabeling data from {relabel_json_path}")
        except FileNotFoundError:
            print(f"Warning: Relabeling file {relabel_json_path} not found. Using original labels.")
            use_relabeling = False

    # Get dataset size (total number of images in dataset)
    total_examples = 5760428
    
    # Calculate the number of shards needed
    num_shards = total_examples // target_images_per_shard
    if total_examples % target_images_per_shard != 0:
        num_shards += 1

    print(f"Dataset has {total_examples} examples. Using {num_shards} shards with ~{target_images_per_shard} images per shard.")
    print(f"Using {'relabeled' if use_relabeling else 'original'} labels")

    # Load the dataset in streaming mode
    token = get_token()
    if token is None:
        raise ValueError("Please login to Hugging Face using `huggingface-cli login` first")

    configure_fast_downloads()
    dataset = load_dataset(dataset_name, split=split, streaming=True, token=token)
    
    # Shard the dataset
    sharded_dataset = dataset.shard(num_shards=num_shards, index=shard_id)

    # Create a directory to save images
    output_dir = f"shard_{shard_id}_human_vs_nohuman"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/human', exist_ok=True)
    os.makedirs(f'{output_dir}/no-human', exist_ok=True)

    # Stats for reporting
    relabeled_count = 0
    total_count = 0
    flipped_labels = 0
    small_person_count = 0

    # Iterate over the shard and download images
    for i, item in enumerate(sharded_dataset):
        if i >= target_images_per_shard:
            break

        img = item['image']
        filename = item['filename']
        original_label = item['person']
        
        # Determine label based on whether we're using relabeling
        if use_relabeling and filename in relabel_data:
            relabeled_count += 1
            new_label = 0  # Default to no-human
            image_data = relabel_data[filename]
            if 'objects' in image_data:
                for obj in image_data['objects']:
                    if obj['label'] == 'person' and obj['confidence'] > confidence_threshold:
                        # Calculate box area percentage
                        area_percentage = calculate_box_area_percentage(obj['box'], img.size)
                        if area_percentage >= min_box_area_percentage:
                            new_label = 1
                            break
                        else:
                            small_person_count += 1
                            print(f"Person detected in {filename} but box too small ({area_percentage:.2f}% < {min_box_area_percentage}%)")
            
            if new_label != original_label:
                flipped_labels += 1
                print(f"Label flipped for {filename}: {original_label} -> {new_label}")
        else:
            # Use original label if not relabeling or no relabel data for this image
            new_label = original_label
            if use_relabeling:
                print(f"Warning: No relabeling data found for {filename}, using original label: {original_label}")

        # Save image to corresponding folder based on new label
        label_dir = 'human' if new_label == 1 else 'no-human'
        save_path = os.path.join(f"{output_dir}/{label_dir}", filename)
        img.save(save_path)

        total_count += 1
        if i % 10 == 0:
            print(f"Saved {i+1} images... (Last: {filename}, Label: {label_dir})")

    # Print final statistics
    print(f"\nProcessing complete for shard {shard_id}:")
    print(f"Total images processed: {total_count}")
    if use_relabeling:
        print(f"Images with relabeling data: {relabeled_count}")
        print(f"Labels flipped: {flipped_labels}")
        print(f"Small person detections ignored: {small_person_count}")
    print(f"Shard saved to {output_dir}")
    
    return output_dir

if __name__ == '__main__':
    # Example usage with relabeling
    load_and_save_shard(
        dataset_name="Harvard-Edge/Wake-Vision-Train-Large", 
        split="train_large", 
        target_images_per_shard=100, 
        shard_id=0,
        confidence_threshold=0.55,
        min_box_area_percentage=5.0
    )