import os
from datasets import load_dataset, config, Dataset
from huggingface_hub import get_token
from torch.utils import data
import shutil
from PIL import Image
import requests
from io import BytesIO
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def load_and_save_shard(
    dataset_name,
    split,
    target_images_per_shard,
    shard_id=0,
    relabel_json_path=None,
    confidence_threshold=0.55,
    min_box_area=5000,  # Now this is absolute area in pixels
    dual_save=False
):
    """
    Load and save dataset with flexible saving options:
    - No relabel_json_path: Save original labels only
    - relabel_json_path provided: Save relabeled version only
    - relabel_json_path and dual_save=True: Save both versions
    
    Args:
        dataset_name (str): Name of the dataset on Hugging Face
        split (str): Split to use
        target_images_per_shard (int): Target number of images per shard
        shard_id (int): ID of the shard to process
        relabel_json_path (str, optional): Path to JSON file containing relabeling data
        confidence_threshold (float): Confidence threshold for detections
        min_box_area (float): Minimum box area in pixels
        dual_save (bool): If True and relabel_json_path is provided, save both versions
    """
    # Configure download settings
    configure_fast_downloads()
    
    # Load dataset
    logger.info(f"Loading dataset: {dataset_name}, split: {split}")
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    
    # Determine save mode
    save_original = not relabel_json_path or dual_save
    save_relabeled = relabel_json_path is not None
    
    # Set up directory names
    base_dir = f"shard_{shard_id}_human_vs_nohuman"
    if save_relabeled and not save_original:
        # Only saving relabeled version
        output_dir = f"{base_dir}_relabeled"
    elif save_original and save_relabeled:
        # Saving both versions
        original_dir = base_dir
        relabeled_dir = f"{base_dir}_relabeled"
    else:
        # Only saving original version
        output_dir = base_dir
    
    # Create directories
    if save_original and save_relabeled:
        for label in ['human', 'no-human']:
            os.makedirs(os.path.join(original_dir, label), exist_ok=True)
            os.makedirs(os.path.join(relabeled_dir, label), exist_ok=True)
    else:
        for label in ['human', 'no-human']:
            os.makedirs(os.path.join(output_dir, label), exist_ok=True)
    
    # Load relabeling data if needed
    relabel_data = {}
    if save_relabeled:
        if not os.path.exists(relabel_json_path):
            raise FileNotFoundError(f"Relabel JSON file not found: {relabel_json_path}")
        with open(relabel_json_path, 'r') as f:
            relabel_data = json.load(f)
        logger.info(f"Loaded relabeling data from {relabel_json_path}")
    
    # Statistics counters
    stats = {
        'processed_images': 0,
        'relabeled_images': 0,
        'flipped_labels': 0,
        'small_detections_ignored': 0,
        'missing_relabel_data': 0
    }
    
    # Process images
    for i, example in enumerate(dataset):
        if i < shard_id * target_images_per_shard:
            continue
        if i >= (shard_id + 1) * target_images_per_shard:
            break
            
        image = example['image']
        original_label = 'human' if example['label'] == 1 else 'no-human'
        image_id = example.get('image_id', str(i))
        
        # Save original version if needed
        if save_original:
            save_path = os.path.join(original_dir if save_relabeled else output_dir, 
                                   original_label, f"{image_id}.jpg")
            image.save(save_path)
        
        # Process and save relabeled version if needed
        if save_relabeled:
            if image_id in relabel_data:
                stats['relabeled_images'] += 1
                detections = relabel_data[image_id]
                
                # Check if any detection meets criteria
                valid_detection = False
                for det in detections:
                    confidence = det.get('c', 0)  # Using 'c' for confidence
                    area = det.get('b', 0)  # Using 'b' for box area
                    
                    if confidence >= confidence_threshold:
                        if area >= min_box_area:
                            valid_detection = True
                            break
                        else:
                            stats['small_detections_ignored'] += 1
                
                # Set label based on valid detections
                relabeled_label = 'human' if valid_detection else 'no-human'
                if relabeled_label != original_label:
                    stats['flipped_labels'] += 1
            else:
                stats['missing_relabel_data'] += 1
                relabeled_label = original_label
            
            # Save relabeled version
            save_path = os.path.join(relabeled_dir if save_original else output_dir,
                                   relabeled_label, f"{image_id}.jpg")
            image.save(save_path)
        
        stats['processed_images'] += 1
        if stats['processed_images'] % 100 == 0:
            logger.info(f"Processed {stats['processed_images']} images...")
    
    # Log statistics
    logger.info("\nDataset Processing Statistics:")
    logger.info(f"Total images processed: {stats['processed_images']}")
    if save_relabeled:
        logger.info(f"Images with relabel data: {stats['relabeled_images']}")
        logger.info(f"Labels flipped: {stats['flipped_labels']}")
        logger.info(f"Small detections ignored: {stats['small_detections_ignored']}")
        logger.info(f"Images missing relabel data: {stats['missing_relabel_data']}")
    
    # Return appropriate paths based on save mode
    if save_original and save_relabeled:
        logger.info(f"\nSaved both versions:")
        logger.info(f"Original labels: {original_dir}")
        logger.info(f"Relabeled version: {relabeled_dir}")
        return original_dir, relabeled_dir
    else:
        logger.info(f"\nSaved to: {output_dir}")
        return output_dir

if __name__ == '__main__':
    # Example usage with relabeling
    load_and_save_shard(
        dataset_name="Harvard-Edge/Wake-Vision-Train-Large", 
        split="train_large", 
        target_images_per_shard=100, 
        shard_id=0,
        confidence_threshold=0.55,
        min_box_area=5000,
        relabel_json_path='/Users/benx13/code/edge_ai_modelcentric/results_new/results/shard_0/images_0-5000.json',
        dual_save=True
    )