import os
from datasets import load_dataset, config
from huggingface_hub import get_token

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

def load_wake_vision_dataset(dataset_name="Harvard-Edge/Wake-Vision-Train-Large", split="train_large", streaming=False, num_proc=8, num_shards=1, shard_id=0):
    """
    Loads the Wake Vision dataset from Hugging Face Datasets with optimized download settings.
    Adds functionality to shard the dataset.

    Args:
        dataset_name (str): Name of the dataset on Hugging Face Hub.
        split (str): Dataset split to load.
        streaming (bool): Whether to load the dataset in streaming mode.
        num_proc (int): Number of processes for data loading.
        num_shards (int): Total number of shards to divide the dataset into.
        shard_id (int):  Shard ID to load (0-indexed).

    Returns:
        datasets.Dataset: The loaded Hugging Face dataset shard.
    """
    configure_fast_downloads()

    token = get_token()
    if token is None:
        raise ValueError("Please login to Hugging Face using `huggingface-cli login` first")

    print(f"Starting dataset download for '{dataset_name}', split '{split}', shard {shard_id+1}/{num_shards} using optimized HF transfer...") # Shard ID is 0-indexed, so display shard_id+1 for user-friendliness
    print("This will download the entire dataset if streaming=False. Please be patient...")

    dataset = load_dataset(
        dataset_name,
        split=split,
        streaming=streaming,
        trust_remote_code=True,
        token=token,
        num_proc=num_proc,
        shard_id=shard_id,
        num_shards=num_shards
    )

    print(f"Download complete! Dataset size: {len(dataset)} samples (if not streaming) for shard {shard_id+1}/{num_shards}")
    return dataset

if __name__ == '__main__':
    # Example usage when running this script directly
    dataset = load_wake_vision_dataset(num_shards=4, shard_id=0) # Example to load the first shard out of 4
    print(dataset)
    dataset_shard_2 = load_wake_vision_dataset(num_shards=4, shard_id=1) # Example to load the second shard out of 4
    print(dataset_shard_2)
    dataset_shard_3 = load_wake_vision_dataset(num_shards=4, shard_id=2) # Example to load the third shard out of 4
    print(dataset_shard_3)
    dataset_shard_4 = load_wake_vision_dataset(num_shards=4, shard_id=3) # Example to load the fourth shard out of 4
    print(dataset_shard_4)
    dataset_full = load_wake_vision_dataset() # Example to load the full dataset without sharding
    print(dataset_full)