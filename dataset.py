import os
import cv2
import torch
from torch.utils.data import Dataset
import logging
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
import numpy as np
import pickle
import lmdb
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_class_dir(args):
    """Process all images in a class directory"""
    class_dir, class_idx = args
    valid_samples = []
    skipped = 0
    total = 0
    
    for filename in os.scandir(class_dir):
        if filename.name.lower().endswith(('.png', '.jpg', '.jpeg')):
            total += 1
            img_path = filename.path
            try:
                # Validate image
                img = cv2.imread(img_path)
                if img is None or img.size == 0:
                    skipped += 1
                    continue
                
                # Store image path and class
                valid_samples.append((img_path, class_idx))
                
            except Exception as e:
                logger.warning(f"Error processing {img_path}: {str(e)}")
                skipped += 1
                continue
    
    return valid_samples, skipped, total

class CachedImageDataset(Dataset):
    """Memory-efficient dataset with LMDB caching for large-scale datasets"""
    def __init__(self, root_dir, transform=None, skip_corrupt=True, num_workers=None, cache_dir=None):
        self.root_dir = root_dir
        self.transform = transform
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(root_dir), '.cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Cache paths
        self.meta_path = os.path.join(self.cache_dir, 'dataset_meta.pkl')
        self.lmdb_path = os.path.join(self.cache_dir, 'images.lmdb')
        
        if num_workers is None:
            # Use more workers for dataset initialization
            num_workers = min(128, os.cpu_count() or 1)
        else:
            # If num_workers is specified, use it directly
            num_workers = min(num_workers, os.cpu_count() or 1)
        
        logger.info(f"Initializing dataset with {num_workers} workers")
        
        # Initialize dataset
        self._initialize_dataset(num_workers, skip_corrupt)
        
        # Initialize LMDB environment
        self.env = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            map_size=1099511627776 * 2  # 2TB map size
        )
        
    def _initialize_dataset(self, num_workers, skip_corrupt):
        """Initialize the dataset and create cache if needed"""
        self.classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Check if cache exists
        if os.path.exists(self.meta_path) and os.path.exists(self.lmdb_path):
            logger.info("Loading cached dataset metadata...")
            with open(self.meta_path, 'rb') as f:
                cached_data = pickle.load(f)
                self.samples = cached_data['samples']
            logger.info(f"Loaded {len(self.samples)} samples from cache")
            return

        # Prepare arguments for parallel processing
        class_dirs = [
            (os.path.join(self.root_dir, class_name), self.class_to_idx[class_name])
            for class_name in self.classes
            if os.path.isdir(os.path.join(self.root_dir, class_name))
        ]
        
        # Process class directories in parallel
        self.samples = []
        total_skipped = 0
        total_images = 0
        
        logger.info(f"Scanning dataset using {num_workers} workers...")
        with mp.Pool(processes=num_workers) as pool:
            results = []
            for result in tqdm(
                pool.imap_unordered(process_class_dir, class_dirs),
                total=len(class_dirs),
                desc="Loading dataset"
            ):
                valid_samples, skipped, total = result
                self.samples.extend(valid_samples)
                total_skipped += skipped
                total_images += total
        
        if total_skipped > 0:
            logger.info(f"Skipped {total_skipped} corrupt/empty images out of {total_images} total images")
        
        # Create LMDB cache
        logger.info("Creating LMDB cache...")
        env = lmdb.open(self.lmdb_path, map_size=1099511627776 * 2)
        
        with env.begin(write=True) as txn:
            for idx, (img_path, class_idx) in enumerate(tqdm(self.samples, desc="Caching images")):
                try:
                    # Read and encode image
                    img = cv2.imread(img_path)
                    success, buf = cv2.imencode('.jpg', img)
                    if not success:
                        continue
                    
                    # Store in LMDB
                    key = f"{idx}".encode()
                    txn.put(key, buf.tobytes())
                except Exception as e:
                    logger.warning(f"Error caching {img_path}: {str(e)}")
                    continue
        
        # Save metadata
        with open(self.meta_path, 'wb') as f:
            pickle.dump({
                'samples': self.samples,
                'classes': self.classes,
                'class_to_idx': self.class_to_idx
            }, f)
        
        logger.info(f"Successfully cached {len(self.samples)} valid images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            # Read image from LMDB
            with self.env.begin(write=False) as txn:
                key = f"{idx}".encode()
                imgbuf = txn.get(key)
                
            # Decode image
            buf = np.frombuffer(imgbuf, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                transformed = self.transform(image=img)
                img = transformed["image"]
            
            return img, label
            
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
            return self.__getitem__((idx + 1) % len(self))

class AlbumentationsDataset(Dataset):
    """Custom dataset that uses Albumentations for augmentation"""
    def __init__(self, root_dir, transform=None, skip_corrupt=True, num_workers=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        if num_workers is None:
            num_workers = min(128, os.cpu_count() or 1)  # Use up to 128 cores
        
        # Prepare arguments for parallel processing
        class_dirs = [
            (os.path.join(root_dir, class_name), self.class_to_idx[class_name])
            for class_name in self.classes
            if os.path.isdir(os.path.join(root_dir, class_name))
        ]
        
        # Process class directories in parallel
        self.samples = []
        total_skipped = 0
        total_images = 0
        
        logger.info(f"Scanning dataset using {num_workers} workers...")
        with mp.Pool(processes=num_workers) as pool:
            results = []
            for result in tqdm(
                pool.imap_unordered(process_class_dir, class_dirs),
                total=len(class_dirs),
                desc="Loading dataset"
            ):
                valid_samples, skipped, total = result
                self.samples.extend(valid_samples)
                total_skipped += skipped
                total_images += total
        
        if total_skipped > 0:
            logger.info(f"Skipped {total_skipped} corrupt/empty images out of {total_images} total images")
        
        logger.info(f"Successfully loaded {len(self.samples)} valid images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            # Read image using cv2 with reduced color first for speed
            image = cv2.imread(img_path, cv2.IMREAD_REDUCED_COLOR_2)
            if image is None:
                # Try full read if reduced read fails
                image = cv2.imread(img_path)
                if image is None:
                    raise ValueError("Image is empty or corrupt")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed["image"]
            
            return image, label
            
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
            # Return a different sample as fallback
            return self.__getitem__((idx + 1) % len(self))

class TransformWrapper(Dataset):
    """Wrapper dataset that applies Albumentations transforms to an existing dataset"""
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        try:
            image, label = self.dataset[idx]
            if isinstance(image, torch.Tensor):
                image = image.numpy().transpose(1, 2, 0)  # Convert to numpy for Albumentations
            transformed = self.transform(image=image)
            return transformed["image"], label
        except Exception as e:
            logger.error(f"Error in TransformWrapper at index {idx}: {str(e)}")
            # Return a different sample as fallback
            return self.__getitem__((idx + 1) % len(self))
    
    @property
    def classes(self):
        """Pass through dataset classes if available"""
        if hasattr(self.dataset, 'classes'):
            return self.dataset.classes
        elif hasattr(self.dataset, 'dataset') and hasattr(self.dataset.dataset, 'classes'):
            return self.dataset.dataset.classes
        return None
