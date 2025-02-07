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

def process_image_batch(args):
    """Process a batch of images"""
    image_paths, class_idx = args
    valid_samples = []
    skipped = 0
    
    for img_path in image_paths:
        try:
            img = cv2.imread(img_path, cv2.IMREAD_REDUCED_COLOR_2)
            if img is None or img.size == 0:
                skipped += 1
                continue
            valid_samples.append((img_path, class_idx))
        except Exception as e:
            skipped += 1
            continue
    
    return valid_samples, skipped, len(image_paths)

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

        logger.info(f"Found {len(self.classes)} classes")
        
        # Collect all images first
        all_images = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            class_images = [
                os.path.join(class_dir, fname)
                for fname in os.listdir(class_dir)
                if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            all_images.extend((img_path, self.class_to_idx[class_name]) for img_path in class_images)
        
        total_images = len(all_images)
        logger.info(f"Found total of {total_images} images")
        
        # Split images into batches for parallel processing
        batch_size = 1000  # Process 1000 images per batch
        batches = []
        for i in range(0, len(all_images), batch_size):
            batch_images = all_images[i:i + batch_size]
            paths, labels = zip(*batch_images)
            batches.append((list(paths), labels[0]))  # All images in batch have same label
        
        # Process batches in parallel
        self.samples = []
        total_skipped = 0
        processed_images = 0
        
        with tqdm(total=total_images, desc="Processing images", position=0) as pbar:
            with mp.Pool(processes=num_workers) as pool:
                for result in pool.imap_unordered(process_image_batch, batches):
                    valid_samples, skipped, total = result
                    self.samples.extend(valid_samples)
                    total_skipped += skipped
                    processed_images += (total - skipped)
                    
                    # Update progress
                    pbar.update(total)
                    pbar.set_postfix({
                        'valid': processed_images,
                        'skipped': total_skipped,
                        'total': total_images
                    })
        
        if total_skipped > 0:
            logger.info(f"Skipped {total_skipped} corrupt/empty images out of {total_images} total images")
        
        # Create LMDB cache
        logger.info("Creating LMDB cache...")
        env = lmdb.open(self.lmdb_path, map_size=1099511627776 * 2)
        
        # Cache images
        total_samples = len(self.samples)
        with env.begin(write=True) as txn:
            with tqdm(total=total_samples, desc="Caching images", position=0) as pbar:
                for idx, (img_path, class_idx) in enumerate(self.samples):
                    try:
                        img = cv2.imread(img_path)
                        success, buf = cv2.imencode('.jpg', img)
                        if success:
                            key = f"{idx}".encode()
                            txn.put(key, buf.tobytes())
                            pbar.update(1)
                            pbar.set_postfix({
                                'cached': idx + 1,
                                'remaining': total_samples - (idx + 1)
                            })
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
        
        # Collect all valid image paths
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            # Read images in binary mode for faster loading
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, filename)
                    self.samples.append((img_path, self.class_to_idx[class_name]))
        
        logger.info(f"Found {len(self.samples)} images across {len(self.classes)} classes")
        
        # Pre-load images into memory if dataset is small enough
        self.preloaded = {}
        total_size_gb = len(self.samples) * 0.5  # Estimate 0.5MB per image
        if total_size_gb < 32:  # If dataset would take less than 32GB RAM
            logger.info("Pre-loading images into memory...")
            for idx, (img_path, _) in enumerate(self.samples):
                try:
                    with open(img_path, 'rb') as f:
                        self.preloaded[idx] = f.read()
                except Exception as e:
                    logger.error(f"Error pre-loading {img_path}: {str(e)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            # Use pre-loaded image if available
            if idx in self.preloaded:
                img_binary = self.preloaded[idx]
                img_array = np.frombuffer(img_binary, dtype=np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            else:
                # Fast image loading
                image = cv2.imread(img_path)
            
            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed["image"]
            
            return image, label
            
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
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
