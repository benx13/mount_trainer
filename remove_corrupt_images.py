import os
import cv2
from tqdm import tqdm
import multiprocessing as mp
import time
from pathlib import Path

def check_image(image_path):
    """
    Check if an image is corrupt using fast validation.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        tuple: (image_path, is_corrupt, error_message)
    """
    try:
        # Quick file size check first
        if os.path.getsize(image_path) < 100:  # Skip tiny files
            return image_path, True, "File too small"
            
        # Read image with reduced size for speed
        img = cv2.imread(image_path, cv2.IMREAD_REDUCED_COLOR_2)  # 1/2 resolution
        if img is None:
            # Double check with full read in case of false positive
            img = cv2.imread(image_path)
            if img is None:
                return image_path, True, "Cannot read image"
            
        return image_path, False, None
    except Exception as e:
        return image_path, True, str(e)

def collect_images_from_subdir(args):
    """
    Collect image paths from a subdirectory.
    """
    subdir, valid_extensions = args
    image_paths = []
    
    try:
        for entry in os.scandir(subdir):
            if entry.is_file():
                ext = os.path.splitext(entry.name.lower())[1]
                if ext in valid_extensions:
                    try:
                        if os.path.getsize(entry.path) >= 100:
                            image_paths.append(entry.path)
                    except OSError:
                        continue
    except Exception:
        pass
    
    return image_paths

def collect_image_paths(directory, num_workers):
    """
    Collect all image paths in the directory with parallel processing.
    """
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    
    # Get all subdirectories including the root
    subdirs = [directory]
    for root, dirs, _ in os.walk(directory):
        subdirs.extend(os.path.join(root, d) for d in dirs)
    
    print(f"Scanning {len(subdirs)} directories...")
    
    # Process subdirectories in parallel
    args_list = [(subdir, valid_extensions) for subdir in subdirs]
    
    image_paths = []
    with mp.Pool(processes=num_workers) as pool:
        with tqdm(total=len(subdirs), desc="Collecting paths") as pbar:
            for result in pool.imap_unordered(collect_images_from_subdir, args_list):
                image_paths.extend(result)
                pbar.update(1)
    
    return image_paths

def remove_corrupt_images(directory, num_workers=None):
    """
    Parallel scan through a directory and remove corrupt images.
    """
    if not num_workers:
        num_workers = max(1, mp.cpu_count() - 1)  # Leave one CPU free

    print(f"Using {num_workers} workers")
    start_time = time.time()

    print("Collecting image paths...")
    image_paths = collect_image_paths(directory, num_workers)
    total_images = len(image_paths)
    print(f"Found {total_images} images to process")

    if total_images == 0:
        print("No images found in directory")
        return

    corrupt_images = 0
    # Larger chunk size for better performance
    chunk_size = max(100, min(2000, total_images // num_workers))

    print("\nScanning for corrupt images...")
    
    with mp.Pool(processes=num_workers) as pool:
        with tqdm(total=total_images, desc="Processing") as pbar:
            for result in pool.imap_unordered(check_image, image_paths, chunksize=chunk_size):
                pbar.update(1)
                image_path, is_corrupt, error = result
                
                if is_corrupt:
                    try:
                        os.remove(image_path)
                        corrupt_images += 1
                        rel_path = os.path.relpath(image_path, directory)
                        print(f"\nRemoved corrupt image: {rel_path} ({error})")
                    except OSError as e:
                        print(f"\nError removing {image_path}: {e}")

                del result

    elapsed_time = time.time() - start_time
    print("\nScan complete!")
    print(f"Total images processed: {total_images}")
    print(f"Corrupt images removed: {corrupt_images}")
    print(f"Time taken: {elapsed_time:.2f} seconds")
    print(f"Processing speed: {total_images / elapsed_time:.2f} images/second")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Remove corrupt images from a directory")
    parser.add_argument("--dir", type=str, required=True,
                      help="Directory containing images to check")
    parser.add_argument("--workers", type=int, default=None,
                      help="Number of worker processes (default: CPU count)")
    
    args = parser.parse_args()
    
    remove_corrupt_images(args.dir, args.workers) 