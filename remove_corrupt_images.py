import os
import cv2
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

def check_image(image_path):
    """
    Check if an image is corrupt.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        tuple: (image_path, is_corrupt, error_message)
    """
    try:
        img = cv2.imread(image_path)
        if img is None or img.size == 0:
            return image_path, True, "Image is None or empty"
        return image_path, False, None
    except Exception as e:
        return image_path, True, str(e)

def collect_image_paths(directory):
    """
    Collect all image paths in the directory.
    
    Args:
        directory (str): Directory to scan
        
    Returns:
        list: List of image file paths
    """
    image_paths = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_paths.append(os.path.join(root, filename))
    return image_paths

def remove_corrupt_images(directory, num_workers=None):
    """
    Parallel scan through a directory and remove any images that can't be opened with OpenCV.
    
    Args:
        directory (str): Path to the directory containing images
        num_workers (int, optional): Number of worker processes. Defaults to CPU count.
    """
    if not num_workers:
        num_workers = mp.cpu_count()

    print(f"Using {num_workers} workers")
    start_time = time.time()

    # Collect all image paths first
    print("Collecting image paths...")
    image_paths = collect_image_paths(directory)
    total_images = len(image_paths)
    print(f"Found {total_images} images to process")

    # Process images in parallel
    corrupt_images = 0
    chunk_size = max(1, min(1000, total_images // (num_workers * 4)))  # Optimize chunk size

    print("\nScanning for corrupt images...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Create progress bar for completed tasks
        with tqdm(total=total_images, desc="Processing") as pbar:
            # Submit all tasks
            future_to_path = {
                executor.submit(check_image, path): path 
                for path in image_paths
            }
            
            # Process completed tasks
            for future in as_completed(future_to_path):
                pbar.update(1)
                image_path, is_corrupt, error = future.result()
                
                if is_corrupt:
                    try:
                        os.remove(image_path)
                        corrupt_images += 1
                        rel_path = os.path.relpath(image_path, directory)
                        print(f"\nRemoved corrupt image: {rel_path} ({error})")
                    except OSError as e:
                        print(f"\nError removing {image_path}: {e}")

    # Print summary with timing information
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