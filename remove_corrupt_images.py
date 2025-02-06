import os
import cv2
from tqdm import tqdm
import multiprocessing as mp
import time
import imghdr

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
            
        # Fast header check
        if imghdr.what(image_path) is None:
            return image_path, True, "Invalid image format"
            
        # Quick image header read instead of full image load
        img = cv2.imread(image_path, cv2.IMREAD_HEADER_ONLY)
        if img is None:
            return image_path, True, "Invalid image header"
            
        return image_path, False, None
    except Exception as e:
        return image_path, True, str(e)

def collect_image_paths(directory):
    """
    Collect all image paths in the directory with basic filtering.
    """
    image_paths = []
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    
    for root, _, files in os.walk(directory):
        for filename in files:
            ext = os.path.splitext(filename.lower())[1]
            if ext in valid_extensions:
                full_path = os.path.join(root, filename)
                # Quick size check during collection
                try:
                    if os.path.getsize(full_path) >= 100:
                        image_paths.append(full_path)
                except OSError:
                    continue
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
    image_paths = collect_image_paths(directory)
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
            # Process images in larger chunks
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

                # Explicitly clear result to manage memory
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