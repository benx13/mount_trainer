import os
import shutil
import pandas as pd
from pathlib import Path

def create_directory(directory):
    """Create directory if it doesn't exist"""
    os.makedirs(directory, exist_ok=True)

def process_folder(source_folder, dest_folder, quality_files):
    """Process images from source folder and move matching ones to destination folder"""
    # Create destination folder
    create_directory(dest_folder)
    
    # Get list of all files in source folder
    source_files = os.listdir(source_folder)
    
    # Counter for moved files
    moved_count = 0
    
    # Process each file
    for filename in source_files:
        if filename in quality_files:
            source_path = os.path.join(source_folder, filename)
            dest_path = os.path.join(dest_folder, filename)
            
            # Move the file
            shutil.copy2(source_path, dest_path)
            moved_count += 1
    
    return moved_count

def main():
    # Define paths
    quality_csv = "quality.csv"  # CSV file containing quality image filenames
    source_base = "full_dataset_human_vs_nohuman_relabeled/"  # Base directory containing human and nohuman folders
    dest_base = "quality_dataset"  # Base directory for quality dataset
    
    # Read quality CSV file
    df = pd.read_csv(quality_csv)
    quality_files = set(df['filename'].values)  # Convert to set for faster lookup
    
    # Process human folder
    source_human = os.path.join(source_base, "human")
    dest_human = os.path.join(dest_base, "human")
    human_count = process_folder(source_human, dest_human, quality_files)
    
    # Process nohuman folder
    source_nohuman = os.path.join(source_base, "no-human")
    dest_nohuman = os.path.join(dest_base, "no-human")
    nohuman_count = process_folder(source_nohuman, dest_nohuman, quality_files)
    
    # Print summary
    print(f"Processed files:")
    print(f"Human: {human_count} files moved")
    print(f"No Human: {nohuman_count} files moved")
    print(f"Total: {human_count + nohuman_count} files moved")

if __name__ == "__main__":
    main()

