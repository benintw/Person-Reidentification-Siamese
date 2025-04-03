#!/usr/bin/env python
"""
Prepare the Market-1501 dataset for training, validation, and testing.

This script processes the Market-1501 dataset and creates the necessary
structure for the person re-identification model.

Usage:
    python scripts/prepare_data.py --data_dir data/market1501
"""

import argparse
import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare Market-1501 dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/market1501",
        help="Path to Market-1501 dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Ratio of data to use for training (default: 0.7)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Ratio of data to use for validation (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def create_dataframe(data_dir):
    """Create a DataFrame with image paths and labels.
    
    Args:
        data_dir: Path to Market-1501 dataset directory
        
    Returns:
        DataFrame with columns: path, person_id, camera_id
    """
    # Check if the dataset has the expected structure
    bounding_box_train = os.path.join(data_dir, "bounding_box_train")
    bounding_box_test = os.path.join(data_dir, "bounding_box_test")
    query = os.path.join(data_dir, "query")
    
    if not all(os.path.exists(p) for p in [bounding_box_train, bounding_box_test, query]):
        raise ValueError(
            f"Dataset at {data_dir} does not have the expected Market-1501 structure. "
            "Expected directories: bounding_box_train, bounding_box_test, query"
        )
    
    # Get all image files
    train_files = [os.path.join("bounding_box_train", f) for f in os.listdir(bounding_box_train) 
                  if f.endswith(".jpg") and not f.startswith(".")]
    test_files = [os.path.join("bounding_box_test", f) for f in os.listdir(bounding_box_test) 
                 if f.endswith(".jpg") and not f.startswith(".")]
    query_files = [os.path.join("query", f) for f in os.listdir(query) 
                  if f.endswith(".jpg") and not f.startswith(".")]
    
    all_files = train_files + test_files + query_files
    
    # Parse file names to extract person ID and camera ID
    # Market-1501 format: 0002_c1s1_000451_03.jpg
    # where 0002 is the person ID and c1 is the camera ID
    data = []
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        if file_name.startswith("-") or file_name.startswith("0000"):
            # Skip distractors and junk images
            continue
        
        parts = file_name.split("_")
        person_id = int(parts[0])
        camera_id = int(parts[1][1])
        
        data.append({
            "path": file_path,
            "person_id": person_id,
            "camera_id": camera_id,
            "is_train": "bounding_box_train" in file_path,
            "is_query": "query" in file_path,
            "is_gallery": "bounding_box_test" in file_path and "query" not in file_path,
        })
    
    return pd.DataFrame(data)


def split_data(df, train_ratio=0.7, val_ratio=0.2, seed=42):
    """Split data into train, validation, and test sets.
    
    Args:
        df: DataFrame with image data
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation
        seed: Random seed for reproducibility
        
    Returns:
        train_df, val_df, test_df: DataFrames for each split
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # If the dataset already has train/test splits, use those
    if "is_train" in df.columns:
        train_val_df = df[df["is_train"]].copy()
        test_df = df[~df["is_train"]].copy()
        
        # Split train into train and validation
        person_ids = train_val_df["person_id"].unique()
        np.random.shuffle(person_ids)
        
        val_size = int(len(person_ids) * val_ratio / (train_ratio + val_ratio))
        val_person_ids = person_ids[:val_size]
        train_person_ids = person_ids[val_size:]
        
        train_df = train_val_df[train_val_df["person_id"].isin(train_person_ids)].copy()
        val_df = train_val_df[train_val_df["person_id"].isin(val_person_ids)].copy()
    else:
        # If no predefined splits, create them from scratch
        person_ids = df["person_id"].unique()
        np.random.shuffle(person_ids)
        
        train_size = int(len(person_ids) * train_ratio)
        val_size = int(len(person_ids) * val_ratio)
        
        train_person_ids = person_ids[:train_size]
        val_person_ids = person_ids[train_size:train_size + val_size]
        test_person_ids = person_ids[train_size + val_size:]
        
        train_df = df[df["person_id"].isin(train_person_ids)].copy()
        val_df = df[df["person_id"].isin(val_person_ids)].copy()
        test_df = df[df["person_id"].isin(test_person_ids)].copy()
    
    # Create split column for easier identification
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    
    return train_df, val_df, test_df


def save_splits(train_df, val_df, test_df, output_dir):
    """Save the data splits to CSV files.
    
    Args:
        train_df: Training data DataFrame
        val_df: Validation data DataFrame
        test_df: Test data DataFrame
        output_dir: Directory to save the CSV files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    # Create a combined CSV with all data
    combined_df = pd.concat([train_df, val_df, test_df])
    combined_df.to_csv(os.path.join(output_dir, "all.csv"), index=False)
    
    print(f"Saved data splits to {output_dir}:")
    print(f"  Train: {len(train_df)} images, {train_df['person_id'].nunique()} identities")
    print(f"  Validation: {len(val_df)} images, {val_df['person_id'].nunique()} identities")
    print(f"  Test: {len(test_df)} images, {test_df['person_id'].nunique()} identities")


def main():
    """Main function to prepare the dataset."""
    args = parse_args()
    
    print(f"Preparing Market-1501 dataset from {args.data_dir}")
    
    # Create DataFrame from dataset
    df = create_dataframe(args.data_dir)
    print(f"Found {len(df)} images with {df['person_id'].nunique()} unique identities")
    
    # Split data into train, validation, and test sets
    train_df, val_df, test_df = split_data(
        df, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed
    )
    
    # Save splits to CSV files
    save_splits(train_df, val_df, test_df, args.output_dir)
    
    print("Dataset preparation complete!")


if __name__ == "__main__":
    main()
