#!/usr/bin/env python
"""
Explore prediction results from the person re-identification model.

This script loads saved prediction results (.npz file) and displays their contents
without requiring the actual images.

Usage:
    python scripts/explore_predictions.py --predictions results/predictions_YYYYMMDD_HHMMSS.npz
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.metrics import evaluate_predictions


def load_predictions(predictions_file):
    """Load prediction data from NPZ file.

    Args:
        predictions_file: Path to the .npz predictions file

    Returns:
        dict: Prediction data
    """
    # Load the predictions file
    predictions_data = np.load(predictions_file)

    # Print all available arrays in the file
    print("\nArrays in the predictions file:")
    for key in predictions_data.files:
        arr = predictions_data[key]
        print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")

    # Create a dictionary with the data
    predictions = {}
    for key in predictions_data.files:
        predictions[key] = predictions_data[key]

    # Try to load image paths from JSON file if it exists
    json_path = Path(predictions_file).stem + "_paths.json"
    json_path = Path(predictions_file).parent / json_path

    if json_path.exists():
        with open(json_path, "r") as f:
            paths_data = json.load(f)
            predictions.update(paths_data)
    else:
        # Create placeholder paths
        print(f"\nWarning: Could not find paths file at {json_path}")
        print("Creating placeholder paths")
        predictions["query_img_paths"] = [
            f"query_{i}" for i in range(len(predictions["query_embeddings"]))
        ]
        predictions["gallery_img_paths"] = [
            f"gallery_{i}" for i in range(len(predictions["gallery_embeddings"]))
        ]

    # Close the file
    predictions_data.close()

    return predictions


def print_basic_stats(predictions):
    """Print basic statistics about the predictions.

    Args:
        predictions: Dictionary containing prediction data
    """
    print("\n===== Basic Statistics =====")
    print(f"Number of queries: {len(predictions['query_embeddings'])}")
    print(f"Number of gallery images: {len(predictions['gallery_embeddings'])}")
    print(f"Embedding dimension: {predictions['query_embeddings'].shape[1]}")

    # Count unique person IDs
    unique_query_ids = len(np.unique(predictions["query_labels"]))
    unique_gallery_ids = len(np.unique(predictions["gallery_labels"]))
    print(f"Unique person IDs in query set: {unique_query_ids}")
    print(f"Unique person IDs in gallery set: {unique_gallery_ids}")

    # Distance statistics
    distances = predictions["distances"]
    print(f"\nDistance statistics:")
    print(f"  Min distance: {np.min(distances):.4f}")
    print(f"  Max distance: {np.max(distances):.4f}")
    print(f"  Mean distance: {np.mean(distances):.4f}")
    print(f"  Median distance: {np.median(distances):.4f}")


def analyze_performance(predictions):
    """Analyze performance metrics in detail.

    Args:
        predictions: Dictionary containing prediction data
    """
    print("\n===== Performance Analysis =====")

    # Calculate standard metrics
    metrics = evaluate_predictions(predictions)
    print(f"Mean Average Precision (mAP): {metrics['mAP']:.4f}")
    print(f"Rank-1 Accuracy: {metrics['rank1']:.4f}")
    print(f"Rank-5 Accuracy: {metrics['rank5']:.4f}")
    print(f"Rank-10 Accuracy: {metrics['rank10']:.4f}")


def analyze_query(predictions, query_idx, no_of_closest=10):
    """Analyze a specific query in detail without requiring images.

    Args:
        predictions: Dictionary containing prediction data
        query_idx: Index of the query to analyze
        no_of_closest: Number of closest matches to display
    """
    print(f"\n===== Analyzing Query {query_idx} =====")

    # Get query information
    query_path = predictions["query_img_paths"][query_idx]
    query_label = predictions["query_labels"][query_idx]
    query_embedding = predictions["query_embeddings"][query_idx]

    print(f"Query image: {query_path}")
    print(f"Person ID: {query_label}")
    print(
        f"Embedding stats: min={query_embedding.min():.4f}, max={query_embedding.max():.4f}, mean={query_embedding.mean():.4f}"
    )

    # Get gallery information
    gallery_paths = predictions["gallery_img_paths"]
    gallery_labels = predictions["gallery_labels"]
    rankings = predictions["rankings"][query_idx]
    distances = predictions["distances"][query_idx]

    # Get the closest matches
    closest_idx = rankings[:no_of_closest]
    closest_paths = [gallery_paths[i] for i in closest_idx]
    closest_labels = gallery_labels[closest_idx]
    closest_distances = distances[closest_idx]

    # Print details about closest matches
    print(f"\nTop {no_of_closest} matches:")
    correct_count = 0
    for i, (idx, path, label, dist) in enumerate(
        zip(closest_idx, closest_paths, closest_labels, closest_distances)
    ):
        is_correct = label == query_label
        if is_correct:
            correct_count += 1
        print(
            f"  {i+1}. {path} (distance: {dist:.4f}) - {'u2713' if is_correct else 'u2717'} [ID: {label}]"
        )

    # Calculate precision@k
    precision_at_k = correct_count / no_of_closest
    print(f"\nPrecision@{no_of_closest}: {precision_at_k:.4f}")

    # Find all matches for this person in the gallery
    matching_gallery_indices = np.where(gallery_labels == query_label)[0]
    print(f"\nAll matching gallery images for person {query_label}:")
    for i, idx in enumerate(matching_gallery_indices):
        rank_position = np.where(rankings == idx)[0][0] if idx in rankings else -1
        print(
            f"  {i+1}. {gallery_paths[idx]} - Rank position: {rank_position+1 if rank_position >= 0 else 'Not found'}"
        )


def plot_distance_distribution(predictions):
    """Plot the distribution of distances.

    Args:
        predictions: Dictionary containing prediction data
    """
    print("\n===== Distance Distribution =====")

    distances = predictions["distances"].flatten()

    # Create a figure
    plt.figure(figsize=(10, 6))

    # Plot histogram
    plt.hist(distances, bins=50, alpha=0.7)
    plt.axvline(
        x=np.median(distances),
        color="r",
        linestyle="--",
        label=f"Median: {np.median(distances):.2f}",
    )
    plt.axvline(
        x=np.mean(distances),
        color="g",
        linestyle="--",
        label=f"Mean: {np.mean(distances):.2f}",
    )

    plt.title("Distribution of Distances")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.legend()

    # Save the figure
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/distance_distribution.png")
    print("Saved distance distribution plot to results/distance_distribution.png")
    plt.show()


def plot_embedding_statistics(predictions):
    """Plot statistics about the embeddings.

    Args:
        predictions: Dictionary containing prediction data
    """
    print("\n===== Embedding Statistics =====")

    # Combine query and gallery embeddings
    all_embeddings = np.vstack(
        [predictions["query_embeddings"], predictions["gallery_embeddings"]]
    )

    # Calculate statistics per dimension
    dim_means = np.mean(all_embeddings, axis=0)
    dim_stds = np.std(all_embeddings, axis=0)
    dim_mins = np.min(all_embeddings, axis=0)
    dim_maxs = np.max(all_embeddings, axis=0)

    # Create a figure
    plt.figure(figsize=(12, 8))

    # Plot mean and standard deviation for each dimension
    x = np.arange(len(dim_means))
    plt.errorbar(
        x, dim_means, yerr=dim_stds, fmt="o", alpha=0.5, markersize=2, elinewidth=0.5
    )

    plt.title("Embedding Statistics per Dimension")
    plt.xlabel("Dimension")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)

    # Add overall statistics as text
    overall_min = np.min(all_embeddings)
    overall_max = np.max(all_embeddings)
    overall_mean = np.mean(all_embeddings)
    overall_std = np.std(all_embeddings)

    stats_text = f"Overall Statistics:\nMin: {overall_min:.4f}\nMax: {overall_max:.4f}\nMean: {overall_mean:.4f}\nStd: {overall_std:.4f}"
    plt.figtext(
        0.02,
        0.02,
        stats_text,
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "pad": 5},
    )

    # Save the figure
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/embedding_statistics.png")
    print("Saved embedding statistics plot to results/embedding_statistics.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Explore prediction results from the person re-identification model"
    )
    parser.add_argument(
        "--predictions",
        default="results/predictions_latest.npz",
        help="Path to predictions file",
    )
    parser.add_argument(
        "--query", type=int, default=None, help="Index of specific query to analyze"
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Number of closest matches to display"
    )
    args = parser.parse_args()

    # Find the latest predictions file if not specified
    if args.predictions == "results/predictions_latest.npz":
        results_dir = Path("results")
        if results_dir.exists():
            prediction_files = list(results_dir.glob("predictions_*.npz"))
            if prediction_files:
                latest_file = max(prediction_files, key=lambda x: x.stat().st_mtime)
                args.predictions = str(latest_file)
                print(f"Using latest predictions file: {args.predictions}")

    # Load predictions
    predictions = load_predictions(args.predictions)

    # Print basic statistics
    print_basic_stats(predictions)

    # Analyze performance
    analyze_performance(predictions)

    # Plot distance distribution
    plot_distance_distribution(predictions)

    # Plot embedding statistics
    plot_embedding_statistics(predictions)

    # Analyze specific query if requested
    if args.query is not None:
        analyze_query(predictions, args.query, args.top_k)
    else:
        # Analyze a random query
        import random

        random_query = random.randint(0, len(predictions["query_embeddings"]) - 1)
        print(f"\nAnalyzing a random query (use --query to specify a specific one)")
        analyze_query(predictions, random_query, args.top_k)


if __name__ == "__main__":
    main()
