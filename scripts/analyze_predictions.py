#!/usr/bin/env python
"""
Analyze prediction results from the person re-identification model.

This script loads saved prediction results and provides various analysis tools:
1. Basic statistics about the predictions
2. Visualization of query results
3. Detailed performance metrics
4. Embedding space visualization
5. Analysis of hard and easy cases

Usage:
    python scripts/analyze_predictions.py --predictions results/predictions_YYYYMMDD_HHMMSS.npz --data-dir data/train
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.utils.metrics import evaluate_predictions
from src.utils.visualization import plot_closest_imgs, visualize_query_results


def load_predictions(predictions_file):
    """Load prediction data from NPZ and JSON files.

    Args:
        predictions_file: Path to the .npz predictions file

    Returns:
        dict: Combined prediction data
    """
    # Load the predictions file
    predictions_data = np.load(predictions_file)

    # Load the image paths from the JSON file
    json_path = Path(predictions_file).stem + "_paths.json"
    json_path = Path(predictions_file).parent / json_path

    try:
        with open(json_path, "r") as f:
            paths_data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Could not find paths file at {json_path}")
        paths_data = {
            "query_img_paths": [
                f"query_{i}" for i in range(len(predictions_data["query_embeddings"]))
            ],
            "gallery_img_paths": [
                f"gallery_{i}"
                for i in range(len(predictions_data["gallery_embeddings"]))
            ],
        }

    # Combine the data
    predictions = {
        "query_embeddings": predictions_data["query_embeddings"],
        "gallery_embeddings": predictions_data["gallery_embeddings"],
        "query_labels": predictions_data["query_labels"],
        "gallery_labels": predictions_data["gallery_labels"],
        "distances": predictions_data["distances"],
        "rankings": predictions_data["rankings"],
        "query_img_paths": paths_data["query_img_paths"],
        "gallery_img_paths": paths_data["gallery_img_paths"],
    }

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

    # Analyze per-query performance
    query_labels = predictions["query_labels"]
    gallery_labels = predictions["gallery_labels"]
    rankings = predictions["rankings"]

    # Calculate AP for each query
    aps = []
    rank1_correct = []

    for i, q_label in enumerate(query_labels):
        # Get ranked gallery labels
        ranked_labels = gallery_labels[rankings[i]]

        # Binary relevance (1 if same person, 0 otherwise)
        relevance = (ranked_labels == q_label).astype(int)

        # Calculate average precision
        if np.sum(relevance) > 0:
            ap = average_precision_score(
                relevance, 1 - np.arange(len(relevance)) / len(relevance)
            )
            aps.append(ap)
        else:
            aps.append(0)

        # Check if rank-1 is correct
        rank1_correct.append(ranked_labels[0] == q_label)

    # Convert to numpy arrays
    aps = np.array(aps)
    rank1_correct = np.array(rank1_correct)

    # Find best and worst performing queries
    best_queries_idx = np.argsort(aps)[-5:]
    worst_queries_idx = np.argsort(aps)[:5]

    print("\nTop 5 best performing queries:")
    for idx in reversed(best_queries_idx):
        print(
            f"  Query {idx} ({predictions['query_img_paths'][idx]}): AP = {aps[idx]:.4f}, Rank-1 {'✓' if rank1_correct[idx] else '✗'}"
        )

    print("\nTop 5 worst performing queries:")
    for idx in worst_queries_idx:
        print(
            f"  Query {idx} ({predictions['query_img_paths'][idx]}): AP = {aps[idx]:.4f}, Rank-1 {'✓' if rank1_correct[idx] else '✗'}"
        )

    return {
        "aps": aps,
        "rank1_correct": rank1_correct,
        "best_queries_idx": best_queries_idx,
        "worst_queries_idx": worst_queries_idx,
    }


def visualize_embeddings_2d(predictions, output_dir="results"):
    """Visualize embeddings in 2D using t-SNE.

    Args:
        predictions: Dictionary containing prediction data
        output_dir: Directory to save the visualization
    """
    print("\n===== Visualizing Embedding Space =====")

    # Combine query and gallery embeddings
    all_embeddings = np.vstack(
        [predictions["query_embeddings"], predictions["gallery_embeddings"]]
    )
    all_labels = np.concatenate(
        [predictions["query_labels"], predictions["gallery_labels"]]
    )

    # Apply t-SNE for dimensionality reduction
    print("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings)

    # Create a figure
    plt.figure(figsize=(12, 10))

    # Get unique labels and assign colors
    unique_labels = np.unique(all_labels)

    # Limit to 20 classes for clarity if there are too many
    if len(unique_labels) > 20:
        print(
            f"Limiting visualization to 20 random classes out of {len(unique_labels)}"
        )
        np.random.seed(42)
        selected_labels = np.random.choice(unique_labels, 20, replace=False)
    else:
        selected_labels = unique_labels

    # Plot each class
    for label in selected_labels:
        idx = all_labels == label
        plt.scatter(
            embeddings_2d[idx, 0],
            embeddings_2d[idx, 1],
            label=str(label)[:4],
            alpha=0.7,
            s=50,
        )

    # Add markers to indicate query points
    query_count = len(predictions["query_embeddings"])
    plt.scatter(
        embeddings_2d[:query_count, 0],
        embeddings_2d[:query_count, 1],
        facecolors="none",
        edgecolors="black",
        s=80,
        label="Query",
    )

    plt.title("t-SNE Visualization of Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))

    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "embedding_visualization.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"Saved embedding visualization to {save_path}")
    plt.show()


def analyze_query(predictions, query_idx, data_dir, no_of_closest=10):
    """Analyze a specific query in detail.

    Args:
        predictions: Dictionary containing prediction data
        query_idx: Index of the query to analyze
        data_dir: Directory containing the images
        no_of_closest: Number of closest matches to display
    """
    print(f"\n===== Analyzing Query {query_idx} =====")

    # Get query information
    query_path = predictions["query_img_paths"][query_idx]
    query_label = predictions["query_labels"][query_idx]
    print(f"Query image: {query_path}")
    print(f"Person ID: {query_label}")

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
        print(f"  {i+1}. {path} (distance: {dist:.4f}) - {'✓' if is_correct else '✗'}")

    # Calculate precision@k
    precision_at_k = correct_count / no_of_closest
    print(f"\nPrecision@{no_of_closest}: {precision_at_k:.4f}")

    # Try to visualize the query results
    try:
        visualize_query_results(predictions, query_idx, data_dir, no_of_closest)
    except Exception as e:
        print(f"Error visualizing query results: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze prediction results from the person re-identification model"
    )
    parser.add_argument(
        "--predictions",
        default="results/predictions_latest.npz",
        help="Path to predictions file",
    )
    parser.add_argument(
        "--data-dir", default="data/train", help="Directory containing the images"
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
    performance = analyze_performance(predictions)

    # Visualize embeddings
    visualize_embeddings_2d(predictions)

    # Analyze specific query if requested
    if args.query is not None:
        analyze_query(predictions, args.query, args.data_dir, args.top_k)
    else:
        # Analyze best and worst performing queries
        print("\n===== Analyzing Best and Worst Queries =====")
        print("Best performing query:")
        best_idx = performance["best_queries_idx"][-1]
        analyze_query(predictions, best_idx, args.data_dir, args.top_k)

        print("\nWorst performing query:")
        worst_idx = performance["worst_queries_idx"][0]
        analyze_query(predictions, worst_idx, args.data_dir, args.top_k)


if __name__ == "__main__":
    main()
