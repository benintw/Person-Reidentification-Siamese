"""
Purpose: predict.py serves as the command-line entry point for the make predict command, orchestrating the inference process by setting up the necessary components and calling Predictor.predict().

Key Responsibilities:
- Parse command-line arguments (e.g., --config, --device).
- Load the configuration from configs/inference.yaml.
- Initialize the model, data loader, and Predictor.
- Run the prediction process and log results.

Context: This script integrates with your Makefile and scripts/predict.sh, providing a user-friendly interface to run inference.

"""

import argparse
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

from src.data.data_loader import get_dataloaders, get_test_dataloaders
from src.inference.predictor import Predictor, save_predictions
from src.models.model import APN_Model
from src.utils.config import load_configs
from src.utils.device import get_device
from src.utils.metrics import evaluate_predictions
from src.utils.visualization import visualize_embedding_space, visualize_query_results


def main():
    # Parse arguments and load config
    parser = argparse.ArgumentParser(
        description="Generate predictions with the person re-identification model"
    )
    parser.add_argument(
        "--config", default="configs/inference.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--device", default="mps", help="Device to use (cpu, cuda, mps)"
    )
    args = parser.parse_args()

    # Load config
    config = load_configs(args.config)
    device = get_device(args.device)

    # Create model

    model = APN_Model(config)
    model.to(device)

    # Load checkpoint
    checkpoint_path = Path(config["checkpoint_path"])
    try:
        # Add safe globals for numpy scalar types
        import numpy as np
        from torch.serialization import add_safe_globals

        add_safe_globals([np._core.multiarray.scalar])

        # Try loading with weights_only=True first (safer)
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=True
            )
            print(f"Loaded checkpoint from {checkpoint_path} with weights_only=True")
        except Exception as e:
            print(f"Warning: Could not load with weights_only=True: {e}")
            print(
                "Attempting to load with weights_only=False (less secure but more compatible)..."
            )
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            print(f"Loaded checkpoint from {checkpoint_path} with weights_only=False")

        model.load_state_dict(checkpoint["model_state_dict"])
    except FileNotFoundError:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    except Exception as e:
        raise Exception(f"Error loading checkpoint: {e}")

    # Get train, val, and test data loaders
    _, _, test_loader = get_dataloaders(config)
    print(f"Test set size: {len(test_loader.dataset)}")

    # Split test set into query and gallery
    query_loader, gallery_loader = get_test_dataloaders(config, test_loader.dataset)
    print(f"Query set size: {len(query_loader.dataset)}")
    print(f"Gallery set size: {len(gallery_loader.dataset)}")

    # Create combined loader for embedding visualization
    combined_dataset = ConcatDataset([query_loader.dataset, gallery_loader.dataset])
    combined_loader = DataLoader(
        combined_dataset, batch_size=config["batch_size"], shuffle=False
    )

    # Create predictor and generate predictions
    predictor = Predictor(model, device, config)
    predictions = predictor.predict(query_loader, gallery_loader)

    # Evaluate predictions
    metrics = evaluate_predictions(predictions)
    print("\n===== Test Set Performance =====")
    print(f"Mean Average Precision (mAP): {metrics['mAP']:.4f}")
    print(f"Rank-1 Accuracy: {metrics['rank1']:.4f}")
    print(f"Rank-5 Accuracy: {metrics['rank5']:.4f}")
    print(f"Rank-10 Accuracy: {metrics['rank10']:.4f}")

    # Save predictions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(config["results_dir"]) / f"predictions_{timestamp}.npz"
    save_predictions(predictions, save_path)

    # Visualize embedding space
    if config.get("visualize_embeddings", True):
        print("\nVisualizing embedding space...")
        visualize_embedding_space(predictions, model, combined_loader, device)

    # Visualize some query results
    if config.get("visualize_queries", True):
        print("\nVisualizing query results...")
        data_dir = config["data_dir"]
        # Check if data_dir ends with a slash
        if not data_dir.endswith("/"):
            data_dir += "/"

        # Get number of queries to visualize
        num_queries = config.get("num_queries_to_visualize", 5)
        # Visualize queries
        for i in range(min(num_queries, len(predictions["query_embeddings"]))):
            try:
                visualize_query_results(predictions, i, data_dir)
            except Exception as e:
                print(f"Error visualizing query {i}: {e}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
