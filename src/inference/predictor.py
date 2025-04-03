"""
Purpose: predictor.py is responsible for the inference phase in a person re-identification system, applying the trained Siamese network model to generate embeddings for query and gallery images and find the closest matches.

Key Responsibilities:
- Process query and gallery images to generate embeddings using the trained model
- Compute distances between query and gallery embeddings to find the closest matches
- Organize prediction results including embeddings, rankings, and image paths for evaluation and visualization
- Save prediction results to disk for later analysis or deployment

Main Components:
- Predictor class: Handles the inference process, generating embeddings and computing distances
- save_predictions function: Saves prediction results to disk in a structured format

Context: This module is used during the evaluation phase after training and validation, to assess how well the model performs on the test set. It supports the calculation of standard person re-identification metrics (mAP, CMC) and visualization of query results.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Predictor:

    def __init__(self, model: nn.Module, device: torch.device, config: dict):
        self.model = model
        self.device = device
        self.config = config
        self.model.eval()
        self.loss_fn = torch.nn.TripletMarginLoss(margin=1.0)

    def predict(self, query_loader: DataLoader, gallery_loader: DataLoader) -> dict:
        """Generate embeddings and find matches for query images in gallery.

        Args:
            query_loader: DataLoader for query images
            gallery_loader: DataLoader for gallery images

        Returns:
            dict: Contains query and gallery information, embeddings, and rankings
        """
        # Extract embeddings
        query_embeddings, query_labels, query_img_paths = [], [], []
        gallery_embeddings, gallery_labels, gallery_img_paths = [], [], []

        print("Extracting query embeddings...")
        with torch.no_grad():
            # Process query images
            for batch in tqdm(query_loader, desc="Processing query images"):
                anchor, positive, negative, person_id = batch

                anchor = anchor.to(self.device)
                embeddings = self.model(anchor).cpu().numpy()
                query_embeddings.extend(embeddings)
                query_labels.extend(person_id)

                # Store image paths if available (for visualization)
                # Since we're now using a direct dataset (not Subset), we can access image paths differently
                try:
                    # Try to get the image paths from the dataset directly
                    if hasattr(query_loader.dataset, "df"):
                        # Get the batch size
                        batch_size = len(person_id)
                        
                        # Get the current position in the dataset based on the batch index
                        # This is an approximation and may not be accurate for all dataloaders
                        batch_idx = len(query_embeddings) - len(embeddings)
                        
                        # Add paths for all images in this batch
                        for i in range(batch_size):
                            if batch_idx + i < len(query_loader.dataset.df):
                                query_img_paths.append(query_loader.dataset.df.iloc[batch_idx + i]["Anchor"])
                            else:
                                # Fallback if index is out of bounds
                                query_img_paths.append(f"query_{len(query_img_paths)}")
                    else:
                        # Fallback if dataset doesn't have df attribute
                        query_img_paths.extend([f"query_{i}" for i in range(len(embeddings))])
                except Exception as e:
                    print(f"Warning: Could not extract query image paths: {e}")
                    # Fallback to generic paths
                    query_img_paths.extend([f"query_{i}" for i in range(len(embeddings))])

            print("Extracting gallery embeddings...")
            # Process gallery images
            for batch in tqdm(gallery_loader, desc="Processing gallery images"):
                anchor, positive, negative, person_id = batch

                anchor = anchor.to(self.device)
                embeddings = self.model(anchor).cpu().numpy()
                gallery_embeddings.extend(embeddings)
                gallery_labels.extend(person_id)

                # Store image paths if available (for visualization)
                # Use the same approach as for query images
                try:
                    # Try to get the image paths from the dataset directly
                    if hasattr(gallery_loader.dataset, "df"):
                        # Get the batch size
                        batch_size = len(person_id)
                        
                        # Get the current position in the dataset based on the batch index
                        # This is an approximation and may not be accurate for all dataloaders
                        batch_idx = len(gallery_embeddings) - len(embeddings)
                        
                        # Add paths for all images in this batch
                        for i in range(batch_size):
                            if batch_idx + i < len(gallery_loader.dataset.df):
                                gallery_img_paths.append(gallery_loader.dataset.df.iloc[batch_idx + i]["Anchor"])
                            else:
                                # Fallback if index is out of bounds
                                gallery_img_paths.append(f"gallery_{len(gallery_img_paths)}")
                    else:
                        # Fallback if dataset doesn't have df attribute
                        gallery_img_paths.extend([f"gallery_{i}" for i in range(len(embeddings))])
                except Exception as e:
                    print(f"Warning: Could not extract gallery image paths: {e}")
                    # Fallback to generic paths
                    gallery_img_paths.extend([f"gallery_{i}" for i in range(len(embeddings))])

        # Convert to numpy arrays
        query_embeddings = np.array(query_embeddings)
        gallery_embeddings = np.array(gallery_embeddings)
        query_labels = np.array(query_labels)
        gallery_labels = np.array(gallery_labels)

        print("Computing distances and rankings...")
        # Compute distances and rankings
        distances = np.zeros((len(query_embeddings), len(gallery_embeddings)))
        rankings = np.zeros((len(query_embeddings), len(gallery_embeddings)), dtype=int)

        for i, q_emb in enumerate(query_embeddings):
            # Calculate distances to all gallery embeddings
            distances[i] = np.linalg.norm(gallery_embeddings - q_emb, axis=1)
            # Get sorted indices (closest first)
            rankings[i] = np.argsort(distances[i])

        return {
            "query_embeddings": query_embeddings,
            "gallery_embeddings": gallery_embeddings,
            "query_labels": query_labels,
            "gallery_labels": gallery_labels,
            "query_img_paths": query_img_paths,
            "gallery_img_paths": gallery_img_paths,
            "distances": distances,
            "rankings": rankings,
        }


def save_predictions(predictions, save_path):
    """Save prediction results to disk.

    Args:
        predictions: Output from predictor.predict()
        save_path: Path to save results
    """
    # Create directory if it doesn't exist
    save_dir = Path(save_path).parent
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    # Save numpy arrays
    np.savez(
        save_path,
        query_embeddings=predictions["query_embeddings"],
        gallery_embeddings=predictions["gallery_embeddings"],
        query_labels=predictions["query_labels"],
        gallery_labels=predictions["gallery_labels"],
        distances=predictions["distances"],
        rankings=predictions["rankings"],
    )

    # Save paths separately (as they're strings, not easily saved in npz)
    with open(f"{save_path.stem}_paths.json", "w") as f:
        import json

        json.dump(
            {
                "query_img_paths": predictions["query_img_paths"],
                "gallery_img_paths": predictions["gallery_img_paths"],
            },
            f,
        )

    print(f"Predictions saved to {save_path}")


"""
# TODO:
1. finish the predictor.py 
2. what should be the output of the predictor?
3. how to visualize the output of predictor? visualization/plot_closest_imgs
4. how to evaluate the output of predictor?
5. how to save the output of predictor?
"""
