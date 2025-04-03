"""Metrics for person re-identification evaluation.

This module provides metrics for evaluating person re-identification models:
- compute_mAP: Calculate Mean Average Precision
- compute_cmc: Calculate Cumulative Matching Characteristics
- evaluate_predictions: Evaluate predictions using standard person re-ID metrics
"""

__all__ = ["compute_mAP", "compute_cmc", "evaluate_predictions"]

import numpy as np
import torch


def evaluate_predictions(predictions):
    """Evaluate predictions using standard person re-ID metrics.

    Args:
        predictions: Output from predictor.predict()

    Returns:
        dict: Evaluation metrics including mAP and CMC scores
    """
    query_labels = predictions["query_labels"]
    gallery_labels = predictions["gallery_labels"]
    rankings = predictions["rankings"]

    # Calculate mAP
    mAP = 0.0
    for i, q_label in enumerate(query_labels):
        ranked_labels = gallery_labels[rankings[i]]
        relevant = (ranked_labels == q_label).astype(int)

        if relevant.sum() > 0:
            precision_at_k = np.cumsum(relevant) / (np.arange(len(relevant)) + 1)
            ap = precision_at_k[relevant > 0].mean()
            mAP += ap

    mAP /= len(query_labels)

    # Calculate CMC curve
    cmc_scores = np.zeros(len(gallery_labels))
    for i, q_label in enumerate(query_labels):
        ranked_labels = gallery_labels[rankings[i]]
        matches = ranked_labels == q_label

        # For each rank position
        for k in range(len(cmc_scores)):
            if np.any(matches[: k + 1]):
                cmc_scores[k] += 1

    cmc_scores = cmc_scores / len(query_labels)

    return {
        "mAP": mAP,
        "rank1": cmc_scores[0],
        "rank5": cmc_scores[4] if len(cmc_scores) > 4 else None,
        "rank10": cmc_scores[9] if len(cmc_scores) > 9 else None,
        "cmc_curve": cmc_scores,
    }


def compute_mAP(model, query_loader, gallery_loader, device):
    """Calculate Mean Average Precision for person re-identification.

    Args:
        model: The neural network model
        query_loader: DataLoader for query images
        gallery_loader: DataLoader for gallery images
        device: Device to run inference on (cuda, cpu, mps)

    Returns:
        float: Mean Average Precision score
    """

    model.eval()
    query_embeddings, query_labels = [], []
    gallery_embeddings, gallery_labels = [], []

    with torch.no_grad():
        # Extract embeddings for query set
        for batch in query_loader:
            anchor, positive, negative, person_id = batch
            anchor = anchor.to(device)
            embeddings = model(anchor).cpu().numpy()
            query_embeddings.extend(embeddings)
            query_labels.extend(person_id)

        # Extract embeddings for gallery set
        for batch in gallery_loader:
            anchor, positive, negative, person_id = batch
            anchor = anchor.to(device)
            embeddings = model(anchor).cpu().numpy()
            gallery_embeddings.extend(embeddings)
            gallery_labels.extend(person_id)

    query_embeddings = np.array(query_embeddings)
    gallery_embeddings = np.array(gallery_embeddings)
    query_labels = np.array(query_labels)
    gallery_labels = np.array(gallery_labels)

    mAP = 0.0
    for i, q_emb in enumerate(query_embeddings):
        distances = np.linalg.norm(gallery_embeddings - q_emb, axis=1)
        sorted_indices = np.argsort(distances)
        ranked_labels = gallery_labels[sorted_indices]
        relevant = (ranked_labels == query_labels[i]).astype(int)
        precision_at_k = np.cumsum(relevant) / (np.arange(len(relevant)) + 1)
        ap = precision_at_k[relevant > 0].mean() if relevant.sum() > 0 else 0
        mAP += ap

    mAP /= len(query_embeddings)

    return mAP


def compute_cmc(model, query_loader, gallery_loader, device, top_k=10):
    """Compute Cumulative Matching Characteristics (CMC) curve for person re-identification.

    The CMC curve is a rank-based metric that evaluates how well a re-identification system ranks
    the correct matches. It measures the probability that a query person appears in different-sized
    candidate lists (e.g., top-1, top-5, top-10).

    Args:
        model (torch.nn.Module): Trained model to evaluate.
        query_loader (DataLoader): DataLoader containing query images.
        gallery_loader (DataLoader): DataLoader containing gallery images to match against.
        device (torch.device): Device to run the model on (e.g., 'cuda', 'cpu', 'mps').
        top_k (int, optional): Number of top ranks to compute. Defaults to 10.

    Returns:
        np.ndarray: CMC scores as a 1D array of length `top_k`, where each element represents
                   the probability of finding the correct match within the top-k positions.
                   For example, cmc_scores[0] is the Rank-1 accuracy (probability of the correct
                   match being the top match), cmc_scores[4] is the Rank-5 accuracy, etc.

    Example:
        >>> cmc_scores = compute_cmc(model, query_loader, gallery_loader, device)
        >>> print(f"Rank-1: {cmc_scores[0]:.4f}, Rank-5: {cmc_scores[4]:.4f}, Rank-10: {cmc_scores[9]:.4f}")
    """
    model.eval()
    query_embeddings, query_labels = [], []
    gallery_embeddings, gallery_labels = [], []

    with torch.no_grad():
        for batch in query_loader:
            anchor, positive, negative, person_id = batch
            anchor = anchor.to(device)
            embeddings = model(anchor).cpu().numpy()
            query_embeddings.extend(embeddings)
            query_labels.extend(person_id)

        for batch in gallery_loader:
            anchor, positive, negative, person_id = batch
            anchor = anchor.to(device)
            embeddings = model(anchor).cpu().numpy()
            gallery_embeddings.extend(embeddings)
            gallery_labels.extend(person_id)

    query_embeddings = np.array(query_embeddings)
    gallery_embeddings = np.array(gallery_embeddings)
    query_labels = np.array(query_labels)
    gallery_labels = np.array(gallery_labels)

    # Initialize CMC scores array
    cmc_scores = np.zeros(top_k)
    num_valid_queries = 0

    for i, q_emb in enumerate(query_embeddings):
        # Skip if query label doesn't exist in gallery (prevents division by zero later)
        if query_labels[i] not in gallery_labels:
            continue

        num_valid_queries += 1

        # Calculate distances between query embedding and all gallery embeddings
        distances = np.linalg.norm(gallery_embeddings - q_emb, axis=1)

        # Sort gallery by distance (closest first)
        sorted_indices = np.argsort(distances)
        ranked_labels = gallery_labels[sorted_indices]

        # Find positions where correct matches occur
        matches = ranked_labels == query_labels[i]

        # Check for matches at each rank position
        for k in range(top_k):
            # For rank-k, check if there's a match in the top-k positions
            # This is the standard CMC calculation - for rank k, we check if the correct match
            # appears in any position from 0 to k
            if np.any(matches[: k + 1]):
                cmc_scores[k] += 1

    # Normalize by number of valid queries
    if num_valid_queries > 0:
        cmc_scores = cmc_scores / num_valid_queries
    else:
        print("Warning: No valid queries found for CMC calculation")

    return cmc_scores
