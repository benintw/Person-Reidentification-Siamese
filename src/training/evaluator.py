"""
Purpose: evaluator.py is responsible for the evaluation or validation phase, assessing the model’s performance on a separate validation or test dataset. It does not update model parameters but computes metrics to gauge generalization and quality.

Key Responsibilities:
- Load a pre-trained model (e.g., from a checkpoint).
- Iterate over validation/test data batches without gradient computation (using torch.no_grad()).
- Compute evaluation metrics (e.g., IoU, Dice score for segmentation) and log results.
- Generate visualizations (e.g., segmentation masks) if needed.
- Handle evaluation-specific configurations (e.g., batch size, metrics) from configs/validation.yaml.

Context: This is run after training or periodically during training to monitor performance on unseen data, ensuring the model generalizes well.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.utils.metrics import compute_cmc, compute_mAP


class Evaluator:

    def __init__(self, model: nn.Module, device: torch.device, config: dict):
        self.model = model
        self.device = device
        self.config = config
        self.model.eval()
        self.loss_fn = torch.nn.TripletMarginLoss(margin=1.0)

    def validate(self, val_dataloader: DataLoader) -> tuple[float, dict]:
        """Validate the model on the validation set.

        Computes validation loss and evaluation metrics (mAP, CMC) if configured.

        Args:
            val_dataloader: DataLoader for validation data

        Returns:
            tuple: (average_loss, metrics_dict)
        """
        print("\n" + "-" * 50)
        print("Validating model...")

        # Compute validation loss
        total_loss = 0.0
        with torch.inference_mode():
            for batch in tqdm(val_dataloader, desc="Validating ..."):
                anchor, positive, negative, person_id = batch

                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)

                a_embedding = self.model(anchor)
                p_embedding = self.model(positive)
                n_embedding = self.model(negative)

                loss = self.loss_fn(a_embedding, p_embedding, n_embedding)

                total_loss += loss.item()

        avg_loss = total_loss / len(val_dataloader)
        print(f"Validation Loss: {avg_loss:.4f}")

        # Compute evaluation metrics if configured
        metrics = {}
        if self.config.get("compute_metrics", False):
            print("Computing evaluation metrics...")

            # Split validation set into query and gallery
            query_ds = Subset(
                val_dataloader.dataset,
                range(0, len(val_dataloader.dataset) // 2),
            )
            gallery_ds = Subset(
                val_dataloader.dataset,
                range(
                    len(val_dataloader.dataset) // 2,
                    len(val_dataloader.dataset),
                ),
            )

            # Create data loaders
            query_loader = DataLoader(
                query_ds, batch_size=self.config["batch_size"], shuffle=False
            )
            gallery_loader = DataLoader(
                gallery_ds, batch_size=self.config["batch_size"], shuffle=False
            )

            # Compute Mean Average Precision
            mAP = compute_mAP(self.model, query_loader, gallery_loader, self.device)
            metrics["mAP"] = mAP
            print(f"Mean Average Precision (mAP): {mAP:.4f}")

            # Compute CMC curve
            cmc_scores = compute_cmc(
                self.model, query_loader, gallery_loader, self.device, top_k=10
            )
            metrics["rank1"] = cmc_scores[0]
            metrics["rank5"] = cmc_scores[4] if len(cmc_scores) > 4 else None
            metrics["rank10"] = cmc_scores[9] if len(cmc_scores) > 9 else None

            # Format and display CMC scores
            rank1_str = f"{cmc_scores[0]:.4f}"
            rank5_str = f"{cmc_scores[4]:.4f}" if len(cmc_scores) > 4 else "N/A"
            rank10_str = f"{cmc_scores[9]:.4f}" if len(cmc_scores) > 9 else "N/A"
            print(
                f"CMC Scores - Rank-1: {rank1_str}, Rank-5: {rank5_str}, Rank-10: {rank10_str}"
            )

        print("-" * 50)
        return avg_loss, metrics
