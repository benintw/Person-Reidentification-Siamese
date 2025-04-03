"""Person Re-Identification Model Training System

This module implements a comprehensive training pipeline for person re-identification models using a Siamese network
architecture with triplet loss. It handles the complete training lifecycle including:

1. Model Initialization and Configuration:
   - Sets up the model architecture, optimizer, and loss function based on configuration
   - Configures training parameters (learning rate, batch size, etc.)
   - Ensures reproducibility through proper seed initialization

2. Training Loop Management:
   - Implements epoch-based training with batch processing
   - Applies triplet loss optimization using anchor-positive-negative image triplets
   - Performs gradient clipping to prevent exploding gradients
   - Tracks and logs training metrics over time

3. Validation and Evaluation:
   - Periodically evaluates model performance on validation data
   - Calculates key person re-identification metrics:
     * Mean Average Precision (mAP)
     * Cumulative Matching Characteristics (CMC) at Rank-1, Rank-5, and Rank-10

4. Model Persistence and Visualization:
   - Implements early stopping to prevent overfitting
   - Saves model checkpoints for best-performing models
   - Generates training history visualizations for performance analysis

Usage:
    trainer = Trainer(config, device_name)
    trainer.train()

Configuration is loaded from configs/training.yaml and includes parameters for:
- Training duration (epochs)
- Optimization settings (learning rate, optimizer type)
- Early stopping criteria
- Model architecture specifications
- Checkpoint saving preferences
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from src.data.data_loader import get_dataloaders
from src.models.model import APN_Model
from src.training.evaluator import Evaluator
from src.utils.device import get_device
from src.utils.visualization import plot_training_history


class Trainer:
    """Manages the training process for person re-identification models.

    This class orchestrates the complete training lifecycle for a person re-identification model,
    including model initialization, training loop execution, validation, metric tracking,
    early stopping, and checkpoint management.

    The Trainer uses a triplet loss approach with anchor-positive-negative image triplets
    to train a Siamese network that learns discriminative embeddings for person re-identification.
    It tracks multiple performance metrics including loss, Mean Average Precision (mAP),
    and Cumulative Matching Characteristics (CMC) scores.

    Attributes:
        config (dict): Configuration dictionary with training parameters
        device (torch.device): Device to run training on (CPU, CUDA, or MPS)
        model (nn.Module): The neural network model being trained
        evaluator (Evaluator): Handles validation and metrics computation
        optimizer (torch.optim.Optimizer): Optimization algorithm
        loss_fn (nn.Module): Loss function (TripletMarginLoss)
        history (dict): Tracks metrics across training epochs
    """

    def __init__(self, config: dict[str, Any], device_name: str) -> None:
        """Initialize the trainer with configuration and device.

        Args:
            config: Dictionary containing training configuration parameters
            device_name: Name of the device to use for training (e.g., 'cpu', 'cuda', 'mps')
        """
        self.config = config
        self.device = get_device(device_name)
        print(f"Using device: {self.device}")

        self.setup_seeds()

        self.train_dataloader, self.val_dataloader, self.test_dataloader = (
            get_dataloaders(config)
        )
        self.model = self._create_model(config)
        self.model.to(self.device)
        self.evaluator = Evaluator(self.model, self.device, config)
        self.optimizer = self._create_optimizer(self.model.parameters(), config)
        self.loss_fn = torch.nn.TripletMarginLoss(margin=1.0)

        self.grad_clip = config.get("grad_clip", 1.0)
        self.history: dict[str, list[float | np.ndarray]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    def setup_seeds(self) -> None:
        torch.manual_seed(self.config["random_seed"])
        np.random.seed(self.config["random_seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config["random_seed"])
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(self.config["random_seed"])

    def _create_optimizer(self, model_params, config):
        optimizer_config = config.get("optimizer", {})
        optimizer_type = optimizer_config.get("type", "Adam").lower()
        optimizer_params = optimizer_config.get(
            "params", {"lr": config["learning_rate"]}
        )

        optimizers = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
        }
        if optimizer_type not in optimizers:
            raise ValueError(
                f"Unsupported optimizer: {optimizer_type}. Use one of {list(optimizers.keys())}"
            )
        return optimizers[optimizer_type](model_params, **optimizer_params)

    def _create_model(self, config):
        model_name = config.get("model", "RoadSegModel")
        models = {
            "apn_model": APN_Model,
        }
        if model_name.lower() not in models:
            raise ValueError(
                f"Unsupported model: {model_name}. Use one of {list(models.keys())}"
            )
        return models[model_name.lower()](config)

    def train_one_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        batch_count = len(self.train_dataloader)

        progress_bar = tqdm(
            self.train_dataloader,
            desc="Training",
            leave=False,  # Don't leave the progress bar
            ncols=100,  # Fixed width
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )

        for batch in progress_bar:
            anchor, positive, negative, person_id = batch
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)

            self.optimizer.zero_grad()
            a_embedding = self.model(anchor)
            p_embedding = self.model(positive)
            n_embedding = self.model(negative)

            loss = self.loss_fn(a_embedding, p_embedding, n_embedding)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()

            # Update progress bar with current loss
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / batch_count

    def validate(self) -> tuple[float, dict]:
        """Validate the model on the validation set using the Evaluator

        Returns:
            tuple: (average_loss, metrics_dict)
        """
        # Use evaluator to get validation loss and metrics
        avg_loss, metrics = self.evaluator.validate(self.val_dataloader)
        return avg_loss, metrics

    def train(self):
        best_val_loss: float = float("inf")
        patience_counter = 0
        start_time = datetime.now()

        # Print training configuration header
        print("\n" + "=" * 70)
        print(f"Starting training with {self.config['epochs']} epochs")
        print(f"   Device: {self.device}")
        print(f"   Batch size: {self.config['batch_size']}")
        print(f"   Learning rate: {self.config['learning_rate']}")
        print("=" * 70)

        # Create a cleaner epoch progress bar
        epochs_pbar = tqdm(
            range(self.config["epochs"]),
            desc="Epochs",
            unit="epoch",
            position=0,
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )

        for epoch in epochs_pbar:
            epoch_start_time = datetime.now()

            # Display epoch header
            print(f"\nEpoch {epoch + 1}/{self.config['epochs']} ")
            print("-" * 50)

            # Train and validate
            train_loss = self.train_one_epoch()
            print(f"Training Loss: {train_loss:.4f}")
            val_loss, metrics = self.validate()

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            if metrics:
                for metric_name, metric_value in metrics.items():
                    if metric_name not in self.history:
                        self.history[metric_name] = []
                    self.history[metric_name].append(metric_value)

            # Calculate epoch time
            epoch_time = datetime.now() - epoch_start_time
            print(f"Epoch time: {epoch_time}")

            # Handle model saving and early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(epoch, val_loss)
                print(f"New best model! Val Loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                print(
                    f"No improvement. Patience: {patience_counter}/{self.config['early_stopping_patience']}"
                )

            if patience_counter >= self.config["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            # Update progress bar with key metrics
            epochs_pbar.set_postfix(
                {"train_loss": f"{train_loss:.4f}", "val_loss": f"{val_loss:.4f}"}
            )

        # Calculate total training time
        total_time = datetime.now() - start_time

        # Display final summary
        print("\n" + "=" * 70)
        print(f"Training completed!")
        print(f"   Total time: {total_time}")
        print(f"   Best validation loss: {best_val_loss:.4f}")
        if "mAP" in self.history and self.history["mAP"]:
            print(f"   Best mAP: {max(self.history['mAP']):.4f}")
        print("=" * 70)

        # Save training history plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_path = (
            Path(self.config["logging"]["log_dir"])
            / f"training_history_{timestamp}.png"
        )
        plot_training_history(self.history, save_path=history_path)
        print(f"Training history saved to {history_path}")

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """Save model checkpoint to disk.

        Args:
            epoch: Current epoch number
            val_loss: Validation loss for this epoch
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "history": self.history,  # Save training history too
        }

        save_path = Path(self.config["save_dir"]) / self.config["save_name"]

        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(checkpoint, save_path)
        print(f"💾 Checkpoint saved to {save_path}")
