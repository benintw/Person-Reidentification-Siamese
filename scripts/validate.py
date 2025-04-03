import argparse
from pathlib import Path

import torch

from src.data.data_loader import get_dataloaders
from src.models.model import APN_Model
from src.training.evaluator import Evaluator
from src.utils.config import load_configs


def load_checkpoint(checkpoint_path, device):
    """Load model checkpoint with proper handling of PyTorch 2.6 weights_only parameter.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the checkpoint to

    Returns:
        Loaded checkpoint dictionary
    """
    try:
        # Add safe globals for numpy scalar types and dtype
        import numpy as np
        from torch.serialization import add_safe_globals

        add_safe_globals([np._core.multiarray.scalar, np.dtype])

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

        return checkpoint
    except FileNotFoundError:
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the model")
    parser.add_argument("--config", type=str, default="configs/validation.yaml")
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["mps", "cuda", "cpu"],
        help="Device to run the training on",
    )
    args = parser.parse_args()

    config = load_configs(args.config)
    model = APN_Model(config).to(args.device)
    checkpoint_path = Path(config["save_dir"]) / config["save_name"]

    # Load checkpoint with proper handling of weights_only parameter
    checkpoint = load_checkpoint(checkpoint_path, args.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, val_dataloader, _ = get_dataloaders(config)
    evaluator = Evaluator(model, args.device, config)
    val_loss = evaluator.validate(val_dataloader)
    print(f"Validation loss: {val_loss}")


if __name__ == "__main__":
    main()
