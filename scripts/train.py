import argparse

from src.training.trainer import Trainer
from src.utils.config import load_configs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--config", type=str, default="configs/training.yaml")
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["mps", "cuda", "cpu"],
        help="Device to run the training on",
    )
    args = parser.parse_args()

    config = load_configs(args.config)
    trainer = Trainer(config, device_name=args.device)
    trainer.train()
