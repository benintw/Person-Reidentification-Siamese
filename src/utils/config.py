from pathlib import Path

import yaml
from icecream import ic


def load_configs(
    base_config_path: str,
) -> dict:
    """
    Load a base config file and inject checkpoint config into it.

    Args:
        base_config_path: Path to the base config file (training.yaml, validation.yaml, etc.)
    """
    # Load base config
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    # Load checkpoint config
    checkpoint_path = Path(base_config_path).parent / "checkpoints.yaml"
    with open(checkpoint_path, "r") as f:
        checkpoint_config = yaml.safe_load(f)

    # Load dataset config
    dataset_path = Path(base_config_path).parent / "dataset.yaml"
    with open(dataset_path, "r") as f:
        dataset_config = yaml.safe_load(f)

    # Load model config
    model_path = Path(base_config_path).parent / "model.yaml"
    with open(model_path, "r") as f:
        model_config = yaml.safe_load(f)

    # Merge configs
    config = {**base_config, **checkpoint_config, **dataset_config, **model_config}

    return config


if __name__ == "__main__":
    config = load_configs("configs/training.yaml")
    config = load_configs("configs/validation.yaml")
    ic(config)
