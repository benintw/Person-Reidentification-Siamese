from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import timm
import torch
import torch.nn as nn
import yaml
from icecream import ic


class APN_Model(nn.Module):
    """Anchor-Positive-Negative model for person re-identification.

    This model uses a pretrained backbone (default: EfficientNet) to extract embeddings
    from images. These embeddings are used in triplet loss for person re-identification.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the model with the given configuration.

        Args:
            config: Dictionary containing model configuration parameters
                - encoder: Name of the timm model to use as backbone
                - pretrained: Whether to use pretrained weights
                - emb_size: Size of the embedding vector

        Raises:
            ValueError: If the encoder model is not available in timm
            RuntimeError: If there's an issue with the model initialization
        """
        super(APN_Model, self).__init__()

        try:
            # Create the backbone model
            self.efficientnet = timm.create_model(
                config["encoder"],
                pretrained=config.get("pretrained", True),
                num_classes=0,  # Remove classifier for feature extraction
            )

            # Get the feature dimension
            self.feature_dim = self.efficientnet.num_features
            # ic(f"Backbone feature dimension: {self.feature_dim}")

            # Add a new classifier/projection head
            self.classifier = nn.Linear(
                in_features=self.feature_dim,
                out_features=config["emb_size"],
            )

            # Initialize weights of the new layer
            nn.init.xavier_uniform_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)

        except Exception as e:
            ic(f"Error initializing model: {str(e)}")
            raise

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            images: Batch of input images [batch_size, channels, height, width]

        Returns:
            Embedding vectors for the input images [batch_size, emb_size]
        """
        # Extract features from the backbone
        features = self.efficientnet(images)

        # Project features to embedding space
        embeddings = self.classifier(features)

        return embeddings

    def get_embedding_dim(self) -> int:
        """Get the dimension of the output embedding.

        Returns:
            Size of the embedding vector
        """
        return self.classifier.out_features


def load_config(config_path: str = "configs/model.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary containing configuration parameters

    Raises:
        FileNotFoundError: If the config file doesn't exist
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file) as f:
        return yaml.safe_load(f)


def test_model(
    model: nn.Module, input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224)
) -> None:
    """Test the model by running a forward pass and displaying information.

    Args:
        model: The model to test
        input_shape: Shape of the input tensor (batch_size, channels, height, width)
    """
    # Print model architecture
    ic("Model Architecture:")
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ic(f"Total parameters: {total_params:,}")
    ic(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    random_input = torch.randn(*input_shape)
    ic(f"Input shape: {random_input.shape}")

    try:
        with torch.no_grad():
            output = model(random_input)
        ic(f"Output shape: {output.shape}")
        ic(
            f"Output stats - Min: {output.min().item():.4f}, Max: {output.max().item():.4f}, Mean: {output.mean().item():.4f}"
        )
    except Exception as e:
        ic(f"Error during forward pass: {str(e)}")


if __name__ == "__main__":
    try:
        # Load configuration
        config = load_config()
        ic("Config loaded successfully:", config)

        # Create and test model
        model = APN_Model(config)
        test_model(model)

        # Test with different batch sizes
        ic("Testing with batch size of 32:")
        test_model(model, (32, 3, 224, 224))

    except Exception as e:
        ic(f"Error: {str(e)}")
