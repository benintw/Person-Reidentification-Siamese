#!/bin/bash
# Training shell script wrapper for the aerial image segmentation model.
# 
# This script provides a convenient CLI interface to:
# - Set training configuration
# - Select compute device
# - Create necessary directories
# - Execute training with proper logging

# default values
CONFIG_PATH="configs/training.yaml"
DEVICE="mps"


# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -c|--config) CONFIG_PATH="$2"; shift ;;
        -d|--device) DEVICE="$2"; shift ;;

        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

# Create necessary directories
mkdir -p checkpoints
mkdir -p logs


# Set up timestamp for logging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/training_${TIMESTAMP}.log"



# Print training configuration
echo "Starting training with:"
echo "  Config: $CONFIG_PATH"
echo "  Device: $DEVICE"
echo

# Run the training script
uv run python scripts/train.py \
    --config "$CONFIG_PATH" \
    --device "$DEVICE" \

echo "Training completed!"
echo "Check the logs and results in the logs directory."