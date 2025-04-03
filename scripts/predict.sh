#!/bin/bash



# Default values
CONFIG="configs/inference.yaml"
DEVICE="mps"



# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;

        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done



# Run prediction
uv run python scripts/predict.py \
    --config "$CONFIG" \
    --device "$DEVICE" \
