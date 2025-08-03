#!/bin/bash
# Inference script for LISA改 (LISA-Kai)

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Model and input paths
MODEL_PATH="output/lisa_kai_experiment/final_model"
IMAGE_PATH="$1"
INSTRUCTION="$2"
OUTPUT_DIR="inference_output/$(date +%Y%m%d_%H%M%S)"

# Check arguments
if [ -z "$IMAGE_PATH" ] || [ -z "$INSTRUCTION" ]; then
    echo "Usage: $0 <image_path> <instruction>"
    echo "Example: $0 sample.jpg \"Please segment the apple in this image\""
    exit 1
fi

# Create output directory
mkdir -p $OUTPUT_DIR

# Run inference
python inference.py \
    --model_path $MODEL_PATH \
    --image "$IMAGE_PATH" \
    --instruction "$INSTRUCTION" \
    --output_dir $OUTPUT_DIR \
    --max_new_tokens 100 \
    --temperature 0.7 \
    --device cuda

echo "Results saved to: $OUTPUT_DIR"