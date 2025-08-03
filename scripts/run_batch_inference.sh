#!/bin/bash
# Batch inference script for LISA改 (LISA-Kai)

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Model and config paths
MODEL_PATH="output/lisa_kai_experiment/final_model"
INPUT_FILE="${1:-configs/inference_samples.json}"
OUTPUT_DIR="batch_output/$(date +%Y%m%d_%H%M%S)"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run batch inference
python batch_inference.py \
    --model_path $MODEL_PATH \
    --input_file $INPUT_FILE \
    --output_dir $OUTPUT_DIR \
    --max_new_tokens 100 \
    --temperature 0.7 \
    --batch_size 1 \
    --device cuda \
    --num_workers 1

echo "Results saved to: $OUTPUT_DIR"