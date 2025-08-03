#!/bin/bash
# Run training integration test for LISA改 (LISA-Kai)

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create necessary directories
mkdir -p checkpoints
mkdir -p test_outputs

echo "Running LISA改 Training Integration Test"
echo "========================================"
echo "This test will:"
echo "1. Download model weights (Qwen2.5-VL-3B and SAM2.1)"
echo "2. Test gradient flow through the model"
echo "3. Run a few training epochs to verify loss reduction"
echo "4. Test model save/load functionality"
echo ""

# Run training test
python tests/training_integration_test.py \
    --device cuda \
    "$@"

echo "Training test completed!"