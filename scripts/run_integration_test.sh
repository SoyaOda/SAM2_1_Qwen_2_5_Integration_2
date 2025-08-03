#!/bin/bash
# Run integration test for LISA改 (LISA-Kai)

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create necessary directories
mkdir -p checkpoints
mkdir -p test_outputs

echo "Running LISA改 Integration Test"
echo "=============================="

# Run integration test
python tests/integration_test.py \
    --device cuda \
    "$@"

echo "Integration test completed!"