#!/bin/bash
# Training script for LISA改 (LISA-Kai)

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Training arguments
MODEL_CONFIG="configs/model_config_example.json"
DATA_CONFIG="configs/train_config_example.json"
OUTPUT_DIR="output/lisa_kai_experiment"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run training
python train.py \
    --model_config $MODEL_CONFIG \
    --data_config $DATA_CONFIG \
    --output_dir $OUTPUT_DIR \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_epochs 10 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --gradient_clip 1.0 \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --language_weight 1.0 \
    --segmentation_weight 1.0 \
    --max_length 512 \
    --image_size 448 \
    --save_steps 500 \
    --eval_steps 100 \
    --logging_steps 10 \
    --num_workers 4 \
    --seed 42 \
    2>&1 | tee $OUTPUT_DIR/train.log