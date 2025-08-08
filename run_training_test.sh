#!/bin/bash
# SEG Loss検証用の訓練スクリプト

echo "======================================"
echo "SEG Loss訓練テスト開始"
echo "======================================"

# 訓練実行（少量データで短時間）
python minimal_train.py \
    --batch_size 2 \
    --num_epochs 2 \
    --samples_per_epoch 20 \
    --seg_loss_weight 3.0 \
    --adapter_lr 0.005 \
    --seg_token_lr 0.001 \
    --save_steps 10 \
    --debug

echo ""
echo "======================================"
echo "訓練完了"
echo "======================================" 