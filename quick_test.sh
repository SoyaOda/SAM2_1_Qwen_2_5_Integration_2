#!/bin/bash
# 高速テスト用スクリプト

echo "🚀 高速テストモードで起動中..."
echo "データセット読み込みを最小限に抑えています。"

# 最小設定で実行
python minimal_train.py \
    --dataset_types "sem_seg" \
    --sample_rates "1" \
    --samples_per_epoch 50 \
    --batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_epochs 1 \
    --save_steps 100 \
    --fast_dev_run \
    --debug

echo "✅ テスト完了"