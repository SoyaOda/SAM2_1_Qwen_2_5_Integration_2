#!/bin/bash
# deepresearch.md 4-4以降の改善をテストするスクリプト
# セグメンテーション損失の安定化版

echo "=========================================="
echo "LISA-Kai改善版訓練テスト"
echo "セグメンテーション損失の安定化版"
echo "=========================================="

# GPUメモリの状況確認
echo "GPU状況:"
nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader

# 小規模テスト用設定（高速確認用）
# - inference_eval_samplesは自動的に128以上になる
# - 段階学習をテストするため、少し更新回数を増やす
# - debugフラグを追加して詳細ログ出力

python minimal_train_backup.py \
    --samples_per_epoch 100 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_epochs 2 \
    --save_steps 20 \
    --visualize_steps 20 \
    --run_inference_eval \
    --inference_eval_samples 3 \
    --warmup_ratio 0.1 \
    --fp16 \
    --max_grad_norm 1.0 \
    --debug \
    --output_dir output/test_improved_$(date +%Y%m%d_%H%M%S) \
    2>&1 | tee test_log_$(date +%Y%m%d_%H%M%S).txt

echo "=========================================="
echo "テスト完了"
echo "ログファイルを確認してください"
echo "=========================================="
