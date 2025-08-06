#!/usr/bin/env python3
"""
データセットの出力形式をデバッグするスクリプト
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.data.dataset import HybridDataset
from src.config import LISAConfig
from transformers import AutoProcessor

# LISAConfig作成
config = LISAConfig()

# プロセッサの準備
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# データセットの作成
dataset = HybridDataset(
    base_image_dir=config.dataset_base_dir,
    qwen_processor=processor,
    samples_per_epoch=10,
    dataset="sem_seg",
    sample_rate=[1.0],
    qwen_image_size=448,
    sam_image_size=1024,
)

# 最初のサンプルを取得
sample = dataset[0]

# データ形式を確認
print("=== データセット出力形式の確認 ===")
for key, value in sample.items():
    if hasattr(value, 'shape'):
        print(f"{key}: shape={value.shape}, dtype={value.dtype}")
    elif hasattr(value, '__len__'):
        print(f"{key}: len={len(value)}, type={type(value)}")
    else:
        print(f"{key}: type={type(value)}")

# pixel_valuesの詳細確認
if 'pixel_values' in sample:
    pv = sample['pixel_values']
    print(f"\npixel_values詳細:")
    print(f"  次元数: {pv.dim()}")
    print(f"  形状: {pv.shape}")
    if pv.dim() == 2:
        print("  ⚠️ 警告: pixel_valuesが2次元です。3次元(C,H,W)が期待されています。")
    elif pv.dim() == 3:
        print(f"  チャンネル数: {pv.shape[0]}")
        print(f"  高さ: {pv.shape[1]}")
        print(f"  幅: {pv.shape[2]}")