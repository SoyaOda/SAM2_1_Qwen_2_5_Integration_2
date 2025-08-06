#!/usr/bin/env python3
"""
apply_chat_templateの出力形式を確認するスクリプト
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from transformers import AutoProcessor
from PIL import Image
import torch
import numpy as np

# プロセッサの準備
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", use_fast=False)

# ダミー画像の作成
image_pil = Image.new('RGB', (448, 336), color='red')

# メッセージの作成
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_pil},
            {"type": "text", "text": "Segment the object in this image. <SEG>"}
        ]
    }
]

# apply_chat_templateの実行
print("=== apply_chat_template実行 ===")
processed = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
)

# 結果の確認
print("\n=== 出力形式の確認 ===")
for key, value in processed.items():
    if hasattr(value, 'shape'):
        print(f"{key}: shape={value.shape}, dtype={value.dtype}")
    elif hasattr(value, '__len__'):
        print(f"{key}: len={len(value)}, type={type(value)}")
    else:
        print(f"{key}: type={type(value)}")

# pixel_valuesの詳細
if 'pixel_values' in processed:
    pv = processed['pixel_values']
    print(f"\npixel_values詳細:")
    print(f"  次元数: {pv.dim()}")
    print(f"  形状: {pv.shape}")
    if pv.dim() == 4:
        print(f"  バッチサイズ: {pv.shape[0]}")
        print(f"  チャンネル数: {pv.shape[1]}")
        print(f"  高さ: {pv.shape[2]}")
        print(f"  幅: {pv.shape[3]}")
        
        # 1つ目のサンプルの形状
        print(f"\n1つ目のサンプル (squeeze(0)後): {pv.squeeze(0).shape}")
        print(f"  squeeze後の次元数: {pv.squeeze(0).dim()}")
    elif pv.dim() == 2:
        print("  ⚠️ 2次元フラット化されたパッチ形式")
        print(f"  パッチ数: {pv.shape[0]}")
        print(f"  特徴次元: {pv.shape[1]}")

# image_grid_thwの確認
if 'image_grid_thw' in processed:
    grid = processed['image_grid_thw']
    print(f"\nimage_grid_thw: {grid}")
    print(f"  形状: {grid.shape}")
    if grid.dim() >= 1:
        print(f"  値: {grid.tolist()}")