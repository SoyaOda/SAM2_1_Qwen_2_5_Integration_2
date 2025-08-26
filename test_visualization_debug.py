#!/usr/bin/env python3
"""
Visualization問題のデバッグスクリプト
"""

import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from src.data.dataset import HybridDataset
from src.data.collators import MultiModalDataCollator
from torch.utils.data import DataLoader

# トークナイザーとプロセッサーをロード
from transformers import AutoTokenizer, AutoProcessor

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    trust_remote_code=True
)

# データセットの設定
dataset = HybridDataset(
    base_image_dir="dataset",
    qwen_processor=processor,
    dataset="sem_seg",  # ADE20Kだけ使う
    sem_seg_data="ade20k",
    num_classes_per_sample=1
)

# 1サンプルだけ取得
sample = dataset[0]

print("=" * 50)
print("Dataset sample keys:")
for key in sample.keys():
    if isinstance(sample[key], torch.Tensor):
        print(f"  {key}: torch.Tensor {sample[key].shape}")
    elif isinstance(sample[key], (list, tuple)):
        print(f"  {key}: {type(sample[key])} len={len(sample[key])}")
    else:
        print(f"  {key}: {type(sample[key])}")

print("\nSpecific checks:")
print(f"  orig_hw: {sample.get('orig_hw')}")
print(f"  original_image: {type(sample.get('original_image'))}")

# Collatorのテスト
from transformers import AutoTokenizer, AutoProcessor

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    trust_remote_code=True
)

collator = MultiModalDataCollator(
    tokenizer=tokenizer,
    processor=processor
)

# バッチ化
batch = collator([sample])

print("\n" + "=" * 50)
print("Batch keys after collation:")
for key in batch.keys():
    if isinstance(batch[key], torch.Tensor):
        print(f"  {key}: torch.Tensor {batch[key].shape}")
    elif isinstance(batch[key], list):
        if len(batch[key]) > 0:
            if isinstance(batch[key][0], torch.Tensor):
                print(f"  {key}: list of torch.Tensor, first shape={batch[key][0].shape}")
            else:
                print(f"  {key}: list, first type={type(batch[key][0])}")
        else:
            print(f"  {key}: empty list")
    else:
        print(f"  {key}: {type(batch[key])}")

print("\nSpecific batch checks:")
print(f"  batch['orig_hw']: {batch.get('orig_hw')}")
print(f"  batch['original_images']: {type(batch.get('original_images'))}")
if 'original_images' in batch and batch['original_images'] is not None:
    if isinstance(batch['original_images'], list) and len(batch['original_images']) > 0:
        print(f"    First image type: {type(batch['original_images'][0])}")
        if hasattr(batch['original_images'][0], 'size'):
            print(f"    First image size: {batch['original_images'][0].size}")