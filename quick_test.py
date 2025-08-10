#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

print("Quick Dataset Randomness Test")
print("=" * 40)

# データセット初期化
from src.config import LISAConfig
from src.data.sem_seg_dataset import SemSegDataset
from transformers import AutoTokenizer

lisa_config = LISAConfig()
print(f"Dataset path: {lisa_config.dataset_base_dir}")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)

print("Creating dataset...")
dataset = SemSegDataset(
    base_image_dir=lisa_config.dataset_base_dir,
    tokenizer=tokenizer,
    samples_per_epoch=10,
    sem_seg_data="ade20k"
)

print(f"Dataset length: {len(dataset)}")

# 同じインデックスで5回アクセス
print("\nAccessing index 0 five times:")
paths = []
for i in range(5):
    sample = dataset[0]
    if isinstance(sample, tuple) and len(sample) > 0:
        image_path = sample[0] if isinstance(sample[0], str) else "Unknown"
        name = Path(image_path).name if image_path != "Unknown" else image_path
        paths.append(image_path)
        print(f"  Access {i+1}: {name}")

# チェック
unique_paths = set(paths)
print(f"\nResult: {len(unique_paths)}/{len(paths)} unique samples")

if len(unique_paths) == len(paths):
    print("✓ SUCCESS: Different samples each time (Original LISA behavior)")
else:
    print("✗ FAILURE: Same sample returned (Not following Original LISA)")