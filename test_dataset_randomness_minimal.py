#!/usr/bin/env python3
"""
データセットのランダム性を確認する最小限のテストスクリプト
"""

import sys
import os
from pathlib import Path

# Suppress warnings
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_randomness():
    """最小限のランダム性テスト"""
    
    print("=" * 50)
    print("Minimal Dataset Randomness Test")
    print("=" * 50)
    
    # sem_seg_datasetを直接インポートしてテスト
    print("\nStep 1: Import modules...")
    from src.data.sem_seg_dataset import SemSegDataset
    from transformers import AutoTokenizer
    
    print("Step 2: Load tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
    
    print("Step 3: Initialize SemSegDataset...")
    dataset = SemSegDataset(
        base_image_dir="/mnt/h/download/LISA-dataset/data/dataset",
        tokenizer=tokenizer,
        samples_per_epoch=10,
        sem_seg_data="ade20k"  # 1つのデータセットのみ
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    print("\nStep 4: Test randomness...")
    # 同じインデックスで複数回アクセス
    samples = []
    for i in range(5):
        try:
            sample = dataset[0]  # 常にindex=0
            if isinstance(sample, tuple) and len(sample) > 0:
                image_path = sample[0] if isinstance(sample[0], str) else 'Unknown'
                image_name = Path(image_path).name if image_path != 'Unknown' else image_path
                samples.append(image_path)
                print(f"  Access {i+1}: {image_name}")
        except Exception as e:
            print(f"  Access {i+1}: Error - {e}")
    
    # 結果の確認
    unique_samples = set(samples)
    print(f"\nResult: {len(unique_samples)}/{len(samples)} unique samples")
    
    if len(unique_samples) == len(samples):
        print("✓ SUCCESS: Random sampling working (Original LISA behavior)")
        return True
    elif len(unique_samples) == 1:
        print("✗ FAILURE: Same sample returned (Deterministic behavior)")
        return False
    else:
        print("△ PARTIAL: Some samples repeated")
        return False

if __name__ == "__main__":
    try:
        success = test_randomness()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)