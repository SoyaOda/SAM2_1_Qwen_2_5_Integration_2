#!/usr/bin/env python3
"""
データセット初期化のデバッグスクリプト
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_dataset_init():
    """データセット初期化を段階的にテスト"""
    
    print("=" * 50)
    print("Step 1: Import modules")
    print("=" * 50)
    
    from src.config import LISAConfig
    lisa_config = LISAConfig()
    print(f"✓ LISAConfig loaded")
    print(f"  Dataset path: {lisa_config.dataset_base_dir}")
    
    print("\n" + "=" * 50)
    print("Step 2: Import processor")
    print("=" * 50)
    
    from transformers import AutoProcessor
    print("✓ AutoProcessor imported")
    
    print("\n" + "=" * 50)
    print("Step 3: Load processor")
    print("=" * 50)
    
    print("Loading Qwen2-VL processor...")
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        trust_remote_code=True
    )
    print("✓ Processor loaded")
    
    print("\n" + "=" * 50)
    print("Step 4: Import HybridDataset")
    print("=" * 50)
    
    from src.data.dataset import HybridDataset
    print("✓ HybridDataset imported")
    
    print("\n" + "=" * 50)
    print("Step 5: Initialize HybridDataset")
    print("=" * 50)
    
    print("Creating HybridDataset instance...")
    print(f"  base_image_dir: {lisa_config.dataset_base_dir}")
    print(f"  dataset: sem_seg")
    print(f"  sample_rate: [1.0]")
    print(f"  samples_per_epoch: 10")
    
    try:
        dataset = HybridDataset(
            base_image_dir=lisa_config.dataset_base_dir,
            qwen_processor=processor,
            dataset="sem_seg",
            sample_rate=[1.0],
            samples_per_epoch=10  # 少ないサンプル数でテスト
        )
        print(f"✓ HybridDataset created with length: {len(dataset)}")
    except Exception as e:
        print(f"✗ Error creating HybridDataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 50)
    print("Step 6: Test __getitem__")
    print("=" * 50)
    
    print("Getting first sample...")
    try:
        sample = dataset[0]
        if isinstance(sample, tuple):
            print(f"✓ Got sample (tuple with {len(sample)} elements)")
            if len(sample) > 0 and isinstance(sample[0], str):
                print(f"  Image path: {sample[0]}")
        else:
            print(f"✓ Got sample (type: {type(sample)})")
    except Exception as e:
        print(f"✗ Error getting sample: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
    return True

if __name__ == "__main__":
    success = test_dataset_init()
    sys.exit(0 if success else 1)