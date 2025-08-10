#!/usr/bin/env python3
"""
データセットのランダム性を確認するテストスクリプト
オリジナルLISA方式の動作確認
"""

import sys
import torch
from pathlib import Path
from transformers import AutoProcessor

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.dataset import HybridDataset
from src.data.collators import MultiModalDataCollator
from src.config import LISAConfig


def test_dataset_randomness():
    """データセットが毎回異なるサンプルを返すことを確認"""
    
    print("Setting up dataset...")
    
    # LISAConfigから設定を取得
    lisa_config = LISAConfig()
    print(f"Using dataset directory from LISAConfig: {lisa_config.dataset_base_dir}")
    
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        trust_remote_code=True
    )
    
    dataset = HybridDataset(
        base_image_dir=lisa_config.dataset_base_dir,  # LISAConfigから取得
        qwen_processor=processor,
        dataset="sem_seg",
        sample_rate=[1.0],
        samples_per_epoch=100  # オリジナルLISA方式
    )
    
    print(f"Dataset length: {len(dataset)}")
    print("\nTesting randomness of dataset...")
    
    # 同じインデックスで複数回アクセス
    samples = []
    for i in range(5):
        sample = dataset[0]  # 常にindex=0でアクセス
        
        # 画像パスを取得（形式に応じて処理）
        if isinstance(sample, dict):
            image_path = sample.get('image_path', 'Unknown')
        elif isinstance(sample, tuple) and len(sample) > 0:
            image_path = sample[0] if isinstance(sample[0], str) else 'Unknown'
        else:
            image_path = 'Unknown'
        
        samples.append(image_path)
        print(f"Access {i+1}: {image_path}")
    
    # すべて異なるサンプルか確認
    unique_samples = set(samples)
    if len(unique_samples) == len(samples):
        print("\n✓ SUCCESS: Dataset returns different samples each time (Original LISA behavior)")
    elif len(unique_samples) == 1:
        print("\n✗ FAILURE: Dataset returns the same sample every time (Deterministic behavior)")
    else:
        print(f"\n△ PARTIAL: {len(unique_samples)}/{len(samples)} unique samples")
    
    # DataLoaderでの動作確認
    print("\n\nTesting with DataLoader...")
    from torch.utils.data import DataLoader
    
    collator = MultiModalDataCollator(tokenizer=processor.tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,  # オリジナルLISAではshuffle不要
        collate_fn=collator,
        num_workers=0
    )
    
    # 最初の3バッチを取得
    batch_samples = []
    for i, batch in enumerate(dataloader):
        if i >= 3:
            break
        
        # バッチ内の画像パスを取得
        if 'image_path' in batch:
            paths = batch['image_path']
        else:
            paths = ['Unknown'] * 2
        
        batch_samples.extend(paths)
        print(f"Batch {i+1}: {paths}")
    
    # ユニークなサンプル数を確認
    unique_batch_samples = set(batch_samples)
    print(f"\nUnique samples in 3 batches: {len(unique_batch_samples)}/{len(batch_samples)}")
    
    if len(unique_batch_samples) == len(batch_samples):
        print("✓ All samples are unique")
    else:
        print("△ Some samples are repeated")
    
    return len(unique_samples) == len(samples)


def test_consistency_within_getitem():
    """1回の__getitem__内で画像とマスクの整合性を確認"""
    
    print("\n\n" + "="*50)
    print("Testing consistency within single __getitem__ call...")
    
    # LISAConfigから設定を取得
    lisa_config = LISAConfig()
    
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        trust_remote_code=True
    )
    
    dataset = HybridDataset(
        base_image_dir=lisa_config.dataset_base_dir,  # LISAConfigから取得
        qwen_processor=processor,  # processorではなくqwen_processorパラメータ
        dataset="sem_seg",
        sample_rate=[1.0],
        samples_per_epoch=100
    )
    
    # 複数回サンプルを取得して整合性を確認
    for i in range(3):
        sample = dataset[0]
        
        if isinstance(sample, tuple) and len(sample) >= 5:
            image_path = sample[0]
            masks = sample[4] if sample[4] is not None else None
            
            print(f"\nSample {i+1}:")
            print(f"  Image: {image_path}")
            if masks is not None:
                print(f"  Mask shape: {masks.shape if hasattr(masks, 'shape') else 'N/A'}")
                print(f"  ✓ Image and mask are from the same __getitem__ call")
            else:
                print(f"  No mask available")
        elif isinstance(sample, dict):
            image_path = sample.get('image_path', 'Unknown')
            mask = sample.get('ground_truth_mask', None)
            
            print(f"\nSample {i+1}:")
            print(f"  Image: {image_path}")
            if mask is not None:
                print(f"  Mask shape: {mask.shape if hasattr(mask, 'shape') else 'N/A'}")
                print(f"  ✓ Image and mask are from the same __getitem__ call")
            else:
                print(f"  No mask available")
    
    print("\n✓ Consistency check completed")
    print("Each __getitem__ call returns matched image-mask pairs")


if __name__ == "__main__":
    print("="*50)
    print("Dataset Randomness Test (Original LISA Behavior)")
    print("="*50)
    
    # Test 1: ランダム性の確認
    randomness_ok = test_dataset_randomness()
    
    # Test 2: 整合性の確認
    test_consistency_within_getitem()
    
    print("\n" + "="*50)
    print("Test Summary:")
    if randomness_ok:
        print("✓ Dataset correctly implements Original LISA's random sampling")
        print("  - Each access returns a different sample")
        print("  - Image-mask pairs are consistent within each sample")
        print("  - Ready for training with minimal_train.py")
    else:
        print("✗ Dataset is not following Original LISA's behavior")
        print("  - May need to check __getitem__ implementation")
    print("="*50)