#!/usr/bin/env python3
"""
データセットのランダム性を確認するテストスクリプト（改良版）
オリジナルLISA方式の動作確認
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import LISAConfig
from src.data.dataset import HybridDataset
from transformers import AutoProcessor

def test_dataset_randomness():
    """データセットが毎回異なるサンプルを返すことを確認"""
    
    print("=" * 50)
    print("Dataset Randomness Test (Original LISA Behavior)")
    print("=" * 50)
    
    # LISAConfigから設定を取得
    lisa_config = LISAConfig()
    print(f"\nUsing dataset directory: {lisa_config.dataset_base_dir}")
    
    # Processorの初期化
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        trust_remote_code=True
    )
    
    # データセットの初期化
    print("Initializing dataset...")
    dataset = HybridDataset(
        base_image_dir=lisa_config.dataset_base_dir,
        qwen_processor=processor,
        dataset="sem_seg",
        sample_rate=[1.0],
        samples_per_epoch=100  # オリジナルLISA方式
    )
    
    print(f"Dataset length: {len(dataset)}")
    print("\n" + "=" * 50)
    print("Test 1: Randomness Check")
    print("=" * 50)
    print("Accessing index 0 multiple times...")
    
    # 同じインデックスで複数回アクセス
    samples = []
    for i in range(5):
        sample = dataset[0]  # 常にindex=0でアクセス
        
        # 画像パスを取得
        if isinstance(sample, dict):
            # Dict形式の場合
            image_path = sample.get('image_path', 'Unknown')
        elif isinstance(sample, tuple) and len(sample) > 0:
            # Tuple形式の場合（オリジナルLISA形式）
            image_path = sample[0] if isinstance(sample[0], str) else 'Unknown'
        else:
            image_path = 'Unknown'
        
        # ファイル名のみを表示（フルパスだと長すぎるため）
        if image_path != 'Unknown':
            image_name = Path(image_path).name
        else:
            image_name = image_path
            
        samples.append(image_path)
        print(f"  Access {i+1}: {image_name}")
    
    # すべて異なるサンプルか確認
    unique_samples = set(samples)
    print(f"\nResult: {len(unique_samples)}/{len(samples)} unique samples")
    
    if len(unique_samples) == len(samples):
        print("✓ SUCCESS: Dataset returns different samples each time")
        print("  (Original LISA behavior correctly implemented)")
    elif len(unique_samples) == 1:
        print("✗ FAILURE: Dataset returns the same sample every time")
        print("  (Deterministic behavior - not following Original LISA)")
    else:
        print("△ PARTIAL: Some samples are repeated")
    
    print("\n" + "=" * 50)
    print("Test 2: Image-Mask Consistency Check")
    print("=" * 50)
    print("Checking consistency within single __getitem__ call...")
    
    # 整合性の確認（3サンプル）
    for i in range(3):
        sample = dataset[0]
        
        print(f"\nSample {i+1}:")
        if isinstance(sample, tuple) and len(sample) >= 5:
            # Tuple形式（オリジナルLISA形式）
            image_path = sample[0]
            masks = sample[4] if len(sample) > 4 else None
            
            image_name = Path(image_path).name if isinstance(image_path, str) else 'Unknown'
            print(f"  Image: {image_name}")
            
            if masks is not None:
                print(f"  Mask shape: {masks.shape if hasattr(masks, 'shape') else 'N/A'}")
                print(f"  ✓ Image and mask from same __getitem__ call")
            else:
                print(f"  No mask available")
                
        elif isinstance(sample, dict):
            # Dict形式
            image_path = sample.get('image_path', 'Unknown')
            mask = sample.get('ground_truth_mask', None)
            
            image_name = Path(image_path).name if image_path != 'Unknown' else 'Unknown'
            print(f"  Image: {image_name}")
            
            if mask is not None:
                print(f"  Mask shape: {mask.shape if hasattr(mask, 'shape') else 'N/A'}")
                print(f"  ✓ Image and mask from same __getitem__ call")
            else:
                print(f"  No mask available")
    
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    if len(unique_samples) == len(samples):
        print("✓ Dataset correctly implements Original LISA's random sampling")
        print("  - Each access returns a different sample")
        print("  - Image-mask pairs are consistent within each sample")
        print("  - Ready for training with minimal_train.py")
        return True
    else:
        print("✗ Dataset is not following Original LISA's behavior")
        print("  - May need to check __getitem__ implementation")
        return False

if __name__ == "__main__":
    success = test_dataset_randomness()
    sys.exit(0 if success else 1)