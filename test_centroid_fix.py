#!/usr/bin/env python3
"""
重心ベースのポイントプロンプト修正のテストスクリプト
"""
import torch
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.models.lisa_model import LISA_Model
from src.config import LISAConfig
import warnings
warnings.filterwarnings('ignore')

def create_test_mask(h=256, w=256, n_regions=1):
    """テスト用のマスクを作成（複数領域対応）"""
    mask = torch.zeros(h, w)
    
    if n_regions == 1:
        # 単一領域（右上）
        mask[h//4:h//2, w//2:3*w//4] = 1.0
    else:
        # 複数領域
        # 左上
        mask[h//4:h//2, w//4:w//2] = 1.0
        # 右下
        mask[h//2:3*h//4, w//2:3*w//4] = 1.0
    
    return mask

def test_compute_mask_centroid():
    """compute_mask_centroid関数のテスト"""
    print("=" * 60)
    print("Testing compute_mask_centroid function")
    print("=" * 60)
    
    # Initialize model
    config = LISAConfig()
    config.qwen_model_name = "Qwen/Qwen2-VL-2B-Instruct"
    config.sam_model_name = "./checkpoints/sam2.1_hiera_large.pt"
    config.use_token_fpn = False  # シンプルなテストのため無効化
    
    # モデルを初期化（軽量モデルでテスト）
    print("\n1. Initializing LISA model...")
    model = LISA_Model(config)
    
    # Test 1: 単一領域のマスク
    print("\n2. Testing single region mask...")
    mask = create_test_mask(256, 256, n_regions=1)
    orig_h, orig_w = 512, 512  # 元画像サイズ
    
    centroid = model.compute_mask_centroid(mask, orig_h, orig_w)
    print(f"   Mask shape: {mask.shape}")
    print(f"   Original image size: {orig_h}x{orig_w}")
    print(f"   Computed centroid: x={centroid[0].item():.1f}, y={centroid[1].item():.1f}")
    
    # 期待される重心位置の確認（右上領域）
    # マスクは [64:128, 128:192] の範囲
    # 重心は約 (160, 96) のマスク座標
    # 元画像座標では 2倍になるので (320, 192)
    expected_x = (128 + 192) / 2 * (orig_w / 256)  # 160 * 2 = 320
    expected_y = (64 + 128) / 2 * (orig_h / 256)   # 96 * 2 = 192
    print(f"   Expected centroid: x={expected_x:.1f}, y={expected_y:.1f}")
    
    # Test 2: 空のマスク（フォールバック）
    print("\n3. Testing empty mask (fallback to center)...")
    empty_mask = torch.zeros(256, 256)
    centroid_empty = model.compute_mask_centroid(empty_mask, orig_h, orig_w)
    print(f"   Computed centroid: x={centroid_empty[0].item():.1f}, y={centroid_empty[1].item():.1f}")
    print(f"   Expected (center): x={orig_w//2}, y={orig_h//2}")
    
    # Test 3: 複数領域のマスク
    print("\n4. Testing multi-region mask...")
    mask_multi = create_test_mask(256, 256, n_regions=2)
    centroid_multi = model.compute_mask_centroid(mask_multi, orig_h, orig_w)
    print(f"   Mask shape: {mask_multi.shape}")
    print(f"   Computed centroid: x={centroid_multi[0].item():.1f}, y={centroid_multi[1].item():.1f}")
    
    # 2つの領域の重心（全体の重心）
    # 左上: [64:128, 64:128] 重心=(96, 96)
    # 右下: [128:192, 128:192] 重心=(160, 160)
    # 全体の重心は約 (128, 128) のマスク座標
    expected_x_multi = 128 * (orig_w / 256)  # 256
    expected_y_multi = 128 * (orig_h / 256)  # 256
    print(f"   Expected centroid (approx): x={expected_x_multi:.1f}, y={expected_y_multi:.1f}")
    
    # Test 4: Forward passでの統合テスト
    print("\n5. Testing in forward pass...")
    
    # ダミーの入力を準備
    B = 1
    seq_len = 100
    input_ids = torch.randint(0, 1000, (B, seq_len))
    
    # SEGトークンを追加（位置50）
    seg_token_id = 151859  # Qwen2.5-VLのSEGトークンID
    input_ids[0, 50] = seg_token_id
    model.seg_token_id = seg_token_id
    
    # 画像とマスクを準備
    pixel_values = torch.randn(B, 3, 448, 448)
    mask_labels = [mask.unsqueeze(0)]  # [1, H, W]形式
    labels = input_ids.clone()
    
    # Forward pass
    try:
        outputs = model.forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            labels=labels,
            mask_labels=mask_labels,
            return_dict=True
        )
        print("   Forward pass successful!")
        print(f"   Output logits shape: {outputs.logits.shape}")
        if outputs.mask_logits and outputs.mask_logits[0]:
            print(f"   Generated mask shape: {outputs.mask_logits[0][0].shape}")
    except Exception as e:
        print(f"   Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    test_compute_mask_centroid()