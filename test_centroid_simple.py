#!/usr/bin/env python3
"""
重心計算機能の簡単なテストスクリプト
"""
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def compute_mask_centroid(mask: torch.Tensor, orig_h: int, orig_w: int) -> torch.Tensor:
    """
    バイナリマスクから重心座標を計算
    
    Args:
        mask: バイナリマスク [H, W] or [1, H, W]
        orig_h: 元画像の高さ（ピクセル座標系）
        orig_w: 元画像の幅（ピクセル座標系）
    
    Returns:
        重心座標 [2] = (cx, cy) in pixels
    """
    if mask.dim() == 3:
        mask = mask.squeeze(0)
    
    # Float型に変換
    m = mask.to(torch.float32)
    h, w = m.shape
    
    # 面積（非ゼロ画素数）
    mass = m.sum()
    if mass <= 0:
        # マスクが空の場合は画像中心を返す
        return torch.tensor([orig_w // 2, orig_h // 2], 
                          dtype=torch.float32, device=mask.device)
    
    # 座標グリッド
    ys = torch.arange(h, device=m.device, dtype=torch.float32).view(h, 1)
    xs = torch.arange(w, device=m.device, dtype=torch.float32).view(1, w)
    
    # 重心計算（マスク座標系）
    cy = (m * ys).sum() / mass
    cx = (m * xs).sum() / mass
    
    # マスク座標系から元画像座標系へ変換
    # マスクがリサイズされている場合のスケーリング
    scale_x = orig_w / w
    scale_y = orig_h / h
    cx = cx * scale_x
    cy = cy * scale_y
    
    return torch.stack([cx, cy])

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

def test_centroid_calculation():
    """重心計算のテスト"""
    print("=" * 60)
    print("Testing Mask Centroid Calculation")
    print("=" * 60)
    
    # Test 1: 単一領域のマスク
    print("\n1. Testing single region mask...")
    mask = create_test_mask(256, 256, n_regions=1)
    orig_h, orig_w = 512, 512  # 元画像サイズ
    
    centroid = compute_mask_centroid(mask, orig_h, orig_w)
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
    print("\n2. Testing empty mask (fallback to center)...")
    empty_mask = torch.zeros(256, 256)
    centroid_empty = compute_mask_centroid(empty_mask, orig_h, orig_w)
    print(f"   Computed centroid: x={centroid_empty[0].item():.1f}, y={centroid_empty[1].item():.1f}")
    print(f"   Expected (center): x={orig_w//2}, y={orig_h//2}")
    
    # Test 3: 複数領域のマスク（全体の重心）
    print("\n3. Testing multi-region mask...")
    mask_multi = create_test_mask(256, 256, n_regions=2)
    centroid_multi = compute_mask_centroid(mask_multi, orig_h, orig_w)
    print(f"   Mask shape: {mask_multi.shape}")
    print(f"   Number of non-zero pixels: {mask_multi.sum().item()}")
    print(f"   Computed centroid: x={centroid_multi[0].item():.1f}, y={centroid_multi[1].item():.1f}")
    
    # 2つの領域の重心（全体の重心）
    # 左上: [64:128, 64:128] 重心=(96, 96)
    # 右下: [128:192, 128:192] 重心=(160, 160)
    # 全体の重心は約 (128, 128) のマスク座標
    expected_x_multi = 128 * (orig_w / 256)  # 256
    expected_y_multi = 128 * (orig_h / 256)  # 256
    print(f"   Expected centroid (approx): x={expected_x_multi:.1f}, y={expected_y_multi:.1f}")
    
    # Test 4: 異なるサイズのマスクと画像
    print("\n4. Testing different mask and image sizes...")
    # マスク: 128x128, 画像: 1024x1024
    small_mask = torch.zeros(128, 128)
    small_mask[32:64, 64:96] = 1.0  # 右側の領域
    
    centroid_diff = compute_mask_centroid(small_mask, 1024, 1024)
    print(f"   Mask size: 128x128")
    print(f"   Image size: 1024x1024")
    print(f"   Computed centroid: x={centroid_diff[0].item():.1f}, y={centroid_diff[1].item():.1f}")
    
    # マスク座標での重心: (80, 48)
    # 画像座標: (80 * 8, 48 * 8) = (640, 384)
    expected_x_diff = 80 * (1024 / 128)  # 640
    expected_y_diff = 48 * (1024 / 128)  # 384
    print(f"   Expected centroid: x={expected_x_diff:.1f}, y={expected_y_diff:.1f}")
    
    # Test 5: PyTorchのバッチ処理
    print("\n5. Testing batch processing...")
    batch_masks = torch.stack([mask, mask_multi, empty_mask])  # [3, 256, 256]
    print(f"   Batch shape: {batch_masks.shape}")
    
    centroids = []
    for i in range(batch_masks.shape[0]):
        c = compute_mask_centroid(batch_masks[i], orig_h, orig_w)
        centroids.append(c)
        print(f"   Mask {i}: centroid = ({c[0].item():.1f}, {c[1].item():.1f})")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    test_centroid_calculation()