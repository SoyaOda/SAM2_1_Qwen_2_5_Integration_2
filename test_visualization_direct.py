#!/usr/bin/env python3
"""
可視化の座標系問題を直接テスト
実際の学習コードの可視化部分だけを抽出してテスト
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def test_visualization_padding():
    """パディング除去処理をテスト"""
    
    # テストケース設定
    orig_h, orig_w = 640, 480  # 元画像サイズ
    sam_size = 1024  # SAM座標系
    
    print(f"Original size: {orig_h} x {orig_w}")
    print(f"SAM size: {sam_size} x {sam_size}")
    
    # SAM座標系でのスケール計算
    scale = sam_size / max(orig_h, orig_w)
    new_h = int(orig_h * scale)
    new_w = int(orig_w * scale)
    
    print(f"Scaled size: {new_h} x {new_w}")
    print(f"Scale factor: {scale:.3f}")
    
    # パディング量
    pad_h = sam_size - new_h
    pad_w = sam_size - new_w
    print(f"Padding: h={pad_h}, w={pad_w}")
    
    # ダミーの予測マスク（SAM座標系: 1024x1024）
    pred_mask = torch.zeros(1, 1, sam_size, sam_size)
    # 有効領域にのみ値を入れる（左上寄せ想定）
    pred_mask[0, 0, 100:300, 100:400] = 1.0
    
    # GTマスク（同じくSAM座標系: 1024x1024）
    gt_mask = torch.zeros(1, 1, sam_size, sam_size)
    gt_mask[0, 0, 100:300, 100:400] = 1.0
    
    print("\n=== Method 1: パディング除去（現在の実装） ===")
    # 現在のsave_visualizationの実装
    pred_mask_unpadded = pred_mask[:, :, :new_h, :new_w]
    print(f"Unpadded pred shape: {pred_mask_unpadded.shape}")
    
    pred_mask_processed = F.interpolate(
        pred_mask_unpadded.float(),
        size=(orig_h, orig_w),
        mode='nearest'
    )
    print(f"Resized pred shape: {pred_mask_processed.shape}")
    
    # GTも同様
    gt_mask_unpadded = gt_mask[:, :, :new_h, :new_w]
    gt_mask_processed = F.interpolate(
        gt_mask_unpadded.float(),
        size=(orig_h, orig_w),
        mode='nearest'
    )
    
    print("\n=== Method 2: 直接リサイズ（修正案） ===")
    # パディングを除去せずに直接リサイズ
    pred_mask_direct = F.interpolate(
        pred_mask.float(),
        size=(orig_h, orig_w),
        mode='nearest'
    )
    print(f"Direct resize pred shape: {pred_mask_direct.shape}")
    
    gt_mask_direct = F.interpolate(
        gt_mask.float(),
        size=(orig_h, orig_w),
        mode='nearest'
    )
    
    # 可視化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 元のSAM座標系マスク
    axes[0, 0].imshow(pred_mask[0, 0].numpy(), cmap='hot')
    axes[0, 0].set_title(f'Pred Mask (SAM)\n{sam_size}x{sam_size}')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gt_mask[0, 0].numpy(), cmap='hot')
    axes[0, 1].set_title(f'GT Mask (SAM)\n{sam_size}x{sam_size}')
    axes[0, 1].axis('off')
    
    # Method 1: パディング除去後
    axes[0, 2].imshow(pred_mask_processed[0, 0].numpy(), cmap='hot')
    axes[0, 2].set_title(f'Method 1: Unpad+Resize\n{orig_h}x{orig_w}')
    axes[0, 2].axis('off')
    
    # Method 2: 直接リサイズ
    axes[1, 0].imshow(pred_mask_direct[0, 0].numpy(), cmap='hot')
    axes[1, 0].set_title(f'Method 2: Direct Resize\n{orig_h}x{orig_w}')
    axes[1, 0].axis('off')
    
    # 差分
    diff1 = np.abs(pred_mask_processed[0, 0].numpy() - gt_mask_processed[0, 0].numpy())
    axes[1, 1].imshow(diff1, cmap='hot')
    axes[1, 1].set_title(f'Diff (Method 1)\nMax: {diff1.max():.3f}')
    axes[1, 1].axis('off')
    
    diff2 = np.abs(pred_mask_direct[0, 0].numpy() - gt_mask_direct[0, 0].numpy())
    axes[1, 2].imshow(diff2, cmap='hot')
    axes[1, 2].set_title(f'Diff (Method 2)\nMax: {diff2.max():.3f}')
    axes[1, 2].axis('off')
    
    plt.suptitle('Visualization Padding Test', fontsize=14)
    plt.tight_layout()
    
    # 保存
    output_dir = Path('/home/soya/SAM2_1_Qwen_2_5_Integration_2/debug_outputs')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'padding_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to {output_dir / 'padding_test.png'}")
    
    # 診断
    print("\n=== DIAGNOSIS ===")
    
    # 有効領域の検出
    pred_np = pred_mask[0, 0].numpy()
    pred_rows = np.any(pred_np > 0, axis=1)
    pred_cols = np.any(pred_np > 0, axis=0)
    
    if np.any(pred_rows):
        pred_valid_rows = np.where(pred_rows)[0]
        p_top, p_bottom = pred_valid_rows[0], pred_valid_rows[-1]
    else:
        p_top, p_bottom = 0, sam_size-1
        
    if np.any(pred_cols):
        pred_valid_cols = np.where(pred_cols)[0]
        p_left, p_right = pred_valid_cols[0], pred_valid_cols[-1]
    else:
        p_left, p_right = 0, sam_size-1
    
    print(f"Pred mask valid region in SAM coords: ({p_top},{p_left}) to ({p_bottom},{p_right})")
    print(f"Pred mask valid size: {p_bottom-p_top+1} x {p_right-p_left+1}")
    
    # パディング問題の診断
    if p_bottom >= new_h or p_right >= new_w:
        print("⚠️ WARNING: Mask extends beyond the valid (non-padded) region!")
        print(f"   Valid region should be within: (0,0) to ({new_h-1},{new_w-1})")
        print(f"   But mask extends to: ({p_bottom},{p_right})")
        print("   This will cause misalignment when unpadding!")
    else:
        print("✅ Mask is within valid region - unpadding should work correctly")

if __name__ == "__main__":
    test_visualization_padding()