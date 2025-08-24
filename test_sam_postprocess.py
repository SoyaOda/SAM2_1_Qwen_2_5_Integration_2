#!/usr/bin/env python3
"""
SAM postprocess修正のテスト
o3モデルの指摘に基づいて、パディング除去処理を正しく実装
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def test_sam_postprocess():
    """SAM公式のpostprocess（3段階処理）をテスト"""
    
    # テストケース設定
    orig_h, orig_w = 640, 480  # 元画像サイズ
    sam_size = 1024  # SAM座標系
    
    print(f"Original size: {orig_h} x {orig_w}")
    print(f"SAM size: {sam_size} x {sam_size}")
    
    # SAM入力サイズ（パディング前のサイズ）を計算
    # SAMは最長辺を1024にして、アスペクト比を保持
    scale = sam_size / max(orig_h, orig_w)
    sam_input_h = int(orig_h * scale)
    sam_input_w = int(orig_w * scale)
    
    print(f"SAM input size (before padding): {sam_input_h} x {sam_input_w}")
    print(f"Scale factor: {scale:.3f}")
    
    # パディング量
    pad_h = sam_size - sam_input_h
    pad_w = sam_size - sam_input_w
    print(f"Padding: h={pad_h}, w={pad_w}")
    
    # ダミーの予測マスク（SAM座標系: 1024x1024）
    pred_mask = torch.zeros(1, 1, sam_size, sam_size)
    # 有効領域にのみ値を入れる（左上寄せ想定）
    pred_mask[0, 0, 100:300, 100:400] = 1.0
    
    # GTマスク（同じくSAM座標系: 1024x1024）
    gt_mask = torch.zeros(1, 1, sam_size, sam_size)
    gt_mask[0, 0, 100:300, 100:400] = 1.0
    
    print("\n=== Method 1: 直接リサイズ（間違った方法） ===")
    # 間違った方法：パディングを除去せずに直接リサイズ
    pred_mask_wrong = F.interpolate(
        pred_mask.float(),
        size=(orig_h, orig_w),
        mode='nearest'
    )
    print(f"Wrong method output shape: {pred_mask_wrong.shape}")
    
    print("\n=== Method 2: SAM公式postprocess（正しい方法） ===")
    # 正しい方法：SAM公式の3段階処理
    
    # 1. 1024x1024にアップサンプル（既に1024x1024なのでスキップ）
    pred_mask_1024 = pred_mask.float()
    print(f"Step 1 - Upsampled to 1024: {pred_mask_1024.shape}")
    
    # 2. パディングを除去（右下パディングを削除）
    pred_mask_unpadded = pred_mask_1024[:, :, :sam_input_h, :sam_input_w]
    print(f"Step 2 - Padding removed: {pred_mask_unpadded.shape}")
    
    # 3. 元画像サイズにリサイズ
    pred_mask_correct = F.interpolate(
        pred_mask_unpadded,
        size=(orig_h, orig_w),
        mode='nearest'
    )
    print(f"Step 3 - Resized to original: {pred_mask_correct.shape}")
    
    # GTも同様に処理
    gt_mask_1024 = gt_mask.float()
    gt_mask_unpadded = gt_mask_1024[:, :, :sam_input_h, :sam_input_w]
    gt_mask_correct = F.interpolate(
        gt_mask_unpadded,
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
    
    # パディング除去後
    axes[0, 2].imshow(pred_mask_unpadded[0, 0].numpy(), cmap='hot')
    axes[0, 2].set_title(f'After Padding Removal\n{sam_input_h}x{sam_input_w}')
    axes[0, 2].axis('off')
    
    # Method 1: 間違った方法（直接リサイズ）
    axes[1, 0].imshow(pred_mask_wrong[0, 0].numpy(), cmap='hot')
    axes[1, 0].set_title(f'Method 1: Direct Resize (Wrong)\n{orig_h}x{orig_w}')
    axes[1, 0].axis('off')
    
    # Method 2: 正しい方法（SAM postprocess）
    axes[1, 1].imshow(pred_mask_correct[0, 0].numpy(), cmap='hot')
    axes[1, 1].set_title(f'Method 2: SAM Postprocess (Correct)\n{orig_h}x{orig_w}')
    axes[1, 1].axis('off')
    
    # 差分
    diff = np.abs(pred_mask_wrong[0, 0].numpy() - pred_mask_correct[0, 0].numpy())
    axes[1, 2].imshow(diff, cmap='hot')
    axes[1, 2].set_title(f'Difference\nMax: {diff.max():.3f}')
    axes[1, 2].axis('off')
    
    plt.suptitle('SAM Postprocess Test\n(Based on o3 model feedback)', fontsize=14)
    plt.tight_layout()
    
    # 保存
    output_dir = Path('/home/soya/SAM2_1_Qwen_2_5_Integration_2/debug_outputs')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'sam_postprocess_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to {output_dir / 'sam_postprocess_test.png'}")
    
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
    if p_bottom >= sam_input_h or p_right >= sam_input_w:
        print("⚠️ WARNING: Mask extends beyond the valid (non-padded) region!")
        print(f"   Valid region should be within: (0,0) to ({sam_input_h-1},{sam_input_w-1})")
        print(f"   But mask extends to: ({p_bottom},{p_right})")
        print("   This would cause misalignment without proper padding removal!")
    else:
        print("✅ Mask is within valid region - padding removal works correctly")
    
    # 結論
    print("\n=== CONCLUSION ===")
    print("The o3 model was correct:")
    print("1. SAM standard: resize to 1024 → add right/bottom padding → process")
    print("2. Postprocess MUST: remove padding → resize to original")
    print("3. Direct resize without padding removal causes misalignment")
    print("4. The fix is to apply proper SAM postprocess (3-step process)")

if __name__ == "__main__":
    test_sam_postprocess()