#!/usr/bin/env python3
"""
マスクの閾値問題をデバッグ
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def test_mask_threshold():
    """マスクの閾値処理をテスト"""
    
    # ダミーのロジット値を生成（実際のモデル出力を模擬）
    # ロジットは通常-10から10の範囲
    logits = torch.randn(1, 1, 256, 256) * 3.0  # 標準偏差3のランダムロジット
    
    # 一部を明確にポジティブ/ネガティブにする
    logits[0, 0, 50:150, 50:150] = 5.0  # 明確にポジティブ（物体領域）
    logits[0, 0, 150:200, 150:200] = -5.0  # 明確にネガティブ（背景）
    
    print("Logits stats:")
    print(f"  Min: {logits.min():.3f}")
    print(f"  Max: {logits.max():.3f}")
    print(f"  Mean: {logits.mean():.3f}")
    print(f"  Std: {logits.std():.3f}")
    
    # シグモイドを適用
    probs = torch.sigmoid(logits)
    probs_np = probs.squeeze().numpy()
    
    print("\nAfter sigmoid:")
    print(f"  Min: {probs_np.min():.3f}")
    print(f"  Max: {probs_np.max():.3f}")
    print(f"  Mean: {probs_np.mean():.3f}")
    print(f"  Std: {probs_np.std():.3f}")
    
    # 異なる閾値で二値化
    thresholds = [0.0, 0.3, 0.5, 0.7]
    binary_masks = {}
    
    for thresh in thresholds:
        binary = (probs_np > thresh).astype(float)
        binary_masks[thresh] = binary
        positive_ratio = binary.mean()
        print(f"\nThreshold {thresh}:")
        print(f"  Positive pixels: {positive_ratio*100:.1f}%")
    
    # 可視化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 元のロジット
    im1 = axes[0, 0].imshow(logits.squeeze().numpy(), cmap='RdBu_r', vmin=-5, vmax=5)
    axes[0, 0].set_title('Raw Logits')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
    
    # シグモイド後
    im2 = axes[0, 1].imshow(probs_np, cmap='jet', vmin=0, vmax=1)
    axes[0, 1].set_title('After Sigmoid (Probabilities)')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    
    # ヒストグラム
    axes[0, 2].hist(probs_np.flatten(), bins=50, edgecolor='black')
    axes[0, 2].set_title('Probability Distribution')
    axes[0, 2].set_xlabel('Probability')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].axvline(x=0.0, color='r', linestyle='--', label='thresh=0.0')
    axes[0, 2].axvline(x=0.5, color='g', linestyle='--', label='thresh=0.5')
    axes[0, 2].legend()
    
    # 異なる閾値での二値化結果
    for i, thresh in enumerate([0.0, 0.5, 0.7]):
        axes[1, i].imshow(binary_masks[thresh], cmap='gray', vmin=0, vmax=1)
        positive_ratio = binary_masks[thresh].mean()
        axes[1, i].set_title(f'Threshold={thresh}\n({positive_ratio*100:.1f}% positive)')
        axes[1, i].axis('off')
    
    plt.suptitle('Mask Threshold Analysis', fontsize=14)
    plt.tight_layout()
    
    # 保存
    output_dir = Path('/home/soya/SAM2_1_Qwen_2_5_Integration_2/debug_outputs')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'mask_threshold_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to {output_dir / 'mask_threshold_test.png'}")
    
    # 診断
    print("\n=== DIAGNOSIS ===")
    print("Problem: Using threshold=0.0 on sigmoid outputs (0-1 range)")
    print("  - Sigmoid(0) = 0.5, so any logit > 0 becomes positive")
    print("  - This causes most pixels to be classified as positive")
    print("Solution: Use threshold=0.5 for sigmoid outputs")
    print("  - This corresponds to logit=0 decision boundary")
    print("  - Standard practice in binary classification")

if __name__ == "__main__":
    test_mask_threshold()