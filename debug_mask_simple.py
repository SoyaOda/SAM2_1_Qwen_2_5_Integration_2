#!/usr/bin/env python3
"""
シンプルなマスク座標系デバッグスクリプト
ダミーデータを使って予測マスクとGTマスクの座標系を調査
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from PIL import Image

# ロギング設定
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# プロジェクトのインポート
import sys
sys.path.append('/home/soya/SAM2_1_Qwen_2_5_Integration_2')

from src.models.lisa_model import LISA_Model
from src.config import LISAConfig
from transformers import Qwen2VLProcessor

def create_dummy_batch(device='cuda'):
    """ダミーバッチデータを作成"""
    batch_size = 1
    
    # ダミー画像データ（オリジナルサイズ: 640x480）
    orig_h, orig_w = 640, 480
    
    # SAM用画像（1024x1024にパディング済み）
    sam_images = torch.randn(batch_size, 3, 1024, 1024).to(device)
    
    # Qwen用画像（448x448にリサイズ+パディング）
    qwen_h = 448
    qwen_w = 448
    pixel_values = torch.randn(batch_size, 3, qwen_h, qwen_w).to(device)
    
    # ダミーテキスト入力
    input_ids = torch.randint(0, 100, (batch_size, 50)).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    labels = torch.full_like(input_ids, -100).to(device)
    
    # SEGトークンを1箇所に挿入
    seg_token_id = 151665  # <SEG>トークンID
    input_ids[0, 25] = seg_token_id
    
    # GTマスク（SAM座標系: 1024x1024）
    mask_labels = torch.zeros(batch_size, 1024, 1024).to(device)
    # 中央に矩形マスクを作成（パディングを考慮）
    scale = 1024 / max(orig_h, orig_w)
    scaled_h = int(orig_h * scale)
    scaled_w = int(orig_w * scale)
    
    # パディング計算（右下パディング）
    pad_h = 1024 - scaled_h
    pad_w = 1024 - scaled_w
    
    # マスクの有効領域（パディング前の領域）
    mask_labels[0, 100:100+200, 100:100+300] = 1.0
    
    # グリッド情報（Qwen2.5-VL用）
    image_grid_thw = torch.tensor([[1, qwen_h//14, qwen_w//14]], dtype=torch.long).to(device)
    
    # orig_hw情報（元画像サイズ）
    orig_hw = [(orig_h, orig_w)]
    
    batch = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'pixel_values': pixel_values,
        'sam_images': sam_images,
        'mask_labels': mask_labels,
        'image_grid_thw': image_grid_thw,
        'orig_hw': orig_hw,
    }
    
    return batch, orig_h, orig_w

def analyze_mask(mask, name):
    """マスクの詳細分析"""
    logger.info(f"\n=== {name} ===")
    logger.info(f"  Shape: {mask.shape}")
    logger.info(f"  Min/Max: {mask.min():.3f} / {mask.max():.3f}")
    logger.info(f"  Mean/Std: {mask.mean():.3f} / {mask.std():.3f}")
    
    # パディング検出
    if mask.dim() >= 2:
        h, w = mask.shape[-2:]
        mask_np = mask.detach().cpu().numpy()
        if mask_np.ndim > 2:
            mask_np = mask_np.squeeze()
        
        # 各行・列の非ゼロ要素をチェック
        row_nonzero = np.any(np.abs(mask_np) > 1e-6, axis=1)
        col_nonzero = np.any(np.abs(mask_np) > 1e-6, axis=0)
        
        # 有効領域の境界を検出
        if np.any(row_nonzero):
            valid_rows = np.where(row_nonzero)[0]
            top, bottom = valid_rows[0], valid_rows[-1]
        else:
            top, bottom = 0, h-1
            
        if np.any(col_nonzero):
            valid_cols = np.where(col_nonzero)[0]
            left, right = valid_cols[0], valid_cols[-1]
        else:
            left, right = 0, w-1
        
        logger.info(f"  Valid region: ({top},{left}) to ({bottom},{right})")
        logger.info(f"  Valid size: {bottom-top+1} x {right-left+1}")
        logger.info(f"  Bottom padding: {h-1-bottom} rows")
        logger.info(f"  Right padding: {w-1-right} cols")
        
        return {
            'shape': mask.shape,
            'valid_region': (top, left, bottom, right),
            'valid_size': (bottom-top+1, right-left+1),
            'bottom_padding': h-1-bottom,
            'right_padding': w-1-right
        }
    
    return None

def visualize_comparison(gt_mask, pred_mask, orig_h, orig_w, save_path):
    """マスクの比較可視化"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # numpy変換
    gt_np = gt_mask.squeeze().detach().cpu().numpy()
    pred_np = pred_mask.squeeze().detach().cpu().numpy()
    
    # サブプロット1: GTマスク
    axes[0, 0].imshow(gt_np, cmap='hot')
    axes[0, 0].set_title(f'GT Mask\nShape: {gt_np.shape}')
    axes[0, 0].axis('off')
    
    # サブプロット2: 予測マスク（生）
    axes[0, 1].imshow(pred_np, cmap='hot')
    axes[0, 1].set_title(f'Pred Mask (Raw)\nShape: {pred_np.shape}')
    axes[0, 1].axis('off')
    
    # サブプロット3: 差分
    diff = np.abs(gt_np - pred_np)
    axes[0, 2].imshow(diff, cmap='hot')
    axes[0, 2].set_title(f'Difference\nMax: {diff.max():.3f}')
    axes[0, 2].axis('off')
    
    # パディング領域の可視化
    h, w = pred_np.shape
    
    # 予測マスクのパディング検出
    pred_nonzero = np.abs(pred_np) > 1e-6
    pred_rows = np.any(pred_nonzero, axis=1)
    pred_cols = np.any(pred_nonzero, axis=0)
    
    # GTマスクのパディング検出
    gt_nonzero = np.abs(gt_np) > 1e-6
    gt_rows = np.any(gt_nonzero, axis=1)
    gt_cols = np.any(gt_nonzero, axis=0)
    
    # パディング可視化（予測）
    pred_viz = np.zeros((h, w, 3))
    pred_viz[pred_nonzero] = [1, 1, 1]  # 白：有効領域
    for i in range(h):
        if not pred_rows[i]:
            pred_viz[i, :] = [1, 0, 0]  # 赤：パディング
    for j in range(w):
        if not pred_cols[j]:
            pred_viz[:, j] = [1, 0, 0]
    
    axes[1, 0].imshow(pred_viz)
    axes[1, 0].set_title('Pred Padding\n(Red = Padding)')
    axes[1, 0].axis('off')
    
    # パディング可視化（GT）
    gt_viz = np.zeros((h, w, 3))
    gt_viz[gt_nonzero] = [1, 1, 1]  # 白：有効領域
    for i in range(h):
        if not gt_rows[i]:
            gt_viz[i, :] = [0, 1, 0]  # 緑：パディング
    for j in range(w):
        if not gt_cols[j]:
            gt_viz[:, j] = [0, 1, 0]
    
    axes[1, 1].imshow(gt_viz)
    axes[1, 1].set_title('GT Padding\n(Green = Padding)')
    axes[1, 1].axis('off')
    
    # 有効領域の境界
    if np.any(pred_rows):
        pred_valid_rows = np.where(pred_rows)[0]
        p_top, p_bottom = pred_valid_rows[0], pred_valid_rows[-1]
    else:
        p_top, p_bottom = 0, h-1
    
    if np.any(pred_cols):
        pred_valid_cols = np.where(pred_cols)[0]
        p_left, p_right = pred_valid_cols[0], pred_valid_cols[-1]
    else:
        p_left, p_right = 0, w-1
    
    if np.any(gt_rows):
        gt_valid_rows = np.where(gt_rows)[0]
        g_top, g_bottom = gt_valid_rows[0], gt_valid_rows[-1]
    else:
        g_top, g_bottom = 0, h-1
        
    if np.any(gt_cols):
        gt_valid_cols = np.where(gt_cols)[0]
        g_left, g_right = gt_valid_cols[0], gt_valid_cols[-1]
    else:
        g_left, g_right = 0, w-1
    
    # 統計情報
    stats_text = f"Original size: {orig_h} x {orig_w}\n"
    stats_text += f"SAM size: 1024 x 1024\n\n"
    stats_text += f"Pred valid: ({p_top},{p_left}) to ({p_bottom},{p_right})\n"
    stats_text += f"Pred size: {p_bottom-p_top+1} x {p_right-p_left+1}\n\n"
    stats_text += f"GT valid: ({g_top},{g_left}) to ({g_bottom},{g_right})\n"
    stats_text += f"GT size: {g_bottom-g_top+1} x {g_right-g_left+1}\n\n"
    
    if (p_top, p_left, p_bottom, p_right) != (g_top, g_left, g_bottom, g_right):
        stats_text += "⚠️ MISALIGNMENT DETECTED!"
    
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=10, 
                    family='monospace', verticalalignment='center')
    axes[1, 2].set_title('Statistics')
    axes[1, 2].axis('off')
    
    plt.suptitle('Mask Coordinate System Debug', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved visualization to {save_path}")

def main():
    """メインのデバッグ処理"""
    # 出力ディレクトリ
    debug_dir = Path('/home/soya/SAM2_1_Qwen_2_5_Integration_2/debug_outputs')
    debug_dir.mkdir(exist_ok=True)
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # モデルの初期化
    logger.info("Loading model...")
    config = LISAConfig()
    model = LISA_Model(config)
    model = model.to(device)
    model.eval()
    
    # ダミーバッチの作成
    logger.info("Creating dummy batch...")
    batch, orig_h, orig_w = create_dummy_batch(device)
    
    logger.info(f"Original image size: {orig_h} x {orig_w}")
    logger.info(f"SAM image size: {batch['sam_images'].shape[-2:]} (should be 1024x1024)")
    logger.info(f"Qwen image size: {batch['pixel_values'].shape[-2:]}")
    
    # GTマスクの分析
    gt_info = analyze_mask(batch['mask_labels'][0], "GT Mask")
    
    # モデル推論
    logger.info("\n=== Running model inference ===")
    with torch.no_grad():
        outputs = model(
            input_ids=batch['input_ids'],
            pixel_values=batch['pixel_values'],
            sam_images=batch['sam_images'],
            labels=batch['labels'],
            attention_mask=batch['attention_mask'],
            image_grid_thw=batch['image_grid_thw']
        )
    
    # 予測マスクの分析
    if hasattr(outputs, 'mask_logits') and outputs.mask_logits is not None:
        if outputs.mask_logits[0] is not None and len(outputs.mask_logits[0]) > 0:
            pred_mask = outputs.mask_logits[0][0] if isinstance(outputs.mask_logits[0], list) else outputs.mask_logits[0]
            pred_info = analyze_mask(pred_mask, "Predicted Mask")
            
            # 可視化
            visualize_comparison(
                batch['mask_labels'][0],
                pred_mask,
                orig_h,
                orig_w,
                debug_dir / 'dummy_debug.png'
            )
            
            # 問題診断
            logger.info("\n=== DIAGNOSIS ===")
            if gt_info and pred_info:
                gt_region = gt_info['valid_region']
                pred_region = pred_info['valid_region']
                
                if gt_region != pred_region:
                    logger.warning("⚠️ COORDINATE MISALIGNMENT DETECTED!")
                    logger.warning(f"   GT valid region: {gt_region}")
                    logger.warning(f"   Pred valid region: {pred_region}")
                    
                    # オフセット計算
                    offset_top = pred_region[0] - gt_region[0]
                    offset_left = pred_region[1] - gt_region[1]
                    logger.warning(f"   Offset: (top={offset_top}, left={offset_left})")
                    
                    if pred_info['bottom_padding'] != gt_info['bottom_padding']:
                        logger.warning(f"   Bottom padding mismatch: Pred={pred_info['bottom_padding']} vs GT={gt_info['bottom_padding']}")
                    if pred_info['right_padding'] != gt_info['right_padding']:
                        logger.warning(f"   Right padding mismatch: Pred={pred_info['right_padding']} vs GT={gt_info['right_padding']}")
                else:
                    logger.info("✅ Coordinates are aligned correctly!")
        else:
            logger.error("No mask predictions generated!")
    else:
        logger.error("Model did not output mask_logits!")
    
    # compute_lossもテスト
    logger.info("\n=== Testing compute_loss ===")
    from minimal_train import MinimalTrainer
    trainer = MinimalTrainer(config)
    trainer.model = model
    
    if hasattr(outputs, 'mask_logits') and outputs.mask_logits is not None:
        try:
            total_loss, lm_loss, seg_loss = trainer.compute_loss(
                outputs,
                batch['labels'],
                batch['mask_labels']
            )
            logger.info(f"Loss computation successful!")
            logger.info(f"  Total loss: {total_loss:.4f}")
            logger.info(f"  LM loss: {lm_loss:.4f}")
            logger.info(f"  Seg loss: {seg_loss:.4f}")
        except Exception as e:
            logger.error(f"Loss computation failed: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("\n=== Debug complete ===")

if __name__ == "__main__":
    main()