#!/usr/bin/env python3
"""
マスク座標系のデバッグスクリプト
予測マスクとGTマスクの座標系・サイズ・パディングを詳細に調査
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Dict, Any, Tuple

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# プロジェクトのインポート
import sys
sys.path.append('/home/soya/SAM2_1_Qwen_2_5_Integration_2')

from src.data.dataset import HybridDataset
from src.data.collate_fn import lisa_collate_fn
from src.models.lisa_model import LISA_Model
from src.config import LISAConfig
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import Qwen2VLProcessor

def analyze_tensor(tensor: torch.Tensor, name: str) -> Dict[str, Any]:
    """テンソルの詳細情報を解析"""
    info = {
        'name': name,
        'shape': tuple(tensor.shape),
        'dtype': str(tensor.dtype),
        'device': str(tensor.device),
        'min': float(tensor.min()),
        'max': float(tensor.max()),
        'mean': float(tensor.mean()),
        'std': float(tensor.std()),
        'has_nan': bool(torch.isnan(tensor).any()),
        'has_inf': bool(torch.isinf(tensor).any())
    }
    
    # パディング検出（ゼロ領域の検出）
    if tensor.dim() >= 2:
        h, w = tensor.shape[-2:]
        # 下端と右端のゼロ領域を検出
        bottom_zeros = 0
        for i in range(h-1, -1, -1):
            if torch.abs(tensor[..., i, :]).max() < 1e-6:
                bottom_zeros += 1
            else:
                break
        
        right_zeros = 0
        for j in range(w-1, -1, -1):
            if torch.abs(tensor[..., :, j]).max() < 1e-6:
                right_zeros += 1
            else:
                break
        
        info['bottom_padding'] = bottom_zeros
        info['right_padding'] = right_zeros
        info['effective_size'] = (h - bottom_zeros, w - right_zeros)
    
    return info

def visualize_mask_debug(
    image: np.ndarray,
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    title: str,
    save_path: Path
) -> None:
    """マスクのデバッグ可視化"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # 予測マスクとGTマスクをnumpyに変換
    if pred_mask.dim() > 2:
        pred_mask = pred_mask.squeeze()
    if gt_mask.dim() > 2:
        gt_mask = gt_mask.squeeze()
    
    pred_np = pred_mask.detach().cpu().numpy()
    gt_np = gt_mask.detach().cpu().numpy()
    
    # Row 1: 生データ
    axes[0, 0].imshow(image)
    axes[0, 0].set_title(f'Original Image\n{image.shape}')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(pred_np, cmap='hot')
    axes[0, 1].set_title(f'Pred Mask (Raw)\n{pred_np.shape}')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(gt_np, cmap='hot')
    axes[0, 2].set_title(f'GT Mask (Raw)\n{gt_np.shape}')
    axes[0, 2].axis('off')
    
    # 差分マップ
    diff = np.abs(pred_np - gt_np)
    axes[0, 3].imshow(diff, cmap='hot')
    axes[0, 3].set_title(f'Absolute Difference\nMax: {diff.max():.3f}')
    axes[0, 3].axis('off')
    
    # Row 2: パディング領域の可視化
    # 予測マスクのパディング検出
    h, w = pred_np.shape
    pred_nonzero = np.abs(pred_np) > 1e-6
    pred_rows = np.any(pred_nonzero, axis=1)
    pred_cols = np.any(pred_nonzero, axis=0)
    
    pred_viz = np.zeros((h, w, 3))
    pred_viz[pred_nonzero] = [1, 1, 1]  # 白：有効領域
    # パディング領域を赤でマーク
    for i in range(h):
        if not pred_rows[i]:
            pred_viz[i, :] = [1, 0, 0]
    for j in range(w):
        if not pred_cols[j]:
            pred_viz[:, j] = [1, 0, 0]
    
    axes[1, 0].imshow(pred_viz)
    axes[1, 0].set_title('Pred Padding (Red=Padding)')
    axes[1, 0].axis('off')
    
    # GTマスクのパディング検出
    gt_nonzero = np.abs(gt_np) > 1e-6
    gt_rows = np.any(gt_nonzero, axis=1)
    gt_cols = np.any(gt_nonzero, axis=0)
    
    gt_viz = np.zeros((h, w, 3))
    gt_viz[gt_nonzero] = [1, 1, 1]  # 白：有効領域
    for i in range(h):
        if not gt_rows[i]:
            gt_viz[i, :] = [0, 1, 0]  # 緑：パディング
    for j in range(w):
        if not gt_cols[j]:
            gt_viz[:, j] = [0, 1, 0]
    
    axes[1, 1].imshow(gt_viz)
    axes[1, 1].set_title('GT Padding (Green=Padding)')
    axes[1, 1].axis('off')
    
    # 有効領域のバウンディングボックス
    pred_valid_rows = np.where(pred_rows)[0]
    pred_valid_cols = np.where(pred_cols)[0]
    gt_valid_rows = np.where(gt_rows)[0]
    gt_valid_cols = np.where(gt_cols)[0]
    
    bbox_viz = np.zeros((h, w, 3))
    if len(pred_valid_rows) > 0 and len(pred_valid_cols) > 0:
        p_top, p_bottom = pred_valid_rows[0], pred_valid_rows[-1]
        p_left, p_right = pred_valid_cols[0], pred_valid_cols[-1]
        # 予測の有効領域を赤で描画
        bbox_viz[p_top:p_bottom+1, p_left] = [1, 0, 0]
        bbox_viz[p_top:p_bottom+1, p_right] = [1, 0, 0]
        bbox_viz[p_top, p_left:p_right+1] = [1, 0, 0]
        bbox_viz[p_bottom, p_left:p_right+1] = [1, 0, 0]
    
    if len(gt_valid_rows) > 0 and len(gt_valid_cols) > 0:
        g_top, g_bottom = gt_valid_rows[0], gt_valid_rows[-1]
        g_left, g_right = gt_valid_cols[0], gt_valid_cols[-1]
        # GTの有効領域を緑で描画
        bbox_viz[g_top:g_bottom+1, g_left] = [0, 1, 0]
        bbox_viz[g_top:g_bottom+1, g_right] = [0, 1, 0]
        bbox_viz[g_top, g_left:g_right+1] = [0, 1, 0]
        bbox_viz[g_bottom, g_left:g_right+1] = [0, 1, 0]
    
    axes[1, 2].imshow(bbox_viz)
    axes[1, 2].set_title('Bounding Boxes\n(Red=Pred, Green=GT)')
    axes[1, 2].axis('off')
    
    # 統計情報
    stats_text = f"Pred valid: ({p_top},{p_left}) to ({p_bottom},{p_right})\n"
    stats_text += f"GT valid: ({g_top},{g_left}) to ({g_bottom},{g_right})\n"
    stats_text += f"Pred size: {p_bottom-p_top+1}x{p_right-p_left+1}\n"
    stats_text += f"GT size: {g_bottom-g_top+1}x{g_right-g_left+1}"
    
    axes[1, 3].text(0.1, 0.5, stats_text, fontsize=10, family='monospace')
    axes[1, 3].set_title('Statistics')
    axes[1, 3].axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved debug visualization to {save_path}")

def main():
    """メインのデバッグ処理"""
    # 出力ディレクトリの準備
    debug_dir = Path('/home/soya/SAM2_1_Qwen_2_5_Integration_2/debug_outputs')
    debug_dir.mkdir(exist_ok=True)
    
    # モデルとデータセットの準備
    logger.info("Loading model and dataset...")
    
    config = LISAConfig()
    config.batch_size = 1
    config.num_workers = 0
    
    # プロセッサとモデルの初期化
    processor = Qwen2VLProcessor.from_pretrained(config.qwen_model_name)
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデルの初期化
    model = LISA_Model(config)
    model = model.to(device)
    model.eval()
    
    # データセットの準備
    dataset = HybridDataset(
        base_image_dir='/home/soya/SAM2_1_Qwen_2_5_Integration_2/data',
        qwen_processor=processor,
        samples_per_epoch=10,
        num_classes_per_sample=1,
        dataset='sem_seg',  # sem_segデータセットのみ使用
        use_cache=False  # デバッグ時はキャッシュ無効
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lisa_collate_fn,
        num_workers=0
    )
    
    # デバッグ情報を収集
    debug_info = []
    
    logger.info("Starting debug analysis...")
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 3:  # 3サンプルだけ解析
            break
        
        logger.info(f"\n=== Batch {batch_idx} ===")
        
        # バッチをデバイスに移動
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        
        # バッチの内容を確認
        batch_info = {
            'batch_idx': batch_idx,
            'keys': list(batch.keys())
        }
        
        # 各テンソルのサイズを記録
        for key, value in batch.items():
            if torch.is_tensor(value):
                info = analyze_tensor(value, key)
                batch_info[key] = info
                logger.info(f"  {key}: shape={info['shape']}, "
                          f"effective={info.get('effective_size', 'N/A')}, "
                          f"padding=(B:{info.get('bottom_padding', 0)}, R:{info.get('right_padding', 0)})")
        
        # モデル推論
        logger.info("  Running model inference...")
        with torch.no_grad():
            outputs = model(
                input_ids=batch['input_ids'],
                pixel_values=batch.get('pixel_values'),
                sam_images=batch.get('sam_images'),
                sam_bboxes=batch.get('sam_bboxes'),
                sam_points=batch.get('sam_points'),
                labels=batch.get('labels'),
                attention_mask=batch.get('attention_mask'),
                image_grid_thw=batch.get('image_grid_thw')
            )
        
        # 出力マスクの解析
        if hasattr(outputs, 'mask_logits') and outputs.mask_logits is not None:
            for i, masks in enumerate(outputs.mask_logits):
                if masks is not None and len(masks) > 0:
                    pred_mask = masks[0] if isinstance(masks, list) else masks
                    pred_info = analyze_tensor(pred_mask, f'pred_mask_{i}')
                    batch_info[f'pred_mask_{i}'] = pred_info
                    logger.info(f"  pred_mask_{i}: shape={pred_info['shape']}, "
                              f"effective={pred_info.get('effective_size', 'N/A')}, "
                              f"padding=(B:{pred_info.get('bottom_padding', 0)}, R:{pred_info.get('right_padding', 0)})")
                    
                    # GTマスクと比較
                    if 'mask_labels' in batch and batch['mask_labels'] is not None:
                        gt_mask = batch['mask_labels'][i]
                        gt_info = analyze_tensor(gt_mask, f'gt_mask_{i}')
                        batch_info[f'gt_mask_{i}'] = gt_info
                        logger.info(f"  gt_mask_{i}: shape={gt_info['shape']}, "
                                  f"effective={gt_info.get('effective_size', 'N/A')}, "
                                  f"padding=(B:{gt_info.get('bottom_padding', 0)}, R:{gt_info.get('right_padding', 0)})")
                        
                        # 可視化
                        if 'sam_images' in batch:
                            sam_image = batch['sam_images'][i]
                            # デノーマライズ
                            mean = torch.tensor([0.485, 0.456, 0.406], device=sam_image.device).view(3, 1, 1)
                            std = torch.tensor([0.229, 0.224, 0.225], device=sam_image.device).view(3, 1, 1)
                            sam_image = (sam_image * std + mean).clamp(0, 1)
                            sam_image = (sam_image * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                            
                            visualize_mask_debug(
                                sam_image,
                                pred_mask,
                                gt_mask,
                                f"Batch {batch_idx} - Mask Comparison",
                                debug_dir / f"debug_batch_{batch_idx}.png"
                            )
        
        debug_info.append(batch_info)
    
    # サマリーレポート作成
    logger.info("\n=== SUMMARY REPORT ===")
    for info in debug_info:
        batch_idx = info['batch_idx']
        logger.info(f"\nBatch {batch_idx}:")
        
        # サイズの不一致を検出
        if f'pred_mask_0' in info and f'gt_mask_0' in info:
            pred_shape = info[f'pred_mask_0']['shape']
            gt_shape = info[f'gt_mask_0']['shape']
            pred_eff = info[f'pred_mask_0'].get('effective_size', pred_shape)
            gt_eff = info[f'gt_mask_0'].get('effective_size', gt_shape)
            
            logger.info(f"  Shape mismatch: Pred={pred_shape} vs GT={gt_shape}")
            logger.info(f"  Effective size: Pred={pred_eff} vs GT={gt_eff}")
            
            if pred_shape != gt_shape:
                logger.warning(f"  ⚠️ SIZE MISMATCH DETECTED!")
            
            if pred_eff != gt_eff:
                logger.warning(f"  ⚠️ EFFECTIVE SIZE MISMATCH - Padding issue likely!")
    
    logger.info("\nDebug analysis complete!")

if __name__ == "__main__":
    main()