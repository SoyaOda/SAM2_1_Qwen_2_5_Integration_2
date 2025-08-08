#!/usr/bin/env python3
"""
SEGトークンのデバッグスクリプト
SEG LOSSが下がらない問題の原因を特定する
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import sys
import logging
from typing import Dict, Any, List, Optional, Tuple

# プロジェクトのパスを追加
sys.path.append(str(Path(__file__).parent))

from src.config import LISAConfig
from src.models.lisa_model import LISA_Model
from src.data.dataset import HybridDataset
from src.data.collators import MultiModalDataCollator
from src.utils.tokenizer_utils import prepare_tokenizer_for_lisa
from transformers import AutoProcessor

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def debug_seg_token():
    """SEGトークンの動作を詳細にデバッグ"""
    
    logger.info("=" * 80)
    logger.info("SEGトークンデバッグ開始")
    logger.info("=" * 80)
    
    # 1. 設定とモデルの初期化
    logger.info("\n1. 設定とモデルの初期化")
    config = LISAConfig(
        qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        sam_model_name="./checkpoints/sam2.1_hiera_large.pt",
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        torch_dtype="auto",
        freeze_qwen=True,
        freeze_sam=True,
        train_seg_token=True,
        use_flash_attention=False
    )
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 2. トークナイザーの準備とSEGトークンの確認
    logger.info("\n2. トークナイザーの準備")
    tokenizer = prepare_tokenizer_for_lisa(
        model_name=config.qwen_model_name,
        seg_token=config.seg_token
    )
    
    # SEGトークンの確認
    seg_token_id = tokenizer.convert_tokens_to_ids(config.seg_token)
    logger.info(f"SEGトークン: '{config.seg_token}'")
    logger.info(f"SEGトークンID: {seg_token_id}")
    logger.info(f"トークナイザーの語彙サイズ: {len(tokenizer)}")
    
    # SEGトークンが正しく追加されているか確認
    if config.seg_token in tokenizer.get_vocab():
        logger.info(f"✓ SEGトークンがトークナイザーに存在します")
    else:
        logger.error(f"✗ SEGトークンがトークナイザーに存在しません！")
    
    # 3. プロセッサの準備
    logger.info("\n3. プロセッサの準備")
    processor = AutoProcessor.from_pretrained(config.qwen_model_name)
    
    # 4. モデルの初期化
    logger.info("\n4. モデルの初期化")
    model = LISA_Model(config)
    model.set_tokenizer(tokenizer)
    model = model.to(device)
    
    # モデルのSEGトークンIDを確認
    logger.info(f"モデル内のSEGトークンID: {model.seg_token_id}")
    
    # SEGトークンの埋め込みが学習可能か確認
    if model.seg_token_id is not None:
        word_embeddings = model.qwen.get_input_embeddings()
        seg_embedding = word_embeddings.weight[model.seg_token_id]
        logger.info(f"SEGトークン埋め込みのrequires_grad: {seg_embedding.requires_grad}")
        logger.info(f"SEGトークン埋め込みのshape: {seg_embedding.shape}")
        logger.info(f"SEGトークン埋め込みのnorm: {seg_embedding.norm().item():.4f}")
    
    # 5. データセットの準備
    logger.info("\n5. データセットの準備")
    dataset = HybridDataset(
        base_image_dir=config.dataset_base_dir,
        qwen_processor=processor,
        samples_per_epoch=10,  # デバッグ用に少なく
        dataset='refcoco||refcoco+||refcocog',
        sample_rate=[0.33, 0.33, 0.34],
        qwen_image_size=config.qwen_image_size,
        sam_image_size=config.sam_image_size,
    )
    
    collator = MultiModalDataCollator(
        tokenizer=tokenizer,
        max_length=config.model_max_length,
        config=config
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collator,
        num_workers=0
    )
    
    # 6. バッチデータの検証
    logger.info("\n6. バッチデータの検証")
    for i, batch in enumerate(dataloader):
        if i >= 3:  # 最初の3バッチのみ確認
            break
        
        logger.info(f"\n--- バッチ {i+1} ---")
        
        # バッチの内容を確認
        input_ids = batch['input_ids']
        labels = batch['labels']
        pixel_values = batch.get('pixel_values', None)
        mask_labels = batch.get('mask_labels', None)
        
        logger.info(f"input_ids shape: {input_ids.shape}")
        logger.info(f"labels shape: {labels.shape}")
        
        # SEGトークンの位置を確認
        seg_positions_input = (input_ids == seg_token_id).nonzero(as_tuple=True)
        seg_positions_labels = (labels == seg_token_id).nonzero(as_tuple=True)
        
        logger.info(f"input_idsのSEGトークン位置: {seg_positions_input}")
        logger.info(f"labelsのSEGトークン位置: {seg_positions_labels}")
        
        if len(seg_positions_labels[1]) > 0:
            logger.info(f"✓ labelsにSEGトークンが含まれています (位置: {seg_positions_labels[1].tolist()})")
        else:
            logger.error(f"✗ labelsにSEGトークンが含まれていません！")
            # デバッグ: labelsの内容を一部表示
            valid_labels = labels[labels != -100][:20]  # 最初の20個の有効なラベル
            logger.info(f"有効なラベル（最初の20個）: {valid_labels.tolist()}")
            tokens = tokenizer.convert_ids_to_tokens(valid_labels.tolist())
            logger.info(f"対応するトークン: {tokens}")
        
        # 7. モデルのforward passを実行
        logger.info("\n7. モデルのforward pass")
        model.eval()
        with torch.no_grad():
            batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                          for k, v in batch.items()}
            
            outputs = model(
                input_ids=batch_device['input_ids'],
                pixel_values=batch_device.get('pixel_values'),
                attention_mask=batch_device.get('attention_mask'),
                labels=batch_device['labels'],
                mask_labels=batch_device.get('mask_labels'),
                image_grid_thw=batch_device.get('image_grid_thw')
            )
            
            logger.info(f"outputs.logits shape: {outputs.logits.shape}")
            logger.info(f"outputs.mask_logits: {outputs.mask_logits}")
            logger.info(f"outputs.seg_token_positions: {outputs.seg_token_positions}")
            
            # マスクが生成されているか確認
            if outputs.mask_logits:
                for j, masks in enumerate(outputs.mask_logits):
                    if masks is not None and len(masks) > 0:
                        logger.info(f"  バッチ{j}: {len(masks)}個のマスクが生成されました")
                        for k, mask in enumerate(masks):
                            logger.info(f"    マスク{k} shape: {mask.shape}")
                    else:
                        logger.warning(f"  バッチ{j}: マスクが生成されませんでした")
            else:
                logger.error("✗ mask_logitsが空です！")
            
            # 8. 損失計算をテスト
            logger.info("\n8. 損失計算のテスト")
            if mask_labels is not None and outputs.mask_logits:
                seg_loss = 0.0
                seg_count = 0
                
                for batch_idx, batch_masks in enumerate(outputs.mask_logits):
                    if batch_masks is not None and len(batch_masks) > 0:
                        gt_mask = mask_labels[batch_idx]
                        
                        for pred_mask in batch_masks:
                            # BCE損失
                            bce_loss = nn.functional.binary_cross_entropy_with_logits(
                                pred_mask.squeeze(0),
                                gt_mask.squeeze(0).to(device).float()
                            )
                            
                            # Dice損失
                            pred_sigmoid = torch.sigmoid(pred_mask.squeeze(0))
                            intersection = (pred_sigmoid * gt_mask.squeeze(0).to(device)).sum()
                            dice = 2 * intersection / (pred_sigmoid.sum() + gt_mask.squeeze(0).sum() + 1e-8)
                            dice_loss = 1 - dice
                            
                            seg_loss += bce_loss + dice_loss
                            seg_count += 1
                            
                            logger.info(f"  BCE Loss: {bce_loss.item():.4f}")
                            logger.info(f"  Dice Loss: {dice_loss.item():.4f}")
                
                if seg_count > 0:
                    avg_seg_loss = seg_loss / seg_count
                    logger.info(f"平均SEG Loss: {avg_seg_loss.item():.4f}")
                else:
                    logger.error("✗ seg_count = 0: SEG損失が計算されませんでした")
            else:
                logger.error("✗ mask_labelsまたはmask_logitsが利用できません")
    
    logger.info("\n" + "=" * 80)
    logger.info("デバッグ完了")
    logger.info("=" * 80)

if __name__ == "__main__":
    debug_seg_token()