#!/usr/bin/env python3
"""
SEGトークンのデバッグスクリプト（簡易版）
データセットに依存せずにSEGトークンの動作を検証
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import logging
from PIL import Image

# プロジェクトのパスを追加
sys.path.append(str(Path(__file__).parent))

from src.config import LISAConfig
from src.models.lisa_model import LISA_Model
from src.utils.tokenizer_utils import prepare_tokenizer_for_lisa
from transformers import AutoProcessor

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_dummy_batch(tokenizer, processor, device, seg_token_id):
    """テスト用のダミーバッチを作成"""
    
    # ダミー画像を作成（448x448のランダム画像）
    dummy_image = Image.new('RGB', (448, 448), color=(128, 128, 128))
    
    # SEGトークンを含むテキストプロンプト
    text_prompt = "Segment the red object in this image. <SEG>"
    
    # messages形式でプロセッサに渡す
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": dummy_image},
                {"type": "text", "text": text_prompt}
            ]
        }
    ]
    
    # apply_chat_templateで処理
    processed = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )
    
    input_ids = processed['input_ids']
    attention_mask = processed['attention_mask']
    pixel_values = processed['pixel_values']
    
    # image_grid_thwを追加
    if 'image_grid_thw' in processed:
        image_grid_thw = processed['image_grid_thw']
    else:
        image_grid_thw = torch.tensor([[1, 32, 32]], dtype=torch.long)
    
    # ラベルを作成（input_idsのコピー）
    labels = input_ids.clone()
    
    # システムプロンプト部分を-100でマスク（学習対象外）
    # 最初の数トークンをマスク
    labels[:, :10] = -100
    
    # ダミーのマスクラベル（セグメンテーション用）
    mask_labels = [torch.ones(1, 448, 448, dtype=torch.float32)]
    
    return {
        'input_ids': input_ids.to(device),
        'labels': labels.to(device),
        'attention_mask': attention_mask.to(device),
        'pixel_values': pixel_values.to(device) if pixel_values is not None else None,
        'mask_labels': mask_labels,
        'image_grid_thw': image_grid_thw.to(device)
    }

def debug_seg_token():
    """SEGトークンの動作を詳細にデバッグ"""
    
    logger.info("=" * 80)
    logger.info("SEGトークンデバッグ開始（簡易版）")
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
        use_flash_attention=False,
        segmentation_loss_weight=1.0
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
        return
    
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
        
        # train_seg_token=Trueの場合、埋め込みを学習可能にする
        if config.train_seg_token and not seg_embedding.requires_grad:
            logger.warning("⚠️ train_seg_token=Trueですが、埋め込みがrequires_grad=Falseです")
            logger.info("埋め込みを学習可能に設定します...")
            seg_embedding.requires_grad = True
            logger.info(f"設定後のrequires_grad: {seg_embedding.requires_grad}")
    
    # 5. テストバッチの作成
    logger.info("\n5. テストバッチの作成")
    
    for test_idx in range(3):  # 3つのテストケース
        logger.info(f"\n--- テストケース {test_idx + 1} ---")
        
        batch = create_dummy_batch(tokenizer, processor, device, seg_token_id)
        
        input_ids = batch['input_ids']
        labels = batch['labels']
        
        logger.info(f"input_ids shape: {input_ids.shape}")
        logger.info(f"labels shape: {labels.shape}")
        
        # input_idsの内容を確認（最初と最後の10トークン）
        logger.info(f"input_ids（最初の10トークン）: {input_ids[0, :10].tolist()}")
        logger.info(f"input_ids（最後の10トークン）: {input_ids[0, -10:].tolist()}")
        
        # SEGトークンの位置を確認
        seg_positions_input = (input_ids == seg_token_id).nonzero(as_tuple=True)
        seg_positions_labels = (labels == seg_token_id).nonzero(as_tuple=True)
        
        logger.info(f"input_idsのSEGトークン位置: {seg_positions_input[1].tolist() if len(seg_positions_input[1]) > 0 else 'なし'}")
        logger.info(f"labelsのSEGトークン位置: {seg_positions_labels[1].tolist() if len(seg_positions_labels[1]) > 0 else 'なし'}")
        
        if len(seg_positions_labels[1]) > 0:
            logger.info(f"✓ labelsにSEGトークンが含まれています")
            # SEGトークン周辺のトークンを表示
            seg_pos = seg_positions_labels[1][0].item()
            start = max(0, seg_pos - 3)
            end = min(labels.shape[1], seg_pos + 4)
            surrounding_tokens = labels[0, start:end].tolist()
            surrounding_text = tokenizer.convert_ids_to_tokens(surrounding_tokens)
            logger.info(f"SEGトークン周辺のトークン: {surrounding_text}")
        else:
            logger.error(f"✗ labelsにSEGトークンが含まれていません！")
            # デバッグ: labelsの内容を一部表示
            valid_labels = labels[labels != -100][:20]  # 最初の20個の有効なラベル
            if len(valid_labels) > 0:
                logger.info(f"有効なラベル（最初の20個）: {valid_labels.tolist()}")
                tokens = tokenizer.convert_ids_to_tokens(valid_labels.tolist())
                logger.info(f"対応するトークン: {tokens}")
        
        # 6. モデルのforward passを実行
        logger.info("\n6. モデルのforward pass")
        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=batch['input_ids'],
                pixel_values=batch['pixel_values'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
                mask_labels=batch['mask_labels'],
                image_grid_thw=batch['image_grid_thw']
            )
            
            logger.info(f"outputs.logits shape: {outputs.logits.shape}")
            logger.info(f"outputs.seg_token_positions: {outputs.seg_token_positions}")
            
            # マスクが生成されているか確認
            if outputs.mask_logits:
                for j, masks in enumerate(outputs.mask_logits):
                    if masks is not None and len(masks) > 0:
                        logger.info(f"  バッチ{j}: {len(masks)}個のマスクが生成されました")
                        for k, mask in enumerate(masks):
                            logger.info(f"    マスク{k} shape: {mask.shape}, "
                                      f"min: {mask.min().item():.4f}, max: {mask.max().item():.4f}")
                    else:
                        logger.warning(f"  バッチ{j}: マスクが生成されませんでした")
            else:
                logger.error("✗ mask_logitsが空です！")
            
            # 7. 損失計算をテスト
            logger.info("\n7. 損失計算のテスト")
            mask_labels = batch['mask_labels']
            if mask_labels is not None and outputs.mask_logits:
                seg_loss_total = 0.0
                seg_count = 0
                
                for batch_idx, batch_masks in enumerate(outputs.mask_logits):
                    if batch_masks is not None and len(batch_masks) > 0:
                        gt_mask = mask_labels[batch_idx].to(device)
                        
                        for pred_mask in batch_masks:
                            # サイズ調整
                            if pred_mask.shape[-2:] != gt_mask.shape[-2:]:
                                pred_mask = nn.functional.interpolate(
                                    pred_mask.unsqueeze(0),
                                    size=gt_mask.shape[-2:],
                                    mode='bilinear',
                                    align_corners=False
                                ).squeeze(0)
                            
                            # BCE損失
                            bce_loss = nn.functional.binary_cross_entropy_with_logits(
                                pred_mask.squeeze(0),
                                gt_mask.squeeze(0).float()
                            )
                            
                            # Dice損失
                            pred_sigmoid = torch.sigmoid(pred_mask.squeeze(0))
                            intersection = (pred_sigmoid * gt_mask.squeeze(0)).sum()
                            dice = 2 * intersection / (pred_sigmoid.sum() + gt_mask.squeeze(0).sum() + 1e-8)
                            dice_loss = 1 - dice
                            
                            seg_loss_total += bce_loss + dice_loss
                            seg_count += 1
                            
                            logger.info(f"  BCE Loss: {bce_loss.item():.4f}")
                            logger.info(f"  Dice Loss: {dice_loss.item():.4f}")
                            logger.info(f"  Combined Loss: {(bce_loss + dice_loss).item():.4f}")
                
                if seg_count > 0:
                    avg_seg_loss = seg_loss_total / seg_count
                    logger.info(f"平均SEG Loss: {avg_seg_loss.item():.4f}")
                else:
                    logger.error("✗ seg_count = 0: SEG損失が計算されませんでした")
            else:
                if mask_labels is None:
                    logger.error("✗ mask_labelsがNoneです")
                if not outputs.mask_logits:
                    logger.error("✗ outputs.mask_logitsが空です")
    
    logger.info("\n" + "=" * 80)
    logger.info("問題の診断")
    logger.info("=" * 80)
    
    if model.seg_token_id is None:
        logger.error("❌ モデルにSEGトークンIDが設定されていません")
    else:
        logger.info("✅ モデルにSEGトークンIDが正しく設定されています")
    
    word_embeddings = model.qwen.get_input_embeddings()
    seg_embedding = word_embeddings.weight[model.seg_token_id]
    if not seg_embedding.requires_grad and config.train_seg_token:
        logger.error("❌ SEGトークンの埋め込みが学習可能になっていません（train_seg_token=True）")
    else:
        logger.info("✅ SEGトークンの埋め込み設定は正しいです")
    
    logger.info("\n" + "=" * 80)
    logger.info("デバッグ完了")
    logger.info("=" * 80)

if __name__ == "__main__":
    debug_seg_token()