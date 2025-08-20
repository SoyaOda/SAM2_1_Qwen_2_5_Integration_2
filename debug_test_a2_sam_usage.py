#!/usr/bin/env python3
"""
Test A2デバッグスクリプト: SAM ImageEncoderが実際に呼ばれているかを確認
"""

import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import logging
from PIL import Image

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.lisa_model import LISA_Model
from src.config import LISAConfig
from src.data.dataset import HybridDataset
from src.data.collators import MultiModalDataCollator
from src.utils.tokenizer_utils import prepare_tokenizer_for_lisa
from transformers import AutoProcessor

# デバッグ用ログ設定
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_sam_usage():
    """SAM ImageEncoderの使用状況をデバッグ"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Test A2と同じ設定でモデルを作成
    config = LISAConfig(
        qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        sam_model_name="./checkpoints/sam2.1_hiera_large.pt",
        device_map="cuda",
        torch_dtype="auto",
        use_flash_attention=False,
        # Test A2と同じFreeze設定
        freeze_qwen_lora=False,
        freeze_seg_token=False,
        sam_lora_r=8,
        freeze_image_adapter=False,
        freeze_text_prompt_projector=False,
        freeze_token_fpn=False,
        freeze_prompt_beta=True,
        freeze_sam_mask_decoder_base=True,
    )
    
    # プロセッサーの準備
    processor = AutoProcessor.from_pretrained(config.qwen_model_name)
    if config.seg_token not in processor.tokenizer.get_vocab():
        processor.tokenizer.add_special_tokens({"additional_special_tokens": [config.seg_token]})
    
    # トークナイザー準備
    tokenizer = prepare_tokenizer_for_lisa(
        model_name=config.qwen_model_name,
        seg_token=config.seg_token
    )
    
    # モデル初期化
    model = LISA_Model(config)
    model.set_tokenizer(tokenizer)
    model = model.to(device)
    model.train()
    
    # SAM ImageEncoderが存在するか確認
    has_sam_encoder = hasattr(model, 'sam_image_encoder') and model.sam_image_encoder is not None
    logger.info(f"**SAM ImageEncoder exists: {has_sam_encoder}**")
    
    if has_sam_encoder:
        # SAM ImageEncoderの詳細情報
        sam_encoder_params = sum(p.numel() for p in model.sam_image_encoder.parameters())
        logger.info(f"SAM ImageEncoder parameters: {sam_encoder_params:,}")
        logger.info(f"SAM ImageEncoder dtype: {next(model.sam_image_encoder.parameters()).dtype}")
        
        # SAM encoder configを確認
        if hasattr(model, 'sam_encoder_config'):
            logger.info(f"SAM encoder config: {model.sam_encoder_config}")
        else:
            logger.warning("SAM encoder config not found!")
    
    # ダミーデータで実際のテスト
    logger.info("=== Creating dummy test data ===")
    
    # ダミー画像データ作成（Test A2と同じ方法）
    batch_size = 1
    dummy_image = torch.randn(batch_size, 3, 224, 224).to(device)  # Qwen用画像
    
    # SAM画像を生成（Test A2と同じロジック）
    sam_images = []
    for i in range(dummy_image.shape[0]):
        qwen_image = dummy_image[i]  # [C, H, W]
        
        # RGB形式に変換してPIL Imageに
        if qwen_image.dim() == 3:
            qwen_np = qwen_image.permute(1, 2, 0).cpu().numpy()
            if qwen_np.max() <= 1.0:
                qwen_np = (qwen_np * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(qwen_np.astype(np.uint8))
            sam_image = pil_image.resize((config.sam_image_size, config.sam_image_size), Image.BICUBIC)
            
            sam_array = np.array(sam_image).astype(np.float32) / 255.0
            sam_tensor = torch.from_numpy(sam_array).permute(2, 0, 1)
            sam_images.append(sam_tensor)
    
    sam_images_batch = torch.stack(sam_images).to(device) if sam_images else None
    logger.info(f"**SAM images batch shape: {sam_images_batch.shape if sam_images_batch is not None else 'None'}**")
    
    # ダミーテキスト入力
    text = "Where is the object? Please segment it <SEG>."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # SEGトークンの位置確認
    seg_token_id = tokenizer.convert_tokens_to_ids(config.seg_token)
    seg_positions = (input_ids == seg_token_id).nonzero(as_tuple=True)
    logger.info(f"**SEG token ID: {seg_token_id}, positions: {seg_positions}**")
    
    # ダミーマスクラベル
    mask_labels = [torch.ones(224, 224).to(device)]  # ダミーGTマスク
    
    logger.info("=== Running forward pass ===")
    
    # **実際のforwardテスト - SAM画像ありとなしを比較**
    logger.info("--- Test 1: WITHOUT SAM images ---")
    with torch.no_grad():  # メモリ節約のため
        outputs_no_sam = model(
            input_ids=input_ids,
            pixel_values=dummy_image,
            attention_mask=attention_mask,
            mask_labels=mask_labels,
            sam_images=None,  # SAM画像なし
            return_dict=True
        )
    logger.info(f"Without SAM - mask_logits: {len(outputs_no_sam.mask_logits) if outputs_no_sam.mask_logits else 'None'}")
    
    logger.info("--- Test 2: WITH SAM images ---")
    with torch.no_grad():  # メモリ節約のため
        outputs_with_sam = model(
            input_ids=input_ids,
            pixel_values=dummy_image,
            attention_mask=attention_mask,
            mask_labels=mask_labels,
            sam_images=sam_images_batch,  # SAM画像あり
            return_dict=True
        )
    logger.info(f"With SAM - mask_logits: {len(outputs_with_sam.mask_logits) if outputs_with_sam.mask_logits else 'None'}")
    
    # 結果の比較
    if outputs_no_sam.mask_logits and outputs_with_sam.mask_logits:
        if outputs_no_sam.mask_logits[0] is not None and outputs_with_sam.mask_logits[0] is not None:
            mask_diff = torch.abs(outputs_no_sam.mask_logits[0][0] - outputs_with_sam.mask_logits[0][0]).mean()
            logger.info(f"**Mask difference (with/without SAM): {mask_diff.item():.6f}**")
            
            if mask_diff.item() < 1e-6:
                logger.error("⚠️ **PROBLEM: Masks are identical! SAM ImageEncoder may not be used!**")
            else:
                logger.info("✅ **SUCCESS: Masks are different! SAM ImageEncoder is being used!**")
        else:
            logger.warning("⚠️ Mask logits are None")
    else:
        logger.warning("⚠️ No mask logits generated")

if __name__ == "__main__":
    debug_sam_usage()