#!/usr/bin/env python3
"""
Test A2 デバッグ版: SAM ImageEncoderの使用状況を詳細にログ出力
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import logging
from tqdm import tqdm
from typing import Optional, Tuple
import math

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 共通設定をインポート
from test_common_config import (
    SEED, NUM_STEPS, BATCH_SIZE, LEARNING_RATE, NUM_FIXED_SAMPLES,
    DATASET_TYPE, SAMPLE_RATE, VIS_INTERVAL, LOG_INTERVAL, EVAL_THRESHOLD,
    set_random_seed, FixedSampleDataset, save_test_config, create_standard_output_structure
)

from src.models.lisa_model import LISA_Model
from src.config import LISAConfig
from src.data.dataset import HybridDataset
from src.data.collators import MultiModalDataCollator
from src.utils.tokenizer_utils import prepare_tokenizer_for_lisa
from transformers import AutoProcessor

# 詳細ログ設定
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 特定のロガーを強制的にDEBUGレベルに
lisa_logger = logging.getLogger('src.models.lisa_model')
lisa_logger.setLevel(logging.DEBUG)

# グローバル変数
num_steps = 3  # デバッグのため少なく設定
batch_size = BATCH_SIZE
learning_rate = LEARNING_RATE
dataset_type = DATASET_TYPE
samples_per_epoch = NUM_FIXED_SAMPLES

def main():
    print("🔍 === Test A2 DEBUG VERSION ===")
    print("🔍 SAM ImageEncoderの使用状況を詳細にトレース")
    
    # シード固定
    set_random_seed(SEED)
    
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔍 Using device: {device}")
    
    # 出力ディレクトリ作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = create_standard_output_structure(f"test_outputs/test_a2_debug_{timestamp}")
    print(f"🔍 Output directory: {output_dir}")
    
    # 1. モデルとトークナイザの準備
    print("🔍 === STEP 1: Model Initialization ===")
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
    print("🔍 Initializing LISA_Model...")
    model = LISA_Model(config)
    model.set_tokenizer(tokenizer)
    
    # **SAM ImageEncoderの存在確認**
    has_sam_encoder = hasattr(model, 'sam_image_encoder') and model.sam_image_encoder is not None
    print(f"🔍 **SAM ImageEncoder exists: {has_sam_encoder}**")
    
    if has_sam_encoder:
        sam_encoder_params = sum(p.numel() for p in model.sam_image_encoder.parameters())
        sam_encoder_dtype = next(model.sam_image_encoder.parameters()).dtype
        print(f"🔍 SAM ImageEncoder parameters: {sam_encoder_params:,}")
        print(f"🔍 SAM ImageEncoder dtype: {sam_encoder_dtype}")
        
        # SAM encoder configを確認
        if hasattr(model, 'sam_encoder_config'):
            print(f"🔍 SAM encoder config: {model.sam_encoder_config}")
        else:
            print("🔍 ⚠️ SAM encoder config not found!")
    else:
        print("🔍 ❌ **SAM ImageEncoder NOT FOUND!**")
        return
    
    model = model.to(device)
    model.train()
    
    # 2. データセット準備（最小限）
    print("🔍 === STEP 2: Dataset Preparation ===")
    base_dir = config.dataset_base_dir
    
    # 最小限のデータセット
    base_dataset = HybridDataset(
        base_image_dir=base_dir,
        qwen_processor=processor,
        samples_per_epoch=2,  # デバッグ用に最小限
        dataset=DATASET_TYPE,
        sample_rate=SAMPLE_RATE,
        qwen_image_size=config.qwen_image_size,
        sam_image_size=config.sam_image_size,
    )
    
    dataset = FixedSampleDataset(base_dataset, num_samples=2, seed=SEED)
    collator = MultiModalDataCollator(
        tokenizer=tokenizer,
        max_length=config.model_max_length,
        config=config
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=1,  # デバッグ用にバッチサイズ1
        shuffle=False,
        collate_fn=collator, 
        num_workers=0,
        pin_memory=True
    )
    
    print(f"🔍 Dataset loaded: {len(dataset)} samples")
    
    # 3. オプティマイザ準備
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # 4. デバッグ学習ループ
    print("🔍 === STEP 3: Debug Training Loop ===")
    step = 0
    data_iter = iter(dataloader)
    
    for step in range(1):  # 1ステップのみ
        print(f"🔍 === DEBUG STEP {step} ===")
        
        # データをロード
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        # バッチからデバイスへ転送
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        
        print(f"🔍 Batch keys: {list(batch.keys())}")
        print(f"🔍 pixel_values shape: {batch['pixel_values'].shape if 'pixel_values' in batch else 'None'}")
        
        # **SAM画像を元画像から生成**
        print("🔍 --- SAM Image Generation ---")
        if 'pixel_values' in batch and batch['pixel_values'] is not None:
            sam_images = []
            for i in range(batch['pixel_values'].shape[0]):
                # Qwen用の画像を取得
                qwen_image = batch['pixel_values'][i]  # [C, H, W]
                print(f"🔍 Qwen image {i} shape: {qwen_image.shape}, dtype: {qwen_image.dtype}")
                
                # RGB形式に変換してPIL Imageに
                if qwen_image.dim() == 3:
                    # [C, H, W] -> [H, W, C]の順番で変換
                    qwen_np = qwen_image.permute(1, 2, 0).cpu().numpy()
                    # 正規化されている場合は0-255に戻す
                    if qwen_np.max() <= 1.0:
                        qwen_np = (qwen_np * 255).astype(np.uint8)
                    
                    # PIL Imageに変換
                    from PIL import Image
                    pil_image = Image.fromarray(qwen_np.astype(np.uint8))
                    
                    # SAM用にリサイズ
                    sam_image = pil_image.resize((config.sam_image_size, config.sam_image_size), Image.BICUBIC)
                    
                    # Tensorに変換
                    sam_array = np.array(sam_image).astype(np.float32) / 255.0
                    sam_tensor = torch.from_numpy(sam_array).permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
                    sam_images.append(sam_tensor)
                    
                    print(f"🔍 Generated SAM image {i} shape: {sam_tensor.shape}, dtype: {sam_tensor.dtype}")
            
            if sam_images:
                sam_images_batch = torch.stack(sam_images).to(device)
                print(f"🔍 **SAM images batch final shape: {sam_images_batch.shape}**")
                print(f"🔍 **SAM images batch dtype: {sam_images_batch.dtype}**")
                print(f"🔍 **SAM images batch device: {sam_images_batch.device}**")
            else:
                sam_images_batch = None
                print("🔍 ⚠️ No SAM images generated!")
        else:
            sam_images_batch = None
            print("🔍 ⚠️ No pixel_values in batch!")
        
        # **forwardメソッド実行**
        print("🔍 --- Model Forward Pass ---")
        print(f"🔍 About to call model.forward with sam_images: {sam_images_batch is not None}")
        
        # **重要: この時点でSAM関連のログが出力されるはず**
        outputs = model(
            input_ids=batch['input_ids'],
            pixel_values=batch['pixel_values'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
            mask_labels=batch['mask_labels'],
            image_grid_thw=batch.get('image_grid_thw', None),
            sam_images=sam_images_batch,  # **ここでSAM画像を渡す**
            return_dict=True
        )
        
        print("🔍 --- Forward Pass Results ---")
        print(f"🔍 Output keys: {list(outputs.__dict__.keys()) if hasattr(outputs, '__dict__') else 'No dict'}")
        print(f"🔍 mask_logits type: {type(outputs.mask_logits)}")
        print(f"🔍 mask_logits length: {len(outputs.mask_logits) if outputs.mask_logits else 'None'}")
        
        if outputs.mask_logits and outputs.mask_logits[0] is not None:
            first_mask = outputs.mask_logits[0]
            if isinstance(first_mask, list) and len(first_mask) > 0:
                mask_tensor = first_mask[0]
                print(f"🔍 First mask shape: {mask_tensor.shape}")
                print(f"🔍 First mask stats: min={mask_tensor.min().item():.6f}, max={mask_tensor.max().item():.6f}, mean={mask_tensor.mean().item():.6f}")
            else:
                print("🔍 ⚠️ First mask is not a list or empty!")
        else:
            print("🔍 ⚠️ No mask logits generated!")
        
        # 簡単な損失計算
        vocab_size = outputs.logits.size(-1)
        lm_loss = F.cross_entropy(outputs.logits.view(-1, vocab_size), batch['labels'].view(-1), ignore_index=-100)
        print(f"🔍 LM Loss: {lm_loss.item():.6f}")
        
        print("🔍 === DEBUG COMPLETE (1 step only) ===")
        break

if __name__ == "__main__":
    main()