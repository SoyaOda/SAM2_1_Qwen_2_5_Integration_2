#!/usr/bin/env python3
"""
SAM ImageEncoderの実際の状態を確認するデバッグスクリプト
"""

import torch
import sys
import os
sys.path.append('/home/soya/SAM2_1_Qwen_2_5_Integration_2')

from src.config import LISAConfig
from src.models.lisa_model import LISA_Model
from transformers import AutoTokenizer, AutoProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_sam_encoder_state():
    print("=== SAM ImageEncoder状態確認 ===")
    
    # Config設定
    config = LISAConfig()
    print(f"Config設定:")
    print(f"  sam_encoder_eval_mode: {config.sam_encoder_eval_mode}")
    print(f"  sam_encoder_precision: {config.sam_encoder_precision}")
    print(f"  freeze_sam_image_encoder: {config.freeze_sam_image_encoder}")
    
    # Tokenizer/Processor設定
    tokenizer = AutoTokenizer.from_pretrained(config.qwen_model_name)
    processor = AutoProcessor.from_pretrained(config.qwen_model_name)
    
    # モデル初期化
    print("\nモデル初期化中...")
    model = LISA_Model(config)
    model.set_tokenizer(tokenizer)
    
    # SAM ImageEncoderの状態確認
    print(f"\n=== SAM ImageEncoder状態 ===")
    print(f"SAM ImageEncoder training mode: {model.sam_image_encoder.training}")
    print(f"SAM encoder config stored: {model.sam_encoder_config}")
    
    # SAM ImageEncoderのパラメータ確認
    sam_params_total = sum(p.numel() for p in model.sam_image_encoder.parameters())
    sam_params_trainable = sum(p.numel() for p in model.sam_image_encoder.parameters() if p.requires_grad)
    
    print(f"\nSAM ImageEncoderパラメータ:")
    print(f"  総パラメータ: {sam_params_total:,}")
    print(f"  学習可能: {sam_params_trainable:,}")
    print(f"  凍結率: {(sam_params_total-sam_params_trainable)/sam_params_total*100:.1f}%")
    
    # モジュール構造を確認
    print(f"\n=== SAM ImageEncoderモジュール構造 ===")
    for name, module in model.sam_image_encoder.named_modules():
        if hasattr(module, 'training'):
            print(f"{name}: training={module.training}")
        if len(list(module.children())) == 0:  # leaf module
            if hasattr(module, 'training'):
                trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in module.parameters())
                if total_params > 0:
                    print(f"  {name}: {trainable_params}/{total_params} trainable")

if __name__ == "__main__":
    check_sam_encoder_state()