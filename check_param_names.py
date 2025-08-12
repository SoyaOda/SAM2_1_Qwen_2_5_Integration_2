#!/usr/bin/env python3
"""
アライメントステージ実行時のパラメータ名を確認
"""

import sys
import torch
import logging
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.config import LISAConfig
from src.models.lisa_model import LISA_Model
from src.utils.tokenizer_utils import prepare_tokenizer_for_lisa
from peft import LoraConfig, get_peft_model, TaskType

# ロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_params_before_lora():
    """LoRA適用前のパラメータ名確認"""
    logger.info("="*80)
    logger.info("LoRA適用前のパラメータ名確認")
    logger.info("="*80)
    
    config = LISAConfig(
        qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        sam_model_name="./checkpoints/sam2.1_hiera_large.pt",
        device_map="cpu",
        torch_dtype="auto",
        use_flash_attention=False,
    )
    
    tokenizer = prepare_tokenizer_for_lisa(
        model_name=config.qwen_model_name,
        seg_token=config.seg_token
    )
    
    model = LISA_Model(config)
    model.set_tokenizer(tokenizer)
    
    # パラメータ名を確認
    logger.info("\n主要なパラメータ名の確認:")
    
    # Qwen関連
    qwen_params = []
    seg_params = []
    for name, param in model.named_parameters():
        if 'qwen' in name:
            if 'word_embeddings' in name or 'embed_tokens' in name:
                seg_params.append(name)
            elif param.requires_grad:  # 学習可能なもののみ
                qwen_params.append(name)
    
    logger.info(f"\nQwenの学習可能パラメータ（最初の5個）:")
    for name in qwen_params[:5]:
        logger.info(f"  - {name}")
    
    logger.info(f"\nSEGトークン関連パラメータ:")
    for name in seg_params[:3]:
        logger.info(f"  - {name}")
    
    # Text Prompt Projector
    text_proj_params = [name for name, _ in model.named_parameters() if 'text_prompt_proj' in name]
    logger.info(f"\nText Prompt Projectorパラメータ:")
    for name in text_proj_params[:3]:
        logger.info(f"  - {name}")
    
    # prompt_beta
    beta_params = [name for name, _ in model.named_parameters() if 'beta' in name.lower()]
    logger.info(f"\nBetaパラメータ:")
    for name in beta_params:
        logger.info(f"  - {name}")
    
    return model

def check_params_after_lora(model):
    """LoRA適用後のパラメータ名確認"""
    logger.info("\n" + "="*80)
    logger.info("LoRA適用後のパラメータ名確認")
    logger.info("="*80)
    
    # Qwen LoRA適用
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model.qwen = get_peft_model(model.qwen, lora_config)
    
    # SAM LoRA適用
    model.add_sam_lora(lora_r=8, lora_alpha=16, lora_dropout=0.1)
    
    # LoRA適用後のパラメータ名確認
    logger.info("\nLoRA適用後の学習可能パラメータ:")
    
    lora_params = []
    seg_params = []
    text_proj_params = []
    beta_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if 'lora' in name.lower():
            lora_params.append(name)
        elif 'word_embeddings' in name or 'embed_tokens' in name:
            seg_params.append(name)
        elif 'text_prompt_proj' in name:
            text_proj_params.append(name)
        elif 'beta' in name.lower():
            beta_params.append(name)
    
    logger.info(f"\nLoRAパラメータ（最初の5個）:")
    for name in lora_params[:5]:
        logger.info(f"  - {name}")
    
    logger.info(f"\nSEGトークン関連（学習可能）:")
    for name in seg_params:
        logger.info(f"  - {name}")
    
    logger.info(f"\nText Prompt Projector（学習可能）:")
    for name in text_proj_params[:3]:
        logger.info(f"  - {name}")
    
    logger.info(f"\nBetaパラメータ（学習可能）:")
    for name in beta_params:
        logger.info(f"  - {name}")
    
    # アライメント条件のテスト
    logger.info("\n" + "="*80)
    logger.info("アライメント条件のマッチングテスト")
    logger.info("="*80)
    
    align_matched = {
        'qwen_lora': [],
        'seg_token': [],
        'text_prompt': [],
        'prompt_beta': []
    }
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # 現在の条件
        if "qwen" in name and "lora" in name:
            align_matched['qwen_lora'].append(name)
        elif "word_embeddings" in name:
            align_matched['seg_token'].append(name)
        elif "text_prompt_proj" in name:
            align_matched['text_prompt'].append(name)
        elif "prompt_beta" in name:
            align_matched['prompt_beta'].append(name)
    
    for category, params in align_matched.items():
        logger.info(f"\n{category}: {len(params)}個マッチ")
        for name in params[:3]:
            logger.info(f"  - {name}")

def main():
    """メイン実行"""
    logger.info("パラメータ名の確認を開始\n")
    
    # LoRA適用前
    model = check_params_before_lora()
    
    # LoRA適用後
    check_params_after_lora(model)
    
    logger.info("\n確認完了")

if __name__ == "__main__":
    main()