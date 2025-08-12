#!/usr/bin/env python3
"""
freeze_*設定の整合性確認テスト
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_freeze_config():
    """freeze_*設定の動作確認"""
    
    logger.info("="*80)
    logger.info("Freeze設定の整合性テスト")
    logger.info("="*80)
    
    # Test Case 1: デフォルト設定（ほぼ全て学習可能）
    logger.info("\n【Test Case 1: デフォルト設定】")
    config1 = LISAConfig(
        qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        sam_model_name="./checkpoints/sam2.1_hiera_large.pt",
        device_map="cpu",
        torch_dtype="auto",
        use_flash_attention=False
    )
    
    # デフォルト値を確認
    logger.info(f"freeze_qwen_lora: {config1.freeze_qwen_lora} (False = Qwen LoRA学習可能)")
    logger.info(f"freeze_seg_token: {config1.freeze_seg_token} (False = SEGトークン学習可能)")
    logger.info(f"freeze_sam_lora: {config1.freeze_sam_lora} (False = SAM LoRA学習可能)")
    logger.info(f"freeze_image_adapter: {config1.freeze_image_adapter} (False = Image Adapter学習可能)")
    logger.info(f"freeze_text_prompt_projector: {config1.freeze_text_prompt_projector} (False = Text Projector学習可能)")
    logger.info(f"freeze_token_fpn: {config1.freeze_token_fpn} (False = Token-FPN学習可能)")
    logger.info(f"freeze_prompt_beta: {config1.freeze_prompt_beta} (False = Beta学習可能)")
    
    # Test Case 2: アライメントステージ設定
    logger.info("\n【Test Case 2: アライメントステージ設定】")
    config2 = LISAConfig(
        qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        sam_model_name="./checkpoints/sam2.1_hiera_large.pt",
        device_map="cpu",
        torch_dtype="auto",
        use_flash_attention=False,
        # アライメントステージでは一部のみ学習
        freeze_qwen_lora=False,  # 学習可能
        freeze_seg_token=False,  # 学習可能
        freeze_text_prompt_projector=False,  # 学習可能
        freeze_prompt_beta=False,  # 学習可能
        # 以下は強制的に凍結
        freeze_sam_lora=True,  # 凍結
        freeze_image_adapter=True,  # 凍結
        freeze_token_fpn=True,  # 凍結
    )
    
    logger.info("アライメントステージでの設定:")
    logger.info(f"  学習可能: Qwen LoRA, SEG Token, Text Projector, Beta")
    logger.info(f"  凍結: SAM LoRA, Image Adapter, Token-FPN")
    
    # Test Case 3: モデル初期化と実際の学習可能パラメータ確認
    logger.info("\n【Test Case 3: 実際のパラメータ状態確認】")
    
    tokenizer = prepare_tokenizer_for_lisa(
        model_name=config2.qwen_model_name,
        seg_token=config2.seg_token
    )
    
    model = LISA_Model(config2)
    model.set_tokenizer(tokenizer)
    
    # prompt_betaを追加
    model.prompt_beta = torch.nn.Parameter(torch.tensor(0.01))
    
    # LoRA適用
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model.qwen = get_peft_model(model.qwen, lora_config)
    
    # SAM LoRAは凍結設定なので適用しない
    if not config2.freeze_sam_lora:
        model.add_sam_lora(lora_r=8, lora_alpha=16, lora_dropout=0.1)
    
    # パラメータカウント
    trainable_count = 0
    frozen_count = 0
    categories = {
        'qwen_lora': 0,
        'seg_token': 0,
        'text_prompt': 0,
        'prompt_beta': 0,
        'sam_lora': 0,
        'image_adapter': 0,
        'token_fpn': 0,
        'other': 0
    }
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_count += 1
            
            if "qwen" in name and "lora" in name:
                categories['qwen_lora'] += 1
            elif "embed_tokens" in name:
                categories['seg_token'] += 1
            elif "text_prompt_proj" in name:
                categories['text_prompt'] += 1
            elif "prompt_beta" in name:
                categories['prompt_beta'] += 1
            elif "sam" in name and "lora" in name:
                categories['sam_lora'] += 1
            elif "image_adapter" in name:
                categories['image_adapter'] += 1
            elif "fpn" in name or "token_fpn" in name:
                categories['token_fpn'] += 1
            else:
                categories['other'] += 1
        else:
            frozen_count += 1
    
    logger.info(f"\n学習可能パラメータ: {trainable_count}個")
    logger.info(f"凍結パラメータ: {frozen_count}個")
    
    logger.info("\n学習可能パラメータの内訳:")
    for category, count in categories.items():
        if count > 0:
            logger.info(f"  {category}: {count}個")
    
    # 検証
    logger.info("\n【検証結果】")
    
    # 期待値との比較
    if categories['qwen_lora'] > 0 and not config2.freeze_qwen_lora:
        logger.info("✅ Qwen LoRA: 正しく学習可能")
    elif categories['qwen_lora'] == 0 and config2.freeze_qwen_lora:
        logger.info("✅ Qwen LoRA: 正しく凍結")
    else:
        logger.warning("❌ Qwen LoRA: 設定と実際の状態が不一致")
    
    if categories['seg_token'] > 0 and not config2.freeze_seg_token:
        logger.info("✅ SEG Token: 正しく学習可能")
    elif categories['seg_token'] == 0 and config2.freeze_seg_token:
        logger.info("✅ SEG Token: 正しく凍結")
    else:
        logger.warning("⚠️ SEG Token: embed_tokensは全体で1つのパラメータ")
    
    if categories['text_prompt'] > 0 and not config2.freeze_text_prompt_projector:
        logger.info("✅ Text Prompt Projector: 正しく学習可能")
    elif categories['text_prompt'] == 0 and config2.freeze_text_prompt_projector:
        logger.info("✅ Text Prompt Projector: 正しく凍結")
    else:
        logger.warning("❌ Text Prompt Projector: 設定と実際の状態が不一致")
    
    if categories['prompt_beta'] > 0 and not config2.freeze_prompt_beta:
        logger.info("✅ Prompt Beta: 正しく学習可能")
    elif categories['prompt_beta'] == 0 and config2.freeze_prompt_beta:
        logger.info("✅ Prompt Beta: 正しく凍結")
    else:
        logger.warning("❌ Prompt Beta: 設定と実際の状態が不一致")
    
    if categories['sam_lora'] == 0 and config2.freeze_sam_lora:
        logger.info("✅ SAM LoRA: 正しく凍結（LoRA未適用）")
    elif categories['sam_lora'] > 0 and not config2.freeze_sam_lora:
        logger.info("✅ SAM LoRA: 正しく学習可能")
    else:
        logger.warning("❌ SAM LoRA: 設定と実際の状態が不一致")
    
    if categories['image_adapter'] == 0 and config2.freeze_image_adapter:
        logger.info("✅ Image Adapter: 正しく凍結")
    elif categories['image_adapter'] > 0 and not config2.freeze_image_adapter:
        logger.info("✅ Image Adapter: 正しく学習可能")
    else:
        logger.warning("❌ Image Adapter: 設定と実際の状態が不一致")
    
    if categories['token_fpn'] == 0 and config2.freeze_token_fpn:
        logger.info("✅ Token-FPN: 正しく凍結")
    elif categories['token_fpn'] > 0 and not config2.freeze_token_fpn:
        logger.info("✅ Token-FPN: 正しく学習可能")
    else:
        logger.warning("❌ Token-FPN: 設定と実際の状態が不一致")
    
    logger.info("\n" + "="*80)
    logger.info("テスト完了")
    logger.info("="*80)

if __name__ == "__main__":
    test_freeze_config()