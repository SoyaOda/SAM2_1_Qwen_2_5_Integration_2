#!/usr/bin/env python3
"""
アライメントステージの修正を簡単にテスト
"""

import sys
import torch
import logging
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from test_common_config import (
    SEED, set_random_seed
)

from src.config import LISAConfig
from src.models.lisa_model import LISA_Model
from src.utils.tokenizer_utils import prepare_tokenizer_for_lisa
from transformers import AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType

# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_alignment_freeze():
    """アライメントステージの凍結設定をテスト"""
    
    set_random_seed(SEED)
    device = torch.device('cpu')  # CPUでテスト
    
    logger.info("="*80)
    logger.info("アライメントステージ凍結設定テスト")
    logger.info("="*80)
    
    # 設定
    config = LISAConfig(
        qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        sam_model_name="./checkpoints/sam2.1_hiera_large.pt",
        device_map="cpu",
        torch_dtype="auto",
        use_flash_attention=False,
        # Freeze flags (False = trainable)
        freeze_qwen_lora=False,
        freeze_seg_token=False,
        freeze_text_prompt_projector=False,
        freeze_prompt_beta=False,
        freeze_sam_lora=False,
        freeze_image_adapter=False,
        freeze_token_fpn=False,
        # Freeze flags
        freeze_sam_mask_decoder_base=False,  # Test A4ではFalse
        freeze_sam_prompt_encoder=True,
        freeze_sam_image_encoder=True
    )
    
    # トークナイザーとプロセッサーの準備
    tokenizer = prepare_tokenizer_for_lisa(
        model_name=config.qwen_model_name,
        seg_token=config.seg_token
    )
    
    processor = AutoProcessor.from_pretrained(config.qwen_model_name)
    if config.seg_token not in processor.tokenizer.get_vocab():
        processor.tokenizer.add_special_tokens({"additional_special_tokens": [config.seg_token]})
    
    # モデル作成
    logger.info("モデルを初期化...")
    model = LISA_Model(config)
    model.set_tokenizer(tokenizer)
    
    # prompt_betaを追加（A4 Addition用）
    model.prompt_beta = torch.nn.Parameter(torch.tensor(0.01))
    
    # LoRA適用
    logger.info("LoRAを適用...")
    
    # Qwen LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model.qwen = get_peft_model(model.qwen, lora_config)
    
    # SAM LoRA
    model.add_sam_lora(lora_r=8, lora_alpha=16, lora_dropout=0.1)
    
    logger.info("\n" + "="*80)
    logger.info("LoRA適用後の学習可能パラメータ")
    logger.info("="*80)
    
    # 学習可能パラメータをカウント
    categories = {
        'qwen_lora': [],
        'seg_token': [],
        'text_prompt': [],
        'prompt_beta': [],
        'sam_lora': [],
        'image_adapter': [],
        'token_fpn': [],
        'other': []
    }
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if "qwen" in name and "lora" in name:
            categories['qwen_lora'].append(name)
        elif "embed_tokens" in name:
            categories['seg_token'].append(name)
        elif "text_prompt_proj" in name:
            categories['text_prompt'].append(name)
        elif "prompt_beta" in name:
            categories['prompt_beta'].append(name)
        elif "sam" in name and "lora" in name:
            categories['sam_lora'].append(name)
        elif "image_adapter" in name:
            categories['image_adapter'].append(name)
        elif "fpn" in name or "token_fpn" in name:
            categories['token_fpn'].append(name)
        else:
            categories['other'].append(name)
    
    for category, params in categories.items():
        if params:
            logger.info(f"{category}: {len(params)}個")
            for p in params[:2]:
                logger.info(f"  - {p[:80]}")
    
    # アライメントステージの凍結設定をシミュレート
    logger.info("\n" + "="*80)
    logger.info("アライメントステージ用の凍結設定を適用（シミュレーション）")
    logger.info("="*80)
    
    align_trainable = {
        'qwen_lora': [],
        'seg_token': [],
        'text_prompt': [],
        'prompt_beta': [],
        'others_frozen': []
    }
    
    for name, param in model.named_parameters():
        should_train = False
        
        # アライメントステージの条件（修正版）
        if "qwen" in name and "lora" in name and not config.freeze_qwen_lora:
            should_train = True
            align_trainable['qwen_lora'].append(name)
        elif "embed_tokens" in name and not config.freeze_seg_token:
            should_train = True
            align_trainable['seg_token'].append(name)
        elif "text_prompt_proj" in name and not config.freeze_text_prompt_projector:
            should_train = True
            align_trainable['text_prompt'].append(name)
        elif "prompt_beta" in name and not config.freeze_prompt_beta:
            should_train = True
            align_trainable['prompt_beta'].append(name)
        else:
            # アライメントステージでは他は全て凍結
            should_train = False
            if param.requires_grad:  # 通常は学習可能だが、アライメントで凍結されるもの
                align_trainable['others_frozen'].append(name)
    
    logger.info("アライメントステージで学習されるパラメータ:")
    for category, params in align_trainable.items():
        if params and category != 'others_frozen':
            logger.info(f"\n{category}: {len(params)}個")
            for p in params[:2]:
                logger.info(f"  - {p[:80]}")
    
    logger.info(f"\nアライメントで凍結されるパラメータ（通常は学習可能）: {len(align_trainable['others_frozen'])}個")
    for p in align_trainable['others_frozen'][:5]:
        logger.info(f"  - {p[:80]}")
    
    # 結果サマリー
    logger.info("\n" + "="*80)
    logger.info("結果サマリー")
    logger.info("="*80)
    
    total_trainable_align = sum(len(params) for category, params in align_trainable.items() if category != 'others_frozen')
    total_frozen_align = len(align_trainable['others_frozen'])
    
    logger.info(f"アライメントステージ:")
    logger.info(f"  - 学習可能: {total_trainable_align}個のパラメータグループ")
    logger.info(f"    - Qwen LoRA: {len(align_trainable['qwen_lora'])}個")
    logger.info(f"    - SEG Token: {len(align_trainable['seg_token'])}個") 
    logger.info(f"    - Text Prompt Projector: {len(align_trainable['text_prompt'])}個")
    logger.info(f"    - Prompt Beta: {len(align_trainable['prompt_beta'])}個")
    logger.info(f"  - 強制凍結: {total_frozen_align}個（SAM LoRA、Image Adapter、Token-FPN等）")
    
    # 検証
    logger.info("\n" + "="*80)
    logger.info("検証結果")
    logger.info("="*80)
    
    issues = []
    
    # Qwen LoRAがマッチするか
    if len(align_trainable['qwen_lora']) == 0:
        issues.append("❌ Qwen LoRAパラメータがマッチしません")
    else:
        logger.info(f"✅ Qwen LoRA: {len(align_trainable['qwen_lora'])}個マッチ")
    
    # SEGトークンがマッチするか
    if len(align_trainable['seg_token']) == 0:
        issues.append("⚠️ SEGトークンembeddingがマッチしません（embed_tokensは全体で1つのパラメータ）")
        logger.info("⚠️ SEGトークン: embed_tokensは全体で1つのパラメータなので個別管理が必要")
    else:
        logger.info(f"✅ SEGトークン: {len(align_trainable['seg_token'])}個マッチ")
    
    # Text Prompt Projectorがマッチするか
    if len(align_trainable['text_prompt']) == 0:
        issues.append("❌ Text Prompt Projectorパラメータがマッチしません")
    else:
        logger.info(f"✅ Text Prompt Projector: {len(align_trainable['text_prompt'])}個マッチ")
    
    # Prompt Betaがマッチするか
    if len(align_trainable['prompt_beta']) == 0:
        issues.append("❌ Prompt Betaパラメータがマッチしません")
    else:
        logger.info(f"✅ Prompt Beta: {len(align_trainable['prompt_beta'])}個マッチ")
    
    # SAM関連が凍結されているか
    sam_in_frozen = [p for p in align_trainable['others_frozen'] if 'sam' in p.lower()]
    if len(sam_in_frozen) > 0:
        logger.info(f"✅ SAM関連: {len(sam_in_frozen)}個が正しく凍結対象")
    
    if issues:
        logger.warning("\n問題点:")
        for issue in issues:
            logger.warning(issue)
    else:
        logger.info("\n✅ 全体的に正しく動作しています")
    
    logger.info("\n" + "="*80)
    logger.info("テスト完了")
    logger.info("="*80)

if __name__ == "__main__":
    test_alignment_freeze()