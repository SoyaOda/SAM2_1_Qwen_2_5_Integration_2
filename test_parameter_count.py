#!/usr/bin/env python3
"""
パラメータカウントの詳細検証スクリプト
"""

import sys
import torch
import logging
from pathlib import Path
from collections import defaultdict

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.config import LISAConfig
from src.models.lisa_model import LISA_Model
from src.utils.tokenizer_utils import prepare_tokenizer_for_lisa

# ロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_parameters():
    """パラメータの詳細分析"""
    logger.info("="*80)
    logger.info("LISA改モデルのパラメータ分析")
    logger.info("="*80)
    
    # モデルの初期化
    config = LISAConfig()
    tokenizer = prepare_tokenizer_for_lisa(model_name=config.qwen_model_name, seg_token=config.seg_token)
    model = LISA_Model(config)
    model.set_tokenizer(tokenizer)
    
    # カテゴリ別パラメータカウント
    param_groups = defaultdict(lambda: {'total': 0, 'trainable': 0, 'params': []})
    
    for name, param in model.named_parameters():
        numel = param.numel()
        is_trainable = param.requires_grad
        
        # カテゴリ分類
        if 'qwen' in name:
            if 'lora' in name:
                category = 'Qwen LoRA'
            elif 'word_embeddings' in name:
                category = 'Qwen Embeddings'
            else:
                category = 'Qwen Base'
        elif 'sam' in name:
            if 'lora' in name:
                category = 'SAM LoRA'
            elif 'mask_decoder' in name:
                category = 'SAM MaskDecoder'
            elif 'prompt_encoder' in name:
                category = 'SAM PromptEncoder'
            elif 'image_encoder' in name:
                category = 'SAM ImageEncoder'
            else:
                category = 'SAM Other'
        elif 'image_adapter' in name:
            category = 'Image Adapter'
        elif 'text_prompt_proj' in name or 'prompt_proj' in name:
            category = 'Text Prompt Projector'
        elif 'prompt_beta' in name:
            category = 'Prompt Beta'
        else:
            category = 'Other'
        
        param_groups[category]['total'] += numel
        if is_trainable:
            param_groups[category]['trainable'] += numel
        param_groups[category]['params'].append((name, numel, is_trainable))
    
    # 結果表示
    logger.info("\n" + "="*80)
    logger.info("カテゴリ別パラメータ数")
    logger.info("="*80)
    
    total_all = 0
    trainable_all = 0
    
    for category in sorted(param_groups.keys()):
        info = param_groups[category]
        total_all += info['total']
        trainable_all += info['trainable']
        
        percentage = (info['trainable'] / info['total'] * 100) if info['total'] > 0 else 0
        logger.info(f"\n【{category}】")
        logger.info(f"  総数: {info['total']:,}")
        logger.info(f"  学習可能: {info['trainable']:,}")
        logger.info(f"  学習可能率: {percentage:.2f}%")
        logger.info(f"  パラメータ数: {len(info['params'])}個")
    
    # 全体統計
    logger.info("\n" + "="*80)
    logger.info("全体統計")
    logger.info("="*80)
    logger.info(f"総パラメータ数: {total_all:,}")
    logger.info(f"学習可能パラメータ数: {trainable_all:,}")
    logger.info(f"学習可能パラメータの割合: {100 * trainable_all / total_all:.2f}%")
    
    # 標準的な計算方法との比較
    logger.info("\n" + "="*80)
    logger.info("標準的な計算方法との比較")
    logger.info("="*80)
    
    standard_total = sum(p.numel() for p in model.parameters())
    standard_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"model.parameters()による総数: {standard_total:,}")
    logger.info(f"model.parameters()による学習可能数: {standard_trainable:,}")
    
    if standard_total != total_all:
        logger.warning(f"⚠️ 総数に差異: {abs(standard_total - total_all):,}")
    else:
        logger.info("✅ 総数は一致")
        
    if standard_trainable != trainable_all:
        logger.warning(f"⚠️ 学習可能数に差異: {abs(standard_trainable - trainable_all):,}")
    else:
        logger.info("✅ 学習可能数は一致")
    
    # 学習可能パラメータの詳細リスト
    logger.info("\n" + "="*80)
    logger.info("学習可能パラメータの詳細（上位20個）")
    logger.info("="*80)
    
    trainable_params = []
    for category, info in param_groups.items():
        for name, numel, is_trainable in info['params']:
            if is_trainable:
                trainable_params.append((name, numel, category))
    
    trainable_params.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, numel, category) in enumerate(trainable_params[:20], 1):
        logger.info(f"{i:2d}. {name[:60]:<60} {numel:>12,} ({category})")
    
    # 特定のパラメータの確認
    logger.info("\n" + "="*80)
    logger.info("重要パラメータの確認")
    logger.info("="*80)
    
    # prompt_betaの確認
    has_prompt_beta = hasattr(model, 'prompt_beta')
    logger.info(f"prompt_beta存在: {has_prompt_beta}")
    if has_prompt_beta:
        logger.info(f"  - 学習可能: {model.prompt_beta.requires_grad}")
        logger.info(f"  - サイズ: {model.prompt_beta.numel()}")
        logger.info(f"  - 値: {model.prompt_beta.item():.6f}")
    
    # SEGトークンの確認
    seg_token_id = tokenizer.convert_tokens_to_ids(config.seg_token)
    # qwen_modelはLISA_Modelの属性
    if hasattr(model, 'qwen'):
        # Qwen2.5-VLの構造: get_input_embeddings()を使用
        embed_layer = model.qwen.get_input_embeddings()
        total_vocab_size = embed_layer.weight.shape[0]
        logger.info(f"\nSEGトークン:")
        logger.info(f"  - ID: {seg_token_id}")
        logger.info(f"  - 語彙サイズ: {total_vocab_size:,}")
        logger.info(f"  - Embedding学習可能: {embed_layer.weight.requires_grad}")
    
    # LoRAの確認
    lora_count = sum(1 for name, _ in model.named_parameters() if 'lora' in name.lower())
    lora_trainable = sum(1 for name, p in model.named_parameters() if 'lora' in name.lower() and p.requires_grad)
    logger.info(f"\nLoRAパラメータ:")
    logger.info(f"  - 総数: {lora_count}")
    logger.info(f"  - 学習可能: {lora_trainable}")
    
    return total_all, trainable_all

def check_duplicate_parameters():
    """重複パラメータのチェック"""
    logger.info("\n" + "="*80)
    logger.info("重複パラメータチェック")
    logger.info("="*80)
    
    config = LISAConfig()
    tokenizer = prepare_tokenizer_for_lisa(model_name=config.qwen_model_name, seg_token=config.seg_token)
    model = LISA_Model(config)
    model.set_tokenizer(tokenizer)
    
    # パラメータIDの収集
    param_ids = {}
    duplicates = []
    
    for name, param in model.named_parameters():
        param_id = id(param)
        if param_id in param_ids:
            duplicates.append((name, param_ids[param_id]))
        else:
            param_ids[param_id] = name
    
    if duplicates:
        logger.warning("⚠️ 重複パラメータが見つかりました:")
        for current_name, original_name in duplicates:
            logger.warning(f"  - {current_name} は {original_name} と同じパラメータ")
    else:
        logger.info("✅ 重複パラメータなし")
    
    return len(duplicates) == 0

def main():
    """メイン実行"""
    logger.info("🔍 パラメータカウント検証を開始します\n")
    
    # パラメータ分析
    total, trainable = analyze_parameters()
    
    # 重複チェック
    no_duplicates = check_duplicate_parameters()
    
    # 結果サマリー
    logger.info("\n" + "="*80)
    logger.info("検証結果サマリー")
    logger.info("="*80)
    
    if no_duplicates:
        logger.info("✅ パラメータカウントは正確です")
        logger.info(f"   総パラメータ数: {total:,}")
        logger.info(f"   学習可能パラメータ数: {trainable:,}")
        logger.info(f"   学習可能率: {100 * trainable / total:.2f}%")
    else:
        logger.warning("⚠️ 重複パラメータがあるため、カウントが不正確な可能性があります")
    
    # minimal_train.pyで表示される値との比較
    logger.info("\n" + "="*80)
    logger.info("minimal_train.pyの表示値との比較")
    logger.info("="*80)
    logger.info("表示されている値:")
    logger.info("  総パラメータ数: 3,988,463,539")
    logger.info("  学習可能パラメータ数: 229,619,234")
    logger.info("  学習可能パラメータの割合: 5.76%")
    logger.info("\n実際の値:")
    logger.info(f"  総パラメータ数: {total:,}")
    logger.info(f"  学習可能パラメータ数: {trainable:,}")
    logger.info(f"  学習可能パラメータの割合: {100 * trainable / total:.2f}%")
    
    return no_duplicates

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)