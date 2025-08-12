#!/usr/bin/env python3
"""
モデル関連のユーティリティ関数
"""

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

def display_parameter_statistics(model, logger_name=None):
    """
    モデルのパラメータ統計を詳細に表示
    
    Args:
        model: PyTorchモデル
        logger_name: ロガー名（指定しない場合は現在のモジュールのロガーを使用）
    """
    if logger_name:
        log = logging.getLogger(logger_name)
    else:
        log = logger
    
    # カテゴリ別にパラメータを分類
    categories = defaultdict(lambda: {'total': 0, 'trainable': 0, 'params': []})
    
    for name, param in model.named_parameters():
        numel = param.numel()
        is_trainable = param.requires_grad
        
        # カテゴリ分類
        if 'qwen' in name:
            if 'lora_A' in name or 'lora_B' in name:
                category = 'Qwen LoRA'
            elif 'word_embeddings' in name or 'embed_tokens' in name:
                if is_trainable:
                    category = 'SEG Token Embedding'
                else:
                    category = 'Qwen Base (frozen)'
            else:
                category = 'Qwen Base (frozen)'
        elif 'sam' in name or 'sam_model' in name:
            if 'lora' in name.lower():
                category = 'SAM LoRA'
            elif 'mask_decoder' in name:
                category = 'SAM MaskDecoder'
            elif 'prompt_encoder' in name:
                category = 'SAM PromptEncoder'
            elif 'image_encoder' in name:
                if 'neck' in name:
                    category = 'SAM Neck (FPN)'
                else:
                    category = 'SAM ImageEncoder'
            elif 'memory_encoder' in name:
                category = 'SAM MemoryEncoder (Video)'
            elif 'memory_attention' in name:
                category = 'SAM MemoryAttention (Video)'
            elif 'obj_ptr_proj' in name:
                category = 'SAM ObjectPointer (Video)'
            elif 'spatial_add_pos_embed' in name:
                category = 'SAM SpatialPosEmbed (Video)'
            elif 'point_emb' in name or 'pe_layer' in name:
                category = 'SAM PositionalEncoding'
            elif 'token_learner' in name:
                category = 'SAM TokenLearner'
            else:
                # より詳細なサブカテゴリ
                if 'conv' in name.lower():
                    category = 'SAM Convolutions'
                elif 'norm' in name.lower():
                    category = 'SAM Normalization'
                elif 'proj' in name.lower():
                    category = 'SAM Projections'
                else:
                    category = 'SAM Other (Misc)'
        elif 'image_adapter' in name:
            category = 'Image Adapter'
        elif 'text_prompt_proj' in name or 'prompt_proj' in name:
            category = 'Text Prompt Projector'
        elif 'prompt_beta' in name:
            category = 'Prompt Beta'
        elif 'fpn' in name or 'token_fpn' in name:
            category = 'Token-FPN'
        else:
            category = 'Other'
        
        categories[category]['total'] += numel
        if is_trainable:
            categories[category]['trainable'] += numel
        categories[category]['params'].append((name, numel, is_trainable))
    
    # 全体統計
    total_all = sum(cat['total'] for cat in categories.values())
    trainable_all = sum(cat['trainable'] for cat in categories.values())
    frozen_all = total_all - trainable_all
    
    # 表示
    log.info("="*80)
    log.info("パラメータ統計詳細")
    log.info("="*80)
    log.info(f"総パラメータ数: {total_all:,}")
    log.info(f"学習可能パラメータ数: {trainable_all:,}")
    log.info(f"凍結パラメータ数: {frozen_all:,}")
    log.info(f"学習可能パラメータの割合: {100 * trainable_all / total_all:.2f}%")
    
    # 学習可能コンポーネントの詳細
    log.info("-"*80)
    log.info("学習可能コンポーネントの内訳:")
    
    trainable_components = []
    for category, info in categories.items():
        if info['trainable'] > 0:
            trainable_components.append((category, info['trainable']))
    
    # サイズ順にソート
    trainable_components.sort(key=lambda x: x[1], reverse=True)
    
    for i, (category, param_count) in enumerate(trainable_components, 1):
        percentage = (param_count / trainable_all * 100) if trainable_all > 0 else 0
        log.info(f"  {i:2}. {category:25} {param_count:12,} ({percentage:5.1f}%)")
    
    # 凍結コンポーネントのサマリー
    log.info("-"*80)
    log.info("凍結コンポーネント:")
    
    frozen_components = []
    for category, info in categories.items():
        frozen_count = info['total'] - info['trainable']
        if frozen_count > 0:
            frozen_components.append((category, frozen_count))
    
    frozen_components.sort(key=lambda x: x[1], reverse=True)
    
    # すべての凍結コンポーネントを表示（省略なし）
    for category, param_count in frozen_components:
        percentage = (param_count / frozen_all * 100) if frozen_all > 0 else 0
        log.info(f"  - {category:25} {param_count:12,} ({percentage:5.1f}%)")
    
    log.info("="*80)

def get_parameter_summary(model):
    """
    モデルのパラメータサマリーを取得（辞書形式で返す）
    
    Args:
        model: PyTorchモデル
        
    Returns:
        dict: パラメータ統計情報
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
        'trainable_ratio': (trainable_params / total_params * 100) if total_params > 0 else 0
    }