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
    
    # カテゴリ別にパラメータを分類（configの命名と完全一致）
    categories = defaultdict(lambda: {'total': 0, 'trainable': 0, 'params': []})
    
    for name, param in model.named_parameters():
        numel = param.numel()
        is_trainable = param.requires_grad
        
        # カテゴリ分類（configの freeze_* 命名に完全対応）
        # Token-FPNを先に判定（upsample_s1などでsamが誤検出されるのを防ぐ）
        if 'token_fpn' in name or 'fpn' in name:
            category = 'Token-FPN (freeze_token_fpn)'
        elif 'qwen' in name:
            if 'lora_A' in name or 'lora_B' in name:
                category = 'Qwen LoRA (freeze_qwen_lora)'
            elif ('word_embeddings' in name or 'embed_tokens' in name):
                # embedding層全体がtrainableかどうかをチェック
                if is_trainable:
                    category = 'SEG Token (freeze_seg_token)'
                else:
                    # embedding層は凍結されているが、SEG tokenの個別要素がtrainableか確認
                    seg_token_trainable = False
                    try:
                        if hasattr(model, 'seg_token_id') and model.seg_token_id is not None:
                            # SEG tokenの特定要素のrequires_gradをチェック
                            if param.dim() >= 2:
                                seg_token_trainable = param[model.seg_token_id].requires_grad
                    except:
                        pass
                    
                    if seg_token_trainable:
                        # SEG tokenのembedding次元数を取得
                        embed_dim = param.shape[-1] if param.dim() >= 2 else 1
                        categories['SEG Token (freeze_seg_token)']['total'] += embed_dim
                        categories['SEG Token (freeze_seg_token)']['trainable'] += embed_dim
                        categories['SEG Token (freeze_seg_token)']['params'].append((f"{name}[seg_token:{model.seg_token_id}]", embed_dim, True))
                        
                        # 残りの部分をQwen Baseに追加
                        remaining_params = numel - embed_dim
                        categories['Qwen Base (freeze_qwen_base)']['total'] += remaining_params
                        categories['Qwen Base (freeze_qwen_base)']['params'].append((f"{name}[except_seg_token]", remaining_params, False))
                        continue  # 通常の処理をスキップ
                    else:
                        category = 'Qwen Base (freeze_qwen_base)'
            else:
                category = 'Qwen Base (freeze_qwen_base)'
        elif 'sam' in name or 'sam_model' in name:
            if 'lora_A' in name or 'lora_B' in name or 'lora' in name.lower():
                # LoRAパラメータ（lora_A, lora_B）またはLoRA関連
                category = 'SAM LoRA (freeze_sam_lora)'
            elif 'mask_decoder' in name:
                # MaskDecoder内のパラメータ
                # LoRAパラメータかどうかで分類を決定
                if 'lora_A' in name or 'lora_B' in name:
                    # 明示的にLoRAパラメータの場合
                    category = 'SAM LoRA (freeze_sam_lora)'
                else:
                    # 通常のMaskDecoderパラメータ
                    # freeze_sam_mask_decoder_base=Falseの場合は学習可能
                    category = 'SAM MaskDecoder Base (freeze_sam_mask_decoder_base)'
            elif 'prompt_encoder' in name:
                category = 'SAM PromptEncoder (freeze_sam_prompt_encoder)'
            elif 'image_encoder' in name:
                category = 'SAM ImageEncoder (freeze_sam_image_encoder)'
            elif 'memory_attention' in name:
                category = 'SAM MemoryAttention (freeze_sam_memory_attention)'
            elif 'memory_encoder' in name:
                # メモリエンコーダー（ビデオトラッキング用）
                category = 'SAM Video Components (frozen)'
            elif 'maskmem' in name or 'no_mem' in name or 'no_obj' in name:
                # ビデオトラッキング関連のメモリコンポーネント
                category = 'SAM Video Components (frozen)'
            elif 'mask_downsample' in name:
                # マスクダウンサンプリング（ビデオ用）
                category = 'SAM Video Components (frozen)'
            elif 'obj_ptr' in name or 'spatial_add_pos_embed' in name:
                # オブジェクトポインタと空間位置埋め込み（ビデオ用）
                category = 'SAM Video Components (frozen)'
            else:
                # その他のSAMコンポーネント
                category = 'SAM Other (Misc)'
        elif 'image_adapter' in name:
            category = 'Image Adapter (freeze_image_adapter)'
        elif 'text_prompt_proj' in name or 'prompt_proj' in name:
            category = 'Text Prompt Projector (freeze_text_prompt_projector)'
        elif 'prompt_beta' in name:
            category = 'Prompt Beta (freeze_prompt_beta)'
        elif 'seg_token_embedding' in name:
            category = 'SEG Token (freeze_seg_token)'
        else:
            category = 'Other'
        
        # 通常の集計
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
        log.info(f"  {i:2}. {category:45} {param_count:12,} ({percentage:5.1f}%)")
    
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
        log.info(f"  - {category:45} {param_count:12,} ({percentage:5.1f}%)")
    
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