#!/usr/bin/env python3
"""
すべての学習可能パラメータを完全に調査するスクリプト
"""

import sys
import torch
import logging
from pathlib import Path
from collections import defaultdict
from peft import get_peft_model, LoraConfig, TaskType

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.config import LISAConfig
from src.models.lisa_model import LISA_Model
from src.utils.tokenizer_utils import prepare_tokenizer_for_lisa

# ロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_all_parameters():
    """すべてのパラメータを徹底的に分析"""
    logger.info("="*80)
    logger.info("LISA改モデルの完全パラメータ分析（LoRA適用後）")
    logger.info("="*80)
    
    # モデルの初期化（minimal_train.pyと同じ手順）
    config = LISAConfig(
        qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        sam_model_name="./checkpoints/sam2.1_hiera_large.pt",
        device_map="cuda",
        torch_dtype="auto",
        freeze_qwen=True,
        freeze_sam=True,
        train_seg_token=True,
        use_flash_attention=False
    )
    
    tokenizer = prepare_tokenizer_for_lisa(
        model_name=config.qwen_model_name,
        seg_token=config.seg_token
    )
    
    model = LISA_Model(config)
    model.set_tokenizer(tokenizer)
    
    # LoRAを適用（minimal_train.pyと同じ）
    logger.info("="*80)
    logger.info("LoRA適用中...")
    logger.info("="*80)
    
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
    
    # SAM LoRA（configにあれば）
    if hasattr(config, 'sam_lora_r') and config.sam_lora_r > 0:
        model.add_sam_lora(
            lora_r=config.sam_lora_r,
            lora_alpha=config.sam_lora_alpha,
            lora_dropout=config.sam_lora_dropout
        )
    
    # カテゴリ別に詳細分析
    categories = defaultdict(lambda: {
        'total': 0, 
        'trainable': 0, 
        'frozen': 0,
        'params': []
    })
    
    # すべてのパラメータを詳細に分類
    for name, param in model.named_parameters():
        numel = param.numel()
        is_trainable = param.requires_grad
        
        # 詳細なカテゴリ分類
        if 'qwen' in name:
            if 'lora_A' in name or 'lora_B' in name:
                category = '1. Qwen LoRA'
            elif 'word_embeddings' in name or 'embed_tokens' in name:
                if is_trainable and config.seg_token in name:
                    category = '2. SEG Token Embedding'
                else:
                    category = '3. Qwen Embeddings (Base)'
            else:
                category = '4. Qwen Base Model'
        
        elif 'sam' in name or 'sam_model' in name:
            if 'lora' in name.lower():
                category = '5. SAM LoRA'
            elif 'mask_decoder' in name:
                if 'transformer' in name:
                    category = '6. SAM MaskDecoder Transformer'
                elif 'output' in name:
                    category = '7. SAM MaskDecoder Output'
                else:
                    category = '8. SAM MaskDecoder Other'
            elif 'prompt_encoder' in name:
                category = '9. SAM PromptEncoder'
            elif 'image_encoder' in name:
                if 'neck' in name:
                    category = '10. SAM Neck (FPN)'
                else:
                    category = '11. SAM ImageEncoder'
            else:
                category = '12. SAM Other'
        
        elif 'image_adapter' in name:
            category = '13. Image Adapter'
        
        elif 'text_prompt_proj' in name or 'prompt_proj' in name:
            category = '14. Text Prompt Projector'
        
        elif 'prompt_beta' in name:
            category = '15. Prompt Beta (Addition Weight)'
        
        elif 'fpn' in name or 'neck' in name:
            category = '16. Token-FPN'
        
        else:
            category = '17. Other/Unknown'
        
        # 統計を更新
        categories[category]['total'] += numel
        if is_trainable:
            categories[category]['trainable'] += numel
        else:
            categories[category]['frozen'] += numel
        
        # パラメータ詳細を保存
        categories[category]['params'].append({
            'name': name,
            'numel': numel,
            'trainable': is_trainable,
            'shape': list(param.shape),
            'dtype': str(param.dtype)
        })
    
    # 結果表示
    logger.info("\n" + "="*80)
    logger.info("【詳細カテゴリ別パラメータ分析】")
    logger.info("="*80)
    
    total_all = 0
    trainable_all = 0
    frozen_all = 0
    
    for category in sorted(categories.keys()):
        info = categories[category]
        if info['total'] == 0:
            continue
            
        total_all += info['total']
        trainable_all += info['trainable']
        frozen_all += info['frozen']
        
        logger.info(f"\n【{category}】")
        logger.info(f"  総パラメータ数: {info['total']:,}")
        logger.info(f"  学習可能: {info['trainable']:,} ({info['trainable']/info['total']*100:.1f}%)")
        logger.info(f"  凍結: {info['frozen']:,}")
        logger.info(f"  パラメータ個数: {len(info['params'])}個")
        
        # 学習可能なパラメータの詳細（上位5個）
        trainable_params = [p for p in info['params'] if p['trainable']]
        if trainable_params:
            logger.info("  主要な学習可能パラメータ:")
            for i, p in enumerate(trainable_params[:5], 1):
                logger.info(f"    {i}. {p['name'][-50:]:50} shape={p['shape']} ({p['numel']:,})")
    
    # 全体統計
    logger.info("\n" + "="*80)
    logger.info("【全体統計サマリー】")
    logger.info("="*80)
    logger.info(f"総パラメータ数: {total_all:,}")
    logger.info(f"学習可能パラメータ数: {trainable_all:,}")
    logger.info(f"凍結パラメータ数: {frozen_all:,}")
    logger.info(f"学習可能率: {100 * trainable_all / total_all:.2f}%")
    
    # 学習可能コンポーネントのみのサマリー
    logger.info("\n" + "="*80)
    logger.info("【学習可能コンポーネントのみのサマリー】")
    logger.info("="*80)
    
    trainable_components = {}
    for category, info in categories.items():
        if info['trainable'] > 0:
            trainable_components[category] = info['trainable']
    
    # サイズ順にソート
    sorted_components = sorted(trainable_components.items(), key=lambda x: x[1], reverse=True)
    
    for category, param_count in sorted_components:
        percentage = (param_count / trainable_all * 100) if trainable_all > 0 else 0
        logger.info(f"{category:40} {param_count:12,} ({percentage:5.1f}%)")
    
    # 特殊なパラメータの確認
    logger.info("\n" + "="*80)
    logger.info("【特殊パラメータの詳細確認】")
    logger.info("="*80)
    
    # SEGトークン関連
    seg_token_params = []
    for name, param in model.named_parameters():
        if 'word_embeddings' in name or 'embed_tokens' in name:
            if param.requires_grad:
                seg_token_params.append((name, param.numel(), param.shape))
    
    if seg_token_params:
        logger.info("\nSEGトークン関連の学習可能パラメータ:")
        for name, numel, shape in seg_token_params:
            logger.info(f"  {name}: shape={shape}, params={numel:,}")
    else:
        logger.info("\nSEGトークン: 学習可能なembeddingパラメータなし（個別管理の可能性）")
    
    # prompt_betaの確認
    if hasattr(model, 'prompt_beta'):
        logger.info(f"\nprompt_beta:")
        logger.info(f"  存在: True")
        logger.info(f"  学習可能: {model.prompt_beta.requires_grad}")
        logger.info(f"  値: {model.prompt_beta.item():.6f}")
        logger.info(f"  サイズ: {model.prompt_beta.numel()}")
    
    # その他の隠れたパラメータを探す
    logger.info("\n" + "="*80)
    logger.info("【隠れたパラメータの探索】")
    logger.info("="*80)
    
    # bufferの確認
    buffers = []
    for name, buffer in model.named_buffers():
        buffers.append((name, buffer.numel(), buffer.shape))
    
    if buffers:
        logger.info(f"\nBuffers（非学習パラメータ）: {len(buffers)}個")
        for name, numel, shape in buffers[:10]:  # 最初の10個
            logger.info(f"  {name}: shape={shape}, params={numel:,}")
    
    return total_all, trainable_all

def compare_with_minimal_train():
    """minimal_train.pyの値と比較"""
    logger.info("\n" + "="*80)
    logger.info("【minimal_train.pyとの比較】")
    logger.info("="*80)
    
    # LoRA適用後の実際の値
    total, trainable = analyze_all_parameters()
    
    # minimal_train.pyで表示される値
    minimal_train_total = 3_988_463_539
    minimal_train_trainable = 229_619_234
    
    logger.info("\nminimal_train.py表示値:")
    logger.info(f"  総パラメータ数: {minimal_train_total:,}")
    logger.info(f"  学習可能パラメータ数: {minimal_train_trainable:,}")
    
    logger.info("\n実際の値（LoRA適用後）:")
    logger.info(f"  総パラメータ数: {total:,}")
    logger.info(f"  学習可能パラメータ数: {trainable:,}")
    
    logger.info("\n差異:")
    logger.info(f"  総パラメータ数の差: {abs(minimal_train_total - total):,}")
    logger.info(f"  学習可能パラメータ数の差: {abs(minimal_train_trainable - trainable):,}")
    
    if abs(minimal_train_total - total) < 1000 and abs(minimal_train_trainable - trainable) < 1000:
        logger.info("\n✅ パラメータカウントは一致しています")
    else:
        logger.info("\n⚠️ パラメータカウントに差異があります")

def main():
    """メイン実行"""
    logger.info("🔍 完全パラメータ分析を開始します\n")
    
    compare_with_minimal_train()
    
    logger.info("\n" + "="*80)
    logger.info("分析完了")
    logger.info("="*80)

if __name__ == "__main__":
    main()