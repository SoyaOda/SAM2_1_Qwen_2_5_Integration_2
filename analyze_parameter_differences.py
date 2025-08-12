#!/usr/bin/env python3
"""
test_a2, test_a3, test_a4の学習可能パラメータの詳細比較分析
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

def get_detailed_parameter_stats(model, test_name):
    """モデルの詳細パラメータ統計を取得"""
    categories = defaultdict(lambda: {'total': 0, 'trainable': 0, 'frozen': 0, 'params': []})
    
    for name, param in model.named_parameters():
        numel = param.numel()
        is_trainable = param.requires_grad
        
        # 詳細なカテゴリ分類
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
                if is_trainable:
                    category = 'SAM MaskDecoder (trainable)'
                else:
                    category = 'SAM MaskDecoder (frozen)'
            elif 'prompt_encoder' in name:
                category = 'SAM PromptEncoder'
            elif 'image_encoder' in name:
                category = 'SAM ImageEncoder'
            elif 'memory_encoder' in name or 'memory_attention' in name:
                category = 'SAM Memory/Video'
            else:
                category = 'SAM Other'
        elif 'image_adapter' in name:
            category = 'Image Adapter'
        elif 'text_prompt_proj' in name or 'prompt_proj' in name:
            category = 'Text Prompt Projector'
        elif 'prompt_beta' in name or 'beta' in name:
            category = 'Prompt Beta'
        elif 'fpn' in name or 'token_fpn' in name:
            category = 'Token-FPN'
        else:
            category = 'Other'
        
        categories[category]['total'] += numel
        if is_trainable:
            categories[category]['trainable'] += numel
        else:
            categories[category]['frozen'] += numel
        
        categories[category]['params'].append({
            'name': name,
            'numel': numel,
            'trainable': is_trainable,
            'shape': list(param.shape),
            'dtype': str(param.dtype)
        })
    
    # 統計計算
    total_all = sum(cat['total'] for cat in categories.values())
    trainable_all = sum(cat['trainable'] for cat in categories.values())
    frozen_all = total_all - trainable_all
    
    return {
        'test_name': test_name,
        'total_params': total_all,
        'trainable_params': trainable_all,
        'frozen_params': frozen_all,
        'trainable_ratio': (trainable_all / total_all * 100) if total_all > 0 else 0,
        'categories': dict(categories)
    }

def analyze_test_a2():
    """Test A2: Unfreeze SAM の分析"""
    print("=== Test A2: Unfreeze SAM ===")
    
    config = LISAConfig(
        qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        sam_model_name="./checkpoints/sam2.1_hiera_large.pt",
        device_map="cpu",  # CPUで軽量実行
        torch_dtype="auto",
        use_flash_attention=False,
        # Training configuration - Test A2: Unfreeze SAM
        train_qwen_lora=True,         # Qwen LoRA有効
        train_seg_token=True,         # SEGトークン学習可能
        train_sam_lora=True,          # SAM MaskDecoder LoRA有効（A2の特徴）
        train_image_adapter=True,     # アダプター学習可能
        train_text_prompt_projector=True,  # プロジェクター学習可能
        train_token_fpn=True,         # Token-FPN学習可能
        train_prompt_beta=False,      # A2ではBetaなし
        # Freeze settings
        freeze_sam_mask_decoder=False # A2: SAM MaskDecoder学習可能
    )
    
    tokenizer = prepare_tokenizer_for_lisa(config.qwen_model_name, config.seg_token)
    model = LISA_Model(config)
    model.set_tokenizer(tokenizer)
    
    return get_detailed_parameter_stats(model, "A2_Unfreeze_SAM")

def analyze_test_a3():
    """Test A3: Gated Embedding の分析"""
    print("=== Test A3: Gated Embedding ===")
    
    config = LISAConfig(
        qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        sam_model_name="./checkpoints/sam2.1_hiera_large.pt",
        device_map="cpu",
        torch_dtype="auto",
        use_flash_attention=False,
        # Training configuration - Test A3: Gated Embedding
        train_qwen_lora=True,         # Qwen LoRA有効
        train_seg_token=True,         # SEGトークン学習可能
        train_sam_lora=True,          # SAM LoRA有効
        train_image_adapter=True,     # アダプター学習可能
        train_text_prompt_projector=True,  # プロジェクター学習可能
        train_token_fpn=True,         # Token-FPN学習可能
        train_prompt_beta=True        # A3: Gated embedding用Beta学習可能
    )
    
    tokenizer = prepare_tokenizer_for_lisa(config.qwen_model_name, config.seg_token)
    
    # A3用のカスタムクラス（Betaパラメータ追加）
    class LISA_Model_Gated(LISA_Model):
        def __init__(self, config):
            super().__init__(config)
            # A3特有のprompt_betaパラメータ
            self.prompt_beta = torch.nn.Parameter(torch.tensor(0.1))
    
    model = LISA_Model_Gated(config)
    model.set_tokenizer(tokenizer)
    
    return get_detailed_parameter_stats(model, "A3_Gated_Embedding")

def analyze_test_a4():
    """Test A4: Addition Embedding の分析"""
    print("=== Test A4: Addition Embedding ===")
    
    config = LISAConfig(
        qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        sam_model_name="./checkpoints/sam2.1_hiera_large.pt",
        device_map="cpu",
        torch_dtype="auto",
        use_flash_attention=False,
        # Training configuration - Test A4: Addition Embedding
        train_qwen_lora=True,         # Qwen LoRA有効
        train_seg_token=True,         # SEGトークン学習可能
        train_sam_lora=True,          # SAM LoRA有効
        train_image_adapter=True,     # アダプター学習可能
        train_text_prompt_projector=True,  # プロジェクター学習可能
        train_token_fpn=True,         # Token-FPN学習可能
        train_prompt_beta=True        # A4: Addition embedding用Beta学習可能
    )
    
    tokenizer = prepare_tokenizer_for_lisa(config.qwen_model_name, config.seg_token)
    
    # A4用のカスタムクラス（Betaパラメータ追加）
    class LISA_Model_Addition(LISA_Model):
        def __init__(self, config):
            super().__init__(config)
            # A4特有のbetaパラメータ
            self.beta = torch.nn.Parameter(torch.tensor(0.01))
    
    model = LISA_Model_Addition(config)
    model.set_tokenizer(tokenizer)
    
    return get_detailed_parameter_stats(model, "A4_Addition_Embedding")

def compare_results(results):
    """結果の詳細比較"""
    print("\n" + "="*100)
    print("詳細パラメータ比較分析")
    print("="*100)
    
    # 基本統計の比較
    print("\n【基本統計比較】")
    print(f"{'テスト':<20} {'総パラメータ':<15} {'学習可能':<15} {'凍結':<15} {'学習可能率':<10}")
    print("-" * 80)
    for result in results:
        print(f"{result['test_name']:<20} "
              f"{result['total_params']:,>14} "
              f"{result['trainable_params']:,>14} "
              f"{result['frozen_params']:,>14} "
              f"{result['trainable_ratio']:>9.2f}%")
    
    # カテゴリ別詳細比較
    all_categories = set()
    for result in results:
        all_categories.update(result['categories'].keys())
    
    print(f"\n【カテゴリ別学習可能パラメータ数比較】")
    print(f"{'カテゴリ':<30} {'A2 (Unfreeze)':<15} {'A3 (Gated)':<15} {'A4 (Addition)':<15}")
    print("-" * 80)
    
    for category in sorted(all_categories):
        row = f"{category:<30}"
        for result in results:
            if category in result['categories']:
                trainable = result['categories'][category]['trainable']
                if trainable > 0:
                    row += f"{trainable:,>14} "
                else:
                    row += f"{'0':>14} "
            else:
                row += f"{'N/A':>14} "
        print(row)
    
    # 差異の詳細分析
    print(f"\n【主要な違いの分析】")
    
    # A2 vs A3の違い
    a2_trainable = results[0]['trainable_params']
    a3_trainable = results[1]['trainable_params']
    a4_trainable = results[2]['trainable_params']
    
    print(f"\nA2 vs A3の違い:")
    print(f"  - A3 - A2 = {a3_trainable - a2_trainable:,} パラメータ")
    
    print(f"\nA2 vs A4の違い:")
    print(f"  - A4 - A2 = {a4_trainable - a2_trainable:,} パラメータ")
    
    print(f"\nA3 vs A4の違い:")
    print(f"  - A4 - A3 = {a4_trainable - a3_trainable:,} パラメータ")
    
    # 設定の違いを詳細分析
    print(f"\n【設定の違い】")
    
    # A2の特徴
    print(f"\nA2 (Unfreeze SAM)の特徴:")
    a2_categories = results[0]['categories']
    for category, info in a2_categories.items():
        if info['trainable'] > 0:
            print(f"  - {category}: {info['trainable']:,} パラメータ")
    
    # A3とA4で追加されたもの
    print(f"\nA3/A4で追加されたBetaパラメータ:")
    for i, result in enumerate(results[1:], 2):  # A3, A4
        test_name = f"A{i+1}"
        categories = result['categories']
        if 'Prompt Beta' in categories:
            beta_info = categories['Prompt Beta']
            print(f"  - {test_name}: {beta_info['trainable']} パラメータ")
    
    # SAM MaskDecoderの状態比較
    print(f"\nSAM MaskDecoderの学習状態:")
    for result in results:
        test_name = result['test_name']
        categories = result['categories']
        
        sam_trainable = 0
        sam_frozen = 0
        
        for category, info in categories.items():
            if 'SAM MaskDecoder' in category:
                if 'trainable' in category:
                    sam_trainable += info['trainable']
                else:
                    sam_frozen += info['frozen']
        
        print(f"  - {test_name}: 学習可能={sam_trainable:,}, 凍結={sam_frozen:,}")

def main():
    """メイン分析実行"""
    print("パラメータ差異分析を開始...")
    
    # 各テストの分析実行
    try:
        print("\n" + "="*50)
        results = []
        
        print("A2分析中...")
        results.append(analyze_test_a2())
        
        print("A3分析中...")
        results.append(analyze_test_a3())
        
        print("A4分析中...")
        results.append(analyze_test_a4())
        
        # 結果比較
        compare_results(results)
        
        print(f"\n✅ 分析完了！")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()