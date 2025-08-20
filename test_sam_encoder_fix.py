#!/usr/bin/env python3
"""
SAM ImageEncoder修正の効果検証テスト

修正内容:
1. sam_image_encoder.eval() の強制適用
2. 学習時/推論時の適切な実行コンテキスト
3. 混合精度設定の最適化
"""

import torch
import sys
import os
sys.path.append('/home/soya/SAM2_1_Qwen_2_5_Integration_2')

from src.config import LISAConfig
from src.models.lisa_model import LISA_Model
from transformers import AutoTokenizer, AutoProcessor
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_sam_encoder_stability():
    """SAM EncoderのEval設定によるFeature安定性テスト"""
    
    print("=== SAM ImageEncoder修正効果検証テスト ===")
    
    # Config設定
    config = LISAConfig()
    config.sam_encoder_eval_mode = True  # 新設定: eval mode強制
    config.sam_encoder_precision = "fp32"  # 最高品質設定
    config.sam_encoder_use_inference_mode = True  # 推論最適化
    config.debug_fusion = True  # デバッグログ有効
    
    print(f"Config: eval_mode={config.sam_encoder_eval_mode}, precision={config.sam_encoder_precision}")
    
    # Tokenizer/Processor設定
    tokenizer = AutoTokenizer.from_pretrained(config.qwen_model_name)
    processor = AutoProcessor.from_pretrained(config.qwen_model_name)
    
    # モデル初期化
    print("モデル初期化中...")
    model = LISA_Model(config)
    model.set_tokenizer(tokenizer)
    model = model.cuda()
    
    # SAM ImageEncoderのeval状態確認
    print(f"SAM ImageEncoder training mode: {model.sam_image_encoder.training}")
    print(f"SAM encoder config: {model.sam_encoder_config}")
    
    # テスト画像作成 (1024x1024 SAM形式)
    sam_images = torch.randn(1, 3, 1024, 1024).cuda()
    
    print("\n=== 学習モード vs 推論モードでの特徴安定性テスト ===")
    
    # 学習モードでのテスト
    model.train()
    with torch.no_grad():  # 学習時でも勾配計算なし
        print("学習モードでSAM特徴抽出中...")
        if hasattr(model, 'sam_image_encoder'):
            model.sam_image_encoder.eval()  # 強制的にeval
        
        # 複数回実行して一貫性確認
        features_train = []
        for i in range(3):
            # SAM encoder部分のみテスト
            sam_encoder_dtype = next(model.sam_image_encoder.parameters()).dtype
            sam_images_converted = sam_images.to(dtype=sam_encoder_dtype)
            
            with torch.no_grad():
                backbone_out = model.sam_image_encoder(sam_images_converted)
                if isinstance(backbone_out, dict):
                    features = backbone_out.get('vision_features', backbone_out.get('image_embeddings'))
                else:
                    features = backbone_out
                features_train.append(features.clone())
        
        # 特徴の一貫性チェック
        diff_train = torch.abs(features_train[0] - features_train[1]).mean().item()
        print(f"学習モード: 特徴差分 (run1 vs run2): {diff_train:.8f}")
    
    # 推論モードでのテスト
    model.eval()
    with torch.no_grad():
        print("推論モードでSAM特徴抽出中...")
        
        features_eval = []
        for i in range(3):
            sam_encoder_dtype = next(model.sam_image_encoder.parameters()).dtype
            sam_images_converted = sam_images.to(dtype=sam_encoder_dtype)
            
            with torch.inference_mode():
                backbone_out = model.sam_image_encoder(sam_images_converted)
                if isinstance(backbone_out, dict):
                    features = backbone_out.get('vision_features', backbone_out.get('image_embeddings'))
                else:
                    features = backbone_out
                features_eval.append(features.clone())
        
        # 特徴の一貫性チェック
        diff_eval = torch.abs(features_eval[0] - features_eval[1]).mean().item()
        print(f"推論モード: 特徴差分 (run1 vs run2): {diff_eval:.8f}")
    
    # 学習モードと推論モードの差分
    mode_diff = torch.abs(features_train[0] - features_eval[0]).mean().item()
    print(f"モード間差分 (train vs eval): {mode_diff:.8f}")
    
    print("\n=== SAM Encoder専用テスト ===")
    
    # SAM ImageEncoderのみをテスト（統合テストは実際のデータで）
    print("SAM ImageEncoderの動作テスト実行中...")
    
    model.eval()
    try:
        with torch.no_grad():
            # SAM Encoderの直接テスト
            sam_encoder_dtype = next(model.sam_image_encoder.parameters()).dtype
            sam_images_converted = sam_images.to(dtype=sam_encoder_dtype)
            
            backbone_out = model.sam_image_encoder(sam_images_converted)
            if isinstance(backbone_out, dict):
                features = backbone_out.get('vision_features', backbone_out.get('image_embeddings'))
            else:
                features = backbone_out
            
            print(f"SAM Encoder成功:")
            print(f"  - 入力形状: {sam_images_converted.shape}")
            print(f"  - 出力形状: {features.shape}")
            print(f"  - dtype: {features.dtype}")
            print(f"  - 値の範囲: [{features.min().item():.4f}, {features.max().item():.4f}]")
            
    except Exception as e:
        print(f"SAM Encoder エラー: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== 結果評価 ===")
    
    # 修正効果の評価
    if diff_train < 1e-6 and diff_eval < 1e-6:
        print("✅ 特徴安定性: 優秀 (完全一致)")
    elif diff_train < 1e-4 and diff_eval < 1e-4:
        print("✅ 特徴安定性: 良好 (微小差分)")
    else:
        print("⚠️ 特徴安定性: 要確認 (差分が大きい)")
    
    if mode_diff < 1e-4:
        print("✅ モード一貫性: 優秀")
    else:
        print("⚠️ モード一貫性: 要確認")
    
    print("\n=== パラメータ凍結状況確認 ===")
    
    frozen_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
        else:
            frozen_params += param.numel()
    
    total_params = frozen_params + trainable_params
    print(f"総パラメータ: {total_params:,}")
    print(f"凍結パラメータ: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
    print(f"学習可能パラメータ: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    # SAM ImageEncoderの状態確認
    sam_encoder_params = sum(p.numel() for p in model.sam_image_encoder.parameters())
    sam_encoder_trainable = sum(p.numel() for p in model.sam_image_encoder.parameters() if p.requires_grad)
    
    print(f"\nSAM ImageEncoder:")
    print(f"  - 総パラメータ: {sam_encoder_params:,}")
    print(f"  - 学習可能: {sam_encoder_trainable:,}")
    print(f"  - 凍結率: {(sam_encoder_params-sam_encoder_trainable)/sam_encoder_params*100:.1f}%")
    print(f"  - Training mode: {model.sam_image_encoder.training}")


if __name__ == "__main__":
    test_sam_encoder_stability()