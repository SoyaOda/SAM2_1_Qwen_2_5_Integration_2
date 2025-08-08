#!/usr/bin/env python3
"""
トークン選択の修正をテストするスクリプト
"""
import torch
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from src.models.lisa_model import LISA_Model
from src.config import LISAConfig
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image
import numpy as np

def test_token_selection():
    """トークン選択無効化のテスト"""
    
    print("=" * 80)
    print("トークン選択修正テスト")
    print("=" * 80)
    
    # 1. モデル準備
    print("\n1. モデル初期化...")
    config = LISAConfig()
    
    # Qwenモデルを先に読み込み
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.qwen_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    
    # LISA_Model初期化
    model = LISA_Model(
        config=config,
        qwen_model=qwen_model,
        sam_predictor=None  # SAMなし
    )
    model.eval()
    
    # トークン選択戦略の確認
    if hasattr(model.qwen.model, 'visual'):
        if hasattr(model.qwen.model.visual, 'config'):
            strategy = model.qwen.model.visual.config.image_feature_select_strategy
            print(f"  Token selection strategy: {strategy}")
            if strategy == "none":
                print("  ✓ トークン選択が無効化されています（全トークン使用）")
            else:
                print(f"  ✗ トークン選択が有効: {strategy}")
    
    # 2. プロセッサ準備
    print("\n2. プロセッサ準備...")
    processor = AutoProcessor.from_pretrained(
        config.qwen_model_name,
        trust_remote_code=True,
        min_pixels=config.qwen_min_pixels,
        max_pixels=config.qwen_max_pixels
    )
    
    # 3. 異なる解像度でテスト
    print("\n3. 異なる解像度でのトークン数テスト...")
    
    test_cases = [
        (448, 448, "Small"),
        (672, 672, "Medium"),
        (896, 896, "Large"),
        (1344, 896, "Wide"),
        (896, 1344, "Tall")
    ]
    
    for width, height, name in test_cases:
        print(f"\n  テスト: {name} ({width}×{height})")
        
        # テスト画像作成
        test_image = Image.new('RGB', (width, height), color='red')
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Segment the red area."}
                ]
            }
        ]
        
        # 前処理
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = processor(
            text=text,
            images=test_image,
            return_tensors="pt"
        )
        
        # デバイスに移動
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        
        # パッチ数の計算
        if 'image_grid_thw' in inputs and inputs['image_grid_thw'] is not None:
            grid = inputs['image_grid_thw'][0]
            H_raw, W_raw = int(grid[1].item()), int(grid[2].item())
            raw_patches = H_raw * W_raw
            expected_tokens_after_merge = raw_patches // 4  # PatchMerge 4:1
            
            print(f"    RAWグリッド: {H_raw}×{W_raw} = {raw_patches} patches")
            print(f"    期待トークン数（PatchMerge後）: {expected_tokens_after_merge}")
        
        # extract_vision_featuresのテスト
        with torch.no_grad():
            try:
                vision_features = model.extract_vision_features(
                    inputs['pixel_values'],
                    inputs.get('image_grid_thw')
                )
                
                actual_tokens = vision_features.shape[1]
                print(f"    実際のトークン数: {actual_tokens}")
                
                # トークン選択の影響を確認
                if strategy == "none":
                    if actual_tokens == expected_tokens_after_merge:
                        print(f"    ✓ 全トークンが使用されています")
                    else:
                        print(f"    ⚠ トークン数が異なります（圧縮率の問題？）")
                else:
                    if actual_tokens <= 128:
                        print(f"    ⚠ top128制限が適用されています")
                    else:
                        print(f"    ✓ 128以上のトークンが使用されています")
                
                print(f"    Vision features shape: {vision_features.shape}")
                
            except Exception as e:
                print(f"    ✗ エラー発生: {e}")
    
    # 4. バッチ処理のテスト
    print("\n4. バッチ処理テスト...")
    
    # 異なる解像度の画像をバッチ処理
    images = [
        Image.new('RGB', (448, 448), color='red'),
        Image.new('RGB', (672, 672), color='green'),
        Image.new('RGB', (896, 896), color='blue')
    ]
    
    batch_messages = []
    for img in images:
        batch_messages.append([
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Segment the colored area."}
                ]
            }
        ])
    
    # バッチ前処理
    batch_inputs = []
    for i, (img, msgs) in enumerate(zip(images, batch_messages)):
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        inp = processor(
            text=text,
            images=img,
            return_tensors="pt"
        )
        batch_inputs.append(inp)
    
    print(f"  バッチサイズ: {len(batch_inputs)}")
    
    # 各サンプルを個別に処理（バッチ処理のシミュレーション）
    for i, inp in enumerate(batch_inputs):
        inp = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inp.items()}
        
        with torch.no_grad():
            try:
                vision_features = model.extract_vision_features(
                    inp['pixel_values'],
                    inp.get('image_grid_thw')
                )
                print(f"  Sample {i}: vision_features shape = {vision_features.shape}")
            except Exception as e:
                print(f"  Sample {i}: エラー = {e}")
    
    print("\n" + "=" * 80)
    print("結論:")
    if strategy == "none":
        print("  ✓ トークン選択が正常に無効化されています")
        print("  ✓ セグメンテーションタスクに適した設定です")
    else:
        print("  ⚠ トークン選択がまだ有効です")
        print("  ⚠ config.token_selection_strategy = 'none' を確認してください")
    print("=" * 80)


if __name__ == "__main__":
    test_token_selection()