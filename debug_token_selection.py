#!/usr/bin/env python3
"""
トークン選択設定のデバッグスクリプト
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
import logging

# ロギング設定
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def check_all_configs(model):
    """すべての設定場所をチェック"""
    print("\n" + "=" * 80)
    print("設定チェック:")
    
    configs_to_check = [
        ("model.qwen.config", model.qwen.config if hasattr(model.qwen, 'config') else None),
        ("model.qwen.model.config", model.qwen.model.config if hasattr(model.qwen.model, 'config') else None),
        ("model.qwen.model.visual.config", model.qwen.model.visual.config if hasattr(model.qwen.model, 'visual') and hasattr(model.qwen.model.visual, 'config') else None),
    ]
    
    for name, config in configs_to_check:
        if config is not None:
            if hasattr(config, 'image_feature_select_strategy'):
                print(f"  {name}.image_feature_select_strategy = {config.image_feature_select_strategy}")
            else:
                print(f"  {name}: image_feature_select_strategy属性なし")
        else:
            print(f"  {name}: 存在しない")
    
    print("=" * 80)

def test_token_selection_debug():
    """トークン選択のデバッグ"""
    
    print("=" * 80)
    print("トークン選択デバッグ")
    print("=" * 80)
    
    # 1. モデル準備
    print("\n1. モデル初期化...")
    config = LISAConfig()
    
    # Qwenモデルを先に読み込み
    print("  Qwenモデルロード中...")
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.qwen_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    
    # 初期状態の設定を確認
    print("\n初期状態（LISA_Model初期化前）:")
    if hasattr(qwen_model.config, 'image_feature_select_strategy'):
        print(f"  qwen_model.config.image_feature_select_strategy = {qwen_model.config.image_feature_select_strategy}")
    
    # LISA_Model初期化
    print("\n  LISA_Model初期化中...")
    model = LISA_Model(
        config=config,
        qwen_model=qwen_model,
        sam_predictor=None
    )
    model.eval()
    
    # 初期化後の設定を確認
    print("\nLISA_Model初期化後:")
    check_all_configs(model)
    
    # 2. プロセッサ準備
    print("\n2. プロセッサ準備...")
    processor = AutoProcessor.from_pretrained(
        config.qwen_model_name,
        trust_remote_code=True,
        min_pixels=config.qwen_min_pixels,
        max_pixels=config.qwen_max_pixels
    )
    
    # 3. 実際のトークン数をテスト
    print("\n3. 実際のトークン数テスト...")
    
    # 900パッチ（225トークン期待）のケース
    test_image = Image.new('RGB', (672, 672), color='red')  # 48×48パッチ = 2304 → 576トークン期待
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Test"}
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
    
    print(f"\n  入力画像: 672×672")
    print(f"  pixel_values shape: {inputs['pixel_values'].shape}")
    
    if 'image_grid_thw' in inputs and inputs['image_grid_thw'] is not None:
        grid = inputs['image_grid_thw'][0]
        H_raw, W_raw = int(grid[1].item()), int(grid[2].item())
        raw_patches = H_raw * W_raw
        expected_tokens = raw_patches // 4
        print(f"  RAWグリッド: {H_raw}×{W_raw} = {raw_patches} patches")
        print(f"  期待トークン数（PatchMerge後）: {expected_tokens}")
    
    # get_image_featuresを直接呼び出し
    print("\n4. get_image_features直接呼び出し...")
    with torch.no_grad():
        # モデルの設定を再確認
        if hasattr(model.qwen.config, 'image_feature_select_strategy'):
            print(f"  実行時のmodel.qwen.config.image_feature_select_strategy = {model.qwen.config.image_feature_select_strategy}")
        
        image_embeds = model.qwen.model.get_image_features(
            inputs['pixel_values'],
            inputs.get('image_grid_thw')
        )
        
        if isinstance(image_embeds, tuple):
            image_embeds = image_embeds[0]
        
        print(f"  get_image_features出力shape: {image_embeds.shape}")
        print(f"  総トークン数: {image_embeds.shape[0]}")
        print(f"  次元: {image_embeds.shape[1]}")
        
        if image_embeds.shape[0] == expected_tokens:
            print(f"  ✓ 期待通り全トークンが返されました")
        elif image_embeds.shape[0] <= 128:
            print(f"  ✗ top128制限が適用されています！")
        else:
            print(f"  ⚠ 予期しないトークン数: {image_embeds.shape[0]} (期待: {expected_tokens})")
    
    # 5. extract_vision_featuresのテスト
    print("\n5. extract_vision_featuresメソッドのテスト...")
    with torch.no_grad():
        try:
            vision_features = model.extract_vision_features(
                inputs['pixel_values'],
                inputs.get('image_grid_thw')
            )
            print(f"  vision_features shape: {vision_features.shape}")
            print(f"  ✓ 正常に処理されました")
        except Exception as e:
            print(f"  ✗ エラー: {e}")
    
    print("\n" + "=" * 80)
    print("結論:")
    if image_embeds.shape[0] == expected_tokens:
        print("  ✓ トークン選択が正常に無効化されています")
    else:
        print("  ✗ トークン選択がまだ有効です")
        print("  → モデルの再初期化または設定の適用方法の見直しが必要")
    print("=" * 80)


if __name__ == "__main__":
    test_token_selection_debug()