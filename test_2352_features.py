#!/usr/bin/env python3
"""
2352次元PatchMerge特徴の動作確認テスト
"""
import torch
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.lisa_model import LISA_Model
from src.config import LISAConfig
from transformers import AutoProcessor, AutoTokenizer
from PIL import Image
import numpy as np

def test_2352_features():
    """2352次元特徴取得のテスト"""
    
    print("=" * 80)
    print("2352次元 PatchMerge特徴 動作確認テスト")
    print("=" * 80)
    
    # 1. 設定を準備
    print("\n1. 設定準備...")
    config = LISAConfig()
    config.use_patchmerge_features = True  # PatchMerge特徴を使用
    config.vision_feature_dim = 2352  # 2352次元に設定
    
    print(f"  use_patchmerge_features: {config.use_patchmerge_features}")
    print(f"  vision_feature_dim: {config.vision_feature_dim}")
    
    # 2. プロセッサとトークナイザーの準備
    print("\n2. プロセッサ準備...")
    processor = AutoProcessor.from_pretrained(
        config.qwen_model_name,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(config.qwen_model_name)
    
    # SEGトークン追加
    seg_token = "<SEG>"
    if seg_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [seg_token]})
    
    # 3. テスト画像の準備
    print("\n3. テスト画像準備...")
    test_image = Image.new('RGB', (448, 448), color='red')
    
    # 4. 画像処理
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Segment the red area."}
            ]
        }
    ]
    
    # プロセッサで画像とテキストを処理
    text = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    inputs = processor(
        text=text,
        images=test_image,
        return_tensors="pt"
    )
    
    # デバイスに移動
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    
    # pixel_valuesが含まれているか確認
    if 'pixel_values' in inputs:
        print(f"  pixel_values shape: {inputs['pixel_values'].shape}")
    else:
        print(f"  Available keys: {inputs.keys()}")
        # pixel_valuesを手動で追加する必要がある場合
        if 'inputs' in inputs:
            # 新しいAPIの場合
            inputs = inputs['inputs']
    
    if 'image_grid_thw' in inputs:
        print(f"  image_grid_thw: {inputs['image_grid_thw']}")
    
    # 5. LISA改モデルの部分的な初期化（extract_vision_featuresのテスト）
    print("\n4. LISA改モデル初期化（簡易版）...")
    
    # Qwenモデルだけ読み込み
    from transformers import Qwen2_5_VLForConditionalGeneration
    
    print("  Qwenモデル読み込み中...")
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.qwen_model_name,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    
    # LISA改モデル初期化（SAMなし）
    model = LISA_Model(
        config=config,
        qwen_model=qwen_model,
        sam_predictor=None,  # SAMは今回不要
        tokenizer=tokenizer
    )
    model.eval()
    model = model.to(device)
    
    # 6. extract_vision_featuresのテスト
    print("\n5. extract_vision_features テスト...")
    
    with torch.no_grad():
        vision_features = model.extract_vision_features(
            inputs['pixel_values'],
            inputs['image_grid_thw']
        )
    
    print(f"\n✓ Vision features shape: {vision_features.shape}")
    print(f"  Batch size: {vision_features.shape[0]}")
    print(f"  Number of patches: {vision_features.shape[1]}")
    print(f"  Feature dimension: {vision_features.shape[2]}")
    
    # 期待される次元かチェック
    if vision_features.shape[2] == 2352:
        print("\n🎉 成功: 2352次元のPatchMerge特徴を取得できました！")
    elif vision_features.shape[2] == 1176:
        print("\n⚠️ 注意: 1176次元の特徴を取得（2352の半分）")
        print("  一部の設定では1176次元が返される場合があります")
    elif vision_features.shape[2] == 2048:
        print("\n❌ エラー: 2048次元のLLM投影後特徴が返されました")
        print("  hidden_statesが正しく取得できていない可能性があります")
    else:
        print(f"\n❓ 予期しない次元: {vision_features.shape[2]}次元")
    
    # 7. ImageFeatureAdapterのテスト
    print("\n6. ImageFeatureAdapter テスト...")
    
    # アダプターを通す
    sam_features = model.image_adapter(vision_features, inputs['image_grid_thw'])
    
    print(f"  SAM features shape: {sam_features.shape}")
    print(f"  Expected: [B, 256, H, W]")
    
    if sam_features.shape[1] == 256:
        print("  ✓ 正しく256次元に変換されました")
    
    # 8. メモリ使用量の確認
    if torch.cuda.is_available():
        print("\n7. GPU メモリ使用量:")
        print(f"  Allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
    
    print("\n" + "=" * 80)
    print("テスト完了")
    print("=" * 80)
    
    return vision_features


if __name__ == "__main__":
    features = test_2352_features()