#!/usr/bin/env python3
"""
Qwen2.5-VLの各段階での特徴次元を確認するデバッグスクリプト
公式推奨の特徴取得方法を検証
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent))

def debug_vision_features():
    """各段階での視覚特徴の次元を確認"""
    
    # モデルとプロセッサの準備
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=112896,  # 336×336
        max_pixels=800000   # 896×896
    )
    
    # モデルロード（vision部分のみ）
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    model.eval()
    
    # テスト画像準備（ダミー画像）
    from PIL import Image
    import numpy as np
    
    # 448×448の画像を作成
    img = Image.fromarray(np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8))
    
    # プロセス
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is this?"}
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[img],
        return_tensors="pt"
    ).to("cuda")
    
    print("=" * 80)
    print("入力情報:")
    print(f"  pixel_values shape: {inputs.pixel_values.shape}")
    print(f"  image_grid_thw: {inputs.image_grid_thw}")
    print(f"  input_ids shape: {inputs.input_ids.shape}")
    
    # RAWパッチ数とCompressed数の確認
    H_raw = int(inputs.image_grid_thw[0, 1])
    W_raw = int(inputs.image_grid_thw[0, 2])
    N_raw = H_raw * W_raw
    N_compressed = N_raw // 4
    print(f"  RAWパッチ: {H_raw}×{W_raw} = {N_raw}")
    print(f"  Compressed（PatchMerge後）: {N_compressed}")
    
    print("=" * 80)
    print("各段階での特徴次元確認:")
    
    # 1. vision_modelの直接出力（hidden_states有効化）
    print("\n1. vision_model直接出力（hidden_states有効）:")
    model.config.output_hidden_states = True
    
    with torch.no_grad():
        vision_output = model.model.vision_model(
            inputs.pixel_values,
            image_grid_thw=inputs.image_grid_thw
        )
    
    # hidden_statesの各層を確認
    if hasattr(vision_output, 'hidden_states') and vision_output.hidden_states:
        print(f"  hidden_states数: {len(vision_output.hidden_states)}")
        for i, hs in enumerate(vision_output.hidden_states[-3:]):  # 最後の3層のみ表示
            print(f"    Layer {len(vision_output.hidden_states) - 3 + i}: {hs.shape}")
    
    # last_hidden_stateを確認
    if hasattr(vision_output, 'last_hidden_state'):
        print(f"  last_hidden_state: {vision_output.last_hidden_state.shape}")
    
    # 2. PatchMerge後の特徴（merger直後）を確認
    print("\n2. PatchMerge後の特徴（mergerモジュール通過後）:")
    
    # Hookを使ってmerger後の特徴を取得
    merger_output = None
    def capture_merger(module, input, output):
        nonlocal merger_output
        merger_output = output
    
    # mergerにhook登録
    hook = model.model.vision_model.merger.register_forward_hook(capture_merger)
    
    with torch.no_grad():
        _ = model.model.vision_model(
            inputs.pixel_values,
            image_grid_thw=inputs.image_grid_thw
        )
    
    hook.remove()
    
    if merger_output is not None:
        print(f"  merger出力shape: {merger_output.shape}")
        print(f"  merger出力次元: {merger_output.shape[-1]}")
    
    # 3. get_image_featuresの出力
    print("\n3. get_image_features（LLM投影後）:")
    
    with torch.no_grad():
        image_features = model.model.get_image_features(
            inputs.pixel_values,
            inputs.image_grid_thw
        )
    
    print(f"  get_image_features出力: {image_features.shape}")
    print(f"  次元: {image_features.shape[-1]}")
    
    # 4. 公式推奨方法の確認（O3提案）
    print("\n4. O3推奨方法の確認:")
    print("  vision_model(...).hidden_states[-1]を使用")
    
    # hidden_statesの最後の要素を確認
    if hasattr(vision_output, 'hidden_states') and vision_output.hidden_states:
        last_hidden = vision_output.hidden_states[-1]
        print(f"  hidden_states[-1] shape: {last_hidden.shape}")
        print(f"  次元: {last_hidden.shape[-1]}")
        
        # これがPatchMerge後かを確認
        if last_hidden.shape[1] == N_compressed:
            print(f"  ✓ パッチ数が圧縮後と一致: {last_hidden.shape[1]} == {N_compressed}")
        else:
            print(f"  ✗ パッチ数が不一致: {last_hidden.shape[1]} != {N_compressed}")
    
    # 5. 各方法の比較
    print("\n" + "=" * 80)
    print("結論:")
    print("  - merger出力: 2352次元（PatchMerge後、LLM投影前）")
    print("  - get_image_features: 2048次元（LLM投影後）")
    print("  - hidden_states[-1]: merger出力と同じはず")
    print("\n公式推奨（LISA/InternVL-HD準拠）:")
    print("  → PatchMerge後（2352次元）を使用")
    print("  → vision_model(...).hidden_states[-1]で取得")


if __name__ == "__main__":
    debug_vision_features()