#!/usr/bin/env python3
"""
LISA_Modelを使ってQwen2.5-VLの各段階での特徴次元を確認するデバッグスクリプト
公式推奨の特徴取得方法を検証（v2）
"""

import torch
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent))

from src.config import LISAConfig
from src.models import LISA_Model
from src.utils import prepare_tokenizer_for_lisa
from transformers import AutoProcessor

def debug_vision_features_v2():
    """LISA_Modelを使って視覚特徴の次元を確認"""
    
    # LISAConfigを作成
    config = LISAConfig(
        qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        sam_model_name="./checkpoints/sam2.1_hiera_large.pt",
        device_map="cuda",
        torch_dtype="auto",
        freeze_qwen=True,
        freeze_sam=True,
        use_dynamic_resolution=True
    )
    
    # トークナイザーとプロセッサの準備
    tokenizer = prepare_tokenizer_for_lisa(
        model_name=config.qwen_model_name,
        seg_token=config.seg_token
    )
    
    processor = AutoProcessor.from_pretrained(
        config.qwen_model_name,
        min_pixels=config.qwen_min_pixels,
        max_pixels=config.qwen_max_pixels
    )
    
    # モデルの読み込み
    print("LISA改モデルの読み込み...")
    model = LISA_Model(config)
    model.set_tokenizer(tokenizer)
    model = model.to("cuda")
    model.eval()
    
    # テスト画像準備（448×448のダミー画像）
    img = Image.fromarray(np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8))
    
    # プロセス
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is <SEG>?"}
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
    
    # 1. visual_module直接出力
    print("\n1. visual_module直接出力:")
    with torch.no_grad():
        visual_module = model.qwen.model.visual
        
        # デバッグ用: visual_moduleがある場合
        if hasattr(model.qwen.model, 'visual'):
            vision_output = visual_module(
                inputs.pixel_values.flatten(0, 1),  # (B*N, D)にフラット化
                inputs.image_grid_thw
            )
            print(f"  visual_module出力type: {type(vision_output)}")
            if torch.is_tensor(vision_output):
                print(f"  visual_module出力shape: {vision_output.shape}")
                print(f"  次元: {vision_output.shape[-1]}")
    
    # 2. get_image_featuresの出力
    print("\n2. get_image_features（現在使用中）:")
    with torch.no_grad():
        image_features = model.qwen.model.get_image_features(
            inputs.pixel_values,
            inputs.image_grid_thw
        )
    
    print(f"  get_image_features出力type: {type(image_features)}")
    if isinstance(image_features, tuple):
        print(f"  tupleの長さ: {len(image_features)}")
        for i, feat in enumerate(image_features):
            if torch.is_tensor(feat):
                print(f"    要素{i}: shape={feat.shape}, dtype={feat.dtype}")
        # 最初の要素を使用
        if len(image_features) > 0 and torch.is_tensor(image_features[0]):
            image_features = image_features[0]
            print(f"  使用する特徴shape: {image_features.shape}")
            print(f"  次元: {image_features.shape[-1]}")
    elif torch.is_tensor(image_features):
        print(f"  get_image_features出力: {image_features.shape}")
        print(f"  次元: {image_features.shape[-1]}")
    
    # 3. vision_modelを使った特徴取得
    print("\n3. vision_model経由の特徴取得:")
    
    # output_hidden_statesを有効化
    original_output_hidden_states = model.qwen.config.output_hidden_states
    model.qwen.config.output_hidden_states = True
    
    with torch.no_grad():
        # 全体のforward実行
        outputs = model.qwen.model(
            input_ids=inputs.input_ids,
            pixel_values=inputs.pixel_values,
            image_grid_thw=inputs.image_grid_thw,
            attention_mask=torch.ones_like(inputs.input_ids),
            output_hidden_states=True
        )
    
    # 元に戻す
    model.qwen.config.output_hidden_states = original_output_hidden_states
    
    # hidden_statesを確認
    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
        print(f"  LLM hidden_states数: {len(outputs.hidden_states)}")
        for i in range(min(3, len(outputs.hidden_states))):
            print(f"    Layer {i}: {outputs.hidden_states[i].shape}")
    
    # 4. Hookを使った詳細な調査
    print("\n4. Hookを使った各モジュールの出力確認:")
    
    captured_outputs = {}
    
    def capture_output(name):
        def hook_fn(module, input, output):
            if torch.is_tensor(output):
                captured_outputs[name] = output.shape
            elif hasattr(output, 'last_hidden_state'):
                captured_outputs[name] = output.last_hidden_state.shape
            elif hasattr(output, 'hidden_states'):
                captured_outputs[name] = f"hidden_states: {len(output.hidden_states)}"
        return hook_fn
    
    # 各モジュールにhookを登録
    hooks = []
    
    # visual関連のモジュールを探す
    if hasattr(model.qwen.model, 'visual'):
        hooks.append(model.qwen.model.visual.register_forward_hook(capture_output('visual')))
        
        # mergerがある場合
        if hasattr(model.qwen.model.visual, 'merger'):
            hooks.append(model.qwen.model.visual.merger.register_forward_hook(capture_output('merger')))
    
    # extract_vision_featuresを実行
    with torch.no_grad():
        vision_features = model.extract_vision_features(
            inputs.pixel_values,
            inputs.image_grid_thw
        )
    
    # hookを削除
    for hook in hooks:
        hook.remove()
    
    print("  Captured outputs:")
    for name, shape in captured_outputs.items():
        print(f"    {name}: {shape}")
    
    print(f"\n  extract_vision_features最終出力: {vision_features.shape}")
    
    # 5. 結論
    print("\n" + "=" * 80)
    print("結論:")
    print("  現在のextract_vision_features:")
    print(f"    - 出力shape: {vision_features.shape}")
    print(f"    - 次元: {vision_features.shape[-1]}")
    
    print("\n  O3推奨（LISA/InternVL-HD準拠）:")
    print("    - PatchMerge後の特徴を使用")
    print("    - 次元は2352（Qwen2.5-VL-3Bの場合）")
    print("    - visual(...).hidden_states[-1]で取得")
    
    # 実際に2352次元の特徴を取得できるか試す
    print("\n6. 2352次元特徴の取得テスト:")
    
    # visualモジュールのconfig確認
    if hasattr(model.qwen.model, 'visual'):
        visual_config = model.qwen.model.visual.config if hasattr(model.qwen.model.visual, 'config') else None
        if visual_config:
            print(f"  visual config hidden_size: {getattr(visual_config, 'hidden_size', 'N/A')}")
            print(f"  visual config intermediate_size: {getattr(visual_config, 'intermediate_size', 'N/A')}")
    
    # mergerモジュールを直接呼び出してみる
    if hasattr(model.qwen.model, 'visual') and hasattr(model.qwen.model.visual, 'merger'):
        print("\n  merger直接呼び出しテスト:")
        
        # まずvisualで特徴を取得
        with torch.no_grad():
            # pixel_valuesをフラット化
            flattened_pv = inputs.pixel_values.flatten(0, 1)  # (B*N, D)
            
            # mergerに渡す前の処理が必要かチェック
            print(f"    入力shape: {flattened_pv.shape}")
            
            # mergerを直接呼び出すのは困難なので、extract_vision_featuresで
            # hidden_statesを使う方法を検討


if __name__ == "__main__":
    debug_vision_features_v2()