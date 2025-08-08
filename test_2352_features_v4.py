#!/usr/bin/env python3
"""
2352次元PatchMerge特徴の確実な取得テスト（v4）
Qwen2_5_VLForConditionalGenerationを直接使用
"""
import torch
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image
import numpy as np

def test_2352_features_v4():
    """2352次元特徴を確実に取得（v4）"""
    
    print("=" * 80)
    print("2352次元 PatchMerge特徴 取得テスト v4")
    print("=" * 80)
    
    # 1. モデルとプロセッサの準備
    print("\n1. モデル準備...")
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    # プロセッサとモデルの読み込み
    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype="auto", 
        device_map="auto"
    )
    
    print(f"  Model type: {type(model)}")
    
    # 2. vision_towerまたはvisualモジュールを探す
    vision_module = None
    
    # 方法1: vision_towerを探す
    if hasattr(model, 'vision_tower'):
        print(f"  ✓ model.vision_tower が存在")
        vision_module = model.vision_tower
    # 方法2: model.model.visualを探す
    elif hasattr(model, 'model') and hasattr(model.model, 'visual'):
        print(f"  ✓ model.model.visual が存在")
        vision_module = model.model.visual
    else:
        print("  Vision module not found, checking model structure...")
        # モデル構造を詳しく調査
        if hasattr(model, 'model'):
            print(f"  model.model attributes: {[attr for attr in dir(model.model) if not attr.startswith('_')][:10]}")
    
    if vision_module is None:
        raise AttributeError("Vision module not found")
    
    print(f"  Vision module type: {type(vision_module)}")
    
    # 3. テスト画像の準備
    print("\n2. テスト画像準備...")
    test_image = Image.new('RGB', (448, 448), color='red')
    
    # 4. 前処理
    print("\n3. 前処理...")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Segment the red area."}
            ]
        }
    ]
    
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = proc(
        text=text,
        images=test_image,
        return_tensors="pt"
    )
    
    pixel_values = inputs["pixel_values"]      
    image_grid_thw = inputs["image_grid_thw"]  
    
    print(f"  pixel_values shape: {pixel_values.shape}")
    print(f"  image_grid_thw: {image_grid_thw}")
    
    # デバイスに移動
    device = next(model.parameters()).device
    pixel_values = pixel_values.to(device)
    image_grid_thw = image_grid_thw.to(device)
    
    # 5. フックを使ってmergerの出力を取得
    print("\n4. フックを使ったPatchMerge特徴の取得...")
    
    captured_features = {}
    
    def capture_hook(name):
        def hook_fn(module, input, output):
            if torch.is_tensor(output):
                captured_features[name] = output.detach()
                print(f"  Captured {name}: shape {output.shape}")
            elif isinstance(output, tuple) and len(output) > 0:
                if torch.is_tensor(output[0]):
                    captured_features[name] = output[0].detach()
                    print(f"  Captured {name}: shape {output[0].shape}")
        return hook_fn
    
    # patch_embedとblocksとmergerにフックを設定
    hooks = []
    
    if hasattr(vision_module, 'patch_embed'):
        hook = vision_module.patch_embed.register_forward_hook(capture_hook('patch_embed'))
        hooks.append(hook)
        print("  ✓ patch_embedにフック設定")
    
    if hasattr(vision_module, 'blocks'):
        # 最初と最後のblockにフック
        if len(vision_module.blocks) > 0:
            hook = vision_module.blocks[0].register_forward_hook(capture_hook('first_block'))
            hooks.append(hook)
            hook = vision_module.blocks[-1].register_forward_hook(capture_hook('last_block'))
            hooks.append(hook)
            print(f"  ✓ blocks（{len(vision_module.blocks)}層）にフック設定")
    
    if hasattr(vision_module, 'merger'):
        hook = vision_module.merger.register_forward_hook(capture_hook('merger'))
        hooks.append(hook)
        print("  ✓ mergerにフック設定")
    
    # 6. get_image_featuresを実行してフックで特徴を取得
    print("\n5. Forward pass実行...")
    
    with torch.no_grad():
        # get_image_featuresを呼び出し
        llm_features = model.get_image_features(pixel_values, image_grid_thw)
        
        if isinstance(llm_features, tuple):
            llm_features = llm_features[0]
        
        print(f"\n  get_image_features出力: {llm_features.shape}")
    
    # フック削除
    for hook in hooks:
        hook.remove()
    
    # 7. キャプチャした特徴の分析
    print("\n6. キャプチャした特徴の分析:")
    
    patchmerge_feat = None
    
    for name, feat in captured_features.items():
        print(f"\n  {name}:")
        print(f"    Shape: {feat.shape}")
        print(f"    次元: {feat.shape[-1] if feat.dim() >= 2 else 'N/A'}")
        
        # 2352次元または1176次元を探す
        if feat.dim() >= 3:
            if feat.shape[-1] == 2352:
                print(f"    ✓✓✓ 2352次元の特徴を発見！")
                patchmerge_feat = feat
            elif feat.shape[-1] == 1176:
                print(f"    → 1176次元 (2352の半分)")
                # 1176次元の場合、2つ結合して2352次元にする可能性を検討
            elif feat.shape[-1] == 1280:
                print(f"    → 1280次元 (Qwen2.5-VLのhidden_size)")
    
    # 8. blocksの最後の出力を2倍にして2352次元を作る試み
    if patchmerge_feat is None and 'last_block' in captured_features:
        last_block_feat = captured_features['last_block']
        if isinstance(last_block_feat, tuple):
            last_block_feat = last_block_feat[0]
        
        print("\n7. last_blockの特徴からPatchMerge特徴を再構成...")
        print(f"  last_block shape: {last_block_feat.shape}")
        
        # mergerが2×2のマージを行う場合、次元が2倍になる可能性
        if last_block_feat.shape[-1] == 1176 or last_block_feat.shape[-1] == 1280:
            print(f"  → {last_block_feat.shape[-1]}次元を2倍にして擬似的な2352/2560次元を作成可能")
    
    # 9. 結果のまとめ
    print("\n" + "=" * 80)
    print("結果:")
    
    if patchmerge_feat is not None:
        print(f"  ✓ 2352次元PatchMerge特徴を取得成功!")
        print(f"  Shape: {patchmerge_feat.shape}")
        return patchmerge_feat
    else:
        print("  ✗ 2352次元特徴が見つかりませんでした")
        print("\n  取得できた特徴:")
        for name, feat in captured_features.items():
            print(f"    - {name}: {feat.shape}")
        
        print("\n  推測:")
        print("  Qwen2.5-VL-3Bでは:")
        print("    - hidden_size = 1280 (旧2.0では1176)")
        print("    - PatchMerge後 = 1280×2 = 2560次元")
        print("    - LLM投影後 = 2048次元")
        print("\n  → 2352次元ではなく2560次元の可能性があります")
    
    print("=" * 80)
    
    return None


if __name__ == "__main__":
    features = test_2352_features_v4()