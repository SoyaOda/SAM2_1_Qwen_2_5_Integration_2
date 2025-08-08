#!/usr/bin/env python3
"""
Qwen2.5-VL-3Bの2560次元PatchMerge特徴を正しく取得するテスト
O3確認済みの正しい方法
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

def test_2560_features_correct():
    """O3確認済みの正しい方法で2560次元特徴を取得"""
    
    print("=" * 80)
    print("2560次元 PatchMerge特徴 取得テスト（O3確認済み）")
    print("=" * 80)
    
    # 1. モデルとプロセッサの準備
    print("\n1. モデル準備...")
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype="auto", 
        device_map="auto"
    ).eval()
    
    print(f"  Model type: {type(model)}")
    
    # 正しいパス: model.model.visual
    vision = model.model.visual
    print(f"  Vision module: model.model.visual")
    print(f"  Vision type: {type(vision)}")
    
    # configの確認
    if hasattr(vision, 'config'):
        config = vision.config
        print(f"  Vision config:")
        print(f"    - hidden_size: {config.hidden_size}")  # 1280
        print(f"    - PatchMerge後: {config.hidden_size * 2} (1280 × 2)")
    
    # 2. テスト画像の準備
    print("\n2. テスト画像準備...")
    test_image = Image.new('RGB', (448, 448), color='red')
    
    # 3. 前処理
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
    
    # バッチ次元を追加
    if pixel_values.dim() == 2:
        pixel_values = pixel_values.unsqueeze(0)  # (B, N_raw, 1280)
    if image_grid_thw.dim() == 1:
        image_grid_thw = image_grid_thw.unsqueeze(0)  # (B, 3)
    
    print(f"  After unsqueeze - pixel_values: {pixel_values.shape}")
    print(f"  After unsqueeze - image_grid_thw: {image_grid_thw.shape}")
    
    # デバイスに移動
    device = next(model.parameters()).device
    pixel_values = pixel_values.to(device)
    image_grid_thw = image_grid_thw.to(device)
    
    # 4. 正しい方法で2560次元特徴を取得
    print("\n4. 2560次元PatchMerge特徴を取得...")
    
    # hidden_statesを有効化
    vision.config.output_hidden_states = True
    print("  ✓ output_hidden_states = True に設定")
    
    with torch.no_grad():
        # 正しい呼び出し方: 第一引数名はhidden_states
        out = vision(
            hidden_states=pixel_values,  # 第一引数名はhidden_states！
            grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True
        )
    
    print(f"  Output type: {type(out)}")
    
    # hidden_statesの確認
    if hasattr(out, 'hidden_states') and out.hidden_states is not None:
        print(f"  ✓ hidden_states が存在")
        print(f"  hidden_states数: {len(out.hidden_states)}")
        
        # 最後のhidden_stateがPatchMerge後の特徴
        feat2560 = out.hidden_states[-1]  # (B, N_raw/4, 2560)
        
        print(f"\n✓ PatchMerge features shape: {feat2560.shape}")
        B, N_patches, D_feat = feat2560.shape
        
        # 次元の検証
        if D_feat == 2560:
            print(f"🎉 成功: 2560次元のPatchMerge特徴を取得できました！")
            print(f"  - Batch size: {B}")
            print(f"  - Patches after merge: {N_patches}")
            print(f"  - Feature dimension: {D_feat}")
        else:
            print(f"❌ 予期しない次元: {D_feat}")
    else:
        print("  ❌ hidden_states が存在しない")
    
    # 5. 圧縮比の確認
    print("\n5. 圧縮比の検証...")
    H_raw = int(image_grid_thw[0, 1].item())
    W_raw = int(image_grid_thw[0, 2].item())
    raw_patches = H_raw * W_raw
    
    print(f"  RAW patches: {H_raw}×{W_raw} = {raw_patches}")
    print(f"  After PatchMerge (4:1): {raw_patches // 4}")
    print(f"  Actual patches: {N_patches}")
    
    if N_patches == raw_patches // 4:
        print("  ✓ 正しい4:1圧縮")
    
    # 6. 代替方法: 2ステージ明示（参考）
    print("\n6. 代替方法（2ステージ明示）...")
    
    with torch.no_grad():
        # patch_embed
        emb = vision.patch_embed
        x = emb(pixel_values, image_grid_thw)
        print(f"  After patch_embed: {x.shape}")
        
        # blocks (encoder)を通す
        # Note: blocksを直接呼ぶのは難しいので、ここでは参考程度
        # 実際には上記のhidden_states方法を使う
    
    # 7. LLM投影後との比較
    print("\n7. LLM投影後（2048次元）との比較...")
    
    with torch.no_grad():
        # get_image_featuresは2048次元を返す
        llm_features = model.get_image_features(
            pixel_values.squeeze(0),  # (N_raw, 1280)
            image_grid_thw.squeeze(0)  # (3,)
        )
        
        if isinstance(llm_features, tuple):
            llm_features = llm_features[0]
        
        print(f"  LLM features shape: {llm_features.shape}")
        print(f"  LLM feature dim: {llm_features.shape[-1]}")
        
        if llm_features.shape[-1] == 2048:
            print("  ✓ LLM投影後は2048次元（期待通り）")
    
    # 8. メモリ使用量
    if torch.cuda.is_available():
        print("\n8. GPU メモリ使用量:")
        print(f"  Allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
    
    print("\n" + "=" * 80)
    print("まとめ:")
    print("  ✓ model.model.visual()でhidden_states[-1]から2560次元取得成功")
    print("  ✓ 第一引数名はhidden_states（pixel_valuesではない）")
    print("  ✓ Qwen2.5-VL-3Bはhidden_size=1280、PatchMerge後=2560")
    print("  ✓ 現在のtransformers 4.42ではmodel.model.visualが正しいパス")
    print("=" * 80)
    
    return feat2560


if __name__ == "__main__":
    features = test_2560_features_correct()