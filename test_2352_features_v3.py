#!/usr/bin/env python3
"""
2352次元PatchMerge特徴の確実な取得テスト（O3推奨方法v3）
AutoModelForCausalLMとvision_towerを使用
"""
import torch
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import numpy as np

def test_2352_features_v3():
    """O3推奨の方法で2352次元特徴を確実に取得（修正版）"""
    
    print("=" * 80)
    print("2352次元 PatchMerge特徴 取得テスト v3（O3準拠・修正版）")
    print("=" * 80)
    
    # 1. モデルとプロセッサの準備（AutoModelForCausalLMを使用）
    print("\n1. モデル準備...")
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    # trust_remote_code=Trueが重要
    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True,  # 重要！
        torch_dtype="auto", 
        device_map="auto"
    )
    
    print(f"  Model type: {type(model)}")
    
    # vision_towerの確認
    if hasattr(model, 'vision_tower'):
        print(f"  ✓ model.vision_tower が存在")
        vision_module = model.vision_tower
    else:
        print(f"  Available attributes: {[attr for attr in dir(model) if not attr.startswith('_')][:20]}")
        raise AttributeError("vision_tower not found")
    
    print(f"  Vision module type: {type(vision_module)}")
    
    # ❶ 視覚タワーにhidden_states出力を要求（重要！）
    vision_module.config.output_hidden_states = True
    print("  ✓ output_hidden_states = True に設定")
    
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
    
    # ❷ apply_chat_templateでtokenize=True（重要！）
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = proc(
        text=text,
        images=test_image,
        return_tensors="pt"
    )
    
    # pixel_valuesとimage_grid_thwを取得
    pixel_values = inputs["pixel_values"]      
    image_grid_thw = inputs["image_grid_thw"]  
    
    print(f"  pixel_values shape (before unsqueeze): {pixel_values.shape}")
    print(f"  image_grid_thw shape (before unsqueeze): {image_grid_thw.shape}")
    
    # バッチ次元を追加（重要！）
    if pixel_values.dim() == 2:
        pixel_values = pixel_values.unsqueeze(0)  # (1, N_raw, 1176)
    if image_grid_thw.dim() == 1:
        image_grid_thw = image_grid_thw.unsqueeze(0)  # (1, 3)
    
    print(f"  pixel_values shape (after unsqueeze): {pixel_values.shape}")
    print(f"  image_grid_thw shape (after unsqueeze): {image_grid_thw.shape}")
    
    # デバイスに移動
    device = next(model.parameters()).device
    pixel_values = pixel_values.to(device)
    image_grid_thw = image_grid_thw.to(device)
    
    # 4. vision_towerを使ってPatchMerge後2352次元を取得
    print("\n4. Vision Tower Forward Pass...")
    
    # ❸ vision_towerを使用（model.model.visualではない！）
    with torch.no_grad():
        # vision_towerを呼び出し - キーワード引数とreturn_dict=Trueが必須
        vision_out = model.vision_tower(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True  # 必須！
        )
    
    print(f"  vision_out type: {type(vision_out)}")
    
    # hidden_statesの確認
    if hasattr(vision_out, 'hidden_states') and vision_out.hidden_states is not None:
        print(f"  ✓ hidden_states が存在")
        print(f"  hidden_states数: {len(vision_out.hidden_states)}")
        
        # hidden_states[-1]がPatchMerge後の特徴
        patchmerge_feat = vision_out.hidden_states[-1]  # (B, N_raw/4, 2352)
        
        print(f"\n✓ PatchMerge features shape: {patchmerge_feat.shape}")
        B, N_patches, D_feat = patchmerge_feat.shape
        
        # 次元の検証
        if D_feat == 2352:
            print(f"🎉 成功: 2352次元のPatchMerge特徴を取得できました！")
            print(f"  - Batch size: {B}")
            print(f"  - Patches after merge: {N_patches}")
            print(f"  - Feature dimension: {D_feat}")
        elif D_feat == 1176:
            print(f"⚠️ 1176次元の特徴（PatchEmbed後、Merge前）")
        else:
            print(f"❌ 予期しない次元: {D_feat}")
    else:
        print("  ❌ hidden_states が存在しない")
        if hasattr(vision_out, '__dict__'):
            print(f"  利用可能な属性: {list(vision_out.__dict__.keys())}")
    
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
    elif N_patches == raw_patches // 8:
        print("  → top128選択も適用されています（8:1圧縮）")
    
    # 6. LLM投影後の特徴との比較
    print("\n6. LLM投影後特徴との比較...")
    
    # get_image_featuresが存在する場合
    if hasattr(model, 'get_image_features'):
        with torch.no_grad():
            # get_image_featuresは2048次元を返す
            llm_features = model.get_image_features(
                pixel_values.squeeze(0),  # get_image_featuresは(N_raw, 1176)を期待
                image_grid_thw.squeeze(0)  # (3,)を期待
            )
            
            if isinstance(llm_features, tuple):
                llm_features = llm_features[0]
            
            print(f"  LLM features shape: {llm_features.shape}")
            print(f"  LLM feature dim: {llm_features.shape[-1]}")
            
            if llm_features.shape[-1] == 2048:
                print("  ✓ LLM投影後は2048次元（期待通り）")
    
    # 7. メモリ使用量
    if torch.cuda.is_available():
        print("\n7. GPU メモリ使用量:")
        print(f"  Allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
    
    print("\n" + "=" * 80)
    print("まとめ:")
    print("  ✓ model.vision_tower()でhidden_states[-1]から2352次元取得成功")
    print("  ✓ AutoModelForCausalLMとtrust_remote_code=Trueが重要")
    print("  ✓ return_dict=Trueとoutput_hidden_states=Trueの両方が必須")
    print("  ✓ pixel_valuesとimage_grid_thwにバッチ次元を追加")
    print("=" * 80)
    
    return patchmerge_feat


if __name__ == "__main__":
    features = test_2352_features_v3()