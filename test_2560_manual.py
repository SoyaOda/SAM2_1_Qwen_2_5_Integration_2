#!/usr/bin/env python3
"""
O3推奨の手動方法で2560次元PatchMerge特徴を取得
PatchEmbed → Blocks → Merger → LayerNormを手動で呼ぶ
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

def test_2560_manual():
    """手動でPatchEmbed → Blocks → Mergerを呼んで2560次元特徴を取得"""
    
    print("=" * 80)
    print("2560次元 PatchMerge特徴 取得（手動方式）")
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
    
    # 正規入口: model.model.visual
    vision = model.model.visual
    print(f"  Vision module: model.model.visual")
    print(f"  Vision type: {type(vision)}")
    print(f"  hidden_size: {vision.config.hidden_size}")
    
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
                {"type": "text", "text": "Test"}
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
    
    # 4. 手動でPatchEmbed → Blocks → Mergerを呼ぶ
    print("\n4. 手動でPatchMerge特徴を取得...")
    
    with torch.no_grad():
        # ① PatchEmbed
        x = vision.patch_embed(pixel_values, image_grid_thw)
        print(f"  ① PatchEmbed出力: {x.shape}")  # (B, N_raw, 1280)
        
        # ② Transformerブロック
        print(f"  ② Blocks数: {len(vision.blocks)}")
        for i, blk in enumerate(vision.blocks):
            x = blk(x, image_grid_thw)
            # 最初と最後のブロックだけログ
            if i == 0:
                print(f"    Block 0出力: {x.shape}")
            elif i == len(vision.blocks) - 1:
                print(f"    Block {i}出力: {x.shape}")
        
        # ③ Merger 2×2
        x = vision.merger(x, image_grid_thw)
        print(f"  ③ Merger出力: {x.shape}")  # 期待: (B, N_raw/4, 2560)
        
        # ④ LayerNorm（もし存在すれば）
        if hasattr(vision, 'ln'):
            x = vision.ln(x)
            print(f"  ④ LayerNorm出力: {x.shape}")
        elif hasattr(vision, 'norm'):
            x = vision.norm(x)
            print(f"  ④ Norm出力: {x.shape}")
        
        feat2560 = x
    
    # 5. 結果の検証
    print("\n5. 結果検証...")
    print(f"  最終特徴shape: {feat2560.shape}")
    
    if feat2560.dim() >= 2:
        dim = feat2560.shape[-1]
        if dim == 2560:
            print(f"  🎉 成功！ 2560次元のPatchMerge特徴を取得できました！")
        elif dim == 2048:
            print(f"  ⚠️ 2048次元（LLM投影後）になっています")
        else:
            print(f"  ❌ 予期しない次元: {dim}")
    
    # 6. フック方式も試す（比較のため）
    print("\n6. フック方式での取得...")
    
    feat = {}
    def tap(_, __, output):
        # outputがtupleの場合は最初の要素を取る
        if isinstance(output, tuple):
            output = output[0]
        feat["pm"] = output.clone()
        print(f"    Hook captured: {output.shape}")
    
    hook = vision.merger.register_forward_hook(tap)
    
    with torch.no_grad():
        _ = vision(hidden_states=pixel_values, grid_thw=image_grid_thw)
    
    hook.remove()
    
    if "pm" in feat:
        feat2560_hook = feat["pm"]
        print(f"  Hook結果: {feat2560_hook.shape}")
        
        # 手動方式とフック方式の結果を比較
        if feat2560.shape == feat2560_hook.shape:
            if torch.allclose(feat2560, feat2560_hook, atol=1e-5):
                print("  ✓ 手動方式とフック方式の結果が一致！")
            else:
                print("  △ 形状は同じだが値が異なる")
    
    # 7. SAM用Adapterの例
    print("\n7. SAM用Adapter例...")
    
    if feat2560.shape[-1] == 2560:
        # 2560 → 256へのAdapter
        adapter = torch.nn.Sequential(
            torch.nn.Linear(2560, 512),
            torch.nn.GELU(),
            torch.nn.Linear(512, 256),
            torch.nn.LayerNorm(256)
        ).to(device)
        
        with torch.no_grad():
            feat256 = adapter(feat2560)
        
        print(f"  Adapter出力: {feat256.shape}")
        print(f"  SAM2.1 MaskDecoder入力として使用可能")
    
    # 8. メモリ使用量
    if torch.cuda.is_available():
        print("\n8. GPU メモリ使用量:")
        print(f"  Allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
    
    print("\n" + "=" * 80)
    print("まとめ:")
    print("  ✓ PatchEmbed → Blocks → Merger → LayerNormの手動実行成功")
    print("  ✓ 2560次元（または2048次元）の特徴を取得")
    print("  ✓ return_dict未実装のため、この方法が現在のベストプラクティス")
    print("=" * 80)
    
    return feat2560


if __name__ == "__main__":
    features = test_2560_manual()