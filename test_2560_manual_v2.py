#!/usr/bin/env python3
"""
O3推奨の手動方法で2560次元PatchMerge特徴を取得（修正版）
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

def test_2560_manual_v2():
    """手動でPatchEmbed → Blocks → Mergerを呼んで2560次元特徴を取得（修正版）"""
    
    print("=" * 80)
    print("2560次元 PatchMerge特徴 取得（手動方式v2）")
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
    
    # 正規入口: model.model.visual
    vision = model.model.visual
    print(f"  Vision module: model.model.visual")
    print(f"  hidden_size: {vision.config.hidden_size}")
    
    # 2. テスト画像の準備と前処理
    print("\n2. 画像準備と前処理...")
    test_image = Image.new('RGB', (448, 448), color='red')
    
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
    
    print(f"  pixel_values shape: {pixel_values.shape}")  # [1024, 1176]
    print(f"  image_grid_thw: {image_grid_thw}")
    
    # デバイスに移動
    device = next(model.parameters()).device
    pixel_values = pixel_values.to(device)
    image_grid_thw = image_grid_thw.to(device)
    
    # 3. patch_embedの引数を調査
    print("\n3. patch_embedの引数調査...")
    
    import inspect
    try:
        sig = inspect.signature(vision.patch_embed.forward)
        print(f"  patch_embed.forward signature: {sig}")
    except:
        pass
    
    # 4. 手動でPatchEmbed → Blocks → Mergerを呼ぶ（修正版）
    print("\n4. 手動でPatchMerge特徴を取得（修正版）...")
    
    with torch.no_grad():
        # ① PatchEmbed（引数は1つだけの可能性）
        try:
            # まずpixel_valuesだけで試す
            x = vision.patch_embed(pixel_values)
            print(f"  ① PatchEmbed出力（1引数）: {x.shape}")
        except:
            # それでもダメならpixel_valuesとimage_grid_thwを試す
            try:
                x = vision.patch_embed(pixel_values, image_grid_thw)
                print(f"  ① PatchEmbed出力（2引数）: {x.shape}")
            except Exception as e:
                print(f"  PatchEmbedエラー: {e}")
                # 代替: get_image_featuresを通して内部を観察
                print("\n  代替方法: フックで内部観察...")
                
                captured = {}
                
                def capture_patch_embed(module, input, output):
                    captured['patch_embed_out'] = output
                    print(f"    patch_embed captured: {output.shape}")
                
                def capture_last_block(module, input, output):
                    captured['last_block_out'] = output
                    print(f"    last_block captured: {output.shape if torch.is_tensor(output) else type(output)}")
                
                def capture_merger(module, input, output):
                    captured['merger_out'] = output
                    print(f"    merger captured: {output.shape}")
                
                hooks = []
                hooks.append(vision.patch_embed.register_forward_hook(capture_patch_embed))
                hooks.append(vision.blocks[-1].register_forward_hook(capture_last_block))
                hooks.append(vision.merger.register_forward_hook(capture_merger))
                
                # get_image_featuresを実行してフックで捕獲
                _ = model.get_image_features(pixel_values, image_grid_thw)
                
                for hook in hooks:
                    hook.remove()
                
                # patch_embed出力を使って続行
                if 'patch_embed_out' in captured:
                    x = captured['patch_embed_out']
                    print(f"\n  PatchEmbed出力を取得: {x.shape}")
                else:
                    print("  PatchEmbed出力が取得できませんでした")
                    return None
        
        # xが取得できた場合、続行
        if 'x' in locals():
            # ② Transformerブロック（grid_thwが必要かも）
            print(f"  ② Blocks数: {len(vision.blocks)}")
            
            for i, blk in enumerate(vision.blocks):
                try:
                    # grid_thwありで試す
                    x = blk(x, image_grid_thw)
                except:
                    # grid_thwなしで試す
                    try:
                        x = blk(x)
                    except Exception as e:
                        print(f"    Block {i} エラー: {e}")
                        break
                
                # 最初と最後のブロックだけログ
                if i == 0:
                    if torch.is_tensor(x):
                        print(f"    Block 0出力: {x.shape}")
                    else:
                        print(f"    Block 0出力: {type(x)}")
                elif i == len(vision.blocks) - 1:
                    if torch.is_tensor(x):
                        print(f"    Block {i}出力: {x.shape}")
            
            # ③ Merger
            try:
                # xがtupleの場合は最初の要素を取る
                if isinstance(x, tuple):
                    x = x[0]
                
                # merger呼び出し
                try:
                    merged = vision.merger(x, image_grid_thw)
                except:
                    merged = vision.merger(x)
                
                print(f"  ③ Merger出力: {merged.shape}")
                feat2560 = merged
                
            except Exception as e:
                print(f"  Mergerエラー: {e}")
                # capturedから取得
                if 'merger_out' in captured:
                    feat2560 = captured['merger_out']
                    print(f"  フックからMerger出力取得: {feat2560.shape}")
                else:
                    feat2560 = None
    
    # 5. 結果の検証
    if feat2560 is not None:
        print("\n5. 結果検証...")
        print(f"  最終特徴shape: {feat2560.shape}")
        
        if feat2560.dim() >= 2:
            dim = feat2560.shape[-1]
            if dim == 2560:
                print(f"  🎉 成功！ 2560次元のPatchMerge特徴を取得できました！")
            elif dim == 2048:
                print(f"  ⚠️ 2048次元（LLM投影後）になっています")
                print("     → mergerモジュールは2560→2048投影も含んでいる可能性")
            else:
                print(f"  ❌ 予期しない次元: {dim}")
    
    print("\n" + "=" * 80)
    print("まとめ:")
    if feat2560 is not None and feat2560.shape[-1] == 2048:
        print("  △ mergerモジュールは既にLLM投影（2560→2048）を含んでいる")
        print("  △ 純粋な2560次元特徴を得るには、merger前の特徴を使う必要がある")
    elif feat2560 is not None and feat2560.shape[-1] == 2560:
        print("  ✓ 2560次元のPatchMerge特徴を取得成功！")
    print("  ✓ 現在のtransformersではreturn_dict未実装のため手動/フック方式が必要")
    print("=" * 80)
    
    return feat2560


if __name__ == "__main__":
    features = test_2560_manual_v2()