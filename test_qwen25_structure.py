#!/usr/bin/env python3
"""
Qwen2.5-VLのモデル構造を詳細に調査するスクリプト
"""
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image

def investigate_qwen25_structure():
    """Qwen2.5-VLの実際の構造を調査"""
    
    print("=" * 80)
    print("Qwen2.5-VL モデル構造調査")
    print("=" * 80)
    
    # 1. モデルロード
    print("\n1. モデルロード...")
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype="auto", 
        device_map="auto"
    )
    
    # 2. モデル属性調査
    print("\n2. モデルのトップレベル属性:")
    main_attrs = [attr for attr in dir(model) if not attr.startswith('_')]
    for attr in sorted(main_attrs)[:20]:
        if hasattr(model, attr):
            obj = getattr(model, attr)
            if not callable(obj):
                print(f"  - {attr}: {type(obj).__name__}")
    
    # 3. model.model の属性調査
    print("\n3. model.model の属性:")
    if hasattr(model, 'model'):
        model_attrs = [attr for attr in dir(model.model) if not attr.startswith('_')]
        for attr in sorted(model_attrs)[:20]:
            if hasattr(model.model, attr):
                obj = getattr(model.model, attr)
                if not callable(obj):
                    print(f"  - {attr}: {type(obj).__name__}")
    
    # 4. vision関連モジュール調査
    print("\n4. Vision関連モジュール:")
    
    # vision_tower の存在確認
    if hasattr(model, 'vision_tower'):
        print("  ✓ model.vision_tower が存在")
        vision = model.vision_tower
    elif hasattr(model, 'model') and hasattr(model.model, 'visual'):
        print("  ✓ model.model.visual が存在")
        vision = model.model.visual
    elif hasattr(model, 'visual'):
        print("  ✓ model.visual が存在")
        vision = model.visual
    else:
        print("  ✗ vision モジュールが見つからない")
        vision = None
    
    if vision:
        print(f"  Vision type: {type(vision).__name__}")
        print(f"  Vision config: {type(vision.config).__name__}" if hasattr(vision, 'config') else "  No config")
        
        # vision内部の構造
        print("\n5. Vision内部モジュール:")
        for name, module in vision.named_children():
            print(f"  - {name}: {type(module).__name__}")
    
    # 5. 実際のforward呼び出しテスト
    print("\n6. Forward呼び出しテスト:")
    
    # テスト画像準備
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
    
    # 前処理
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = proc(
        text=text,
        images=test_image,
        return_tensors="pt"
    )
    
    device = next(model.parameters()).device
    pixel_values = inputs["pixel_values"].to(device)
    image_grid_thw = inputs["image_grid_thw"].to(device)
    
    print(f"  pixel_values shape: {pixel_values.shape}")
    print(f"  image_grid_thw: {image_grid_thw}")
    
    # バッチ次元を追加
    if pixel_values.dim() == 2:
        pixel_values = pixel_values.unsqueeze(0)
    
    # 7. 各種呼び出し方法を試す
    print("\n7. 各種呼び出し方法の検証:")
    
    # 方法1: visual()を直接呼び出し
    if vision:
        print("\n  方法1: visual()直接呼び出し")
        try:
            # configにoutput_hidden_statesを設定
            if hasattr(vision, 'config'):
                vision.config.output_hidden_states = True
                print("    output_hidden_states = True 設定済み")
            
            with torch.no_grad():
                # キーワード引数で呼び出し
                result = vision(
                    pixel_values=pixel_values,
                    grid_thw=image_grid_thw
                )
            
            print(f"    結果type: {type(result)}")
            if torch.is_tensor(result):
                print(f"    Tensor shape: {result.shape}")
            elif hasattr(result, '__dict__'):
                print(f"    属性: {list(result.__dict__.keys())}")
            
        except Exception as e:
            print(f"    エラー: {e}")
    
    # 方法2: return_dict=Trueを明示
    if vision:
        print("\n  方法2: return_dict=True指定")
        try:
            with torch.no_grad():
                result = vision(
                    pixel_values=pixel_values,
                    grid_thw=image_grid_thw,
                    return_dict=True
                )
            
            print(f"    結果type: {type(result)}")
            if hasattr(result, 'hidden_states'):
                print(f"    ✓ hidden_states が存在")
                if result.hidden_states:
                    print(f"    hidden_states数: {len(result.hidden_states)}")
                    print(f"    最後のhidden_state shape: {result.hidden_states[-1].shape}")
            else:
                print(f"    ✗ hidden_states が存在しない")
                if hasattr(result, '__dict__'):
                    print(f"    利用可能な属性: {list(result.__dict__.keys())}")
            
        except Exception as e:
            print(f"    エラー: {e}")
    
    # 方法3: output_hidden_states=Trueを明示
    if vision:
        print("\n  方法3: output_hidden_states=True指定")
        try:
            with torch.no_grad():
                result = vision(
                    pixel_values=pixel_values,
                    grid_thw=image_grid_thw,
                    output_hidden_states=True,
                    return_dict=True
                )
            
            print(f"    結果type: {type(result)}")
            if hasattr(result, 'hidden_states'):
                print(f"    ✓ hidden_states が存在")
                if result.hidden_states:
                    print(f"    hidden_states数: {len(result.hidden_states)}")
                    for i, hs in enumerate(result.hidden_states[-3:]):
                        print(f"      Layer {len(result.hidden_states) - 3 + i}: shape {hs.shape}")
            else:
                print(f"    ✗ hidden_states が存在しない")
            
        except Exception as e:
            print(f"    エラー: {e}")
    
    print("\n" + "=" * 80)
    print("調査完了")
    print("=" * 80)


if __name__ == "__main__":
    investigate_qwen25_structure()