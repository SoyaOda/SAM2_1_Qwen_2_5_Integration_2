#!/usr/bin/env python3
"""
transformers 4.54.1でvision_towerが使えるか確認
互換性のある方法も試す
"""
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoModelForCausalLM
from PIL import Image
import transformers

def test_vision_tower_v2():
    """vision_tower属性の確認（v4.54.1）"""
    
    print("=" * 80)
    print(f"Transformers version: {transformers.__version__}")
    print("vision_tower属性確認テスト")
    print("=" * 80)
    
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    # 1. 通常のロード方法
    print("\n1. 通常のロード（Qwen2_5_VLForConditionalGeneration）...")
    model1 = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype="auto", 
        device_map="auto"
    )
    
    print(f"  Model type: {type(model1)}")
    
    # vision_towerチェック
    if hasattr(model1, 'vision_tower'):
        print("  ✓ model.vision_tower が存在！")
        print(f"    Type: {type(model1.vision_tower)}")
    else:
        print("  ✗ model.vision_tower が存在しない")
    
    # model.model.visualチェック
    if hasattr(model1, 'model') and hasattr(model1.model, 'visual'):
        print("  ✓ model.model.visual も存在")
        print(f"    Type: {type(model1.model.visual)}")
    
    # 2. trust_remote_code=Trueでロード
    print("\n2. trust_remote_code=Trueでロード...")
    model2 = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype="auto", 
        device_map="auto"
    )
    
    if hasattr(model2, 'vision_tower'):
        print("  ✓ model.vision_tower が存在！")
    else:
        print("  ✗ model.vision_tower が存在しない")
    
    # 3. AutoModelForCausalLMでロード（試行）
    print("\n3. AutoModelForCausalLMでロード（試行）...")
    try:
        model3 = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype="auto", 
            device_map="auto"
        )
        
        print(f"  Model type: {type(model3)}")
        
        if hasattr(model3, 'vision_tower'):
            print("  ✓ model.vision_tower が存在！")
        else:
            print("  ✗ model.vision_tower が存在しない")
    except Exception as e:
        print(f"  エラー: {str(e)[:100]}...")
    
    # 4. 互換性のあるアクセス方法
    print("\n4. 互換性のあるアクセス方法...")
    
    def get_vision_module(model):
        """vision_towerまたはmodel.model.visualを取得"""
        vision = getattr(model, "vision_tower", None)
        if vision is None:
            if hasattr(model, 'model') and hasattr(model.model, 'visual'):
                vision = model.model.visual
        return vision
    
    vision = get_vision_module(model1)
    if vision:
        print(f"  ✓ Vision module取得成功: {type(vision)}")
        
        # configの確認
        if hasattr(vision, 'config'):
            config = vision.config
            print(f"    - hidden_size: {config.hidden_size}")
            print(f"    - PatchMerge後: {config.hidden_size * 2}")
    
    # 5. 実際に2560次元特徴を取得してみる
    print("\n5. 2560次元特徴取得テスト...")
    
    # テスト画像
    test_image = Image.new('RGB', (448, 448), color='red')
    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
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
    
    # バッチ次元追加
    if pixel_values.dim() == 2:
        pixel_values = pixel_values.unsqueeze(0)
    if image_grid_thw.dim() == 1:
        image_grid_thw = image_grid_thw.unsqueeze(0)
    
    device = next(model1.parameters()).device
    pixel_values = pixel_values.to(device)
    image_grid_thw = image_grid_thw.to(device)
    
    # hidden_states取得を試す
    if vision:
        vision.config.output_hidden_states = True
        
        try:
            with torch.no_grad():
                # vision_towerが存在する場合
                if hasattr(model1, 'vision_tower'):
                    out = model1.vision_tower(
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        output_hidden_states=True,
                        return_dict=True
                    )
                # model.model.visualを使う場合
                else:
                    out = model1.model.visual(
                        hidden_states=pixel_values,  # 引数名注意
                        grid_thw=image_grid_thw,
                        output_hidden_states=True,
                        return_dict=True
                    )
                
                if hasattr(out, 'hidden_states') and out.hidden_states:
                    feat = out.hidden_states[-1]
                    print(f"  ✓ Hidden states取得成功: {feat.shape}")
                    print(f"    次元: {feat.shape[-1]}")
                else:
                    print(f"  Output type: {type(out)}")
                    if torch.is_tensor(out):
                        print(f"  Tensor shape: {out.shape}")
        except Exception as e:
            print(f"  エラー: {e}")
    
    print("\n" + "=" * 80)
    print("結論:")
    print(f"  Transformers {transformers.__version__}での状況：")
    if hasattr(model1, 'vision_tower'):
        print("  ✓ vision_towerが利用可能！")
    else:
        print("  ✗ vision_towerはまだ利用不可")
        print("  → model.model.visualを使用")
    print("=" * 80)


if __name__ == "__main__":
    test_vision_tower_v2()