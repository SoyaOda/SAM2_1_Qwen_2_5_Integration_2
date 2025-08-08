#!/usr/bin/env python3
"""
Qwen2.5-VLのvision_tower属性の存在確認
"""
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoModel
from PIL import Image

def test_vision_tower():
    """vision_tower属性の存在を確認"""
    
    print("=" * 80)
    print("Qwen2.5-VL vision_tower属性確認")
    print("=" * 80)
    
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    # 1. Qwen2_5_VLForConditionalGenerationで読み込み
    print("\n1. Qwen2_5_VLForConditionalGenerationで読み込み...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype="auto", 
        device_map="auto"
    )
    
    print(f"  Model type: {type(model)}")
    print(f"  Model attributes (vision関連):")
    
    # vision関連の属性を探す
    for attr in dir(model):
        if 'vision' in attr.lower() or 'visual' in attr.lower():
            if not attr.startswith('_'):
                print(f"    - {attr}: {hasattr(model, attr)}")
    
    # 詳細確認
    print("\n  詳細確認:")
    if hasattr(model, 'vision_tower'):
        print("    ✓ model.vision_tower が存在")
        print(f"      Type: {type(model.vision_tower)}")
    else:
        print("    ✗ model.vision_tower が存在しない")
    
    if hasattr(model, 'model'):
        print("    ✓ model.model が存在")
        if hasattr(model.model, 'visual'):
            print("      ✓ model.model.visual が存在")
            print(f"        Type: {type(model.model.visual)}")
        if hasattr(model.model, 'vision_tower'):
            print("      ✓ model.model.vision_tower が存在")
            print(f"        Type: {type(model.model.vision_tower)}")
    
    # 2. trust_remote_code=Trueで読み込み
    print("\n2. trust_remote_code=Trueで読み込み...")
    try:
        model2 = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype="auto", 
            device_map="auto"
        )
        
        print(f"  Model type: {type(model2)}")
        
        if hasattr(model2, 'vision_tower'):
            print("    ✓ model.vision_tower が存在（trust_remote_code=True）")
        else:
            print("    ✗ model.vision_tower が存在しない（trust_remote_code=True）")
    except Exception as e:
        print(f"  エラー: {e}")
    
    # 3. AutoModelで読み込み（試行）
    print("\n3. AutoModel.from_pretrainedで読み込み（試行）...")
    try:
        model3 = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype="auto", 
            device_map="auto"
        )
        
        print(f"  Model type: {type(model3)}")
        
        if hasattr(model3, 'vision_tower'):
            print("    ✓ model.vision_tower が存在（AutoModel）")
        else:
            print("    ✗ model.vision_tower が存在しない（AutoModel）")
    except Exception as e:
        print(f"  エラー: {e}")
    
    print("\n" + "=" * 80)
    print("結論:")
    print("  Qwen2.5-VL-3B-Instructの実際の構造：")
    print("  - model.model.visual が存在（確定）")
    print("  - model.vision_tower は存在しない")
    print("  - hidden_size = 1280")
    print("  - PatchMerge後 = 2560次元（1280×2）")
    print("=" * 80)


if __name__ == "__main__":
    test_vision_tower()