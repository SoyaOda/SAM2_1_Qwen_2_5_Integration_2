#!/usr/bin/env python3
"""
Qwen2.5-VL-3Bの構造を探索して理解するスクリプト
中間層特徴抽出の前準備
"""

import torch
from transformers import Qwen2VLForConditionalGeneration
import sys

print("=" * 80)
print("Qwen2.5-VL-3B モデル構造探索")
print("=" * 80)

# モデルのロード（メタデータのみ）
print("\n1. モデルのロード...")
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

try:
    # CPUで軽量にロード
    from transformers import AutoModelForVision2Seq
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype=torch.float16
    )
    print("  ✅ モデルロード成功")
except Exception as e:
    print(f"  ❌ エラー: {e}")
    print("  別の方法でロード...")
    try:
        # Qwen2VLの新しいAPIを試す
        from transformers import Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cpu"
        )
        print("  ✅ モデルロード成功（Qwen2VL API）")
    except Exception as e2:
        print(f"  ❌ 最終エラー: {e2}")
        sys.exit(1)

print("\n2. トップレベル構造:")
for name, module in model.named_children():
    print(f"  {name}: {type(module).__name__}")

print("\n3. 視覚モデル (model.visual) の詳細構造:")
if hasattr(model, 'visual'):
    visual = model.visual
    for name, module in visual.named_children():
        print(f"  visual.{name}: {type(module).__name__}")
        
        # サブモジュールの数を確認
        if hasattr(module, '__len__'):
            try:
                length = len(module)
                print(f"    -> {length} sub-modules")
            except:
                pass

print("\n4. 視覚エンコーダーの詳細探索:")
# Qwen2VLの視覚モデルの実際の構造を探索
def explore_module(module, prefix="", max_depth=3, current_depth=0):
    if current_depth >= max_depth:
        return
    
    for name, submodule in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        module_type = type(submodule).__name__
        
        # ブロックやレイヤーを探す
        if any(keyword in name.lower() for keyword in ['block', 'layer', 'encoder', 'decoder']):
            print(f"  {'  ' * current_depth}{full_name}: {module_type}", end="")
            
            # ModuleListの場合は要素数を表示
            if 'ModuleList' in module_type or 'Sequential' in module_type:
                try:
                    print(f" [{len(submodule)} modules]")
                    # 最初と最後のモジュールの型を表示
                    if len(submodule) > 0:
                        print(f"  {'  ' * (current_depth+1)}First: {type(submodule[0]).__name__}")
                        if len(submodule) > 1:
                            print(f"  {'  ' * (current_depth+1)}Last: {type(submodule[-1]).__name__}")
                except:
                    print()
            else:
                print()
            
            # 再帰的に探索
            explore_module(submodule, full_name, max_depth, current_depth + 1)

print("\n視覚モデルの階層構造:")
explore_module(model.visual, "visual", max_depth=4)

print("\n5. 言語モデル (model.model) の構造:")
if hasattr(model, 'model'):
    llm = model.model
    for name, module in llm.named_children():
        module_type = type(module).__name__
        print(f"  model.{name}: {module_type}", end="")
        
        if 'ModuleList' in module_type:
            try:
                print(f" [{len(module)} layers]")
            except:
                print()
        else:
            print()

print("\n6. 特徴次元の確認:")
# 設定から次元情報を取得
if hasattr(model, 'config'):
    config = model.config
    print(f"  Hidden size: {getattr(config, 'hidden_size', 'N/A')}")
    print(f"  Vision config: {getattr(config, 'vision_config', 'N/A')}")
    
    if hasattr(config, 'vision_config'):
        vision_config = config.vision_config
        print("\n  Vision config details:")
        for key in dir(vision_config):
            if not key.startswith('_'):
                value = getattr(vision_config, key)
                if isinstance(value, (int, float, str, bool)):
                    print(f"    {key}: {value}")

print("\n7. 中間層アクセス方法の確認:")
# 実際のブロックへのアクセス方法を確認
try:
    # 一般的なパターンを試す
    paths_to_try = [
        "visual.blocks",
        "visual.encoder.layers",
        "visual.transformer.blocks",
        "visual.layers",
        "visual.encoder.blocks",
    ]
    
    for path in paths_to_try:
        try:
            parts = path.split('.')
            current = model
            for part in parts:
                current = getattr(current, part)
            print(f"  ✅ Found blocks at: {path}")
            print(f"     Number of blocks: {len(current)}")
            print(f"     Block type: {type(current[0]).__name__}")
            
            # 最初のブロックの構造
            print(f"\n     First block structure:")
            for name, module in current[0].named_children():
                print(f"       {name}: {type(module).__name__}")
            
            break
        except (AttributeError, IndexError):
            continue
    else:
        print("  ⚠️ 標準的なブロック構造が見つかりません")
        print("\n  代替: visual内の全てのModuleListを列挙")
        for name, module in model.visual.named_modules():
            if 'ModuleList' in type(module).__name__:
                try:
                    print(f"    {name}: ModuleList with {len(module)} items")
                    if len(module) > 0:
                        print(f"      Item type: {type(module[0]).__name__}")
                except:
                    pass
                    
except Exception as e:
    print(f"  エラー: {e}")

print("\n" + "=" * 80)
print("構造探索完了")
print("次のステップ: この情報を基に中間層特徴抽出を実装")
print("=" * 80)