#!/usr/bin/env python3
"""
2352次元のPatchMerge後のhidden_states取得を調査するスクリプト
O3推奨の方法を実装・検証
"""

import torch
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from PIL import Image
import numpy as np

def investigate_2352_features():
    """2352次元特徴の取得方法を調査"""
    
    print("=" * 80)
    print("2352次元 Hidden States 取得調査")
    print("=" * 80)
    
    # 1. LISA改モデルを直接ロード
    print("\n1. LISA改モデル準備中...")
    from src.models.lisa_model import LISA_Model
    from src.config import LISAConfig
    from transformers import Qwen2VLProcessor
    
    config = LISAConfig()
    processor = Qwen2VLProcessor.from_pretrained(config.qwen_model_name)
    
    # LISA改モデルを初期化（簡易版）
    import torch
    from transformers import Qwen2VLForConditionalGeneration
    
    print("  Qwenモデル読み込み中...")
    qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
        config.qwen_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    
    # visualモジュールを取得
    visual = qwen_model.model.visual
    visual.eval()
    
    # 2. テスト画像の準備
    print("\n2. テスト画像準備...")
    test_image = Image.new('RGB', (448, 448), color='red')
    
    # 3. 画像処理
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Test"}
            ]
        }
    ]
    
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in inputs.items()}
    
    print(f"  pixel_values shape: {inputs['pixel_values'].shape}")
    print(f"  image_grid_thw: {inputs['image_grid_thw']}")
    
    # 4. Visual モジュールの構造を調査
    print("\n3. Visual モジュール構造:")
    visual = model.model.visual
    
    print(f"  Visual type: {type(visual)}")
    print(f"  Visual config:")
    if hasattr(visual, 'config'):
        config = visual.config
        print(f"    - hidden_size: {config.hidden_size}")
        print(f"    - intermediate_size: {config.intermediate_size}")
        print(f"    - num_hidden_layers: {config.num_hidden_layers}")
    
    # 5. Visual モジュールの内部構造
    print("\n4. Visual 内部モジュール:")
    for name, module in visual.named_children():
        print(f"  - {name}: {type(module).__name__}")
        if hasattr(module, 'out_features'):
            print(f"      out_features: {module.out_features}")
        elif hasattr(module, 'hidden_size'):
            print(f"      hidden_size: {module.hidden_size}")
    
    # 6. 実際に visual() を呼び出して hidden_states を取得
    print("\n5. Visual forward pass で hidden_states 取得:")
    
    with torch.no_grad():
        # pixel_valuesとgrid_thwを渡す
        pixel_values = inputs['pixel_values']
        grid_thw = inputs['image_grid_thw']
        
        print(f"  Input pixel_values: {pixel_values.shape}")
        print(f"  Input grid_thw: {grid_thw}")
        
        # visual()を直接呼び出し - return_dict=Trueで詳細な出力を取得
        try:
            # 方法1: visual()を直接呼び出し
            visual_outputs = visual(pixel_values, grid_thw=grid_thw)
            
            print(f"\n  visual() 出力:")
            print(f"    - Type: {type(visual_outputs)}")
            
            # BaseModelOutputの場合
            if hasattr(visual_outputs, 'last_hidden_state'):
                print(f"    - last_hidden_state shape: {visual_outputs.last_hidden_state.shape}")
                print(f"    - last_hidden_state dtype: {visual_outputs.last_hidden_state.dtype}")
            
            # hidden_statesがある場合（output_hidden_states=Trueが必要かも）
            if hasattr(visual_outputs, 'hidden_states') and visual_outputs.hidden_states is not None:
                print(f"    - hidden_states length: {len(visual_outputs.hidden_states)}")
                for i, hs in enumerate(visual_outputs.hidden_states[-3:]):  # 最後の3層
                    print(f"      Layer {i-3}: shape {hs.shape}")
            
        except Exception as e:
            print(f"  Error in visual(): {e}")
            print("  Trying alternative approach...")
    
    # 7. merger と blocks の詳細調査
    print("\n6. Merger と Blocks の詳細:")
    
    if hasattr(visual, 'merger') and hasattr(visual, 'blocks'):
        print("  merger found!")
        merger = visual.merger
        blocks = visual.blocks
        
        print(f"    - Merger type: {type(merger)}")
        print(f"    - Merger config: {merger.config if hasattr(merger, 'config') else 'N/A'}")
        
        # Mergerの出力次元を調べる
        if hasattr(merger, 'hidden_size'):
            print(f"    - Merger hidden_size: {merger.hidden_size}")
        if hasattr(merger, 'd_model'):
            print(f"    - Merger d_model: {merger.d_model}")
        if hasattr(merger, 'ln_q'):
            # ln_qがある場合、その正規化次元が出力次元のヒント
            print(f"    - Merger ln_q normalized_shape: {merger.ln_q.normalized_shape}")
        
        print(f"    - Blocks length: {len(blocks)}")
        print(f"    - Last block type: {type(blocks[-1])}")
    
    # 8. フックを使って中間出力を取得
    print("\n7. フックを使った中間特徴取得:")
    
    captured_features = {}
    
    def capture_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            captured_features[name] = output.detach()
            print(f"  Captured {name}: shape {output.shape}, dtype {output.dtype}")
        return hook_fn
    
    # mergerの出力をキャプチャ
    hooks = []
    if hasattr(visual, 'merger'):
        hook = visual.merger.register_forward_hook(capture_hook('merger_output'))
        hooks.append(hook)
    
    # 最後のblockの出力をキャプチャ
    if hasattr(visual, 'blocks') and len(visual.blocks) > 0:
        hook = visual.blocks[-1].register_forward_hook(capture_hook('last_block_output'))
        hooks.append(hook)
    
    # forward pass
    with torch.no_grad():
        try:
            _ = visual(pixel_values, grid_thw=grid_thw)
        except:
            pass
    
    # フックを削除
    for hook in hooks:
        hook.remove()
    
    # 9. キャプチャした特徴の分析
    print("\n8. キャプチャした特徴の分析:")
    for name, feat in captured_features.items():
        print(f"  {name}:")
        print(f"    - Shape: {feat.shape}")
        print(f"    - Dtype: {feat.dtype}")
        print(f"    - 次元: {feat.shape[-1]}")
        
        # 2352次元かチェック
        if feat.shape[-1] == 2352:
            print(f"    ✓ 2352次元の特徴を発見！")
        elif feat.shape[-1] == 1176:
            print(f"    → 1176次元 (2352の半分) - これを2つ結合すれば2352次元")
    
    # 10. 結論と推奨実装
    print("\n" + "=" * 80)
    print("結論と推奨実装:")
    print("=" * 80)
    
    if 'merger_output' in captured_features:
        feat = captured_features['merger_output']
        if feat.shape[-1] == 2352 or feat.shape[-1] == 1176:
            print("\n✓ 2352次元特徴の取得方法が判明しました！")
            print("\n推奨実装:")
            print("```python")
            print("# extract_vision_features メソッドの修正")
            print("def extract_vision_features(self, pixel_values, image_grid_thw):")
            print("    # Visualモジュールを直接使用")
            print("    visual_outputs = self.qwen.model.visual(pixel_values, grid_thw=image_grid_thw)")
            print("    ")
            print("    # merger出力を取得するためのフック")
            print("    merger_output = None")
            print("    def capture_merger(module, input, output):")
            print("        nonlocal merger_output")
            print("        merger_output = output[0] if isinstance(output, tuple) else output")
            print("    ")
            print("    hook = self.qwen.model.visual.merger.register_forward_hook(capture_merger)")
            print("    _ = visual_outputs  # forward実行")
            print("    hook.remove()")
            print("    ")
            print("    # merger_outputが2352次元の特徴")
            print("    return merger_output")
            print("```")
    
    return captured_features


if __name__ == "__main__":
    features = investigate_2352_features()