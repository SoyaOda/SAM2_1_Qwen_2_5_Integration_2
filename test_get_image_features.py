#!/usr/bin/env python3
"""
get_image_featuresの内部動作を調査して2352次元特徴を取得する方法を探る
"""
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image
import inspect

def investigate_get_image_features():
    """get_image_featuresの内部実装を調査"""
    
    print("=" * 80)
    print("get_image_features 内部調査")
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
    
    # 2. get_image_featuresメソッドの確認
    print("\n2. get_image_featuresメソッドの存在確認:")
    if hasattr(model, 'get_image_features'):
        print("  ✓ model.get_image_features が存在")
        # ソースコードを確認
        try:
            source = inspect.getsource(model.get_image_features)
            print("\n  get_image_featuresのソースコード（最初の500文字）:")
            print("  " + source[:500].replace("\n", "\n  "))
        except:
            pass
    
    # 3. テスト画像準備
    print("\n3. テスト画像準備...")
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
    
    # 4. get_image_featuresを実行
    print("\n4. get_image_features実行:")
    
    with torch.no_grad():
        image_features = model.get_image_features(
            pixel_values,
            image_grid_thw
        )
    
    print(f"  結果type: {type(image_features)}")
    if isinstance(image_features, tuple):
        print(f"  tupleの長さ: {len(image_features)}")
        for i, feat in enumerate(image_features):
            if torch.is_tensor(feat):
                print(f"    要素{i}: shape={feat.shape}, dtype={feat.dtype}")
    elif torch.is_tensor(image_features):
        print(f"  Tensor shape: {image_features.shape}")
    
    # 5. フックを使って内部の特徴を取得
    print("\n5. フックを使った内部特徴の取得:")
    
    captured_features = {}
    hooks = []
    
    def capture_hook(name):
        def hook_fn(module, input, output):
            if torch.is_tensor(output):
                captured_features[name] = output.shape
            elif hasattr(output, 'last_hidden_state'):
                captured_features[name] = output.last_hidden_state.shape
            elif isinstance(output, tuple) and len(output) > 0:
                if torch.is_tensor(output[0]):
                    captured_features[name] = output[0].shape
        return hook_fn
    
    # patch_embedとmergerにフックを設定
    if hasattr(model.model.visual, 'patch_embed'):
        hook = model.model.visual.patch_embed.register_forward_hook(capture_hook('patch_embed'))
        hooks.append(hook)
    
    if hasattr(model.model.visual, 'merger'):
        hook = model.model.visual.merger.register_forward_hook(capture_hook('merger'))
        hooks.append(hook)
    
    # blocksの最後にもフック
    if hasattr(model.model.visual, 'blocks'):
        if len(model.model.visual.blocks) > 0:
            hook = model.model.visual.blocks[-1].register_forward_hook(capture_hook('last_block'))
            hooks.append(hook)
    
    # get_image_featuresを再実行
    with torch.no_grad():
        _ = model.get_image_features(pixel_values, image_grid_thw)
    
    # フック削除
    for hook in hooks:
        hook.remove()
    
    print("  キャプチャした特徴:")
    for name, shape in captured_features.items():
        print(f"    {name}: {shape}")
        if len(shape) >= 3:
            dim = shape[-1]
            if dim == 2352:
                print(f"      ✓ 2352次元を発見!")
            elif dim == 1176:
                print(f"      → 1176次元 (2352の半分)")
    
    # 6. visual.encoderを直接呼び出し
    print("\n6. encoder経由の特徴取得:")
    
    if hasattr(model.model.visual, 'blocks'):
        print("  blocks（エンコーダ）が存在")
        
        # patch_embedで処理
        with torch.no_grad():
            if pixel_values.dim() == 2:
                pixel_values = pixel_values.unsqueeze(0)
            
            # patch_embed
            if hasattr(model.model.visual, 'patch_embed'):
                x = model.model.visual.patch_embed(pixel_values, image_grid_thw)
                print(f"    patch_embed後: {x.shape}")
                
                # rotary_pos_embを適用
                if hasattr(model.model.visual, 'rotary_pos_emb'):
                    # ロータリー埋め込みの準備
                    grid_t = image_grid_thw[:, 0]
                    grid_h = image_grid_thw[:, 1] 
                    grid_w = image_grid_thw[:, 2]
                    
                    try:
                        # blocksを通す
                        for i, block in enumerate(model.model.visual.blocks):
                            x = block(x, grid_thw=image_grid_thw)
                            if i == 0 or i == len(model.model.visual.blocks) - 1:
                                print(f"    Block {i}後: {x[0].shape if isinstance(x, tuple) else x.shape}")
                    except Exception as e:
                        print(f"    Blocksエラー: {e}")
                
                # mergerを通す
                if hasattr(model.model.visual, 'merger'):
                    try:
                        merged = model.model.visual.merger(x, image_grid_thw)
                        print(f"    merger後: {merged.shape if torch.is_tensor(merged) else type(merged)}")
                        if torch.is_tensor(merged) and merged.shape[-1] == 2352:
                            print("      ✓✓✓ 2352次元特徴を取得成功!")
                    except Exception as e:
                        print(f"    Mergerエラー: {e}")
    
    print("\n" + "=" * 80)
    print("調査完了")
    print("=" * 80)


if __name__ == "__main__":
    investigate_get_image_features()