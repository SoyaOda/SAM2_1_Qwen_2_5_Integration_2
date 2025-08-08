#!/usr/bin/env python3
"""
Qwen2.5-VLの中間層特徴抽出テスト
Token-FPN実装の基盤となる特徴抽出の検証
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

print("=" * 80)
print("Qwen2.5-VL 中間層特徴抽出テスト")
print("=" * 80)

class QwenIntermediateFeatureExtractor:
    """Qwenの中間層特徴を抽出するクラス"""
    
    def __init__(self, model):
        self.model = model
        self.intermediate_features = {}
        self.hooks = []
        
    def hook_fn(self, layer_name):
        """フック関数を作成"""
        def hook(module, input, output):
            # outputは通常 (hidden_states, attention_weights) のタプル
            # hidden_statesのみを保存
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            self.intermediate_features[layer_name] = hidden_states.detach()
        return hook
    
    def register_hooks(self, layer_indices):
        """指定した層にフックを登録"""
        # Qwen2.5-VLのViTブロック構造を確認
        # model.model.visual がビジョンモジュール
        if hasattr(self.model.model, 'visual'):
            vision_model = self.model.model.visual
        else:
            raise ValueError("ビジョンモジュールが見つかりません")
        
        # ViTのブロックを探す
        if hasattr(vision_model, 'blocks'):
            blocks = vision_model.blocks
            print(f"Found blocks at: model.model.visual.blocks ({len(blocks)} blocks)")
        else:
            # Qwen2.5-VLは通常blocksを持つはず
            print("blocks属性が見つかりません。モジュール構造を確認中...")
            for name, module in vision_model.named_children():
                print(f"  visual.{name}: {type(module).__name__}")
                if 'ModuleList' in type(module).__name__:
                    try:
                        print(f"    -> {len(module)} modules")
                    except:
                        pass
            raise ValueError("ViTブロックが見つかりません")
        
        print(f"\n総ブロック数: {len(blocks)}")
        
        # 指定した層にフックを登録
        for idx in layer_indices:
            if idx < len(blocks):
                hook = blocks[idx].register_forward_hook(
                    self.hook_fn(f"block_{idx}")
                )
                self.hooks.append(hook)
                print(f"  Block {idx} にフック登録")
    
    def remove_hooks(self):
        """フックを削除"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.intermediate_features = {}
    
    def get_features(self):
        """抽出した特徴を取得"""
        return self.intermediate_features


def visualize_intermediate_features(features_dict, save_path="qwen_intermediate_features.png"):
    """中間層特徴を可視化"""
    num_layers = len(features_dict)
    if num_layers == 0:
        print("可視化する特徴がありません")
        return
        
    fig, axes = plt.subplots(2, num_layers, figsize=(4*num_layers, 8))
    
    if num_layers == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, (layer_name, features) in enumerate(features_dict.items()):
        # 特徴の形状を確認
        print(f"\n{layer_name} shape: {features.shape}")
        
        # [B, N, C] -> チャネル方向の統計
        B, N, C = features.shape
        
        # 平均活性化（トークン×チャネル）
        feat_mean = features[0].mean(dim=-1).cpu().numpy()  # [N]
        
        # 最大活性化
        feat_max = features[0].max(dim=-1)[0].cpu().numpy()  # [N]
        
        # トークンインデックス
        token_indices = np.arange(N)
        
        # 平均活性化の表示
        axes[0, idx].plot(token_indices, feat_mean)
        axes[0, idx].set_title(f"{layer_name}\nMean Activation")
        axes[0, idx].set_xlabel("Token Index")
        axes[0, idx].set_ylabel("Mean Activation")
        axes[0, idx].grid(True, alpha=0.3)
        
        # 最大活性化の表示
        axes[1, idx].plot(token_indices, feat_max)
        axes[1, idx].set_title(f"Max Activation")
        axes[1, idx].set_xlabel("Token Index")
        axes[1, idx].set_ylabel("Max Activation")
        axes[1, idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n特徴可視化を保存: {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nデバイス: {device}")
    
    # 1. モデルとプロセッサの準備
    print("\n1. Qwen2.5-VLモデルのロード...")
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=336*336,
        max_pixels=1344*1344
    )
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map=str(device),
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model.eval()
    
    # 2. テスト画像の準備
    print("\n2. テスト画像の準備...")
    # グラデーション付きテスト画像
    image = Image.new('RGB', (448, 448))
    pixels = np.zeros((448, 448, 3), dtype=np.uint8)
    
    # グラデーションパターン
    for i in range(448):
        for j in range(448):
            pixels[i, j, 0] = int(255 * (i / 448))  # 赤: 縦グラデーション
            pixels[i, j, 1] = int(255 * (j / 448))  # 緑: 横グラデーション
            # 青: チェッカーボード
            if (i // 56 + j // 56) % 2 == 0:
                pixels[i, j, 2] = 200
            else:
                pixels[i, j, 2] = 50
    
    # 中央に白い円
    center = (224, 224)
    radius = 80
    for i in range(448):
        for j in range(448):
            if (i - center[0])**2 + (j - center[1])**2 < radius**2:
                pixels[i, j] = [255, 255, 255]
    
    image = Image.fromarray(pixels)
    image.save("test_pattern.png")
    print("  テストパターン画像を保存: test_pattern.png")
    
    # 3. 入力の準備
    print("\n3. 入力データの準備...")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image."}
        ]
    }]
    
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )
    
    # デバイスに移動
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    
    print(f"  input_ids shape: {inputs['input_ids'].shape}")
    print(f"  pixel_values shape: {inputs['pixel_values'].shape}")
    if 'image_grid_thw' in inputs:
        print(f"  image_grid_thw: {inputs['image_grid_thw']}")
    
    # 4. モデル構造の確認
    print("\n4. モデル構造の確認...")
    vision_model = model.model.visual
    
    # ブロック数を確認
    if hasattr(vision_model, 'blocks'):
        num_blocks = len(vision_model.blocks)
        print(f"  ビジョンモデルのブロック数: {num_blocks}")
        
        # ブロックの詳細
        first_block = vision_model.blocks[0]
        print(f"  最初のブロックの構造:")
        for name, module in first_block.named_children():
            print(f"    {name}: {type(module).__name__}")
    else:
        print("  blocksが見つかりません")
        # 代替構造を探す
        for name, module in vision_model.named_children():
            if 'ModuleList' in type(module).__name__:
                print(f"  Found ModuleList at {name}: {len(module)} items")
                num_blocks = len(module)
                break
        else:
            num_blocks = 28  # デフォルト値
    
    # 5. 特徴抽出器のセットアップ
    print("\n5. 中間層特徴の抽出...")
    extractor = QwenIntermediateFeatureExtractor(model)
    
    # 抽出する層のインデックス（早期、中期、後期の層）
    layer_indices = [
        num_blocks // 4,      # 1/4地点
        num_blocks // 2,      # 1/2地点
        3 * num_blocks // 4,  # 3/4地点
        num_blocks - 1        # 最終層
    ]
    
    print(f"  抽出する層: {layer_indices}")
    
    # フックを登録
    try:
        extractor.register_hooks(layer_indices)
        
        with torch.no_grad():
            # Forward pass
            outputs = model(**inputs)
            
        # 抽出した特徴を取得
        features = extractor.get_features()
        
        print(f"\n抽出した特徴層数: {len(features)}")
        
        # 6. 特徴の分析
        print("\n6. 特徴の分析:")
        
        for layer_name, feat in features.items():
            print(f"\n{layer_name}:")
            print(f"  形状: {feat.shape}")
            print(f"  dtype: {feat.dtype}")
            print(f"  デバイス: {feat.device}")
            print(f"  最小値: {feat.min().item():.4f}")
            print(f"  最大値: {feat.max().item():.4f}")
            print(f"  平均値: {feat.mean().item():.4f}")
            print(f"  標準偏差: {feat.std().item():.4f}")
            
            # トークン次元を確認
            if feat.dim() == 3:
                B, N, C = feat.shape
                print(f"  バッチサイズ: {B}")
                print(f"  トークン数: {N}")
                print(f"  チャネル数: {C}")
            elif feat.dim() == 2:
                N, C = feat.shape
                B = 1
                print(f"  バッチサイズ: {B} (推定)")
                print(f"  トークン数: {N}")
                print(f"  チャネル数: {C}")
                # 3次元に変換して辞書を更新
                features[layer_name] = feat.unsqueeze(0)
        
        # 7. 特徴の可視化
        if features:
            print("\n7. 特徴の可視化...")
            visualize_intermediate_features(features, "qwen_intermediate_features.png")
        
        # 8. Token-FPN用の分析
        print("\n8. Token-FPN実装のための分析:")
        
        # チャネル数の確認
        channels = [feat.shape[2] if feat.dim() == 3 else feat.shape[1] for feat in features.values()]
        unique_channels = list(set(channels))
        
        if len(unique_channels) == 1:
            print(f"  ✅ 全層で同一チャネル数: {unique_channels[0]}")
            print(f"     -> 1x1 Convで256chに変換後、FPN構築可能")
        else:
            print(f"  ⚠️ 異なるチャネル数: {unique_channels}")
            print(f"     -> 各層で個別に1x1 Conv適用が必要")
        
        # トークン数の確認
        token_counts = [feat.shape[1] if feat.dim() == 3 else feat.shape[0] for feat in features.values()]
        unique_tokens = list(set(token_counts))
        
        if len(unique_tokens) == 1:
            print(f"  ✅ 全層で同一トークン数: {unique_tokens[0]}")
            print(f"     -> 擬似階層構築のためDownsample/Upsample必要")
            
            # 空間解像度の推定
            N = unique_tokens[0]
            # CLSトークンを除外
            spatial_tokens = N - 1 if N > 1 else N
            H = W = int(np.sqrt(spatial_tokens))
            if H * W == spatial_tokens:
                print(f"     -> 推定空間解像度: {H}x{W}")
            else:
                print(f"     -> 非正方形または不規則なトークン配置")
        else:
            print(f"  ⚠️ 異なるトークン数: {unique_tokens}")
        
        print("\n9. Token-FPN実装への推奨:")
        print("  1. 各中間層から特徴を抽出 ✅")
        print("  2. [B, N, C] -> [B, C, H, W]への変換が必要")
        print("  3. 1x1 Convで256chに統一")
        print("  4. Downsample/Upsampleで擬似階層を構築")
        print("  5. FPNのtop-down pathwayで融合")
        print("  6. stride-4/8特徴をSAM2.1 MaskDecoderへ供給")
        
    finally:
        # フックを削除
        extractor.remove_hooks()
    
    print("\n" + "=" * 80)
    print("テスト完了")
    print("=" * 80)


if __name__ == "__main__":
    main()