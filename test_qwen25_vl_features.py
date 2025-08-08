#!/usr/bin/env python3
"""
Qwen2.5-VL動的解像度機能の統合テスト
すべての新機能が正しく動作することを確認
"""

import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent))

from src.config import LISAConfig
from src.utils.resolution_utils import ResolutionBucketManager, QualityScoreCalculator, calculate_image_pad_tokens
from src.utils.multi_image_utils import prepare_multi_image_messages, calculate_multi_image_tokens, validate_multi_image_context
from transformers import AutoProcessor
from src.utils import prepare_tokenizer_for_lisa


def test_resolution_buckets():
    """解像度バケット戦略のテスト"""
    print("\n" + "="*60)
    print("1. 解像度バケット戦略のテスト")
    print("="*60)
    
    config = LISAConfig()
    bucket_manager = ResolutionBucketManager(config)
    
    # テスト画像
    test_cases = [
        ("Small", (224, 224)),
        ("Medium", (512, 768)),
        ("Large", (1024, 768)),
        ("Very Large", (2048, 1536)),
    ]
    
    for name, size in test_cases:
        img = Image.new('RGB', size, color='white')
        bucket = bucket_manager.get_bucket_for_image(img)
        optimal_size = bucket_manager.calculate_optimal_size(img, bucket)
        
        pixels, grid, tokens = bucket
        print(f"\n{name} {size}:")
        print(f"  → バケット: {int(np.sqrt(pixels))}×{int(np.sqrt(pixels))} ({tokens} tokens)")
        print(f"  → 最適サイズ: {optimal_size}")


def test_quality_score():
    """品質スコア機能のテスト"""
    print("\n" + "="*60)
    print("2. 品質スコア機能のテスト")
    print("="*60)
    
    config = LISAConfig()
    quality_calc = QualityScoreCalculator(config)
    
    # テスト画像の作成
    test_images = []
    
    # 高品質画像（シャープ、良好なコントラスト）
    high_quality = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
    high_quality[200:300, 200:300] = 255  # 高コントラストエリア
    test_images.append(("High Quality", Image.fromarray(high_quality)))
    
    # 低品質画像（ぼやけ、低コントラスト）
    low_quality = np.full((128, 128, 3), 128, dtype=np.uint8)
    low_quality += np.random.randint(-10, 10, (128, 128, 3), dtype=np.int16).astype(np.uint8)
    test_images.append(("Low Quality", Image.fromarray(low_quality)))
    
    # 中品質画像
    medium_quality = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    test_images.append(("Medium Quality", Image.fromarray(medium_quality)))
    
    for name, img in test_images:
        score = quality_calc.calculate_quality_score(img)
        weight = quality_calc.get_loss_weight(score)
        print(f"\n{name}:")
        print(f"  品質スコア: {score:.3f}")
        print(f"  損失重み: {weight:.3f}")
        print(f"  判定: {'高品質' if score >= 0.7 else '低品質'}")


def test_image_pad_tokens():
    """image_padトークンの計算テスト"""
    print("\n" + "="*60)
    print("3. image_padトークンの計算テスト")
    print("="*60)
    
    # テストケース
    test_cases = [
        ("Small", 576, 576),     # 同じサイズ（パディング不要）
        ("Medium", 576, 1536),   # 大きくパディング
        ("Large", 1536, 6144),   # さらに大きくパディング
    ]
    
    for name, original, padded in test_cases:
        added_tokens = calculate_image_pad_tokens(original, padded, merge_ratio=4)
        print(f"\n{name}:")
        print(f"  元のパッチ数: {original} ({original//4} tokens)")
        print(f"  パディング後: {padded} ({padded//4} tokens)")
        print(f"  追加image_pad: {added_tokens} tokens")


def test_multi_image_support():
    """マルチ画像サポートのテスト"""
    print("\n" + "="*60)
    print("4. マルチ画像サポートのテスト")
    print("="*60)
    
    # テスト画像
    images = [
        Image.new('RGB', (448, 448), color='red'),
        Image.new('RGB', (672, 672), color='green'),
        Image.new('RGB', (896, 896), color='blue'),
    ]
    
    prompts = [
        "Describe the first image.",
        "What's in the second image?",
        "Analyze the third image.",
    ]
    
    # メッセージフォーマット
    messages = prepare_multi_image_messages(images, prompts)
    print(f"\nメッセージ構造: {len(messages[0]['content'])} 要素")
    
    # トークン数計算
    image_tokens = calculate_multi_image_tokens(images)
    text_tokens = len(" ".join(prompts).split()) * 2  # 概算
    
    print(f"\n画像トークン数: {image_tokens}")
    print(f"テキストトークン数（概算）: {text_tokens}")
    print(f"合計: {image_tokens + text_tokens}")
    
    # コンテキスト長の検証
    valid, message = validate_multi_image_context(images, text_tokens)
    print(f"\nコンテキスト検証:")
    print(f"  {message}")


def test_full_pipeline():
    """完全なパイプラインのテスト"""
    print("\n" + "="*60)
    print("5. 完全なパイプラインテスト")
    print("="*60)
    
    config = LISAConfig()
    print(f"\n設定:")
    print(f"  動的解像度: {config.use_dynamic_resolution}")
    print(f"  品質スコア: {config.use_quality_score}")
    print(f"  min_pixels: {config.qwen_min_pixels} ({int(np.sqrt(config.qwen_min_pixels))}×{int(np.sqrt(config.qwen_min_pixels))})")
    print(f"  max_pixels: {config.qwen_max_pixels} ({int(np.sqrt(config.qwen_max_pixels))}×{int(np.sqrt(config.qwen_max_pixels))})")
    print(f"  max_length: {config.model_max_length}")
    
    # プロセッサの初期化
    tokenizer = prepare_tokenizer_for_lisa(
        model_name=config.qwen_model_name,
        seg_token=config.seg_token
    )
    
    processor = AutoProcessor.from_pretrained(
        config.qwen_model_name,
        min_pixels=config.qwen_min_pixels,
        max_pixels=config.qwen_max_pixels
    )
    
    # テスト画像
    img = Image.new('RGB', (600, 800), color='white')
    
    # 処理
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Segment the object. <SEG>"}
            ]
        }
    ]
    
    processed = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )
    
    # 結果の確認
    pixel_values = processed['pixel_values'].squeeze(0)
    input_ids = processed['input_ids'].squeeze(0)
    
    if pixel_values.dim() == 2:  # Flattened patches
        n_patches = pixel_values.shape[0]
        n_tokens = n_patches // 4
    else:  # 3D image
        _, h, w = pixel_values.shape
        n_patches = (h // 14) * (w // 14)
        n_tokens = n_patches // 4
    
    print(f"\n処理結果:")
    print(f"  元の画像: 600×800")
    print(f"  処理後: {pixel_values.shape}")
    print(f"  パッチ数: {n_patches}")
    print(f"  画像トークン: {n_tokens}")
    print(f"  総トークン: {len(input_ids)}")
    
    # image_padトークンの確認
    IMAGE_PAD_ID = 151655
    image_pad_count = (input_ids == IMAGE_PAD_ID).sum().item()
    print(f"  image_padトークン: {image_pad_count}")
    
    if image_pad_count == n_tokens:
        print("  ✅ image_padトークン数が正しい")
    else:
        print(f"  ⚠️ image_padトークン数が不一致（期待: {n_tokens}）")


def main():
    print("\n" + "="*80)
    print("Qwen2.5-VL 動的解像度機能 統合テスト")
    print("="*80)
    
    # 各テストの実行
    test_resolution_buckets()
    test_quality_score()
    test_image_pad_tokens()
    test_multi_image_support()
    test_full_pipeline()
    
    print("\n" + "="*80)
    print("✅ すべてのテスト完了")
    print("="*80)


if __name__ == "__main__":
    main()