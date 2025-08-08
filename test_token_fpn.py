#!/usr/bin/env python3
"""
Token-FPN動作確認テスト
Token-FPNの統合と動的解像度対応の検証
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
import logging

# ロギング設定
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

print("=" * 80)
print("Token-FPN 動作確認テスト")
print("=" * 80)

def test_token_fpn():
    """Token-FPNの動作を確認"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"デバイス: {device}")
    
    # 1. モデルのロード
    logger.info("1. モデルとToken-FPNのセットアップ...")
    
    from src.config import LISAConfig
    from src.models.lisa_model import LISA_Model
    from src.utils import prepare_tokenizer_for_lisa
    
    # Config作成
    config = LISAConfig(
        qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        sam_model_name="./checkpoints/sam2.1_hiera_large.pt",
        use_token_fpn=True,  # Token-FPNを有効化
        fpn_layer_indices=[8, 16, 24, 31],
        use_dynamic_resolution=True,
        qwen_min_pixels=336*336,
        qwen_max_pixels=1344*1344
    )
    
    # トークナイザー準備
    tokenizer = prepare_tokenizer_for_lisa(
        model_name=config.qwen_model_name,
        seg_token=config.seg_token
    )
    
    # プロセッサ準備
    processor = AutoProcessor.from_pretrained(
        config.qwen_model_name,
        min_pixels=config.qwen_min_pixels,
        max_pixels=config.qwen_max_pixels
    )
    
    # SEGトークンを追加
    if config.seg_token not in processor.tokenizer.get_vocab():
        processor.tokenizer.add_special_tokens({"additional_special_tokens": [config.seg_token]})
    
    # モデル作成
    logger.info("LISA改モデルを作成中...")
    model = LISA_Model(config)
    model.set_tokenizer(tokenizer)
    model = model.to(device)
    model.eval()
    
    # 2. テスト画像の準備
    logger.info("\n2. テスト画像の準備...")
    
    # 448x448のテスト画像（動的解像度対応）
    image = Image.new('RGB', (448, 448))
    pixels = np.zeros((448, 448, 3), dtype=np.uint8)
    
    # グラデーションパターン
    for i in range(448):
        for j in range(448):
            pixels[i, j, 0] = int(255 * (i / 448))
            pixels[i, j, 1] = int(255 * (j / 448))
            pixels[i, j, 2] = int(255 * ((i + j) / 896))
    
    # 中央に白い円
    center = (224, 224)
    radius = 80
    for i in range(448):
        for j in range(448):
            if (i - center[0])**2 + (j - center[1])**2 < radius**2:
                pixels[i, j] = [255, 255, 255]
    
    image = Image.fromarray(pixels)
    image.save("test_token_fpn_pattern.png")
    logger.info("  テストパターン画像を保存: test_token_fpn_pattern.png")
    
    # 3. 入力データの準備
    logger.info("\n3. 入力データの準備...")
    
    # SEGトークンを含むプロンプト
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Segment the white circle in the image <SEG>"}
        ]
    }]
    
    # 入力の処理
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True
    )
    
    # デバイスに移動
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    
    logger.info(f"  input_ids shape: {inputs['input_ids'].shape}")
    logger.info(f"  pixel_values shape: {inputs['pixel_values'].shape}")
    if 'image_grid_thw' in inputs:
        logger.info(f"  image_grid_thw: {inputs['image_grid_thw']}")
    
    # 4. Token-FPNの動作確認
    logger.info("\n4. Token-FPNの動作確認...")
    
    with torch.no_grad():
        # Forward pass
        outputs = model(
            input_ids=inputs['input_ids'],
            pixel_values=inputs['pixel_values'],
            attention_mask=inputs.get('attention_mask'),
            image_grid_thw=inputs.get('image_grid_thw')
        )
        
        logger.info(f"  言語モデル出力形状: {outputs.logits.shape}")
        
        if outputs.mask_logits and len(outputs.mask_logits) > 0:
            logger.info(f"  マスク予測数: {len(outputs.mask_logits)}")
            for i, masks in enumerate(outputs.mask_logits):
                if masks is not None:
                    logger.info(f"    バッチ{i}: {len(masks)}個のマスク")
                    for j, mask in enumerate(masks):
                        logger.info(f"      マスク{j}形状: {mask.shape}")
        else:
            logger.info("  マスク予測なし（SEGトークンが処理されなかった可能性）")
        
        if outputs.seg_token_positions:
            logger.info(f"  SEGトークン位置: {outputs.seg_token_positions}")
    
    # 5. Token-FPNの中間特徴を確認
    logger.info("\n5. Token-FPNの中間特徴を確認...")
    
    if hasattr(model, 'token_fpn') and model.use_token_fpn:
        token_fpn = model.token_fpn
        
        # フックから取得した中間特徴を確認
        if token_fpn.intermediate_features:
            logger.info(f"  抽出された中間層数: {len(token_fpn.intermediate_features)}")
            
            for layer_name, features in token_fpn.intermediate_features.items():
                if features.dim() == 2:
                    features = features.unsqueeze(0)
                B, N, C = features.shape if features.dim() == 3 else (1, *features.shape)
                logger.info(f"    {layer_name}: [B={B}, N={N}, C={C}]")
                
                # トークン数から空間解像度を推定
                H_est = W_est = int(np.sqrt(N))
                if H_est * W_est == N:
                    logger.info(f"      推定空間解像度: {H_est}x{W_est}")
                else:
                    logger.info(f"      非正方形配置（N={N}）")
        else:
            logger.info("  中間特徴が抽出されていません（フックが動作していない可能性）")
    else:
        logger.info("  Token-FPNが無効化されています")
    
    # 6. 結果の可視化
    logger.info("\n6. 結果の可視化...")
    
    if outputs.mask_logits and len(outputs.mask_logits) > 0:
        for batch_idx, masks in enumerate(outputs.mask_logits):
            if masks is not None and len(masks) > 0:
                # 最初のマスクを可視化
                mask = masks[0]
                if mask.dim() == 3:
                    mask = mask.squeeze(0)
                
                # シグモイドを適用して確率に変換
                mask_prob = torch.sigmoid(mask).cpu().numpy()
                
                # 可視化
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # 元画像
                axes[0].imshow(image)
                axes[0].set_title("Original Image")
                axes[0].axis('off')
                
                # マスク予測
                axes[1].imshow(mask_prob, cmap='gray')
                axes[1].set_title("Predicted Mask")
                axes[1].axis('off')
                
                # オーバーレイ
                axes[2].imshow(image)
                axes[2].imshow(mask_prob, alpha=0.5, cmap='jet')
                axes[2].set_title("Overlay")
                axes[2].axis('off')
                
                plt.tight_layout()
                plt.savefig("token_fpn_results.png", dpi=150, bbox_inches='tight')
                plt.close()
                
                logger.info("  結果を保存: token_fpn_results.png")
                break
    
    # 7. パフォーマンス評価
    logger.info("\n7. パフォーマンス評価...")
    
    import time
    
    # Token-FPNありの推論時間
    start_time = time.time()
    with torch.no_grad():
        for _ in range(5):
            _ = model(
                input_ids=inputs['input_ids'],
                pixel_values=inputs['pixel_values'],
                attention_mask=inputs.get('attention_mask'),
                image_grid_thw=inputs.get('image_grid_thw')
            )
    fpn_time = (time.time() - start_time) / 5
    
    logger.info(f"  Token-FPN推論時間（平均）: {fpn_time:.3f}秒")
    
    # メモリ使用量
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        logger.info(f"  GPU メモリ使用量: {allocated:.2f}GB / {reserved:.2f}GB")
    
    logger.info("\n" + "=" * 80)
    logger.info("Token-FPN テスト完了")
    logger.info("=" * 80)
    
    return True


if __name__ == "__main__":
    try:
        success = test_token_fpn()
        if success:
            print("\n✅ Token-FPNテスト成功")
        else:
            print("\n❌ Token-FPNテスト失敗")
    except Exception as e:
        print(f"\n❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()