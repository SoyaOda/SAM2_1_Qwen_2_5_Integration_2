#!/usr/bin/env python3
"""
動的解像度機能のテストスクリプト
"""

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor
from src.config import LISAConfig
from src.utils import prepare_tokenizer_for_lisa
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dynamic_resolution():
    """動的解像度の設定と処理をテスト"""
    
    # 1. LISAConfigの動的解像度設定を確認
    config = LISAConfig()
    logger.info(f"動的解像度設定:")
    logger.info(f"  min_pixels: {config.qwen_min_pixels} ({int(np.sqrt(config.qwen_min_pixels))}×{int(np.sqrt(config.qwen_min_pixels))})")
    logger.info(f"  max_pixels: {config.qwen_max_pixels} ({int(np.sqrt(config.qwen_max_pixels))}×{int(np.sqrt(config.qwen_max_pixels))})")
    logger.info(f"  use_dynamic_resolution: {config.use_dynamic_resolution}")
    logger.info(f"  model_max_length: {config.model_max_length}")
    
    # 2. Processorの動的解像度設定をテスト
    logger.info("\nProcessorテスト:")
    tokenizer = prepare_tokenizer_for_lisa(
        model_name=config.qwen_model_name,
        seg_token=config.seg_token
    )
    
    processor = AutoProcessor.from_pretrained(
        config.qwen_model_name,
        min_pixels=config.qwen_min_pixels,
        max_pixels=config.qwen_max_pixels
    )
    processor.tokenizer = tokenizer
    
    # 3. 異なるサイズの画像でテスト
    test_sizes = [
        (224, 224, "Small"),
        (448, 448, "Medium"),
        (896, 896, "Large"),
        (1344, 896, "Wide"),
        (896, 1344, "Tall"),
        (2048, 2048, "Very Large"),
    ]
    
    for width, height, desc in test_sizes:
        logger.info(f"\n{desc} Image ({width}×{height}):")
        
        # ダミー画像を作成
        img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        image = Image.fromarray(img_array)
        
        # apply_chat_template
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image. <SEG>"}
            ]
        }]
        
        processed = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # 結果を表示
        pixel_values = processed['pixel_values'].squeeze(0)
        if 'image_grid_thw' in processed and processed['image_grid_thw'] is not None:
            grid = processed['image_grid_thw'].squeeze(0)
            logger.info(f"  Grid (T,H,W): {grid.tolist()}")
            H_grid = int(grid[1])
            W_grid = int(grid[2])
            logger.info(f"  Grid size: {H_grid}×{W_grid} = {H_grid*W_grid} patches")
        
        if pixel_values.dim() == 2:
            n_patches = pixel_values.shape[0]
            logger.info(f"  Pixel values: {pixel_values.shape} (2D format)")
            logger.info(f"  Compressed patches: {n_patches}")
            logger.info(f"  Final tokens (after 2×2 merge): {n_patches // 4}")
        else:
            logger.info(f"  Pixel values: {pixel_values.shape} (3D format)")
        
        # トークン数を計算
        input_ids = processed['input_ids'].squeeze(0)
        n_image_pads = (input_ids == 151655).sum().item()  # IMAGE_PAD_ID
        logger.info(f"  Image pad tokens: {n_image_pads}")
        logger.info(f"  Total tokens: {len(input_ids)}")

if __name__ == "__main__":
    test_dynamic_resolution()
    logger.info("\n✅ 動的解像度テスト完了!")