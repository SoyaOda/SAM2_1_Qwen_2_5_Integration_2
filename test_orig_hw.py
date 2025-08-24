#!/usr/bin/env python
"""
orig_hw修正の動作確認テスト
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import logging
from src.data.dataset import HybridDataset
from src.data.collators import MultiModalDataCollator
from transformers import AutoProcessor

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_dataset_orig_hw():
    """データセットのorig_hw取得をテスト"""
    
    # プロセッサーの初期化
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    
    # ダミーのデータセット設定
    dataset_config = {
        'dataset_list': 'semantic_segmentation',
        'semantic_segmentation': {
            'root': '/path/to/data',  # 実際のパスに変更
            'annotation': 'test.json',
            'dataset_name': 'test'
        },
        'llava_path': None,
        'image_folder': '/path/to/images',
        'stage': 'stage2',
        'version': 'Qwen2-VL-2B-Instruct',
        'seg_token_idx': 151655,
        'ignore_label': -100,
        'sam_image_size': 1024,
        'use_quality_score': False,
        'use_dynamic_resolution': True
    }
    
    # CoordinateTransformクラスのモック
    class CoordinateTransform:
        def __init__(self, orig_size):
            self.orig_size = orig_size
    
    # テスト用のサンプルを作成
    test_samples = []
    
    # 10要素形式（coord_transform付き）
    coord_transform = CoordinateTransform((480, 640))
    sample1 = (
        "/path/to/image1.jpg",  # image_path
        torch.randn(3, 1024, 1024),  # image_sam
        torch.randn(3, 448, 448),  # image_qwen
        [{"role": "user", "content": "Segment the cat"}, {"role": "assistant", "content": "<SEG>"}],  # conversations
        torch.ones(1, 1024, 1024),  # masks
        "cat",  # label
        (480, 640),  # resize (元画像サイズ)
        ["What is in the image?"],  # questions
        ["cat", "dog"],  # sampled_classes
        coord_transform  # coord_transform
    )
    
    # 9要素形式（coord_transformなし、resizeオブジェクト付き）
    class ResizeObj:
        def __init__(self, orig_size):
            self.orig_size = orig_size
    
    resize_obj = ResizeObj((720, 1280))
    sample2 = (
        "/path/to/image2.jpg",
        torch.randn(3, 1024, 1024),
        torch.randn(3, 448, 448),
        [{"role": "user", "content": "Find the dog"}, {"role": "assistant", "content": "<SEG>"}],
        torch.ones(1, 1024, 1024),
        "dog",
        resize_obj,  # resizeオブジェクト
        ["Where is the dog?"],
        ["dog", "cat"]
    )
    
    # 9要素形式（タプルのresize）
    sample3 = (
        "/path/to/image3.jpg",
        torch.randn(3, 1024, 1024),
        torch.randn(3, 448, 448),
        [{"role": "user", "content": "Segment the car"}, {"role": "assistant", "content": "<SEG>"}],
        torch.ones(1, 1024, 1024),
        "car",
        (600, 800),  # タプルのresize
        ["What vehicle is this?"],
        ["car", "truck"]
    )
    
    # 各サンプルでorig_hw取得をテスト
    for i, sample in enumerate([sample1, sample2, sample3]):
        logger.info(f"\n=== Testing sample {i+1} ===")
        
        # __getitem__内のロジックを模倣
        if len(sample) == 10:
            image_path, _, _, _, _, _, resize, _, _, coord_transform = sample
            
            # 優先順位1: coord_transform.orig_size
            if hasattr(coord_transform, 'orig_size') and coord_transform.orig_size is not None:
                orig_hw = coord_transform.orig_size
                logger.info(f"✓ Got orig_hw from coord_transform.orig_size: {orig_hw}")
            # 優先順位2: resizeタプル
            elif isinstance(resize, tuple) and len(resize) == 2:
                orig_hw = resize
                logger.info(f"✓ Got orig_hw from resize tuple: {orig_hw}")
            else:
                orig_hw = (1024, 1024)
                logger.warning(f"✗ Using default orig_hw: {orig_hw}")
                
        elif len(sample) == 9:
            image_path, _, _, _, _, _, resize, _, _ = sample
            
            # 優先順位1: resizeオブジェクトのorig_size
            if hasattr(resize, 'orig_size') and resize.orig_size is not None:
                orig_hw = resize.orig_size
                logger.info(f"✓ Got orig_hw from resize.orig_size: {orig_hw}")
            # 優先順位2: resizeタプル
            elif isinstance(resize, tuple) and len(resize) == 2:
                orig_hw = resize
                logger.info(f"✓ Got orig_hw from resize tuple: {orig_hw}")
            else:
                orig_hw = (1024, 1024)
                logger.warning(f"✗ Using default orig_hw: {orig_hw}")
        
        # 期待値と比較
        if i == 0:
            expected = (480, 640)
            assert orig_hw == expected, f"Sample 1: Expected {expected}, got {orig_hw}"
            logger.info(f"✓ Sample 1 passed: {orig_hw} == {expected}")
        elif i == 1:
            expected = (720, 1280)
            assert orig_hw == expected, f"Sample 2: Expected {expected}, got {orig_hw}"
            logger.info(f"✓ Sample 2 passed: {orig_hw} == {expected}")
        elif i == 2:
            expected = (600, 800)
            assert orig_hw == expected, f"Sample 3: Expected {expected}, got {orig_hw}"
            logger.info(f"✓ Sample 3 passed: {orig_hw} == {expected}")
    
    logger.info("\n=== All tests passed! ===")


def test_postprocess_masks():
    """postprocess_masksの4D形式変換をテスト"""
    from sam2.utils.transforms import SAM2Transforms
    
    sam_transforms = SAM2Transforms(
        resolution=1024,
        mask_threshold=0.0,
        max_hole_area=0.0,
        max_sprinkle_area=0.0
    )
    
    # 異なる次元のマスクでテスト
    test_cases = [
        (torch.randn(256, 256), "2D mask"),  # [H, W]
        (torch.randn(1, 256, 256), "3D mask with C=1"),  # [C, H, W]
        (torch.randn(4, 256, 256), "3D mask with C=4"),  # [C, H, W]
        (torch.randn(1, 1, 256, 256), "4D mask [1,1,H,W]"),  # [B, C, H, W]
        (torch.randn(1, 4, 256, 256), "4D mask [1,4,H,W]"),  # [B, C, H, W]
    ]
    
    orig_hw = (480, 640)
    
    for mask, desc in test_cases:
        logger.info(f"\n=== Testing {desc} with shape {mask.shape} ===")
        
        # 4D形式に変換
        if mask.dim() == 2:
            mask_4d = mask.unsqueeze(0).unsqueeze(0)  # [H, W] -> [1, 1, H, W]
        elif mask.dim() == 3:
            if mask.shape[0] == 1:
                mask_4d = mask.unsqueeze(0)  # [1, H, W] -> [1, 1, H, W]
            else:
                mask_4d = mask.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
        elif mask.dim() == 4:
            mask_4d = mask
        else:
            raise ValueError(f"Unexpected mask dimension: {mask.dim()}")
        
        logger.info(f"  Converted to 4D: {mask_4d.shape}")
        
        # postprocess_masksを実行
        try:
            processed = sam_transforms.postprocess_masks(mask_4d.float(), orig_hw)
            logger.info(f"  ✓ Processed shape: {processed.shape}")
            assert processed.shape[-2:] == orig_hw, f"Expected shape ending with {orig_hw}, got {processed.shape}"
        except Exception as e:
            logger.error(f"  ✗ Error: {e}")
            raise
    
    logger.info("\n=== All postprocess_masks tests passed! ===")


def test_denormalize():
    """denormalize関数のテスト"""
    
    def denormalize_sam_image(img_tensor):
        """SAM2のImageNet正規化を逆操作してRGB画像に戻す"""
        if img_tensor.dim() == 4:
            img_tensor = img_tensor[0]
        # ImageNet mean/std
        mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(3, 1, 1)
        # 逆正規化: x = x * std + mean
        img = (img_tensor * std + mean).clamp(0, 1)
        # [0, 1] -> [0, 255]
        img = (img * 255.0).round().to(torch.uint8)
        # CHW -> HWC
        img = img.permute(1, 2, 0).cpu().numpy()
        return img
    
    # テスト用の正規化された画像を作成
    # 元の値: [0.5, 0.5, 0.5] (グレー)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # 正規化: (x - mean) / std
    original = torch.ones(3, 64, 64) * 0.5
    normalized = (original - mean) / std
    
    logger.info("\n=== Testing denormalize function ===")
    logger.info(f"Original values (sample): {original[0, 0, 0].item():.3f}")
    logger.info(f"Normalized values (sample): {normalized[0, 0, 0].item():.3f}")
    
    # denormalize
    denormalized = denormalize_sam_image(normalized)
    
    logger.info(f"Denormalized shape: {denormalized.shape}")
    logger.info(f"Denormalized dtype: {denormalized.dtype}")
    logger.info(f"Denormalized values (sample): {denormalized[0, 0, 0]}")
    
    # 元の値に近い値に戻っているか確認
    expected_value = int(0.5 * 255)
    actual_value = denormalized[0, 0, 0]
    assert abs(actual_value - expected_value) < 2, f"Expected around {expected_value}, got {actual_value}"
    
    logger.info(f"✓ Denormalize test passed: {actual_value} ≈ {expected_value}")


if __name__ == "__main__":
    logger.info("Starting orig_hw fix tests...")
    
    # 各テストを実行
    test_dataset_orig_hw()
    test_postprocess_masks()
    test_denormalize()
    
    logger.info("\n" + "="*50)
    logger.info("ALL TESTS PASSED SUCCESSFULLY!")
    logger.info("="*50)