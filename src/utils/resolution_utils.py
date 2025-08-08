"""
Qwen2.5-VL動的解像度のための解像度管理ユーティリティ
"""

import torch
import numpy as np
from typing import Tuple, List, Dict, Optional
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class ResolutionBucketManager:
    """解像度バケット管理クラス"""
    
    def __init__(self, config):
        """
        Args:
            config: LISAConfig object with resolution_buckets
        """
        self.config = config
        self.buckets = config.resolution_buckets if config.resolution_buckets else [
            (448 * 448, (32, 32), 256),   # Small
            (672 * 672, (48, 48), 576),   # Medium  
            (896 * 896, (64, 64), 1024),  # Large
        ]
        self.patch_size = config.qwen_patch_size  # 14
        self.merge_ratio = 4  # 4:1 patch compression
        
    def get_bucket_for_image(self, image: Image.Image) -> Tuple[int, Tuple[int, int], int]:
        """
        画像を最適なバケットに振り分け
        
        Args:
            image: PIL Image
            
        Returns:
            (target_pixels, (H_grid, W_grid), token_count)
        """
        w, h = image.size
        pixels = w * h
        
        # Find appropriate bucket
        for bucket_pixels, grid_size, tokens in self.buckets:
            if pixels <= bucket_pixels:
                return bucket_pixels, grid_size, tokens
        
        # Use largest bucket if image is too large
        return self.buckets[-1]
    
    def calculate_optimal_size(self, image: Image.Image, bucket: Tuple[int, Tuple[int, int], int]) -> Tuple[int, int]:
        """
        バケットに基づいて最適なリサイズサイズを計算
        
        Args:
            image: PIL Image
            bucket: (pixels, grid_size, tokens)
            
        Returns:
            (new_width, new_height) maintaining aspect ratio
        """
        w, h = image.size
        target_pixels, (target_h_grid, target_w_grid), _ = bucket
        
        # Calculate target dimensions based on grid
        target_h = target_h_grid * self.patch_size
        target_w = target_w_grid * self.patch_size
        
        # Maintain aspect ratio
        aspect_ratio = w / h
        
        if aspect_ratio > target_w / target_h:
            # Width-constrained
            new_w = target_w
            new_h = int(new_w / aspect_ratio)
            # Round to patch_size multiple
            new_h = (new_h // self.patch_size) * self.patch_size
        else:
            # Height-constrained
            new_h = target_h
            new_w = int(new_h * aspect_ratio)
            # Round to patch_size multiple
            new_w = (new_w // self.patch_size) * self.patch_size
        
        # Ensure minimum size
        new_w = max(new_w, self.patch_size * 2)
        new_h = max(new_h, self.patch_size * 2)
        
        return new_w, new_h
    
    def group_by_buckets(self, images: List[Image.Image]) -> Dict[int, List[int]]:
        """
        画像をバケットごとにグループ化（効率的なバッチ処理用）
        
        Args:
            images: List of PIL Images
            
        Returns:
            Dict mapping bucket_index to list of image indices
        """
        bucket_groups = {i: [] for i in range(len(self.buckets))}
        
        for idx, img in enumerate(images):
            bucket = self.get_bucket_for_image(img)
            # Find bucket index
            for bucket_idx, b in enumerate(self.buckets):
                if b == bucket:
                    bucket_groups[bucket_idx].append(idx)
                    break
        
        # Remove empty buckets
        bucket_groups = {k: v for k, v in bucket_groups.items() if v}
        
        return bucket_groups


class QualityScoreCalculator:
    """画像品質スコア計算クラス"""
    
    def __init__(self, config):
        self.config = config
        self.threshold = config.quality_score_threshold if hasattr(config, 'quality_score_threshold') else 0.7
        
    def calculate_quality_score(self, image: Image.Image) -> float:
        """
        画像の品質スコアを計算
        
        Args:
            image: PIL Image
            
        Returns:
            Quality score between 0 and 1
        """
        # Convert to numpy for analysis
        img_array = np.array(image)
        
        # Basic quality metrics
        scores = []
        
        # 1. Resolution score (normalized)
        w, h = image.size
        pixels = w * h
        # Normalize to [0, 1] based on expected range
        resolution_score = min(pixels / (1024 * 1024), 1.0)  # 1MP as reference
        scores.append(resolution_score)
        
        # 2. Sharpness score (based on gradient magnitude)
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
            
        # Calculate gradients
        gy, gx = np.gradient(gray.astype(np.float32))
        gnorm = np.sqrt(gx**2 + gy**2)
        sharpness = np.mean(gnorm) / 255.0  # Normalize
        sharpness_score = min(sharpness * 10, 1.0)  # Scale and cap at 1
        scores.append(sharpness_score)
        
        # 3. Contrast score
        if img_array.max() > img_array.min():
            contrast = (img_array.std() / 127.5)  # Normalized standard deviation
            contrast_score = min(contrast, 1.0)
        else:
            contrast_score = 0.0
        scores.append(contrast_score)
        
        # 4. Brightness score (penalize too dark or too bright)
        mean_brightness = img_array.mean() / 255.0
        brightness_score = 1.0 - abs(mean_brightness - 0.5) * 2  # Peak at 0.5
        scores.append(brightness_score)
        
        # Weighted average
        weights = [0.3, 0.3, 0.2, 0.2]  # Resolution, sharpness, contrast, brightness
        quality_score = sum(s * w for s, w in zip(scores, weights))
        
        return quality_score
    
    def get_loss_weight(self, quality_score: float) -> float:
        """
        品質スコアに基づいて損失の重みを計算
        
        Args:
            quality_score: Quality score between 0 and 1
            
        Returns:
            Loss weight
        """
        if quality_score < self.threshold:
            # Reduce weight for low-quality images
            return 0.5 + 0.5 * (quality_score / self.threshold)
        else:
            # Full weight for high-quality images
            return 1.0


def calculate_image_pad_tokens(original_patches: int, padded_patches: int, merge_ratio: int = 4) -> int:
    """
    パディング後に必要な<image_pad>トークン数を計算
    
    Args:
        original_patches: 元のパッチ数
        padded_patches: パディング後のパッチ数
        merge_ratio: パッチ圧縮比率（デフォルト4:1）
        
    Returns:
        追加すべき<image_pad>トークン数
    """
    added_patches = padded_patches - original_patches
    # 4パッチ = 1トークン
    added_tokens = added_patches // merge_ratio
    return added_tokens


def apply_right_bottom_padding(image_tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """
    右下パディング戦略を適用（Qwen2.5-VL公式推奨）
    
    Args:
        image_tensor: (C, H, W) or (B, C, H, W)
        target_h: Target height
        target_w: Target width
        
    Returns:
        Padded tensor
    """
    if image_tensor.dim() == 3:
        c, h, w = image_tensor.shape
        pad_h = target_h - h
        pad_w = target_w - w
        
        # Right-bottom padding (left=0, right=pad_w, top=0, bottom=pad_h)
        padded = torch.nn.functional.pad(image_tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
    elif image_tensor.dim() == 4:
        b, c, h, w = image_tensor.shape
        pad_h = target_h - h
        pad_w = target_w - w
        
        # Right-bottom padding
        padded = torch.nn.functional.pad(image_tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)
    else:
        raise ValueError(f"Unexpected tensor dimension: {image_tensor.dim()}")
    
    return padded


def validate_token_count(image_tokens: int, text_tokens: int, max_tokens: int = 16384) -> bool:
    """
    トークン数が上限を超えていないか検証
    
    Args:
        image_tokens: Number of image tokens
        text_tokens: Number of text tokens  
        max_tokens: Maximum allowed tokens
        
    Returns:
        True if within limit, False otherwise
    """
    total = image_tokens + text_tokens
    if total > max_tokens:
        logger.warning(f"Token count ({total}) exceeds limit ({max_tokens})")
        logger.warning(f"  Image tokens: {image_tokens}")
        logger.warning(f"  Text tokens: {text_tokens}")
        return False
    return True