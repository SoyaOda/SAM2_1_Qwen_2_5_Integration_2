"""
Qwen2.5-VLのマルチ画像対応ユーティリティ
"""

import torch
from typing import List, Dict, Any, Tuple
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def prepare_multi_image_messages(images: List[Image.Image], prompts: List[str]) -> List[Dict[str, Any]]:
    """
    複数画像用のメッセージフォーマットを準備
    
    Args:
        images: List of PIL Images
        prompts: List of text prompts (one per image)
        
    Returns:
        Messages in Qwen2.5-VL format
    """
    if len(images) != len(prompts):
        raise ValueError(f"Number of images ({len(images)}) must match prompts ({len(prompts)})")
    
    content = []
    for img, prompt in zip(images, prompts):
        content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})
    
    messages = [
        {
            "role": "user", 
            "content": content
        }
    ]
    
    return messages


def calculate_multi_image_tokens(images: List[Image.Image], patch_size: int = 14, merge_ratio: int = 4) -> int:
    """
    複数画像の合計トークン数を計算
    
    Args:
        images: List of PIL Images
        patch_size: ViT patch size (14 for Qwen2.5-VL)
        merge_ratio: Patch compression ratio (4:1)
        
    Returns:
        Total number of image tokens
    """
    total_tokens = 0
    
    for img in images:
        w, h = img.size
        # Round to patch_size multiple
        h_patches = (h + patch_size - 1) // patch_size
        w_patches = (w + patch_size - 1) // patch_size
        
        # Calculate tokens after compression
        raw_patches = h_patches * w_patches
        img_tokens = raw_patches // merge_ratio
        total_tokens += img_tokens
    
    return total_tokens


def validate_multi_image_context(
    images: List[Image.Image], 
    text_tokens: int,
    max_context: int = 16384,
    patch_size: int = 14,
    merge_ratio: int = 4
) -> Tuple[bool, str]:
    """
    マルチ画像のコンテキスト長を検証
    
    Args:
        images: List of PIL Images
        text_tokens: Estimated text token count
        max_context: Maximum context length
        patch_size: ViT patch size
        merge_ratio: Patch compression ratio
        
    Returns:
        (is_valid, message)
    """
    image_tokens = calculate_multi_image_tokens(images, patch_size, merge_ratio)
    total_tokens = image_tokens + text_tokens
    
    if total_tokens > max_context:
        message = (
            f"Context length exceeded: {total_tokens} > {max_context}\n"
            f"  Image tokens: {image_tokens} ({len(images)} images)\n"
            f"  Text tokens: {text_tokens}"
        )
        return False, message
    
    message = (
        f"Context within limit: {total_tokens}/{max_context}\n"
        f"  Image tokens: {image_tokens} ({len(images)} images)\n"
        f"  Text tokens: {text_tokens}"
    )
    return True, message


def group_images_by_resolution(
    images: List[Image.Image],
    tolerance: float = 0.1
) -> Dict[Tuple[int, int], List[int]]:
    """
    解像度が近い画像をグループ化（効率的なバッチ処理用）
    
    Args:
        images: List of PIL Images
        tolerance: Resolution similarity tolerance (0.1 = 10%)
        
    Returns:
        Dict mapping (approx_width, approx_height) to list of image indices
    """
    groups = {}
    
    for idx, img in enumerate(images):
        w, h = img.size
        
        # Find existing group with similar resolution
        found_group = False
        for (gw, gh), indices in groups.items():
            if abs(w - gw) / gw < tolerance and abs(h - gh) / gh < tolerance:
                indices.append(idx)
                found_group = True
                break
        
        # Create new group if no match found
        if not found_group:
            groups[(w, h)] = [idx]
    
    return groups


class MultiImageBatchProcessor:
    """マルチ画像バッチ処理のヘルパークラス"""
    
    def __init__(self, config):
        self.config = config
        self.max_context = config.model_max_length
        self.patch_size = config.qwen_patch_size
        self.merge_ratio = 4
        
    def can_batch_together(self, images: List[Image.Image], text_tokens: int) -> bool:
        """
        画像群が同一バッチで処理可能か判定
        
        Args:
            images: List of PIL Images
            text_tokens: Estimated text tokens
            
        Returns:
            True if can batch together
        """
        valid, _ = validate_multi_image_context(
            images, text_tokens, self.max_context, 
            self.patch_size, self.merge_ratio
        )
        return valid
    
    def split_into_valid_batches(
        self, 
        images: List[Image.Image], 
        text_tokens_per_image: int = 100
    ) -> List[List[int]]:
        """
        コンテキスト長制限内でバッチに分割
        
        Args:
            images: List of PIL Images
            text_tokens_per_image: Estimated text tokens per image
            
        Returns:
            List of image index lists for each batch
        """
        batches = []
        current_batch = []
        current_tokens = 0
        
        for idx, img in enumerate(images):
            img_tokens = calculate_multi_image_tokens([img], self.patch_size, self.merge_ratio)
            total_if_added = current_tokens + img_tokens + text_tokens_per_image
            
            if total_if_added > self.max_context:
                # Start new batch
                if current_batch:
                    batches.append(current_batch)
                current_batch = [idx]
                current_tokens = img_tokens + text_tokens_per_image
            else:
                # Add to current batch
                current_batch.append(idx)
                current_tokens = total_if_added
        
        # Add remaining batch
        if current_batch:
            batches.append(current_batch)
        
        return batches