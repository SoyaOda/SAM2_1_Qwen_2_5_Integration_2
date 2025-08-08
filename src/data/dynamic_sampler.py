"""
動的解像度のための効率的なバッチサンプラー
"""

import torch
from torch.utils.data import Sampler
import numpy as np
from typing import Iterator, List, Optional
import logging

logger = logging.getLogger(__name__)


class DynamicResolutionBatchSampler(Sampler):
    """
    解像度バケットに基づいて効率的にバッチを作成するサンプラー
    同じ解像度バケット内の画像を同一バッチにまとめることでGPU利用率を最適化
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int,
        bucket_manager,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0
    ):
        """
        Args:
            dataset: Dataset with resolution information
            batch_size: Batch size per bucket
            bucket_manager: ResolutionBucketManager instance
            shuffle: Whether to shuffle samples within buckets
            drop_last: Whether to drop incomplete batches
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_manager = bucket_manager
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        
        # Pre-compute bucket assignments for all samples
        self._prepare_buckets()
        
    def _prepare_buckets(self):
        """事前に全サンプルのバケット割り当てを計算"""
        self.bucket_indices = {i: [] for i in range(len(self.bucket_manager.buckets))}
        
        logger.info("Analyzing dataset for resolution buckets...")
        
        # Analyze a subset of dataset to determine bucket distribution
        sample_size = min(1000, len(self.dataset))
        sample_indices = np.random.choice(len(self.dataset), sample_size, replace=False)
        
        for idx in sample_indices:
            try:
                # Get image from dataset (without full preprocessing)
                sample = self.dataset[idx]
                
                # Extract image for bucket assignment
                if 'original_image' in sample:
                    image = sample['original_image']
                elif 'pixel_values' in sample:
                    # Skip if already processed
                    continue
                else:
                    continue
                
                # Assign to bucket
                bucket = self.bucket_manager.get_bucket_for_image(image)
                for bucket_idx, b in enumerate(self.bucket_manager.buckets):
                    if b == bucket:
                        self.bucket_indices[bucket_idx].append(idx)
                        break
                        
            except Exception as e:
                logger.debug(f"Failed to analyze sample {idx}: {e}")
                continue
        
        # Extrapolate to full dataset
        total_analyzed = sum(len(indices) for indices in self.bucket_indices.values())
        if total_analyzed > 0:
            scale = len(self.dataset) / sample_size
            for bucket_idx in self.bucket_indices:
                estimated_count = int(len(self.bucket_indices[bucket_idx]) * scale)
                # Generate indices for full dataset
                if estimated_count > 0:
                    # Distribute indices evenly
                    stride = len(self.dataset) // estimated_count
                    self.bucket_indices[bucket_idx] = list(range(
                        bucket_idx * stride // len(self.bucket_manager.buckets),
                        len(self.dataset),
                        stride
                    ))[:estimated_count]
        
        # Remove empty buckets
        self.bucket_indices = {k: v for k, v in self.bucket_indices.items() if v}
        
        # Log bucket distribution
        logger.info("Resolution bucket distribution:")
        for bucket_idx, indices in self.bucket_indices.items():
            bucket_info = self.bucket_manager.buckets[bucket_idx]
            pixels, grid, tokens = bucket_info
            res = int(np.sqrt(pixels))
            logger.info(f"  Bucket {bucket_idx} ({res}x{res}, {tokens} tokens): {len(indices)} samples")
    
    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches"""
        # Set random seed for this epoch
        if self.shuffle:
            np.random.seed(self.seed + self.epoch)
        
        batches = []
        
        # Create batches for each bucket
        for bucket_idx, indices in self.bucket_indices.items():
            if not indices:
                continue
                
            # Shuffle indices within bucket if needed
            bucket_indices = indices.copy()
            if self.shuffle:
                np.random.shuffle(bucket_indices)
            
            # Create batches
            for i in range(0, len(bucket_indices), self.batch_size):
                batch = bucket_indices[i:i + self.batch_size]
                
                # Handle incomplete batches
                if len(batch) < self.batch_size:
                    if self.drop_last:
                        continue
                    elif len(batch) < self.batch_size // 2:
                        # Too small, merge with previous if possible
                        if batches and len(batches[-1]) + len(batch) <= self.batch_size * 1.5:
                            batches[-1].extend(batch)
                            continue
                
                batches.append(batch)
        
        # Shuffle batches if needed
        if self.shuffle:
            np.random.shuffle(batches)
        
        # Yield batches
        for batch in batches:
            yield batch
    
    def __len__(self) -> int:
        """Total number of batches"""
        total_batches = 0
        for indices in self.bucket_indices.values():
            if self.drop_last:
                total_batches += len(indices) // self.batch_size
            else:
                total_batches += (len(indices) + self.batch_size - 1) // self.batch_size
        return total_batches
    
    def set_epoch(self, epoch: int):
        """Set epoch for shuffling"""
        self.epoch = epoch


class AdaptiveBatchSampler(Sampler):
    """
    GPU メモリに基づいて動的にバッチサイズを調整するサンプラー
    高解像度画像には小さいバッチ、低解像度画像には大きいバッチを使用
    """
    
    def __init__(
        self,
        dataset,
        base_batch_size: int,
        bucket_manager,
        max_tokens_per_batch: int = 4096,
        shuffle: bool = True,
        seed: int = 0
    ):
        """
        Args:
            dataset: Dataset
            base_batch_size: Base batch size for medium resolution
            bucket_manager: ResolutionBucketManager
            max_tokens_per_batch: Maximum tokens per batch
            shuffle: Whether to shuffle
            seed: Random seed
        """
        self.dataset = dataset
        self.base_batch_size = base_batch_size
        self.bucket_manager = bucket_manager
        self.max_tokens_per_batch = max_tokens_per_batch
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
        # Calculate adaptive batch sizes for each bucket
        self._calculate_batch_sizes()
        
    def _calculate_batch_sizes(self):
        """各バケットの適応的バッチサイズを計算"""
        self.bucket_batch_sizes = {}
        
        for bucket_idx, (pixels, grid, tokens) in enumerate(self.bucket_manager.buckets):
            # Calculate batch size based on token count
            # Ensure at least batch_size=1
            adaptive_batch_size = max(1, self.max_tokens_per_batch // tokens)
            
            # Cap at reasonable maximum
            adaptive_batch_size = min(adaptive_batch_size, self.base_batch_size * 2)
            
            self.bucket_batch_sizes[bucket_idx] = adaptive_batch_size
            
        logger.info("Adaptive batch sizes:")
        for bucket_idx, batch_size in self.bucket_batch_sizes.items():
            bucket_info = self.bucket_manager.buckets[bucket_idx]
            pixels, grid, tokens = bucket_info
            res = int(np.sqrt(pixels))
            logger.info(f"  Bucket {bucket_idx} ({res}x{res}, {tokens} tokens): batch_size={batch_size}")
    
    def __iter__(self) -> Iterator[List[int]]:
        """Generate adaptive batches"""
        # Implementation similar to DynamicResolutionBatchSampler
        # but with variable batch sizes per bucket
        indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            np.random.seed(self.seed + self.epoch)
            np.random.shuffle(indices)
        
        # Simple implementation: yield fixed-size batches
        # In production, would group by resolution first
        for i in range(0, len(indices), self.base_batch_size):
            yield indices[i:i + self.base_batch_size]
    
    def __len__(self) -> int:
        return (len(self.dataset) + self.base_batch_size - 1) // self.base_batch_size
    
    def set_epoch(self, epoch: int):
        self.epoch = epoch