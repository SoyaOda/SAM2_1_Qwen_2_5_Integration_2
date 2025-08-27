"""
Data collators for LISA改 (LISA-Kai) training
Handles batching of mixed data types
"""
import torch
from typing import Dict, List, Optional, Union
from dataclasses import dataclass


class MultiModalDataCollator:
    """
    Data collator for multimodal data (image + text + optional masks)
    Handles batching for LISA training
    """
    
    def __init__(
        self,
        tokenizer,
        max_length: int = 512,
        padding: str = "longest",
        return_tensors: str = "pt",
        config=None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.return_tensors = return_tensors
        self.config = config
    
    def __call__(self, features: List[Dict[str, any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of multimodal samples - test_a1準拠版
        
        Args:
            features: List of sample dictionaries containing:
                - pixel_values: Image tensors (2D or 3D format)
                - input_ids: Token IDs
                - labels: Target token IDs
                - attention_mask: Attention masks
                - ground_truth_mask/mask_labels (optional): Segmentation masks
                - image_grid_thw: Grid dimensions for dynamic resolution
                - original_image: PIL images for visualization
        
        Returns:
            Batch dictionary with properly shaped tensors
        """
        batch = {}
        
        # Handle image features with dynamic resolution (Qwen2.5-VL requirement)
        if 'pixel_values' in features[0]:
            # Check if pixel_values are already flattened (Qwen2.5-VL format)
            first_pix = features[0]['pixel_values']
            
            if first_pix.dim() == 2:
                # 2D patch format from apply_chat_template with tokenize=True
                # Format: (N_patch, D_v) where N_patch = H_grid × W_grid
                
                # Collect all pixel_values and grid info
                all_pixel_values = []
                all_image_grids = []
                for f in features:
                    all_pixel_values.append(f['pixel_values'])
                    if 'image_grid_thw' in f and f['image_grid_thw'] is not None:
                        all_image_grids.append(f['image_grid_thw'])
                    else:
                        # 必須: image_grid_thwが必要
                        raise ValueError("image_grid_thw is required for 2D pixel_values format")
                
                # Find max grid dimensions
                max_raw_patches = 0
                all_raw_patches = []
                
                for i, pv in enumerate(all_pixel_values):
                    # pixel_values contains RAW patches (before PatchMerge)
                    n_raw_patches = pv.shape[0]
                    all_raw_patches.append(n_raw_patches)
                    
                    grid = all_image_grids[i]
                    H_grid_raw = int(grid[1])
                    W_grid_raw = int(grid[2])
                    
                    # Verify RAW patches consistency
                    expected_raw_patches = H_grid_raw * W_grid_raw
                    
                    if expected_raw_patches != n_raw_patches:
                        raise ValueError(
                            f"Sample {i}: Patch count mismatch! "
                            f"Grid: {H_grid_raw}×{W_grid_raw} = {expected_raw_patches} RAW patches. "
                            f"Actual: {n_raw_patches} patches in pixel_values."
                        )
                    
                    max_raw_patches = max(max_raw_patches, n_raw_patches)
                
                # Calculate proper grid dimensions using square policy
                import math
                
                # Square policy: make it as square as possible
                grid_size = math.ceil(math.sqrt(max_raw_patches))
                # Round up to nearest even number (REQUIRED for 2x2 PatchMerge)
                H_grid_max = W_grid_max = (grid_size + 1) // 2 * 2
                
                # Actual padded patch count (must be H*W exactly)
                N_padded_raw = H_grid_max * W_grid_max
                
                # Image tokens after PatchMerge
                max_img_tokens = N_padded_raw // 4
                
                # Debug output for batch processing (disabled for performance)
                # if len(all_image_grids) > 0:
                #     print(f"[Batch Collator] Max RAW patches: {max_raw_patches}")
                #     print(f"  Target grid (even): {H_grid_max}×{W_grid_max} = {N_padded_raw} patches")
                #     print(f"  Image tokens after PatchMerge: {max_img_tokens}")
                #     for i, n_patches in enumerate(all_raw_patches):
                #         grid = all_image_grids[i]
                #         print(f"  Sample {i}: {n_patches} patches (grid: {int(grid[1])}×{int(grid[2])})")
                
                # IMAGE_PAD_ID for Qwen2.5-VL
                IMAGE_PAD_ID = 151655
                
                # Process each sample
                padded_pixel_values = []
                padded_input_ids = []
                padded_attention_masks = []
                padded_labels = []
                new_image_grid_thw = []
                
                # Debug output removed for performance
                
                for i, f in enumerate(features):
                    pv = f['pixel_values']
                    input_ids = f['input_ids']
                    attention_mask = f.get('attention_mask', torch.ones_like(input_ids))
                    labels = f.get('labels', torch.full_like(input_ids, -100))
                    
                    # Verify original grid info
                    orig_grid = f['image_grid_thw']
                    orig_H_raw = int(orig_grid[1])
                    orig_W_raw = int(orig_grid[2])
                    expected_raw_patches = orig_H_raw * orig_W_raw
                    
                    if pv.shape[0] != expected_raw_patches:
                        raise ValueError(
                            f"Sample {i}: Inconsistent data! "
                            f"Grid: {orig_H_raw}×{orig_W_raw} = {expected_raw_patches} RAW patches. "
                            f"Actual: {pv.shape[0]} patches in pixel_values."
                        )
                    
                    # 1. Pad pixel_values to N_padded_raw
                    current_patches = pv.shape[0]
                    pad_len = N_padded_raw - current_patches
                    
                    if pad_len > 0:
                        # Zero padding for pixel values
                        padding = torch.zeros(pad_len, pv.shape[1], dtype=pv.dtype, device=pv.device)
                        pv_padded = torch.cat([pv, padding], dim=0)
                    else:
                        pv_padded = pv[:N_padded_raw]  # Truncate if needed
                    
                    padded_pixel_values.append(pv_padded)
                    
                    # 2. Handle input_ids padding - need to adjust image_pad tokens
                    # Count existing image_pad tokens first
                    existing_image_pads = (input_ids == IMAGE_PAD_ID).sum().item()
                    
                    # The correct number should match the image tokens after PatchMerge
                    target_image_pads = N_padded_raw // 4  # This matches padded pixel_values
                    
                    # Debug: Sample padding info
                    
                    # Calculate how many to add or remove
                    image_pad_diff = target_image_pads - existing_image_pads
                    
                    if image_pad_diff > 0:
                        # Need to add more image_pad tokens
                        image_pad_mask = (input_ids == IMAGE_PAD_ID)
                        
                        if image_pad_mask.any():
                            # Get existing image_pad positions
                            pad_indices = image_pad_mask.nonzero(as_tuple=True)[0]
                            last_pad_idx = pad_indices[-1].item()
                            
                            # Insert additional image_pad tokens after the last existing one
                            ids_padded = torch.cat([
                                input_ids[:last_pad_idx + 1],  # Up to and including last image_pad
                                input_ids.new_full((image_pad_diff,), IMAGE_PAD_ID),  # Additional image_pads
                                input_ids[last_pad_idx + 1:]  # Rest of the sequence
                            ])
                            
                            # 同様にlabelsとattention_maskも調整
                            labels_padded = torch.cat([
                                labels[:last_pad_idx + 1],
                                labels.new_full((image_pad_diff,), -100),  # 追加分は-100
                                labels[last_pad_idx + 1:]
                            ])
                            
                            mask_padded = torch.cat([
                                attention_mask[:last_pad_idx + 1],
                                attention_mask.new_ones((image_pad_diff,)),
                                attention_mask[last_pad_idx + 1:]
                            ])
                        else:
                            # No existing image_pad tokens - shouldn't happen
                            print(f"Warning: No existing image_pad tokens found in sample {i}")
                            ids_padded = input_ids
                            labels_padded = labels
                            mask_padded = attention_mask
                    elif image_pad_diff < 0:
                        # Need to remove excess image_pad tokens
                        image_pad_mask = (input_ids == IMAGE_PAD_ID)
                        if image_pad_mask.any():
                            pad_indices = image_pad_mask.nonzero(as_tuple=True)[0]
                            # Keep only the target number of image_pads
                            keep_indices = []
                            image_pad_count = 0
                            for idx in range(len(input_ids)):
                                if input_ids[idx] == IMAGE_PAD_ID:
                                    if image_pad_count < target_image_pads:
                                        keep_indices.append(idx)
                                        image_pad_count += 1
                                else:
                                    keep_indices.append(idx)
                            ids_padded = input_ids[keep_indices]
                            labels_padded = labels[keep_indices]
                            mask_padded = attention_mask[keep_indices]
                        else:
                            ids_padded = input_ids
                            labels_padded = labels
                            mask_padded = attention_mask
                    else:
                        # Correct number already
                        ids_padded = input_ids
                        labels_padded = labels
                        mask_padded = attention_mask
                    
                    padded_input_ids.append(ids_padded)
                    padded_labels.append(labels_padded)
                    padded_attention_masks.append(mask_padded)
                    
                    # 3. Update grid to unified dimensions for batch processing
                    unified_grid = torch.tensor([1, H_grid_max, W_grid_max], dtype=torch.long)
                    new_image_grid_thw.append(unified_grid)
                
                # Stack all tensors - ensure all have same shape
                batch['pixel_values'] = torch.stack(padded_pixel_values)  # (B, N_max, D_v)
                batch['image_grid_thw'] = torch.stack(new_image_grid_thw)  # (B, 3)
                
                # Handle text features with variable length
                # Pad input_ids, attention_mask, and labels to the same length
                max_len = max(ids.shape[0] for ids in padded_input_ids)
                
                final_input_ids = []
                final_attention_masks = []
                final_labels = []
                
                for ids, mask, labs in zip(padded_input_ids, padded_attention_masks, padded_labels):
                    if ids.shape[0] < max_len:
                        padding_len = max_len - ids.shape[0]
                        ids = torch.cat([ids, torch.full((padding_len,), self.tokenizer.pad_token_id, dtype=ids.dtype)])
                        mask = torch.cat([mask, torch.zeros(padding_len, dtype=mask.dtype)])
                        labs = torch.cat([labs, torch.full((padding_len,), -100, dtype=labs.dtype)])
                    final_input_ids.append(ids)
                    final_attention_masks.append(mask)
                    final_labels.append(labs)
                
                batch['input_ids'] = torch.stack(final_input_ids)
                batch['attention_mask'] = torch.stack(final_attention_masks)
                batch['labels'] = torch.stack(final_labels)
                
                # Validate the batch
                # Debug: Final batch validation (disabled for performance)
                
            else:
                # 3D format (C, H, W) - handle dynamic resolution
                pix_list = []
                grid_list = []
                H_grid_max = W_grid_max = 0
                
                # Find maximum grid size in batch
                for f in features:
                    pix = f['pixel_values']  # (3, H, W)
                    pix_list.append(pix)
                    
                    if 'image_grid_thw' in f:
                        grid = f['image_grid_thw']  # (3,) = [T, H_grid, W_grid]
                        grid_list.append(grid)
                        H_grid_max = max(H_grid_max, int(grid[1]))
                        W_grid_max = max(W_grid_max, int(grid[2]))
                    else:
                        # Calculate grid from image size if not provided
                        _, H, W = pix.shape
                        H_grid = H // 14
                        W_grid = W // 14
                        grid = torch.tensor([1, H_grid, W_grid], dtype=torch.long)
                        grid_list.append(grid)
                        H_grid_max = max(H_grid_max, H_grid)
                        W_grid_max = max(W_grid_max, W_grid)
                
                # Round up to nearest even number for PatchMerge
                H_grid_max = (H_grid_max + 1) // 2 * 2
                W_grid_max = (W_grid_max + 1) // 2 * 2
                
                # Calculate new size (multiple of 14)
                New_H = H_grid_max * 14
                New_W = W_grid_max * 14
                
                # Pad all images to unified size
                padded_imgs = []
                new_grids = []
                
                for pix in pix_list:
                    _, H, W = pix.shape
                    pad_h = New_H - H
                    pad_w = New_W - W
                    
                    # Pad right and bottom with zeros
                    pix_padded = torch.nn.functional.pad(pix, (0, pad_w, 0, pad_h), value=0.0)
                    padded_imgs.append(pix_padded)
                    
                    # Update image_grid_thw to padded size
                    new_grid = torch.tensor([1, H_grid_max, W_grid_max], dtype=torch.long)
                    new_grids.append(new_grid)
                
                batch['pixel_values'] = torch.stack(padded_imgs)  # (B, 3, New_H, New_W)
                batch['image_grid_thw'] = torch.stack(new_grids)  # (B, 3)
                
                # Handle text features
                text_features = []
                for f in features:
                    text_feat = {
                        'input_ids': f['input_ids'],
                        'attention_mask': f.get('attention_mask', torch.ones_like(f['input_ids']))
                    }
                    if 'labels' in f:
                        text_feat['labels'] = f['labels']
                    text_features.append(text_feat)
                
                # Pad text features to the same length manually
                max_len = max(f['input_ids'].shape[0] for f in text_features)
                
                padded_input_ids = []
                padded_attention_masks = []
                padded_labels = []
                
                for text_feat in text_features:
                    ids = text_feat['input_ids']
                    mask = text_feat['attention_mask']
                    
                    if ids.shape[0] < max_len:
                        padding_len = max_len - ids.shape[0]
                        ids = torch.cat([ids, torch.full((padding_len,), self.tokenizer.pad_token_id, dtype=ids.dtype)])
                        mask = torch.cat([mask, torch.zeros(padding_len, dtype=mask.dtype)])
                    
                    padded_input_ids.append(ids)
                    padded_attention_masks.append(mask)
                    
                    if 'labels' in text_feat:
                        labels = text_feat['labels']
                        if labels.shape[0] < max_len:
                            padding_len = max_len - labels.shape[0]
                            labels = torch.cat([labels, torch.full((padding_len,), -100, dtype=labels.dtype)])
                        padded_labels.append(labels)
                
                batch['input_ids'] = torch.stack(padded_input_ids)
                batch['attention_mask'] = torch.stack(padded_attention_masks)
                if padded_labels:
                    batch['labels'] = torch.stack(padded_labels)
        
        # Handle mask labels - test_a1準拠のキー名変換
        # HybridDatasetは'ground_truth_mask'を返すが、モデルは'mask_labels'を期待
        # VQAなどマスクを使わないタスクも混在するため、適切に処理
        has_any_mask = False
        mask_labels = []
        
        for f in features:
            gt_mask = f.get('ground_truth_mask')
            if gt_mask is not None:
                has_any_mask = True
                mask_labels.append(gt_mask)
            elif f.get('mask_labels') is not None:
                has_any_mask = True
                mask_labels.append(f['mask_labels'])
        
        # マスクが1つでも存在する場合のみ、mask_labelsをセット
        if has_any_mask and len(mask_labels) > 0:
            batch['mask_labels'] = torch.stack(mask_labels)
        else:
            # VQAのみのバッチなど、マスクが全くない場合はNone
            batch['mask_labels'] = None
        
        # Handle SAM images for high-resolution processing
        if 'sam_images' in features[0] and features[0]['sam_images'] is not None:
            sam_images_list = []
            for i, f in enumerate(features):
                sam_img = f.get('sam_images')
                if sam_img is not None:
                    sam_images_list.append(sam_img)
                else:
                    # Create dummy SAM image if missing
                    sam_images_list.append(torch.zeros(3, 1024, 1024))
            batch['sam_images'] = torch.stack(sam_images_list)
        else:
            batch['sam_images'] = None
        
        # Preserve original images and sizes for visualization
        if any('original_image' in f for f in features):
            batch['original_images'] = [f.get('original_image') for f in features]
        
        # Preserve original image sizes for SAM postprocessing
        if any('orig_hw' in f for f in features):
            batch['orig_hw'] = [f.get('orig_hw') for f in features]
        
        # Preserve has_mask flags for each sample
        if any('has_mask' in f for f in features):
            batch['has_mask'] = [f.get('has_mask', False) for f in features]
        
        return batch


@dataclass
class LISADataCollator:
    """
    Data collator for LISA training
    Handles variable-length sequences and optional mask data
    """
    
    tokenizer: any
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    
    def __call__(self, features: List[Dict[str, any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples
        
        Args:
            features: List of sample dictionaries
        
        Returns:
            Batch dictionary with padded tensors
        """
        batch = {}
        
        # Separate features by type
        has_masks = any('mask_labels' in f for f in features)
        
        # Handle text features (input_ids, attention_mask, labels)
        text_features = []
        for f in features:
            text_feat = {
                'input_ids': f['input_ids'],
                'attention_mask': f['attention_mask'],
                'labels': f['labels']
            }
            text_features.append(text_feat)
        
        # Pad text features
        text_batch = self.tokenizer.pad(
            text_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors
        )
        
        batch['input_ids'] = text_batch['input_ids']
        batch['attention_mask'] = text_batch['attention_mask']
        
        # Handle labels padding
        if 'labels' in text_batch:
            batch['labels'] = text_batch['labels']
        else:
            # Manually pad labels
            labels = [f['labels'] for f in text_features]
            batch['labels'] = self._pad_labels(labels)
        
        # Handle image features
        pixel_values = torch.stack([f['pixel_values'] for f in features])
        batch['pixel_values'] = pixel_values
        
        # Handle mask labels if present
        if has_masks:
            mask_labels_list = []
            for f in features:
                if 'mask_labels' in f:
                    mask_labels_list.append([f['mask_labels']])
                else:
                    mask_labels_list.append(None)
            batch['mask_labels'] = mask_labels_list
        
        return batch
    
    def _pad_labels(self, labels_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Pad label sequences with -100
        
        Args:
            labels_list: List of label tensors
        
        Returns:
            Padded label tensor
        """
        # Find max length
        max_len = max(len(labels) for labels in labels_list)
        if self.max_length is not None:
            max_len = min(max_len, self.max_length)
        
        # Pad each sequence
        padded_labels = []
        for labels in labels_list:
            if len(labels) < max_len:
                # Pad with -100 (ignore index)
                padding = torch.full((max_len - len(labels),), -100, dtype=labels.dtype)
                padded = torch.cat([labels, padding])
            else:
                padded = labels[:max_len]
            padded_labels.append(padded)
        
        return torch.stack(padded_labels)


@dataclass
class LISAEvalDataCollator(LISADataCollator):
    """
    Data collator for evaluation
    Preserves original data for metric computation
    """
    
    def __call__(self, features: List[Dict[str, any]]) -> Dict[str, any]:
        """
        Collate batch for evaluation
        
        Preserves original masks and metadata
        """
        batch = super().__call__(features)
        
        # Add original data for evaluation
        if any('original_image' in f for f in features):
            batch['original_images'] = [f.get('original_image') for f in features]
            # Debug logging
            import logging
            logger = logging.getLogger(__name__)
            if batch['original_images'] and batch['original_images'][0] is not None:
                from PIL import Image
                if isinstance(batch['original_images'][0], Image.Image):
                    logger.debug(f"Added PIL original_images to batch: {batch['original_images'][0].size}")
                else:
                    logger.debug(f"Added original_images to batch: type={type(batch['original_images'][0])}")
        else:
            # Always add empty list if no original_images
            batch['original_images'] = []
        
        if any('metadata' in f for f in features):
            batch['metadata'] = [f.get('metadata', {}) for f in features]
        
        return batch