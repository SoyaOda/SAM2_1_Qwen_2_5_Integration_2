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
        Collate batch of multimodal samples
        
        Args:
            features: List of sample dictionaries containing:
                - pixel_values: Image tensors
                - input_ids: Token IDs
                - labels: Target token IDs
                - attention_mask: Attention masks
                - mask_labels (optional): Segmentation masks
        
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
                # This is the official behavior when tokenize=True
                # Format: (N_patch, D_v) where N_patch = H_grid × W_grid
                
                # Collect all pixel_values to analyze
                all_pixel_values = []
                all_image_grids = []
                for f in features:
                    all_pixel_values.append(f['pixel_values'])
                    if 'image_grid_thw' in f and f['image_grid_thw'] is not None:
                        all_image_grids.append(f['image_grid_thw'])
                
                # Find max grid dimensions
                # IMPORTANT (Corrected Spec):
                # - image_grid_thw contains RAW patch dimensions (H_img/14, W_img/14) 
                # - pixel_values contains RAW patches (NOT compressed): N_raw = H_grid * W_grid
                # - PatchMerge happens INSIDE the model, not in preprocessing
                # - image_pad tokens = RAW patches / 4 (after PatchMerge)
                
                # First find the maximum number of patches needed
                max_raw_patches = 0
                all_raw_patches = []
                
                for i, pv in enumerate(all_pixel_values):
                    # pixel_values contains RAW patches (before PatchMerge)
                    n_raw_patches = pv.shape[0]
                    all_raw_patches.append(n_raw_patches)
                    
                    # Grid info is required for dynamic resolution
                    if i >= len(all_image_grids) or all_image_grids[i] is None:
                        raise ValueError(f"Sample {i}: image_grid_thw is required for 2D pixel_values format")
                    
                    grid = all_image_grids[i]
                    # image_grid_thw contains RAW patch dimensions (before PatchMerge)
                    H_grid_raw = int(grid[1])
                    W_grid_raw = int(grid[2])
                    
                    # Verify RAW patches consistency
                    expected_raw_patches = H_grid_raw * W_grid_raw
                    
                    if expected_raw_patches != n_raw_patches:
                        raise ValueError(
                            f"Sample {i}: Patch count mismatch! "
                            f"Grid: {H_grid_raw}×{W_grid_raw} = {expected_raw_patches} RAW patches. "
                            f"Actual: {n_raw_patches} patches in pixel_values. "
                            f"This usually means the image preprocessing is incorrect."
                        )
                    
                    max_raw_patches = max(max_raw_patches, n_raw_patches)
                
                # Calculate proper grid dimensions using square policy (O3 recommendation)
                # CRITICAL: H_max and W_max must be EVEN for PatchMerge compatibility
                import math
                
                # Square policy: make it as square as possible
                grid_size = math.ceil(math.sqrt(max_raw_patches))
                # Round up to nearest even number (REQUIRED for 2x2 PatchMerge)
                H_grid_max = W_grid_max = (grid_size + 1) // 2 * 2
                
                # Actual padded patch count (must be H*W exactly)
                N_padded_raw = H_grid_max * W_grid_max
                
                # Image tokens after PatchMerge
                max_img_tokens = N_padded_raw // 4
                
                # Debug output for batch processing
                if len(all_image_grids) > 0:
                    print(f"[Batch Collator] Max RAW patches: {max_raw_patches}")
                    print(f"  Target grid (even): {H_grid_max}×{W_grid_max} = {N_padded_raw} patches")
                    print(f"  Image tokens after PatchMerge: {max_img_tokens}")
                    individual_info = []
                    for i, n_patches in enumerate(all_raw_patches):
                        if i < len(all_image_grids):
                            grid = all_image_grids[i]
                            individual_info.append(f"Sample {i}: {n_patches} patches (grid: {int(grid[1])}×{int(grid[2])})")
                        else:
                            individual_info.append(f"Sample {i}: {n_patches} patches")
                    for info in individual_info:
                        print(f"  {info}")
                
                # IMAGE_PAD_ID for Qwen2.5-VL
                IMAGE_PAD_ID = 151655
                VISION_START_ID = 151652  # <|vision_start|>
                VISION_END_ID = 151653     # <|vision_end|>
                
                # IMPORTANT: image_pad tokens = RAW patches / 4 (after PatchMerge)
                # Already calculated as max_img_tokens above
                
                # Process each sample
                padded_pixel_values = []
                padded_input_ids = []
                padded_attention_masks = []
                new_image_grid_thw = []
                
                print(f"[DEBUG] Processing {len(features)} samples:")
                print(f"  Max RAW patches: {max_raw_patches}")
                print(f"  Max image tokens (after PatchMerge): {max_img_tokens}")
                
                for i, f in enumerate(features):
                    pv = f['pixel_values']
                    input_ids = f['input_ids']
                    attention_mask = f.get('attention_mask', torch.ones_like(input_ids))
                    
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
                    
                    # 1. Pad pixel_values to N_padded_raw (not just max_raw_patches)
                    # CRITICAL: Must pad to H_grid_max * W_grid_max exactly
                    current_patches = pv.shape[0]
                    pad_len = N_padded_raw - current_patches
                    
                    if pad_len > 0:
                        # Zero padding for pixel values
                        padding = torch.zeros(pad_len, pv.shape[1], dtype=pv.dtype, device=pv.device)
                        pv_padded = torch.cat([pv, padding], dim=0)
                    else:
                        pv_padded = pv[:N_padded_raw]  # Truncate if needed (shouldn't happen)
                    
                    padded_pixel_values.append(pv_padded)
                    
                    # 2. Handle input_ids padding - need to adjust image_pad tokens
                    # Count existing image_pad tokens first
                    existing_image_pads = (input_ids == IMAGE_PAD_ID).sum().item()
                    
                    # The correct number should match the image tokens after PatchMerge
                    # CRITICAL: Use the actual padded raw patches divided by 4
                    target_image_pads = N_padded_raw // 4  # This matches padded pixel_values
                    
                    print(f"  Sample {i}: existing_pads={existing_image_pads}, target_pads={target_image_pads}, "
                          f"orig_patches={current_patches}, padded_patches={N_padded_raw}")
                    
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
                        else:
                            # No existing image_pad tokens - shouldn't happen
                            print(f"Warning: No existing image_pad tokens found in sample {i}")
                            ids_padded = input_ids
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
                        else:
                            ids_padded = input_ids
                    else:
                        # Correct number already
                        ids_padded = input_ids
                    
                    padded_input_ids.append(ids_padded)
                    
                    # 3. Update attention_mask accordingly
                    if len(ids_padded) != len(attention_mask):
                        # Adjust mask to match new length
                        if len(ids_padded) > len(attention_mask):
                            # Added tokens - extend mask with 1s
                            extra = len(ids_padded) - len(attention_mask)
                            mask_padded = torch.cat([
                                attention_mask[:len(attention_mask)//2],  # First half
                                torch.ones(extra, dtype=attention_mask.dtype),  # New tokens
                                attention_mask[len(attention_mask)//2:]  # Second half
                            ])
                        else:
                            # Removed tokens - adjust mask
                            mask_padded = torch.ones(len(ids_padded), dtype=attention_mask.dtype)
                    else:
                        mask_padded = attention_mask
                    
                    padded_attention_masks.append(mask_padded)
                    
                    # 4. Update grid to unified dimensions for batch processing
                    # CRITICAL: All samples must have the same grid size (H_grid_max, W_grid_max)
                    # This ensures RoPE embeddings work correctly
                    unified_grid = torch.tensor([1, H_grid_max, W_grid_max], dtype=torch.long)
                    new_image_grid_thw.append(unified_grid)
                
                # Stack all tensors - ensure all have same shape
                # Check dimensions before stacking
                shapes = [pv.shape for pv in padded_pixel_values]
                if len(set(shapes)) > 1:
                    print(f"Error: Inconsistent pixel_values shapes after padding: {shapes}")
                    # Force all to have the same shape by additional padding
                    max_patches_actual = max(pv.shape[0] for pv in padded_pixel_values)
                    fixed_pixel_values = []
                    for pv in padded_pixel_values:
                        if pv.shape[0] < max_patches_actual:
                            extra_pad = max_patches_actual - pv.shape[0]
                            pv = torch.cat([pv, torch.zeros(extra_pad, pv.shape[1], dtype=pv.dtype, device=pv.device)], dim=0)
                        fixed_pixel_values.append(pv)
                    batch['pixel_values'] = torch.stack(fixed_pixel_values)  # (B, N_max, D_v)
                else:
                    batch['pixel_values'] = torch.stack(padded_pixel_values)  # (B, N_max, D_v)
                batch['image_grid_thw'] = torch.stack(new_image_grid_thw)  # (B, 3)
                
                # Handle text features with variable length
                # Pad input_ids and attention_mask to the same length
                max_len = max(ids.shape[0] for ids in padded_input_ids)
                
                final_input_ids = []
                final_attention_masks = []
                final_labels = []
                
                for i, (ids, mask) in enumerate(zip(padded_input_ids, padded_attention_masks)):
                    if ids.shape[0] < max_len:
                        padding_len = max_len - ids.shape[0]
                        ids = torch.cat([ids, torch.full((padding_len,), self.tokenizer.pad_token_id, dtype=ids.dtype)])
                        mask = torch.cat([mask, torch.zeros(padding_len, dtype=mask.dtype)])
                    final_input_ids.append(ids)
                    final_attention_masks.append(mask)
                    
                    # Handle labels if present
                    if 'labels' in features[i]:
                        labels = features[i]['labels']
                        # Ensure labels match the padded length
                        if len(labels) < max_len:
                            padding_len = max_len - len(labels)
                            labels = torch.cat([labels, torch.full((padding_len,), -100, dtype=labels.dtype)])
                        final_labels.append(labels)
                
                batch['input_ids'] = torch.stack(final_input_ids)
                batch['attention_mask'] = torch.stack(final_attention_masks)
                if final_labels:
                    batch['labels'] = torch.stack(final_labels)
                
                # Validate the batch
                print(f"[DEBUG] Final batch validation:")
                print(f"  pixel_values shape: {batch['pixel_values'].shape}")
                print(f"  input_ids shape: {batch['input_ids'].shape}")
                for b_idx in range(batch['input_ids'].shape[0]):
                    n_image_pads = (batch['input_ids'][b_idx] == IMAGE_PAD_ID).sum().item()
                    print(f"  Sample {b_idx}: {n_image_pads} image_pad tokens")
                
                # トークン数上限チェック（Qwen2.5-VLのRoPE制限）
                total_tokens = batch['input_ids'].shape[1]
                max_length = getattr(self.config, 'model_max_length', 16384)  # 動的解像度対応
                if total_tokens > max_length:
                    print(f"Warning: Total tokens ({total_tokens}) exceeds {max_length} limit!")
                    print(f"  Image tokens: {max_img_tokens}")
                    print(f"  Text tokens: {total_tokens - max_img_tokens}")
                    # 必要に応じてエラーにする
                    # raise ValueError(f"Token count {total_tokens} exceeds maximum 2048")
            else:
                # 3D format (C, H, W) - need to handle dynamic resolution
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
                    text_features.append(text_feat)
                
                # Pad text features
                padded_text = self.tokenizer.pad(
                    text_features,
                    padding=self.padding,
                    max_length=self.max_length,
                    return_tensors=self.return_tensors
                )
                
                batch['input_ids'] = padded_text['input_ids']
                batch['attention_mask'] = padded_text['attention_mask']
                
                # Handle labels
                if 'labels' in features[0]:
                    labels = []
                    for f in features:
                        label = f['labels']
                        # Pad labels to match input_ids length
                        if len(label) < batch['input_ids'].size(1):
                            # Pad with -100 (ignore index)
                            padding_length = batch['input_ids'].size(1) - len(label)
                            label = torch.cat([
                                label,
                                torch.full((padding_length,), -100, dtype=label.dtype)
                            ])
                        labels.append(label)
                    batch['labels'] = torch.stack(labels)
        
        # Handle mask labels (optional) - support both 'mask_labels' and 'ground_truth_mask'
        mask_key = 'mask_labels' if 'mask_labels' in features[0] else 'ground_truth_mask'
        if mask_key in features[0] and features[0][mask_key] is not None:
            mask_labels = []
            
            # Determine padded size based on pixel_values format
            if 'pixel_values' in batch:
                if batch['pixel_values'].dim() == 3:
                    # Flattened format - masks are kept at original size
                    # Just stack them as is
                    for f in features:
                        if f.get(mask_key) is not None:
                            mask_labels.append(f[mask_key])
                        else:
                            # Create dummy mask with default size
                            mask_labels.append(torch.zeros(1, 1024, 1024))
                else:
                    # 3D format - use padded size
                    _, _, padded_h, padded_w = batch['pixel_values'].shape
                    
                    for f in features:
                        if f.get(mask_key) is not None:
                            mask = f[mask_key]
                            # Pad mask to match padded image size
                            if mask.dim() == 2:
                                mask = mask.unsqueeze(0)  # Add channel dimension
                            _, orig_h, orig_w = mask.shape
                            pad_h = padded_h - orig_h
                            pad_w = padded_w - orig_w
                            mask_padded = torch.nn.functional.pad(mask, (0, pad_w, 0, pad_h), value=0)
                            mask_labels.append(mask_padded)
                        else:
                            # Create dummy mask if missing
                            mask_labels.append(torch.zeros(1, padded_h, padded_w))
            else:
                # No pixel_values - use default size
                for f in features:
                    if f.get(mask_key) is not None:
                        mask_labels.append(f[mask_key])
                    else:
                        mask_labels.append(torch.zeros(1, 1024, 1024))
                        
            batch['mask_labels'] = torch.stack(mask_labels)
        
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
        
        if any('metadata' in f for f in features):
            batch['metadata'] = [f.get('metadata', {}) for f in features]
        
        return batch