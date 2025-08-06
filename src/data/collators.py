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
        return_tensors: str = "pt"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.return_tensors = return_tensors
    
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
                H_grid_max = W_grid_max = 0
                for i, pv in enumerate(all_pixel_values):
                    n_patches = pv.shape[0]
                    
                    # Use provided grid if available
                    if i < len(all_image_grids):
                        grid = all_image_grids[i]
                        H_grid = int(grid[1])
                        W_grid = int(grid[2])
                        if H_grid * W_grid != n_patches:
                            print(f"Warning: Grid mismatch - grid says {H_grid}×{W_grid}={H_grid*W_grid}, but has {n_patches} patches")
                    else:
                        # Estimate grid from patch count
                        # Find factors closest to square
                        factors = []
                        for h in range(1, int(n_patches**0.5) + 1):
                            if n_patches % h == 0:
                                w = n_patches // h
                                factors.append((h, w))
                        # Choose most square-like
                        H_grid, W_grid = min(factors, key=lambda x: abs(x[0] - x[1]))
                    
                    H_grid_max = max(H_grid_max, H_grid)
                    W_grid_max = max(W_grid_max, W_grid)
                
                # Calculate max patches (must be 4の倍数)
                max_patches_raw = H_grid_max * W_grid_max
                # Qwen2.5-VL requires patches to be multiple of 4 for compression
                max_patches = ((max_patches_raw + 3) // 4) * 4
                
                # Adjust grid if needed to match 4の倍数
                if max_patches != max_patches_raw:
                    # Need to expand grid slightly
                    # Try expanding width first
                    W_grid_max = (max_patches + H_grid_max - 1) // H_grid_max
                    if H_grid_max * W_grid_max != max_patches:
                        # Adjust height if needed
                        H_grid_max = (max_patches + W_grid_max - 1) // W_grid_max
                
                # Debug output for batch processing
                if len(all_image_grids) > 0:
                    print(f"[Batch Collator] Unified grid: {H_grid_max}×{W_grid_max} = {max_patches} patches ({max_patches//4} tokens)")
                    individual_info = []
                    for i, (pv, grid) in enumerate(zip(all_pixel_values, all_image_grids if all_image_grids else [])):
                        if i < len(all_image_grids):
                            individual_info.append(f"({int(grid[1])},{int(grid[2])}={int(grid[1])*int(grid[2])})")
                        else:
                            individual_info.append("estimated")
                    print(f"  Original sample grids: {individual_info}")
                
                # IMAGE_PAD_ID for Qwen2.5-VL
                IMAGE_PAD_ID = 151655
                
                # Process each sample
                padded_pixel_values = []
                padded_input_ids = []
                padded_attention_masks = []
                new_image_grid_thw = []
                
                for f in features:
                    pv = f['pixel_values']
                    input_ids = f['input_ids']
                    attention_mask = f.get('attention_mask', torch.ones_like(input_ids))
                    
                    # Calculate padding size
                    pad_len = max_patches - pv.shape[0]
                    
                    # Debug: Check original grid size
                    if 'image_grid_thw' in f and f['image_grid_thw'] is not None:
                        orig_grid = f['image_grid_thw']
                        orig_patches = int(orig_grid[1]) * int(orig_grid[2])
                        if orig_patches != pv.shape[0]:
                            print(f"Warning: Original grid mismatch - grid says {orig_patches} patches ({orig_grid[1]}x{orig_grid[2]}), but pixel_values has {pv.shape[0]}")
                    
                    # 1. Pad pixel_values with zeros in the patch dimension
                    if pad_len > 0:
                        padding = torch.zeros(pad_len, pv.shape[1], dtype=pv.dtype, device=pv.device)
                        pv_padded = torch.cat([pv, padding], dim=0)
                    else:
                        pv_padded = pv
                    padded_pixel_values.append(pv_padded)
                    
                    # 2. Add corresponding <|image_pad|> tokens to input_ids
                    # Count existing image_pad tokens
                    image_pad_mask = (input_ids == IMAGE_PAD_ID)
                    existing_pads = image_pad_mask.sum().item()
                    
                    # Debug: Qwen2.5-VL uses 4:1 patch compression
                    # So image_pad tokens = pixel patches / 4
                    # Ensure patches are multiple of 4
                    if pv.shape[0] % 4 != 0:
                        print(f"Warning: Original patches {pv.shape[0]} not multiple of 4")
                    expected_pads = pv.shape[0] // 4
                    if existing_pads != expected_pads:
                        print(f"Warning: Mismatch before padding - existing image_pad tokens: {existing_pads}, expected (patches/4): {expected_pads}, pixel patches: {pv.shape[0]}")
                    
                    # For Qwen2.5-VL, we need to add pad_len/4 image_pad tokens
                    # because of the 4:1 patch compression
                    image_pad_to_add = pad_len // 4
                    
                    if image_pad_to_add > 0:
                        if image_pad_mask.any():
                            # Find the first occurrence of image_pad token
                            first_img_pad_idx = image_pad_mask.nonzero(as_tuple=True)[0][0].item()
                            
                            # Insert additional image_pad tokens
                            ids_padded = torch.cat([
                                input_ids[:first_img_pad_idx],  # Before image_pad block
                                input_ids.new_full((existing_pads + image_pad_to_add,), IMAGE_PAD_ID),  # All image_pad tokens
                                input_ids[first_img_pad_idx + existing_pads:]  # After image_pad block
                            ])
                        else:
                            # No existing image_pad tokens, append at the end
                            print(f"Warning: No existing image_pad tokens found in input_ids!")
                            ids_padded = torch.cat([
                                input_ids,
                                input_ids.new_full((image_pad_to_add,), IMAGE_PAD_ID)
                            ])
                    else:
                        ids_padded = input_ids
                    
                    # Verify after padding
                    new_image_pad_count = (ids_padded == IMAGE_PAD_ID).sum().item()
                    expected_final_pads = max_patches // 4
                    if new_image_pad_count != expected_final_pads:
                        print(f"Error: After padding - image_pad tokens: {new_image_pad_count}, expected (max_patches/4): {expected_final_pads}")
                        print(f"  Sample {len(padded_input_ids)}: original patches={pv.shape[0]}, padded to {max_patches}")
                        print(f"  Original image_pads={existing_pads}, added={image_pad_to_add}, total={new_image_pad_count}")
                    
                    padded_input_ids.append(ids_padded)
                    
                    # 3. Update attention_mask
                    if image_pad_to_add > 0:
                        mask_padded = torch.cat([
                            attention_mask,
                            torch.ones(image_pad_to_add, dtype=attention_mask.dtype)
                        ])
                    else:
                        mask_padded = attention_mask
                    padded_attention_masks.append(mask_padded)
                    
                    # 4. Update image_grid_thw to match padded patches
                    # All samples in batch must have the same grid size
                    new_grid = torch.tensor([1, H_grid_max, W_grid_max], dtype=torch.long)
                    new_image_grid_thw.append(new_grid)
                
                # Stack all tensors
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
                
                # トークン数上限チェック（Qwen2.5-VLのRoPE制限）
                total_tokens = batch['input_ids'].shape[1]
                if total_tokens > 2048:
                    print(f"Warning: Total tokens ({total_tokens}) exceeds 2048 limit!")
                    print(f"  Image tokens: {max_patches // 4}")
                    print(f"  Text tokens: {total_tokens - max_patches // 4}")
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