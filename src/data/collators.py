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
        
        # Handle image features
        if 'pixel_values' in features[0]:
            pixel_values = torch.stack([f['pixel_values'] for f in features])
            batch['pixel_values'] = pixel_values
            
        # Handle image_grid_thw
        if 'image_grid_thw' in features[0]:
            image_grid_thw = torch.stack([f['image_grid_thw'] for f in features])
            batch['image_grid_thw'] = image_grid_thw
        
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
        
        # Handle mask labels (optional)
        if 'mask_labels' in features[0] and features[0]['mask_labels'] is not None:
            mask_labels = []
            for f in features:
                if f.get('mask_labels') is not None:
                    mask_labels.append(f['mask_labels'])
                else:
                    # Create dummy mask if missing
                    h, w = f['pixel_values'].shape[-2:]
                    mask_labels.append(torch.zeros(1, h, w))
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