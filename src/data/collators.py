"""
Data collators for LISA改 (LISA-Kai) training
Handles batching of mixed data types
"""
import torch
from typing import Dict, List, Optional, Union
from dataclasses import dataclass


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