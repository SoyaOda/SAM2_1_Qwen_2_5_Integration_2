"""
Dataset classes for LISA改 (LISA-Kai) training
Supports multiple dataset types: captioning, VQA, and referring segmentation
"""
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Union, Tuple
import json
from pathlib import Path
from PIL import Image
import numpy as np
from transformers import AutoProcessor


class LISADataset(Dataset):
    """
    Base dataset class for LISA改 training
    Supports mixed datasets with different task types
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        processor: AutoProcessor,
        tokenizer,
        max_length: int = 512,
        image_size: int = 448,
        task_type: str = "mixed",
        seg_token: str = "<SEG>"
    ):
        """
        Initialize LISA dataset
        
        Args:
            data_path: Path to dataset JSON or directory
            processor: Image processor for Qwen2.5-VL
            tokenizer: Tokenizer with SEG token
            max_length: Maximum sequence length
            image_size: Target image size
            task_type: Dataset type (caption, vqa, refseg, mixed)
            seg_token: Segmentation special token
        """
        self.data_path = Path(data_path)
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_size = image_size
        self.task_type = task_type
        self.seg_token = seg_token
        self.seg_token_id = tokenizer.convert_tokens_to_ids(seg_token)
        
        # Load dataset annotations
        self.samples = self._load_annotations()
    
    def _load_annotations(self) -> List[Dict]:
        """Load dataset annotations from JSON"""
        if self.data_path.suffix == '.json':
            with open(self.data_path, 'r') as f:
                data = json.load(f)
        else:
            # Load from directory structure
            data = []
            for json_file in self.data_path.glob('*.json'):
                with open(json_file, 'r') as f:
                    data.extend(json.load(f))
        
        return data
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample
        
        Returns dict with keys:
            - input_ids: Token IDs including image tokens
            - attention_mask: Attention mask
            - pixel_values: Processed image tensor
            - labels: Language modeling labels
            - mask_labels: Segmentation mask (if available)
        """
        sample = self.samples[idx]
        
        # Load and process image
        image_path = Path(sample['image_path'])
        image = Image.open(image_path).convert('RGB')
        
        # Process image using Qwen processor
        pixel_values = self.processor.image_processor(
            images=image,
            return_tensors="pt"
        )['pixel_values'][0]
        
        # Prepare text based on task type
        text_input, has_mask = self._prepare_text_input(sample)
        
        # Tokenize text
        encoding = self.tokenizer(
            text_input,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'][0]
        attention_mask = encoding['attention_mask'][0]
        
        # Prepare labels (mask non-response tokens with -100)
        labels = self._prepare_labels(input_ids, sample)
        
        # Prepare output dict
        output = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'labels': labels
        }
        
        # Add mask labels if this is a segmentation task
        if has_mask and 'mask_path' in sample:
            mask = self._load_mask(sample['mask_path'])
            output['mask_labels'] = mask
        
        return output
    
    def _prepare_text_input(self, sample: Dict) -> Tuple[str, bool]:
        """
        Prepare text input based on task type
        
        Returns:
            text_input: Formatted text input
            has_mask: Whether this sample includes segmentation
        """
        task = sample.get('task_type', self.task_type)
        
        if task == 'caption':
            # Image captioning task
            text_input = f"Human: Please describe this image.\nAssistant: {sample['caption']}"
            return text_input, False
        
        elif task == 'vqa':
            # Visual question answering
            text_input = f"Human: {sample['question']}\nAssistant: {sample['answer']}"
            return text_input, False
        
        elif task == 'refseg':
            # Referring segmentation
            instruction = sample['instruction']
            # Response includes SEG token
            response = f"{self.seg_token}"
            if 'description' in sample:
                response += f" {sample['description']}"
            text_input = f"Human: {instruction}\nAssistant: {response}"
            return text_input, True
        
        else:
            raise ValueError(f"Unknown task type: {task}")
    
    def _prepare_labels(self, input_ids: torch.Tensor, sample: Dict) -> torch.Tensor:
        """
        Prepare labels by masking non-response tokens
        
        Args:
            input_ids: Tokenized input IDs
            sample: Data sample
        
        Returns:
            labels: Token IDs with non-response positions masked (-100)
        """
        labels = input_ids.clone()
        
        # Find where assistant response starts
        # Simple approach: look for "Assistant:" token sequence
        assistant_tokens = self.tokenizer.encode("Assistant:", add_special_tokens=False)
        
        # Find the position of assistant response
        response_start = None
        for i in range(len(input_ids) - len(assistant_tokens)):
            if torch.equal(input_ids[i:i+len(assistant_tokens)], 
                          torch.tensor(assistant_tokens)):
                response_start = i + len(assistant_tokens)
                break
        
        if response_start is not None:
            # Mask everything before the response
            labels[:response_start] = -100
        
        # Also mask padding tokens
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        
        return labels
    
    def _load_mask(self, mask_path: Union[str, Path]) -> torch.Tensor:
        """
        Load segmentation mask
        
        Args:
            mask_path: Path to mask file
        
        Returns:
            mask: Binary mask tensor [H, W]
        """
        mask_path = Path(mask_path)
        
        if mask_path.suffix in ['.png', '.jpg', '.jpeg']:
            # Load from image file
            mask = Image.open(mask_path).convert('L')
            mask = np.array(mask) > 128  # Binarize
        elif mask_path.suffix == '.npy':
            # Load from numpy file
            mask = np.load(mask_path)
        else:
            raise ValueError(f"Unsupported mask format: {mask_path.suffix}")
        
        # Convert to tensor
        mask = torch.from_numpy(mask).float()
        
        return mask


class RefCOCODataset(LISADataset):
    """
    Dataset for RefCOCO/RefCOCO+/RefCOCOg referring segmentation
    """
    
    def __init__(self, *args, **kwargs):
        kwargs['task_type'] = 'refseg'
        super().__init__(*args, **kwargs)
    
    def _load_annotations(self) -> List[Dict]:
        """Load RefCOCO-style annotations"""
        # RefCOCO specific loading logic
        # This is a simplified version - adapt to actual RefCOCO format
        annotations = []
        
        if self.data_path.name == 'refcoco':
            # Load RefCOCO annotations
            ann_file = self.data_path / 'annotations.json'
            with open(ann_file, 'r') as f:
                data = json.load(f)
            
            for ann in data['annotations']:
                sample = {
                    'image_path': self.data_path / 'images' / ann['image_file'],
                    'mask_path': self.data_path / 'masks' / ann['mask_file'],
                    'instruction': ann['expression'],
                    'task_type': 'refseg'
                }
                annotations.append(sample)
        
        return annotations


class MixedDataset(Dataset):
    """
    Mixed dataset combining multiple task types
    """
    
    def __init__(
        self,
        datasets: List[Dataset],
        sampling_weights: Optional[List[float]] = None
    ):
        """
        Initialize mixed dataset
        
        Args:
            datasets: List of datasets to mix
            sampling_weights: Sampling weights for each dataset
        """
        self.datasets = datasets
        self.sampling_weights = sampling_weights or [1.0] * len(datasets)
        
        # Calculate dataset sizes and cumulative indices
        self.dataset_sizes = [len(d) for d in datasets]
        self.cumulative_sizes = np.cumsum([0] + self.dataset_sizes)
        self.total_size = sum(self.dataset_sizes)
    
    def __len__(self) -> int:
        return self.total_size
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item from appropriate dataset"""
        # Find which dataset this index belongs to
        dataset_idx = np.searchsorted(self.cumulative_sizes[1:], idx, side='right')
        
        # Get index within that dataset
        local_idx = idx - self.cumulative_sizes[dataset_idx]
        
        # Return item from that dataset
        return self.datasets[dataset_idx][local_idx]


def create_dataset(
    data_config: Dict,
    processor: AutoProcessor,
    tokenizer,
    is_train: bool = True
) -> Dataset:
    """
    Create dataset based on configuration
    
    Args:
        data_config: Dataset configuration
        processor: Image processor
        tokenizer: Text tokenizer
        is_train: Whether this is for training
    
    Returns:
        Dataset instance
    """
    dataset_type = data_config.get('type', 'mixed')
    
    if dataset_type == 'mixed':
        # Create multiple datasets and mix them
        datasets = []
        
        for dataset_cfg in data_config['datasets']:
            ds_type = dataset_cfg['type']
            
            if ds_type == 'refcoco':
                ds = RefCOCODataset(
                    data_path=dataset_cfg['path'],
                    processor=processor,
                    tokenizer=tokenizer,
                    **dataset_cfg.get('params', {})
                )
            else:
                ds = LISADataset(
                    data_path=dataset_cfg['path'],
                    processor=processor,
                    tokenizer=tokenizer,
                    task_type=ds_type,
                    **dataset_cfg.get('params', {})
                )
            
            datasets.append(ds)
        
        # Create mixed dataset
        weights = data_config.get('sampling_weights')
        return MixedDataset(datasets, weights)
    
    else:
        # Create single dataset
        return LISADataset(
            data_path=data_config['path'],
            processor=processor,
            tokenizer=tokenizer,
            task_type=dataset_type,
            **data_config.get('params', {})
        )