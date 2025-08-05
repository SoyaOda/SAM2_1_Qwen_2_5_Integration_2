"""
Quick test for dataset-collator compatibility
"""
import sys
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.utils import prepare_tokenizer_for_lisa
from src.data.collators import MultiModalDataCollator
from transformers import AutoProcessor
import numpy as np
from PIL import Image

# Setup
tokenizer = prepare_tokenizer_for_lisa("Qwen/Qwen2.5-VL-3B-Instruct")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# Create dummy sample that mimics dataset output
sample = {
    'input_ids': torch.randint(0, 32000, (50,)),
    'labels': torch.randint(0, 32000, (50,)),
    'attention_mask': torch.ones(50),
    'sam_images': torch.randn(3, 1024, 1024),
    'pixel_values': torch.randn(3, 448, 448),
    'ground_truth_mask': torch.randint(0, 2, (1, 1024, 1024)).float(),
    'has_mask': True,
    'seg_token_mask': torch.zeros(50).bool(),
    'image_path': '/path/to/test.jpg',
    'text_prompt': 'Test prompt',
    'image_grid_thw': torch.tensor([1, 24, 24]),
    'resize': None,
    'questions': None,
    'sampled_classes': None,
    'original_image': Image.new('RGB', (224, 224))
}

# Test collator
collator = MultiModalDataCollator(tokenizer=tokenizer, max_length=512)

try:
    # Single sample
    batch = collator([sample])
    print("✓ Single sample collation successful")
    print(f"  Batch keys: {list(batch.keys())}")
    
    # Check mask handling
    if 'mask_labels' in batch:
        print(f"  ✓ mask_labels found in batch: shape {batch['mask_labels'].shape}")
    else:
        print("  ✗ mask_labels NOT found in batch")
        if 'ground_truth_mask' in batch:
            print(f"  ! ground_truth_mask found instead: shape {batch['ground_truth_mask'].shape}")
    
    # Multiple samples
    batch2 = collator([sample, sample])
    print("\n✓ Multiple samples collation successful")
    print(f"  Batch shape: {batch2['input_ids'].shape}")
    
except Exception as e:
    print(f"✗ Collator failed: {e}")
    import traceback
    traceback.print_exc()