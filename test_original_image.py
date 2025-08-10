#!/usr/bin/env python3
"""
original_imageがDataCollatorに正しく渡されているかテスト
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.config import LISAConfig
from src.data.dataset import HybridDataset
from src.data.collators import MultiModalDataCollator
from transformers import AutoProcessor, AutoTokenizer
from src.utils import prepare_tokenizer_for_lisa
from torch.utils.data import DataLoader

# Config
config = LISAConfig()

# Tokenizer & Processor
tokenizer = prepare_tokenizer_for_lisa(
    model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    seg_token="<SEG>"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# Dataset
dataset = HybridDataset(
    base_image_dir=config.dataset_base_dir,
    qwen_processor=processor,
    samples_per_epoch=1,
    dataset='sem_seg',
    sample_rate=[1.0],
)

# Get single sample
print("\n=== Single Sample Test ===")
sample = dataset[0]
print(f"Sample type: {type(sample)}")
if isinstance(sample, dict):
    print(f"Sample keys: {sample.keys()}")
    if 'original_image' in sample:
        print(f"✓ original_image found: {type(sample['original_image'])}")
    else:
        print("✗ original_image NOT found")

# Test with DataCollator
print("\n=== DataCollator Test ===")
collator = MultiModalDataCollator(
    tokenizer=tokenizer,
    max_length=config.model_max_length,
    config=config
)

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=1,
    collate_fn=collator,
    num_workers=0
)

# Get batch
for batch in dataloader:
    print(f"Batch keys: {batch.keys()}")
    if 'original_images' in batch:
        print(f"✓ original_images found in batch: {len(batch['original_images'])} items")
        if batch['original_images']:
            print(f"  Type: {type(batch['original_images'][0])}")
    else:
        print("✗ original_images NOT in batch")
    break