#!/usr/bin/env python3
"""
Debug frozen SAM Other (Misc) parameters
"""

import torch
from transformers import AutoTokenizer
from src.models.lisa_model import LISA_Model
from src.config import LISAConfig
from collections import defaultdict

# Create model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True)
tokenizer.add_special_tokens({"additional_special_tokens": ["<SEG>"]})

config = LISAConfig(
    qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    sam_model_name="./checkpoints/sam2.1_hiera_large.pt",
    freeze_sam_mask_decoder_base=False,
    sam_lora_r=0
)

print("Loading model...")
model = LISA_Model(config, tokenizer=tokenizer)

print("\n" + "="*80)
print("Analyzing FROZEN SAM Other (Misc) parameters:")
print("="*80)

# Find frozen SAM parameters that would be classified as Other (Misc)
frozen_misc = []
total_frozen_misc = 0

for name, param in model.named_parameters():
    if param.requires_grad:
        continue  # Skip trainable params
    
    # Apply classification logic
    if 'token_fpn' in name or 'fpn' in name:
        continue  # This is Token-FPN
    elif 'qwen' in name:
        continue  # This is Qwen
    elif 'sam' in name or 'sam_model' in name:
        # Check if it's a known SAM component
        if 'mask_decoder' in name:
            continue
        elif 'prompt_encoder' in name:
            continue
        elif 'image_encoder' in name:
            continue
        elif 'memory_attention' in name:
            continue
        elif 'memory_encoder' in name:
            continue
        elif 'obj_ptr' in name:
            continue
        elif 'spatial_add_pos_embed' in name:
            continue
        elif 'maskmem' in name:
            continue
        elif 'no_mem' in name:
            continue
        elif 'no_obj' in name:
            continue
        elif 'mask_downsample' in name:
            continue
        else:
            # This would be SAM Other (Misc)
            frozen_misc.append({
                'name': name,
                'shape': list(param.shape),
                'numel': param.numel()
            })
            total_frozen_misc += param.numel()

print(f"\nFound {len(frozen_misc)} frozen SAM Other (Misc) parameters")
print(f"Total parameters: {total_frozen_misc:,}")

# Group by component prefix
component_groups = defaultdict(list)
for p in frozen_misc:
    parts = p['name'].split('.')
    if 'sam_model' in parts:
        idx = parts.index('sam_model')
        if idx + 1 < len(parts):
            component = parts[idx + 1]
            component_groups[component].append(p)
    elif 'sam' in parts[0]:
        component = parts[0]
        component_groups[component].append(p)

print("\nGrouped by component:")
for component, params in sorted(component_groups.items()):
    total = sum(p['numel'] for p in params)
    print(f"\n  {component}: {len(params)} tensors, {total:,} parameters")
    for p in params[:5]:  # Show first 5
        print(f"    - {p['name']}")
        print(f"      Shape: {p['shape']}, Size: {p['numel']:,}")
    if len(params) > 5:
        print(f"    ... and {len(params)-5} more")

# Check if these match the expected 1,599,729
print(f"\n{'='*80}")
print(f"Expected frozen SAM Other (Misc): 1,599,729")
print(f"Found frozen SAM Other (Misc): {total_frozen_misc:,}")
if total_frozen_misc == 1599729:
    print("✅ Match!")
else:
    print(f"❌ Mismatch by {abs(total_frozen_misc - 1599729):,}")