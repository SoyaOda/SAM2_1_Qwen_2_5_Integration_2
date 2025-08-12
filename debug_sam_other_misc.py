#!/usr/bin/env python3
"""
Debug script to identify SAM Other (Misc) parameters
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
    freeze_sam_mask_decoder_base=False,  # Unfreeze to see trainable params
    sam_lora_r=0  # No LoRA
)

print("Loading model...")
model = LISA_Model(config, tokenizer=tokenizer)
model.set_tokenizer(tokenizer, "<SEG>")

print("\n" + "="*80)
print("Analyzing SAM Other (Misc) parameters:")
print("="*80)

# Track SAM parameters by category
sam_categories = defaultdict(list)

for name, param in model.named_parameters():
    if 'sam' in name.lower() or 'sam_model' in name.lower():
        # Categorize the parameter
        if 'lora_A' in name or 'lora_B' in name:
            category = 'LoRA'
        elif 'mask_decoder' in name:
            category = 'MaskDecoder'
        elif 'prompt_encoder' in name:
            category = 'PromptEncoder'
        elif 'image_encoder' in name:
            category = 'ImageEncoder'
        elif 'memory_attention' in name:
            category = 'MemoryAttention'
        elif 'memory_encoder' in name:
            category = 'MemoryEncoder'
        elif 'obj_ptr' in name:
            category = 'ObjectPointer'
        elif 'spatial_add_pos_embed' in name:
            category = 'SpatialPosEmbed'
        elif 'maskmem' in name:
            category = 'MaskMemory'
        elif 'no_mem' in name:
            category = 'NoMemory'
        elif 'no_obj' in name:
            category = 'NoObject'
        elif 'mask_downsample' in name:
            category = 'MaskDownsample'
        else:
            # これがSAM Other (Misc)に該当するパラメータ
            category = 'Other (Misc)'
        
        sam_categories[category].append({
            'name': name,
            'shape': list(param.shape),
            'numel': param.numel(),
            'trainable': param.requires_grad
        })

# Display results
for category, params in sorted(sam_categories.items()):
    if category == 'Other (Misc)' and params:
        print(f"\n{category}: {len(params)} parameters")
        print("-" * 40)
        
        # Group by prefix for better understanding
        prefix_groups = defaultdict(list)
        for p in params:
            # Extract prefix (first few parts of the name)
            parts = p['name'].split('.')
            if len(parts) > 2:
                prefix = '.'.join(parts[:3])
            else:
                prefix = '.'.join(parts[:2])
            prefix_groups[prefix].append(p)
        
        for prefix, group in sorted(prefix_groups.items()):
            total_params = sum(p['numel'] for p in group)
            trainable_count = sum(1 for p in group if p['trainable'])
            print(f"\n  Prefix: {prefix}")
            print(f"    Total params: {total_params:,}")
            print(f"    Trainable: {trainable_count}/{len(group)} tensors")
            
            # Show first few examples
            for p in group[:3]:
                status = "TRAINABLE" if p['trainable'] else "FROZEN"
                print(f"      [{status}] {p['name']}")
                print(f"          Shape: {p['shape']}, Size: {p['numel']:,}")
            
            if len(group) > 3:
                print(f"      ... and {len(group)-3} more")

# Summary statistics
print("\n" + "="*80)
print("Summary of SAM Other (Misc) parameters:")
print("="*80)

if 'Other (Misc)' in sam_categories:
    misc_params = sam_categories['Other (Misc)']
    total_misc = sum(p['numel'] for p in misc_params)
    trainable_misc = sum(p['numel'] for p in misc_params if p['trainable'])
    frozen_misc = total_misc - trainable_misc
    
    print(f"Total Other (Misc) parameters: {total_misc:,}")
    print(f"  - Trainable: {trainable_misc:,} ({trainable_misc/total_misc*100:.1f}%)")
    print(f"  - Frozen: {frozen_misc:,} ({frozen_misc/total_misc*100:.1f}%)")
    
    # Find unique components
    unique_components = set()
    for p in misc_params:
        parts = p['name'].split('.')
        if 'sam_model' in parts:
            idx = parts.index('sam_model')
            if idx + 1 < len(parts):
                unique_components.add(parts[idx + 1])
    
    print(f"\nUnique SAM components in Other (Misc):")
    for comp in sorted(unique_components):
        print(f"  - {comp}")
else:
    print("No SAM Other (Misc) parameters found")

print("\n" + "="*80)
print("All SAM categories found:")
for category in sorted(sam_categories.keys()):
    count = len(sam_categories[category])
    total = sum(p['numel'] for p in sam_categories[category])
    print(f"  - {category}: {count} tensors, {total:,} parameters")