#!/usr/bin/env python3
"""
Debug SAM MaskDecoder parameter names
"""

import torch
from transformers import AutoTokenizer
from src.models.lisa_model import LISA_Model
from src.config import LISAConfig

# Create model with LoRA
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True)
tokenizer.add_special_tokens({"additional_special_tokens": ["<SEG>"]})

config = LISAConfig(
    qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    sam_model_name="facebook/sam2.1-hiera-large",
    freeze_seg_token=False,
    sam_lora_r=8  # Enable SAM LoRA
)

print("Loading model...")
model = LISA_Model(config, tokenizer=tokenizer)
model.set_tokenizer(tokenizer, "<SEG>")

# Add SAM LoRA
model.add_sam_lora(lora_r=8, lora_alpha=16)

print("\nSAM MaskDecoder parameter names:")
for name, param in model.named_parameters():
    if 'mask_decoder' in name:
        status = "TRAINABLE" if param.requires_grad else "FROZEN"
        print(f"  {status}: {name} ({param.numel():,} params)")
        
print("\nLoRA parameter names:")
for name, param in model.named_parameters():
    if 'lora' in name.lower():
        status = "TRAINABLE" if param.requires_grad else "FROZEN"
        print(f"  {status}: {name} ({param.numel():,} params)")