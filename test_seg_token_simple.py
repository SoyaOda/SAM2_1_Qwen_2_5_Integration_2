#!/usr/bin/env python3
"""
Simple SEG token test
"""

import torch
from transformers import AutoTokenizer
from src.models.lisa_model import LISA_Model
from src.config import LISAConfig
from src.utils.model_utils import display_parameter_statistics

# Create minimal model with SEG token
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True)
tokenizer.add_special_tokens({"additional_special_tokens": ["<SEG>"]})

config = LISAConfig(
    qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    sam_model_name="facebook/sam2.1-hiera-large",
    freeze_seg_token=False,
    use_token_fpn=False
)

print("Loading model...")
model = LISA_Model(config, tokenizer=tokenizer)
model.set_tokenizer(tokenizer, "<SEG>")

print("\nParameter statistics:")
display_parameter_statistics(model)

print(f"\nSEG token embedding exists: {hasattr(model, 'seg_token_embedding')}")
if hasattr(model, 'seg_token_embedding'):
    print(f"SEG token embedding trainable: {model.seg_token_embedding.requires_grad}")
    print(f"SEG token embedding shape: {model.seg_token_embedding.shape}")