#!/usr/bin/env python3
"""
Debug parameter classification
"""

from collections import defaultdict

# Simulate the parameter names from the actual log
sample_params = [
    # From training log - trainable SAM MaskDecoder parameters
    ("sam_model.sam_mask_decoder.transformer.layers.0.self_attn.q_proj.weight", True),
    ("sam_model.sam_mask_decoder.transformer.layers.0.self_attn.q_proj.bias", True),
    ("sam_model.sam_mask_decoder.transformer.layers.0.self_attn.k_proj.weight", True),
    ("sam_model.sam_mask_decoder.transformer.layers.0.self_attn.k_proj.bias", True),
    ("sam_model.sam_mask_decoder.transformer.layers.0.self_attn.v_proj.weight", True),
    ("sam_model.sam_mask_decoder.transformer.layers.0.self_attn.v_proj.bias", True),
    ("sam_model.sam_mask_decoder.transformer.layers.0.self_attn.out_proj.weight", True),
    ("sam_model.sam_mask_decoder.transformer.layers.0.self_attn.out_proj.bias", True),
    ("sam_model.sam_mask_decoder.transformer.layers.0.norm1.weight", True),
    ("sam_model.sam_mask_decoder.transformer.layers.0.norm1.bias", True),
    
    # LoRA parameters
    ("sam_model.sam_mask_decoder.transformer.layers.0.self_attn.q_proj.lora_A", True),
    ("sam_model.sam_mask_decoder.transformer.layers.0.self_attn.q_proj.lora_B", True),
    ("sam_model.sam_mask_decoder.transformer.layers.0.self_attn.k_proj.lora_A", True),
    ("sam_model.sam_mask_decoder.transformer.layers.0.self_attn.k_proj.lora_B", True),
    
    # Frozen SAM MaskDecoder parameters (hypothetical)
    ("sam_model.sam_mask_decoder.mask_tokens.weight", False),
    ("sam_model.sam_mask_decoder.output_upscaling.0.weight", False),
]

def classify_parameter(name, is_trainable):
    """Classify parameter using updated logic"""
    if 'sam' in name or 'sam_model' in name:
        if 'lora_A' in name or 'lora_B' in name or 'lora' in name.lower():
            # LoRAパラメータ（lora_A, lora_B）またはLoRA関連
            category = 'SAM LoRA (freeze_sam_lora)'
        elif 'mask_decoder' in name:
            # MaskDecoder内のパラメータ
            # LoRAが適用されているレイヤーかどうかで判断
            if ('q_proj' in name or 'k_proj' in name or 'v_proj' in name or 
                'out_proj' in name) and is_trainable:
                # LoRAが適用されたattentionレイヤー（学習可能な場合はLoRAとして分類）
                category = 'SAM LoRA (freeze_sam_lora)'
            else:
                # 通常のMaskDecoderパラメータ
                category = 'SAM MaskDecoder Base (freeze_sam_mask_decoder_base)'
        else:
            category = 'SAM Other'
    else:
        category = 'Other'
    
    return category

print("Parameter classification analysis:")
print("="*80)

categories = defaultdict(lambda: {'total': 0, 'trainable': 0})

for name, is_trainable in sample_params:
    category = classify_parameter(name, is_trainable)
    print(f"{category:50} {'TRAINABLE' if is_trainable else 'FROZEN':9} {name}")
    
    # Simulate parameter counts
    param_count = 1000  # dummy value
    categories[category]['total'] += param_count
    if is_trainable:
        categories[category]['trainable'] += param_count

print("\n" + "="*80)
print("Category summary:")
for category, counts in categories.items():
    frozen = counts['total'] - counts['trainable']
    print(f"{category:50} Total: {counts['total']:5}, Trainable: {counts['trainable']:5}, Frozen: {frozen:5}")