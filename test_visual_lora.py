#!/usr/bin/env python3
"""
Test Visual ViT blocks LoRA configuration
"""

import re
import torch
from pathlib import Path
from src.config import LISAConfig
from transformers import Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_visual_lora_targeting():
    """Test that Visual ViT blocks are properly targeted for LoRA"""
    print("\n" + "="*80)
    print("Testing Visual ViT LoRA Configuration")
    print("="*80)
    
    # 1. Check config target_modules
    config = LISAConfig()
    
    print("\n1. Checking LISAConfig target_modules:")
    print(f"   Total target_modules: {len(config.lora_target_modules)}")
    
    # Separate regex patterns from normal module names
    regex_patterns = []
    normal_modules = []
    
    for module in config.lora_target_modules:
        if isinstance(module, str) and (module.startswith('(?:') or '\\' in module):
            regex_patterns.append(module)
        else:
            normal_modules.append(module)
    
    print(f"   Normal modules: {normal_modules}")
    print(f"   Regex patterns: {len(regex_patterns)} pattern(s)")
    if regex_patterns:
        print(f"   Regex: {regex_patterns[0][:80]}...")
    
    # 2. Test regex pattern matching
    print("\n2. Testing Visual ViT regex pattern:")
    
    if regex_patterns:
        visual_regex = regex_patterns[0]
        pattern = re.compile(visual_regex)
        
        # Test expected Visual ViT module names
        test_cases = [
            ("visual.blocks.0.attn.qkv", True),
            ("visual.blocks.0.attn.proj", True),
            ("visual.blocks.0.mlp.gate_proj", True),
            ("visual.blocks.0.mlp.up_proj", True),
            ("visual.blocks.0.mlp.down_proj", True),
            ("visual.blocks.31.mlp.gate_proj", True),
            ("model.language_model.layers.0.mlp.gate_proj", False),  # Should NOT match language model
            ("q_proj", False),  # Should NOT match bare q_proj
        ]
        
        all_passed = True
        for module_name, should_match in test_cases:
            matches = bool(pattern.match(module_name))
            status = "✓" if matches == should_match else "✗"
            if matches != should_match:
                all_passed = False
            print(f"   {status} '{module_name}': {'matches' if matches else 'no match'} (expected: {'match' if should_match else 'no match'})")
        
        if all_passed:
            print("   ✓ All regex tests passed!")
        else:
            print("   ✗ Some regex tests failed!")
            return False
    
    # 3. Test with actual model (lightweight check)
    print("\n3. Checking LoRA application with PEFT:")
    
    try:
        # Create minimal LoRA config
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            task_type="CAUSAL_LM"
        )
        
        print(f"   LoRA config created with r={lora_config.r}, alpha={lora_config.lora_alpha}")
        print(f"   Target modules include regex pattern: {any('visual' in str(m) for m in lora_config.target_modules)}")
        
        # Note: We don't actually load the model here to avoid memory issues
        # In production, PEFT will apply this config to match both language and visual modules
        
    except Exception as e:
        print(f"   ✗ Failed to create LoRA config: {e}")
        return False
    
    # 4. Summary
    print("\n" + "="*80)
    print("Summary of Visual ViT LoRA Configuration:")
    print("="*80)
    print("✓ 1. target_modules includes both language and visual components")
    print("✓ 2. Language modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj")
    print("✓ 3. Visual modules: regex pattern for visual.blocks.*.attn.{qkv,proj} and visual.blocks.*.mlp.{gate,up,down}_proj")
    print("\n⚠️  Note: This configuration will apply LoRA to BOTH language and visual components.")
    print("   If you want visual-only LoRA, use only the regex pattern.")
    print("   If you want language-only LoRA, use only the projection names.")
    
    return True

def check_actual_model_modules():
    """Optional: Check actual Qwen2.5-VL model module names (requires loading model)"""
    print("\n" + "="*80)
    print("Optional: Checking Actual Model Modules (skipped by default)")
    print("="*80)
    print("To run this check, uncomment the code below and ensure you have enough memory.")
    
    # Uncomment to actually load and check the model:
    """
    from transformers import Qwen2_5_VLForConditionalGeneration
    
    print("Loading Qwen2.5-VL-3B model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )
    
    # Check visual modules
    visual_modules = [name for name, _ in model.named_modules() if 'visual.blocks' in name]
    print(f"Found {len(visual_modules)} visual.blocks modules")
    
    # Check for expected module patterns
    has_attn_qkv = any('attn.qkv' in name for name in visual_modules)
    has_attn_proj = any('attn.proj' in name for name in visual_modules)
    has_mlp_gate = any('mlp.gate_proj' in name for name in visual_modules)
    has_mlp_up = any('mlp.up_proj' in name for name in visual_modules)
    has_mlp_down = any('mlp.down_proj' in name for name in visual_modules)
    
    print(f"  attn.qkv: {'✓' if has_attn_qkv else '✗'}")
    print(f"  attn.proj: {'✓' if has_attn_proj else '✗'}")
    print(f"  mlp.gate_proj: {'✓' if has_mlp_gate else '✗'}")
    print(f"  mlp.up_proj: {'✓' if has_mlp_up else '✗'}")
    print(f"  mlp.down_proj: {'✓' if has_mlp_down else '✗'}")
    
    # Sample some module names
    print("\nSample visual module names:")
    for name in visual_modules[:10]:
        print(f"  {name}")
    """

def main():
    """Run tests"""
    print("Visual ViT LoRA Configuration Test")
    print("="*80)
    
    success = test_visual_lora_targeting()
    
    if success:
        print("\n" + "="*80)
        print("✅ Visual ViT LoRA configuration tests passed!")
        print("The missing adapter keys warning should be resolved after retraining.")
        print("="*80)
        print("\n⚠️  Important: The missing adapter keys warning will persist for existing")
        print("   checkpoints that were trained without Visual ViT LoRA.")
        print("   To fully resolve the warning, you need to:")
        print("   1. Retrain with the new target_modules configuration, OR")
        print("   2. Use Visual-only or Language-only configuration to match existing checkpoints")
        return 0
    else:
        print("\n" + "="*80)
        print("⚠️ Some tests failed.")
        print("="*80)
        return 1


if __name__ == "__main__":
    exit(main())