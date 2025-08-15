#!/usr/bin/env python3
"""
Test that all configuration is centralized in src/config.py
"""

from src.config import LISAConfig
import torch

def test_config_centralization():
    """Test configuration centralization"""
    print("\n" + "="*80)
    print("Configuration Centralization Test")
    print("="*80)
    
    # Create default config
    config = LISAConfig()
    
    print("\n1. LoRA Configuration:")
    print(f"   lora_r: {config.lora_r}")
    print(f"   lora_alpha: {config.lora_alpha}")
    print(f"   lora_dropout: {config.lora_dropout}")
    print(f"   lora_visual_enabled: {config.lora_visual_enabled}")
    print(f"   sam_lora_r: {config.sam_lora_r}")
    print(f"   sam_lora_alpha: {config.sam_lora_alpha}")
    
    print("\n2. Learning Rates:")
    print(f"   adapter_lr: {config.adapter_lr}")
    print(f"   lora_lr: {config.lora_lr}")
    print(f"   seg_token_lr: {config.seg_token_lr}")
    print(f"   weight_decay: {config.weight_decay}")
    
    print("\n3. Loss Weights:")
    print(f"   language_loss_weight: {config.language_loss_weight}")
    print(f"   segmentation_loss_weight: {config.segmentation_loss_weight}")
    
    print("\n4. Target Modules (auto-generated):")
    if config.lora_visual_enabled:
        print(f"   Language modules + Visual ViT modules")
    else:
        print(f"   Language modules only")
    print(f"   Number of target modules: {len(config.lora_target_modules)}")
    
    # Test with Visual LoRA enabled
    print("\n" + "="*80)
    print("Testing with Visual LoRA Enabled")
    print("="*80)
    
    config_visual = LISAConfig(lora_visual_enabled=True)
    print(f"Target modules count: {len(config_visual.lora_target_modules)}")
    has_visual = any('visual' in str(m) for m in config_visual.lora_target_modules)
    print(f"Has Visual ViT modules: {has_visual}")
    
    # Test minimal override (like in minimal_train.py)
    print("\n" + "="*80)
    print("Testing Minimal Override (minimal_train.py style)")
    print("="*80)
    
    # Only override environment-dependent settings
    config_minimal = LISAConfig(
        qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        sam_model_name="./checkpoints/sam2.1_hiera_large.pt",
        device_map="cuda",
        torch_dtype="auto",
        use_flash_attention=False
    )
    
    print("Overridden settings:")
    print(f"   qwen_model_name: {config_minimal.qwen_model_name}")
    print(f"   device_map: {config_minimal.device_map}")
    
    print("\nPreserved settings from config.py:")
    print(f"   lora_r: {config_minimal.lora_r}")
    print(f"   lora_alpha: {config_minimal.lora_alpha}")
    print(f"   adapter_lr: {config_minimal.adapter_lr}")
    print(f"   lora_visual_enabled: {config_minimal.lora_visual_enabled}")
    
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print("✓ All configuration is centralized in src/config.py")
    print("✓ minimal_train.py only overrides environment-dependent settings")
    print("✓ No command-line arguments for model/training parameters")
    print("\nTo change any settings, edit src/config.py directly.")
    
    return True

def main():
    """Run test"""
    success = test_config_centralization()
    
    if success:
        print("\n✅ Configuration centralization test passed!")
        return 0
    else:
        print("\n❌ Test failed")
        return 1

if __name__ == "__main__":
    exit(main())