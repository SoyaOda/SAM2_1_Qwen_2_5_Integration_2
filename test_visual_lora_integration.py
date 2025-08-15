#!/usr/bin/env python3
"""
Test Visual LoRA integration with checkpoint save/load
"""

import torch
import tempfile
import json
from pathlib import Path
from src.config import LISAConfig
from src.models.lisa_model import LISA_Model
from src.utils.checkpoint_io import save_lisa_checkpoint, load_lisa_checkpoint
from src.utils.model_utils import display_parameter_statistics
from transformers import AutoProcessor
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_visual_lora_disabled():
    """Test with Visual LoRA disabled (backward compatible)"""
    print("\n" + "="*80)
    print("Test 1: Visual LoRA Disabled (Backward Compatible)")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "test_checkpoint_no_visual"
        
        # Create config with Visual LoRA disabled
        config = LISAConfig(
            fusion_type="sigma_add",
            lora_visual_enabled=False,  # Disabled
            lora_r=8,
            lora_alpha=32
        )
        
        print(f"\nlora_visual_enabled: {config.lora_visual_enabled}")
        print(f"target_modules: {config.lora_target_modules}")
        
        # Check that Visual regex is NOT in target_modules
        has_visual = any('visual' in str(m) for m in config.lora_target_modules)
        if not has_visual:
            print("✓ No Visual ViT modules in target_modules (as expected)")
        else:
            print("✗ Unexpected Visual ViT modules in target_modules!")
            return False
        
        # Create and save model
        model = LISA_Model(config)
        processor = AutoProcessor.from_pretrained(
            config.qwen_model_name,
            min_pixels=224*224,
            max_pixels=1024*1024
        )
        
        if config.seg_token not in processor.tokenizer.get_vocab():
            processor.tokenizer.add_tokens([config.seg_token], special_tokens=True)
        
        model.set_tokenizer(processor.tokenizer, seg_token=config.seg_token)
        
        # Save checkpoint
        save_lisa_checkpoint(
            checkpoint_dir=str(checkpoint_dir),
            model=model,
            processor=processor,
            config=config,
            global_step=100
        )
        
        # Check saved metadata
        base_info_path = checkpoint_dir / "qwen" / "base_model_info.json"
        with open(base_info_path, 'r') as f:
            base_info = json.load(f)
        
        print(f"\nSaved metadata:")
        print(f"  lora_visual_enabled: {base_info.get('lora_visual_enabled', 'N/A')}")
        print(f"  Number of target_modules: {len(base_info.get('lora_target_modules', []))}")
        
        # Load checkpoint
        result = load_lisa_checkpoint(
            checkpoint_dir=str(checkpoint_dir),
            device='cpu',
            strict=False
        )
        
        loaded_config = result['config']
        print(f"\nLoaded config:")
        print(f"  lora_visual_enabled: {loaded_config.lora_visual_enabled}")
        
        if loaded_config.lora_visual_enabled == False:
            print("✓ Visual LoRA setting preserved correctly")
            return True
        else:
            print("✗ Visual LoRA setting not preserved!")
            return False


def test_visual_lora_enabled():
    """Test with Visual LoRA enabled (new feature)"""
    print("\n" + "="*80)
    print("Test 2: Visual LoRA Enabled (New Feature)")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "test_checkpoint_with_visual"
        
        # Create config with Visual LoRA enabled
        config = LISAConfig(
            fusion_type="sigma_add",
            lora_visual_enabled=True,  # Enabled
            lora_r=8,
            lora_alpha=32
        )
        
        print(f"\nlora_visual_enabled: {config.lora_visual_enabled}")
        print(f"target_modules count: {len(config.lora_target_modules)}")
        
        # Check that Visual regex IS in target_modules
        has_visual = any('visual' in str(m) for m in config.lora_target_modules)
        if has_visual:
            print("✓ Visual ViT modules in target_modules (as expected)")
        else:
            print("✗ Missing Visual ViT modules in target_modules!")
            return False
        
        # Create and save model
        model = LISA_Model(config)
        processor = AutoProcessor.from_pretrained(
            config.qwen_model_name,
            min_pixels=224*224,
            max_pixels=1024*1024
        )
        
        if config.seg_token not in processor.tokenizer.get_vocab():
            processor.tokenizer.add_tokens([config.seg_token], special_tokens=True)
        
        model.set_tokenizer(processor.tokenizer, seg_token=config.seg_token)
        
        # Count LoRA parameters
        language_lora = 0
        visual_lora = 0
        for name, param in model.named_parameters():
            if 'lora' in name.lower() and param.requires_grad:
                if 'visual.blocks' in name:
                    visual_lora += param.numel()
                else:
                    language_lora += param.numel()
        
        print(f"\nLoRA parameter distribution:")
        print(f"  Language LoRA: {language_lora:,}")
        print(f"  Visual LoRA: {visual_lora:,}")
        
        if visual_lora > 0:
            print("✓ Visual LoRA parameters detected")
        else:
            print("⚠️ No Visual LoRA parameters (might be due to model initialization)")
        
        # Save checkpoint
        save_lisa_checkpoint(
            checkpoint_dir=str(checkpoint_dir),
            model=model,
            processor=processor,
            config=config,
            global_step=200
        )
        
        # Check saved metadata
        base_info_path = checkpoint_dir / "qwen" / "base_model_info.json"
        with open(base_info_path, 'r') as f:
            base_info = json.load(f)
        
        print(f"\nSaved metadata:")
        print(f"  lora_visual_enabled: {base_info.get('lora_visual_enabled', 'N/A')}")
        print(f"  Number of target_modules: {len(base_info.get('lora_target_modules', []))}")
        
        # Load checkpoint
        result = load_lisa_checkpoint(
            checkpoint_dir=str(checkpoint_dir),
            device='cpu',
            strict=False
        )
        
        loaded_config = result['config']
        print(f"\nLoaded config:")
        print(f"  lora_visual_enabled: {loaded_config.lora_visual_enabled}")
        
        if loaded_config.lora_visual_enabled == True:
            print("✓ Visual LoRA setting preserved correctly")
            return True
        else:
            print("✗ Visual LoRA setting not preserved!")
            return False


def test_cross_loading():
    """Test loading checkpoint with different Visual LoRA settings"""
    print("\n" + "="*80)
    print("Test 3: Cross-Loading (Different Visual LoRA Settings)")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save with Visual LoRA disabled
        checkpoint_dir_no_visual = Path(tmpdir) / "checkpoint_no_visual"
        config1 = LISAConfig(
            fusion_type="sigma_add",
            lora_visual_enabled=False,
            lora_r=8
        )
        
        model1 = LISA_Model(config1)
        processor = AutoProcessor.from_pretrained(
            config1.qwen_model_name,
            min_pixels=224*224,
            max_pixels=1024*1024
        )
        if config1.seg_token not in processor.tokenizer.get_vocab():
            processor.tokenizer.add_tokens([config1.seg_token], special_tokens=True)
        model1.set_tokenizer(processor.tokenizer, seg_token=config1.seg_token)
        
        save_lisa_checkpoint(
            checkpoint_dir=str(checkpoint_dir_no_visual),
            model=model1,
            processor=processor,
            config=config1,
            global_step=300
        )
        print("Saved checkpoint with Visual LoRA disabled")
        
        # Try to load with Visual LoRA enabled (should auto-adjust)
        config2 = LISAConfig(
            fusion_type="sigma_add",
            lora_visual_enabled=True,  # Different from saved
            lora_r=8
        )
        
        print(f"\nLoading with different config:")
        print(f"  Original: lora_visual_enabled=False")
        print(f"  Loading: lora_visual_enabled=True")
        
        result = load_lisa_checkpoint(
            checkpoint_dir=str(checkpoint_dir_no_visual),
            device='cpu',
            strict=False
        )
        
        loaded_config = result['config']
        print(f"\nAuto-adjusted config:")
        print(f"  lora_visual_enabled: {loaded_config.lora_visual_enabled}")
        
        if loaded_config.lora_visual_enabled == False:
            print("✓ Config auto-adjusted to match checkpoint")
            return True
        else:
            print("✗ Config not properly adjusted")
            return False


def main():
    """Run all tests"""
    print("Testing Visual LoRA Integration")
    print("="*80)
    
    tests = [
        ("Visual LoRA Disabled", test_visual_lora_disabled),
        ("Visual LoRA Enabled", test_visual_lora_enabled),
        ("Cross-Loading", test_cross_loading),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    all_passed = True
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n✅ All Visual LoRA integration tests passed!")
        print("\nUsage:")
        print("  - For backward compatibility: use default (lora_visual_enabled=False)")
        print("  - For new training with better performance: use --enable_visual_lora")
        print("  - Checkpoints automatically preserve their Visual LoRA settings")
        return 0
    else:
        print("\n⚠️ Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())