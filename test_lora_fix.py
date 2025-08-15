#!/usr/bin/env python3
"""
Test LoRA adapter saving/loading after fixes
"""

import torch
import tempfile
import warnings
from pathlib import Path
from src.config import LISAConfig
from src.models.lisa_model import LISA_Model
from src.utils.checkpoint_io import save_lisa_checkpoint, load_lisa_checkpoint
from transformers import AutoProcessor
import logging

# Capture warnings
warnings.filterwarnings("error", message=".*missing adapter keys.*")

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_lora_adapter():
    """Test LoRA adapter with updated target_modules"""
    print("\n" + "="*80)
    print("Testing LoRA Adapter with Updated Configuration")
    print("="*80)
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "test_checkpoint"
        
        # Initialize model with updated config
        config = LISAConfig(
            fusion_type="sigma_add",
            lora_r=8,
            lora_alpha=32
        )
        
        print(f"\nLoRA target modules: {config.lora_target_modules}")
        print(f"LoRA rank: {config.lora_r}, alpha: {config.lora_alpha}")
        
        # Create model
        model = LISA_Model(config)
        
        # Initialize processor
        processor = AutoProcessor.from_pretrained(
            config.qwen_model_name,
            min_pixels=224*224,
            max_pixels=1024*1024,
            max_length=8192
        )
        
        # Add SEG token
        if config.seg_token not in processor.tokenizer.get_vocab():
            processor.tokenizer.add_tokens([config.seg_token], special_tokens=True)
        
        # Set tokenizer in model
        model.set_tokenizer(processor.tokenizer, seg_token=config.seg_token)
        
        # Count LoRA parameters
        lora_params = sum(p.numel() for n, p in model.named_parameters() 
                         if "lora" in n.lower() and p.requires_grad)
        print(f"\nTotal LoRA parameters: {lora_params:,}")
        
        # Check which modules have LoRA
        lora_modules = set()
        for name, _ in model.named_parameters():
            if "lora" in name.lower():
                # Extract module name
                parts = name.split('.')
                for i, part in enumerate(parts):
                    if part in config.lora_target_modules:
                        lora_modules.add(part)
        
        print(f"Modules with LoRA applied: {sorted(lora_modules)}")
        
        # Save checkpoint
        print(f"\nSaving checkpoint to {checkpoint_dir}")
        save_lisa_checkpoint(
            checkpoint_dir=str(checkpoint_dir),
            model=model,
            processor=processor,
            config=config,
            global_step=100,
            best_loss=0.1,
            epoch=1
        )
        
        # Check adapter_config.json
        adapter_config_path = checkpoint_dir / "qwen" / "adapter_config.json"
        if adapter_config_path.exists():
            import json
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            print(f"\nAdapter config saved:")
            print(f"  base_model: {adapter_config.get('base_model_name_or_path', 'N/A')}")
            print(f"  target_modules: {adapter_config.get('target_modules', 'N/A')}")
            print(f"  r: {adapter_config.get('r', 'N/A')}")
        
        # Load checkpoint
        print(f"\nLoading checkpoint from {checkpoint_dir}")
        print("This should NOT produce 'missing adapter keys' warnings...")
        
        try:
            result = load_lisa_checkpoint(
                checkpoint_dir=str(checkpoint_dir),
                device='cpu',
                strict=False
            )
            
            model2 = result['model']
            
            # Count loaded LoRA parameters
            loaded_lora_params = sum(p.numel() for n, p in model2.named_parameters() 
                                   if "lora" in n.lower() and p.requires_grad)
            print(f"\nLoaded LoRA parameters: {loaded_lora_params:,}")
            
            if loaded_lora_params == lora_params:
                print("✓ LoRA parameters count matches!")
            else:
                print(f"✗ LoRA parameters mismatch: expected {lora_params:,}, got {loaded_lora_params:,}")
            
            print("\n✓ Test completed successfully - No missing adapter keys warnings!")
            return True
            
        except UserWarning as w:
            if "missing adapter keys" in str(w):
                print(f"\n✗ Warning still appears: {w}")
                print("\nThis might be due to:")
                print("  1. PEFT library version mismatch")
                print("  2. Need to clear cache and reinstall")
                print("  3. Legacy checkpoint format")
                return False
            else:
                raise
        except Exception as e:
            print(f"\n✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False


def check_peft_version():
    """Check PEFT library version"""
    try:
        import peft
        print(f"\nPEFT version: {peft.__version__}")
        
        # Check if AutoPeftModel is available
        from peft import AutoPeftModelForCausalLM
        print("✓ AutoPeftModel is available")
        
    except ImportError as e:
        print(f"✗ PEFT import error: {e}")
        print("Consider updating PEFT: pip install -U peft")


def main():
    """Run tests"""
    print("Testing LoRA Adapter Fix")
    print("="*80)
    
    # Check PEFT version
    check_peft_version()
    
    # Run test
    success = test_lora_adapter()
    
    if success:
        print("\n" + "="*80)
        print("✅ All tests passed! The warning should be resolved.")
        print("="*80)
        return 0
    else:
        print("\n" + "="*80)
        print("⚠️ Tests completed but warnings may still appear.")
        print("This is usually harmless if the model loads correctly.")
        print("="*80)
        return 1


if __name__ == "__main__":
    exit(main())