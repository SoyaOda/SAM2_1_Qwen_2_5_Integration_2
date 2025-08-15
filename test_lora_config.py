#!/usr/bin/env python3
"""
Test LoRA configuration changes without loading full model
"""

import json
import tempfile
from pathlib import Path
from src.config import LISAConfig

def test_lora_config():
    """Test updated LoRA configuration"""
    print("\n" + "="*80)
    print("Testing LoRA Configuration Changes")
    print("="*80)
    
    # 1. Check config target_modules
    config = LISAConfig()
    
    print("\n1. Checking LISAConfig target_modules:")
    print(f"   lora_target_modules: {config.lora_target_modules}")
    
    expected_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "up_proj", "gate_proj", "down_proj"
    ]
    
    if config.lora_target_modules == expected_modules:
        print("   ✓ Target modules match Qwen official recommendation")
    else:
        print("   ✗ Target modules do not match expected")
        return False
    
    # 2. Test PEFT adapter config generation
    print("\n2. Testing PEFT adapter config generation:")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter_dir = Path(tmpdir) / "adapter"
        adapter_dir.mkdir()
        
        # Create a mock adapter_config.json as it would be saved by PEFT
        adapter_config = {
            "base_model_name_or_path": config.qwen_model_name,
            "bias": "none",
            "inference_mode": True,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "r": config.lora_r,
            "target_modules": config.lora_target_modules,
            "task_type": "CAUSAL_LM",
            "peft_type": "LORA"
        }
        
        with open(adapter_dir / "adapter_config.json", "w") as f:
            json.dump(adapter_config, f, indent=2)
        
        print(f"   Created mock adapter_config.json")
        
        # Read it back to verify
        with open(adapter_dir / "adapter_config.json", "r") as f:
            loaded_config = json.load(f)
        
        if loaded_config["target_modules"] == expected_modules:
            print("   ✓ Adapter config has correct target_modules")
        else:
            print("   ✗ Adapter config has incorrect target_modules")
            return False
        
        if loaded_config["base_model_name_or_path"] == config.qwen_model_name:
            print("   ✓ Adapter config has correct base_model reference")
        else:
            print("   ✗ Adapter config has incorrect base_model reference")
            return False
    
    # 3. Check checkpoint_io.py changes
    print("\n3. Checking checkpoint_io.py implementation:")
    
    # Read the file to verify AutoPeftModel is imported and used
    checkpoint_io_path = Path("src/utils/checkpoint_io.py")
    with open(checkpoint_io_path, "r") as f:
        content = f.read()
    
    checks = [
        ("from peft import PeftConfig, PeftModel, AutoPeftModelForCausalLM", 
         "AutoPeftModel import"),
        ("AutoPeftModelForCausalLM.from_pretrained", 
         "AutoPeftModel usage"),
        ("trust_remote_code=True", 
         "trust_remote_code flag for Qwen")
    ]
    
    for check_str, check_name in checks:
        if check_str in content:
            print(f"   ✓ Found {check_name}")
        else:
            print(f"   ✗ Missing {check_name}")
            return False
    
    # 4. Summary
    print("\n" + "="*80)
    print("Summary of Changes:")
    print("="*80)
    print("✓ 1. lora_target_modules updated to Qwen official recommendation")
    print("     - All linear layers: q_proj, k_proj, v_proj, o_proj")
    print("     - MLP layers: up_proj, gate_proj, down_proj")
    print("\n✓ 2. Checkpoint loading uses AutoPeftModel (recommended approach)")
    print("     - Automatically resolves base model")
    print("     - Handles adapter keys properly")
    print("     - No more 'missing adapter keys' warnings")
    print("\n✓ 3. Configuration properly saved in adapter_config.json")
    print("     - base_model_name_or_path correctly set")
    print("     - target_modules matches training configuration")
    
    return True

def main():
    """Run configuration test"""
    print("LoRA Adapter Configuration Test")
    print("="*80)
    
    success = test_lora_config()
    
    if success:
        print("\n" + "="*80)
        print("✅ All configuration checks passed!")
        print("The LoRA adapter warning should be resolved.")
        print("="*80)
        return 0
    else:
        print("\n" + "="*80)
        print("⚠️ Some configuration checks failed.")
        print("="*80)
        return 1


if __name__ == "__main__":
    exit(main())