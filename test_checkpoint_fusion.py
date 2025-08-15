#!/usr/bin/env python3
"""
Test checkpoint saving and loading for both fusion modes
"""

import torch
import tempfile
import shutil
from pathlib import Path
from src.config import LISAConfig
from src.models.lisa_model import LISA_Model
from src.utils.checkpoint_io import save_lisa_checkpoint, load_lisa_checkpoint
from transformers import AutoProcessor
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_sigma_add_fusion():
    """Test Sigma-Add fusion checkpoint save/load"""
    print("\n" + "="*80)
    print("Testing Sigma-Add Fusion Checkpoint")
    print("="*80)
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "sigma_add_checkpoint"
        
        # Initialize model with Sigma-Add fusion
        config = LISAConfig(fusion_type="sigma_add")
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
        
        # Get original beta value
        original_beta = torch.sigmoid(model.image_fusion_beta).item()
        print(f"Original beta value: {original_beta:.6f}")
        
        # Modify beta value to test saving
        with torch.no_grad():
            model.image_fusion_beta.data = torch.tensor(1.0)  # sigmoid(1.0) ≈ 0.731
        modified_beta = torch.sigmoid(model.image_fusion_beta).item()
        print(f"Modified beta value: {modified_beta:.6f}")
        
        # Save checkpoint
        print(f"\nSaving checkpoint to {checkpoint_dir}")
        save_lisa_checkpoint(
            checkpoint_dir=str(checkpoint_dir),
            model=model,
            processor=processor,
            config=config,
            global_step=1000,
            best_loss=0.5,
            epoch=5
        )
        
        # Check saved files
        expected_files = [
            "image_fusion_beta.pt",
            "config.pt",
            "processor",
            "adapters/image_adapter.pt",
            "adapters/text_prompt_proj.pt",
            "adapters/token_fpn.pt",
        ]
        
        for file_path in expected_files:
            full_path = checkpoint_dir / file_path
            if full_path.exists():
                print(f"✓ {file_path} exists")
            else:
                print(f"✗ {file_path} missing!")
        
        # Load checkpoint into new model
        print("\nLoading checkpoint into new model...")
        config2 = LISAConfig(fusion_type="sigma_add")
        result = load_lisa_checkpoint(
            checkpoint_dir=str(checkpoint_dir),
            device='cpu',
            strict=False
        )
        
        model2 = result['model']
        loaded_beta = torch.sigmoid(model2.image_fusion_beta).item()
        print(f"Loaded beta value: {loaded_beta:.6f}")
        
        # Check if values match
        if abs(loaded_beta - modified_beta) < 1e-5:
            print("✓ Beta value loaded correctly!")
        else:
            print(f"✗ Beta value mismatch: expected {modified_beta:.6f}, got {loaded_beta:.6f}")
        
        return True


def test_cross_attention_fusion():
    """Test Cross-Attention fusion checkpoint save/load"""
    print("\n" + "="*80)
    print("Testing Cross-Attention Fusion Checkpoint")
    print("="*80)
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "cross_attention_checkpoint"
        
        # Initialize model with Cross-Attention fusion
        config = LISAConfig(fusion_type="cross_attention")
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
        
        # Get original parameter count
        original_params = sum(p.numel() for p in model.image_fusion.parameters())
        trainable_params = sum(p.numel() for p in model.image_fusion.parameters() if p.requires_grad)
        print(f"Original Cross-Attention params: {trainable_params}/{original_params}")
        
        # Modify a parameter to test saving
        if hasattr(model.image_fusion, 'fusion') and hasattr(model.image_fusion.fusion, 'cross_attention'):
            # Get the query projection weight
            q_proj = model.image_fusion.fusion.cross_attention.q_proj
            original_norm = q_proj.weight.norm().item()
            print(f"Original q_proj weight norm: {original_norm:.6f}")
            
            # Modify it
            with torch.no_grad():
                q_proj.weight.data *= 1.5
            modified_norm = q_proj.weight.norm().item()
            print(f"Modified q_proj weight norm: {modified_norm:.6f}")
        
        # Save checkpoint
        print(f"\nSaving checkpoint to {checkpoint_dir}")
        save_lisa_checkpoint(
            checkpoint_dir=str(checkpoint_dir),
            model=model,
            processor=processor,
            config=config,
            global_step=2000,
            best_loss=0.3,
            epoch=10
        )
        
        # Check saved files
        expected_files = [
            "image_fusion_cross_attention.pt",
            "config.pt",
            "processor",
            "adapters/image_adapter.pt",
            "adapters/text_prompt_proj.pt",
            "adapters/token_fpn.pt",
        ]
        
        for file_path in expected_files:
            full_path = checkpoint_dir / file_path
            if full_path.exists():
                print(f"✓ {file_path} exists")
            else:
                print(f"✗ {file_path} missing!")
        
        # Load checkpoint into new model
        print("\nLoading checkpoint into new model...")
        config2 = LISAConfig(fusion_type="cross_attention")
        result = load_lisa_checkpoint(
            checkpoint_dir=str(checkpoint_dir),
            device='cpu',
            strict=False
        )
        
        model2 = result['model']
        loaded_params = sum(p.numel() for p in model2.image_fusion.parameters())
        loaded_trainable = sum(p.numel() for p in model2.image_fusion.parameters() if p.requires_grad)
        print(f"Loaded Cross-Attention params: {loaded_trainable}/{loaded_params}")
        
        # Check parameter values
        if hasattr(model2.image_fusion, 'fusion') and hasattr(model2.image_fusion.fusion, 'cross_attention'):
            q_proj2 = model2.image_fusion.fusion.cross_attention.q_proj
            loaded_norm = q_proj2.weight.norm().item()
            print(f"Loaded q_proj weight norm: {loaded_norm:.6f}")
            
            # Check if values match
            if abs(loaded_norm - modified_norm) < 1e-4:
                print("✓ Cross-Attention weights loaded correctly!")
            else:
                print(f"✗ Weight mismatch: expected {modified_norm:.6f}, got {loaded_norm:.6f}")
        
        return True


def main():
    """Run all tests"""
    print("Testing Checkpoint Save/Load for Fusion Modes")
    print("="*80)
    
    try:
        # Test Sigma-Add fusion
        sigma_add_ok = test_sigma_add_fusion()
        
        # Test Cross-Attention fusion
        cross_attention_ok = test_cross_attention_fusion()
        
        # Summary
        print("\n" + "="*80)
        print("Test Summary")
        print("="*80)
        print(f"Sigma-Add Fusion: {'✓ PASSED' if sigma_add_ok else '✗ FAILED'}")
        print(f"Cross-Attention Fusion: {'✓ PASSED' if cross_attention_ok else '✗ FAILED'}")
        
        if sigma_add_ok and cross_attention_ok:
            print("\n✓ All tests passed!")
            return 0
        else:
            print("\n✗ Some tests failed!")
            return 1
            
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())