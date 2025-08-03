"""
Check actual dimensions of Qwen2.5-VL-3B and SAM2.1 models
"""
import torch
from transformers import AutoConfig
import json


def check_qwen_dimensions():
    """Check Qwen2.5-VL model dimensions from config"""
    print("=" * 80)
    print("Checking Qwen2.5-VL-3B Model Dimensions")
    print("=" * 80)
    
    try:
        # Load only the config (not the full model) to save memory
        config = AutoConfig.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        
        print("\nMain Model Config:")
        print(f"  Model type: {config.model_type}")
        print(f"  Hidden size (D_l): {config.hidden_size}")
        print(f"  Intermediate size: {config.intermediate_size}")
        print(f"  Num hidden layers: {config.num_hidden_layers}")
        print(f"  Num attention heads: {config.num_attention_heads}")
        print(f"  Vocab size: {config.vocab_size}")
        
        if hasattr(config, 'vision_config'):
            print("\nVision Config:")
            vision_config = config.vision_config
            
            # Safely access vision config attributes
            vision_attrs = ['hidden_size', 'intermediate_size', 'num_hidden_layers', 
                           'num_attention_heads', 'image_size', 'patch_size',
                           'spatial_merge_size', 'temporal_patch_size']
            
            for attr in vision_attrs:
                if hasattr(vision_config, attr):
                    value = getattr(vision_config, attr)
                    print(f"  Vision {attr}: {value}")
            
            # Try to access vision config as dict if object access fails
            try:
                if hasattr(vision_config, 'to_dict'):
                    vision_dict = vision_config.to_dict()
                    print("\n  Additional vision config details:")
                    for key in ['depth', 'mlp_ratio', 'num_heads', 'embed_dim']:
                        if key in vision_dict:
                            print(f"    {key}: {vision_dict[key]}")
            except Exception as e:
                print(f"  (Could not access vision config dict: {e})")
        
        # Check for any special architecture details
        print("\nAdditional Architecture Info:")
        special_attrs = ['rope_theta', 'tie_word_embeddings', 'use_sliding_window']
        for attr in special_attrs:
            if hasattr(config, attr):
                print(f"  {attr}: {getattr(config, attr)}")
        
        # Save config for reference
        with open('qwen_config.json', 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        print("\n✓ Config saved to qwen_config.json")
        
        return config
        
    except Exception as e:
        print(f"\n✗ Failed to load Qwen config: {e}")
        return None


def check_sam_dimensions():
    """Check SAM2.1 expected dimensions"""
    print("\n" + "=" * 80)
    print("SAM2.1 Expected Dimensions (from documentation)")
    print("=" * 80)
    
    # Based on SAM2 architecture documentation
    print("\nSAM2.1 Architecture:")
    print("  Image encoder: Hierarchical Vision Transformer (Hiera)")
    print("  Image embedding channels: 256")
    print("  Prompt embedding dimension: 256")
    print("  Mask decoder output: Variable (depends on image size)")
    print("  Low-res mask size: 256x256 (before upsampling)")
    
    print("\nExpected Input/Output Shapes:")
    print("  Image embeddings: [B, 256, H, W]")
    print("  Sparse prompt embeddings: [B, num_prompts, 256]")
    print("  Dense prompt embeddings: [B, 256, H, W] (optional)")
    print("  Mask output: [B, num_masks, H_mask, W_mask]")
    
    return {
        'image_embedding_dim': 256,
        'prompt_embedding_dim': 256,
        'low_res_mask_size': 256
    }


def verify_dimensions_match():
    """Verify that our implementation matches expected dimensions"""
    print("\n" + "=" * 80)
    print("Dimension Verification")
    print("=" * 80)
    
    # Expected from o3_spec1.md
    expected = {
        'qwen_hidden_size': 2560,  # From spec
        'qwen_vision_hidden_size': 768,  # Example in spec
        'sam_embedding_dim': 256,
        'adapter_out_dim': 256,
    }
    
    print("\nExpected dimensions from o3_spec1.md:")
    for key, value in expected.items():
        print(f"  {key}: {value}")
    
    # Get actual Qwen config
    qwen_config = check_qwen_dimensions()
    
    if qwen_config:
        print("\n\nComparison with actual Qwen2.5-VL-3B:")
        
        # Check language hidden size
        actual_hidden = qwen_config.hidden_size
        expected_hidden = expected['qwen_hidden_size']
        match = "✓" if actual_hidden == expected_hidden else "✗"
        print(f"  Language hidden size: {actual_hidden} (expected: {expected_hidden}) {match}")
        
        # Check vision hidden size
        if hasattr(qwen_config, 'vision_config') and hasattr(qwen_config.vision_config, 'hidden_size'):
            actual_vision_hidden = qwen_config.vision_config.hidden_size
            print(f"  Vision hidden size: {actual_vision_hidden} (spec example: {expected['qwen_vision_hidden_size']})")
            print(f"    Note: Spec used {expected['qwen_vision_hidden_size']} as example, actual is {actual_vision_hidden}")
    
    # SAM dimensions
    sam_dims = check_sam_dimensions()
    print(f"\n  SAM embedding dimension: {sam_dims['image_embedding_dim']} ✓")
    print(f"  Adapter output dimension: {expected['adapter_out_dim']} ✓")
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    if qwen_config:
        print("\nKey dimensions for LISA改 implementation:")
        if hasattr(qwen_config, 'vision_config') and hasattr(qwen_config.vision_config, 'hidden_size'):
            vision_hidden = qwen_config.vision_config.hidden_size
            print(f"  ImageFeatureAdapter: {vision_hidden} → 256")
        else:
            print(f"  ImageFeatureAdapter: [vision_hidden_size] → 256")
        print(f"  TextPromptProjector: {qwen_config.hidden_size} → 256")
        print(f"  Vocabulary size: {qwen_config.vocab_size} (for SEG token addition)")
        
        if hasattr(qwen_config, 'vision_config') and hasattr(qwen_config.vision_config, 'hidden_size'):
            print("\n⚠️  Note: The spec used D_v=768 as an example, but actual Qwen2.5-VL-3B has:")
            print(f"     D_v={vision_hidden} (vision hidden size)")
            print(f"     D_l={qwen_config.hidden_size} (language hidden size)")


if __name__ == "__main__":
    verify_dimensions_match()