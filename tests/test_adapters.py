"""
Test script for ImageFeatureAdapter and TextPromptProjector modules
"""
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.adapters import ImageFeatureAdapter, TextPromptProjector
from src.config import LISAConfig


def test_image_feature_adapter():
    """Test ImageFeatureAdapter with various input shapes"""
    print("=" * 60)
    print("Testing ImageFeatureAdapter")
    print("=" * 60)
    
    # Test configurations
    test_configs = [
        # (batch_size, num_patches, input_dim, output_dim)
        (1, 256, 768, 256),    # 16x16 patches
        (2, 1024, 1024, 256),  # 32x32 patches
        (1, 576, 768, 256),    # 24x24 patches (non-power-of-2)
        (4, 64, 512, 256),     # 8x8 patches
    ]
    
    for batch_size, num_patches, in_dim, out_dim in test_configs:
        print(f"\nTest case: B={batch_size}, N={num_patches}, D_in={in_dim}, D_out={out_dim}")
        
        # Create adapter
        adapter = ImageFeatureAdapter(in_dim=in_dim, out_dim=out_dim)
        
        # Create dummy input
        vision_features = torch.randn(batch_size, num_patches, in_dim)
        
        # Forward pass
        output = adapter(vision_features)
        
        # Calculate expected spatial dimensions
        expected_shape = adapter.get_output_shape((batch_size, num_patches, in_dim))
        
        print(f"  Input shape: {vision_features.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected shape: {expected_shape}")
        
        # Verify output shape
        assert output.shape == expected_shape, f"Shape mismatch: {output.shape} != {expected_shape}"
        assert output.shape[1] == out_dim, f"Output channel mismatch: {output.shape[1]} != {out_dim}"
        
        # Check that output is contiguous
        assert output.is_contiguous(), "Output tensor is not contiguous"
        
        print(f"  ✓ Test passed!")
    
    # Test gradient flow
    print("\nTesting gradient flow...")
    adapter = ImageFeatureAdapter(in_dim=768, out_dim=256)
    vision_features = torch.randn(2, 256, 768, requires_grad=True)
    output = adapter(vision_features)
    loss = output.sum()
    loss.backward()
    
    assert vision_features.grad is not None, "No gradient computed for input"
    assert all(p.grad is not None for p in adapter.parameters()), "No gradient computed for adapter parameters"
    print("  ✓ Gradient flow test passed!")


def test_text_prompt_projector():
    """Test TextPromptProjector with various configurations"""
    print("\n" + "=" * 60)
    print("Testing TextPromptProjector")
    print("=" * 60)
    
    # Test configurations
    test_configs = [
        # (batch_size, in_dim, out_dim, use_mlp)
        (1, 2048, 256, False),  # Qwen2.5-VL-3B actual dimension, linear projection
        (4, 2048, 256, True),   # With MLP
        (2, 4096, 256, False),  # Larger model dimension
        (8, 1024, 256, True),   # Smaller model dimension
    ]
    
    for batch_size, in_dim, out_dim, use_mlp in test_configs:
        print(f"\nTest case: B={batch_size}, D_in={in_dim}, D_out={out_dim}, MLP={use_mlp}")
        
        # Create projector
        projector = TextPromptProjector(in_dim=in_dim, out_dim=out_dim, use_mlp=use_mlp)
        
        # Test with 2D input [B, D_l]
        text_hidden_2d = torch.randn(batch_size, in_dim)
        output_2d = projector(text_hidden_2d)
        
        print(f"  2D Input shape: {text_hidden_2d.shape}")
        print(f"  2D Output shape: {output_2d.shape}")
        
        assert output_2d.shape == (batch_size, out_dim), f"2D shape mismatch: {output_2d.shape}"
        
        # Test with 3D input [B, 1, D_l]
        text_hidden_3d = torch.randn(batch_size, 1, in_dim)
        output_3d = projector(text_hidden_3d)
        
        print(f"  3D Input shape: {text_hidden_3d.shape}")
        print(f"  3D Output shape: {output_3d.shape}")
        
        assert output_3d.shape == (batch_size, out_dim), f"3D shape mismatch: {output_3d.shape}"
        
        print(f"  ✓ Basic test passed!")
    
    # Test forward_multiple method
    print("\nTesting forward_multiple method...")
    batch_size = 2
    seq_len = 100
    num_segs = 3
    in_dim = 2048
    out_dim = 256
    
    projector = TextPromptProjector(in_dim=in_dim, out_dim=out_dim)
    
    # Create sequence hidden states
    text_hidden_seq = torch.randn(batch_size, seq_len, in_dim)
    
    # Create SEG positions (random positions in sequence)
    seg_positions = torch.randint(0, seq_len, (batch_size, num_segs))
    
    # Forward pass
    prompt_embeds = projector.forward_multiple(text_hidden_seq, seg_positions)
    
    print(f"  Sequence shape: {text_hidden_seq.shape}")
    print(f"  SEG positions shape: {seg_positions.shape}")
    print(f"  Output shape: {prompt_embeds.shape}")
    
    assert prompt_embeds.shape == (batch_size, num_segs, out_dim), \
        f"Multiple SEG shape mismatch: {prompt_embeds.shape}"
    
    print(f"  ✓ forward_multiple test passed!")
    
    # Test gradient flow
    print("\nTesting gradient flow...")
    projector = TextPromptProjector(in_dim=2048, out_dim=256, use_mlp=True)
    text_hidden = torch.randn(4, 2048, requires_grad=True)
    output = projector(text_hidden)
    loss = output.sum()
    loss.backward()
    
    assert text_hidden.grad is not None, "No gradient computed for input"
    assert all(p.grad is not None for p in projector.parameters()), "No gradient computed for projector parameters"
    print("  ✓ Gradient flow test passed!")


def test_parameter_counts():
    """Test and report parameter counts for adapters"""
    print("\n" + "=" * 60)
    print("Parameter Counts")
    print("=" * 60)
    
    # ImageFeatureAdapter
    adapter_configs = [
        (1280, 256),   # Qwen2.5-VL-3B actual vision dimension
        (1024, 256),   # Alternative vision model
    ]
    
    for in_dim, out_dim in adapter_configs:
        adapter = ImageFeatureAdapter(in_dim=in_dim, out_dim=out_dim)
        param_count = sum(p.numel() for p in adapter.parameters())
        print(f"\nImageFeatureAdapter({in_dim} -> {out_dim}):")
        print(f"  Total parameters: {param_count:,}")
        print(f"  Memory (FP32): {param_count * 4 / 1024 / 1024:.2f} MB")
    
    # TextPromptProjector
    projector_configs = [
        (2048, 256, False),  # Qwen2.5-VL-3B actual, linear
        (2048, 256, True),   # Qwen2.5-VL-3B actual, MLP
        (4096, 256, False),  # Larger model, linear
        (4096, 256, True),   # Larger model, MLP
    ]
    
    for in_dim, out_dim, use_mlp in projector_configs:
        projector = TextPromptProjector(in_dim=in_dim, out_dim=out_dim, use_mlp=use_mlp)
        param_count = sum(p.numel() for p in projector.parameters())
        mlp_str = "MLP" if use_mlp else "Linear"
        print(f"\nTextPromptProjector({in_dim} -> {out_dim}, {mlp_str}):")
        print(f"  Total parameters: {param_count:,}")
        print(f"  Memory (FP32): {param_count * 4 / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    # Run all tests
    test_image_feature_adapter()
    test_text_prompt_projector()
    test_parameter_counts()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)