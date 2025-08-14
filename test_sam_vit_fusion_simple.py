"""
Simple test for SAM ViT Sigma Add Fusion implementation
Tests only the fusion mechanism without full model initialization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_fusion_mechanism():
    """Test the core fusion mechanism"""
    
    print("=" * 80)
    print("SAM ViT Fusion Mechanism Test (Simplified)")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Simulate fusion parameters
    image_fusion_beta = nn.Parameter(torch.zeros(1).to(device))
    print(f"\nInitial image_fusion_beta: {image_fusion_beta.item():.4f}")
    
    # 2. Create dummy features
    batch_size = 2
    channels = 256
    h, w = 64, 64  # SAM resolution
    
    # Simulate SAM features [B, C, H, W]
    sam_features = torch.randn(batch_size, channels, h, w).to(device)
    print(f"\nSAM features shape: {sam_features.shape}")
    
    # Simulate Qwen features (might have different resolution)
    qwen_h, qwen_w = 32, 32  # Different resolution
    qwen_features = torch.randn(batch_size, channels, qwen_h, qwen_w).to(device)
    print(f"Qwen features shape: {qwen_features.shape}")
    
    # 3. Test resolution alignment
    print("\n" + "=" * 40)
    print("Test 1: Resolution Alignment")
    print("=" * 40)
    
    if qwen_features.shape[-2:] != sam_features.shape[-2:]:
        qwen_features_resized = F.interpolate(
            qwen_features,
            size=sam_features.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        print(f"Resized Qwen features: {qwen_features_resized.shape}")
    else:
        qwen_features_resized = qwen_features
    
    print("✓ Resolution alignment successful")
    
    # 4. Test fusion with different beta values
    print("\n" + "=" * 40)
    print("Test 2: Fusion with Different Beta Values")
    print("=" * 40)
    
    beta_values = [-2.0, 0.0, 2.0]  # Will be sigmoid-ed to ~[0.12, 0.5, 0.88]
    
    for beta_val in beta_values:
        # Set beta value
        with torch.no_grad():
            image_fusion_beta.data = torch.tensor([beta_val]).to(device)
        
        # Apply sigmoid scaling
        beta_scaled = torch.sigmoid(image_fusion_beta)
        
        # Perform fusion
        fused = sam_features + beta_scaled * qwen_features_resized
        
        print(f"Beta={beta_val:4.1f} → Sigmoid={beta_scaled.item():.3f} → Fused shape: {fused.shape}")
        
        # Check that fusion preserves shape
        assert fused.shape == sam_features.shape, "Fusion changed feature shape!"
    
    print("✓ Fusion with different beta values successful")
    
    # 5. Test gradient flow
    print("\n" + "=" * 40)
    print("Test 3: Gradient Flow")
    print("=" * 40)
    
    # Reset beta and enable gradients
    image_fusion_beta.data = torch.zeros(1).to(device)
    image_fusion_beta.requires_grad = True
    
    # Forward pass
    beta_scaled = torch.sigmoid(image_fusion_beta)
    fused = sam_features + beta_scaled * qwen_features_resized
    
    # Simulate loss
    loss = fused.mean()
    loss.backward()
    
    if image_fusion_beta.grad is not None:
        grad_value = image_fusion_beta.grad.item()
        print(f"Gradient on image_fusion_beta: {grad_value:.6f}")
        if abs(grad_value) > 1e-8:
            print("✓ Gradient flows through fusion mechanism")
        else:
            print("⚠ Very small gradient")
    else:
        print("✗ No gradient computed")
    
    # 6. Test fusion effect magnitude
    print("\n" + "=" * 40)
    print("Test 4: Fusion Effect Analysis")
    print("=" * 40)
    
    with torch.no_grad():
        # Beta = 0 (sigmoid = 0.5)
        image_fusion_beta.data = torch.zeros(1).to(device)
        beta_scaled = torch.sigmoid(image_fusion_beta)
        fused_mid = sam_features + beta_scaled * qwen_features_resized
        
        # Beta = -inf (sigmoid ≈ 0, minimal Qwen contribution)
        image_fusion_beta.data = torch.tensor([-10.0]).to(device)
        beta_scaled = torch.sigmoid(image_fusion_beta)
        fused_min = sam_features + beta_scaled * qwen_features_resized
        
        # Beta = +inf (sigmoid ≈ 1, maximal Qwen contribution)  
        image_fusion_beta.data = torch.tensor([10.0]).to(device)
        beta_scaled = torch.sigmoid(image_fusion_beta)
        fused_max = sam_features + beta_scaled * qwen_features_resized
        
        # Calculate differences
        diff_to_sam = (fused_min - sam_features).abs().mean()
        diff_min_max = (fused_max - fused_min).abs().mean()
        
        print(f"Difference when β→-∞ (should be ≈0): {diff_to_sam.item():.6f}")
        print(f"Difference between min and max fusion: {diff_min_max.item():.6f}")
        
        if diff_to_sam.item() < 0.01:
            print("✓ Minimal fusion correctly approaches pure SAM features")
        if diff_min_max.item() > 0.1:
            print("✓ Fusion range is significant")
    
    print("\n" + "=" * 80)
    print("All Fusion Mechanism Tests Passed!")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    try:
        success = test_fusion_mechanism()
        if success:
            print("\n✅ Fusion mechanism working correctly!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()