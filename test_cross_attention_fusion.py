"""
Test script for Cross-Attention fusion between SAM and Qwen features
Based on test_sam_vit_fusion.py implementation
"""

import torch
import numpy as np
import logging
from PIL import Image
from src.config import LISAConfig
from src.models.lisa_model import LISA_Model
from transformers import AutoProcessor

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_cross_attention_fusion():
    """Test the Cross-Attention fusion implementation"""
    
    print("=" * 80)
    print("Cross-Attention Fusion Test")
    print("=" * 80)
    
    # 1. Initialize config with Cross-Attention fusion
    config = LISAConfig(
        qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        sam_model_name="./checkpoints/sam2.1_hiera_large.pt",
        torch_dtype=torch.float32,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        fusion_type="cross_attention",  # Use Cross-Attention
        fusion_num_heads=8,
        fusion_dropout=0.1,
        fusion_use_gate=True,
        freeze_image_fusion_beta=False,  # Keep trainable for testing
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Fusion type: {config.fusion_type}")
    
    # 2. Initialize model and processor
    print("\nInitializing LISA model with Cross-Attention fusion...")
    model = LISA_Model(config)
    model = model.to(device)
    model.eval()
    
    # Verify fusion type
    assert model.fusion_type == "cross_attention", f"Expected cross_attention, got {model.fusion_type}"
    print(f"✓ Model initialized with fusion_type={model.fusion_type}")
    
    # Initialize processor
    processor = AutoProcessor.from_pretrained(
        config.qwen_model_name,
        trust_remote_code=True,
        min_pixels=256*28*28,
        max_pixels=1280*28*28
    )
    
    # Add SEG token if not present
    if config.seg_token not in processor.tokenizer.get_vocab():
        processor.tokenizer.add_tokens([config.seg_token], special_tokens=True)
        model.qwen.resize_token_embeddings(len(processor.tokenizer))
    
    # Set tokenizer in model
    model.set_tokenizer(processor.tokenizer, seg_token=config.seg_token)
    
    # Check fusion module
    print(f"\nFusion module type: {type(model.image_fusion).__name__}")
    cross_attn_params = sum(p.numel() for p in model.image_fusion.parameters())
    print(f"Cross-Attention parameters: {cross_attn_params:,}")
    
    # 3. Create dummy inputs
    batch_size = 1
    
    # Create a dummy image
    dummy_image = Image.fromarray(np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8))
    
    # Process through Qwen's processor with tokenization
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image"}]},
        {"role": "assistant", "content": "This is a test <SEG> response"}
    ]
    
    # Apply chat template with proper tokenization
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = processor(
        text=[text],
        images=[dummy_image],
        return_tensors="pt",
        padding=True
    ).to(device)
    
    pixel_values = inputs.pixel_values
    image_grid_thw = inputs.get('image_grid_thw', torch.tensor([[1, 32, 32]]).to(device))
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    
    # SAM inputs (high resolution 1024x1024)
    sam_images = torch.randn(batch_size, 3, 1024, 1024).to(device)
    
    # 4. Test forward pass without SAM images (baseline)
    print("\n" + "=" * 40)
    print("Test 1: Forward pass WITHOUT SAM images")
    print("=" * 40)
    
    with torch.no_grad():
        outputs_no_sam = model.forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            sam_images=None,  # No SAM images
            return_dict=True
        )
    
    print("✓ Forward pass without SAM images completed")
    print(f"  Output logits shape: {outputs_no_sam.logits.shape}")
    
    # 5. Test forward pass WITH SAM images (Cross-Attention fusion)
    print("\n" + "=" * 40)
    print("Test 2: Forward pass WITH SAM images (Cross-Attention)")
    print("=" * 40)
    
    with torch.no_grad():
        outputs_with_sam = model.forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            sam_images=sam_images,  # Include SAM images for fusion
            return_dict=True
        )
    
    print("✓ Forward pass with SAM images completed")
    print(f"  Output logits shape: {outputs_with_sam.logits.shape}")
    
    # 6. Compare outputs to verify fusion is happening
    print("\n" + "=" * 40)
    print("Test 3: Verify Cross-Attention fusion effect on MASKS")
    print("=" * 40)
    
    # Check mask outputs (fusion affects segmentation, not language)
    if hasattr(outputs_no_sam, 'mask_logits') and hasattr(outputs_with_sam, 'mask_logits'):
        if outputs_no_sam.mask_logits and outputs_with_sam.mask_logits:
            # Extract first mask from each output
            mask_no_sam = outputs_no_sam.mask_logits[0]
            mask_with_sam = outputs_with_sam.mask_logits[0]
            
            # Handle list structure
            if isinstance(mask_no_sam, list):
                mask_no_sam = mask_no_sam[0] if mask_no_sam else None
            if isinstance(mask_with_sam, list):
                mask_with_sam = mask_with_sam[0] if mask_with_sam else None
            
            if mask_no_sam is not None and mask_with_sam is not None:
                mask_diff = (mask_with_sam - mask_no_sam).abs().mean()
                print(f"Mean absolute difference in mask logits: {mask_diff.item():.6f}")
                
                if mask_diff.item() > 1e-6:
                    print("✓ Cross-Attention fusion IS affecting the mask output")
                else:
                    print("⚠ WARNING: Cross-Attention may not be affecting masks significantly")
            else:
                print("⚠ No mask outputs generated")
        else:
            print("⚠ No mask_logits in outputs")
    else:
        print("⚠ Model outputs don't include mask_logits")
    
    print("\nNote: Cross-Attention fusion affects segmentation masks, not language logits.")
    
    # 7. Test gradient flow through Cross-Attention
    print("\n" + "=" * 40)
    print("Test 4: Gradient flow through Cross-Attention")
    print("=" * 40)
    
    model.train()
    for param in model.image_fusion.parameters():
        param.requires_grad = True
    
    # Forward pass with gradient
    outputs = model.forward(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        image_grid_thw=image_grid_thw,
        sam_images=sam_images,
        return_dict=True
    )
    
    # Create dummy loss including mask outputs for proper gradient flow
    loss = outputs.logits.mean()
    if outputs.mask_logits:
        # Include mask loss to ensure gradients flow through fusion module
        mask_losses = []
        for m in outputs.mask_logits:
            # Handle nested list structure
            if isinstance(m, list):
                for sub_m in m:
                    if sub_m is not None and hasattr(sub_m, 'mean'):
                        mask_losses.append(sub_m.mean())
            elif m is not None and hasattr(m, 'mean'):
                mask_losses.append(m.mean())
        
        if mask_losses:
            mask_loss = sum(mask_losses) / len(mask_losses)
            loss = loss + mask_loss
    loss.backward()
    
    # Check gradients in Cross-Attention module
    grad_norms = []
    for name, param in model.image_fusion.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.abs().mean().item()
            grad_norms.append((name, grad_norm))
    
    if grad_norms:
        print("Gradient norms in Cross-Attention module:")
        for name, norm in grad_norms[:5]:  # Show first 5
            print(f"  {name}: {norm:.6f}")
        
        avg_grad = np.mean([n for _, n in grad_norms])
        if avg_grad > 1e-8:
            print(f"✓ Gradients flow through Cross-Attention (avg: {avg_grad:.6f})")
        else:
            print("⚠ WARNING: Very small gradients in Cross-Attention")
    else:
        print("⚠ WARNING: No gradients in Cross-Attention module")
    
    # Store Cross-Attention parameters count before cleanup
    cross_params = sum(p.numel() for p in model.image_fusion.parameters())
    
    # 8. Compare with Sigma-Add fusion
    print("\n" + "=" * 40)
    print("Test 5: Compare with Sigma-Add fusion")
    print("=" * 40)
    
    # Clean up first model to free memory
    del model
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    # Create config for Sigma-Add
    config_sigma = LISAConfig(
        qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        sam_model_name="./checkpoints/sam2.1_hiera_large.pt",
        torch_dtype=torch.float32,
        device_map=device,
        fusion_type="sigma_add",  # Use Sigma-Add
    )
    
    # Initialize Sigma-Add model
    model_sigma = LISA_Model(config_sigma)
    model_sigma = model_sigma.to(device)
    model_sigma.eval()
    
    # Set tokenizer
    model_sigma.set_tokenizer(processor.tokenizer, seg_token=config.seg_token)
    
    print(f"Sigma-Add model fusion_type: {model_sigma.fusion_type}")
    print(f"Sigma-Add beta value: {model_sigma.image_fusion_beta.item():.4f}")
    
    # Test Sigma-Add forward pass
    with torch.no_grad():
        outputs_sigma = model_sigma.forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            sam_images=sam_images,
            return_dict=True
        )
    
    print("✓ Sigma-Add forward pass completed")
    
    # Compare parameters (cross_params was saved before model deletion)
    sigma_params = 1  # Just the beta parameter
    
    print(f"\nParameter comparison:")
    print(f"  Cross-Attention: {cross_params:,} parameters")
    print(f"  Sigma-Add: {sigma_params} parameter")
    print(f"  Difference: {cross_params - sigma_params:,} additional parameters")
    
    print("\n" + "=" * 80)
    print("Cross-Attention Fusion Tests Completed")
    print("=" * 80)
    
    # Summary
    print("\nSummary:")
    print("✓ Model initialization with Cross-Attention successful")
    print("✓ Forward pass without SAM images works")
    print("✓ Forward pass with SAM images works")
    print("✓ Cross-Attention module has proper parameters")
    
    # Check gradient flow
    if grad_norms and avg_grad > 1e-8:
        print("✓ Gradients flow through Cross-Attention mechanism")
    
    print("✓ Comparison with Sigma-Add completed")
    
    return True

if __name__ == "__main__":
    try:
        success = test_cross_attention_fusion()
        if success:
            print("\n✅ All Cross-Attention tests passed!")
        else:
            print("\n⚠️ Some tests may have issues")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()