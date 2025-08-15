"""
Advanced tests for Cross-Attention fusion including visualization and real image testing
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import logging
from pathlib import Path
import os

from src.config import LISAConfig
from src.models.lisa_model import LISA_Model
from src.models.fusion_layers import CrossAttentionFusion
from transformers import AutoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def visualize_attention_weights(attn_weights, save_path="attention_vis.png"):
    """Visualize attention weights as a heatmap"""
    # attn_weights shape: [B, N, M] or [B, H, N, M]
    if attn_weights.dim() == 4:
        # Average over heads
        attn_weights = attn_weights.mean(dim=1)
    
    # Take first batch
    weights = attn_weights[0].cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Show as heatmap
    im = ax.imshow(weights, cmap='hot', interpolation='nearest')
    ax.set_title("Cross-Attention Weights (Query: SAM, Key/Value: Qwen)")
    ax.set_xlabel("Qwen Feature Positions")
    ax.set_ylabel("SAM Feature Positions")
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Save figure
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved attention visualization to {save_path}")
    return save_path


def test_attention_visualization():
    """Test Cross-Attention with attention weight visualization"""
    print("=" * 80)
    print("Attention Visualization Test")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create fusion module
    fusion = CrossAttentionFusion(dim=256, num_heads=8, use_gate=True)
    fusion = fusion.to(device)
    fusion.eval()
    
    # Create dummy features (smaller for visualization)
    batch_size = 1
    sam_features = torch.randn(batch_size, 16*16, 256).to(device)  # 16x16 spatial
    qwen_features = torch.randn(batch_size, 16*16, 256).to(device)
    
    # Get attention weights
    with torch.no_grad():
        fused, attn_weights = fusion(
            sam_features, 
            qwen_features, 
            return_attention_weights=True
        )
    
    print(f"Attention weights shape: {attn_weights.shape}")
    
    # Visualize
    vis_path = visualize_attention_weights(attn_weights)
    print(f"✓ Attention weights visualized and saved to {vis_path}")
    
    # Analyze attention patterns
    # Check if attention is focused or distributed
    attn_flat = attn_weights.flatten()
    entropy = -(attn_flat * torch.log(attn_flat + 1e-10)).sum()
    print(f"Attention entropy: {entropy.item():.4f} (higher = more distributed)")
    
    # Find most attended positions
    max_attn, max_idx = attn_weights[0].max(dim=1)
    print(f"Maximum attention values: mean={max_attn.mean():.4f}, std={max_attn.std():.4f}")
    
    return True


def test_real_image_fusion(image_path=None):
    """Test fusion with a real image"""
    print("=" * 80)
    print("Real Image Fusion Test")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load or create test image
    if image_path and os.path.exists(image_path):
        image = Image.open(image_path).convert("RGB")
        print(f"Loaded image from {image_path}")
    else:
        # Create a synthetic test image with clear objects
        print("Creating synthetic test image...")
        img_array = np.zeros((512, 512, 3), dtype=np.uint8)
        # Add a red square
        img_array[100:200, 100:200, 0] = 255
        # Add a green circle (approximate)
        center = (350, 350)
        radius = 50
        y, x = np.ogrid[:512, :512]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        img_array[mask, 1] = 255
        # Add blue rectangle
        img_array[300:400, 50:150, 2] = 255
        
        image = Image.fromarray(img_array)
        image.save("test_synthetic.png")
        print("Saved synthetic test image to test_synthetic.png")
    
    # Initialize models with different fusion types
    configs = {
        "cross_attention": LISAConfig(
            fusion_type="cross_attention",
            debug_fusion=True,
            torch_dtype=torch.float32,
            device_map=device
        ),
        "sigma_add": LISAConfig(
            fusion_type="sigma_add",
            debug_fusion=True,
            torch_dtype=torch.float32,
            device_map=device
        )
    }
    
    # Process with both fusion types
    for fusion_type, config in configs.items():
        print(f"\nTesting {fusion_type} fusion...")
        
        # Initialize model
        model = LISA_Model(config)
        model = model.to(device)
        model.eval()
        
        # Initialize processor
        processor = AutoProcessor.from_pretrained(
            config.qwen_model_name,
            trust_remote_code=True
        )
        
        # Add SEG token
        if config.seg_token not in processor.tokenizer.get_vocab():
            processor.tokenizer.add_tokens([config.seg_token], special_tokens=True)
            model.qwen.resize_token_embeddings(len(processor.tokenizer))
        
        model.set_tokenizer(processor.tokenizer, seg_token=config.seg_token)
        
        # Prepare inputs
        messages = [
            {"role": "user", "content": [
                {"type": "image"}, 
                {"type": "text", "text": "Segment the red object<SEG> in the image"}
            ]}
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt"
        ).to(device)
        
        # Create SAM image (1024x1024)
        sam_image = image.resize((1024, 1024))
        sam_tensor = torch.tensor(np.array(sam_image)).permute(2, 0, 1).float() / 255.0
        sam_tensor = sam_tensor.unsqueeze(0).to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pixel_values=inputs.pixel_values,
                image_grid_thw=inputs.get('image_grid_thw', torch.tensor([[1, 32, 32]]).to(device)),
                sam_images=sam_tensor,
                return_dict=True
            )
        
        print(f"✓ {fusion_type}: Generated outputs successfully")
        
        # Analyze mask outputs
        if outputs.mask_logits:
            mask = outputs.mask_logits[0]
            if isinstance(mask, list):
                mask = mask[0]
            
            if mask is not None:
                # Convert to probability
                mask_prob = torch.sigmoid(mask)
                
                # Statistics
                print(f"  Mask statistics:")
                print(f"    Shape: {mask.shape}")
                print(f"    Min: {mask_prob.min().item():.4f}")
                print(f"    Max: {mask_prob.max().item():.4f}")
                print(f"    Mean: {mask_prob.mean().item():.4f}")
                print(f"    Std: {mask_prob.std().item():.4f}")
                
                # Save mask visualization
                if mask_prob.dim() == 4:
                    mask_np = mask_prob[0, 0].cpu().numpy()
                elif mask_prob.dim() == 3:
                    mask_np = mask_prob[0].cpu().numpy()
                else:
                    mask_np = mask_prob.cpu().numpy()
                
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(image)
                plt.title("Original Image")
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.imshow(mask_np, cmap='hot')
                plt.title(f"Mask Output ({fusion_type})")
                plt.colorbar()
                plt.axis('off')
                
                save_path = f"mask_output_{fusion_type}.png"
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"  Saved mask visualization to {save_path}")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    return True


def compare_fusion_convergence():
    """Compare training convergence between fusion methods"""
    print("=" * 80)
    print("Fusion Convergence Comparison")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create small test models
    fusion_modules = {
        "cross_attention": CrossAttentionFusion(dim=256, num_heads=8, use_gate=True),
        "sigma_add": nn.Sequential(
            nn.Identity()  # Placeholder for sigma-add
        )
    }
    
    # Training simulation
    num_steps = 100
    batch_size = 4
    losses = {name: [] for name in fusion_modules}
    
    for name, module in fusion_modules.items():
        module = module.to(device)
        module.train()
        
        optimizer = torch.optim.Adam(module.parameters(), lr=1e-4)
        
        print(f"\nSimulating training for {name}...")
        
        for step in range(num_steps):
            # Create dummy data
            sam_feats = torch.randn(batch_size, 256, 256).to(device)
            qwen_feats = torch.randn(batch_size, 256, 256).to(device)
            target = torch.randn(batch_size, 256, 256).to(device)
            
            # Forward pass
            if name == "cross_attention":
                output = module(sam_feats, qwen_feats)
            else:
                # Simple addition for sigma-add simulation
                output = sam_feats + 0.5 * qwen_feats
            
            # Compute loss
            loss = nn.functional.mse_loss(output, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses[name].append(loss.item())
            
            if (step + 1) % 20 == 0:
                print(f"  Step {step+1}: Loss = {loss.item():.6f}")
    
    # Plot convergence curves
    plt.figure(figsize=(10, 6))
    for name, loss_values in losses.items():
        plt.plot(loss_values, label=name, linewidth=2)
    
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Fusion Method Convergence Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = "convergence_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved convergence comparison to {save_path}")
    
    # Compare final losses
    for name, loss_values in losses.items():
        final_loss = np.mean(loss_values[-10:])
        print(f"{name}: Final loss = {final_loss:.6f}")
    
    return True


if __name__ == "__main__":
    print("Running Advanced Fusion Tests")
    print("=" * 80)
    
    try:
        # Run tests
        success = True
        
        # 1. Attention visualization
        if success:
            success = test_attention_visualization()
        
        # 2. Real image test
        if success:
            success = test_real_image_fusion()
        
        # 3. Convergence comparison
        if success:
            success = compare_fusion_convergence()
        
        if success:
            print("\n" + "=" * 80)
            print("✅ All advanced tests passed!")
            print("=" * 80)
        else:
            print("\n❌ Some tests failed")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()