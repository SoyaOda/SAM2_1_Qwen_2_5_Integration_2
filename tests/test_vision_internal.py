"""
Test to find pure vision features (1280D) from Qwen2.5-VL
"""
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import numpy as np

# Load model
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cuda"
)
processor = AutoProcessor.from_pretrained(model_name)

# Create dummy image
img_array = np.random.randint(0, 255, (336, 336, 3), dtype=np.uint8)
image = Image.fromarray(img_array)

# Process
messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": "test"}
]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, images=[image], return_tensors="pt").to("cuda")

print("Exploring Qwen2.5-VL Visual Module Structure")
print("=" * 50)

# Get visual module
visual = model.model.visual

# Check visual blocks
print(f"\n1. Visual blocks: {len(visual.blocks)} blocks")

# Test getting features before projection
with torch.no_grad():
    # Get pixel values
    pixel_values = inputs['pixel_values']
    grid_thw = inputs['image_grid_thw']
    
    print(f"\n2. Input shapes:")
    print(f"   pixel_values: {pixel_values.shape}")
    print(f"   grid_thw: {grid_thw}")
    
    # Try to trace through visual module
    x = pixel_values
    
    # Patch embedding
    if hasattr(visual, 'patch_embed'):
        x = visual.patch_embed(x, grid_thw)
        print(f"\n3. After patch_embed: {x.shape}")
    
    # Go through blocks
    for i, block in enumerate(visual.blocks[:3]):  # First 3 blocks
        x = block(x)
        if i == 0:
            print(f"\n4. After first vision block: {x.shape}")
            print(f"   This should be vision features!")
    
    # Check if there's a projection at the end
    if hasattr(visual, 'merger'):
        print(f"\n5. Visual has 'merger' module")
        print(f"   Type: {type(visual.merger)}")
        
        # Try to get features before merger
        # Save features before merger
        features_before_merger = x.clone()
        print(f"   Features before merger: {features_before_merger.shape}")
        
        # Apply merger
        x = visual.merger(x)
        print(f"   Features after merger: {x.shape}")

# Alternative: Hook into forward to capture intermediate features
print("\n6. Using forward hooks to capture features:")

features_dict = {}

def hook_fn(name):
    def hook(module, input, output):
        features_dict[name] = output
    return hook

# Register hooks
hooks = []
if hasattr(visual, 'blocks'):
    # Hook after last block
    hook = visual.blocks[-1].register_forward_hook(hook_fn('last_block'))
    hooks.append(hook)

if hasattr(visual, 'merger'):
    # Hook before merger
    hook = visual.merger.register_forward_hook(hook_fn('merger'))
    hooks.append(hook)

# Forward pass
with torch.no_grad():
    vision_out = visual(inputs['pixel_values'], grid_thw=inputs['image_grid_thw'])

# Check captured features
for name, feat in features_dict.items():
    if isinstance(feat, tuple):
        print(f"\n   {name} output (tuple):")
        for i, f in enumerate(feat):
            if torch.is_tensor(f):
                print(f"     [{i}] shape: {f.shape}, dtype: {f.dtype}")
    else:
        print(f"\n   {name} output shape: {feat.shape}, dtype: {feat.dtype}")

# Clean up hooks
for hook in hooks:
    hook.remove()

print("\n" + "=" * 50)