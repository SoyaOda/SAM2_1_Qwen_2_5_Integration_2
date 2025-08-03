"""
Test script to understand Qwen2.5-VL vision feature extraction
"""
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import numpy as np

# Load model and processor
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

# Process with chat template
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "What is this?"}
        ]
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(
    text=text,
    images=[image],
    return_tensors="pt"
).to("cuda")

print("=" * 50)
print("Testing Qwen2.5-VL Vision Features")
print("=" * 50)

# Test 1: Check model attributes
print("\n1. Model attributes:")
for attr in ['visual', 'vision_tower', 'vision_model', 'model']:
    print(f"  hasattr(model, '{attr}'): {hasattr(model, attr)}")

# Test 2: Check model.model attributes
if hasattr(model, 'model'):
    print("\n2. model.model attributes:")
    for attr in ['visual', 'vision_tower', 'vision_model']:
        print(f"  hasattr(model.model, '{attr}'): {hasattr(model.model, attr)}")

# Test 3: Try to get vision features
print("\n3. Attempting to extract vision features:")

# Approach 1: Direct visual attribute
if hasattr(model.model, 'visual'):
    try:
        with torch.no_grad():
            vision_out = model.model.visual(
                inputs['pixel_values'],
                grid_thw=inputs['image_grid_thw']
            )
        print(f"  Direct visual() output shape: {vision_out.shape}")
        print(f"  Direct visual() output dtype: {vision_out.dtype}")
    except Exception as e:
        print(f"  Direct visual() failed: {e}")

# Approach 2: Through forward pass with output_hidden_states
try:
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
    
    print(f"\n4. Model outputs analysis:")
    print(f"  outputs.keys(): {outputs.keys()}")
    
    if hasattr(outputs, 'hidden_states'):
        print(f"  len(outputs.hidden_states): {len(outputs.hidden_states)}")
        for i, hs in enumerate(outputs.hidden_states[:3]):  # First 3 layers
            print(f"  hidden_states[{i}].shape: {hs.shape}")
    
    # Check for vision-specific outputs
    for attr in ['vision_outputs', 'vision_hidden_states', 'encoder_hidden_states']:
        if hasattr(outputs, attr):
            val = getattr(outputs, attr)
            if val is not None:
                if isinstance(val, (list, tuple)):
                    print(f"  {attr}: list/tuple of length {len(val)}")
                    if len(val) > 0:
                        print(f"    First element shape: {val[0].shape}")
                else:
                    print(f"  {attr}.shape: {val.shape}")
except Exception as e:
    print(f"  Forward pass failed: {e}")

# Test 4: Inspect internal structure
print("\n5. Model internal structure:")
if hasattr(model, 'model'):
    if hasattr(model.model, 'visual'):
        visual_module = model.model.visual
        print(f"  Visual module type: {type(visual_module)}")
        print(f"  Visual module output channels: {getattr(visual_module, 'output_channels', 'Not found')}")
        
        # Try to understand visual module structure
        if hasattr(visual_module, 'blocks'):
            print(f"  Number of visual blocks: {len(visual_module.blocks)}")
        
        # Check for projection layers
        for attr in ['proj', 'merger', 'output_proj']:
            if hasattr(visual_module, attr):
                module = getattr(visual_module, attr)
                print(f"  Found {attr}: {type(module)}")
                if hasattr(module, 'out_features'):
                    print(f"    out_features: {module.out_features}")

print("\n" + "=" * 50)