"""
Test Qwen2.5-VL processor to understand image token generation
"""
import torch
from transformers import AutoProcessor, AutoTokenizer
from PIL import Image
import numpy as np

# Load processor and tokenizer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# Create a dummy image
img_array = np.random.randint(0, 255, (336, 336, 3), dtype=np.uint8)
image = Image.fromarray(img_array)

# Test different processing approaches
print("=" * 50)
print("Testing Qwen2.5-VL Processor")
print("=" * 50)

# Approach 1: Simple text + image
text = "Human: Please segment the object in this image\nAssistant: The object is here."
inputs = processor(
    text=text,
    images=image,
    return_tensors="pt"
)

print("\n1. Simple processing:")
print(f"Input IDs shape: {inputs['input_ids'].shape}")
print(f"Pixel values shape: {inputs['pixel_values'].shape}")
print(f"Has image_grid_thw: {'image_grid_thw' in inputs}")

# Check for image tokens
input_ids = inputs['input_ids'][0]
# Common image token IDs for Qwen2.5-VL
image_token_ids = [151859, 151860, 151861, 151862, 151863]
image_tokens_found = []
for token_id in image_token_ids:
    count = (input_ids == token_id).sum().item()
    if count > 0:
        image_tokens_found.append((token_id, count))

print(f"Image tokens found: {image_tokens_found}")
print(f"First 30 tokens: {input_ids[:30].tolist()}")
print(f"Token decode: {tokenizer.decode(input_ids[:30])}")

# Approach 2: With max_length
print("\n2. With max_length:")
inputs2 = processor(
    text=text,
    images=image,
    max_length=2048,
    truncation=True,
    return_tensors="pt"
)
print(f"Input IDs shape: {inputs2['input_ids'].shape}")

# Approach 3: Using chat template
print("\n3. Using chat template:")
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image,
            },
            {"type": "text", "text": "Please segment the object in this image"},
        ],
    },
    {
        "role": "assistant",
        "content": "The object is here."
    }
]

text_chat = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False
)
print(f"Chat template text: {text_chat[:200]}...")

inputs3 = processor(
    text=text_chat,
    images=[image],
    return_tensors="pt"
)
print(f"Input IDs shape: {inputs3['input_ids'].shape}")

# Check special tokens
print("\n4. Special tokens info:")
print(f"Tokenizer vocab size: {len(tokenizer)}")
print(f"Special tokens map: {tokenizer.special_tokens_map}")
print(f"Additional special tokens: {tokenizer.additional_special_tokens}")

# Find image token pattern
print("\n5. Finding image token pattern in chat template:")
unique_tokens = torch.unique(inputs3['input_ids'])
high_id_tokens = unique_tokens[unique_tokens > 150000]
print(f"High ID tokens (>150000): {high_id_tokens.tolist()}")

# Check actual token IDs for special tokens
print("\n6. Special token IDs:")
for token in ['<|image_pad|>', '<|vision_start|>', '<|vision_end|>']:
    token_id = tokenizer.convert_tokens_to_ids(token)
    print(f"{token}: {token_id}")

# Count image pad tokens
image_pad_id = tokenizer.convert_tokens_to_ids('<|image_pad|>')
image_pad_count = (inputs3['input_ids'] == image_pad_id).sum().item()
print(f"\nNumber of <|image_pad|> tokens: {image_pad_count}")
print(f"image_grid_thw: {inputs3.get('image_grid_thw', 'Not found')}")