"""
Test SEG token generation in dummy dataset
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from transformers import AutoProcessor
from PIL import Image
import numpy as np
from src.utils import prepare_tokenizer_for_lisa

# Setup
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
tokenizer = prepare_tokenizer_for_lisa(model_name, seg_token="<SEG>")
processor = AutoProcessor.from_pretrained(model_name)

# Create dummy image
img_array = np.random.randint(0, 255, (336, 336, 3), dtype=np.uint8)
image = Image.fromarray(img_array)

# Test case 1: Basic template
print("=" * 50)
print("Test 1: Basic SEG token usage")
print("=" * 50)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Can you segment the car in this image?"},
        ],
    },
    {
        "role": "assistant",
        "content": "Sure, the car is <SEG>."
    }
]

# Apply chat template
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
print(f"Generated text:\n{text}")
print()

# Process
inputs = processor(
    text=text,
    images=[image],
    return_tensors="pt"
)

# Check for SEG token
seg_token_id = tokenizer.convert_tokens_to_ids("<SEG>")
print(f"SEG token ID: {seg_token_id}")

# Count SEG tokens in input_ids
seg_count = (inputs['input_ids'] == seg_token_id).sum().item()
print(f"Number of SEG tokens in input_ids: {seg_count}")

# Decode to verify
decoded = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
print(f"\nDecoded text (first 300 chars):\n{decoded[:300]}...")

# Find SEG token position if exists
if seg_count > 0:
    seg_positions = (inputs['input_ids'][0] == seg_token_id).nonzero(as_tuple=True)[0]
    print(f"\nSEG token positions: {seg_positions.tolist()}")
    
    # Show context around SEG token
    for pos in seg_positions:
        start = max(0, pos - 5)
        end = min(len(inputs['input_ids'][0]), pos + 5)
        context_ids = inputs['input_ids'][0][start:end]
        context_text = tokenizer.decode(context_ids, skip_special_tokens=False)
        print(f"Context around position {pos}: '{context_text}'")
else:
    print("\nNo SEG tokens found!")
    print("Checking if '<SEG>' appears as text in the decoded string...")
    if "<SEG>" in decoded:
        print("Found '<SEG>' as text in the decoded string!")
        seg_index = decoded.index("<SEG>")
        print(f"Text around '<SEG>': '{decoded[max(0, seg_index-20):seg_index+20]}'")

# Test case 2: Check vocabulary
print("\n" + "=" * 50)
print("Test 2: Vocabulary check")
print("=" * 50)

vocab = tokenizer.get_vocab()
seg_in_vocab = "<SEG>" in vocab
print(f"Is '<SEG>' in vocabulary? {seg_in_vocab}")

if seg_in_vocab:
    print(f"<SEG> token ID from vocab: {vocab['<SEG>']}")
    
# Show some high ID tokens
print("\nHigh ID tokens (>151660):")
high_tokens = {k: v for k, v in vocab.items() if v > 151660}
for token, id in sorted(high_tokens.items(), key=lambda x: x[1])[:10]:
    print(f"  {token}: {id}")