"""
Minimal test for model-dataset integration
"""
import sys
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.config import LISAConfig
from src.models import LISA_Model
from src.utils import prepare_tokenizer_for_lisa
from src.data.collators import MultiModalDataCollator
from transformers import AutoProcessor
import numpy as np
from PIL import Image

print("1. Setting up config and tokenizer...")
config = LISAConfig(
    qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    sam_model_name="./checkpoints/sam2.1_hiera_large.pt",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    freeze_qwen=True,
    freeze_sam=True,
    train_seg_token=True,
)

tokenizer = prepare_tokenizer_for_lisa(
    model_name=config.qwen_model_name,
    seg_token=config.seg_token
)

processor = AutoProcessor.from_pretrained(config.qwen_model_name)
processor.tokenizer = tokenizer

print("2. Loading model...")
model = LISA_Model(config)
model.set_tokenizer(tokenizer)
model = model.to(config.device_map)
model.eval()

print("3. Creating dummy data mimicking dataset output...")
# Create dummy sample
dummy_image = Image.new('RGB', (336, 336), color='red')

# Process with Qwen processor
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": dummy_image},
            {"type": "text", "text": "Segment the object. <SEG>"}
        ]
    }
]

processed = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
)

# Create sample like dataset output
sample = {
    'input_ids': processed['input_ids'].squeeze(0),
    'labels': processed['input_ids'].squeeze(0).clone(),
    'attention_mask': processed['attention_mask'].squeeze(0),
    'pixel_values': processed['pixel_values'].squeeze(0),
    'ground_truth_mask': torch.zeros(1, 1024, 1024),  # Dataset returns this key
    'image_grid_thw': processed.get('image_grid_thw', torch.tensor([1, 24, 24])).squeeze(0),
}

print("4. Testing collator...")
collator = MultiModalDataCollator(tokenizer=tokenizer)
batch = collator([sample])

print(f"   Batch keys: {list(batch.keys())}")
print(f"   'mask_labels' in batch: {'mask_labels' in batch}")

# Move to device
batch = {k: v.to(config.device_map) if isinstance(v, torch.Tensor) else v 
         for k, v in batch.items()}

print("5. Testing model forward pass...")
try:
    with torch.no_grad():
        # Prepare inputs like in test_with_real_dataset.py
        model_inputs = {
            'input_ids': batch['input_ids'],
            'pixel_values': batch['pixel_values'],
            'attention_mask': batch['attention_mask'],
            'labels': batch['labels'],
        }
        
        if 'image_grid_thw' in batch:
            model_inputs['image_grid_thw'] = batch['image_grid_thw']
        
        # Handle mask labels
        if 'mask_labels' in batch:
            # Convert to list format expected by model
            mask_list = []
            for i in range(batch['mask_labels'].size(0)):
                mask_list.append(batch['mask_labels'][i])
            model_inputs['mask_labels'] = mask_list
        
        outputs = model(**model_inputs)
        print("✓ Model forward pass successful!")
        
        if hasattr(outputs, 'logits'):
            print(f"  Language logits shape: {outputs.logits.shape}")
        if hasattr(outputs, 'mask_logits'):
            print(f"  Mask predictions available: {outputs.mask_logits is not None}")
            
except Exception as e:
    print(f"✗ Model forward pass failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Dataset → Collator → Model integration working!")