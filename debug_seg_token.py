#!/usr/bin/env python3
"""
SEG token embedding detection debug script
"""

import torch
from transformers import AutoTokenizer, Qwen2VLProcessor
from src.models.lisa_model import LISAModel
from src.config import LISAConfig
from src.utils.model_utils import display_parameter_statistics

def debug_seg_token():
    """Debug SEG token embedding detection"""
    
    print("=== SEG Token Embedding Debug ===")
    
    # 1. Setup tokenizer
    print("1. Setting up tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True)
    processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True)
    
    # Add SEG token
    seg_token = "<SEG>"
    if seg_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [seg_token]})
        processor.tokenizer = tokenizer
    
    seg_token_id = tokenizer.convert_tokens_to_ids(seg_token)
    print(f"SEG token ID: {seg_token_id}")
    print(f"Vocab size: {len(tokenizer)}")
    
    # 2. Create minimal config
    print("
2. Creating model config...")
    config = LISAConfig(
        qwen_model_path="Qwen/Qwen2.5-VL-3B-Instruct",
        sam_model_path="facebook/sam2.1-hiera-large",
        freeze_seg_token=False,  # Enable SEG token training
        use_token_fpn=False  # Simplify for debug
    )
    
    # 3. Load model
    print("
3. Loading model...")
    model = LISAModel(config, tokenizer=tokenizer)
    model.set_tokenizer(tokenizer, seg_token)
    
    # 4. Check embedding layer details
    print("
4. Checking embedding layer details...")
    embed_layer = model.qwen.get_input_embeddings()
    print(f"Embedding layer type: {type(embed_layer)}")
    print(f"Embedding weight shape: {embed_layer.weight.shape}")
    print(f"Embedding weight requires_grad: {embed_layer.weight.requires_grad}")
    
    # Check specific SEG token embedding
    seg_embedding = embed_layer.weight[seg_token_id]
    print(f"SEG token embedding shape: {seg_embedding.shape}")
    print(f"SEG token embedding requires_grad: {seg_embedding.requires_grad}")
    
    # 5. Manual parameter inspection
    print("
5. Manual parameter inspection...")
    for name, param in model.named_parameters():
        if 'embed' in name:
            print(f"Parameter: {name}")
            print(f"  Shape: {param.shape}")
            print(f"  Requires grad: {param.requires_grad}")
            print(f"  Numel: {param.numel()}")
            
            # Check if this is the embedding layer with SEG token
            if 'embed_tokens' in name or 'word_embeddings' in name:
                try:
                    seg_grad = param[seg_token_id].requires_grad
                    print(f"  SEG token [{seg_token_id}] requires_grad: {seg_grad}")
                    print(f"  SEG token embedding dim: {param.shape[-1]}")
                except:
                    print(f"  Could not check SEG token gradient")
            print()
    
    # 6. Test parameter statistics
    print("
6. Testing parameter statistics...")
    display_parameter_statistics(model)
    
    # 7. Count trainable parameters manually
    print("
7. Manual trainable parameter count...")
    trainable_count = 0
    seg_token_found = False
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_count += param.numel()
            print(f"Trainable: {name} ({param.numel():,} params)")
        
        # Special check for SEG token
        if ('embed_tokens' in name or 'word_embeddings' in name) and not param.requires_grad:
            try:
                if hasattr(model, 'seg_token_id') and model.seg_token_id is not None:
                    seg_grad = param[model.seg_token_id].requires_grad
                    if seg_grad:
                        seg_token_found = True
                        embed_dim = param.shape[-1]
                        print(f"SEG Token found: {name}[{model.seg_token_id}] ({embed_dim} params)")
            except Exception as e:
                print(f"Error checking SEG token in {name}: {e}")
    
    print(f"
Total trainable parameters (manual count): {trainable_count:,}")
    print(f"SEG token trainable found: {seg_token_found}")

if __name__ == "__main__":
    debug_seg_token()#!/usr/bin/env python3
"""
SEG token embedding detection debug script
"""

import torch
from transformers import AutoTokenizer, Qwen2VLProcessor
from src.models.lisa_model import LISAModel
from src.config import LISAConfig
from src.utils.model_utils import display_parameter_statistics

def debug_seg_token():
    """Debug SEG token embedding detection"""
    
    print("=== SEG Token Embedding Debug ===")
    
    # 1. Setup tokenizer
    print("1. Setting up tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True)
    processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True)
    
    # Add SEG token
    seg_token = "<SEG>"
    if seg_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [seg_token]})
        processor.tokenizer = tokenizer
    
    seg_token_id = tokenizer.convert_tokens_to_ids(seg_token)
    print(f"SEG token ID: {seg_token_id}")
    print(f"Vocab size: {len(tokenizer)}")
    
    # 2. Create minimal config
    print("
2. Creating model config...")
    config = LISAConfig(
        qwen_model_path="Qwen/Qwen2.5-VL-3B-Instruct",
        sam_model_path="facebook/sam2.1-hiera-large",
        freeze_seg_token=False,  # Enable SEG token training
        use_token_fpn=False  # Simplify for debug
    )
    
    # 3. Load model
    print("
3. Loading model...")
    model = LISAModel(config, tokenizer=tokenizer)
    model.set_tokenizer(tokenizer, seg_token)
    
    # 4. Check embedding layer details
    print("
4. Checking embedding layer details...")
    embed_layer = model.qwen.get_input_embeddings()
    print(f"Embedding layer type: {type(embed_layer)}")
    print(f"Embedding weight shape: {embed_layer.weight.shape}")
    print(f"Embedding weight requires_grad: {embed_layer.weight.requires_grad}")
    
    # Check specific SEG token embedding
    seg_embedding = embed_layer.weight[seg_token_id]
    print(f"SEG token embedding shape: {seg_embedding.shape}")
    print(f"SEG token embedding requires_grad: {seg_embedding.requires_grad}")
    
    # 5. Manual parameter inspection
    print("
5. Manual parameter inspection...")
    for name, param in model.named_parameters():
        if 'embed' in name:
            print(f"Parameter: {name}")
            print(f"  Shape: {param.shape}")
            print(f"  Requires grad: {param.requires_grad}")
            print(f"  Numel: {param.numel()}")
            
            # Check if this is the embedding layer with SEG token
            if 'embed_tokens' in name or 'word_embeddings' in name:
                try:
                    seg_grad = param[seg_token_id].requires_grad
                    print(f"  SEG token [{seg_token_id}] requires_grad: {seg_grad}")
                    print(f"  SEG token embedding dim: {param.shape[-1]}")
                except:
                    print(f"  Could not check SEG token gradient")
            print()
    
    # 6. Test parameter statistics
    print("
6. Testing parameter statistics...")
    display_parameter_statistics(model)
    
    # 7. Count trainable parameters manually
    print("
7. Manual trainable parameter count...")
    trainable_count = 0
    seg_token_found = False
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_count += param.numel()
            print(f"Trainable: {name} ({param.numel():,} params)")
        
        # Special check for SEG token
        if ('embed_tokens' in name or 'word_embeddings' in name) and not param.requires_grad:
            try:
                if hasattr(model, 'seg_token_id') and model.seg_token_id is not None:
                    seg_grad = param[model.seg_token_id].requires_grad
                    if seg_grad:
                        seg_token_found = True
                        embed_dim = param.shape[-1]
                        print(f"SEG Token found: {name}[{model.seg_token_id}] ({embed_dim} params)")
            except Exception as e:
                print(f"Error checking SEG token in {name}: {e}")
    
    print(f"
Total trainable parameters (manual count): {trainable_count:,}")
    print(f"SEG token trainable found: {seg_token_found}")

if __name__ == "__main__":
    debug_seg_token()