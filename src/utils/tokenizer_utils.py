"""
Tokenizer utilities for LISA改 (LISA-Kai)
Handles special token addition and configuration
"""
from typing import Dict, List, Optional
from transformers import AutoTokenizer


def add_seg_token(
    tokenizer: AutoTokenizer,
    seg_token: str = "<SEG>",
    additional_special_tokens: Optional[List[str]] = None
) -> Dict[str, int]:
    """
    Add SEG token and other special tokens to tokenizer
    
    Args:
        tokenizer: HuggingFace tokenizer
        seg_token: Segmentation special token (default: "<SEG>")
        additional_special_tokens: Other special tokens to add
    
    Returns:
        Dictionary mapping token names to their IDs
    """
    special_tokens_dict = {}
    tokens_to_add = []
    
    # Check if SEG token already exists
    if seg_token not in tokenizer.get_vocab():
        tokens_to_add.append(seg_token)
    
    # Add any additional special tokens
    if additional_special_tokens:
        for token in additional_special_tokens:
            if token not in tokenizer.get_vocab():
                tokens_to_add.append(token)
    
    # Add all new tokens at once
    if tokens_to_add:
        num_added = tokenizer.add_special_tokens({
            "additional_special_tokens": tokens_to_add
        })
        print(f"Added {num_added} special tokens to tokenizer")
    
    # Get token IDs
    special_tokens_dict[seg_token] = tokenizer.convert_tokens_to_ids(seg_token)
    if additional_special_tokens:
        for token in additional_special_tokens:
            special_tokens_dict[token] = tokenizer.convert_tokens_to_ids(token)
    
    return special_tokens_dict


def prepare_tokenizer_for_lisa(
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    seg_token: str = "<SEG>",
    padding_side: str = "left"
) -> AutoTokenizer:
    """
    Prepare tokenizer for LISA model with special tokens
    
    Args:
        model_name: Model name or path
        seg_token: Segmentation special token
        padding_side: Padding side for tokenizer
    
    Returns:
        Configured tokenizer
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set padding side (left for generation, right for training)
    tokenizer.padding_side = padding_side
    
    # Add special tokens
    special_tokens = add_seg_token(tokenizer, seg_token)
    
    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Tokenizer prepared with vocab size: {len(tokenizer)}")
    print(f"Special tokens: {special_tokens}")
    
    return tokenizer


def create_training_labels(
    input_ids: List[int],
    tokenizer: AutoTokenizer,
    response_start_idx: Optional[int] = None,
    ignore_index: int = -100
) -> List[int]:
    """
    Create training labels by masking non-response tokens
    
    Args:
        input_ids: List of token IDs
        tokenizer: Tokenizer instance
        response_start_idx: Index where model response starts
        ignore_index: Value to use for ignored positions
    
    Returns:
        List of labels with ignored positions set to ignore_index
    """
    labels = input_ids.copy()
    
    if response_start_idx is not None:
        # Mask everything before response
        for i in range(response_start_idx):
            labels[i] = ignore_index
    else:
        # Try to find response start automatically
        # Look for common patterns like "Assistant:" or special tokens
        # This is a simplified version - adjust based on your data format
        pass
    
    return labels