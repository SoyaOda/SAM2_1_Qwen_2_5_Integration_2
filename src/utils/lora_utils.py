"""
LoRA utilities for LISA改 (LISA-Kai)
Handles LoRA configuration and application
"""
import torch
from typing import List, Optional, Union
from peft import LoraConfig, get_peft_model, TaskType
from transformers import PreTrainedModel


def get_qwen_lora_target_modules(model_name: str) -> List[str]:
    """
    Get target modules for LoRA based on Qwen model architecture
    
    Args:
        model_name: Name of the Qwen model
    
    Returns:
        List of module names to apply LoRA to
    """
    # Common attention module patterns in Qwen2.5-VL
    # These target the cross-attention layers between vision and language
    target_modules = [
        "self_attn.q_proj",
        "self_attn.k_proj", 
        "self_attn.v_proj",
        "self_attn.o_proj",
        # MLPs can also be targeted for more capacity
        # "mlp.gate_proj",
        # "mlp.up_proj",
        # "mlp.down_proj",
    ]
    
    # For vision-language cross attention specifically
    # Note: Exact module names may vary - these are common patterns
    cross_attn_modules = [
        "encoder_attn.q_proj",
        "encoder_attn.k_proj",
        "encoder_attn.v_proj",
        "encoder_attn.o_proj",
        "cross_attn.q_proj",
        "cross_attn.k_proj",
        "cross_attn.v_proj",
        "cross_attn.o_proj",
    ]
    
    return target_modules + cross_attn_modules


def apply_lora_to_model(
    model: PreTrainedModel,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
    task_type: str = "CAUSAL_LM"
) -> PreTrainedModel:
    """
    Apply LoRA to a model
    
    Args:
        model: Model to apply LoRA to
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling parameter
        lora_dropout: Dropout probability for LoRA layers
        target_modules: List of module names to apply LoRA to
        task_type: Task type for LoRA
    
    Returns:
        Model with LoRA applied
    """
    if target_modules is None:
        # Use default target modules for Qwen
        target_modules = get_qwen_lora_target_modules(model.config._name_or_path)
    
    # Create LoRA configuration
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM if task_type == "CAUSAL_LM" else TaskType.SEQ_2_SEQ_LM,
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print LoRA info
    model.print_trainable_parameters()
    
    return model


def freeze_base_model(
    model: Union[PreTrainedModel, torch.nn.Module],
    except_modules: Optional[List[str]] = None
):
    """
    Freeze all parameters in base model except specified modules
    
    Args:
        model: Model to freeze
        except_modules: List of module names to keep trainable
    """
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze specified modules
    if except_modules:
        for name, param in model.named_parameters():
            for module_name in except_modules:
                if module_name in name:
                    param.requires_grad = True
                    break


def configure_lisa_for_training(
    lisa_model,
    config,
    tokenizer
) -> dict:
    """
    Configure LISA model for training with LoRA and frozen parameters
    
    Args:
        lisa_model: LISA model instance
        config: LISA configuration
        tokenizer: Tokenizer with SEG token
    
    Returns:
        Dictionary with training configuration info
    """
    training_info = {
        "total_params": 0,
        "trainable_params": 0,
        "frozen_params": 0,
        "lora_params": 0,
    }
    
    # 1. Freeze base models as specified
    if config.freeze_qwen:
        freeze_base_model(lisa_model.qwen)
    
    if config.freeze_sam:
        freeze_base_model(lisa_model.sam_mask_decoder)
        freeze_base_model(lisa_model.sam_prompt_encoder)
    
    # 2. Apply LoRA to Qwen if not frozen
    if not config.freeze_qwen and config.lora_r > 0:
        lisa_model.qwen = apply_lora_to_model(
            lisa_model.qwen,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules
        )
        
        # Count LoRA parameters
        for name, param in lisa_model.qwen.named_parameters():
            if "lora" in name and param.requires_grad:
                training_info["lora_params"] += param.numel()
    
    # 3. Enable SEG token embedding training
    if config.train_seg_token and hasattr(lisa_model, 'seg_token_id'):
        word_embeddings = lisa_model.qwen.get_input_embeddings()
        # Only make SEG token embedding trainable
        word_embeddings.weight.requires_grad = False
        word_embeddings.weight[lisa_model.seg_token_id].requires_grad = True
        training_info["trainable_params"] += word_embeddings.weight.shape[1]  # Embedding dimension
    
    # 4. Ensure adapters are trainable
    if config.train_adapters:
        for param in lisa_model.image_adapter.parameters():
            param.requires_grad = True
        for param in lisa_model.text_prompt_proj.parameters():
            param.requires_grad = True
    
    # Count parameters
    for name, param in lisa_model.named_parameters():
        training_info["total_params"] += param.numel()
        if param.requires_grad:
            training_info["trainable_params"] += param.numel()
        else:
            training_info["frozen_params"] += param.numel()
    
    # Print summary
    print("\n" + "=" * 60)
    print("LISA Model Training Configuration")
    print("=" * 60)
    print(f"Total parameters: {training_info['total_params']:,}")
    print(f"Trainable parameters: {training_info['trainable_params']:,} "
          f"({training_info['trainable_params']/training_info['total_params']*100:.2f}%)")
    print(f"Frozen parameters: {training_info['frozen_params']:,}")
    if training_info['lora_params'] > 0:
        print(f"LoRA parameters: {training_info['lora_params']:,}")
    print("=" * 60 + "\n")
    
    return training_info