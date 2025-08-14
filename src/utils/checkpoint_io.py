# -*- coding: utf-8 -*-
"""
Checkpoint I/O utilities for LISA model
Handles saving and loading of all model components with proper organization
"""
import os
import json
import hashlib
import datetime
import subprocess
import sys
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, Optional, Union

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerBase, ProcessorMixin

logger = logging.getLogger(__name__)

try:
    from safetensors.torch import save_file as safetensors_save_file, load_file as safetensors_load_file
    _HAS_SAFE = True
except Exception:
    _HAS_SAFE = False
    logger.warning("safetensors not available, using torch.save instead")


def _short_hash(d: Dict[str, Any]) -> str:
    """Generate short hash from dictionary for unique naming"""
    s = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]


def build_checkpoint_dir(cfg, output_root: str, name_suffix: str = "best") -> str:
    """Build checkpoint directory name based on config parameters"""
    # Key parameters for directory naming
    key_fields = {
        "qwen_model": cfg.qwen_model_name.split("/")[-1] if hasattr(cfg, "qwen_model_name") else "qwen",
        "sam_model": cfg.sam_model_name.split("/")[-1] if hasattr(cfg, "sam_model_name") else "sam",
        "lora_r": getattr(cfg, "lora_r", 0),
        "sam_lora_r": getattr(cfg, "sam_lora_r", 0),
        "freeze_qwen_lora": getattr(cfg, "freeze_qwen_lora", False),
        "freeze_sam_mask_decoder": getattr(cfg, "freeze_sam_mask_decoder_base", True),
        "use_token_fpn": getattr(cfg, "use_token_fpn", True),
    }
    
    # Generate hash for uniqueness
    hh = _short_hash(key_fields)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build directory name
    qwen_part = f"qwen_{key_fields['lora_r']}" if key_fields['lora_r'] > 0 else "qwen_base"
    sam_part = f"sam_lora{key_fields['sam_lora_r']}" if key_fields['sam_lora_r'] > 0 else "sam_base"
    
    dir_name = f"checkpoint_{qwen_part}_{sam_part}_{ts}_{hh}/{name_suffix}"
    full_path = os.path.join(output_root, dir_name)
    os.makedirs(full_path, exist_ok=True)
    
    return full_path


def _write_metadata(checkpoint_dir: str, cfg, additional_info: Dict[str, Any] = None) -> None:
    """Write metadata files for reproducibility"""
    meta_dir = os.path.join(checkpoint_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    
    # Save config
    config_path = os.path.join(meta_dir, "config.json")
    config_dict = asdict(cfg) if hasattr(cfg, "__dict__") else cfg.__dict__
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)
    
    # Save git SHA
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        diff = subprocess.check_output(["git", "diff", "--stat"]).decode().strip()
    except Exception:
        sha = "NA"
        diff = "NA"
    
    with open(os.path.join(meta_dir, "git_info.txt"), "w") as f:
        f.write(f"SHA: {sha}\n")
        f.write(f"Diff:\n{diff}\n")
    
    # Save environment info
    import torch
    import transformers
    try:
        import peft
        peft_version = peft.__version__
    except:
        peft_version = "NA"
    
    env_info = {
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft_version,
        "cuda": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "NA",
    }
    
    with open(os.path.join(meta_dir, "environment.json"), "w") as f:
        json.dump(env_info, f, indent=2)
    
    # Save additional info if provided
    if additional_info:
        with open(os.path.join(meta_dir, "training_info.json"), "w") as f:
            json.dump(additional_info, f, indent=2, default=str)


def save_processor(checkpoint_dir: str, processor: ProcessorMixin) -> None:
    """Save processor (tokenizer + image preprocessor)"""
    processor_dir = os.path.join(checkpoint_dir, "processor")
    os.makedirs(processor_dir, exist_ok=True)
    processor.save_pretrained(processor_dir)
    logger.info(f"Saved processor to {processor_dir}")


def save_qwen_component(checkpoint_dir: str, qwen_model, config) -> None:
    """Save Qwen model or LoRA adapter"""
    qwen_dir = os.path.join(checkpoint_dir, "qwen")
    os.makedirs(qwen_dir, exist_ok=True)
    
    # Check if it's a PEFT model
    try:
        from peft import PeftModel
        is_peft = isinstance(qwen_model, PeftModel)
    except ImportError:
        is_peft = False
    
    if is_peft:
        # Save only LoRA adapter
        qwen_model.save_pretrained(qwen_dir)
        logger.info("Saved Qwen LoRA adapter")
        
        # Save base model reference
        base_info = {
            "base_model": config.qwen_model_name,
            "is_lora": True,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
        }
        with open(os.path.join(qwen_dir, "base_model_info.json"), "w") as f:
            json.dump(base_info, f, indent=2)
    else:
        # Check if we should save full model or just reference
        if config.freeze_qwen_base:
            # Just save reference since base is frozen
            base_info = {
                "base_model": config.qwen_model_name,
                "is_frozen": True,
            }
            with open(os.path.join(qwen_dir, "base_model_info.json"), "w") as f:
                json.dump(base_info, f, indent=2)
            logger.info("Saved Qwen base model reference (frozen)")
        else:
            # Save full model (unlikely but possible)
            qwen_model.save_pretrained(qwen_dir, safe_serialization=True)
            logger.info("Saved full Qwen model")


def save_seg_token_embedding(checkpoint_dir: str, model, tokenizer: PreTrainedTokenizerBase) -> None:
    """Save SEG token embedding if trainable"""
    if hasattr(model, 'seg_token_embedding') and model.seg_token_embedding is not None:
        seg_data = {
            'seg_token_id': model.seg_token_id,
            'seg_token': model.config.seg_token,
            'embedding': model.seg_token_embedding.detach().cpu(),
        }
        
        save_path = os.path.join(checkpoint_dir, "seg_token_embedding.pt")
        torch.save(seg_data, save_path)
        logger.info(f"Saved SEG token embedding for ID {model.seg_token_id}")


def save_prompt_beta(checkpoint_dir: str, prompt_beta: torch.nn.Parameter) -> None:
    """Save prompt beta parameter"""
    save_path = os.path.join(checkpoint_dir, "prompt_beta.pt")
    torch.save({'prompt_beta': prompt_beta.detach().cpu()}, save_path)
    logger.info(f"Saved prompt_beta: {prompt_beta.item():.6f}")


def _gather_sam_lora_weights(sam_mask_decoder) -> Dict[str, torch.Tensor]:
    """Gather SAM LoRA weights from custom implementation"""
    lora_weights = {}
    
    for name, module in sam_mask_decoder.named_modules():
        # Check for our custom LoRALinear modules
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            lora_weights[f"{name}.lora_A"] = module.lora_A.detach().cpu()
            lora_weights[f"{name}.lora_B"] = module.lora_B.detach().cpu()
            if hasattr(module, 'scaling'):
                lora_weights[f"{name}.scaling"] = torch.tensor(module.scaling)
    
    return lora_weights


def save_sam_lora(checkpoint_dir: str, sam_mask_decoder, config) -> None:
    """Save SAM LoRA weights or full MaskDecoder"""
    sam_dir = os.path.join(checkpoint_dir, "sam")
    os.makedirs(sam_dir, exist_ok=True)
    
    if config.sam_lora_r > 0 and config.freeze_sam_mask_decoder_base:
        # Save LoRA weights
        lora_weights = _gather_sam_lora_weights(sam_mask_decoder)
        
        if lora_weights:
            if _HAS_SAFE:
                save_path = os.path.join(sam_dir, "lora.safetensors")
                safetensors_save_file(lora_weights, save_path)
            else:
                save_path = os.path.join(sam_dir, "lora.pt")
                torch.save(lora_weights, save_path)
            
            logger.info(f"Saved SAM LoRA weights: {len(lora_weights)} tensors")
            
            # Save LoRA config
            lora_config = {
                "lora_r": config.sam_lora_r,
                "lora_alpha": config.sam_lora_alpha,
                "lora_dropout": config.sam_lora_dropout,
                "is_lora": True,
            }
            with open(os.path.join(sam_dir, "lora_config.json"), "w") as f:
                json.dump(lora_config, f, indent=2)
    
    elif not config.freeze_sam_mask_decoder_base:
        # Save full MaskDecoder
        save_path = os.path.join(sam_dir, "mask_decoder.pt")
        torch.save(sam_mask_decoder.state_dict(), save_path)
        logger.info("Saved full SAM MaskDecoder")
    
    # Save SAM model reference
    sam_info = {
        "sam_model": config.sam_model_name,
        "freeze_mask_decoder": config.freeze_sam_mask_decoder_base,
        "has_lora": config.sam_lora_r > 0 and config.freeze_sam_mask_decoder_base,
    }
    with open(os.path.join(sam_dir, "sam_info.json"), "w") as f:
        json.dump(sam_info, f, indent=2)


def save_adapters(checkpoint_dir: str, model) -> None:
    """Save adapter modules"""
    adapters_dir = os.path.join(checkpoint_dir, "adapters")
    os.makedirs(adapters_dir, exist_ok=True)
    
    # Image adapter
    if hasattr(model, 'image_adapter'):
        save_path = os.path.join(adapters_dir, "image_adapter.pt")
        torch.save(model.image_adapter.state_dict(), save_path)
        logger.info("Saved image_adapter")
    
    # Text prompt projector
    if hasattr(model, 'text_prompt_proj'):
        save_path = os.path.join(adapters_dir, "text_prompt_proj.pt")
        torch.save(model.text_prompt_proj.state_dict(), save_path)
        logger.info("Saved text_prompt_proj")
    
    # Token-FPN or HighResGenerator
    if hasattr(model, 'token_fpn') and model.use_token_fpn:
        save_path = os.path.join(adapters_dir, "token_fpn.pt")
        torch.save(model.token_fpn.state_dict(), save_path)
        logger.info("Saved token_fpn")
    elif hasattr(model, 'high_res_generator'):
        save_path = os.path.join(adapters_dir, "high_res_generator.pt")
        torch.save(model.high_res_generator.state_dict(), save_path)
        logger.info("Saved high_res_generator")


def save_training_state(checkpoint_dir: str, optimizer, scheduler=None, scaler=None, 
                        global_step=0, best_loss=float('inf'), epoch=0) -> None:
    """Save training state for resuming"""
    training_dir = os.path.join(checkpoint_dir, "training")
    os.makedirs(training_dir, exist_ok=True)
    
    # Save optimizer
    torch.save(optimizer.state_dict(), os.path.join(training_dir, "optimizer.pt"))
    
    # Save scheduler if exists
    if scheduler is not None:
        torch.save(scheduler.state_dict(), os.path.join(training_dir, "scheduler.pt"))
    
    # Save scaler if exists (for AMP)
    if scaler is not None:
        torch.save(scaler.state_dict(), os.path.join(training_dir, "scaler.pt"))
    
    # Save training metadata
    training_meta = {
        "global_step": global_step,
        "best_loss": best_loss,
        "epoch": epoch,
    }
    with open(os.path.join(training_dir, "training_state.json"), "w") as f:
        json.dump(training_meta, f, indent=2)
    
    logger.info(f"Saved training state at step {global_step}, epoch {epoch}")


def save_lisa_checkpoint(
    checkpoint_dir: str,
    model,
    processor: ProcessorMixin,
    config,
    optimizer=None,
    scheduler=None,
    scaler=None,
    global_step: int = 0,
    best_loss: float = float('inf'),
    epoch: int = 0,
    additional_info: Dict[str, Any] = None,
) -> None:
    """
    Save complete LISA model checkpoint
    
    Args:
        checkpoint_dir: Directory to save checkpoint
        model: LISA_Model instance
        processor: AutoProcessor instance
        config: LISAConfig instance
        optimizer: Optimizer (optional)
        scheduler: Learning rate scheduler (optional)
        scaler: GradScaler for AMP (optional)
        global_step: Current training step
        best_loss: Best validation loss so far
        epoch: Current epoch
        additional_info: Additional metadata to save
    """
    logger.info(f"Saving checkpoint to {checkpoint_dir}")
    
    # 1. Save metadata
    _write_metadata(checkpoint_dir, config, additional_info)
    
    # 2. Save processor (tokenizer + image preprocessor)
    save_processor(checkpoint_dir, processor)
    
    # 3. Save model config
    config_path = os.path.join(checkpoint_dir, "config.pt")
    torch.save(config, config_path)
    
    # 4. Save Qwen component
    save_qwen_component(checkpoint_dir, model.qwen, config)
    
    # 5. Save special parameters
    save_seg_token_embedding(checkpoint_dir, model, processor.tokenizer)
    save_prompt_beta(checkpoint_dir, model.prompt_beta)
    
    # 6. Save SAM components
    save_sam_lora(checkpoint_dir, model.sam_mask_decoder, config)
    
    # 7. Save adapters
    save_adapters(checkpoint_dir, model)
    
    # 8. Save training state if optimizer provided
    if optimizer is not None:
        save_training_state(checkpoint_dir, optimizer, scheduler, scaler, 
                          global_step, best_loss, epoch)
    
    # 9. Create checkpoint index file
    checkpoint_info = {
        "checkpoint_dir": checkpoint_dir,
        "timestamp": datetime.datetime.now().isoformat(),
        "global_step": global_step,
        "epoch": epoch,
        "best_loss": best_loss,
        "model_type": "LISA_Model",
        "config_class": "LISAConfig",
    }
    
    with open(os.path.join(checkpoint_dir, "checkpoint_info.json"), "w") as f:
        json.dump(checkpoint_info, f, indent=2)
    
    logger.info(f"Checkpoint saved successfully to {checkpoint_dir}")


def load_qwen_component(checkpoint_dir: str, base_model=None, device='cuda'):
    """Load Qwen model or apply LoRA adapter"""
    qwen_dir = os.path.join(checkpoint_dir, "qwen")
    
    # Read base model info
    base_info_path = os.path.join(qwen_dir, "base_model_info.json")
    if os.path.exists(base_info_path):
        with open(base_info_path, 'r') as f:
            base_info = json.load(f)
        
        # Load base model if not provided
        if base_model is None:
            from transformers import Qwen2_5_VLForConditionalGeneration
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_info["base_model"],
                torch_dtype=torch.bfloat16,
                device_map=device
            )
            logger.info(f"Loaded Qwen base model from {base_info['base_model']}")
        
        # Apply LoRA if exists
        if base_info.get("is_lora", False):
            adapter_config_path = os.path.join(qwen_dir, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                try:
                    from peft import PeftModel
                    qwen_model = PeftModel.from_pretrained(base_model, qwen_dir)
                    logger.info("Applied Qwen LoRA adapter")
                    return qwen_model
                except Exception as e:
                    logger.warning(f"Failed to load PEFT adapter: {e}")
                    return base_model
    
    # Check if full model saved
    model_path = os.path.join(qwen_dir, "model.safetensors")
    if os.path.exists(model_path) or os.path.exists(os.path.join(qwen_dir, "pytorch_model.bin")):
        from transformers import Qwen2_5_VLForConditionalGeneration
        qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(qwen_dir)
        logger.info("Loaded full Qwen model")
        return qwen_model
    
    return base_model


def apply_sam_lora_weights(sam_mask_decoder, checkpoint_dir: str):
    """Apply SAM LoRA weights to MaskDecoder"""
    sam_dir = os.path.join(checkpoint_dir, "sam")
    
    # Try safetensors first
    lora_path = os.path.join(sam_dir, "lora.safetensors")
    if not os.path.exists(lora_path):
        lora_path = os.path.join(sam_dir, "lora.pt")
    
    if os.path.exists(lora_path):
        if lora_path.endswith(".safetensors"):
            lora_weights = safetensors_load_file(lora_path)
        else:
            lora_weights = torch.load(lora_path, map_location='cpu')
        
        # Apply weights to LoRALinear modules
        for name, module in sam_mask_decoder.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                lora_a_key = f"{name}.lora_A"
                lora_b_key = f"{name}.lora_B"
                
                if lora_a_key in lora_weights:
                    module.lora_A.data.copy_(lora_weights[lora_a_key])
                if lora_b_key in lora_weights:
                    module.lora_B.data.copy_(lora_weights[lora_b_key])
                
                scaling_key = f"{name}.scaling"
                if scaling_key in lora_weights and hasattr(module, 'scaling'):
                    module.scaling = float(lora_weights[scaling_key])
        
        logger.info(f"Applied SAM LoRA weights from {lora_path}")
    
    # Or load full MaskDecoder
    mask_decoder_path = os.path.join(sam_dir, "mask_decoder.pt")
    if os.path.exists(mask_decoder_path):
        state_dict = torch.load(mask_decoder_path, map_location='cpu')
        sam_mask_decoder.load_state_dict(state_dict)
        logger.info("Loaded full SAM MaskDecoder")


def load_lisa_checkpoint(
    checkpoint_dir: str,
    model=None,
    processor=None,
    device='cuda',
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load LISA model checkpoint
    
    Args:
        checkpoint_dir: Directory containing checkpoint
        model: Existing LISA_Model to load weights into (optional)
        processor: Existing processor (optional)
        device: Device to load model on
        strict: Whether to strictly enforce state dict matching
    
    Returns:
        Dictionary with loaded components and metadata
    """
    checkpoint_dir = Path(checkpoint_dir)
    logger.info(f"Loading checkpoint from {checkpoint_dir}")
    
    # 1. Load config
    config_path = checkpoint_dir / "config.pt"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = torch.load(config_path, map_location=device, weights_only=False)
    
    # 2. Load processor if not provided
    if processor is None:
        from transformers import AutoProcessor
        processor_dir = checkpoint_dir / "processor"
        if processor_dir.exists():
            processor = AutoProcessor.from_pretrained(str(processor_dir))
            logger.info("Loaded processor")
    
    # 3. Initialize model if not provided
    if model is None:
        from src.models.lisa_model import LISA_Model
        model = LISA_Model(config=config)
        logger.info("Initialized new LISA model")
    
    # 4. Load Qwen component
    model.qwen = load_qwen_component(str(checkpoint_dir), model.qwen, device)
    
    # 5. Load special parameters
    # SEG token embedding
    seg_token_path = checkpoint_dir / "seg_token_embedding.pt"
    if seg_token_path.exists():
        seg_data = torch.load(seg_token_path, map_location=device, weights_only=False)
        if 'embedding' in seg_data:
            model.seg_token_embedding = nn.Parameter(seg_data['embedding'].to(device))
            model.seg_token_id = seg_data['seg_token_id']
            logger.info(f"Loaded SEG token embedding for ID {model.seg_token_id}")
    
    # Prompt beta
    prompt_beta_path = checkpoint_dir / "prompt_beta.pt"
    if prompt_beta_path.exists():
        beta_data = torch.load(prompt_beta_path, map_location=device, weights_only=False)
        model.prompt_beta.data = beta_data['prompt_beta'].to(device)
        logger.info(f"Loaded prompt_beta: {model.prompt_beta.item():.6f}")
    
    # 6. Load SAM components
    apply_sam_lora_weights(model.sam_mask_decoder, str(checkpoint_dir))
    
    # 7. Load adapters
    adapters_dir = checkpoint_dir / "adapters"
    
    # Image adapter
    image_adapter_path = adapters_dir / "image_adapter.pt"
    if image_adapter_path.exists():
        state_dict = torch.load(image_adapter_path, map_location=device, weights_only=True)
        model.image_adapter.load_state_dict(state_dict, strict=strict)
        logger.info("Loaded image_adapter")
    
    # Text prompt projector
    text_proj_path = adapters_dir / "text_prompt_proj.pt"
    if text_proj_path.exists():
        state_dict = torch.load(text_proj_path, map_location=device, weights_only=True)
        model.text_prompt_proj.load_state_dict(state_dict, strict=strict)
        logger.info("Loaded text_prompt_proj")
    
    # Token-FPN or HighResGenerator
    token_fpn_path = adapters_dir / "token_fpn.pt"
    high_res_path = adapters_dir / "high_res_generator.pt"
    
    if token_fpn_path.exists() and hasattr(model, 'token_fpn'):
        state_dict = torch.load(token_fpn_path, map_location=device, weights_only=True)
        model.token_fpn.load_state_dict(state_dict, strict=strict)
        logger.info("Loaded token_fpn")
    elif high_res_path.exists() and hasattr(model, 'high_res_generator'):
        state_dict = torch.load(high_res_path, map_location=device, weights_only=True)
        model.high_res_generator.load_state_dict(state_dict, strict=strict)
        logger.info("Loaded high_res_generator")
    
    # 8. Load training state if exists
    training_state = {}
    training_dir = checkpoint_dir / "training"
    if training_dir.exists():
        training_meta_path = training_dir / "training_state.json"
        if training_meta_path.exists():
            with open(training_meta_path, 'r') as f:
                training_state = json.load(f)
            logger.info(f"Loaded training state: step {training_state.get('global_step', 0)}")
    
    # 9. Load checkpoint info
    checkpoint_info = {}
    info_path = checkpoint_dir / "checkpoint_info.json"
    if info_path.exists():
        with open(info_path, 'r') as f:
            checkpoint_info = json.load(f)
    
    model.to(device)
    logger.info(f"Checkpoint loaded successfully from {checkpoint_dir}")
    
    return {
        'model': model,
        'processor': processor,
        'config': config,
        'training_state': training_state,
        'checkpoint_info': checkpoint_info,
    }