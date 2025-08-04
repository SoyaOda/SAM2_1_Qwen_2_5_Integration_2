"""
Configuration for LISA改 (LISA-Kai) model
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class LISAConfig:
    """Configuration for LISA改 model"""
    
    # Model paths
    qwen_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    sam_model_name: str = "facebook/sam2.1-hiera-large"
    
    # Model dimensions
    # Note: These are the actual dimensions for Qwen2.5-VL-3B (not the spec examples)
    qwen_hidden_size: Optional[int] = None  # Will be auto-detected (actual: 2048)
    qwen_vision_hidden_size: Optional[int] = None  # Will be auto-detected (actual: 1280)
    sam_image_embedding_dim: int = 256
    
    # Adapter dimensions
    image_adapter_out_dim: int = 256
    text_prompt_out_dim: int = 256
    
    # Special tokens
    seg_token: str = "<SEG>"
    
    # LoRA configuration
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: list = None  # Will be set based on model architecture
    
    # Training configuration
    freeze_qwen: bool = True
    freeze_sam: bool = True
    train_seg_token: bool = True
    train_adapters: bool = True
    
    # Loss weights
    language_loss_weight: float = 1.0
    segmentation_loss_weight: float = 1.0
    
    # Device settings
    device_map: str = "auto"
    torch_dtype: str = "auto"
    use_flash_attention: bool = False
    
    # Dataset configuration
    dataset_base_dir: str = "/mnt/h/download/LISA-dataset/data/dataset"
    model_max_length: int = 2048
    qwen_image_size: int = 448
    sam_image_size: int = 1024
    
    # Dataset paths
    sem_seg_data: str = "ade20k||cocostuff"
    refer_seg_data: str = "refcoco||refcoco+||refcocog"
    vqa_data: str = "llava_instruct_150k"
    reason_seg_data: str = "ReasonSeg|train"
    
    # Dataset sampling
    dataset_config: str = "sem_seg||refer_seg||vqa||reason_seg"
    sample_rates: list = None  # Will be initialized in __post_init__
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            # Default target modules for Qwen cross-attention
            self.lora_target_modules = [
                "cross_attn",
                "cross_attention",
                "encoder_attn",
                "encoder_attention"
            ]
        
        if self.sample_rates is None:
            # Default sample rates for datasets [sem_seg, refer_seg, vqa, reason_seg]
            self.sample_rates = [9, 3, 3, 1]