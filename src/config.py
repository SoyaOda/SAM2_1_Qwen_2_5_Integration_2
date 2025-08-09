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
    model_max_length: int = 16384  # Extended for dynamic resolution
    
    # Dynamic resolution configuration for Qwen2.5-VL (Following o3_spec3.md)
    qwen_min_pixels: int = 336 * 336  # 112,896 - Minimum ~24×24 patches
    qwen_max_pixels: int = 2688 * 2688  # 7,225,344 - Maximum ~192×192 patches
    qwen_patch_size: int = 14  # ViT patch size
    qwen_merge_size: int = 2  # 2x2 patch merging for compression
    
    # Resolution buckets for efficient batching
    resolution_buckets: list = None  # Will be initialized in __post_init__
    use_quality_score: bool = True  # Enable quality-based sample weighting
    quality_score_threshold: float = 0.7  # Threshold for low-quality images
    
    # Legacy fixed size settings (kept for backward compatibility)
    qwen_image_size: int = 448  # Used as fallback when dynamic resolution is disabled
    sam_image_size: int = 1024
    
    # Dynamic resolution control
    use_dynamic_resolution: bool = True  # Enable/disable dynamic resolution
    
    # Token selection strategy
    use_token_selection: bool = False  # Disabled for segmentation tasks (need all tokens)
    
    # Vision feature extraction settings
    use_patchmerge_features: bool = False  # Currently False (2048-dim); will be True when PatchMerge features are available
    vision_feature_dim: int = 2048  # Currently 2048 (LLM-projected); will be 2560 for PatchMerge
    token_selection_strategy: str = "none"  # Options: "top128", "none" - Use "none" for segmentation
    
    # SAM2.1 MaskDecoder LoRA configuration
    sam_lora_r: int = 8  # 0 to disable, 4-8 for enabling LoRA on MaskDecoder
    sam_lora_alpha: int = 16  # LoRA alpha for MaskDecoder
    sam_lora_dropout: float = 0.1  # LoRA dropout for MaskDecoder
    sam_lora_target_modules: list = None  # Will be set in __post_init__
    
    # Dataset paths
    sem_seg_data: str = "ade20k||cocostuff"
    refer_seg_data: str = "refcoco||refcoco+||refcocog"
    vqa_data: str = "llava_instruct_150k"
    reason_seg_data: str = "ReasonSeg|train"
    
    # Token-FPN configuration
    use_token_fpn: bool = True  # Enable Token-FPN for multi-scale feature extraction
    fpn_layer_indices: list = None  # Will be initialized in __post_init__
    
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
        
        if self.sam_lora_target_modules is None:
            # Default target modules for SAM2.1 MaskDecoder attention layers
            self.sam_lora_target_modules = [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "cross_attn_token_to_image.q_proj",
                "cross_attn_token_to_image.k_proj",
                "cross_attn_token_to_image.v_proj",
                "cross_attn_image_to_token.q_proj",
                "cross_attn_image_to_token.k_proj",
                "cross_attn_image_to_token.v_proj"
            ]
        
        if self.fpn_layer_indices is None:
            # Default layer indices for Token-FPN (early, mid, late, final)
            # For Qwen2.5-VL-3B with 32 blocks: 8, 16, 24, 31
            self.fpn_layer_indices = [8, 16, 24, 31]
        
        if self.sample_rates is None:
            # Default sample rates for datasets [sem_seg, refer_seg, vqa, reason_seg]
            self.sample_rates = [9, 3, 3, 1]
        
        if self.resolution_buckets is None:
            # Optimal resolution buckets for batching
            # Format: (pixels, grid_size, token_count)
            self.resolution_buckets = [
                (448 * 448, (32, 32), 256),   # Small: 32×32 patches, 256 tokens
                (672 * 672, (48, 48), 576),   # Medium: 48×48 patches, 576 tokens
                (896 * 896, (64, 64), 1024),  # Large: 64×64 patches, 1024 tokens
            ]