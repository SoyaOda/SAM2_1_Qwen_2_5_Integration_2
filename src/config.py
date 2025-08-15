"""
Configuration for LISA改 (LISA-Kai) model
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class LISAConfig:
    """
    Configuration for LISA改 (LISA-Kai) model
    
    すべての学習パラメータとモデル設定はこのファイルで一元管理されています。
    minimal_train.pyのコマンドライン引数では変更できません（設計による制約）。
    パラメータを変更する場合は、このファイルを直接編集してください。
    """
    
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
    
    # ========================================================================
    # LoRA Configuration - すべてのLoRA設定はここで一元管理
    # ========================================================================
    
    # Qwen2.5-VL LoRA settings
    # これらの値を変更する場合は、このファイルを直接編集してください
    # minimal_train.pyのコマンドライン引数では変更できません（設計による制約）
    # NOTE: lora_r=0 にするとQwenのLoRAは完全に無効化されます（言語モデル・Visual ViT両方）
    #       この場合、freeze_qwen_lora設定に関わらずLoRAは適用されません
    lora_r: int = 8                    # 0 to disable, 4-16 for enabling LoRA on Qwen
    lora_alpha: int = 32               # LoRAのスケーリング係数（通常はrの2-4倍、学習初期の安定性に影響）
    lora_dropout: float = 0.1          # LoRAのドロップアウト率（過学習防止）
    lora_target_modules: list = None   # 自動設定（lora_visual_enabledに基づいて__post_init__で決定）
    lora_visual_enabled: bool = False  # Visual ViT blocksへのLoRA適用（lora_r>0の場合のみ有効）
                                       # False: 言語モデルのみ（既存チェックポイントと互換、デフォルト）
                                       # True: 言語モデル＋Visual ViT（より高性能だが既存チェックポイントと非互換）
    
    # SAM2.1 MaskDecoder LoRA settings
    # NOTE: sam_lora_r=0 にするとSAMのLoRAは完全に無効化されます
    #       sam_lora_r>0 の場合、freeze_sam_mask_decoder_base=True のときのみLoRAが適用されます
    #       freeze_sam_mask_decoder_base=False の場合はMaskDecoder全体を直接学習（LoRA無視）
    sam_lora_r: int = 8  # 0 to disable, 4-8 for enabling LoRA on SAM MaskDecoder
    sam_lora_alpha: int = 16  # LoRA alpha for MaskDecoder (通常はlora_alphaの半分程度が推奨)
    sam_lora_dropout: float = 0.1  # LoRA dropout for MaskDecoder
    sam_lora_target_modules: list = None  # Will be set in __post_init__
    
    # ========================================================================
    # Training Configuration - Centralized control for all components
    # ========================================================================
    
    # ---- Freeze Flags (What NOT to train) ----
    # Qwen2.5-VL
    freeze_qwen_lora: bool = False          # Qwen LoRA adapters (2.5M params) - False = trainable
    freeze_seg_token: bool = False          # SEG token embedding - False = trainable
    
    # SAM2.1
    # NOTE: freeze_sam_lora は現在未使用です。SAM LoRAの適用は以下の条件で自動制御されます：
    #   - freeze_sam_mask_decoder_base=True かつ sam_lora_r>0 → LoRA適用（61K params学習可能）
    #   - freeze_sam_mask_decoder_base=False → MaskDecoder直接学習（4.2M params学習可能）
    freeze_sam_lora: bool = False           # [DEPRECATED] この設定は無視されます
    
    # Adapter Components  
    freeze_image_adapter: bool = False      # Qwen→SAM feature adapter (1.2M params) - False = trainable
    freeze_text_prompt_projector: bool = False  # LLM→SAM embedding projector (1.2M params) - False = trainable
    freeze_token_fpn: bool = False          # Multi-scale feature extractor (3.7M params) - False = trainable
    freeze_prompt_beta: bool = False        # Embedding fusion weight β (1 param) - False = trainable
    freeze_image_fusion_beta: bool = False  # Image feature fusion weight β (1 param) - False = trainable
    
    # Fusion configuration
    # Based on research: Cross-Attention recommended for accuracy (Reasoning Segmentation)
    # Sigma-Add as fallback for computational constraints
    #
    # Recommended settings based on GPU resources:
    # - GPU 16GB+: fusion_type="cross_attention" (best accuracy)
    # - GPU 8-16GB: fusion_type="cross_attention" with smaller batch_size
    # - GPU <8GB: fusion_type="sigma_add" (lightest computation)
    fusion_type: str = "sigma_add"  # Options: "sigma_add", "cross_attention" 
    fusion_num_heads: int = 8  # 8 heads optimal for 256-dim (SAM standard)
    fusion_dropout: float = 0.1  # Transformer/ViT standard for generalization
    fusion_use_gate: bool = True  # Dynamic mixing improves stability & accuracy
    debug_fusion: bool = False  # Enable debug logging for fusion operations
    
    # ---- Always Frozen Components ----
    # Qwen2.5-VL
    freeze_qwen_base: bool = True          # ALWAYS freeze Qwen base model (3.7B params)
    
    # SAM2.1
    freeze_sam_image_encoder: bool = True  # ALWAYS True - not used (saves 212M params!)
    freeze_sam_mask_decoder_base: bool = True   # True: LoRA適用（sam_lora_r>0なら）, False: 直接学習（4.2M）
    freeze_sam_prompt_encoder: bool = True # Freeze SAM PromptEncoder
    freeze_sam_memory_attention: bool = True  # ALWAYS True - video only (saves 8.3M params!)
    
    # ---- Legacy/Compatibility (DO NOT USE) ----
    freeze_qwen: bool = True               # Deprecated - use freeze_qwen_base
    freeze_sam: bool = True                # Deprecated - use specific freeze_sam_* flags
    train_seg_token: bool = None           # Deprecated - use freeze_seg_token=False instead
    train_adapters: bool = None            # Deprecated - use specific freeze_* flags
    
    # ========================================================================
    # Training Configuration - 学習パラメータの一元管理
    # ========================================================================
    
    # Learning rates (各コンポーネントの学習率)
    adapter_lr: float = 1e-3           # アダプター（Image Adapter, Text Projector等）の学習率
    lora_lr: float = 1e-4              # LoRA（Qwen/SAM）の学習率
    seg_token_lr: float = 5e-5         # SEGトークン埋め込みの学習率
    weight_decay: float = 0.01         # Weight decay (AdamW用)
    
    # Loss weights
    language_loss_weight: float = 1.0      # 言語モデリング損失の重み
    segmentation_loss_weight: float = 1.0  # セグメンテーション損失の重み
    
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
            # Qwen2.5-VL has two parts that can be targeted with LoRA:
            # 1. Language model layers: standard q,k,v,o,gate,up,down projections
            # 2. Visual ViT blocks: visual.blocks.*.attn.{qkv,proj} and visual.blocks.*.mlp.{gate,up,down}_proj
            # 
            # Language model projections (always included for Qwen SFT)
            language_modules = [
                "q_proj",
                "k_proj", 
                "v_proj",
                "o_proj",      # Output projection
                "up_proj",     # MLP up projection
                "gate_proj",   # MLP gate projection  
                "down_proj",   # MLP down projection
            ]
            
            # Visual ViT blocks - only included if lora_visual_enabled=True
            # This matches visual.blocks.{0-31}.attn.{qkv,proj} and visual.blocks.{0-31}.mlp.{gate_proj,up_proj,down_proj}
            visual_regex = r"(?:^|.*)visual\.blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.(?:up_proj|gate_proj|down_proj))$"
            
            if self.lora_visual_enabled:
                # Include both language and visual components
                self.lora_target_modules = language_modules + [visual_regex]
            else:
                # Language-only (backward compatible with existing checkpoints)
                self.lora_target_modules = language_modules
        
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