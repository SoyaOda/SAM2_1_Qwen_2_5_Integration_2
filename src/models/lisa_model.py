"""
LISA改 (LISA-Kai) Model Implementation
Integration of Qwen2.5-VL-3B and SAM2.1 for reasoning segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings
import logging

logger = logging.getLogger(__name__)

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor
)
from sam2.sam2_image_predictor import SAM2ImagePredictor
from peft import LoraConfig, get_peft_model, TaskType

from .adapters import ImageFeatureAdapter, TextPromptProjector
from .token_fpn import TokenFPN
from .fusion_layers import HybridFusionModule
from ..config import LISAConfig


@dataclass
class LISAModelOutput:
    """Output format for LISA model"""
    logits: torch.Tensor  # Language model logits [B, seq_len, vocab_size]
    mask_logits: Optional[List[torch.Tensor]]  # List of mask predictions
    language_hidden_states: Optional[torch.Tensor] = None  # LLM hidden states
    vision_hidden_states: Optional[torch.Tensor] = None  # Vision encoder hidden states
    seg_token_positions: Optional[List[int]] = None  # Positions of SEG tokens


class HighResFeatureGenerator(nn.Module):
    """
    Generates high-resolution features from Qwen vision features for SAM2.1 mask decoder.
    Implements a lightweight FPN (Feature Pyramid Network) approach.
    """
    def __init__(self, in_channels: int, sam_channels: int = 256):
        """
        Args:
            in_channels: Input channels from Qwen vision features (256 after adapter)
            sam_channels: SAM2.1 transformer_dim (256 by default)
        """
        super().__init__()
        
        # We don't need to reduce channels since input is already 256
        # from the image adapter
        
        # Important: SAM2.1's MaskDecoder expects the high-res features to have
        # the SAME channels as image_embeddings (256), then it will apply its own
        # conv_s0 and conv_s1 to reduce to 32 and 64 channels respectively
        
        # Upsampling layers for generating high-res features
        # For stride-8 feature (2x upsampling)
        self.upsample_2x = nn.Sequential(
            nn.ConvTranspose2d(in_channels, sam_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(sam_channels),
            nn.GELU()
        )
        
        # For stride-4 feature (4x upsampling)
        self.upsample_4x = nn.Sequential(
            nn.ConvTranspose2d(in_channels, sam_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(sam_channels),
            nn.GELU(),
            nn.ConvTranspose2d(sam_channels, sam_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(sam_channels),
            nn.GELU()
        )
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: Input features [B, C, H, W] from image adapter (already 256 channels)
               
        Returns:
            List of high-res features [feat_s0, feat_s1] where:
            - feat_s0: [B, 256, 4*H, 4*W] for stride-4
            - feat_s1: [B, 256, 2*H, 2*W] for stride-8
            
        Note:
            We output 256 channels (same as sam_channels) and let SAM2.1's
            MaskDecoder apply its own conv_s0/conv_s1 to reduce to 32/64 channels.
        """
        # Generate multi-scale features
        feat_s1 = self.upsample_2x(x)   # [B, 256, 2*H, 2*W]
        feat_s0 = self.upsample_4x(x)   # [B, 256, 4*H, 4*W]
        
        # Return in the order expected by SAM2.1 (stride-4, stride-8)
        # MaskDecoder will apply conv_s0/conv_s1 to reduce channels to 32/64
        return [feat_s0, feat_s1]

class LISA_Model(nn.Module):
    """
    LISA改 Model
    
    Integrates Qwen2.5-VL for vision-language understanding with SAM2.1 for
    high-precision mask generation, connected through learned adapters.
    """
    
    def __init__(
        self,
        config: LISAConfig,
        qwen_model: Optional[Qwen2_5_VLForConditionalGeneration] = None,
        sam_predictor: Optional[SAM2ImagePredictor] = None,
        tokenizer: Optional[AutoTokenizer] = None,
    ):
        """
        Initialize LISA model
        
        Args:
            config: Model configuration
            qwen_model: Pre-loaded Qwen model (optional)
            sam_predictor: Pre-loaded SAM predictor (optional)
            tokenizer: Pre-loaded tokenizer (optional)
        """
        super().__init__()
        self.config = config
        
        # Load Qwen2.5-VL model
        if qwen_model is not None:
            self.qwen = qwen_model
        else:
            self.qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                config.qwen_model_name,
                torch_dtype=config.torch_dtype,
                device_map=config.device_map,
                attn_implementation="flash_attention_2" if config.use_flash_attention else "eager"
            )
        
        # Disable top128 token selection for segmentation tasks (use all tokens)
        # This is critical for dense prediction tasks like segmentation
        # Reference: md_files/current/o3_query_answers.md line 1585
        
        # Set on multiple levels to ensure it takes effect
        # 1. Visual module config
        if hasattr(self.qwen.model, 'visual'):
            if hasattr(self.qwen.model.visual, 'config'):
                self.qwen.model.visual.config.image_feature_select_strategy = "none"
                logger.info("Set visual.config.image_feature_select_strategy = 'none'")
        
        # 2. Model config (sometimes used by get_image_features)
        if hasattr(self.qwen, 'config'):
            self.qwen.config.image_feature_select_strategy = "none"
            logger.info("Set model.config.image_feature_select_strategy = 'none'")
        
        # 3. Model.model config
        if hasattr(self.qwen.model, 'config'):
            self.qwen.model.config.image_feature_select_strategy = "none"
            logger.info("Set model.model.config.image_feature_select_strategy = 'none'")
        
        logger.info("Disabled top128 token selection - using all visual tokens for segmentation")
        
        # Load SAM2.1 components
        if sam_predictor is not None:
            self.sam_predictor = sam_predictor
            self.sam_model = sam_predictor.model
        else:
            # Build SAM2 model
            from sam2.build_sam import build_sam2, build_sam2_video_predictor
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            # Download checkpoint if using HF model name
            if config.sam_model_name.startswith("facebook/"):
                # For HF models, we need to download and use local checkpoint
                # This is a simplified approach - in production, handle this better
                import warnings
                warnings.warn(
                    "Loading SAM2 from HuggingFace requires manual checkpoint download. "
                    "Using default checkpoint path."
                )
                checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
            else:
                checkpoint = config.sam_model_name
            
            # Build SAM2 model using the automatic model builder
            try:
                # Try the newer API
                self.sam_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large")
            except:
                # Fallback to manual loading
                # For SAM2.1, we need to specify the full config path
                import os
                import sam2
                sam2_root = os.path.dirname(sam2.__file__)
                config_path = os.path.join(sam2_root, "configs", "sam2.1", "sam2.1_hiera_l.yaml")
                
                # Build model with absolute config path
                sam2_model = build_sam2(config_path, checkpoint, device=config.device_map)
                self.sam_predictor = SAM2ImagePredictor(sam2_model)
            
            self.sam_model = self.sam_predictor.model
        
        # Access mask decoder, prompt encoder, and image encoder
        # SAM2 architecture may vary, so we check for the correct attributes
        if hasattr(self.sam_model, 'sam_mask_decoder'):
            self.sam_mask_decoder = self.sam_model.sam_mask_decoder
        elif hasattr(self.sam_model, 'mask_decoder'):
            self.sam_mask_decoder = self.sam_model.mask_decoder
        else:
            raise AttributeError("Cannot find mask decoder in SAM model")
        
        if hasattr(self.sam_model, 'sam_prompt_encoder'):
            self.sam_prompt_encoder = self.sam_model.sam_prompt_encoder
        elif hasattr(self.sam_model, 'prompt_encoder'):
            self.sam_prompt_encoder = self.sam_model.prompt_encoder
        else:
            raise AttributeError("Cannot find prompt encoder in SAM model")
        
        # Access image encoder for high-res features
        if hasattr(self.sam_model, 'image_encoder'):
            self.sam_image_encoder = self.sam_model.image_encoder
        elif hasattr(self.sam_model, 'sam_image_encoder'):
            self.sam_image_encoder = self.sam_model.sam_image_encoder
        else:
            raise AttributeError("Cannot find image encoder in SAM model")
        
        # Get model dimensions
        if config.qwen_hidden_size is None:
            config.qwen_hidden_size = self.qwen.config.hidden_size
        if config.qwen_vision_hidden_size is None:
            # Currently using 2048-dim LLM-projected features
            # Future: will be 2560-dim when PatchMerge features are available
            config.qwen_vision_hidden_size = 2048  # Qwen2.5-VL-3B LLM-projected dimension
        
        # Initialize adapters with the same dtype and device as Qwen model
        model_dtype = next(self.qwen.parameters()).dtype
        model_device = next(self.qwen.parameters()).device
        
        # Image adapter: currently using 2048-dim input (LLM-projected features)
        # Future: will use 2560-dim when PatchMerge features become available
        self.image_adapter = ImageFeatureAdapter(
            in_dim=config.vision_feature_dim if hasattr(config, 'vision_feature_dim') else 2048,  # Default to LLM-projected features
            out_dim=config.sam_image_embedding_dim  # 256 for SAM2.1
        ).to(device=model_device, dtype=model_dtype)
        
        self.text_prompt_proj = TextPromptProjector(
            in_dim=config.qwen_hidden_size,
            out_dim=config.text_prompt_out_dim,
            use_mlp=True  # O3推奨: 2層MLP + LayerNormでマッチング精度向上
        ).to(device=model_device, dtype=model_dtype)
        
        # Initialize Token-FPN for multi-scale feature extraction
        # Token-FPNを使用するかどうかの設定
        self.use_token_fpn = getattr(config, 'use_token_fpn', True)
        
        if self.use_token_fpn:
            # Token-FPNを初期化（Qwen中間層からのマルチスケール特徴抽出）
            self.token_fpn = TokenFPN(
                in_channels=1280,  # Qwen2.5-VL-3Bの中間層チャネル数
                out_channels=config.sam_image_embedding_dim,  # 256
                layer_indices=getattr(config, 'fpn_layer_indices', [8, 16, 24, 31]),
                sam_image_size=config.sam_image_size
            ).to(device=model_device, dtype=model_dtype)
            
            # Token-FPNのフックを登録
            self.token_fpn.register_hooks(self.qwen)
            logger.info("Token-FPN initialized with intermediate layer extraction")
        else:
            # 従来のHighResFeatureGeneratorを使用
            self.high_res_generator = HighResFeatureGenerator(
                in_channels=config.sam_image_embedding_dim,  # Already 256 after adapter
                sam_channels=config.sam_image_embedding_dim
            ).to(device=model_device, dtype=model_dtype)
            logger.info("Using simple HighResFeatureGenerator")
        
        # ========================================================================
        # Apply freeze settings from config - Centralized control
        # ========================================================================
        
        # ---- Qwen2.5-VL Components ----
        if config.freeze_qwen_base:
            for param in self.qwen.parameters():
                param.requires_grad = False
            logger.info("Froze Qwen2.5-VL base model")
        
        # ---- SAM2.1 Components (Fine-grained control) ----
        
        # SAM ImageEncoder - Advanced configuration for high-quality feature extraction
        if config.freeze_sam_image_encoder and hasattr(self, 'sam_image_encoder'):
            for param in self.sam_image_encoder.parameters():
                param.requires_grad = False
            logger.info("Froze SAM ImageEncoder parameters (212M params - gradient updates disabled)")
        
        # Force SAM ImageEncoder to eval mode for consistent inference behavior
        if hasattr(config, 'sam_encoder_eval_mode') and config.sam_encoder_eval_mode:
            if hasattr(self, 'sam_image_encoder'):
                self.sam_image_encoder.eval()
                logger.info("Set SAM ImageEncoder to eval() mode (Dropout/BN disabled for stable features)")
        
        # Store SAM encoder configuration for forward pass
        self.sam_encoder_config = {
            'precision': getattr(config, 'sam_encoder_precision', 'bf16'),
            'use_inference_mode': getattr(config, 'sam_encoder_use_inference_mode', True),
            'cache_embeddings': getattr(config, 'sam_encoder_cache_embeddings', False)
        }
        
        # SAM MaskDecoder
        if config.freeze_sam_mask_decoder_base:
            for param in self.sam_mask_decoder.parameters():
                param.requires_grad = False
            logger.info("Froze SAM MaskDecoder (will use LoRA if enabled)")
        
        # SAM PromptEncoder  
        if config.freeze_sam_prompt_encoder:
            for param in self.sam_prompt_encoder.parameters():
                param.requires_grad = False
            logger.info("Froze SAM PromptEncoder")
        
        # SAM Memory/Video components - Always freeze for image tasks
        # These are for video tracking and not needed (saves 8.3M params!)
        if config.freeze_sam_memory_attention and hasattr(self.sam_model, 'memory_attention'):
            for param in self.sam_model.memory_attention.parameters():
                param.requires_grad = False
            logger.info("Froze SAM memory_attention (video components - saves 8.3M params)")
            
            # Also freeze other video-related components
            video_components = [
                'maskmem_tpos_enc', 'no_mem_embed', 'no_mem_pos_enc',
                'no_obj_ptr', 'no_obj_embed_spatial', 'mask_downsample',
                'memory_encoder', 'obj_ptr_proj', 'spatial_add_pos_embed'  # 追加
            ]
            for comp_name in video_components:
                if hasattr(self.sam_model, comp_name):
                    comp = getattr(self.sam_model, comp_name)
                    if hasattr(comp, 'parameters'):
                        for param in comp.parameters():
                            param.requires_grad = False
                    elif isinstance(comp, nn.Parameter):
                        comp.requires_grad = False
            logger.info("Froze other SAM video components")
            
            # さらに、sam_model内のすべてのパラメータを走査して凍結
            # memory_encoder, obj_ptr_proj等が確実に凍結されるようにする
            for name, param in self.sam_model.named_parameters():
                if ('memory_encoder' in name or 'obj_ptr' in name or 
                    'spatial_add_pos_embed' in name or 'mem_' in name):
                    param.requires_grad = False
        
        # Freeze SAM components individually
        elif config.freeze_sam_mask_decoder_base and config.freeze_sam_prompt_encoder:
            # Original behavior: freeze MaskDecoder and PromptEncoder
            for param in self.sam_mask_decoder.parameters():
                param.requires_grad = False
            for param in self.sam_prompt_encoder.parameters():
                param.requires_grad = False
            logger.info("Froze SAM components (legacy freeze_sam=True)")
        
        # ---- Adapter Components ----
        # Control training of adapter components
        if hasattr(self, 'image_adapter') and config.freeze_image_adapter:
            for param in self.image_adapter.parameters():
                param.requires_grad = False
            logger.info("Froze Image Adapter")
        
        if hasattr(self, 'text_prompt_proj') and config.freeze_text_prompt_projector:
            for param in self.text_prompt_proj.parameters():
                param.requires_grad = False
            logger.info("Froze Text Prompt Projector")
        
        if hasattr(self, 'token_fpn') and config.freeze_token_fpn:
            for param in self.token_fpn.parameters():
                param.requires_grad = False
            logger.info("Froze Token-FPN")
        
        # 加算アプローチ用の学習可能なスケーリング係数β
        self.prompt_beta = nn.Parameter(torch.tensor(0.01))
        
        # Image feature fusion module for SAM-Qwen feature fusion
        self.fusion_type = config.fusion_type if hasattr(config, 'fusion_type') else 'sigma_add'
        self.debug_fusion = config.debug_fusion if hasattr(config, 'debug_fusion') else False
        
        if self.fusion_type == "cross_attention":
            # Use cross-attention fusion
            self.image_fusion = HybridFusionModule(
                dim=256,
                fusion_type="cross_attention",
                num_heads=config.fusion_num_heads if hasattr(config, 'fusion_num_heads') else 8,
                dropout=config.fusion_dropout if hasattr(config, 'fusion_dropout') else 0.1,
                use_gate=config.fusion_use_gate if hasattr(config, 'fusion_use_gate') else True
            )
            logger.info(f"Added Cross-Attention fusion module for SAM-Qwen features")
        else:
            # Use sigma-add fusion (backward compatibility)
            # Create a single beta parameter that will be shared
            self.image_fusion_beta = nn.Parameter(torch.zeros(1))
            self.image_fusion = HybridFusionModule(
                dim=256,
                fusion_type="sigma_add",
                external_beta=self.image_fusion_beta  # Share the beta parameter
            )
            logger.info("Added Sigma-Add fusion (image_fusion_beta) for SAM-Qwen features")
        
        # Freeze prompt_beta if specified
        if config.freeze_prompt_beta:
            self.prompt_beta.requires_grad = False
            logger.info("Froze Prompt Beta")
        
        # Freeze image fusion parameters if specified
        if hasattr(config, 'freeze_image_fusion_beta') and config.freeze_image_fusion_beta:
            if self.fusion_type == "sigma_add" and hasattr(self, 'image_fusion_beta'):
                self.image_fusion_beta.requires_grad = False
                logger.info("Froze image_fusion_beta")
            # Apply freeze to all fusion types uniformly
            for param in self.image_fusion.parameters():
                param.requires_grad = False
            logger.info(f"Froze {self.fusion_type} fusion module")
        
        # Store tokenizer and SEG token info
        self.tokenizer = tokenizer
        self.seg_token_id = None
        
        # ========================================================================
        # SAM LoRA Configuration
        # ========================================================================
        # IMPORTANT: SAM LoRAはMaskDecoderがfreezeされている場合のみ適用
        # MaskDecoderがunfreeze（学習可能）の場合、LoRAは無効化される
        if config.sam_lora_r > 0 and config.freeze_sam_mask_decoder_base:
            # MaskDecoderがfreezeの場合のみLoRAを適用
            logger.info(f"SAM MaskDecoder is frozen. Applying LoRA (r={config.sam_lora_r})...")
            self.add_sam_lora(
                lora_r=config.sam_lora_r,
                lora_alpha=config.sam_lora_alpha,
                lora_dropout=config.sam_lora_dropout
            )
        elif config.sam_lora_r > 0 and not config.freeze_sam_mask_decoder_base:
            # MaskDecoderがunfreezeの場合、LoRAを無効化
            logger.warning(
                f"SAM MaskDecoder is unfrozen (freeze_sam_mask_decoder_base=False). "
                f"LoRA will be disabled even though sam_lora_r={config.sam_lora_r} is set. "
                f"MaskDecoder will be trained directly."
            )
        else:
            logger.info("SAM LoRA disabled (sam_lora_r=0)")
        
        # ========================================================================
        # Qwen LoRA Configuration
        # ========================================================================
        # Apply LoRA to Qwen if freeze_qwen_lora=False and lora_r > 0
        if not config.freeze_qwen_lora and config.lora_r > 0:
            logger.info(f"Applying LoRA to Qwen2.5-VL (r={config.lora_r}, alpha={config.lora_alpha})...")
            self.add_qwen_lora(
                lora_r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules
            )
        elif config.freeze_qwen_lora:
            logger.info("Qwen LoRA is frozen (freeze_qwen_lora=True)")
        else:
            logger.info("Qwen LoRA disabled (lora_r=0)")
        
    def add_qwen_lora(self, lora_r: int = 8, lora_alpha: int = 32, lora_dropout: float = 0.1, target_modules: list = None):
        """
        Add LoRA adapters to Qwen2.5-VL model
        
        Args:
            lora_r: LoRA rank
            lora_alpha: LoRA alpha scaling parameter
            lora_dropout: Dropout probability for LoRA layers
            target_modules: List of module names to apply LoRA to
        """
        if lora_r <= 0:
            logger.info("Qwen LoRA disabled (lora_r=0)")
            return
        
        # Check if Qwen base is frozen
        qwen_frozen = not any(p.requires_grad for p in self.qwen.parameters())
        if not qwen_frozen:
            logger.warning(
                "Qwen base model is not frozen! This will result in a very large number of trainable parameters. "
                "Consider setting freeze_qwen_base=True in config."
            )
        
        # Default target modules for Qwen
        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "k_proj"]  # Attention projections
            logger.info(f"Using default target modules for Qwen LoRA: {target_modules}")
        
        # Create LoRA configuration
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA to Qwen
        self.qwen = get_peft_model(self.qwen, lora_config)
        
        # Count LoRA parameters
        lora_params = sum(p.numel() for n, p in self.qwen.named_parameters() if "lora" in n.lower() and p.requires_grad)
        logger.info(f"Added {lora_params:,} LoRA parameters to Qwen2.5-VL")
        
        # Print trainable parameters summary
        self.qwen.print_trainable_parameters()
        
    def set_tokenizer(self, tokenizer: AutoTokenizer, seg_token: str = "<SEG>"):
        """
        Set tokenizer and configure SEG token
        
        Args:
            tokenizer: Tokenizer to use
            seg_token: Special token for segmentation
        """
        self.tokenizer = tokenizer
        
        # Add SEG token if not present
        if seg_token not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": [seg_token]})
            # Resize embeddings
            self.qwen.resize_token_embeddings(len(tokenizer))
            
        self.seg_token_id = tokenizer.convert_tokens_to_ids(seg_token)
        
        # Enable SEG token embedding training
        if not self.config.freeze_seg_token:
            # Since the entire embedding layer is frozen, we need to create a separate
            # learnable parameter for the SEG token and replace it dynamically
            embed_dim = self.qwen.get_input_embeddings().weight.shape[1]
            device = self.qwen.get_input_embeddings().weight.device
            dtype = self.qwen.get_input_embeddings().weight.dtype
            
            # Create a learnable SEG token embedding
            self.seg_token_embedding = nn.Parameter(
                torch.randn(embed_dim, device=device, dtype=dtype) * 0.02
            )
            
            # Replace the SEG token in the frozen embedding with our learnable version
            with torch.no_grad():
                self.qwen.get_input_embeddings().weight[self.seg_token_id] = self.seg_token_embedding.data
            
            logger.info(f"SEG token embedding training enabled for token ID {self.seg_token_id} (separate parameter: {embed_dim} dims)")
        else:
            self.seg_token_embedding = None
    
    def add_sam_lora(self, lora_r: int = 4, lora_alpha: int = 16, lora_dropout: float = 0.1):
        """
        Add LoRA adapters to SAM2.1 MaskDecoder attention layers
        
        IMPORTANT: LoRA should only be applied when MaskDecoder is frozen.
        If MaskDecoder is unfrozen, it will be trained directly without LoRA.
        
        Args:
            lora_r: LoRA rank (4-8 recommended)
            lora_alpha: LoRA alpha scaling factor
            lora_dropout: LoRA dropout rate
        """
        import torch.nn as nn
        
        # Check if MaskDecoder is frozen
        mask_decoder_frozen = not any(p.requires_grad for p in self.sam_mask_decoder.parameters())
        if not mask_decoder_frozen:
            logger.warning(
                "SAM MaskDecoder is not frozen! LoRA should only be applied to frozen layers. "
                "Consider setting freeze_sam_mask_decoder_base=True in config."
            )
            return
        
        if lora_r <= 0:
            logger.info("SAM LoRA disabled (lora_r=0)")
            return
        
        logger.info(f"Adding LoRA to SAM2.1 MaskDecoder (r={lora_r}, alpha={lora_alpha})")
        
        # Simple LoRA implementation for SAM MaskDecoder
        class LoRALinear(nn.Module):
            def __init__(self, original_layer, r=4, alpha=16, dropout=0.1):
                super().__init__()
                self.original_layer = original_layer
                self.r = r
                self.alpha = alpha
                
                # Freeze original weights
                for param in self.original_layer.parameters():
                    param.requires_grad = False
                
                # Add LoRA matrices
                self.lora_A = nn.Parameter(torch.randn(r, original_layer.in_features) * 0.01)
                self.lora_B = nn.Parameter(torch.zeros(original_layer.out_features, r))
                self.dropout = nn.Dropout(dropout)
                self.scaling = alpha / r
                
            def forward(self, x):
                # Original forward
                result = self.original_layer(x)
                # Add LoRA path
                lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
                return result + lora_out
        
        # Apply LoRA to MaskDecoder attention layers
        lora_applied = []
        
        # Check for transformer blocks in MaskDecoder
        if hasattr(self.sam_mask_decoder, 'transformer'):
            transformer = self.sam_mask_decoder.transformer
            
            # Apply to each transformer layer
            for i, layer in enumerate(transformer.layers):
                # Self attention
                if hasattr(layer, 'self_attn'):
                    for proj_name in ['q_proj', 'k_proj', 'v_proj']:
                        if hasattr(layer.self_attn, proj_name):
                            original = getattr(layer.self_attn, proj_name)
                            setattr(layer.self_attn, proj_name, LoRALinear(original, lora_r, lora_alpha, lora_dropout))
                            lora_applied.append(f"transformer.layers.{i}.self_attn.{proj_name}")
                
                # Cross attention (token to image)
                if hasattr(layer, 'cross_attn_token_to_image'):
                    for proj_name in ['q_proj', 'k_proj', 'v_proj']:
                        if hasattr(layer.cross_attn_token_to_image, proj_name):
                            original = getattr(layer.cross_attn_token_to_image, proj_name)
                            setattr(layer.cross_attn_token_to_image, proj_name, LoRALinear(original, lora_r, lora_alpha, lora_dropout))
                            lora_applied.append(f"transformer.layers.{i}.cross_attn_token_to_image.{proj_name}")
                
                # Cross attention (image to token)
                if hasattr(layer, 'cross_attn_image_to_token'):
                    for proj_name in ['q_proj', 'k_proj', 'v_proj']:
                        if hasattr(layer.cross_attn_image_to_token, proj_name):
                            original = getattr(layer.cross_attn_image_to_token, proj_name)
                            setattr(layer.cross_attn_image_to_token, proj_name, LoRALinear(original, lora_r, lora_alpha, lora_dropout))
                            lora_applied.append(f"transformer.layers.{i}.cross_attn_image_to_token.{proj_name}")
        
        # Count LoRA parameters
        lora_params = 0
        for name, param in self.sam_mask_decoder.named_parameters():
            if "lora" in name.lower() and param.requires_grad:
                lora_params += param.numel()
        
        logger.info(f"Applied LoRA to {len(lora_applied)} modules in SAM MaskDecoder")
        logger.info(f"Total SAM LoRA parameters: {lora_params:,}")
        
        # Store LoRA config for later reference
        self.sam_lora_config = {
            'r': lora_r,
            'alpha': lora_alpha,
            'dropout': lora_dropout,
            'applied_to': lora_applied
        }
    
    def extract_vision_features(self, pixel_values: torch.Tensor, image_grid_thw: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract vision features from Qwen's vision encoder
        
        Args:
            pixel_values: Qwen2.5-VL format - 2D tensor [total_patches, C*P*P*T]
            image_grid_thw: Grid dimensions [num_images, 3] where 3=[T,H,W] (BEFORE merge)
        
        Returns:
            Vision features [B, N_compressed_patches, D] after LLM projection
        """
        # image_grid_thw is REQUIRED for tokenized format
        if image_grid_thw is None:
            raise ValueError("image_grid_thw is required for dynamic resolution support")
        
        # ========================================================================
        # 正しい実装（O3調査結果に基づく）:
        # Qwen2.5-VLのpixel_valuesは[total_patches, C*P*P*T]の2D形式
        # visualエンコーダーは2×2の空間マージを行うため、出力は入力の1/4
        # ========================================================================
        
        logger.info(f"[Vision Features] pixel_values shape: {pixel_values.shape}, image_grid_thw shape: {image_grid_thw.shape if image_grid_thw is not None else 'None'}")
        
        # Determine actual batch size first
        actual_batch_size = None
        
        # 重要: pixel_valuesが誤った3D形式の場合を検出
        if pixel_values.dim() == 3:
            logger.error(f"[Vision Features] ERROR: pixel_values has wrong dimensions! Expected 2D [total_patches, feat_dim], got 3D {pixel_values.shape}")
            logger.error(f"[Vision Features] This suggests the collator is incorrectly stacking pixel_values instead of concatenating them.")
            # バッチ次元を除去して2Dに変換（一時的な対処）
            actual_batch_size = pixel_values.shape[0]
            if image_grid_thw is not None and actual_batch_size != image_grid_thw.shape[0]:
                raise ValueError(f"Batch size mismatch: pixel_values batch={actual_batch_size}, image_grid_thw batch={image_grid_thw.shape[0]}")
            
            # 各画像のpixel_valuesを縦に連結して正しい2D形式にする
            pixel_values_list = []
            for i in range(actual_batch_size):
                pixel_values_list.append(pixel_values[i])  # [patches_i, feat_dim]
            pixel_values = torch.cat(pixel_values_list, dim=0)  # [total_patches, feat_dim]
            logger.warning(f"[Vision Features] Converted to 2D format: {pixel_values.shape}")
        
        # image_grid_thwが3Dの場合は平坦化 (B, num_images, 3) -> (num_images_total, 3)
        if image_grid_thw is not None and image_grid_thw.dim() == 3:
            batch_size = image_grid_thw.shape[0]
            num_images_per_batch = image_grid_thw.shape[1]
            image_grid_thw_flat = image_grid_thw.reshape(-1, 3)
            logger.info(f"[Vision Features] Flattened image_grid_thw from ({batch_size}, {num_images_per_batch}, 3) to {image_grid_thw_flat.shape}")
        elif image_grid_thw is not None:
            batch_size = image_grid_thw.shape[0]
            image_grid_thw_flat = image_grid_thw
        else:
            # image_grid_thw is None - use batch size from pixel_values
            batch_size = actual_batch_size if actual_batch_size is not None else 1
            image_grid_thw_flat = None
        
        # PeftModelの場合、base_modelを通じてアクセス
        if hasattr(self.qwen, 'model'):
            base_model = self.qwen.model
        elif hasattr(self.qwen, 'base_model'):
            base_model = self.qwen.base_model.model
        else:
            base_model = self.qwen
        
        # spatial_merge_sizeを取得（通常は2）
        if hasattr(base_model, 'config') and hasattr(base_model.config, 'vision_config'):
            spatial_merge_size = getattr(base_model.config.vision_config, 'spatial_merge_size', 2)
        else:
            spatial_merge_size = 2  # デフォルト値
        logger.info(f"[Vision Features] spatial_merge_size = {spatial_merge_size}")
        
        # マージ前のパッチ数を計算（入力時）
        input_patch_counts = (image_grid_thw_flat[:, 0] * image_grid_thw_flat[:, 1] * image_grid_thw_flat[:, 2]).tolist()
        total_input_patches = sum(input_patch_counts)
        
        # マージ後のパッチ数を計算（出力時）
        output_patch_counts = []
        for t, h, w in image_grid_thw_flat.tolist():
            # 2×2の空間マージにより、H//2, W//2になる
            output_patches = int(t * (h // spatial_merge_size) * (w // spatial_merge_size))
            output_patch_counts.append(output_patches)
        total_output_patches = sum(output_patch_counts)
        
        logger.info(f"[Vision Features] Input patch counts per image: {input_patch_counts}, total: {total_input_patches}")
        logger.info(f"[Vision Features] Output patch counts per image (after {spatial_merge_size}x{spatial_merge_size} merge): {output_patch_counts}, total: {total_output_patches}")
        logger.info(f"[Vision Features] Actual pixel_values patches: {pixel_values.shape[0]}")
        
        if pixel_values.shape[0] != total_input_patches:
            logger.error(f"[Vision Features] Input patch count mismatch! Expected {total_input_patches}, got {pixel_values.shape[0]}")
            raise ValueError(f"Input patch count mismatch: expected {total_input_patches} patches based on grid_thw, but got {pixel_values.shape[0]} in pixel_values")
        
        # 0パッチの画像を検出
        zero_patch_indices = [i for i, count in enumerate(output_patch_counts) if count == 0]
        if zero_patch_indices:
            logger.error(f"[Vision Features] Found {len(zero_patch_indices)} images with 0 output patches at indices: {zero_patch_indices}")
            logger.error(f"[Vision Features] Grid values for 0-patch images: {[image_grid_thw_flat[i].tolist() for i in zero_patch_indices]}")
            raise ValueError(f"Cannot process images with 0 patches. Check data preprocessing. Indices: {zero_patch_indices}")
        
        # バッチ全体を一度に処理（推奨方法）
        if hasattr(base_model, 'visual'):
            # Qwen2.5-VLのvisualエンコーダーを直接呼び出す
            # pixel_valuesは全バッチのパッチが連結された2D形式
            # image_grid_thwは(num_images_total, 3)
            all_image_embeds = base_model.visual(
                hidden_states=pixel_values,
                grid_thw=image_grid_thw_flat
            )
        else:
            raise AttributeError(f"Cannot find visual encoder. Available attributes: {dir(base_model)[:20]}")
        
        # Handle tuple return
        if isinstance(all_image_embeds, tuple):
            all_image_embeds = all_image_embeds[0]
        
        logger.info(f"[Vision Features] Total embeddings shape: {all_image_embeds.shape}")
        
        # 出力パッチ数の検証
        if all_image_embeds.shape[0] != total_output_patches:
            logger.error(f"[Vision Features] Output mismatch! Expected {total_output_patches} patches after merge, got {all_image_embeds.shape[0]}")
            raise ValueError(f"Output patch count mismatch: expected {total_output_patches}, got {all_image_embeds.shape[0]}")
        
        # 各画像の特徴を分割（マージ後のパッチ数を使用）
        per_image_features = []
        offset = 0
        for i, num_patches in enumerate(output_patch_counts):
            image_features = all_image_embeds[offset:offset + num_patches]  # [num_patches, D]
            per_image_features.append(image_features)
            offset += num_patches
            logger.debug(f"[Vision Features] Image {i}: extracted {num_patches} patches")
        
        # バッチ形式に整形
        if batch_size == 1:
            # 単一画像の場合
            vision_features = per_image_features[0].unsqueeze(0)  # [1, N_patches, D]
        else:
            # 複数画像の場合
            # すべての画像が同じパッチ数を持つか確認
            if len(set(output_patch_counts)) == 1:
                # 同じパッチ数 - スタック可能
                vision_features = torch.stack(per_image_features, dim=0)  # [B, N_patches, D]
                logger.info(f"[Vision Features] Stacked {batch_size} images with {output_patch_counts[0]} patches each")
            else:
                # 異なるパッチ数 - パディングが必要
                logger.warning(f"[Vision Features] Variable output patch counts: {output_patch_counts}, applying padding")
                max_patches = max(output_patch_counts)
                
                padded_features = []
                for features in per_image_features:
                    if features.shape[0] < max_patches:
                        # ゼロパディング
                        padding = torch.zeros(max_patches - features.shape[0], features.shape[1],
                                            dtype=features.dtype, device=features.device)
                        features = torch.cat([features, padding], dim=0)
                    padded_features.append(features)
                
                vision_features = torch.stack(padded_features, dim=0)  # [B, N_patches, D]
        
        logger.info(f"[Vision Features] Final shape: {vision_features.shape}")
        return vision_features

    
    def extract_sam_features(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        [DEPRECATED] This method is no longer used.
        High-resolution features are now generated using HighResFeatureGenerator
        from Qwen vision features instead of SAM2.1's native extraction.
        
        This method is kept for compatibility but should not be called.
        """
        raise NotImplementedError(
            "extract_sam_features is deprecated. "
            "High-resolution features are now generated from Qwen vision features "
            "using the HighResFeatureGenerator class."
        )
    
    def compute_mask_centroid(self, mask: torch.Tensor, orig_h: int, orig_w: int) -> torch.Tensor:
        """
        バイナリマスクから重心座標を計算
        
        Args:
            mask: バイナリマスク [H, W] or [1, H, W]
            orig_h: 元画像の高さ（ピクセル座標系）
            orig_w: 元画像の幅（ピクセル座標系）
        
        Returns:
            重心座標 [2] = (cx, cy) in pixels
        """
        if mask.dim() == 3:
            mask = mask.squeeze(0)
        
        # Float型に変換
        m = mask.to(torch.float32)
        h, w = m.shape
        
        # 面積（非ゼロ画素数）
        mass = m.sum()
        if mass <= 0:
            # マスクが空の場合は画像中心を返す
            return torch.tensor([orig_w // 2, orig_h // 2], 
                              dtype=torch.float32, device=mask.device)
        
        # 座標グリッド
        ys = torch.arange(h, device=m.device, dtype=torch.float32).view(h, 1)
        xs = torch.arange(w, device=m.device, dtype=torch.float32).view(1, w)
        
        # 重心計算（マスク座標系）
        cy = (m * ys).sum() / mass
        cx = (m * xs).sum() / mass
        
        # マスク座標系から元画像座標系へ変換
        # マスクがリサイズされている場合のスケーリング
        scale_x = orig_w / w
        scale_y = orig_h / h
        cx = cx * scale_x
        cy = cy * scale_y
        
        return torch.stack([cx, cy])

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        mask_labels: Optional[List[torch.Tensor]] = None,
        sam_images: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[LISAModelOutput, Tuple]:
        """
        Forward pass with SAM2.1 integration for LISA改 model
        
        Args:
            input_ids: Text input token IDs
            pixel_values: Image inputs (2D patch or 3D image format)
            attention_mask: Attention mask for text
            labels: Target token IDs for training
            mask_labels: Ground truth masks for training
            sam_images: SAM-formatted images (1024x1024)
            image_grid_thw: Grid dimensions for dynamic resolution
            return_dict: Whether to return LISAModelOutput or tuple
        """
        B = input_ids.shape[0]
        
        # Log batch composition for debugging
        logger.debug(f"[FORWARD] Batch composition:")
        logger.debug(f"  input_ids: {input_ids.shape}")
        logger.debug(f"  pixel_values: {pixel_values.shape if pixel_values is not None else 'None'}")
        logger.debug(f"  sam_images: {sam_images.shape if sam_images is not None else 'None'}")
        logger.debug(f"  mask_labels: {len(mask_labels) if mask_labels is not None else 'None'}")
        logger.debug(f"  image_grid_thw: {image_grid_thw.shape if image_grid_thw is not None else 'None'}")
        
        # Determine if this is a segmentation task or VQA task
        # Segmentation tasks have SEG tokens in input_ids or labels
        is_segmentation_task = False
        if self.seg_token_id is not None:
            if labels is not None:
                # Training: check labels for SEG tokens
                is_segmentation_task = any((labels[i] == self.seg_token_id).any() for i in range(B))
            else:
                # Inference: check input_ids for SEG tokens
                is_segmentation_task = any((input_ids[i] == self.seg_token_id).any() for i in range(B))
        
        logger.debug(f"[FORWARD] Task type: {'Segmentation' if is_segmentation_task else 'VQA'}")
        
        # 1. Extract features from both vision encoders (always needed for LISA改)
        # 1.1 Extract features from Qwen vision encoder
        vision_features = self.extract_vision_features(pixel_values, image_grid_thw)
        
        # 1.2 Extract features from SAM vision encoder 
        sam_image_features = None  # Features from SAM ImageEncoder
        if sam_images is not None:
            logger.debug(f"[FORWARD] Processing SAM images: {sam_images.shape}")
            with torch.no_grad():
                # Ensure SAM encoder is in eval mode for stability
                self.sam_image_encoder.eval()
                
                # Convert to appropriate dtype
                sam_encoder_dtype = next(self.sam_image_encoder.parameters()).dtype
                sam_images_converted = sam_images.to(dtype=sam_encoder_dtype)
                
                # Extract features using SAM encoder
                backbone_out = self.sam_image_encoder(sam_images_converted)
                
                # Handle different output formats
                if isinstance(backbone_out, dict):
                    sam_image_features = backbone_out.get('vision_features', backbone_out.get('image_embeddings'))
                else:
                    sam_image_features = backbone_out
                
                logger.debug(f"[FORWARD] SAM ImageEncoder features extracted: {sam_image_features.shape}")
        
        # Check if segmentation task requires SAM features
        if is_segmentation_task and sam_image_features is None:
            logger.warning("⚠️ SAM images are None for segmentation task! This may impact mask quality.")
            logger.warning("SAM ImageEncoder cannot function without SAM images.")
        
        # 2. Convert Qwen vision features to SAM-compatible format (256 channels)
        image_features_sam = None
        if vision_features is not None:
            # Apply image adapter to get 256-channel features
            image_features_sam = self.image_adapter(vision_features, image_grid_thw)  # [B, 256, H, W]
            logger.debug(f"[FORWARD] Qwen features after image_adapter: {image_features_sam.shape}")
        
        # 3. Fuse SAM ImageEncoder features with Qwen features
        if sam_image_features is not None and image_features_sam is not None:
            if self.fusion_type == "sigma_add":
                # Apply sigma-add fusion as per sam_vit_sigma_add_fusion20250814.md
                # Both features should be at 256 channels
                # sam_image_features: [B, 256, 64, 64] from SAM encoder
                # image_features_sam: [B, 256, H, W] from Qwen after image_adapter
                
                B_sam, C_sam, H_sam, W_sam = sam_image_features.shape
                B_qwen, C_qwen, H_qwen, W_qwen = image_features_sam.shape
                
                logger.debug(f"[FUSION] SAM encoder features: {sam_image_features.shape}, Qwen adapter features: {image_features_sam.shape}")
                
                # Align spatial dimensions (resize Qwen features to match SAM)
                if (H_qwen, W_qwen) != (H_sam, W_sam):
                    image_features_sam_resized = F.interpolate(
                        image_features_sam,
                        size=(H_sam, W_sam),
                        mode='bilinear',
                        align_corners=False
                    )
                    logger.debug(f"[FUSION] Resized Qwen features from ({H_qwen}, {W_qwen}) to ({H_sam}, {W_sam})")
                else:
                    image_features_sam_resized = image_features_sam
                
                # Apply sigma-add fusion: sam_features + sigmoid(beta) * qwen_features
                beta_scaled = torch.sigmoid(self.image_fusion_beta)
                image_features_sam = sam_image_features + beta_scaled * image_features_sam_resized
                
                logger.info(f"[FUSION] Applied sigma-add fusion with beta={beta_scaled.item():.3f}")
            
            elif self.fusion_type == "cross_attention":
                # Use the CrossAttentionFusion module
                # Align spatial dimensions first
                B_sam, C_sam, H_sam, W_sam = sam_image_features.shape
                B_qwen, C_qwen, H_qwen, W_qwen = image_features_sam.shape
                
                if (H_qwen, W_qwen) != (H_sam, W_sam):
                    image_features_sam = F.interpolate(
                        image_features_sam,
                        size=(H_sam, W_sam),
                        mode='bilinear',
                        align_corners=False
                    )
                
                # Apply cross-attention fusion via HybridFusionModule
                image_features_sam = self.image_fusion(sam_image_features, image_features_sam)
                logger.info(f"[FUSION] Applied cross-attention fusion")
        elif sam_image_features is not None:
            # Only SAM features available
            image_features_sam = sam_image_features
            logger.debug("[FUSION] Using only SAM encoder features (no Qwen features)")
        # else: image_features_sam stays as Qwen-only features
        
        # 4. Language modeling through Qwen
        # Pass through Qwen model with proper inputs based on format
        if pixel_values is not None and pixel_values.dim() == 2:
            # 2D format with image_grid_thw (apply_chat_template format)
            outputs = self.qwen(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
                output_hidden_states=True
            )
        elif pixel_values is not None and pixel_values.dim() == 3:
            # 3D format (batch of images) - still need image_grid_thw for Qwen2.5-VL
            outputs = self.qwen(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,  # Pass grid info for 3D format too
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
                output_hidden_states=True
            )
        else:
            # No pixel_values or other format
            outputs = self.qwen(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
                output_hidden_states=True
            )
        
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1] if outputs.hidden_states else None
        
        # 5. Generate high-resolution features for SAM2.1 mask decoder
        sam_high_res_features = None
        if image_features_sam is not None and is_segmentation_task:
            if self.use_token_fpn:
                # Use Token-FPN for multi-scale features
                image_features_sam, sam_high_res_features = self.token_fpn(
                    image_features_sam, image_grid_thw, use_hooks=True
                )
                logger.debug(f"[FORWARD] Features after Token-FPN: {image_features_sam.shape}")
            else:
                # Use HighResFeatureGenerator
                sam_high_res_features = self.high_res_generator(image_features_sam)
                logger.debug(f"[FORWARD] Generated high-res features using HighResFeatureGenerator")
        
        
        # 7. Find SEG token positions and generate masks (only for segmentation tasks)
        mask_logits = []
        seg_positions = []
        
        if is_segmentation_task:
            # Check if seg_token_id is set
            if self.seg_token_id is None:
                logger.error("[FORWARD] seg_token_id is None for segmentation task!")
                raise ValueError("seg_token_id must be set for segmentation tasks")
            
            # Verify SAM features are available
            if image_features_sam is None:
                logger.error("[FORWARD] SAM features not available for segmentation task!")
                raise ValueError("SAM vision features are required for segmentation tasks")
            
            # Determine SEG positions based on labels (training) or generated tokens (inference)
            if labels is not None:
                # Training: find SEG tokens in labels
                for i in range(B):
                    seg_pos = (labels[i] == self.seg_token_id).nonzero(as_tuple=True)[0]
                    if len(seg_pos) > 0:
                        seg_positions.append(seg_pos.tolist())
                        
                        # Training with segmentation: warn if mask_labels missing
                        if mask_labels is None or i >= len(mask_labels) or mask_labels[i] is None:
                            logger.warning(f"[FORWARD] Training segmentation task but mask_labels missing for batch {i}")
                    else:
                        seg_positions.append([])
            else:
                # Inference: find SEG tokens in input_ids
                for i in range(B):
                    seg_pos = (input_ids[i] == self.seg_token_id).nonzero(as_tuple=True)[0]
                    logger.debug(f"[SEG SEARCH] Batch {i}: Found {len(seg_pos)} SEG tokens at positions {seg_pos.tolist() if len(seg_pos) > 0 else []}")
                    if len(seg_pos) > 0:
                        seg_positions.append(seg_pos.tolist())
                    else:
                        seg_positions.append([])
            
            # 4. Generate masks for each SEG position using proper SAM2.1 implementation
            logger.debug(f"[MASK GEN] Total seg_positions: {seg_positions}")
            logger.debug(f"[MASK GEN] Total batches: {B}")
            for i in range(B):
                sample_masks = []
                
                # Debug: Log seg_positions for this batch
                logger.debug(f"[MASK GEN] Batch {i}: seg_positions = {seg_positions[i]}")
                logger.debug(f"[MASK GEN] image_features_sam shape: {image_features_sam.shape if image_features_sam is not None else 'None'}")
                
                for j, seg_idx in enumerate(seg_positions[i]):
                    logger.debug(f"[MASK GEN] Processing batch {i}, SEG token {j} at position {seg_idx}")
                    try:
                        # Extract text embedding at SEG position
                        text_embed = hidden_states[i, seg_idx]  # [hidden_dim]
                        
                        # Project to SAM prompt space
                        # Project from Qwen hidden_dim (3584) to SAM prompt_embed_dim (256)
                        prompt_embed = self.text_prompt_proj(text_embed)  # [256]
                        
                        # Get position encoding for SAM
                        if not hasattr(self, '_pe_cache'):
                            self._pe_cache = {}
                        
                        pe_key = image_features_sam[i:i+1].shape[-2:]
                        if pe_key not in self._pe_cache:
                            self._pe_cache[pe_key] = self.sam_prompt_encoder.get_dense_pe().to(image_features_sam.device)
                        image_pe = self._pe_cache[pe_key]
                        
                        # Handle different image_pe shapes
                        if image_pe.dim() == 3:  # Missing batch dimension
                            image_pe = image_pe.unsqueeze(0)
                        elif image_pe.shape[0] != 1:
                            # Ensure batch dimension is 1
                            image_pe = image_pe[:1]
                        
                        # Check spatial dimensions
                        if image_pe.shape[-2:] != image_features_sam[i:i+1].shape[-2:]:
                            # Interpolate position encoding to match image features
                            image_pe = F.interpolate(
                                image_pe,
                                size=image_features_sam[i:i+1].shape[-2:],
                                mode='bilinear',
                                align_corners=False
                            )
                        
                        # 真値マスクから重心座標を計算（訓練時）、または画像中心を使用（推論時）
                        h_feat, w_feat = image_features_sam[i:i+1].shape[-2:]
                        # Map to original image coordinates (feature stride is 16)
                        h_img, w_img = h_feat * 16, w_feat * 16
                        
                        if mask_labels is not None and i < len(mask_labels) and mask_labels[i] is not None:
                            # 訓練時: GTマスクから重心を計算
                            gt_mask = mask_labels[i]
                            
                            # GTマスクが複数SEGに対応している場合、最初のマスクを使用
                            if isinstance(gt_mask, list):
                                if len(gt_mask) > j:
                                    gt_mask = gt_mask[j]
                                else:
                                    gt_mask = gt_mask[0] if len(gt_mask) > 0 else None
                            
                            if gt_mask is not None:
                                # 重心を計算
                                centroid = self.compute_mask_centroid(gt_mask, h_img, w_img)
                                center_x, center_y = centroid[0].item(), centroid[1].item()
                                logger.debug(f"[CENTROID] Batch {i}, SEG {j}: Computed centroid from GT mask - x={center_x:.1f}, y={center_y:.1f} (image center would be {w_img//2}, {h_img//2})")
                            else:
                                # GTマスクがない場合は画像中心を使用
                                center_x, center_y = w_img // 2, h_img // 2
                                logger.debug(f"[CENTROID] Batch {i}, SEG {j}: No GT mask - using image center ({center_x}, {center_y})")
                        else:
                            # 推論時: 画像中心を使用（フォールバック）
                            center_x, center_y = w_img // 2, h_img // 2
                            logger.debug(f"[CENTROID] Batch {i}, SEG {j}: Inference mode - using image center ({center_x}, {center_y})")
                        
                        # Create point coordinates [N, 2] where N=1 for center point
                        point_coords = torch.tensor([[center_x, center_y]], 
                                                  dtype=torch.float32, 
                                                  device=prompt_embed.device).unsqueeze(0)  # [1, 1, 2]
                        point_labels = torch.tensor([1], dtype=torch.int32, 
                                                  device=prompt_embed.device).unsqueeze(0)  # [1, 1] (positive point)
                        
                        # Use SAM's PromptEncoder to generate proper embeddings
                        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
                            points=(point_coords, point_labels),
                            boxes=None,
                            masks=None,
                        )
                        
                        # 加算アプローチによる埋め込み融合
                        # sparse_embeddings: [1, N, C] where C=256 for SAM2.1
                        if sparse_embeddings.shape[1] > 0:
                            # 元の位置埋め込みを保持
                            e_pos = sparse_embeddings[:, 0, :]  # [1, 256] SAMの位置埋め込み
                            
                            # βをsigmoidで0〜1に制限
                            beta_scaled = torch.sigmoid(self.prompt_beta)
                            
                            # 加算による融合（位置情報を100%保持しつつLLM情報を追加）
                            e_add = e_pos + beta_scaled * prompt_embed.unsqueeze(0)
                            
                            # 融合結果で埋め込みを更新
                            sparse_embeddings[:, 0, :] = e_add
                        
                        # Ensure dense_embeddings matches image_features spatial size
                        if dense_embeddings.shape[-2:] != image_features_sam[i:i+1].shape[-2:]:
                            h_feat, w_feat = image_features_sam[i:i+1].shape[-2:]
                            dense_embeddings = F.interpolate(
                                dense_embeddings,
                                size=(h_feat, w_feat),
                                mode='bilinear',
                                align_corners=False
                            )
                        
                        # Get SAM model dtype for consistency
                        sam_dtype = next(self.sam_mask_decoder.parameters()).dtype
                        
                        # Use generated high-res features from Qwen vision encoder
                        if sam_high_res_features is not None and len(sam_high_res_features) >= 2:
                            # Apply SAM2.1's conv_s0/conv_s1 to compress channels before passing to MaskDecoder
                            # This is required because MaskDecoder doesn't apply these automatically
                            feat_s0 = sam_high_res_features[0][i:i+1].to(dtype=sam_dtype)  # stride-4
                            feat_s1 = sam_high_res_features[1][i:i+1].to(dtype=sam_dtype)  # stride-8
                            
                            # Apply channel compression: 256 -> 32 for s0, 256 -> 64 for s1
                            if hasattr(self.sam_mask_decoder, 'conv_s0') and hasattr(self.sam_mask_decoder, 'conv_s1'):
                                feat_s0 = self.sam_mask_decoder.conv_s0(feat_s0)  # 256 -> 32 channels
                                feat_s1 = self.sam_mask_decoder.conv_s1(feat_s1)  # 256 -> 64 channels
                            else:
                                # This should not happen with SAM2.1 initialized with use_high_res_features=True
                                # According to spec: "異常時には明示的に例外を投げるよう修正します（ダミー入力でエラーを隠蔽しないようにする）"
                                raise RuntimeError(
                                    "SAM2.1 MaskDecoder missing conv_s0/conv_s1. "
                                    "Ensure SAM2.1 is initialized with use_high_res_features=True"
                                )
                            
                            high_res_features = [feat_s0, feat_s1]
                        else:
                            # This should not happen since we always generate high-res features
                            raise RuntimeError("High-resolution features not available. This should not happen.")
                        
                        # Use repeat_image=True to handle batch size mismatch
                        # Re-enable high-res features with proper 256-channel format
                        low_res_masks, iou_predictions, sam_tokens_out, obj_score_logits = self.sam_mask_decoder(
                            image_embeddings=image_features_sam[i:i+1].to(dtype=sam_dtype),
                            image_pe=image_pe.to(dtype=sam_dtype),
                            sparse_prompt_embeddings=sparse_embeddings.to(dtype=sam_dtype),
                            dense_prompt_embeddings=dense_embeddings.to(dtype=sam_dtype),
                            multimask_output=False,  # Single mask per prompt
                            repeat_image=True,  # Allow SAM to handle batch size mismatch
                            high_res_features=high_res_features,  # Pass 256-channel features - SAM will apply conv_s0/s1
                        )
                        
                        # Upscale mask to original image size
                        # For dynamic resolution, determine target size from grid_thw
                        if image_grid_thw is not None and i < image_grid_thw.shape[0]:
                            # Grid dimensions are in RAW patches (before PatchMerge)
                            H_grid_raw = int(image_grid_thw[i, 1].item())
                            W_grid_raw = int(image_grid_thw[i, 2].item())
                            # Convert to pixel dimensions (14px per patch)
                            orig_h = H_grid_raw * 14
                            orig_w = W_grid_raw * 14
                        elif pixel_values is not None and pixel_values.dim() == 4:
                            # Fallback to pixel_values dimensions if available
                            orig_h, orig_w = pixel_values.shape[-2:]
                        else:
                            # Use feature map size * stride as fallback
                            orig_h = h_feat * 16
                            orig_w = w_feat * 16
                        
                        mask_logit = F.interpolate(
                            low_res_masks,
                            size=(orig_h, orig_w),
                            mode='bilinear',
                            align_corners=False
                        )
                        sample_masks.append(mask_logit.squeeze(0))  # Remove batch dim
                            
                    except Exception as e:
                        logger.error(f"SAM2.1 mask generation failed with error: {e}")
                        import traceback
                        traceback.print_exc()
                        print(f"Warning: SAM2.1 mask generation failed with error: {e}")
                        import traceback
                        print(f"Traceback: {traceback.format_exc()}")
                        # Fall back to dummy mask if SAM fails
                        if pixel_values is not None:
                            orig_h, orig_w = pixel_values.shape[-2:]
                            dummy_mask = torch.zeros(1, orig_h, orig_w, device=pixel_values.device, dtype=torch.float32)
                            sample_masks.append(dummy_mask)
                
                mask_logits.append(sample_masks if len(sample_masks) > 0 else None)
        else:
            # VQA task: No mask generation needed
            logger.debug("[FORWARD] VQA task - skipping mask generation")
            mask_logits = None
            seg_positions = None
        
        if return_dict:
            return LISAModelOutput(
                logits=logits,
                mask_logits=mask_logits,
                language_hidden_states=hidden_states,
                vision_hidden_states=vision_features,
                seg_token_positions=seg_positions
            )
        else:
            return logits, mask_logits
    
    def generate_with_masks(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        **kwargs
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Generate text with automatic mask generation at SEG tokens
        
        Args:
            input_ids: Input token IDs [B, seq_len]
            pixel_values: Input images [B, C, H, W]
            attention_mask: Attention mask [B, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            top_p: Top-p sampling parameter
            **kwargs: Additional generation arguments
        
        Returns:
            Dictionary with:
                - generated_ids: Full generated token sequence
                - generated_text: List of decoded text
                - masks: List of generated masks (None if no SEG token)
                - seg_positions: Positions where SEG tokens were generated
        """
        B = input_ids.size(0)
        device = input_ids.device
        
        # Initialize outputs
        generated_ids = input_ids.clone()
        all_masks = [[] for _ in range(B)]
        all_seg_positions = [[] for _ in range(B)]
        
        # Get vision features once (they don't change during generation)
        # For generation, we need to infer image_grid_thw from pixel_values
        B = pixel_values.shape[0]
        image_grid_thw = kwargs.get('image_grid_thw', torch.tensor([[1, 24, 24]], device=pixel_values.device).repeat(B, 1))
        vision_features = self.extract_vision_features(pixel_values, image_grid_thw)
        image_features_sam = self.image_adapter(vision_features, image_grid_thw)
        
        # Generate high-res features from Qwen vision features
        sam_high_res_features = self.high_res_generator(image_features_sam)
        
        # Generation loop
        for step in range(max_new_tokens):
            # Get model outputs
            with torch.no_grad():
                outputs = self.qwen(
                    input_ids=generated_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                    return_dict=True
                )
            
            # Get next token logits
            next_token_logits = outputs.logits[:, -1, :]  # [B, vocab_size]
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Sample or take argmax
            if do_sample:
                # Apply top-p sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Keep at least one token
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    
                    # Set logits to -inf for removed tokens
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # Append generated tokens
            generated_ids = torch.cat([
                generated_ids,
                next_tokens.unsqueeze(1)
            ], dim=1)
            
            # Update attention mask if provided
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((B, 1), device=device, dtype=attention_mask.dtype)
                ], dim=1)
            
            # Check for SEG tokens
            seg_mask = (next_tokens == self.seg_token_id)
            
            if seg_mask.any():
                # Get hidden states at current position
                current_hidden = outputs.hidden_states[-1][:, -1, :]  # [B, D_l]
                
                # Generate masks for samples that output SEG
                for b in range(B):
                    if seg_mask[b]:
                        # Record SEG position
                        seg_pos = generated_ids.size(1) - 1
                        all_seg_positions[b].append(seg_pos)
                        
                        # Generate mask using proper SAM2.1 implementation
                        seg_hidden = current_hidden[b]  # [D_l]
                        prompt_embed = self.text_prompt_proj(seg_hidden)  # [256]
                        
                        try:
                            # Get positional encoding from SAM prompt encoder
                            image_pe = self.sam_prompt_encoder.get_dense_pe()
                            
                            # Ensure PE matches image embeddings size
                            if image_pe.shape[-2:] != image_features_sam[b:b+1].shape[-2:]:
                                h_feat, w_feat = image_features_sam[b:b+1].shape[-2:]
                                image_pe = F.interpolate(
                                    image_pe,
                                    size=(h_feat, w_feat),
                                    mode='bilinear',
                                    align_corners=False
                                )
                            
                            # Create a center point as anchor for text-guided segmentation
                            h_img, w_img = pixel_values.shape[-2:]
                            center_x, center_y = w_img // 2, h_img // 2
                            
                            # Create point coordinates [N, 2] where N=1 for center point
                            point_coords = torch.tensor([[center_x, center_y]], 
                                                      dtype=torch.float32, 
                                                      device=device).unsqueeze(0)  # [1, 1, 2]
                            point_labels = torch.tensor([1], dtype=torch.int32, 
                                                      device=device).unsqueeze(0)  # [1, 1] (positive point)
                            
                            # Use SAM's PromptEncoder to generate proper embeddings
                            sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
                                points=(point_coords, point_labels),
                                boxes=None,
                                masks=None,
                            )
                            
                            # Replace the point embedding with our text-derived embedding
                            # sparse_embeddings: [1, N, C] where C=256 for SAM2.1
                            if sparse_embeddings.shape[1] > 0:
                                sparse_embeddings[:, 0, :] = prompt_embed.unsqueeze(0)
                            
                            # Ensure dense_embeddings matches image_features spatial size
                            if dense_embeddings.shape[-2:] != image_features_sam[b:b+1].shape[-2:]:
                                h_feat, w_feat = image_features_sam[b:b+1].shape[-2:]
                                dense_embeddings = F.interpolate(
                                    dense_embeddings,
                                    size=(h_feat, w_feat),
                                    mode='bilinear',
                                    align_corners=False
                                )
                            
                            # Get SAM model dtype for consistency
                            sam_dtype = next(self.sam_mask_decoder.parameters()).dtype
                            
                            # Run mask decoder
                            with torch.no_grad():
                                # Use generated high-res features from Qwen vision encoder
                                if sam_high_res_features is not None and len(sam_high_res_features) >= 2:
                                    # Apply SAM2.1's conv_s0/conv_s1 to compress channels
                                    feat_s0 = sam_high_res_features[0][b:b+1].to(dtype=sam_dtype)  # stride-4
                                    feat_s1 = sam_high_res_features[1][b:b+1].to(dtype=sam_dtype)  # stride-8
                                    
                                    # Apply channel compression: 256 -> 32 for s0, 256 -> 64 for s1
                                    if hasattr(self.sam_mask_decoder, 'conv_s0') and hasattr(self.sam_mask_decoder, 'conv_s1'):
                                        feat_s0 = self.sam_mask_decoder.conv_s0(feat_s0)  # 256 -> 32 channels
                                        feat_s1 = self.sam_mask_decoder.conv_s1(feat_s1)  # 256 -> 64 channels
                                    else:
                                        # This should not happen with SAM2.1
                                        raise RuntimeError("SAM2.1 MaskDecoder missing conv_s0/conv_s1")
                                    
                                    high_res_features = [feat_s0, feat_s1]
                                else:
                                    # This should not happen since we always generate high-res features
                                    raise RuntimeError("High-resolution features not available during generation.")
                                
                                low_res_masks, _, _, _ = self.sam_mask_decoder(
                                    image_embeddings=image_features_sam[b:b+1].to(dtype=sam_dtype),
                                    image_pe=image_pe.to(dtype=sam_dtype),
                                    sparse_prompt_embeddings=sparse_embeddings.to(dtype=sam_dtype),
                                    dense_prompt_embeddings=dense_embeddings.to(dtype=sam_dtype),
                                    multimask_output=False,  # Single mask per prompt
                                    repeat_image=False,  # No need to repeat for single image
                                    high_res_features=high_res_features,
                                )
                            
                            # Upscale mask
                            orig_h, orig_w = pixel_values.shape[-2:]
                            mask = F.interpolate(
                                low_res_masks,
                                size=(orig_h, orig_w),
                                mode='bilinear',
                                align_corners=False
                            ).squeeze(0).squeeze(0)  # Remove batch and channel dims
                            
                            all_masks[b].append(mask)
                            
                        except Exception as e:
                            print(f"Warning: SAM2.1 mask generation failed during inference: {e}")
                            import traceback
                            print(f"Traceback: {traceback.format_exc()}")
                            # Fall back to dummy mask if SAM fails
                            orig_h, orig_w = pixel_values.shape[-2:]
                            dummy_mask = torch.zeros(orig_h, orig_w, device=device, dtype=torch.float32)
                            all_masks[b].append(dummy_mask)
            
            # Check for EOS tokens
            eos_mask = (next_tokens == self.tokenizer.eos_token_id)
            if eos_mask.all():
                break
        
        # Decode generated text
        generated_text = []
        for b in range(B):
            # Get only the newly generated tokens
            new_tokens = generated_ids[b, input_ids.size(1):]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
            generated_text.append(text)
        
        return {
            'generated_ids': generated_ids,
            'generated_text': generated_text,
            'masks': all_masks,
            'seg_positions': all_seg_positions
        }
    
    def save_pretrained(self, save_directory: str):
        """Save all model components and trainable parameters"""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        logger.info(f"Saving model to {save_directory}")
        
        # 1. Save config
        torch.save(self.config, os.path.join(save_directory, "config.pt"))
        
        # 2. Save all adapter components (always trainable in our setup)
        # Image Adapter
        torch.save(self.image_adapter.state_dict(), 
                  os.path.join(save_directory, "image_adapter.pt"))
        logger.info(f"Saved image_adapter: {sum(p.numel() for p in self.image_adapter.parameters())} params")
        
        # Text Prompt Projector
        torch.save(self.text_prompt_proj.state_dict(), 
                  os.path.join(save_directory, "text_prompt_proj.pt"))
        logger.info(f"Saved text_prompt_proj: {sum(p.numel() for p in self.text_prompt_proj.parameters())} params")
        
        # Token-FPN or High-res Generator
        if self.use_token_fpn:
            torch.save(self.token_fpn.state_dict(),
                      os.path.join(save_directory, "token_fpn.pt"))
            logger.info(f"Saved token_fpn: {sum(p.numel() for p in self.token_fpn.parameters())} params")
        else:
            torch.save(self.high_res_generator.state_dict(),
                      os.path.join(save_directory, "high_res_generator.pt"))
            logger.info(f"Saved high_res_generator: {sum(p.numel() for p in self.high_res_generator.parameters())} params")
        
        # 3. Save trainable special parameters
        # Prompt Beta (residual addition scaling)
        torch.save({'prompt_beta': self.prompt_beta.data}, 
                  os.path.join(save_directory, "prompt_beta.pt"))
        logger.info(f"Saved prompt_beta: {self.prompt_beta.data.item()}")
        
        # Image Fusion parameters (SAM-Qwen feature fusion)
        if self.fusion_type == "sigma_add" and hasattr(self, 'image_fusion_beta'):
            torch.save({'image_fusion_beta': self.image_fusion_beta.data}, 
                      os.path.join(save_directory, "image_fusion_beta.pt"))
            logger.info(f"Saved image_fusion_beta: {self.image_fusion_beta.data.item()}")
        elif self.fusion_type == "cross_attention":
            torch.save(self.image_fusion.state_dict(),
                      os.path.join(save_directory, "image_fusion_cross_attention.pt"))
            logger.info("Saved Cross-Attention fusion module")
        
        # SEG Token Embedding (if trainable)
        if hasattr(self, 'seg_token_embedding') and self.seg_token_embedding is not None:
            torch.save({
                'seg_token_embedding': self.seg_token_embedding.data,
                'seg_token_id': self.seg_token_id
            }, os.path.join(save_directory, "seg_token_embedding.pt"))
            logger.info(f"Saved SEG token embedding for ID {self.seg_token_id}")
        
        # 4. Save SAM components
        # SAM LoRA weights (if enabled)
        if self.config.sam_lora_r > 0 and self.config.freeze_sam_mask_decoder_base:
            sam_lora_state = {}
            for name, param in self.sam_mask_decoder.named_parameters():
                if 'lora_' in name and param.requires_grad:
                    sam_lora_state[name] = param.data
            if sam_lora_state:
                torch.save(sam_lora_state, 
                          os.path.join(save_directory, "sam_lora.pt"))
                logger.info(f"Saved SAM LoRA: {len(sam_lora_state)} modules, {sum(p.numel() for p in sam_lora_state.values())} params")
        
        # If SAM MaskDecoder is not frozen (trained directly)
        elif not self.config.freeze_sam_mask_decoder_base:
            torch.save(self.sam_mask_decoder.state_dict(),
                      os.path.join(save_directory, "sam_mask_decoder.pt"))
            logger.info(f"Saved full SAM MaskDecoder: {sum(p.numel() for p in self.sam_mask_decoder.parameters())} params")
        
        # 5. Save Qwen components
        # Qwen LoRA (if enabled)
        if not self.config.freeze_qwen_lora and self.config.lora_r > 0:
            # Save using PEFT's save_pretrained method
            os.makedirs(os.path.join(save_directory, "qwen"), exist_ok=True)
            try:
                # For PEFT models
                self.qwen.save_pretrained(os.path.join(save_directory, "qwen"))
                logger.info(f"Saved Qwen LoRA adapter")
            except Exception as e:
                # Fallback: manually save LoRA weights
                logger.warning(f"PEFT save failed: {e}, trying manual save")
                qwen_lora_state = {}
                for name, param in self.qwen.named_parameters():
                    if 'lora_' in name and param.requires_grad:
                        qwen_lora_state[name] = param.data
                if qwen_lora_state:
                    torch.save(qwen_lora_state,
                              os.path.join(save_directory, "qwen", "lora_weights.pt"))
                    logger.info(f"Saved Qwen LoRA manually: {len(qwen_lora_state)} modules")
        
        # If Qwen base is not frozen (unlikely but possible)
        elif not self.config.freeze_qwen_base:
            self.qwen.save_pretrained(os.path.join(save_directory, "qwen"))
            logger.info(f"Saved full Qwen model")
        
        # 6. Create a metadata file for easy loading
        metadata = {
            'model_type': 'LISA_Model',
            'config_class': 'LISAConfig',
            'use_token_fpn': self.use_token_fpn,
            'has_seg_token_embedding': hasattr(self, 'seg_token_embedding') and self.seg_token_embedding is not None,
            'has_sam_lora': self.config.sam_lora_r > 0 and self.config.freeze_sam_mask_decoder_base,
            'has_qwen_lora': not self.config.freeze_qwen_lora and self.config.lora_r > 0,
            'seg_token_id': self.seg_token_id if hasattr(self, 'seg_token_id') else None,
        }
        torch.save(metadata, os.path.join(save_directory, "metadata.pt"))
        
        logger.info(f"Model saved successfully to {save_directory}")

    @classmethod
    def load_pretrained(cls, load_directory: str, device='cuda', **kwargs):
        """Load model from saved checkpoint
        
        Args:
            load_directory: Directory containing saved model
            device: Device to load model on
            **kwargs: Additional arguments to override config
        
        Returns:
            Loaded LISA_Model instance
        """
        import os
        from pathlib import Path
        
        load_dir = Path(load_directory)
        logger.info(f"Loading model from {load_dir}")
        
        # 1. Load config
        config_path = load_dir / "config.pt"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        config = torch.load(config_path, map_location=device, weights_only=False)
        
        # Override config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # 2. Load metadata if exists
        metadata_path = load_dir / "metadata.pt"
        metadata = {}
        if metadata_path.exists():
            metadata = torch.load(metadata_path, map_location=device, weights_only=False)
            logger.info(f"Loaded metadata: {metadata}")
        
        # 3. Initialize model
        model = cls(config=config)
        
        # 4. Load adapter components
        # Image Adapter
        image_adapter_path = load_dir / "image_adapter.pt"
        if image_adapter_path.exists():
            state_dict = torch.load(image_adapter_path, map_location=device, weights_only=True)
            model.image_adapter.load_state_dict(state_dict)
            logger.info(f"Loaded image_adapter")
        
        # Text Prompt Projector
        text_prompt_proj_path = load_dir / "text_prompt_proj.pt"
        if text_prompt_proj_path.exists():
            state_dict = torch.load(text_prompt_proj_path, map_location=device, weights_only=True)
            model.text_prompt_proj.load_state_dict(state_dict)
            logger.info(f"Loaded text_prompt_proj")
        
        # Token-FPN or High-res Generator
        token_fpn_path = load_dir / "token_fpn.pt"
        high_res_gen_path = load_dir / "high_res_generator.pt"
        
        if token_fpn_path.exists():
            state_dict = torch.load(token_fpn_path, map_location=device, weights_only=True)
            model.token_fpn.load_state_dict(state_dict)
            logger.info(f"Loaded token_fpn")
        elif high_res_gen_path.exists():
            state_dict = torch.load(high_res_gen_path, map_location=device, weights_only=True)
            model.high_res_generator.load_state_dict(state_dict)
            logger.info(f"Loaded high_res_generator")
        
        # 5. Load special parameters
        # Prompt Beta
        prompt_beta_path = load_dir / "prompt_beta.pt"
        if prompt_beta_path.exists():
            state = torch.load(prompt_beta_path, map_location=device, weights_only=False)
            if 'prompt_beta' in state:
                model.prompt_beta.data = state['prompt_beta'].to(device)
                logger.info(f"Loaded prompt_beta: {state['prompt_beta'].item()}")
        
        # Image Fusion parameters
        if model.fusion_type == "sigma_add":
            image_fusion_beta_path = load_dir / "image_fusion_beta.pt"
            if image_fusion_beta_path.exists():
                state = torch.load(image_fusion_beta_path, map_location=device, weights_only=False)
                if 'image_fusion_beta' in state and hasattr(model, 'image_fusion_beta'):
                    model.image_fusion_beta.data = state['image_fusion_beta'].to(device)
                    logger.info(f"Loaded image_fusion_beta: {state['image_fusion_beta'].item()}")
        elif model.fusion_type == "cross_attention":
            image_fusion_cross_path = load_dir / "image_fusion_cross_attention.pt"
            if image_fusion_cross_path.exists():
                state_dict = torch.load(image_fusion_cross_path, map_location=device, weights_only=False)
                model.image_fusion.load_state_dict(state_dict)
                logger.info("Loaded Cross-Attention fusion module")
        
        # SEG Token Embedding
        seg_embedding_path = load_dir / "seg_token_embedding.pt"
        if seg_embedding_path.exists():
            state = torch.load(seg_embedding_path, map_location=device, weights_only=False)
            if 'seg_token_embedding' in state:
                model.seg_token_embedding = nn.Parameter(state['seg_token_embedding'].to(device))
                if 'seg_token_id' in state:
                    model.seg_token_id = state['seg_token_id']
                logger.info(f"Loaded SEG token embedding for ID {model.seg_token_id}")
        
        # 6. Load SAM components
        # SAM LoRA
        sam_lora_path = load_dir / "sam_lora.pt"
        sam_mask_decoder_path = load_dir / "sam_mask_decoder.pt"
        
        if sam_lora_path.exists():
            sam_lora_state = torch.load(sam_lora_path, map_location=device, weights_only=True)
            # Apply LoRA weights
            for name, param in model.sam_mask_decoder.named_parameters():
                if name in sam_lora_state:
                    param.data = sam_lora_state[name].to(device)
            logger.info(f"Loaded SAM LoRA: {len(sam_lora_state)} modules")
        elif sam_mask_decoder_path.exists():
            # Load full MaskDecoder
            state_dict = torch.load(sam_mask_decoder_path, map_location=device, weights_only=True)
            model.sam_mask_decoder.load_state_dict(state_dict)
            logger.info(f"Loaded full SAM MaskDecoder")
        
        # 7. Load Qwen components
        qwen_dir = load_dir / "qwen"
        if qwen_dir.exists():
            # Check if it's LoRA or full model
            adapter_config_path = qwen_dir / "adapter_config.json"
            lora_weights_path = qwen_dir / "lora_weights.pt"
            
            if adapter_config_path.exists():
                # Load PEFT LoRA adapter
                try:
                    from peft import PeftModel
                    model.qwen = PeftModel.from_pretrained(model.qwen, str(qwen_dir))
                    logger.info(f"Loaded Qwen LoRA adapter")
                except Exception as e:
                    logger.warning(f"Failed to load PEFT adapter: {e}")
                    # Try alternative loading
                    if lora_weights_path.exists():
                        lora_state = torch.load(lora_weights_path, map_location=device, weights_only=True)
                        for name, param in model.qwen.named_parameters():
                            if name in lora_state:
                                param.data = lora_state[name].to(device)
                        logger.info(f"Loaded Qwen LoRA manually")
            elif lora_weights_path.exists():
                # Load manual LoRA weights
                lora_state = torch.load(lora_weights_path, map_location=device, weights_only=True)
                for name, param in model.qwen.named_parameters():
                    if name in lora_state:
                        param.data = lora_state[name].to(device)
                logger.info(f"Loaded Qwen LoRA weights")
            else:
                # Load full Qwen model
                from transformers import Qwen2_5_VLForConditionalGeneration
                model.qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(qwen_dir))
                logger.info(f"Loaded full Qwen model")
        
        model.to(device)
        logger.info(f"Model loaded successfully from {load_dir}")
        
        return model
    
    @classmethod
    def from_pretrained(cls, load_directory: str, **kwargs):
        """Load model from saved components"""
        import os
        
        # Load config
        config = torch.load(os.path.join(load_directory, "config.pt"), weights_only=False)
        
        # Initialize model
        model = cls(config, **kwargs)
        
        # Load adapter weights
        model.image_adapter.load_state_dict(
            torch.load(os.path.join(load_directory, "image_adapter.pt"), weights_only=True)
        )
        model.text_prompt_proj.load_state_dict(
            torch.load(os.path.join(load_directory, "text_prompt_proj.pt"), weights_only=True)
        )
        # Load high-res generator if exists
        high_res_gen_path = os.path.join(load_directory, "high_res_generator.pt")
        if os.path.exists(high_res_gen_path):
            model.high_res_generator.load_state_dict(
                torch.load(high_res_gen_path, weights_only=True)
            )
        
        # Load Qwen if saved
        qwen_path = os.path.join(load_directory, "qwen")
        if os.path.exists(qwen_path):
            model.qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(qwen_path)
        
        # Load tokenizer if saved
        tokenizer_path = os.path.join(load_directory, "tokenizer.json")
        if os.path.exists(tokenizer_path):
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(load_directory)
            model.set_tokenizer(tokenizer)
        
        return model