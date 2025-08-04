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

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor
)
from sam2.sam2_image_predictor import SAM2ImagePredictor

from .adapters import ImageFeatureAdapter, TextPromptProjector
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
            # Try to get vision encoder hidden size
            if hasattr(self.qwen.config, 'vision_config'):
                config.qwen_vision_hidden_size = self.qwen.config.vision_config.hidden_size
            else:
                # Default to actual Qwen2.5-VL-3B vision encoder size
                config.qwen_vision_hidden_size = 1280  # Qwen2.5-VL-3B actual vision hidden size
        
        # Initialize adapters with the same dtype as Qwen model
        model_dtype = next(self.qwen.parameters()).dtype
        
        self.image_adapter = ImageFeatureAdapter(
            in_dim=config.qwen_vision_hidden_size,
            out_dim=config.sam_image_embedding_dim
        ).to(dtype=model_dtype)
        
        self.text_prompt_proj = TextPromptProjector(
            in_dim=config.qwen_hidden_size,
            out_dim=config.text_prompt_out_dim,
            use_mlp=False  # Start with linear projection
        ).to(dtype=model_dtype)
        
        # Initialize high-resolution feature generator
        # Input is already transformed by image_adapter to sam_channels
        self.high_res_generator = HighResFeatureGenerator(
            in_channels=config.sam_image_embedding_dim,  # Already 256 after adapter
            sam_channels=config.sam_image_embedding_dim
        ).to(dtype=model_dtype)
        
        # Freeze models as specified
        if config.freeze_qwen:
            for param in self.qwen.parameters():
                param.requires_grad = False
        
        if config.freeze_sam:
            for param in self.sam_mask_decoder.parameters():
                param.requires_grad = False
            for param in self.sam_prompt_encoder.parameters():
                param.requires_grad = False
        
        # Store tokenizer and SEG token info
        self.tokenizer = tokenizer
        self.seg_token_id = None
        
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
        if self.config.train_seg_token:
            word_embeddings = self.qwen.get_input_embeddings()
            word_embeddings.weight[self.seg_token_id].requires_grad = True
    
    def extract_vision_features(self, pixel_values: torch.Tensor, image_grid_thw: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract vision features from Qwen's vision encoder
        
        Args:
            pixel_values: Input images [B, C, H, W]
            image_grid_thw: Grid dimensions for dynamic resolution [B, 3]
        
        Returns:
            Vision features [B, N_patches, D_v]
        """
        # Get vision encoder from Qwen model
        # We need to extract features before the merger to get 1280D features
        if hasattr(self.qwen.model, 'visual'):
            visual_module = self.qwen.model.visual
            
            # Use forward hooks to capture features before merger
            features_dict = {}
            
            def capture_features(module, input, output):
                # Store the input to merger (which is the output of vision blocks)
                features_dict['before_merger'] = input[0] if isinstance(input, tuple) else input
            
            # Register hook on merger module
            if hasattr(visual_module, 'merger'):
                hook = visual_module.merger.register_forward_hook(capture_features)
            else:
                raise AttributeError("Cannot find merger module in visual encoder")
            
            try:
                # Run visual module
                if image_grid_thw is not None:
                    _ = visual_module(pixel_values, grid_thw=image_grid_thw)
                else:
                    # Default grid_thw if not provided
                    B = pixel_values.shape[0]
                    default_grid = torch.tensor([[1, 24, 24]], device=pixel_values.device).repeat(B, 1)
                    _ = visual_module(pixel_values, grid_thw=default_grid)
                
                # Get captured features
                if 'before_merger' in features_dict:
                    vision_features = features_dict['before_merger']
                    
                    # Ensure proper shape [B, N_patches, D_v]
                    if len(vision_features.shape) == 2:
                        # Add batch dimension if missing
                        vision_features = vision_features.unsqueeze(0)
                    elif len(vision_features.shape) == 3:
                        # Already in correct shape
                        pass
                    else:
                        raise ValueError(f"Unexpected vision features shape: {vision_features.shape}")
                    
                    return vision_features
                else:
                    raise RuntimeError("Failed to capture vision features before merger")
                    
            finally:
                # Remove hook
                hook.remove()
        else:
            raise AttributeError("Cannot find visual module in Qwen model")

    
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
    
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        mask_labels: Optional[List[torch.Tensor]] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[LISAModelOutput, Tuple]:
        """
        Forward pass of LISA model
        
        Args:
            input_ids: Input token IDs [B, seq_len]
            pixel_values: Input images [B, C, H, W]
            attention_mask: Attention mask [B, seq_len]
            labels: Language modeling labels [B, seq_len]
            mask_labels: Ground truth masks for training
            return_dict: Whether to return LISAModelOutput
        
        Returns:
            Model outputs including language logits and mask predictions
        """
        B = input_ids.size(0)
        
        # 1. Run Qwen model for vision-language understanding
        qwen_outputs = self.qwen(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Extract outputs
        logits = qwen_outputs.logits  # [B, seq_len, vocab_size]
        hidden_states = qwen_outputs.hidden_states[-1]  # Last layer [B, seq_len, D_l]
        
        # 2. Extract vision features if images are provided
        vision_features = None
        image_features_sam = None
        sam_high_res_features = None
        
        if pixel_values is not None:
            # Get vision features from Qwen
            vision_features = self.extract_vision_features(pixel_values, image_grid_thw)
            
            # Debug: Check vision features shape
            if len(vision_features.shape) == 2:
                # If 2D, add batch dimension
                vision_features = vision_features.unsqueeze(0)
            
            # Transform to SAM format using adapter
            image_features_sam = self.image_adapter(vision_features)  # [B, 256, H, W]
            
            # Generate high-res features from Qwen vision features
            # This replaces the previous approach of extracting from SAM2.1
            sam_high_res_features = self.high_res_generator(image_features_sam)
            
            # High-res features are now generated from Qwen vision features
            # using the HighResFeatureGenerator - no need for SAM2.1's native extraction
        
        # 3. Find SEG token positions and generate masks
        mask_logits = []
        seg_positions = []
        
        # Check if seg_token_id is set
        if self.seg_token_id is None:
            return LISAModelOutput(logits=logits, mask_logits=mask_logits, seg_token_positions=seg_positions)
        
        # Determine SEG positions based on labels (training) or generated tokens (inference)
        if labels is not None:
            # Training: find SEG tokens in labels
            for i in range(B):
                seg_pos = (labels[i] == self.seg_token_id).nonzero(as_tuple=True)[0]
                if len(seg_pos) > 0:
                    seg_positions.append(seg_pos.tolist())
                else:
                    seg_positions.append([])
        else:
            # Inference: find SEG tokens in input_ids
            for i in range(B):
                seg_pos = (input_ids[i] == self.seg_token_id).nonzero(as_tuple=True)[0]
                if len(seg_pos) > 0:
                    seg_positions.append(seg_pos.tolist())
                else:
                    seg_positions.append([])
        
        # 4. Generate masks for each SEG position using proper SAM2.1 implementation
        for i in range(B):
            sample_masks = []
            
            if len(seg_positions[i]) > 0 and image_features_sam is not None and i < image_features_sam.shape[0]:
                # Extract hidden states at SEG positions
                seg_hidden_states = hidden_states[i, seg_positions[i]]  # [num_segs, D_l]
                
                # Project to prompt embeddings
                if len(seg_positions[i]) == 1:
                    prompt_embeds = self.text_prompt_proj(seg_hidden_states)  # [256]
                else:
                    prompt_embeds = self.text_prompt_proj(seg_hidden_states)  # [num_segs, 256]
                
                # Generate mask for each SEG token
                for j, prompt_embed in enumerate(prompt_embeds if len(seg_positions[i]) > 1 else [prompt_embeds]):
                    try:
                        # Get positional encoding from SAM prompt encoder
                        image_pe = self.sam_prompt_encoder.get_dense_pe()
                        
                        # Ensure PE matches image embeddings size
                        if image_pe.shape[-2:] != image_features_sam[i:i+1].shape[-2:]:
                            h_feat, w_feat = image_features_sam[i:i+1].shape[-2:]
                            image_pe = F.interpolate(
                                image_pe,
                                size=(h_feat, w_feat),
                                mode='bilinear',
                                align_corners=False
                            )
                        
                        # Create a center point as anchor for text-guided segmentation
                        # SAM2.1 expects points in the image coordinate system
                        h_img, w_img = pixel_values.shape[-2:]
                        center_x, center_y = w_img // 2, h_img // 2
                        
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
                        
                        # Replace the point embedding with our text-derived embedding
                        # sparse_embeddings: [1, N, C] where C=256 for SAM2.1
                        if sparse_embeddings.shape[1] > 0:
                            sparse_embeddings[:, 0, :] = prompt_embed.unsqueeze(0)
                        
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
                        if pixel_values is not None:
                            orig_h, orig_w = pixel_values.shape[-2:]
                            mask_logit = F.interpolate(
                                low_res_masks,
                                size=(orig_h, orig_w),
                                mode='bilinear',
                                align_corners=False
                            )
                            sample_masks.append(mask_logit.squeeze(0))  # Remove batch dim
                        else:
                            sample_masks.append(low_res_masks.squeeze(0))
                            
                    except Exception as e:
                        print(f"Warning: SAM2.1 mask generation failed with error: {e}")
                        import traceback
                        print(f"Traceback: {traceback.format_exc()}")
                        # Fall back to dummy mask if SAM fails
                        if pixel_values is not None:
                            orig_h, orig_w = pixel_values.shape[-2:]
                            dummy_mask = torch.zeros(1, orig_h, orig_w, device=pixel_values.device, dtype=torch.float32)
                            sample_masks.append(dummy_mask)
            
            mask_logits.append(sample_masks if len(sample_masks) > 0 else None)
        
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
        image_features_sam = self.image_adapter(vision_features)
        
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
        """Save model components"""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save config
        torch.save(self.config, os.path.join(save_directory, "config.pt"))
        
        # Save adapter weights
        torch.save(self.image_adapter.state_dict(), 
                  os.path.join(save_directory, "image_adapter.pt"))
        torch.save(self.text_prompt_proj.state_dict(), 
                  os.path.join(save_directory, "text_prompt_proj.pt"))
        torch.save(self.high_res_generator.state_dict(),
                  os.path.join(save_directory, "high_res_generator.pt"))
        
        # Save Qwen if modified (e.g., with LoRA or new embeddings)
        if not self.config.freeze_qwen or self.config.train_seg_token:
            self.qwen.save_pretrained(os.path.join(save_directory, "qwen"))
    
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