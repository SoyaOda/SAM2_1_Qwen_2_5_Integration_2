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
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor as SAM2IP
            
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
                model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            else:
                checkpoint = config.sam_model_name
                model_cfg = config.sam_model_name.replace(".pt", ".yaml")
            
            # Build SAM2 model
            sam2_model = build_sam2(model_cfg, checkpoint, device=config.device_map)
            self.sam_predictor = SAM2IP(sam2_model)
            self.sam_model = self.sam_predictor.model
        
        # Access mask decoder and prompt encoder
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
        
        # Initialize adapters
        self.image_adapter = ImageFeatureAdapter(
            in_dim=config.qwen_vision_hidden_size,
            out_dim=config.sam_image_embedding_dim
        )
        
        self.text_prompt_proj = TextPromptProjector(
            in_dim=config.qwen_hidden_size,
            out_dim=config.text_prompt_out_dim,
            use_mlp=False  # Start with linear projection
        )
        
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
    
    def extract_vision_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract vision features from Qwen's vision encoder
        
        Args:
            pixel_values: Input images [B, C, H, W]
        
        Returns:
            Vision features [B, N_patches, D_v]
        """
        # Get vision encoder from Qwen model
        # The exact method depends on Qwen2.5-VL's implementation
        if hasattr(self.qwen, 'visual'):
            # Direct access to vision model
            vision_features = self.qwen.visual(pixel_values)
        elif hasattr(self.qwen, 'vision_model'):
            vision_features = self.qwen.vision_model(pixel_values)
        else:
            # Fallback: run full model and extract vision features
            # This is less efficient but ensures compatibility
            with torch.no_grad():
                outputs = self.qwen(
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                    return_dict=True
                )
                # Extract vision features from appropriate layer
                if hasattr(outputs, 'vision_hidden_states'):
                    vision_features = outputs.vision_hidden_states
                else:
                    raise AttributeError("Cannot extract vision features from Qwen model")
        
        return vision_features
    
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        mask_labels: Optional[List[torch.Tensor]] = None,
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
            output_hidden_states=True,
            return_dict=True
        )
        
        # Extract outputs
        logits = qwen_outputs.logits  # [B, seq_len, vocab_size]
        hidden_states = qwen_outputs.hidden_states[-1]  # Last layer [B, seq_len, D_l]
        
        # 2. Extract vision features if images are provided
        vision_features = None
        image_features_sam = None
        
        if pixel_values is not None:
            # Get vision features from Qwen
            vision_features = self.extract_vision_features(pixel_values)
            
            # Transform to SAM format
            image_features_sam = self.image_adapter(vision_features)  # [B, 256, H, W]
        
        # 3. Find SEG token positions and generate masks
        mask_logits = []
        seg_positions = []
        
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
        
        # 4. Generate masks for each SEG position
        for i in range(B):
            sample_masks = []
            
            if len(seg_positions[i]) > 0 and image_features_sam is not None:
                # Extract hidden states at SEG positions
                seg_hidden_states = hidden_states[i, seg_positions[i]]  # [num_segs, D_l]
                
                # Project to prompt embeddings
                if len(seg_positions[i]) == 1:
                    prompt_embeds = self.text_prompt_proj(seg_hidden_states)  # [1, 256]
                else:
                    prompt_embeds = self.text_prompt_proj(seg_hidden_states)  # [num_segs, 256]
                
                # Generate mask for each SEG token
                for j, prompt_embed in enumerate(prompt_embeds):
                    # Prepare inputs for SAM mask decoder
                    # Get positional encoding for the image
                    image_pe = self.sam_prompt_encoder.get_dense_pe()
                    
                    # SAM2 expects specific input format for mask decoder
                    # Create sparse embeddings (points/boxes)
                    sparse_embeddings = prompt_embed.unsqueeze(0).unsqueeze(0)  # [1, 1, 256]
                    
                    # No dense embeddings for text-only prompts
                    dense_embeddings = torch.empty(
                        1, 0, self.sam_prompt_encoder.embed_dim,
                        device=prompt_embed.device
                    )
                    
                    # Run mask decoder
                    low_res_masks, iou_predictions = self.sam_mask_decoder(
                        image_embeddings=image_features_sam[i:i+1],  # [1, 256, H, W]
                        image_pe=image_pe,  # Positional encoding
                        sparse_prompt_embeddings=sparse_embeddings,  # [1, 1, 256]
                        dense_prompt_embeddings=dense_embeddings,  # [1, 0, 256]
                        multimask_output=False,  # Single mask per prompt
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
        vision_features = self.extract_vision_features(pixel_values)
        image_features_sam = self.image_adapter(vision_features)
        
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
                        
                        # Generate mask
                        seg_hidden = current_hidden[b]  # [D_l]
                        prompt_embed = self.text_prompt_proj(seg_hidden)  # [256]
                        
                        # Get positional encoding
                        image_pe = self.sam_prompt_encoder.get_dense_pe()
                        
                        # Prepare inputs for mask decoder
                        sparse_embeddings = prompt_embed.unsqueeze(0).unsqueeze(0)  # [1, 1, 256]
                        dense_embeddings = torch.empty(
                            1, 0, self.sam_prompt_encoder.embed_dim,
                            device=device
                        )
                        
                        # Run mask decoder
                        with torch.no_grad():
                            low_res_masks, _ = self.sam_mask_decoder(
                                image_embeddings=image_features_sam[b:b+1],
                                image_pe=image_pe,
                                sparse_prompt_embeddings=sparse_embeddings,
                                dense_prompt_embeddings=dense_embeddings,
                                multimask_output=False
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
        
        # Save Qwen if modified (e.g., with LoRA or new embeddings)
        if not self.config.freeze_qwen or self.config.train_seg_token:
            self.qwen.save_pretrained(os.path.join(save_directory, "qwen"))
    
    @classmethod
    def from_pretrained(cls, load_directory: str, **kwargs):
        """Load model from saved components"""
        import os
        
        # Load config
        config = torch.load(os.path.join(load_directory, "config.pt"))
        
        # Initialize model
        model = cls(config, **kwargs)
        
        # Load adapter weights
        model.image_adapter.load_state_dict(
            torch.load(os.path.join(load_directory, "image_adapter.pt"))
        )
        model.text_prompt_proj.load_state_dict(
            torch.load(os.path.join(load_directory, "text_prompt_proj.pt"))
        )
        
        # Load Qwen if saved
        qwen_path = os.path.join(load_directory, "qwen")
        if os.path.exists(qwen_path):
            model.qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(qwen_path)
        
        return model