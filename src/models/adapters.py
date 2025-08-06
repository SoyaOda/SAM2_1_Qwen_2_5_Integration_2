"""
Adapter modules for LISA改 (LISA-Kai)
Includes ImageFeatureAdapter and TextPromptProjector
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class ImageFeatureAdapter(nn.Module):
    """
    Adapter to transform Qwen vision features to SAM-compatible format
    
    This module takes vision features from Qwen2.5-VL's vision encoder
    and transforms them to match SAM2.1's expected input format.
    """
    
    def __init__(self, in_dim: int, out_dim: int = 256):
        """
        Args:
            in_dim: Input dimension from Qwen vision encoder
            out_dim: Output dimension for SAM mask decoder (default: 256)
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Linear projection to match SAM's expected dimension
        self.proj = nn.Linear(in_dim, out_dim)
        
        # Layer normalization for stability
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Transform vision features from Qwen to SAM format
        
        Args:
            vision_features: Tensor of shape [B, N_patches, D_v]
                            where B is batch size, N_patches is number of patches,
                            D_v is Qwen vision hidden dimension
        
        Returns:
            Tensor of shape [B, out_dim, H, W] suitable for SAM mask decoder
        """
        B, N, D_v = vision_features.shape
        
        # Project features to target dimension
        features = self.proj(vision_features)  # [B, N, out_dim]
        features = self.norm(features)
        
        # Calculate spatial dimensions assuming square patches
        # N should be a perfect square for standard vision transformers
        H = W = int(math.sqrt(N))
        
        if H * W != N:
            # Handle non-square patch grids
            # Find the closest factors
            factors = []
            for i in range(1, int(math.sqrt(N)) + 1):
                if N % i == 0:
                    factors.append((i, N // i))
            
            # Choose the most square-like factorization
            H, W = min(factors, key=lambda x: abs(x[0] - x[1]))
        
        # Reshape to spatial format
        # [B, N, out_dim] -> [B, H, W, out_dim] -> [B, out_dim, H, W]
        features_2d = features.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        return features_2d
    
    def get_output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        """
        Calculate output shape given input shape
        
        Args:
            input_shape: (B, N_patches, D_v)
        
        Returns:
            output_shape: (B, out_dim, H, W)
        """
        B, N, _ = input_shape
        H = W = int(math.sqrt(N))
        
        if H * W != N:
            factors = []
            for i in range(1, int(math.sqrt(N)) + 1):
                if N % i == 0:
                    factors.append((i, N // i))
            H, W = min(factors, key=lambda x: abs(x[0] - x[1]))
            
        return (B, self.out_dim, H, W)


class TextPromptProjector(nn.Module):
    """
    Projects text hidden states from Qwen LLM to prompt embeddings for SAM
    
    This module transforms the hidden state at <SEG> token positions
    into prompt embeddings that guide SAM's mask generation.
    """
    
    def __init__(self, in_dim: int, out_dim: int = 256, use_mlp: bool = False):
        """
        Args:
            in_dim: Input dimension from Qwen LLM hidden states
            out_dim: Output dimension for SAM prompt embedding (default: 256)
            use_mlp: Whether to use MLP projection instead of linear
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        if use_mlp:
            # O3推奨: Two-layer MLP with LayerNorm and GELU activation
            hidden_dim = 512  # Fixed hidden dimension as recommended
            self.proj = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim)
            )
        else:
            # Simple linear projection
            self.proj = nn.Linear(in_dim, out_dim)
        
        # Layer normalization for output stability
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, text_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Project text hidden states to SAM prompt embeddings
        
        Args:
            text_hidden_states: Tensor of shape [B, D_l] or [B, 1, D_l]
                               where B is batch size, D_l is LLM hidden dimension
        
        Returns:
            Tensor of shape [B, out_dim] containing prompt embeddings
        """
        # Handle both [B, D_l] and [B, 1, D_l] inputs
        if text_hidden_states.dim() == 3:
            text_hidden_states = text_hidden_states.squeeze(1)
        
        # Project to target dimension
        prompt_embeds = self.proj(text_hidden_states)  # [B, out_dim]
        prompt_embeds = self.norm(prompt_embeds)
        
        return prompt_embeds
    
    def forward_multiple(self, text_hidden_states: torch.Tensor, 
                        seg_positions: torch.Tensor) -> torch.Tensor:
        """
        Project multiple SEG token positions from a sequence
        
        Args:
            text_hidden_states: Tensor of shape [B, seq_len, D_l]
            seg_positions: Tensor of shape [B, num_segs] containing SEG positions
        
        Returns:
            Tensor of shape [B, num_segs, out_dim] containing prompt embeddings
        """
        B, seq_len, D_l = text_hidden_states.shape
        num_segs = seg_positions.shape[1]
        
        # Gather hidden states at SEG positions
        # Create batch indices
        batch_indices = torch.arange(B, device=seg_positions.device).unsqueeze(1).expand(-1, num_segs)
        
        # Gather SEG hidden states
        seg_hidden = text_hidden_states[batch_indices, seg_positions]  # [B, num_segs, D_l]
        
        # Project each SEG hidden state
        seg_hidden_flat = seg_hidden.view(-1, D_l)  # [B*num_segs, D_l]
        prompt_embeds_flat = self.proj(seg_hidden_flat)  # [B*num_segs, out_dim]
        prompt_embeds_flat = self.norm(prompt_embeds_flat)
        
        # Reshape back
        prompt_embeds = prompt_embeds_flat.view(B, num_segs, self.out_dim)
        
        return prompt_embeds