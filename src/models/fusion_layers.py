import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention based fusion module for SAM and Qwen features.
    SAM features are used as Query, Qwen features as Key/Value.
    """
    def __init__(
        self, 
        dim: int = 256, 
        num_heads: int = 8, 
        dropout: float = 0.1,
        use_gate: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.use_gate = use_gate
        
        # Multi-head Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        # LayerNorm for residual connection
        self.norm = nn.LayerNorm(dim)
        
        # Optional gate mechanism for dynamic fusion control
        if use_gate:
            self.gate_fc = nn.Linear(dim * 2, dim)
            self.gate_sigmoid = nn.Sigmoid()
            # Initialize gate bias to favor original features initially
            nn.init.constant_(self.gate_fc.bias, -2.0)
        
        logger.info(f"Initialized CrossAttentionFusion: dim={dim}, heads={num_heads}, gate={use_gate}")

    def forward(
        self, 
        sam_feats: torch.Tensor, 
        qwen_feats: torch.Tensor,
        return_attention_weights: bool = False
    ) -> torch.Tensor:
        """
        Args:
            sam_feats: [B, N, D] SAM features (Query)
            qwen_feats: [B, M, D] Qwen features (Key/Value)
            return_attention_weights: If True, return attention weights
        Returns:
            fused_feats: [B, N, D] Fused features
        """
        # Validate input shapes
        if sam_feats.dim() != 3 or qwen_feats.dim() != 3:
            raise ValueError(f"Expected 3D tensors, got sam_feats: {sam_feats.dim()}D, qwen_feats: {qwen_feats.dim()}D")
        
        if sam_feats.size(-1) != qwen_feats.size(-1):
            raise ValueError(f"Feature dimensions must match: sam_feats: {sam_feats.size(-1)}, qwen_feats: {qwen_feats.size(-1)}")
        
        B, N, D = sam_feats.shape
        
        # Cross-Attention: SAM queries attend to Qwen features
        attn_output, attn_weights = self.cross_attn(
            query=sam_feats,
            key=qwen_feats, 
            value=qwen_feats,
            need_weights=return_attention_weights
        )
        
        # Residual connection + LayerNorm
        fused = self.norm(attn_output + sam_feats)
        
        # Optional gate mechanism for controlled fusion
        if self.use_gate:
            # Concatenate original and fused features
            gate_input = torch.cat([sam_feats, fused], dim=-1)  # [B, N, 2*D]
            gate = self.gate_sigmoid(self.gate_fc(gate_input))  # [B, N, D]
            # Weighted combination: gate controls how much fusion to apply
            output = gate * fused + (1 - gate) * sam_feats
        else:
            output = fused
        
        if return_attention_weights:
            return output, attn_weights
        return output


class SigmaAddFusion(nn.Module):
    """
    Sigma-based additive fusion module (existing implementation).
    Uses a learnable scalar parameter with sigmoid activation.
    """
    def __init__(self, init_value: float = 0.0, external_beta: torch.nn.Parameter = None):
        super().__init__()
        if external_beta is not None:
            # Use external beta parameter (shared with parent model)
            self.beta = external_beta
            self.owns_beta = False
            logger.info(f"Initialized SigmaAddFusion with external beta")
        else:
            # Create own beta parameter
            self.beta = nn.Parameter(torch.tensor(init_value))
            self.owns_beta = True
            logger.info(f"Initialized SigmaAddFusion with internal beta={init_value}")
    
    def forward(
        self, 
        sam_feats: torch.Tensor, 
        qwen_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            sam_feats: [B, C, H, W] SAM features
            qwen_feats: [B, C, H, W] Qwen features (same spatial size)
        Returns:
            fused_feats: [B, C, H, W] Fused features
        """
        # Validate input shapes
        if sam_feats.shape != qwen_feats.shape:
            raise ValueError(f"Shape mismatch: sam_feats {sam_feats.shape} vs qwen_feats {qwen_feats.shape}")
        
        beta_scaled = torch.sigmoid(self.beta)
        fused = sam_feats + beta_scaled * qwen_feats
        return fused


class HybridFusionModule(nn.Module):
    """
    Hybrid fusion module that can switch between Cross-Attention and Sigma-Add fusion.
    """
    def __init__(
        self,
        dim: int = 256,
        fusion_type: str = "cross_attention",  # "cross_attention" or "sigma_add"
        num_heads: int = 8,
        dropout: float = 0.1,
        use_gate: bool = True,
        external_beta: torch.nn.Parameter = None  # For sigma_add fusion
    ):
        super().__init__()
        self.dim = dim
        self.fusion_type = fusion_type
        
        if fusion_type == "cross_attention":
            self.fusion = CrossAttentionFusion(
                dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                use_gate=use_gate
            )
        elif fusion_type == "sigma_add":
            self.fusion = SigmaAddFusion(init_value=0.0, external_beta=external_beta)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        logger.info(f"HybridFusionModule initialized with fusion_type={fusion_type}")
    
    def forward(
        self,
        sam_features: torch.Tensor,
        qwen_features: torch.Tensor,
        return_attention_weights: bool = False
    ) -> torch.Tensor:
        """
        Unified interface for different fusion methods.
        
        Args:
            sam_features: SAM features (shape depends on fusion type)
            qwen_features: Qwen features (shape depends on fusion type)
            return_attention_weights: Only used for cross_attention
        Returns:
            Fused features
        """
        if self.fusion_type == "cross_attention":
            # Expect flattened features for cross-attention
            if sam_features.dim() == 4:  # [B, C, H, W]
                B, C, H, W = sam_features.shape
                sam_features = sam_features.flatten(2).transpose(1, 2)  # [B, H*W, C]
                qwen_features = qwen_features.flatten(2).transpose(1, 2)  # [B, H*W, C]
                
                if return_attention_weights:
                    fused, attn_weights = self.fusion(
                        sam_features, qwen_features, return_attention_weights=True
                    )
                    # Reshape back to spatial format
                    fused = fused.transpose(1, 2).reshape(B, C, H, W)
                    return fused, attn_weights
                else:
                    fused = self.fusion(sam_features, qwen_features)
                    # Reshape back to spatial format
                    fused = fused.transpose(1, 2).reshape(B, C, H, W)
                    return fused
            else:
                # Already flattened
                return self.fusion(sam_features, qwen_features, return_attention_weights)
        
        elif self.fusion_type == "sigma_add":
            # Expect spatial features for sigma-add
            return self.fusion(sam_features, qwen_features)