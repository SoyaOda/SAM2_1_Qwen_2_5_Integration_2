"""
Loss functions for LISA改 (LISA-Kai) model
Includes language modeling loss and segmentation losses
"""
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List


def compute_language_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100
) -> torch.Tensor:
    """
    Compute cross-entropy loss for language modeling
    
    Args:
        logits: Model output logits [B, seq_len, vocab_size]
        labels: Target token IDs [B, seq_len]
        ignore_index: Index to ignore in loss computation (default: -100)
    
    Returns:
        Scalar loss tensor
    """
    # Flatten for cross entropy
    vocab_size = logits.size(-1)
    logits_flat = logits.view(-1, vocab_size)
    labels_flat = labels.view(-1)
    
    # Compute cross entropy loss
    loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=ignore_index)
    
    return loss


def compute_dice_loss(
    pred_masks: torch.Tensor,
    true_masks: torch.Tensor,
    smooth: float = 1e-5
) -> torch.Tensor:
    """
    Compute soft Dice loss for segmentation
    
    Args:
        pred_masks: Predicted masks (after sigmoid) [B, H, W] or [B, 1, H, W]
        true_masks: Ground truth masks [B, H, W] or [B, 1, H, W]
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Scalar Dice loss (1 - Dice coefficient)
    """
    # Ensure masks have same shape
    if pred_masks.dim() == 4:
        pred_masks = pred_masks.squeeze(1)
    if true_masks.dim() == 4:
        true_masks = true_masks.squeeze(1)
    
    # Flatten spatial dimensions
    pred_flat = pred_masks.view(pred_masks.size(0), -1)
    true_flat = true_masks.view(true_masks.size(0), -1)
    
    # Compute Dice coefficient
    intersection = (pred_flat * true_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + true_flat.sum(dim=1)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    # Return Dice loss (1 - Dice coefficient)
    return 1.0 - dice.mean()


def compute_segmentation_loss(
    mask_logits: torch.Tensor,
    mask_labels: torch.Tensor,
    loss_type: str = "bce_dice"
) -> Tuple[torch.Tensor, dict]:
    """
    Compute segmentation loss combining BCE and Dice
    
    Args:
        mask_logits: Predicted mask logits [B, H, W] or [B, 1, H, W]
        mask_labels: Ground truth masks [B, H, W] or [B, 1, H, W]
        loss_type: Type of loss ("bce", "dice", or "bce_dice")
    
    Returns:
        Total loss and dictionary of individual losses
    """
    # Ensure float labels for BCE
    mask_labels = mask_labels.float()
    
    losses = {}
    total_loss = 0.0
    
    if loss_type in ["bce", "bce_dice"]:
        # Binary cross entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(mask_logits, mask_labels)
        losses["bce_loss"] = bce_loss
        total_loss = total_loss + bce_loss
    
    if loss_type in ["dice", "bce_dice"]:
        # Dice loss (apply sigmoid first)
        pred_masks = torch.sigmoid(mask_logits)
        dice_loss = compute_dice_loss(pred_masks, mask_labels)
        losses["dice_loss"] = dice_loss
        total_loss = total_loss + dice_loss
    
    losses["total_seg_loss"] = total_loss
    
    return total_loss, losses



def compute_lisa_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask_logits_list: Optional[List[List[torch.Tensor]]] = None,
    mask_labels_list: Optional[List[List[torch.Tensor]]] = None,
    language_weight: float = 1.0,
    segmentation_weight: float = 1.0,
    ignore_index: int = -100
) -> Tuple[torch.Tensor, dict]:
    """
    Compute combined loss for LISA model
    
    Args:
        logits: Language model logits [B, seq_len, vocab_size]
        labels: Language modeling labels [B, seq_len]
        mask_logits_list: List of mask predictions per sample
        mask_labels_list: List of ground truth masks per sample
        language_weight: Weight for language modeling loss
        segmentation_weight: Weight for segmentation loss
        ignore_index: Index to ignore in language loss
    
    Returns:
        Total loss and dictionary of all loss components
    """
    losses = {}
    
    # Language modeling loss
    lm_loss = compute_language_loss(logits, labels, ignore_index)
    losses["language_loss"] = lm_loss
    
    # Segmentation loss (if masks are provided)
    seg_loss = 0.0
    num_masks = 0
    
    if mask_logits_list is not None and mask_labels_list is not None:
        seg_losses = []
        
        for sample_preds, sample_labels in zip(mask_logits_list, mask_labels_list):
            if sample_preds is not None and sample_labels is not None:
                # Handle multiple masks per sample
                for pred_mask, true_mask in zip(sample_preds, sample_labels):
                    # Use standard segmentation loss
                    mask_loss, mask_loss_dict = compute_segmentation_loss(
                        pred_mask, true_mask
                    )
                    seg_losses.append(mask_loss)
                    num_masks += 1
        
        if num_masks > 0:
            seg_loss = torch.stack(seg_losses).mean()
            losses["segmentation_loss"] = seg_loss
            losses["num_masks"] = num_masks
    
    # Combine losses
    total_loss = language_weight * lm_loss
    if num_masks > 0:
        total_loss = total_loss + segmentation_weight * seg_loss
    
    losses["total_loss"] = total_loss
    
    return total_loss, losses


class LISALoss(torch.nn.Module):
    """
    Loss module for LISA model training
    """
    
    def __init__(
        self,
        language_weight: float = 1.0,
        segmentation_weight: float = 1.0,
        seg_loss_type: str = "bce_dice",
        ignore_index: int = -100
    ):
        """
        Initialize LISA loss module
        
        Args:
            language_weight: Weight for language modeling loss
            segmentation_weight: Weight for segmentation loss
            seg_loss_type: Type of segmentation loss
            ignore_index: Index to ignore in language loss
        """
        super().__init__()
        self.language_weight = language_weight
        self.segmentation_weight = segmentation_weight
        self.seg_loss_type = seg_loss_type
        self.ignore_index = ignore_index
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask_logits_list: Optional[List[List[torch.Tensor]]] = None,
        mask_labels_list: Optional[List[List[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute LISA loss
        
        Args:
            logits: Language model outputs
            labels: Language targets
            mask_logits_list: Mask predictions
            mask_labels_list: Mask targets
        
        Returns:
            Total loss and loss components
        """
        return compute_lisa_loss(
            logits=logits,
            labels=labels,
            mask_logits_list=mask_logits_list,
            mask_labels_list=mask_labels_list,
            language_weight=self.language_weight,
            segmentation_weight=self.segmentation_weight,
            ignore_index=self.ignore_index
        )