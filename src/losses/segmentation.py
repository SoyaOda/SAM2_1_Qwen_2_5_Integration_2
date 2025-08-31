"""
セグメンテーション損失関数
4-2の要件に従って安定化されたDice損失とBCE損失を実装
"""
import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Dice損失計算（logitsから直接計算）
    
    Args:
        logits: 予測logits (N,1,H,W) or (N,C,H,W) for binary C=1
        targets: ターゲットマスク（0/1のfloat tensor）same spatial size
        eps: 数値安定性のためのイプシロン
    
    Returns:
        Dice損失（スカラー）
    """
    # Sigmoid変換
    probs = torch.sigmoid(logits)
    
    # バッチごとに計算（空間次元でsum）
    dims = tuple(range(2, probs.ndim))  # (H, W)次元
    
    # 2 * intersection / (sum_pred + sum_target)
    num = 2.0 * (probs * targets).sum(dims)
    den = (probs.pow(2) + targets.pow(2)).sum(dims) + eps
    dice = 1.0 - (num + eps) / den
    
    return dice.mean()


def bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, 
                   pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Binary Cross Entropy with logits
    
    Args:
        logits: 予測logits
        targets: ターゲットマスク（0/1）
        pos_weight: 正例の重み（クラス不均衡対策）
    
    Returns:
        BCE損失（スカラー）
    """
    return F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction='mean'
    )


def safe_seg_loss(logits: torch.Tensor, targets: torch.Tensor,
                  pos_weight: Optional[torch.Tensor] = None,
                  dice_weight: float = 1.0,
                  bce_weight: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    安定化されたセグメンテーション損失
    空マスクに対応し、Dice損失の数値不安定性を回避
    
    Args:
        logits: 予測logits (N,1,H,W) or (H,W)
        targets: ターゲットマスク（0/1のfloat tensor）
        pos_weight: BCEの正例重み
        dice_weight: Dice損失の重み
        bce_weight: BCE損失の重み
    
    Returns:
        total_loss, bce_loss, dice_loss のタプル
    """
    # 形状を統一（H,W）-> (1,1,H,W)
    if logits.dim() == 2:
        logits = logits.unsqueeze(0).unsqueeze(0)
    elif logits.dim() == 3:
        logits = logits.unsqueeze(0)
    
    if targets.dim() == 2:
        targets = targets.unsqueeze(0).unsqueeze(0)
    elif targets.dim() == 3:
        targets = targets.unsqueeze(0)
    
    # targetsをfloatに変換
    targets = targets.float()
    
    # 前景があるかチェック（バッチごと）
    with torch.no_grad():
        has_fg = (targets.sum(dim=(2, 3)) > 0)  # (N,1)
        has_fg = has_fg.squeeze(1)  # (N,)
    
    # BCE損失（常に計算）
    bce = bce_with_logits(logits, targets, pos_weight=pos_weight)
    
    # Dice損失（前景がある場合のみ）
    if has_fg.any():
        # 前景があるサンプルのみでDice損失を計算
        fg_logits = logits[has_fg]
        fg_targets = targets[has_fg]
        
        if fg_logits.numel() > 0:
            dice = dice_loss_with_logits(fg_logits, fg_targets)
        else:
            dice = torch.tensor(0.0, device=logits.device)
    else:
        # 全バッチが空前景ならDiceは0
        dice = torch.tensor(0.0, device=logits.device)
    
    # 総合損失
    loss = bce_weight * bce + dice_weight * dice
    
    return loss, bce, dice


def compute_pos_weight(masks: torch.Tensor, min_ratio: float = 0.01) -> torch.Tensor:
    """
    正例の重みを計算（クラス不均衡対策）
    
    Args:
        masks: バイナリマスク
        min_ratio: 最小比率（ゼロ除算回避）
    
    Returns:
        pos_weight tensor
    """
    pos_frac = masks.mean().clamp(min=min_ratio).item()
    neg_frac = 1.0 - pos_frac
    pos_weight = torch.tensor([neg_frac / pos_frac], device=masks.device)
    return pos_weight