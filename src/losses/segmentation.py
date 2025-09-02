"""
セグメンテーション損失関数
4-2の要件に従って安定化されたDice損失とBCE損失を実装
"""
import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Dice損失計算（logitsから直接計算）- 改善版
    数値安定性を向上させ、スパイクを抑制
    
    Args:
        logits: 予測logits (N,1,H,W) or (N,C,H,W) for binary C=1
        targets: ターゲットマスク（0/1のfloat tensor）same spatial size
        eps: 数値安定性のためのイプシロン
    
    Returns:
        Dice損失（スカラー）
    """
    # Sigmoid変換（数値安定性のためclampを追加）
    probs = torch.sigmoid(logits)
    # 極端な値を回避
    probs = torch.clamp(probs, min=eps, max=1.0 - eps)
    
    # バッチごとに計算（空間次元でsum）
    dims = tuple(range(2, probs.ndim))  # (H, W)次元
    
    # Dice係数の計算方法を改善（分母をより安定化）
    # 標準的なDice: 2*intersection / (pred + target)
    intersection = (probs * targets).sum(dims)
    pred_sum = probs.sum(dims)
    target_sum = targets.sum(dims)
    
    # 分母を安定化（両方が0の場合を考慮）
    denominator = pred_sum + target_sum + eps
    
    # Dice係数（0-1の範囲）
    dice_coeff = (2.0 * intersection + eps) / denominator
    
    # Dice損失（1 - Dice係数）
    dice_loss = 1.0 - dice_coeff
    
    # 平均を取る前にクランプして異常値を除去
    dice_loss = torch.clamp(dice_loss, min=0.0, max=1.0)
    
    return dice_loss.mean()


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
    安定化されたセグメンテーション損失（改善版）
    空マスクに対応し、Dice損失の数値不安定性を回避
    勾配フローを維持するよう修正
    
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
        has_fg = (targets.sum(dim=(2, 3)) > 0).squeeze(1)  # (N,)
        # 前景の割合を計算（極端に小さい場合の対策）
        fg_ratio = targets.sum() / targets.numel()
    
    # BCE損失（常に計算）
    # pos_weightが極端に大きくならないよう制限
    if pos_weight is not None:
        pos_weight = torch.clamp(pos_weight, min=0.1, max=10.0)
    bce = bce_with_logits(logits, targets, pos_weight=pos_weight)
    
    # Dice損失の計算（安定化版）
    if has_fg.any() and fg_ratio > 1e-4:  # 前景が十分にある場合のみ
        # 前景があるサンプルのみでDice損失を計算
        fg_logits = logits[has_fg]
        fg_targets = targets[has_fg]
        
        if fg_logits.numel() > 0:
            dice = dice_loss_with_logits(fg_logits, fg_targets)
            # Dice損失が異常に大きい場合はクランプ
            dice = torch.clamp(dice, min=0.0, max=2.0)
        else:
            # 勾配が流れるよう、logitsから派生したゼロテンソルを作成
            dice = (logits * 0).mean()
    else:
        # 全バッチが空前景または前景が極端に少ない場合
        # BCEから派生させることで確実に勾配グラフに接続
        dice = bce * 0
    
    # 総合損失（重み付けを適用）
    # 損失の値が異常に大きくならないようクランプ
    loss = bce_weight * bce + dice_weight * dice
    loss = torch.clamp(loss, min=0.0, max=10.0)
    
    return loss, bce, dice


def compute_pos_weight(masks: torch.Tensor, min_ratio: float = 0.01, max_weight: float = 10.0) -> torch.Tensor:
    """
    正例の重みを計算（クラス不均衡対策）- 改善版
    極端な重みを防ぐため上限を設定
    
    Args:
        masks: バイナリマスク
        min_ratio: 最小比率（ゼロ除算回避）
        max_weight: 最大重み（極端な値を防ぐ）
    
    Returns:
        pos_weight tensor
    """
    # float型に変換してから計算（dtype問題を回避）
    masks_float = masks.float() if masks.dtype != torch.float32 else masks
    pos_frac = masks_float.mean().clamp(min=min_ratio).item()
    neg_frac = 1.0 - pos_frac
    
    # 重みを計算し、上限でクランプ
    weight = neg_frac / pos_frac
    weight = min(weight, max_weight)
    
    pos_weight = torch.tensor([weight], device=masks.device, dtype=torch.float32)
    return pos_weight