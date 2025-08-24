"""
座標系変換とメタデータ管理
SAM2.1の正しい前処理（アスペクト比保持+パディング）をサポート
"""
from dataclasses import dataclass
from typing import Tuple, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
import cv2


@dataclass
class SamSquareMeta:
    """SAM前処理のメタデータ
    
    SAMの標準的な前処理：
    1. アスペクト比を保持して最長辺を1024にリサイズ
    2. 右下にパディングして1024x1024の正方形にする
    3. このメタデータで逆変換を可能にする
    """
    raw_h: int  # 元画像の高さ
    raw_w: int  # 元画像の幅
    side: int = 1024  # SAMの入力サイズ（1024固定）
    scale: float = 1.0  # リサイズのスケール（side / max(raw_h, raw_w)）
    new_h: int = 0  # リサイズ後の高さ（パディング前）
    new_w: int = 0  # リサイズ後の幅（パディング前）
    pad_top: int = 0  # 上パディング（通常0）
    pad_left: int = 0  # 左パディング（通常0）
    pad_bottom: int = 0  # 下パディング
    pad_right: int = 0  # 右パディング
    
    @classmethod
    def from_image_size(cls, height: int, width: int, target_size: int = 1024) -> 'SamSquareMeta':
        """画像サイズからメタデータを作成"""
        scale = target_size / max(height, width)
        new_h = int(height * scale)
        new_w = int(width * scale)
        
        # 右下にパディング（SAM標準）
        pad_bottom = target_size - new_h
        pad_right = target_size - new_w
        
        return cls(
            raw_h=height,
            raw_w=width,
            side=target_size,
            scale=scale,
            new_h=new_h,
            new_w=new_w,
            pad_top=0,
            pad_left=0,
            pad_bottom=pad_bottom,
            pad_right=pad_right
        )


def apply_sam_transform_to_image(
    image: np.ndarray,
    meta: Optional[SamSquareMeta] = None,
    target_size: int = 1024
) -> Tuple[np.ndarray, SamSquareMeta]:
    """画像にSAMの標準的な前処理を適用
    
    Args:
        image: 入力画像 (H, W, C) or (H, W)
        meta: 既存のメタデータ（Noneの場合は作成）
        target_size: 目標サイズ（デフォルト1024）
    
    Returns:
        変換後の画像とメタデータ
    """
    h, w = image.shape[:2]
    
    if meta is None:
        meta = SamSquareMeta.from_image_size(h, w, target_size)
    
    # アスペクト比を保持してリサイズ
    resized = cv2.resize(image, (meta.new_w, meta.new_h), interpolation=cv2.INTER_LINEAR)
    
    # パディング（右下）
    if len(resized.shape) == 3:
        padded = np.pad(resized, 
                       ((meta.pad_top, meta.pad_bottom), 
                        (meta.pad_left, meta.pad_right), 
                        (0, 0)), 
                       mode='constant', constant_values=0)
    else:
        padded = np.pad(resized, 
                       ((meta.pad_top, meta.pad_bottom), 
                        (meta.pad_left, meta.pad_right)), 
                       mode='constant', constant_values=0)
    
    return padded, meta


def apply_sam_transform_to_mask(
    mask: np.ndarray,
    meta: SamSquareMeta,
    ignore_value: int = 255
) -> np.ndarray:
    """マスクにSAMの標準的な前処理を適用
    
    Args:
        mask: 入力マスク (H, W)
        meta: SAM変換のメタデータ
        ignore_value: パディング部分の値（デフォルト255）
    
    Returns:
        変換後のマスク
    """
    # アスペクト比を保持してリサイズ（最近傍補間）
    resized = cv2.resize(mask, (meta.new_w, meta.new_h), interpolation=cv2.INTER_NEAREST)
    
    # パディング（右下）
    padded = np.full((meta.side, meta.side), ignore_value, dtype=mask.dtype)
    padded[:meta.new_h, :meta.new_w] = resized
    
    return padded


def sam_square_to_raw_numpy(
    data: np.ndarray,
    meta: SamSquareMeta,
    interpolation: str = 'nearest'
) -> np.ndarray:
    """SAM座標系から元画像座標系への逆変換（NumPy版）
    
    Args:
        data: SAM座標系のデータ (1024, 1024) or (1024, 1024, C)
        meta: SAM変換のメタデータ
        interpolation: 補間方法 ('nearest' or 'linear')
    
    Returns:
        元画像座標系のデータ
    """
    # パディングを除去
    unpadded = data[:meta.new_h, :meta.new_w]
    
    # 元サイズにリサイズ
    interp = cv2.INTER_NEAREST if interpolation == 'nearest' else cv2.INTER_LINEAR
    restored = cv2.resize(unpadded, (meta.raw_w, meta.raw_h), interpolation=interp)
    
    return restored


def sam_square_to_raw_tensor(
    data: torch.Tensor,
    meta: SamSquareMeta,
    mode: str = 'nearest-exact'
) -> torch.Tensor:
    """SAM座標系から元画像座標系への逆変換（Tensor版）
    
    Args:
        data: SAM座標系のデータ (B, C, 1024, 1024) or (B, 1024, 1024)
        meta: SAM変換のメタデータ
        mode: 補間方法 ('nearest-exact', 'bilinear')
    
    Returns:
        元画像座標系のデータ
    """
    # 入力の次元を調整
    squeeze_last = False
    if data.dim() == 3:  # (B, H, W) -> (B, 1, H, W)
        data = data.unsqueeze(1)
        squeeze_last = True
    elif data.dim() == 2:  # (H, W) -> (1, 1, H, W)
        data = data.unsqueeze(0).unsqueeze(0)
        squeeze_last = True
    
    # パディングを除去
    unpadded = data[:, :, :meta.new_h, :meta.new_w]
    
    # 元サイズにリサイズ
    # PyTorchのnearestモードには既知の問題があるため、nearest-exactを使用
    if mode == 'nearest-exact':
        # PyTorch 2.0以降でサポート
        restored = F.interpolate(
            unpadded.float(),
            size=(meta.raw_h, meta.raw_w),
            mode='nearest-exact'
        )
    elif mode == 'nearest':
        # 互換性のためのフォールバック（推奨しない）
        restored = F.interpolate(
            unpadded.float(),
            size=(meta.raw_h, meta.raw_w),
            mode='nearest'
        )
    else:  # bilinear
        restored = F.interpolate(
            unpadded.float(),
            size=(meta.raw_h, meta.raw_w),
            mode='bilinear',
            align_corners=False
        )
    
    # 元の次元に戻す
    if squeeze_last:
        restored = restored.squeeze(1)
        if restored.shape[0] == 1:
            restored = restored.squeeze(0)
    
    return restored


def raw_to_sam_square_tensor(
    data: torch.Tensor,
    meta: SamSquareMeta,
    mode: str = 'nearest-exact',
    padding_value: float = 0.0
) -> torch.Tensor:
    """元画像座標系からSAM座標系への変換（Tensor版）
    
    Args:
        data: 元画像座標系のデータ (B, C, H, W) or (B, H, W)
        meta: SAM変換のメタデータ
        mode: 補間方法 ('nearest-exact', 'bilinear')
        padding_value: パディング値
    
    Returns:
        SAM座標系のデータ
    """
    # 入力の次元を調整
    squeeze_last = False
    if data.dim() == 3:  # (B, H, W) -> (B, 1, H, W)
        data = data.unsqueeze(1)
        squeeze_last = True
    elif data.dim() == 2:  # (H, W) -> (1, 1, H, W)
        data = data.unsqueeze(0).unsqueeze(0)
        squeeze_last = True
    
    # リサイズ
    if mode == 'nearest-exact':
        resized = F.interpolate(
            data.float(),
            size=(meta.new_h, meta.new_w),
            mode='nearest-exact'
        )
    elif mode == 'nearest':
        resized = F.interpolate(
            data.float(),
            size=(meta.new_h, meta.new_w),
            mode='nearest'
        )
    else:  # bilinear
        resized = F.interpolate(
            data.float(),
            size=(meta.new_h, meta.new_w),
            mode='bilinear',
            align_corners=False
        )
    
    # パディング（右下）
    padded = F.pad(
        resized,
        (meta.pad_left, meta.pad_right, meta.pad_top, meta.pad_bottom),
        mode='constant',
        value=padding_value
    )
    
    # 元の次元に戻す
    if squeeze_last:
        padded = padded.squeeze(1)
        if padded.shape[0] == 1:
            padded = padded.squeeze(0)
    
    return padded


# エイリアス（使いやすさのため）
def resize_mask_to_sam_square(
    mask: Union[np.ndarray, torch.Tensor],
    meta: SamSquareMeta,
    mode: str = 'nearest-exact'
) -> Union[np.ndarray, torch.Tensor]:
    """マスクをSAM座標系にリサイズ"""
    if isinstance(mask, torch.Tensor):
        return raw_to_sam_square_tensor(mask, meta, mode=mode, padding_value=255)
    else:
        return apply_sam_transform_to_mask(mask, meta, ignore_value=255)


def resize_mask_from_sam_square(
    mask: Union[np.ndarray, torch.Tensor],
    meta: SamSquareMeta,
    mode: str = 'nearest-exact'
) -> Union[np.ndarray, torch.Tensor]:
    """マスクを元画像座標系に戻す"""
    if isinstance(mask, torch.Tensor):
        return sam_square_to_raw_tensor(mask, meta, mode=mode)
    else:
        return sam_square_to_raw_numpy(mask, meta, interpolation='nearest')