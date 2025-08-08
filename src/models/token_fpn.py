"""
Token-FPN: Feature Pyramid Network for Vision Transformer (Qwen2.5-VL)
動的解像度対応のマルチスケール特徴抽出
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TokenFPN(nn.Module):
    """
    Token-based Feature Pyramid Network for Qwen2.5-VL
    
    ViTの中間層から特徴を抽出し、FPN構造でマルチスケール特徴を生成
    動的解像度（image_grid_thw）に対応
    """
    
    def __init__(
        self,
        in_channels: int = 1280,  # Qwen2.5-VL-3Bの中間層チャネル数
        out_channels: int = 256,   # SAM2.1のtransformer_dim
        layer_indices: List[int] = [8, 16, 24, 31],  # 抽出する層のインデックス
        sam_image_size: int = 1024,  # SAM2.1の期待する画像サイズ
    ):
        """
        Args:
            in_channels: Qwen中間層のチャネル数
            out_channels: 出力チャネル数（SAM互換の256）
            layer_indices: 特徴を抽出するブロックのインデックス
            sam_image_size: SAM2.1の入力画像サイズ
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer_indices = sorted(layer_indices)
        self.sam_image_size = sam_image_size
        
        # フックで取得した特徴を保存
        self.intermediate_features = {}
        self.hooks = []
        
        # 各層に対する1x1畳み込み（lateral connections）
        self.lateral_convs = nn.ModuleList()
        for _ in layer_indices:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
        
        # FPNの各レベルに対する3x3畳み込み（エイリアシング除去）
        self.fpn_convs = nn.ModuleList()
        for _ in layer_indices:
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU()
                )
            )
        
        # 高解像度特徴生成用のアップサンプリング
        # stride-8 feature (2x upsampling)
        self.upsample_s1 = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        # stride-4 feature (4x upsampling)
        self.upsample_s0 = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        # PatchMergeシミュレーション（2x2プーリング）
        self.patch_merge = nn.AvgPool2d(kernel_size=2, stride=2)
        
    def hook_fn(self, layer_name: str):
        """フック関数を作成"""
        def hook(module, input, output):
            # outputは通常 (hidden_states, attention_weights) のタプル
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            self.intermediate_features[layer_name] = hidden_states.detach()
        return hook
    
    def register_hooks(self, model):
        """Qwenモデルにフックを登録"""
        self.remove_hooks()  # 既存のフックをクリア
        
        # Qwen2.5-VLのビジョンモデルへアクセス
        if hasattr(model.model, 'visual'):
            vision_model = model.model.visual
        else:
            raise ValueError("ビジョンモジュールが見つかりません")
        
        # ViTブロックへアクセス
        if hasattr(vision_model, 'blocks'):
            blocks = vision_model.blocks
            logger.debug(f"Found {len(blocks)} vision blocks")
        else:
            raise ValueError("ViTブロックが見つかりません")
        
        # 指定した層にフックを登録
        for idx in self.layer_indices:
            if idx < len(blocks):
                hook = blocks[idx].register_forward_hook(
                    self.hook_fn(f"block_{idx}")
                )
                self.hooks.append(hook)
                logger.debug(f"Registered hook at block {idx}")
    
    def remove_hooks(self):
        """フックを削除"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.intermediate_features = {}
    
    def reshape_tokens_to_2d(
        self, 
        tokens: torch.Tensor, 
        image_grid_thw: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        トークンを2D形状に復元（動的解像度対応）
        
        Args:
            tokens: [B, N, C] or [N, C] 形状のトークン
            image_grid_thw: [B, 3] 形状の[T, H_raw, W_raw]情報
        
        Returns:
            [B, C, H, W] 形状の2D特徴マップ
        """
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)  # [N, C] -> [B, N, C]
        
        B, N, C = tokens.shape
        
        if image_grid_thw is not None:
            # 動的解像度モード
            H_raw = image_grid_thw[0, 1].item()
            W_raw = image_grid_thw[0, 2].item()
            
            # トークン数の確認
            expected_tokens = H_raw * W_raw
            if N != expected_tokens:
                logger.warning(f"Token count mismatch: got {N}, expected {expected_tokens}")
                # 最も近い正方形に調整
                H_raw = W_raw = int(torch.sqrt(torch.tensor(N, dtype=torch.float32)).item())
        else:
            # 固定解像度モード（正方形を仮定）
            H_raw = W_raw = int(torch.sqrt(torch.tensor(N, dtype=torch.float32)).item())
        
        # [B, N, C] -> [B, H, W, C] -> [B, C, H, W]
        features_2d = tokens.view(B, H_raw, W_raw, C).permute(0, 3, 1, 2)
        
        return features_2d
    
    def build_fpn_features(
        self, 
        features_dict: Dict[str, torch.Tensor],
        image_grid_thw: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        FPN構造で特徴を融合
        
        Args:
            features_dict: 各層の特徴辞書 {"block_i": [B, N, C]}
            image_grid_thw: 動的解像度情報
        
        Returns:
            FPN特徴辞書 {"P_i": [B, C, H, W]}
        """
        fpn_features = {}
        
        # 1. 各層の特徴を2D化して1x1 convで射影（lateral connections）
        laterals = []
        for i, (layer_name, tokens) in enumerate(features_dict.items()):
            # 2D形状に復元
            feat_2d = self.reshape_tokens_to_2d(tokens, image_grid_thw)
            
            # PatchMerge相当の処理（RAWからの圧縮）
            feat_2d = self.patch_merge(feat_2d)  # [B, C, H/2, W/2]
            
            # 1x1 convで256chに射影
            lateral = self.lateral_convs[i](feat_2d)
            laterals.append(lateral)
        
        # 2. Top-down pathway（上位層から下位層へ）
        # 最後の層から開始
        for i in range(len(laterals) - 1, -1, -1):
            if i == len(laterals) - 1:
                # 最上位層はそのまま使用
                fpn_feat = laterals[i]
            else:
                # 上位層をアップサンプルして加算
                prev_shape = laterals[i].shape[-2:]
                upsampled = F.interpolate(
                    fpn_features[f"P{i+1}"], 
                    size=prev_shape, 
                    mode='bilinear', 
                    align_corners=False
                )
                fpn_feat = laterals[i] + upsampled
            
            # 3x3 convでエイリアシング除去
            fpn_feat = self.fpn_convs[i](fpn_feat)
            fpn_features[f"P{i}"] = fpn_feat
        
        return fpn_features
    
    def forward(
        self, 
        qwen_features: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
        use_hooks: bool = True
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass
        
        Args:
            qwen_features: Qwenからの最終特徴 [B, C, H, W]（image_adapterの出力）
            image_grid_thw: 動的解像度情報 [B, 3]
            use_hooks: フックから中間特徴を使用するか
        
        Returns:
            (image_embeddings, high_res_features) のタプル
            - image_embeddings: [B, 256, H, W] SAM2.1用の基本埋め込み
            - high_res_features: [feat_s0, feat_s1] 高解像度特徴のリスト
        """
        
        if use_hooks and self.intermediate_features:
            # フックから取得した中間特徴を使用してFPNを構築
            fpn_features = self.build_fpn_features(
                self.intermediate_features, 
                image_grid_thw
            )
            
            # 最も解像度の高いP0を基本埋め込みとして使用
            if "P0" in fpn_features:
                base_features = fpn_features["P0"]
            else:
                # P0がない場合は最初の特徴を使用
                base_features = list(fpn_features.values())[0]
        else:
            # フックを使わない場合は入力特徴をそのまま使用
            base_features = qwen_features
        
        # SAM2.1互換のサイズに調整
        B, C, H, W = base_features.shape
        
        # SAMの期待するサイズ（1024x1024入力の場合、stride-16で64x64）
        target_size = self.sam_image_size // 16
        
        if H != target_size or W != target_size:
            # サイズ調整が必要
            image_embeddings = F.interpolate(
                base_features,
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=False
            )
        else:
            image_embeddings = base_features
        
        # 高解像度特徴の生成
        # stride-8 (2x upsampling)
        feat_s1 = self.upsample_s1(image_embeddings)
        
        # stride-4 (4x upsampling)
        feat_s0 = self.upsample_s0(image_embeddings)
        
        # SAM2.1のMaskDecoderが期待する順序で返す
        high_res_features = [feat_s0, feat_s1]
        
        return image_embeddings, high_res_features
    
    def __del__(self):
        """デストラクタでフックを確実に削除"""
        self.remove_hooks()