"""
Token-FPN: Feature Pyramid Network for Vision Transformer (Qwen2.5-VL)
動的解像度対応のマルチスケール特徴抽出
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
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
        image_grid_thw: Optional[torch.Tensor] = None,
        spatial_merge_size: int = 2
    ) -> torch.Tensor:
        """
        トークンを2D形状に復元（動的解像度対応）
        
        重要: フックから取得した特徴はPatchMerge前なので、
        ここではPatchMerge前のグリッドサイズ（H_raw, W_raw）をそのまま使用する
        
        Args:
            tokens: [B*N, C] or [N, C] 形状のトークン（フックから取得、PatchMerge前）
            image_grid_thw: [B, 3] 形状の[T, H_raw, W_raw]情報（PatchMerge前のグリッド）
            spatial_merge_size: 使用しない（フックの特徴は既にPatchMerge前）
        
        Returns:
            [B, C, H, W] 形状の2D特徴マップ
        """
        # トークンの次元を確認
        if tokens.dim() == 2:
            # [N_total, C] 形状（バッチ全体が連結されている）
            N_total, C = tokens.shape
            
            if image_grid_thw is not None:
                B = image_grid_thw.shape[0]
                
                # 各画像のトークン数を計算（PatchMerge前のサイズ）
                token_counts = []
                grids = []
                for b in range(B):
                    T = image_grid_thw[b, 0].item()
                    H_raw = image_grid_thw[b, 1].item()  # PatchMerge前
                    W_raw = image_grid_thw[b, 2].item()  # PatchMerge前
                    
                    # フックの特徴はPatchMerge前なので、そのままのサイズを使用
                    n_tokens = T * H_raw * W_raw
                    token_counts.append(n_tokens)
                    grids.append((T, H_raw, W_raw))
                
                # トークンの総数が一致するか確認
                expected_total = sum(token_counts)
                if N_total == expected_total:
                    # バッチ全体のトークンが連結されている場合
                    # 各画像ごとに分割して処理
                    features_2d_list = []
                    offset = 0
                    
                    for b in range(B):
                        n_tokens = token_counts[b]
                        tokens_b = tokens[offset:offset + n_tokens]  # [n_tokens, C]
                        
                        T, H_raw, W_raw = grids[b]
                        
                        if T > 1:
                            # 動画の場合
                            tokens_reshaped = tokens_b.view(T, H_raw, W_raw, C)
                            tokens_spatial = tokens_reshaped.mean(dim=0)  # 時間平均
                        else:
                            # 静止画の場合
                            tokens_spatial = tokens_b.view(H_raw, W_raw, C)
                        
                        # [H, W, C] -> [C, H, W]
                        feat_2d = tokens_spatial.permute(2, 0, 1).contiguous()
                        features_2d_list.append(feat_2d)
                        
                        offset += n_tokens
                    
                    # バッチ次元でスタック
                    return torch.stack(features_2d_list, dim=0)  # [B, C, H, W]
                    
                else:
                    # トークン数が合わない場合のフォールバック
                    logger.error(
                        f"Token count mismatch! Total tokens: {N_total}, "
                        f"Expected: {expected_total} (sum of {token_counts})"
                    )
                    # 単一画像として扱う
                    B = 1
                    H = W = int(torch.sqrt(torch.tensor(N_total, dtype=torch.float32)).item())
                    tokens = tokens.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                    return tokens
            else:
                # image_grid_thwがない場合は正方形を仮定
                H = W = int(torch.sqrt(torch.tensor(N_total, dtype=torch.float32)).item())
                tokens = tokens.view(1, H, W, C).permute(0, 3, 1, 2).contiguous()
                return tokens
        
        # 3次元の場合（通常はここには来ない）
        elif tokens.dim() == 3:
            B, N, C = tokens.shape
            
            if image_grid_thw is not None and B == image_grid_thw.shape[0]:
                features_2d_list = []
                
                for b in range(B):
                    T = image_grid_thw[b, 0].item()
                    H_raw = image_grid_thw[b, 1].item()
                    W_raw = image_grid_thw[b, 2].item()
                    
                    expected_tokens = T * H_raw * W_raw
                    
                    if N != expected_tokens:
                        logger.error(
                            f"Image {b}: Token count mismatch! Got {N} tokens, expected {expected_tokens}"
                        )
                        # フォールバック
                        size = int(torch.sqrt(torch.tensor(N // T, dtype=torch.float32)).item())
                        H_raw = W_raw = size
                    
                    tokens_b = tokens[b]  # [N, C]
                    
                    if T > 1:
                        tokens_reshaped = tokens_b.view(T, H_raw, W_raw, C)
                        tokens_spatial = tokens_reshaped.mean(dim=0)
                    else:
                        tokens_spatial = tokens_b.view(H_raw, W_raw, C)
                    
                    feat_2d = tokens_spatial.permute(2, 0, 1).contiguous()
                    features_2d_list.append(feat_2d)
                
                return torch.stack(features_2d_list, dim=0)
            else:
                # 固定解像度モード
                H = W = int(torch.sqrt(torch.tensor(N, dtype=torch.float32)).item())
                features_2d_list = []
                for b in range(B):
                    tokens_b = tokens[b]
                    tokens_spatial = tokens_b.view(H, W, C)
                    feat_2d = tokens_spatial.permute(2, 0, 1).contiguous()
                    features_2d_list.append(feat_2d)
                return torch.stack(features_2d_list, dim=0)
        
        raise ValueError(f"Unexpected token shape: {tokens.shape}")
    
    def build_fpn_features(
        self, 
        intermediate_features: Union[List[torch.Tensor], Dict[str, torch.Tensor]],
        image_grid_thw: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        中間特徴からFPNピラミッドを構築
        
        Args:
            intermediate_features: 異なる層からの特徴リストまたは辞書
            image_grid_thw: [B, 3] 形状のグリッド情報（マージ前）
        
        Returns:
            ピラミッド特徴の辞書 {"P0": ..., "P1": ..., ...}
        """
        fpn_features = {}
        
        # 辞書型の場合はリストに変換
        if isinstance(intermediate_features, dict):
            # block_8, block_16などのキーから特徴を取り出す（フックで登録した名前と一致）
            feature_list = []
            for layer_idx in self.layer_indices:
                block_key = f"block_{layer_idx}"
                if block_key in intermediate_features:
                    feat = intermediate_features[block_key]
                    feature_list.append(feat)
                else:
                    logger.warning(f"Feature for {block_key} not found in intermediate_features")
            intermediate_features = feature_list
        
        # 特徴が取得できていない場合はエラー
        if not intermediate_features:
            raise ValueError("No intermediate features found. Check hook registration.")
        
        for i, tokens in enumerate(intermediate_features):
            # tokenがテンソルであることを確認
            if not isinstance(tokens, torch.Tensor):
                logger.error(f"Feature {i} is not a tensor: type={type(tokens)}")
                continue
                
            # PatchMerge前のトークンを2D形状に復元
            # フックから取得した特徴は既にPatchMerge前なので、spatial_merge_sizeは使用しない
            feat_2d = self.reshape_tokens_to_2d(
                tokens, 
                image_grid_thw,
                spatial_merge_size=2  # 実際には使用されない
            )
            
            # チャンネル次元を統一
            B, C, H, W = feat_2d.shape
            
            if C != self.out_channels:
                # lateral convolutionを使用
                if i < len(self.lateral_convs):
                    feat_2d = self.lateral_convs[i](feat_2d)
                else:
                    # 動的に作成（必要に応じて）
                    lateral = nn.Conv2d(C, self.out_channels, 1).to(feat_2d.device)
                    feat_2d = lateral(feat_2d)
            
            # FPN convolutionを適用
            if i < len(self.fpn_convs):
                feat_2d = self.fpn_convs[i](feat_2d)
            
            # ピラミッドレベルを割り当て
            # 最も解像度の高いものをP0とする
            level_name = f"P{i}"
            fpn_features[level_name] = feat_2d
        
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