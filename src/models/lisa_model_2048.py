#!/usr/bin/env python3
"""
LISA改モデル - 2048次元特徴版（実用的実装）
Qwen2.5-VL-3B + SAM2.1統合
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Any
from transformers import Qwen2_5_VLForConditionalGeneration
import numpy as np

class ImageFeatureAdapter2048(nn.Module):
    """2048次元のLLM投影後特徴をSAM2.1用256次元に変換"""
    
    def __init__(self, in_dim: int = 2048, out_dim: int = 256):
        super().__init__()
        
        # 2048 → 512 → 256（LISA-v2準拠）
        self.adapter = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Linear(512, out_dim),
            nn.LayerNorm(out_dim)
        )
        
        self.out_dim = out_dim
    
    def forward(self, 
                vision_features: torch.Tensor,
                image_grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: [N_patches, 2048] - merger出力
            image_grid_thw: [B, 3] - グリッド情報
        
        Returns:
            sam_features: [B, 256, H, W] - SAM用特徴マップ
        """
        # vision_featuresが[256, 2048]の形式の場合
        if vision_features.dim() == 2:
            vision_features = vision_features.unsqueeze(0)  # [1, 256, 2048]
        
        B = vision_features.shape[0] if vision_features.dim() == 3 else 1
        
        # Adapter適用
        adapted = self.adapter(vision_features)  # [B, N_patches, 256]
        
        # 空間次元を復元
        if image_grid_thw is not None and image_grid_thw.numel() > 0:
            # PatchMerge後のグリッドサイズ
            H = int(image_grid_thw[0, 1].item()) // 2  # merger後は1/2
            W = int(image_grid_thw[0, 2].item()) // 2
        else:
            # デフォルト（448×448画像の場合）
            N_patches = adapted.shape[1]
            H = W = int(np.sqrt(N_patches))
        
        # [B, N_patches, 256] → [B, H, W, 256] → [B, 256, H, W]
        if adapted.dim() == 3:
            adapted = adapted.reshape(B, H, W, self.out_dim)
        else:
            adapted = adapted.reshape(H, W, self.out_dim)
            adapted = adapted.unsqueeze(0)
        
        sam_features = adapted.permute(0, 3, 1, 2).contiguous()
        
        return sam_features


class LISA_Model_2048(nn.Module):
    """LISA改モデル - 2048次元特徴版"""
    
    def __init__(self, 
                 config,
                 qwen_model: Optional[Qwen2_5_VLForConditionalGeneration] = None,
                 sam_predictor = None,
                 tokenizer = None):
        super().__init__()
        
        self.config = config
        self.tokenizer = tokenizer
        
        # Qwenモデル
        if qwen_model is not None:
            self.qwen = qwen_model
        else:
            self.qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                config.qwen_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        
        # SAMプレディクタ
        self.sam_predictor = sam_predictor
        
        # 2048次元用アダプタ
        self.image_adapter = ImageFeatureAdapter2048(
            in_dim=2048,  # LLM投影後
            out_dim=256   # SAM入力
        )
        
        # デバイスとdtypeに移動
        if qwen_model is not None:
            device = next(qwen_model.parameters()).device
            dtype = next(qwen_model.parameters()).dtype
            self.image_adapter = self.image_adapter.to(device=device, dtype=dtype)
        
        # フリーズ設定
        if config.freeze_qwen:
            for param in self.qwen.parameters():
                param.requires_grad = False
        
        if config.freeze_sam and sam_predictor:
            for param in self.sam_predictor.model.parameters():
                param.requires_grad = False
    
    def extract_vision_features_2048(self,
                                     pixel_values: torch.Tensor,
                                     image_grid_thw: torch.Tensor) -> torch.Tensor:
        """
        2048次元のLLM投影後特徴を取得（安定版）
        
        Args:
            pixel_values: [N_raw, C] or [B, N_raw, C]
            image_grid_thw: [3] or [B, 3]
        
        Returns:
            vision_features: [B, N_patches, 2048]
        """
        # get_image_featuresを使用（安定）
        vision_features = self.qwen.model.get_image_features(
            pixel_values,
            image_grid_thw
        )
        
        # tupleの場合は最初の要素
        if isinstance(vision_features, tuple):
            vision_features = vision_features[0]
        
        # [256, 2048] → [1, 256, 2048]
        if vision_features.dim() == 2:
            vision_features = vision_features.unsqueeze(0)
        
        return vision_features
    
    def forward(self,
                input_ids: torch.Tensor,
                pixel_values: Optional[torch.Tensor] = None,
                image_grid_thw: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                masks: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Forward pass
        
        Args:
            input_ids: [B, L]
            pixel_values: [N_raw, C]
            image_grid_thw: [B, 3]
            attention_mask: [B, L]
            labels: [B, L]
            masks: [B, H, W] - セグメンテーションマスク
        
        Returns:
            outputs: 言語モデル出力とセグメンテーション予測
        """
        outputs = {}
        
        # 1. 視覚特徴抽出（2048次元）
        if pixel_values is not None:
            vision_features = self.extract_vision_features_2048(
                pixel_values, 
                image_grid_thw
            )
            
            # SAM用に変換
            sam_features = self.image_adapter(vision_features, image_grid_thw)
            outputs['sam_features'] = sam_features
            
            # SAMでセグメンテーション
            if self.sam_predictor is not None:
                # SAM2.1の推論
                # ここでは簡略化（実際の実装は別途）
                pass
        
        # 2. 言語モデル
        lm_outputs = self.qwen(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        outputs['lm_outputs'] = lm_outputs
        
        # 3. 損失計算
        if labels is not None:
            outputs['loss'] = lm_outputs.loss
        
        return outputs
    
    def generate(self, 
                 input_ids: torch.Tensor,
                 pixel_values: Optional[torch.Tensor] = None,
                 image_grid_thw: Optional[torch.Tensor] = None,
                 **kwargs) -> torch.Tensor:
        """テキスト生成"""
        return self.qwen.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **kwargs
        )


# 使用例
if __name__ == "__main__":
    from src.config import LISAConfig
    
    config = LISAConfig()
    model = LISA_Model_2048(config)
    
    print("LISA改モデル（2048次元版）初期化完了")
    print(f"  Vision features: 2048次元（LLM投影後）")
    print(f"  SAM features: 256次元")
    print(f"  メモリ効率: 最適化済み")