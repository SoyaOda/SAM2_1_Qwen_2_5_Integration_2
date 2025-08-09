# LISA改 Segmentation Loss改善のための詳細実装計画

## 実行日: 2025年1月7日
## 目標: Seg Loss 1.3 → 0.6以下への段階的改善

---

## 📊 現状分析サマリー

### 現在の問題点
1. **Seg Loss 1.3で停滞**（期待値: 0.35-0.6）
2. **動的解像度が実際には機能していない可能性**（固定448×448で動作）
3. **Qwen2.5-VL（言語寄り）とSAM2.1（ピクセル精度）の特徴分布ギャップ**
4. **Cross-Attentionなし**で単純な線形変換のみ
5. **SAM2.1 MaskDecoder完全凍結**で適応性なし

### 現在の実装構造
```
Qwen ViT (1280dim) 
    ↓ [フック取得]
純粋視覚特徴 (1280dim)
    ↓ [ImageFeatureAdapter: Linear + LayerNorm]
SAM用特徴 (256dim)
    ↓
SAM2.1 MaskDecoder（完全凍結）
```

---

## 🎯 Phase 1: 即効性のある改善（3日間）

### 1.1 動的解像度の動作確認と修正

#### ステップ1: 現状の動作確認（Day 1 AM）

**ファイル**: `minimal_train.py`
**行番号**: 327行目付近（train_epochメソッド内）

```python
# 既存のデバッグコードの後に追加
if batch_idx % 50 == 0:  # 50バッチごとに詳細ログ
    # 動的解像度の実際の動作を確認
    if 'image_grid_thw' in batch and batch['image_grid_thw'] is not None:
        grid_thw = batch['image_grid_thw']
        H_grids = grid_thw[:, 1].float()
        W_grids = grid_thw[:, 2].float()
        
        # 統計情報を計算
        H_mean = H_grids.mean().item()
        H_std = H_grids.std().item()
        W_mean = W_grids.mean().item()
        W_std = W_grids.std().item()
        
        # パッチ数から実際の解像度を逆算
        actual_H = H_mean * 14  # パッチサイズ14
        actual_W = W_mean * 14
        
        logger.info(f"[動的解像度チェック] Grid: H={H_mean:.1f}±{H_std:.1f}, W={W_mean:.1f}±{W_std:.1f}")
        logger.info(f"[動的解像度チェック] 推定解像度: {actual_H:.0f}×{actual_W:.0f}px")
        
        # 固定値チェック
        if H_std < 0.1 and W_std < 0.1:
            logger.warning("⚠️ グリッドサイズが固定値の可能性があります！")
        
        # WandBログ（使用している場合）
        if self.config.use_wandb:
            wandb.log({
                "dynamic_resolution/grid_H_mean": H_mean,
                "dynamic_resolution/grid_H_std": H_std,
                "dynamic_resolution/grid_W_mean": W_mean,
                "dynamic_resolution/grid_W_std": W_std,
                "dynamic_resolution/estimated_resolution": actual_H,
                "train/step": self.global_step
            })
```

#### ステップ2: 動的解像度の修正（Day 1 PM）

**ファイル**: `src/data/dataset.py`
**行番号**: 690-721行目（HybridDataset.__getitem__内）

```python
# 既存のコード（690行目付近）を以下に置き換え
# apply_chat_templateで画像とテキストを処理
qwen_processed = self.qwen_processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
)

input_ids = qwen_processed['input_ids'].squeeze(0)
attention_mask = qwen_processed['attention_mask'].squeeze(0)
pixel_values = qwen_processed['pixel_values'].squeeze(0)  # 処理済み画像

# 動的解像度モード: processorが自動計算したimage_grid_thwを使用
if self.use_dynamic_resolution:
    # processorが計算したグリッドサイズを取得
    if 'image_grid_thw' in qwen_processed and qwen_processed['image_grid_thw'] is not None:
        image_grid_thw = qwen_processed['image_grid_thw'].squeeze(0)
        
        # デバッグ: 実際のグリッドサイズを確認
        if hasattr(self, 'debug_counter'):
            self.debug_counter += 1
        else:
            self.debug_counter = 0
        
        if self.debug_counter % 100 == 0:  # 100サンプルごとにログ
            _, h, w = pixel_values.shape
            H_grid, W_grid = image_grid_thw[1].item(), image_grid_thw[2].item()
            print(f"[Dataset] 動的解像度: 画像 {h}×{w}px → グリッド {H_grid}×{W_grid}")
    else:
        # フォールバック: pixel_valuesから計算
        _, h, w = pixel_values.shape
        H_grid = h // 14
        W_grid = w // 14
        image_grid_thw = torch.tensor([1, H_grid, W_grid], dtype=torch.long)
        print(f"[Dataset] 警告: image_grid_thw自動計算 {H_grid}×{W_grid}")
else:
    # 固定解像度モード
    image_grid_thw = torch.tensor([1, 32, 32], dtype=torch.long)  # 448×448用
```

### 1.2 EdgeLoss実装（Day 2 AM）

**ファイル**: `src/models/lisa_model.py`
**行番号**: 580行目付近（ファイル末尾に追加）

```python
def compute_edge_loss(pred_masks: torch.Tensor, gt_masks: torch.Tensor, 
                     edge_weight: float = 0.25) -> torch.Tensor:
    """
    エッジ損失の計算（LISA-v2準拠）
    
    Args:
        pred_masks: 予測マスク [B, 1, H, W] (logits)
        gt_masks: 正解マスク [B, 1, H, W] (0/1)
        edge_weight: エッジ損失の重み係数
    
    Returns:
        エッジ損失値
    """
    # Sobelフィルタの定義
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32, device=pred_masks.device)
    sobel_y = torch.tensor([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=torch.float32, device=pred_masks.device)
    
    # フィルタを適切な形状に変換
    sobel_x = sobel_x.view(1, 1, 3, 3)
    sobel_y = sobel_y.view(1, 1, 3, 3)
    
    # 予測マスクをシグモイド活性化
    pred_sigmoid = torch.sigmoid(pred_masks)
    
    # エッジ検出（予測マスク）
    pred_edge_x = F.conv2d(pred_sigmoid, sobel_x, padding=1)
    pred_edge_y = F.conv2d(pred_sigmoid, sobel_y, padding=1)
    pred_edges = torch.sqrt(pred_edge_x**2 + pred_edge_y**2 + 1e-8)
    
    # エッジ検出（正解マスク）
    gt_float = gt_masks.float()
    gt_edge_x = F.conv2d(gt_float, sobel_x, padding=1)
    gt_edge_y = F.conv2d(gt_float, sobel_y, padding=1)
    gt_edges = torch.sqrt(gt_edge_x**2 + gt_edge_y**2 + 1e-8)
    
    # L1損失
    edge_loss = F.l1_loss(pred_edges, gt_edges)
    
    return edge_loss * edge_weight
```

**ファイル**: `minimal_train.py`
**行番号**: 301行目付近（compute_lossメソッド内）

```python
# seg_loss計算部分（301行目付近）を以下に修正
seg_loss += bce_loss + dice_loss

# EdgeLoss追加
if self.config.use_edge_loss:  # 設定で有効化
    from src.models.lisa_model import compute_edge_loss
    edge_loss = compute_edge_loss(
        pred_mask.unsqueeze(0).unsqueeze(0),  # [B=1, C=1, H, W]
        gt_mask_resized.unsqueeze(0).unsqueeze(0),
        edge_weight=self.config.edge_loss_weight
    )
    seg_loss += edge_loss
    
    # デバッグログ
    if batch_idx % 100 == 0:
        logger.debug(f"EdgeLoss: {edge_loss.item():.4f}")
```

### 1.3 学習率の最適化（Day 2 PM）

**ファイル**: `minimal_train.py`
**行番号**: 80-90行目（setup_model_and_data内）、594-600行目（argparse）

```python
# コマンドライン引数のデフォルト値を変更（594-600行目）
parser.add_argument('--adapter_lr', type=float, default=1.0e-4,  # 1.5e-4 → 1.0e-4
                   help='アダプター学習率')
parser.add_argument('--lora_lr', type=float, default=5e-5,  # 1.5e-4 → 5e-5
                   help='LoRA学習率')
parser.add_argument('--seg_token_lr', type=float, default=2.5e-5,  # 5e-5 → 2.5e-5
                   help='SEGトークン学習率')

# EdgeLoss設定を追加
parser.add_argument('--use_edge_loss', action='store_true',
                   help='EdgeLossを使用')
parser.add_argument('--edge_loss_weight', type=float, default=0.25,
                   help='EdgeLoss係数')
```

### 1.4 検証スクリプト（Day 3）

**新規ファイル**: `validate_improvements.py`

```python
#!/usr/bin/env python3
"""
Phase 1改善の効果を検証するスクリプト
"""
import torch
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt

def analyze_training_logs(log_dir: Path):
    """学習ログから改善効果を分析"""
    
    # loss_history.jsonを読み込み
    with open(log_dir / "loss_history.json", 'r') as f:
        history = json.load(f)
    
    # 改善前後の比較
    steps = np.array(history['steps'])
    seg_loss = np.array(history['seg_loss'])
    
    # 移動平均でスムージング
    window = 100
    seg_loss_smooth = np.convolve(seg_loss, np.ones(window)/window, mode='valid')
    steps_smooth = steps[:len(seg_loss_smooth)]
    
    # 改善ポイントを特定
    improvement_points = []
    
    # Phase 1開始点（仮に10000ステップ）
    phase1_start = 10000
    if phase1_start in steps:
        idx = np.where(steps == phase1_start)[0][0]
        before_loss = np.mean(seg_loss[max(0, idx-500):idx])
        after_loss = np.mean(seg_loss[idx:min(len(seg_loss), idx+500)])
        improvement = before_loss - after_loss
        improvement_points.append({
            'phase': 'Phase 1',
            'step': phase1_start,
            'before': before_loss,
            'after': after_loss,
            'improvement': improvement,
            'percentage': (improvement / before_loss) * 100
        })
    
    # 結果を出力
    print("="*60)
    print("📊 改善効果分析レポート")
    print("="*60)
    
    for point in improvement_points:
        print(f"\n{point['phase']} (Step {point['step']}):")
        print(f"  改善前: {point['before']:.4f}")
        print(f"  改善後: {point['after']:.4f}")
        print(f"  改善量: {point['improvement']:.4f} ({point['percentage']:.1f}%)")
    
    # 最終的な損失
    final_loss = np.mean(seg_loss[-100:])
    print(f"\n最終Seg Loss: {final_loss:.4f}")
    
    if final_loss < 0.6:
        print("✅ 目標達成！LISA相当の性能に到達")
    elif final_loss < 0.9:
        print("⚠️ 改善は見られるが、追加の最適化が必要")
    else:
        print("❌ Phase 2以降の実装が必要")
    
    # グラフ生成
    plt.figure(figsize=(12, 6))
    plt.plot(steps_smooth, seg_loss_smooth, 'b-', alpha=0.7, label='Seg Loss')
    
    # 改善ポイントをマーク
    for point in improvement_points:
        plt.axvline(x=point['step'], color='r', linestyle='--', alpha=0.5)
        plt.text(point['step'], plt.ylim()[1]*0.9, point['phase'], rotation=90)
    
    plt.xlabel('Steps')
    plt.ylabel('Segmentation Loss')
    plt.title('Segmentation Loss改善の推移')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(log_dir / 'improvement_analysis.png', dpi=150)
    print(f"\nグラフを保存: {log_dir / 'improvement_analysis.png'}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, required=True,
                       help='ログディレクトリのパス')
    args = parser.parse_args()
    
    analyze_training_logs(Path(args.log_dir))
```

---

## 🔧 Phase 2: 構造的改善（1週間）

### 2.1 SAM2.1 MaskDecoderへのLoRA追加（Day 4-5）

**ファイル**: `src/models/lisa_model.py`
**行番号**: 120-130行目付近（__init__メソッド内）

```python
# SAM2.1のMaskDecoder取得後に追加（120行目付近）
# Freeze SAM mask decoder parameters by default
for param in self.sam_mask_decoder.parameters():
    param.requires_grad = False

# SAM MaskDecoderへのLoRA適用（新規追加）
if config.use_sam_lora:
    from peft import LoraConfig, get_peft_model, TaskType
    
    # SAM2.1 MaskDecoder用のLoRA設定
    sam_lora_config = LoraConfig(
        r=4,  # rank 4（軽量）
        lora_alpha=16,  # スケーリング係数
        target_modules=[
            "transformer.self_attn.q_proj",
            "transformer.self_attn.k_proj", 
            "transformer.self_attn.v_proj",
            "transformer.cross_attn_token_to_image.q_proj",
            "transformer.cross_attn_token_to_image.k_proj",
            "transformer.cross_attn_token_to_image.v_proj",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,  # SAMは特徴抽出タスク
    )
    
    # LoRAを適用
    self.sam_mask_decoder = get_peft_model(self.sam_mask_decoder, sam_lora_config)
    
    # LoRA部分のみ学習可能に
    for name, param in self.sam_mask_decoder.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
            
    logger.info(f"SAM MaskDecoderにLoRAを適用しました（r={sam_lora_config.r}）")
    
    # パラメータ数の確認
    sam_lora_params = sum(p.numel() for n, p in self.sam_mask_decoder.named_parameters() 
                          if p.requires_grad and "lora" in n.lower())
    logger.info(f"SAM LoRAパラメータ数: {sam_lora_params:,}")
```

**ファイル**: `src/config.py`
**行番号**: 35-40行目付近

```python
# Training configuration に追加
freeze_qwen: bool = True
freeze_sam: bool = True
use_sam_lora: bool = False  # 新規追加：SAM MaskDecoderへのLoRA
sam_lora_r: int = 4  # 新規追加：LoRAのrank
sam_lora_alpha: int = 16  # 新規追加：LoRAのalpha
```

### 2.2 Cross-Attention融合層（Day 6-7）

**新規ファイル**: `src/models/fusion_layers.py`

```python
"""
Vision-Language特徴の深い融合のためのモジュール
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class CrossAttentionFusion(nn.Module):
    """
    テキスト特徴と視覚特徴をCross-Attentionで融合
    Otter-SAMのablation studyで2-3ポイントの改善が確認された手法
    """
    
    def __init__(self, 
                 dim: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 use_residual: bool = True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.use_residual = use_residual
        
        # Multi-head Cross Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
        # Optional: ゲート機構で融合の強度を制御
        self.fusion_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
    
    def forward(self, 
                text_embed: torch.Tensor,
                vision_features: torch.Tensor,
                vision_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            text_embed: [B, D] or [B, 1, D] - SEGトークンの隠れ状態
            vision_features: [B, N_patches, D] - 視覚特徴
            vision_mask: [B, N_patches] - 有効なパッチのマスク（オプション）
        
        Returns:
            融合された特徴 [B, D]
        """
        # text_embedを[B, 1, D]形状に統一
        if text_embed.dim() == 2:
            text_embed = text_embed.unsqueeze(1)  # [B, D] -> [B, 1, D]
        
        B, _, D = text_embed.shape
        
        # Cross-attention: textがquery、visionがkey/value
        # これにより、テキストが視覚情報のどこに注目すべきかを学習
        attended_text, attn_weights = self.cross_attn(
            query=text_embed,  # [B, 1, D]
            key=vision_features,  # [B, N_patches, D]
            value=vision_features,  # [B, N_patches, D]
            key_padding_mask=vision_mask  # マスクがあれば適用
        )
        
        # Residual connection + LayerNorm
        if self.use_residual:
            text_fused = self.norm1(attended_text + text_embed)
        else:
            text_fused = self.norm1(attended_text)
        
        # Feed-forward network
        text_out = self.ffn(text_fused)
        text_out = self.norm2(text_out + text_fused)  # [B, 1, D]
        
        # ゲート機構：元のtext_embedとの融合度合いを制御
        text_concat = torch.cat([text_embed, text_out], dim=-1)  # [B, 1, D*2]
        gate = self.fusion_gate(text_concat)  # [B, 1, D]
        
        # ゲートを適用した最終出力
        final_output = gate * text_out + (1 - gate) * text_embed
        final_output = self.norm3(final_output)
        
        return final_output.squeeze(1)  # [B, D]

class BidirectionalFusion(nn.Module):
    """
    双方向の特徴融合（Vision→Text、Text→Vision）
    より深い相互作用を実現
    """
    
    def __init__(self, 
                 dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        
        # 複数層のCross-Attention
        self.text_to_vision_layers = nn.ModuleList([
            CrossAttentionFusion(dim, num_heads) 
            for _ in range(num_layers)
        ])
        
        self.vision_to_text_layers = nn.ModuleList([
            nn.MultiheadAttention(dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(num_layers * 2)
        ])
    
    def forward(self, 
                text_embed: torch.Tensor,
                vision_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        双方向融合
        
        Returns:
            (融合されたtext特徴, 融合されたvision特徴)
        """
        for i in range(self.num_layers):
            # Text → Vision
            text_embed = self.text_to_vision_layers[i](text_embed, vision_features)
            
            # Vision → Text（vision特徴も更新）
            vision_attended, _ = self.vision_to_text_layers[i](
                vision_features,
                text_embed.unsqueeze(1).expand(-1, vision_features.size(1), -1),
                text_embed.unsqueeze(1).expand(-1, vision_features.size(1), -1)
            )
            vision_features = self.norm_layers[i*2](vision_features + vision_attended)
        
        return text_embed, vision_features
```

**ファイル**: `src/models/lisa_model.py`
**行番号**: 140行目付近（__init__内でtext_prompt_projの後）

```python
# Text prompt projectorの後に追加
self.text_prompt_proj = TextPromptProjector(
    in_dim=config.qwen_hidden_size,
    out_dim=config.text_prompt_out_dim,
    use_mlp=True
).to(dtype=model_dtype)

# Cross-Attention融合層を追加（新規）
if config.use_cross_attention_fusion:
    from src.models.fusion_layers import CrossAttentionFusion
    self.cross_attention_fusion = CrossAttentionFusion(
        dim=config.text_prompt_out_dim,
        num_heads=8,
        dropout=0.1
    ).to(dtype=model_dtype)
    logger.info("Cross-Attention融合層を追加しました")
else:
    self.cross_attention_fusion = None
```

**ファイル**: `src/models/lisa_model.py`
**行番号**: 470行目付近（forward内、prompt_embeds生成後）

```python
# Project to prompt embeddings（既存コード）
if len(seg_positions[i]) == 1:
    prompt_embeds = self.text_prompt_proj(seg_hidden_states)  # [256]
else:
    prompt_embeds = self.text_prompt_proj(seg_hidden_states)  # [num_segs, 256]

# Cross-Attention融合を適用（新規追加）
if self.cross_attention_fusion is not None and vision_features is not None:
    # vision_featuresを256次元に変換済みのものを使用
    vision_features_256 = self.image_adapter(vision_features[i:i+1], image_grid_thw)
    # [B, 256, H, W] -> [B, H*W, 256]
    B_v, C_v, H_v, W_v = vision_features_256.shape
    vision_features_flat = vision_features_256.view(B_v, C_v, H_v*W_v).transpose(1, 2)
    
    # Cross-Attention融合
    if len(seg_positions[i]) == 1:
        prompt_embeds = self.cross_attention_fusion(
            prompt_embeds,
            vision_features_flat[0]  # [H*W, 256]
        )
    else:
        # 複数SEGトークンの場合
        fused_embeds = []
        for emb in prompt_embeds:
            fused_emb = self.cross_attention_fusion(
                emb,
                vision_features_flat[0]
            )
            fused_embeds.append(fused_emb)
        prompt_embeds = torch.stack(fused_embeds)
```

---

## 🚀 Phase 3: 動的解像度の完全活用（3日間）

### 3.1 解像度適応的なバッチ処理（Day 8）

**ファイル**: `src/data/collators.py`
**行番号**: 340行目付近（MultiModalDataCollator.__call__の最後）

```python
# 動的解像度に応じたバッチ最適化を追加
def _optimize_batch_for_resolution(self, batch: Dict) -> Dict:
    """
    動的解像度に応じてバッチを最適化
    高解像度画像が多い場合は警告を出す
    """
    if 'image_grid_thw' in batch and batch['image_grid_thw'] is not None:
        grid_thw = batch['image_grid_thw']
        total_patches = (grid_thw[:, 1] * grid_thw[:, 2]).sum().item()
        avg_patches = total_patches / len(grid_thw)
        
        # メモリ使用量の推定
        estimated_memory_gb = (total_patches * 256 * 4) / (1024**3)  # 概算
        
        if avg_patches > 1024:  # 32×32以上
            logger.warning(f"⚠️ 高解像度バッチ検出: 平均{avg_patches:.0f}パッチ")
            logger.warning(f"推定メモリ使用量: {estimated_memory_gb:.2f}GB")
            
            # 必要に応じてバッチを分割する処理を追加可能
            if estimated_memory_gb > 8.0:  # 8GB以上の場合
                logger.error("メモリ不足の可能性があります！バッチサイズを減らしてください")
    
    return batch

# __call__メソッドの最後で呼び出し
collated = self._optimize_batch_for_resolution(collated)
```

### 3.2 動的Gradient Accumulation（Day 9）

**ファイル**: `minimal_train.py`
**行番号**: 320行目付近（train_epoch内）

```python
# Gradient Accumulation用の変数（既存）
accumulation_steps = self.config.gradient_accumulation_steps

# 動的調整を追加（新規）
if self.config.use_dynamic_grad_accum:
    # バッチ内の平均グリッドサイズに基づいて調整
    if 'image_grid_thw' in batch and batch['image_grid_thw'] is not None:
        avg_grid = batch['image_grid_thw'][:, 1:].float().mean().item()
        
        if avg_grid < 24:  # 小さい画像（336px相当）
            accumulation_steps = max(4, self.config.gradient_accumulation_steps // 2)
        elif avg_grid < 36:  # 中サイズ（504px相当）
            accumulation_steps = self.config.gradient_accumulation_steps
        else:  # 大きい画像（504px以上）
            accumulation_steps = min(32, self.config.gradient_accumulation_steps * 2)
        
        # 初回または変更時のみログ
        if batch_idx == 0 or (batch_idx % 100 == 0 and self.config.debug):
            logger.debug(f"動的Grad Accum: グリッド{avg_grid:.1f} → accumulation_steps={accumulation_steps}")
```

### 3.3 解像度別の性能分析（Day 10）

**新規ファイル**: `analyze_resolution_performance.py`

```python
#!/usr/bin/env python3
"""
解像度別のセグメンテーション性能を分析
"""
import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_by_resolution(model, dataloader, device='cuda'):
    """
    解像度別にモデル性能を分析
    """
    resolution_buckets = defaultdict(list)
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            # バッチをデバイスに移動
            batch = {k: v.to(device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # モデル推論
            outputs = model(
                input_ids=batch['input_ids'],
                pixel_values=batch['pixel_values'],
                attention_mask=batch['attention_mask'],
                image_grid_thw=batch.get('image_grid_thw'),
            )
            
            # 解像度別に分類
            if 'image_grid_thw' in batch and batch['image_grid_thw'] is not None:
                for i in range(len(batch['image_grid_thw'])):
                    H_grid = batch['image_grid_thw'][i, 1].item()
                    W_grid = batch['image_grid_thw'][i, 2].item()
                    resolution = H_grid * W_grid * 14 * 14  # 実際のピクセル数
                    
                    # Dice scoreを計算（簡易版）
                    if outputs.mask_logits[i] is not None and batch['mask_labels'][i] is not None:
                        pred_mask = torch.sigmoid(outputs.mask_logits[i][0])
                        gt_mask = batch['mask_labels'][i]
                        
                        intersection = (pred_mask * gt_mask).sum()
                        dice = 2 * intersection / (pred_mask.sum() + gt_mask.sum() + 1e-8)
                        
                        # 解像度カテゴリに分類
                        if resolution < 250000:  # ~500×500
                            bucket = "Low (≤500px)"
                        elif resolution < 500000:  # 500-700px
                            bucket = "Medium (500-700px)"
                        else:  # 700px+
                            bucket = "High (≥700px)"
                        
                        resolution_buckets[bucket].append(dice.item())
    
    # 統計を計算
    stats = {}
    for bucket, scores in resolution_buckets.items():
        stats[bucket] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'count': len(scores)
        }
    
    # 結果を表示
    print("="*60)
    print("📊 解像度別セグメンテーション性能")
    print("="*60)
    
    for bucket in ["Low (≤500px)", "Medium (500-700px)", "High (≥700px)"]:
        if bucket in stats:
            s = stats[bucket]
            print(f"\n{bucket}:")
            print(f"  Dice Score: {s['mean']:.4f} ± {s['std']:.4f}")
            print(f"  サンプル数: {s['count']}")
    
    # 改善提案
    print("\n" + "="*60)
    print("💡 改善提案")
    print("="*60)
    
    if stats.get("High (≥700px)", {}).get('mean', 0) < stats.get("Low (≤500px)", {}).get('mean', 1):
        print("⚠️ 高解像度での性能が低い")
        print("  → 高解像度特徴の品質改善が必要")
        print("  → SAM2.1のconv_s0/s1の確認")
    
    if all(s.get('mean', 0) < 0.5 for s in stats.values()):
        print("⚠️ 全体的に性能が低い")
        print("  → Cross-Attention融合の追加を検討")
        print("  → SAM MaskDecoderのLoRA追加を検討")
    
    return stats
```

---

## 📈 実装効果の検証とモニタリング

### 検証メトリクス実装

**ファイル**: `src/utils/metrics.py`（新規作成）

```python
"""
セグメンテーション性能の詳細メトリクス
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple

def compute_segmentation_metrics(
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    包括的なセグメンテーションメトリクスを計算
    
    Args:
        pred_masks: 予測マスク（logits） [B, H, W]
        gt_masks: 正解マスク [B, H, W]
        threshold: 2値化閾値
    
    Returns:
        メトリクスの辞書
    """
    # Sigmoid活性化と2値化
    pred_sigmoid = torch.sigmoid(pred_masks)
    pred_binary = (pred_sigmoid > threshold).float()
    
    # 基本メトリクス
    intersection = (pred_binary * gt_masks).sum(dim=(1, 2))
    pred_sum = pred_binary.sum(dim=(1, 2))
    gt_sum = gt_masks.sum(dim=(1, 2))
    union = pred_sum + gt_sum - intersection
    
    # Dice係数
    dice = (2 * intersection + 1e-8) / (pred_sum + gt_sum + 1e-8)
    
    # IoU (Jaccard Index)
    iou = (intersection + 1e-8) / (union + 1e-8)
    
    # Pixel Accuracy
    correct = (pred_binary == gt_masks).float().sum(dim=(1, 2))
    total = gt_masks.shape[1] * gt_masks.shape[2]
    pixel_acc = correct / total
    
    # Boundary F1 Score（簡易版）
    from scipy import ndimage
    
    boundary_f1_scores = []
    for i in range(len(pred_masks)):
        # エッジ検出
        pred_boundary = ndimage.binary_erosion(pred_binary[i].cpu().numpy()) ^ pred_binary[i].cpu().numpy()
        gt_boundary = ndimage.binary_erosion(gt_masks[i].cpu().numpy()) ^ gt_masks[i].cpu().numpy()
        
        # F1計算
        if gt_boundary.sum() > 0:
            precision = (pred_boundary & gt_boundary).sum() / (pred_boundary.sum() + 1e-8)
            recall = (pred_boundary & gt_boundary).sum() / (gt_boundary.sum() + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            boundary_f1_scores.append(f1)
    
    metrics = {
        'dice': dice.mean().item(),
        'iou': iou.mean().item(),
        'pixel_accuracy': pixel_acc.mean().item(),
        'boundary_f1': np.mean(boundary_f1_scores) if boundary_f1_scores else 0.0,
    }
    
    return metrics

def log_metrics_to_wandb(metrics: Dict[str, float], step: int, prefix: str = "val"):
    """WandBにメトリクスをログ"""
    import wandb
    
    wandb_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
    wandb_metrics["step"] = step
    wandb.log(wandb_metrics)
```

---

## 🎯 実行コマンドと期待結果

### Phase 1実行（即効性のある改善）

```bash
# Phase 1の改善を適用して学習再開
python minimal_train.py \
    --dataset_types "sem_seg||refer_seg||vqa||reason_seg" \
    --sample_rates "9,3,3,1" \
    --samples_per_epoch 35000 \
    --batch_size 2 \
    --gradient_accumulation_steps 16 \
    --num_epochs 3 \
    --adapter_lr 1.0e-4 \
    --lora_lr 5e-5 \
    --seg_token_lr 2.5e-5 \
    --use_edge_loss \
    --edge_loss_weight 0.25 \
    --save_steps 1000 \
    --use_wandb \
    --wandb_project "lisa_kai_phase1"
```

**期待結果**:
- Seg Loss: 1.3 → 1.0-1.1
- 動的解像度の実際の動作確認
- EdgeLossによる境界精度向上

### Phase 2実行（構造的改善）

```bash
# Phase 2: SAM LoRA + Cross-Attention
python minimal_train.py \
    --dataset_types "sem_seg||refer_seg||vqa||reason_seg" \
    --sample_rates "9,3,3,1" \
    --samples_per_epoch 35000 \
    --batch_size 2 \
    --gradient_accumulation_steps 16 \
    --num_epochs 3 \
    --adapter_lr 1.0e-4 \
    --lora_lr 5e-5 \
    --seg_token_lr 2.5e-5 \
    --sam_lora_lr 5e-4 \
    --use_edge_loss \
    --edge_loss_weight 0.25 \
    --use_sam_lora \
    --use_cross_attention_fusion \
    --save_steps 1000 \
    --use_wandb \
    --wandb_project "lisa_kai_phase2"
```

**期待結果**:
- Seg Loss: 1.0-1.1 → 0.7-0.9
- SAM MaskDecoderの適応性向上
- Vision-Language特徴の深い融合

### Phase 3実行（動的解像度最適化）

```bash
# Phase 3: 動的解像度の完全活用
python minimal_train.py \
    --dataset_types "sem_seg||refer_seg||vqa||reason_seg" \
    --sample_rates "9,3,3,1" \
    --samples_per_epoch 35000 \
    --batch_size 2 \
    --use_dynamic_grad_accum \
    --num_epochs 3 \
    --adapter_lr 5e-5 \
    --lora_lr 2.5e-5 \
    --seg_token_lr 1e-5 \
    --sam_lora_lr 2.5e-4 \
    --use_edge_loss \
    --edge_loss_weight 0.25 \
    --use_sam_lora \
    --use_cross_attention_fusion \
    --save_steps 1000 \
    --use_wandb \
    --wandb_project "lisa_kai_phase3"
```

**期待結果**:
- Seg Loss: 0.7-0.9 → 0.5-0.6（LISA相当）
- 解像度適応的な学習
- 最終的な性能最適化

---

## 📊 成功基準とチェックポイント

### Phase 1完了基準（3日後）
- [ ] 動的解像度が実際に可変グリッドで動作
- [ ] Seg Loss < 1.1
- [ ] EdgeLossが正常に計算される
- [ ] 学習が安定して進行

### Phase 2完了基準（1週間後）
- [ ] SAM LoRAパラメータが学習されている
- [ ] Cross-Attention融合が機能
- [ ] Seg Loss < 0.9
- [ ] Dice Score > 0.5

### Phase 3完了基準（10日後）
- [ ] Seg Loss < 0.6（LISA基準達成）
- [ ] mIoU > 35%
- [ ] 高解像度画像での性能向上確認
- [ ] 学習の収束確認

---

## 🚨 トラブルシューティング

### 問題1: OOMエラー
```python
# 解決策: バッチサイズ自動調整
if "CUDA out of memory" in str(e):
    logger.warning("OOM検出: バッチサイズを半分にして再試行")
    batch_size = max(1, batch_size // 2)
```

### 問題2: 勾配爆発
```python
# 解決策: Gradient Clippingの強化
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # 1.0 → 0.5
```

### 問題3: Seg Lossが下がらない
```python
# 解決策: 学習率スケジューリングの調整
if epoch > 2 and seg_loss > 1.0:
    logger.warning("Seg Loss停滞検出: 学習率を半分に")
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.5
```

---

## 📝 実装チェックリスト

### 実装前の準備
- [ ] 現在のコードのバックアップ
- [ ] 新しいブランチの作成: `git checkout -b segmentation-improvement`
- [ ] WandBプロジェクトの準備

### Phase 1実装
- [ ] 動的解像度のデバッグコード追加
- [ ] EdgeLoss関数の実装
- [ ] 学習率の調整
- [ ] 検証スクリプトの作成

### Phase 2実装
- [ ] SAM LoRA設定の追加
- [ ] CrossAttentionFusionモジュールの実装
- [ ] forwardメソッドへの統合
- [ ] パラメータ数の確認

### Phase 3実装
- [ ] 動的Gradient Accumulation
- [ ] 解像度別性能分析
- [ ] メトリクス計算の強化
- [ ] 最終評価

---

## 📌 重要な注意事項

1. **各Phaseは段階的に実装** - 一度にすべてを変更しない
2. **各変更後は必ず動作確認** - 小さな変更でもテストを実行
3. **ログとメトリクスの記録** - すべての実験結果を記録
4. **問題が発生したら即座にロールバック** - gitでの管理を徹底

この実装計画に従って段階的に改善を進めることで、Seg Lossを1.3から0.6以下まで改善し、LISA相当の性能を達成できる見込みです。