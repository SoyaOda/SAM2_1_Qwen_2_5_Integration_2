# LISA改 現状実装分析 - 動的解像度実装のための詳細調査

## 1. 概要

このドキュメントは、LISA改（Qwen2.5-VL + SAM2.1統合モデル）の現在の実装状況を詳細に分析し、動的解像度機能の実装に必要な情報をまとめたものです。

## 2. 現在の画像処理フロー

### 2.1 固定解像度の現状

現在の実装では、すべての画像処理が固定解像度で行われています：

- **Qwen用画像**: 448×448ピクセル（固定）
- **SAM用画像**: 1024×1024ピクセル（固定）
- **グリッドサイズ**: 24×24（576パッチ、144トークン@4:1圧縮）

### 2.2 画像前処理の実装詳細

#### 2.2.1 Qwen用画像前処理（`src/data/dataset.py: preprocess_qwen_image`）

```python
def preprocess_qwen_image(image: Image.Image, processor: AutoProcessor, target_size: Optional[int] = None) -> torch.Tensor:
    """
    Qwen用画像前処理：アスペクト比を維持して448x448以下にリサイズし、14の倍数にパディング
    """
    if target_size is None:
        target_size = getattr(config, 'QWEN_IMAGE_SIZE', 448)
    
    # アスペクト比を維持してリサイズ
    scale = target_size / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # 14の倍数になるようにパディング（Qwen2.5-VLのパッチサイズ）
    pad_h = (-new_h) % 14
    pad_w = (-new_w) % 14
```

**実装箇所**: `src/data/dataset.py` lines 137-196

**重要な点**:
- 448×448固定サイズへのリサイズ
- 14ピクセル（パッチサイズ）の倍数への調整
- 正規化パラメータ: mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]

#### 2.2.2 SAM用画像前処理（`src/data/dataset.py: preprocess_sam_image`）

```python
def preprocess_sam_image(image: Image.Image, target_size: Optional[int] = None) -> torch.Tensor:
    """
    SAM用画像前処理：1024x1024にリサイズ・パディング・正規化
    """
    # 最長辺を1024にリサイズ
    # 1024x1024にパディング
    # 正規化: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
```

**実装箇所**: `src/data/dataset.py` lines 101-135

### 2.3 データセットでの画像処理（`src/data/dataset.py: HybridDataset.__getitem__`）

```python
# Qwen2.5-VLのapply_chat_templateで画像とテキストを一緒に処理
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_pil},
            {"type": "text", "text": text_prompt}
        ]
    }
]

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
image_grid_thw = qwen_processed.get('image_grid_thw')  # 常に[1, 24, 24]
```

**実装箇所**: `src/data/dataset.py` lines 468-727

**重要な点**:
- `apply_chat_template`が画像のトークン化を処理
- `image_grid_thw`は取得されるが、常に固定値[1, 24, 24]
- 動的解像度の情報は完全に無視されている

## 3. バッチ処理とコレーター実装

### 3.1 MultiModalDataCollator（`src/data/collators.py`）

現在のコレーターは2つの形式の画像データを処理：

1. **2Dパッチ形式**（`pixel_values.dim() == 2`）
   - Qwen2.5-VLの`apply_chat_template(tokenize=True)`の出力
   - 形式: (N_patch, D_v) where N_patch = H_grid × W_grid
   - 4:1パッチ圧縮対応

2. **3D画像形式**（`pixel_values.dim() == 3`）
   - 通常の画像テンソル形式: (C, H, W)
   - 動的解像度の可能性があるが、現在は使用されていない

```python
# 2Dパッチ形式の処理
if first_pix.dim() == 2:
    # バッチ内で最大のグリッドサイズを見つける
    for i, pv in enumerate(all_pixel_values):
        n_patches = pv.shape[0]
        # グリッドサイズを推定
        H_grid = int(grid[1])
        W_grid = int(grid[2])
        H_grid_max = max(H_grid_max, H_grid)
        W_grid_max = max(W_grid_max, W_grid)
    
    # 4の倍数に調整（Qwen2.5-VLの4:1パッチ圧縮）
    max_patches = ((max_patches_raw + 3) // 4) * 4
```

**実装箇所**: `src/data/collators.py` lines 45-337

**重要な点**:
- バッチ内で統一されたグリッドサイズが必要
- トークン数上限チェック（2048トークン制限）
- 動的グリッドサイズの処理ロジックは存在するが、実際には使用されていない

## 4. Vision Feature抽出とアダプター

### 4.1 Vision Feature抽出（`src/models/lisa_model.py: extract_vision_features`）

```python
def extract_vision_features(self, pixel_values: torch.Tensor, image_grid_thw: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Extract vision features from Qwen's vision encoder
    """
    # フックを使用してmerger前の1280次元特徴を取得
    def capture_features(module, input, output):
        features_dict['before_merger'] = input[0]
    
    # デフォルトグリッド（常に24×24）
    if image_grid_thw is not None:
        _ = visual_module(pixel_values, grid_thw=image_grid_thw)
    else:
        default_grid = torch.tensor([[1, 24, 24]], device=pixel_values.device).repeat(B, 1)
        _ = visual_module(pixel_values, grid_thw=default_grid)
```

**実装箇所**: `src/models/lisa_model.py` lines 265-327

**重要な点**:
- merger前の純粋な視覚特徴（1280次元）を抽出
- `image_grid_thw`パラメータは渡されるが、常に固定値
- 動的解像度の能力は完全に未使用

### 4.2 ImageFeatureAdapter（`src/models/adapters.py`）

```python
class ImageFeatureAdapter(nn.Module):
    """
    Adapter to transform Qwen vision features to SAM-compatible format
    """
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        # [B, N_patches, D_v] -> [B, out_dim, H, W]
        B, N, D_v = vision_features.shape
        features = self.proj(vision_features)  # [B, N, 256]
        features = self.norm(features)
        
        # 空間次元を計算（正方形パッチを仮定）
        H = W = int(math.sqrt(N))
```

**実装箇所**: `src/models/adapters.py` lines 12-96

**重要な点**:
- 1280次元 → 256次元への変換
- パッチ数から空間次元を推定
- 非正方形グリッドのサポートあり（動的解像度対応の基盤）

## 5. 高解像度特徴生成とMask Decoder統合

### 5.1 HighResFeatureGenerator（`src/models/lisa_model.py`）

```python
class HighResFeatureGenerator(nn.Module):
    """
    Generates high-resolution features from Qwen vision features for SAM2.1 mask decoder.
    """
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Generate multi-scale features
        feat_s1 = self.upsample_2x(x)   # [B, 256, 2*H, 2*W] stride-8
        feat_s0 = self.upsample_4x(x)   # [B, 256, 4*H, 4*W] stride-4
        
        # SAM2.1のMaskDecoderがconv_s0/conv_s1を適用
        return [feat_s0, feat_s1]
```

**実装箇所**: `src/models/lisa_model.py` lines 33-92

**重要な点**:
- Qwen視覚特徴から高解像度特徴を生成
- SAM2.1のネイティブimage_encoderは使用しない
- 256チャンネルで出力（SAMが32/64チャンネルに圧縮）

### 5.2 SAM2.1 MaskDecoderへの特徴伝達

```python
# SAM2.1 MaskDecoder呼び出し
low_res_masks, iou_predictions, sam_tokens_out, obj_score_logits = self.sam_mask_decoder(
    image_embeddings=image_features_sam[i:i+1].to(dtype=sam_dtype),  # [B, 256, H, W]
    image_pe=image_pe.to(dtype=sam_dtype),                           # 位置エンコーディング
    sparse_prompt_embeddings=sparse_embeddings.to(dtype=sam_dtype),  # テキストプロンプト
    dense_prompt_embeddings=dense_embeddings.to(dtype=sam_dtype),
    multimask_output=False,
    repeat_image=True,
    high_res_features=high_res_features,  # [feat_s0, feat_s1]
)
```

**実装箇所**: `src/models/lisa_model.py` lines 527-535

**重要な点**:
- 主特徴（stride-16）: Qwen視覚特徴をアダプター変換
- 高解像度特徴（stride-4,8）: HighResFeatureGeneratorで生成
- SAM2.1の`conv_s0`/`conv_s1`でチャンネル圧縮（256→32/64）

## 6. 動的解像度実装に必要な変更箇所

### 6.1 データ処理層

1. **`preprocess_qwen_image`関数**
   - 固定448×448ではなく、動的サイズをサポート
   - 最大解像度とトークン数制限の実装

2. **`HybridDataset.__getitem__`**
   - 画像サイズに基づく動的`image_grid_thw`の計算
   - 可変パッチ数への対応

### 6.2 コレーター層

1. **`MultiModalDataCollator`**
   - 可変グリッドサイズのバッチ処理
   - トークン数上限（16384）への対応
   - 効率的なパディング戦略

### 6.3 モデル層

1. **`extract_vision_features`**
   - 動的`image_grid_thw`の適切な処理
   - 可変パッチ数への対応

2. **`ImageFeatureAdapter`**
   - 非正方形グリッドの完全サポート
   - 動的空間次元の処理

3. **`HighResFeatureGenerator`**
   - 可変入力サイズへの対応
   - 適応的アップサンプリング

## 7. 設定パラメータと制限

### 7.1 現在の設定（`src/config/config_linux.py`）

```python
QWEN_IMAGE_SIZE = 448        # 固定サイズ
SAM_IMAGE_SIZE = 1024       # 固定サイズ
MODEL_MAX_LENGTH = 2048     # トークン上限（本来は16384可能）
```

### 7.2 動的解像度実装時の推奨設定

```python
QWEN_MAX_IMAGE_SIZE = 2688  # 最大解像度
QWEN_MIN_IMAGE_SIZE = 336   # 最小解像度
QWEN_PATCH_SIZE = 14        # パッチサイズ
MODEL_MAX_LENGTH = 16384    # 最大トークン数
```

## 8. まとめ

現在の実装は固定解像度（448×448）で動作しており、Qwen2.5-VLの動的解像度機能は完全に未活用です。しかし、コードベースには動的解像度をサポートするための基本的な構造（可変グリッドサイズ処理、非正方形グリッドサポート）が存在しており、適切な修正により動的解像度機能を実装することが可能です。

主な課題は：
1. データ前処理での動的サイズ対応
2. バッチ処理での効率的なパディング戦略
3. トークン数制限（16384）の管理
4. メモリ効率的な実装

これらの課題に対処することで、Qwen2.5-VLの高機能なViTを最大限活用し、セグメンテーション精度の向上が期待できます。