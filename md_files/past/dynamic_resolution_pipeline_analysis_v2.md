# LISA改 動的解像度パイプライン完全解析（詳細コード版）
## 2025年8月8日 作成

# 1. 概要

LISA改は、Qwen2.5-VL-3BとSAM2.1を統合した次世代のマルチモーダル理解モデルです。本文書では、動的解像度（Dynamic Resolution）環境下での画像処理パイプライン全体を詳細に解析し、元画像からLoss計算までの完全なデータフローを実際のコードと共に記録します。

## 主要コンポーネント
- **Qwen2.5-VL-3B**: Vision-Language Model（動的解像度対応）
- **SAM2.1**: Segment Anything Model 2.1（高精度セグメンテーション）
- **動的解像度**: min_pixels=112896, max_pixels=7225344（336×336 〜 2688×2688）

---

# 2. 画像処理パイプライン全体フロー

## 2.1 元画像の入力と前処理

### ステップ1: Dataset.__getitem__での画像読み込み

#### 実際のコード実装:
```python
# src/data/dataset.py: HybridDataset.__getitem__ (line 652-956)

def __getitem__(self, idx) -> Dict[str, Any]:
    # データセット選択とサンプル取得
    dataset_idx = np.random.choice(len(self.all_datasets), p=self.sample_rate)
    selected_dataset = self.all_datasets[dataset_idx]
    sample = selected_dataset[idx % len(selected_dataset)]
    
    # PIL画像への変換（テンソルから）
    if isinstance(image_qwen, torch.Tensor):
        # 正規化を元に戻す
        qwen_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        qwen_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        image_denorm = image_qwen * qwen_std + qwen_mean
        image_denorm = torch.clamp(image_denorm, 0, 1)
        
        # (C, H, W) -> (H, W, C)
        image_np = image_denorm.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        image_pil = Image.fromarray(image_np)
    
    # Qwen2.5-VLのapply_chat_templateで画像とテキストを処理
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_pil},
                {"type": "text", "text": text_prompt}
            ]
        }
    ]
    
    # tokenize=Trueで2D形式（N_patches, D_v）を取得
    qwen_processed = self.qwen_processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )
    
    # 出力の取得
    pixel_values = qwen_processed['pixel_values'].squeeze(0)  # [N_raw_patches, 1176]
    image_grid_thw = qwen_processed['image_grid_thw'].squeeze(0)  # [1, H_grid, W_grid]
    
    # SAM用画像前処理（1024×1024）
    image_sam = preprocess_sam_image(image_pil, self.sam_image_size)
```

#### 出力形状:
- **pixel_values**: [N_raw_patches, 1176] - RAWパッチの2D配列
- **image_grid_thw**: [1, H_grid, W_grid] - RAWグリッドサイズ
- **sam_images**: [3, 1024, 1024] - SAM用正規化画像

### ステップ2: 動的解像度処理（Qwen2.5-VL）

#### AutoProcessor内部の動的解像度アルゴリズム:
```python
# transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py相当の処理

def calculate_dynamic_resolution(image_pil, min_pixels=112896, max_pixels=7225344):
    """
    動的解像度の計算
    min_pixels: 336×336 = 112,896
    max_pixels: 2688×2688 = 7,225,344
    """
    width, height = image_pil.size
    current_pixels = width * height
    
    # スケーリング比率の計算
    if current_pixels < min_pixels:
        scale = math.sqrt(min_pixels / current_pixels)
    elif current_pixels > max_pixels:
        scale = math.sqrt(max_pixels / current_pixels)
    else:
        scale = 1.0
    
    # 新しいサイズ（14の倍数に調整）
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # 14の倍数にパディング
    new_width = ((new_width + 13) // 14) * 14
    new_height = ((new_height + 13) // 14) * 14
    
    # RAWパッチ数の計算
    H_grid = new_height // 14
    W_grid = new_width // 14
    
    return new_width, new_height, H_grid, W_grid

# 実際の処理例（640×480画像）
width, height = 640, 480
new_w, new_h, H_grid, W_grid = calculate_dynamic_resolution(
    Image.new('RGB', (width, height))
)
# 結果: new_w=672, new_h=504, H_grid=48, W_grid=36
# RAWパッチ数: 48 × 36 = 1728
# image_grid_thw: [1, 48, 36]
```

---

## 2.2 Collatorでのバッチ処理

### MultiModalDataCollator.__call__の処理フロー

#### 実際のコード実装:
```python
# src/data/collators.py (line 29-459)

def __call__(self, features: List[Dict[str, any]]) -> Dict[str, torch.Tensor]:
    """
    動的解像度対応のバッチ処理
    """
    batch = {}
    
    # 1. RAWパッチ数の分析と最大値決定
    max_raw_patches = 0
    all_raw_patches = []
    
    for i, pv in enumerate(all_pixel_values):
        # pixel_values contains RAW patches (before PatchMerge)
        n_raw_patches = pv.shape[0]
        all_raw_patches.append(n_raw_patches)
        
        grid = all_image_grids[i]
        H_grid_raw = int(grid[1])
        W_grid_raw = int(grid[2])
        
        # RAWパッチ数の検証
        expected_raw_patches = H_grid_raw * W_grid_raw
        if expected_raw_patches != n_raw_patches:
            raise ValueError(f"Patch count mismatch!")
        
        max_raw_patches = max(max_raw_patches, n_raw_patches)
    
    # 2. 偶数グリッドへの調整（PatchMerge互換性のため）
    import math
    grid_size = math.ceil(math.sqrt(max_raw_patches))
    # 偶数に切り上げ（2x2 PatchMergeに必須）
    H_grid_max = W_grid_max = (grid_size + 1) // 2 * 2
    N_padded_raw = H_grid_max * W_grid_max
    
    # PatchMerge後のトークン数
    max_img_tokens = N_padded_raw // 4
    
    print(f"[Batch Collator] Max RAW patches: {max_raw_patches}")
    print(f"  Target grid (even): {H_grid_max}×{W_grid_max} = {N_padded_raw} patches")
    print(f"  Image tokens after PatchMerge: {max_img_tokens}")
    
    # 3. 各サンプルの処理
    IMAGE_PAD_ID = 151655  # <image_pad>トークンID
    
    for i, f in enumerate(features):
        pv = f['pixel_values']
        input_ids = f['input_ids']
        
        # pixel_valuesのパディング
        current_patches = pv.shape[0]
        pad_len = N_padded_raw - current_patches
        
        if pad_len > 0:
            padding = torch.zeros(pad_len, pv.shape[1], dtype=pv.dtype, device=pv.device)
            pv_padded = torch.cat([pv, padding], dim=0)
        else:
            pv_padded = pv[:N_padded_raw]
        
        padded_pixel_values.append(pv_padded)
        
        # image_padトークンの調整
        existing_image_pads = (input_ids == IMAGE_PAD_ID).sum().item()
        target_image_pads = N_padded_raw // 4  # PatchMerge後
        image_pad_diff = target_image_pads - existing_image_pads
        
        if image_pad_diff > 0:
            # image_padトークンを追加
            image_pad_mask = (input_ids == IMAGE_PAD_ID)
            pad_indices = image_pad_mask.nonzero(as_tuple=True)[0]
            last_pad_idx = pad_indices[-1].item()
            
            ids_padded = torch.cat([
                input_ids[:last_pad_idx + 1],
                input_ids.new_full((image_pad_diff,), IMAGE_PAD_ID),
                input_ids[last_pad_idx + 1:]
            ])
        elif image_pad_diff < 0:
            # 余分なimage_padトークンを削除
            # (コード省略)
        
        # image_grid_thwを統一サイズに更新
        unified_grid = torch.tensor([1, H_grid_max, W_grid_max], dtype=torch.long)
        new_image_grid_thw.append(unified_grid)
    
    # 4. バッチテンソルの作成
    batch['pixel_values'] = torch.stack(padded_pixel_values)  # [B, N_padded, 1176]
    batch['image_grid_thw'] = torch.stack(new_image_grid_thw)  # [B, 3]
    batch['input_ids'] = torch.stack(final_input_ids)
    
    return batch
```

#### 実際のログ出力例:
```
[Batch Collator] Max RAW patches: 832
  Target grid (even): 30×30 = 900 patches
  Image tokens after PatchMerge: 225
  Sample 0: 704 patches (grid: 22×32)
  Sample 1: 832 patches (grid: 32×26)
  Sample 2: 768 patches (grid: 24×32)
  Sample 3: 768 patches (grid: 24×32)
```

### 重要な変換公式
```
RAWパッチ数 = H_grid × W_grid
PatchMerge後トークン数 = RAWパッチ数 ÷ 4
image_padトークン数 = PatchMerge後トークン数
```

---

## 2.3 モデル内部での特徴抽出

### LISA_Model.forwardの処理

#### 実際のコード実装:

##### 1. Qwen視覚特徴抽出（修正版）
```python
# src/models/lisa_model.py (line 290-386)

def extract_vision_features(self, pixel_values: torch.Tensor, image_grid_thw: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    バッチ処理時のトークン選択を回避する修正版
    """
    B = pixel_values.shape[0] if pixel_values.dim() == 3 else 1
    
    if B == 1:
        # Single sample - process directly
        image_embeds = self.qwen.model.get_image_features(pixel_values, image_grid_thw)
        
        if isinstance(image_embeds, tuple):
            image_embeds = image_embeds[0]
        
        if image_embeds.dim() == 2:
            image_embeds = image_embeds.unsqueeze(0)  # [N, D] -> [1, N, D]
        
        vision_features = image_embeds
        
    else:
        # Batch processing - 各サンプルを個別に処理してトークン選択を回避
        all_features = []
        
        for i in range(B):
            # 単一サンプルを抽出
            single_pixel_values = pixel_values[i:i+1]  # バッチ次元を保持
            single_grid = image_grid_thw[i:i+1]
            
            # 個別に処理
            single_embeds = self.qwen.model.get_image_features(single_pixel_values, single_grid)
            
            if isinstance(single_embeds, tuple):
                single_embeds = single_embeds[0]
            
            if single_embeds.dim() == 2:
                single_embeds = single_embeds.unsqueeze(0)
            
            all_features.append(single_embeds)
        
        # すべての特徴を結合
        vision_features = torch.cat(all_features, dim=0)  # [B, N_patches, 2048]
        logger.debug(f"Batch processing successful: {B} samples, {vision_features.shape[1]} tokens each")
    
    return vision_features

# 実際の呼び出し（forward内）
vision_features = self.extract_vision_features(pixel_values, image_grid_thw)
# 入力: [4, 900, 1176], [4, 3]
# 出力: [4, 225, 2048]  # PatchMerge後
```

##### 2. ImageFeatureAdapterによる変換
```python
# src/models/adapters.py (line 10-73)

class ImageFeatureAdapter(nn.Module):
    def __init__(self, in_dim: int = 2048, out_dim: int = 256):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)
        
    def forward(self, vision_features: torch.Tensor, image_grid_thw: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        vision_features: [B, N_patches, in_dim=2048]
        出力: [B, out_dim=256, H_feat, W_feat]
        """
        B = vision_features.shape[0]
        
        # 線形変換: 2048 -> 256
        adapted = self.linear(vision_features)  # [B, N_patches, 256]
        adapted = self.layer_norm(adapted)
        
        # グリッド形状を計算（PatchMerge後）
        if image_grid_thw is not None:
            # RAWグリッドからPatchMerge後のサイズを計算
            H_raw = int(image_grid_thw[0, 1].item())
            W_raw = int(image_grid_thw[0, 2].item())
            H_feat = H_raw // 2  # PatchMerge: 2×2結合
            W_feat = W_raw // 2
        else:
            # デフォルト（正方形を仮定）
            N_patches = adapted.shape[1]
            H_feat = W_feat = int(math.sqrt(N_patches))
        
        # [B, N, C] -> [B, H, W, C] -> [B, C, H, W]
        adapted = adapted.view(B, H_feat, W_feat, 256)
        adapted = adapted.permute(0, 3, 1, 2)  # [B, 256, H_feat, W_feat]
        
        return adapted

# 実際の使用例
image_features_sam = self.image_adapter(vision_features, image_grid_thw)
# 入力: [4, 225, 2048], [4, 3]（グリッド[1, 30, 30]）
# 出力: [4, 256, 15, 15]  # 30÷2=15
```

##### 3. 高解像度特徴の生成
```python
# src/models/adapters.py (line 157-229)

class HighResFeatureGenerator(nn.Module):
    def __init__(self, in_channels: int = 256):
        super().__init__()
        # Stride-4用のアップサンプリング
        self.upsample_s0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=4),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Stride-8用のアップサンプリング  
        self.upsample_s1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, image_features: torch.Tensor) -> List[torch.Tensor]:
        """
        入力: [B, 256, H, W]
        出力: [stride-4特徴, stride-8特徴]
        """
        # Stride-4: 4倍アップサンプリング
        feat_s0 = self.upsample_s0(image_features)  # [B, 256, H*4, W*4]
        
        # Stride-8: 2倍アップサンプリング
        feat_s1 = self.upsample_s1(image_features)  # [B, 256, H*2, W*2]
        
        return [feat_s0, feat_s1]

# 実際の使用例
sam_high_res_features = self.high_res_generator(image_features_sam)
# 入力: [4, 256, 15, 15]
# 出力: [[4, 256, 60, 60], [4, 256, 30, 30]]
```

### 位置情報の保持（M-RoPE）
```python
# Qwen2.5-VL内部でのM-RoPE処理
def apply_multimodal_rotary_pos_emb(hidden_states, image_grid_thw):
    """
    2D位置エンコーディングの適用
    image_grid_thw: [B, 3] where [T=1, H_grid, W_grid]
    """
    # グリッド位置を2D座標に展開
    H_grid, W_grid = image_grid_thw[0, 1], image_grid_thw[0, 2]
    
    # 各パッチの(x, y)座標を生成
    y_pos = torch.arange(H_grid).unsqueeze(1).expand(-1, W_grid)
    x_pos = torch.arange(W_grid).unsqueeze(0).expand(H_grid, -1)
    
    # RoPEエンコーディングを適用
    # PatchMerge後も相対位置関係を保持
    return apply_rotary_pos_emb_2d(hidden_states, x_pos, y_pos)
```

---

## 2.4 セグメンテーションマスク生成

### SEGトークンからマスクへの変換

#### 実際のコード実装:

##### 1. SEGトークン位置の特定と処理
```python
# src/models/lisa_model.py (line 500-540)

# SEGトークンIDの設定
self.seg_token_id = 151665  # <SEG>

# SEG位置の検出
seg_positions = []
if labels is not None:
    # Training: labelsから検索
    for i in range(B):
        seg_pos = (labels[i] == self.seg_token_id).nonzero(as_tuple=True)[0]
        if len(seg_pos) > 0:
            seg_positions.append(seg_pos.tolist())
        else:
            seg_positions.append([])
else:
    # Inference: input_idsから検索
    for i in range(B):
        seg_pos = (input_ids[i] == self.seg_token_id).nonzero(as_tuple=True)[0]
        if len(seg_pos) > 0:
            seg_positions.append(seg_pos.tolist())
        else:
            seg_positions.append([])

# SEG位置のhidden states抽出
for i in range(B):
    if len(seg_positions[i]) > 0:
        # hidden_states: [B, seq_len, 3584]
        seg_hidden_states = hidden_states[i, seg_positions[i]]  # [num_segs, 3584]
```

##### 2. TextPromptProjectorによる変換
```python
# src/models/adapters.py (line 76-111)

class TextPromptProjector(nn.Module):
    def __init__(self, text_dim: int = 3584, prompt_dim: int = 256):
        super().__init__()
        # 3層MLPで次元削減
        self.proj = nn.Sequential(
            nn.Linear(text_dim, text_dim // 2),  # 3584 -> 1792
            nn.ReLU(),
            nn.Linear(text_dim // 2, text_dim // 4),  # 1792 -> 896
            nn.ReLU(),
            nn.Linear(text_dim // 4, prompt_dim),  # 896 -> 256
            nn.LayerNorm(prompt_dim)
        )
        
    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        # text_features: [num_prompts, 3584]
        # 出力: [num_prompts, 256]
        return self.proj(text_features)

# 実際の使用
prompt_embeds = self.text_prompt_proj(seg_hidden_states)  # [num_segs, 256]
```

##### 3. SAM2.1 MaskDecoderでのマスク生成
```python
# src/models/lisa_model.py (line 541-660)

for j, prompt_embed in enumerate(prompt_embeds if len(seg_positions[i]) > 1 else [prompt_embeds]):
    # 位置エンコーディングの取得
    image_pe = self.sam_prompt_encoder.get_dense_pe()
    
    # PEサイズの調整
    if image_pe.shape[-2:] != image_features_sam[i:i+1].shape[-2:]:
        h_feat, w_feat = image_features_sam[i:i+1].shape[-2:]
        image_pe = F.interpolate(
            image_pe,
            size=(h_feat, w_feat),
            mode='bilinear',
            align_corners=False
        )
    
    # 中心点を基準点として使用（テキストガイドセグメンテーション用）
    h_feat, w_feat = image_features_sam[i:i+1].shape[-2:]
    h_img, w_img = h_feat * 16, w_feat * 16  # feature stride = 16
    center_x, center_y = w_img // 2, h_img // 2
    
    # SAMプロンプトエンコーダで埋め込み生成
    point_coords = torch.tensor([[center_x, center_y]], 
                              dtype=torch.float32, 
                              device=prompt_embed.device).unsqueeze(0)  # [1, 1, 2]
    point_labels = torch.tensor([1], dtype=torch.int32, 
                              device=prompt_embed.device).unsqueeze(0)  # [1, 1] (positive)
    
    sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
        points=(point_coords, point_labels),
        boxes=None,
        masks=None,
    )
    
    # テキスト由来の埋め込みで置換
    if sparse_embeddings.shape[1] > 0:
        sparse_embeddings[:, 0, :] = prompt_embed.unsqueeze(0)
    
    # 高解像度特徴の準備（チャンネル圧縮）
    feat_s0 = sam_high_res_features[0][i:i+1]  # [1, 256, H*4, W*4]
    feat_s1 = sam_high_res_features[1][i:i+1]  # [1, 256, H*2, W*2]
    
    # SAM2.1のconv_s0/conv_s1でチャンネル圧縮: 256 -> 32/64
    if hasattr(self.sam_mask_decoder, 'conv_s0'):
        feat_s0 = self.sam_mask_decoder.conv_s0(feat_s0)  # 256 -> 32 channels
        feat_s1 = self.sam_mask_decoder.conv_s1(feat_s1)  # 256 -> 64 channels
    
    high_res_features = [feat_s0, feat_s1]
    
    # MaskDecoder実行
    low_res_masks, iou_predictions, sam_tokens_out, obj_score_logits = self.sam_mask_decoder(
        image_embeddings=image_features_sam[i:i+1],  # [1, 256, H_feat, W_feat]
        image_pe=image_pe,
        sparse_prompt_embeddings=sparse_embeddings,  # テキスト由来
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,  # 単一マスク出力
        repeat_image=True,  # バッチサイズ不一致を処理
        high_res_features=high_res_features,  # 高解像度特徴
    )
```

##### 4. 元画像サイズへのアップスケール
```python
# 動的解像度に基づくサイズ計算
if image_grid_thw is not None and i < image_grid_thw.shape[0]:
    # RAWパッチグリッドから元画像サイズを計算
    H_grid_raw = int(image_grid_thw[i, 1].item())
    W_grid_raw = int(image_grid_thw[i, 2].item())
    # パッチサイズ14pxで元画像サイズを復元
    orig_h = H_grid_raw * 14
    orig_w = W_grid_raw * 14
elif pixel_values is not None and pixel_values.dim() == 4:
    # フォールバック: pixel_valuesから
    orig_h, orig_w = pixel_values.shape[-2:]
else:
    # デフォルト: feature map size * stride
    orig_h = h_feat * 16
    orig_w = w_feat * 16

# Bilinear補間でリサイズ
mask_logit = F.interpolate(
    low_res_masks,  # [1, 1, H_mask, W_mask]
    size=(orig_h, orig_w),
    mode='bilinear',
    align_corners=False
)
sample_masks.append(mask_logit.squeeze(0))  # [1, orig_h, orig_w]

# 実際の例
# image_grid_thw[i] = [1, 48, 36]
# orig_h = 48 * 14 = 672
# orig_w = 36 * 14 = 504
# 最終マスク: [1, 672, 504]
```

---

## 2.5 Loss計算と整合性

### compute_lossでの処理

#### 実際のコード実装:
```python
# minimal_train.py (line 228-312)

def compute_loss(self, outputs, labels, mask_labels):
    """損失計算"""
    # 1. 言語モデリング損失
    vocab_size = outputs.logits.size(-1)  # 151666
    lm_loss = nn.functional.cross_entropy(
        outputs.logits.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=-100  # パディングトークンを無視
    )
    
    # 2. セグメンテーション損失
    seg_loss = 0.0
    seg_count = 0
    
    if outputs.mask_logits is not None:
        for batch_idx, batch_masks in enumerate(outputs.mask_logits):
            if batch_masks is not None and len(batch_masks) > 0:
                gt_mask = mask_labels[batch_idx]  # 正解マスク [1, 1024, 1024]
                
                for pred_mask in batch_masks:
                    # デバッグ情報
                    logger.debug(f"pred_mask shape: {pred_mask.shape}")
                    logger.debug(f"gt_mask shape: {gt_mask.shape}")
                    
                    # マスクの次元を確認して適切に処理
                    if pred_mask.dim() == 2:
                        pred_h, pred_w = pred_mask.shape
                    else:
                        pred_h, pred_w = pred_mask.shape[-2:]
                        
                    if gt_mask.dim() == 2:
                        gt_h, gt_w = gt_mask.shape
                    else:
                        gt_h, gt_w = gt_mask.shape[-2:]
                    
                    # サイズが異なる場合はリサイズ（動的解像度対応）
                    if (pred_h, pred_w) != (gt_h, gt_w):
                        # gt_maskを予測サイズに合わせる
                        if gt_mask.dim() == 2:
                            gt_mask_4d = gt_mask.unsqueeze(0).unsqueeze(0)  # [H,W] -> [1,1,H,W]
                        elif gt_mask.dim() == 3:
                            gt_mask_4d = gt_mask.unsqueeze(1)  # [B,H,W] -> [B,1,H,W]
                        else:
                            gt_mask_4d = gt_mask
                        
                        # float型に変換してinterpolate
                        gt_mask_4d = gt_mask_4d.float()
                        gt_mask_resized = nn.functional.interpolate(
                            gt_mask_4d,
                            size=(pred_h, pred_w),  # 予測マスクのサイズに合わせる
                            mode='nearest'  # セグメンテーションマスクはnearest補間
                        )
                        
                        # 元の次元に戻す
                        if gt_mask.dim() == 2:
                            gt_mask_resized = gt_mask_resized.squeeze(0).squeeze(0)
                        elif gt_mask.dim() == 3:
                            gt_mask_resized = gt_mask_resized.squeeze(1)
                    else:
                        gt_mask_resized = gt_mask
                    
                    # BCE損失（Binary Cross Entropy）
                    bce_loss = nn.functional.binary_cross_entropy_with_logits(
                        pred_mask.squeeze(0),
                        gt_mask_resized.squeeze(0)
                    )
                    
                    # Dice損失
                    pred_sigmoid = torch.sigmoid(pred_mask.squeeze(0))
                    intersection = (pred_sigmoid * gt_mask_resized.squeeze(0)).sum()
                    dice = 2 * intersection / (pred_sigmoid.sum() + gt_mask_resized.squeeze(0).sum() + 1e-8)
                    dice_loss = 1 - dice
                    
                    # 合計損失
                    seg_loss += bce_loss + dice_loss
                    seg_count += 1
    
    # 3. 平均セグメンテーション損失
    if seg_count > 0:
        seg_loss = seg_loss / seg_count
    
    # 4. 総合損失
    total_loss = lm_loss + self.config.seg_loss_weight * seg_loss
    
    return total_loss, lm_loss, seg_loss
```

### マスクサイズの整合性保証

#### 動的解像度環境での整合性処理:
```python
# 実際のサイズ例

# ケース1: 元画像640×480の場合
# 1. Qwen処理後: 672×504（14の倍数）
# 2. 予測マスク: [1, 672, 504]（image_grid_thwから計算）
# 3. 正解マスク: [1, 1024, 1024]（SAM標準サイズ）
# 4. Loss計算時: 正解を672×504にnearest補間

# ケース2: バッチ処理（統一グリッド30×30）
# 1. 統一後: 420×420ピクセル（30×14）
# 2. 予測マスク: [1, 420, 420]
# 3. 正解マスク: [1, 1024, 1024]
# 4. Loss計算時: 正解を420×420にnearest補間

# サイズ調整の具体例
def adjust_mask_size(pred_mask, gt_mask):
    """
    予測マスクと正解マスクのサイズを合わせる
    """
    pred_h, pred_w = pred_mask.shape[-2:]
    gt_h, gt_w = gt_mask.shape[-2:]
    
    if (pred_h, pred_w) != (gt_h, gt_w):
        # 動的解像度により予測サイズが異なる
        print(f"Adjusting mask: GT {gt_h}×{gt_w} -> Pred {pred_h}×{pred_w}")
        
        # nearest補間でリサイズ（セグメンテーションマスクの標準）
        gt_mask_resized = F.interpolate(
            gt_mask.unsqueeze(0).unsqueeze(0).float(),
            size=(pred_h, pred_w),
            mode='nearest'
        ).squeeze(0).squeeze(0)
        
        return gt_mask_resized
    
    return gt_mask
```

---

# 3. データ形状の変遷まとめ

## 3.1 単一サンプルの変遷

### 視覚的フロー図:
```
元画像 (PIL.Image) 
│ 例: 640×480ピクセル
│
├─[Qwen処理]────────────────────────────────────────┐
│  apply_chat_template(tokenize=True)              │
│  ↓                                                │
│  pixel_values [1728, 1176]                       │
│  (48×36 RAWパッチ × 1176次元)                     │
│  ↓                                                │
│  [PatchMerge 2×2結合]                             │
│  ↓                                                │
│  [432, 2048] (1728÷4=432トークン)                 │
│  ↓                                                │
│  [ImageAdapter]                                   │
│  ↓                                                │
│  [256, 24, 18] (24×18グリッド)                    │
│                                                   │
└─[SAM処理]─────────────────────────────────────────┐
   preprocess_sam_image()                          │
   ↓                                                │
   sam_images [3, 1024, 1024]                      │
   ↓                                                │
   [MaskDecoder処理]                                │
   ↓                                                │
   masks [1, 672, 504]                             │
   (元画像解像度に復元)                              │
```

## 3.2 バッチ処理での形状

### 詳細な形状変化テーブル:

| ステージ | 形状 | 例（バッチサイズ4） | 備考 |
|---------|------|-------------------|------|
| **入力段階** |
| 元画像 | 可変 | 各サンプル異なる | PIL.Image形式 |
| Dataset出力 pixel_values | [N_raw_i, 1176] | [704, 1176], [832, 1176], [768, 1176], [768, 1176] | 各サンプル個別 |
| Dataset出力 image_grid_thw | [1, H_i, W_i] | [1,22,32], [1,32,26], [1,24,32], [1,24,32] | RAWグリッド |
| **Collator処理後** |
| pixel_values | [B, N_max, 1176] | [4, 900, 1176] | 30×30に統一 |
| image_grid_thw | [B, 3] | [4, 3] | 全て[1,30,30] |
| input_ids | [B, seq_len] | [4, 263] | image_pad含む |
| **モデル内部** |
| vision_features | [B, N_merge, 2048] | [4, 225, 2048] | 900÷4=225 |
| image_features_sam | [B, 256, H, W] | [4, 256, 15, 15] | 30÷2=15 |
| high_res_features[0] | [B, 256, H×4, W×4] | [4, 256, 60, 60] | Stride-4 |
| high_res_features[1] | [B, 256, H×2, W×2] | [4, 256, 30, 30] | Stride-8 |
| **出力段階** |
| 予測マスク | [B, 1, H_orig, W_orig] | 各サンプル異なる | 動的サイズ |
| 正解マスク | [B, 1, 1024, 1024] | [4, 1, 1024, 1024] | SAM標準 |

### 実際のコードでの確認方法:
```python
# デバッグ用の形状確認コード
def debug_shapes(batch, model_outputs):
    print("=== バッチ処理の形状確認 ===")
    
    # 入力
    print(f"pixel_values: {batch['pixel_values'].shape}")
    print(f"image_grid_thw: {batch['image_grid_thw'].shape}")
    print(f"  Grid values: {batch['image_grid_thw']}")
    
    # 中間特徴
    if hasattr(model_outputs, 'vision_hidden_states'):
        print(f"vision_features: {model_outputs.vision_hidden_states.shape}")
    
    # マスク出力
    if model_outputs.mask_logits:
        for i, masks in enumerate(model_outputs.mask_logits):
            if masks:
                for j, mask in enumerate(masks):
                    print(f"  Sample {i}, Mask {j}: {mask.shape}")
```

---

# 4. 重要な発見と修正点

## 4.1 トークン選択問題の解決

**問題**: バッチ処理時にget_image_featuresが自動的にトークン選択を適用（256→64）

**解決策**（実装済み）:
```python
# extract_vision_featuresの修正
- バッチの各サンプルを個別処理
- トークン選択を完全回避
- 全視覚トークンを保持
```

## 4.2 RAW基準パディングの実装

**Collatorの正しい実装**:
1. RAWパッチ数で統一（PatchMerge前）
2. image_grid_thwも更新
3. image_padトークン数を正確に計算

## 4.3 位置情報の一貫性

**M-RoPEによる位置エンコーディング**:
- パディング後もimage_grid_thwで正しい位置を保持
- セグメンテーション精度の向上に寄与

---

# 5. パフォーマンス最適化の推奨事項

## 5.1 バケット化戦略
```python
推奨実装:
- 類似解像度のサンプルをグループ化
- パディング量の最小化
- メモリ効率の向上
```

## 5.2 将来の拡張性

### 2560次元PatchMerge特徴の利用（将来実装）
```python
# transformers v4.57+で実装予定
- return_dict=True のサポート
- hidden_states[-1]から2560次元特徴取得
- より豊富な視覚情報の活用
```

---

# 6. デバッグとモニタリング

## 6.1 重要なチェックポイント

1. **Collator出力の検証**:
   - pixel_values形状
   - image_padトークン数
   - image_grid_thwの一致

2. **視覚特徴の確認**:
   - extract_vision_features出力形状
   - トークン数の保持

3. **マスク生成の検証**:
   - 予測マスクサイズ
   - 正解マスクとの整合性

## 6.2 ログ出力の活用
```python
logger.debug(f"vision_features shape: {vision_features.shape}")
logger.debug(f"Total tokens: {total_tokens}, Expected: {expected}")
```

---

# 7. まとめ

LISA改の動的解像度パイプラインは、以下の特徴を持つ高度な画像処理システムです：

1. **柔軟な解像度対応**: 336×336から2688×2688まで対応
2. **効率的なバッチ処理**: RAW基準パディングによる一貫性
3. **正確な位置情報保持**: M-RoPEによる2D位置エンコーディング
4. **マルチスケール特徴**: 複数のstrideで高解像度特徴生成
5. **ロバストなLoss計算**: 動的サイズ調整による整合性保証

本システムは、将来のFoodLMM改への発展を見据え、高精度な画像理解とセグメンテーションを実現する基盤となっています。