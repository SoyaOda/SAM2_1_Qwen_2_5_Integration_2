# LISA改 技術詳細ドキュメント

## アーキテクチャ詳細

### 1. モデル統合の詳細

#### Qwen2.5-VL-3Bの統合
```python
# Vision Encoder
- 入力: RGB画像 [B, C, H, W]
- 出力: Vision Features [B, N_patches, 1280]
- パッチサイズ: 14x14
- 特徴次元: 1280

# Language Model
- 入力: Token IDs [B, seq_len]
- 出力: Hidden States [B, seq_len, 2048]
- 語彙サイズ: 152000
- 最大シーケンス長: 32768
```

#### SAM2.1の統合
```python
# Image Encoder (Hiera-Large)
- 入力: Image Features [B, 256, H/4, W/4]
- 出力: Image Embeddings [B, 256, 64, 64]
- アーキテクチャ: Hierarchical Vision Transformer
- ステージ: [2, 6, 36, 4]

# Mask Decoder
- 入力: Image Embeddings + Prompt Embeddings
- 出力: Segmentation Masks [B, 1, H, W]
- マルチマスク出力: サポート（最初のクリックで3マスク）
```

### 2. アダプターモジュールの詳細実装

#### ImageFeatureAdapter
```python
class ImageFeatureAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        # vision_features: [B, N_patches, D_v]
        B, N, D = vision_features.shape
        
        # プロジェクション
        features = self.proj(vision_features)  # [B, N, 256]
        features = self.norm(features)
        
        # 空間的な再形成（正方形を仮定）
        H = W = int(math.sqrt(N))
        features = features.view(B, H, W, -1)  # [B, H, W, 256]
        features = features.permute(0, 3, 1, 2)  # [B, 256, H, W]
        
        return features
```

#### TextPromptProjector
```python
class TextPromptProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256, use_mlp: bool = False):
        super().__init__()
        
        if use_mlp:
            self.proj = nn.Sequential(
                nn.Linear(in_dim, in_dim // 2),
                nn.ReLU(),
                nn.Linear(in_dim // 2, out_dim)
            )
        else:
            self.proj = nn.Linear(in_dim, out_dim)
        
        self.norm = nn.LayerNorm(out_dim)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [num_prompts, D_l] or [B, num_prompts, D_l]
        prompt_embeds = self.proj(hidden_states)
        prompt_embeds = self.norm(prompt_embeds)
        return prompt_embeds
```

### 3. generate_with_masks()の実装詳細

```python
def generate_with_masks(self, input_ids, pixel_values, **kwargs):
    """
    ストリーミング生成とリアルタイムマスク生成
    """
    # 初期化
    generated_ids = input_ids.clone()
    all_masks = [[] for _ in range(B)]
    
    # ビジョン特徴の抽出（一度だけ）
    vision_features = self.extract_vision_features(pixel_values)
    image_features_sam = self.image_adapter(vision_features)
    
    # 生成ループ
    for step in range(max_new_tokens):
        # 次トークンの予測
        outputs = self.qwen(generated_ids, pixel_values=pixel_values, ...)
        next_token_logits = outputs.logits[:, -1, :]
        
        # サンプリングまたはargmax
        if do_sample:
            next_tokens = sample_with_temperature(next_token_logits, temperature)
        else:
            next_tokens = torch.argmax(next_token_logits, dim=-1)
        
        # SEGトークンの検出
        seg_mask = (next_tokens == self.seg_token_id)
        
        if seg_mask.any():
            # SEGトークンが生成された場合、マスクを生成
            for b in range(B):
                if seg_mask[b]:
                    # 隠れ状態をプロンプトに変換
                    seg_hidden = outputs.hidden_states[-1][b, -1, :]
                    prompt_embed = self.text_prompt_proj(seg_hidden)
                    
                    # SAMでマスク生成
                    mask = self.generate_mask_with_sam(
                        image_features_sam[b:b+1],
                        prompt_embed
                    )
                    all_masks[b].append(mask)
        
        # 生成されたトークンを追加
        generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(1)], dim=1)
        
        # EOSチェック
        if (next_tokens == self.tokenizer.eos_token_id).all():
            break
    
    return {
        'generated_ids': generated_ids,
        'generated_text': decode_tokens(generated_ids),
        'masks': all_masks
    }
```

### 4. 学習の詳細

#### 損失関数
```python
def compute_loss(outputs, labels, mask_labels):
    # 言語モデリング損失
    lm_loss = F.cross_entropy(
        outputs.logits.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=-100
    )
    
    # セグメンテーション損失
    seg_loss = 0
    if mask_labels is not None:
        for pred_masks, gt_masks in zip(outputs.mask_logits, mask_labels):
            if pred_masks is not None:
                # Binary Cross Entropy + Dice Loss
                bce_loss = F.binary_cross_entropy_with_logits(pred_masks, gt_masks)
                dice_loss = compute_dice_loss(pred_masks, gt_masks)
                seg_loss += bce_loss + dice_loss
    
    # 合計損失
    total_loss = lm_loss + 0.5 * seg_loss
    return total_loss
```

#### LoRA設定
```python
peft_config = LoraConfig(
    r=16,  # ランク
    lora_alpha=32,  # スケーリング係数
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "up_proj", "down_proj", "gate_proj"      # FFN
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### 5. データ処理の詳細

#### RefCOCOデータセットの処理
```python
def process_refcoco_sample(image, expression, mask):
    # プロンプトの構築
    prompt = f"Human: Please segment {expression} in this image\nAssistant:"
    
    # マスクの処理
    mask = torch.from_numpy(mask).float()
    
    # ラベルの作成（SEGトークンを適切な位置に挿入）
    # "The <SEG> object is located here"のような形式
    
    return {
        'pixel_values': process_image(image),
        'input_ids': tokenize(prompt),
        'labels': create_labels_with_seg(response),
        'mask_labels': mask
    }
```

#### バッチ処理の最適化
```python
class MultiModalDataCollator:
    def __call__(self, features):
        # 可変長シーケンスのパディング
        max_length = max(f['input_ids'].size(0) for f in features)
        
        # 画像サイズの統一（必要に応じてリサイズ）
        target_size = self._determine_target_size(features)
        
        # バッチの構築
        batch = {
            'input_ids': pad_sequences([f['input_ids'] for f in features]),
            'pixel_values': stack_images([f['pixel_values'] for f in features]),
            'attention_mask': create_attention_masks(...),
            'labels': pad_labels([f['labels'] for f in features]),
            'mask_labels': process_masks([f.get('mask_labels') for f in features])
        }
        
        return batch
```

### 6. メモリ最適化

#### 勾配チェックポイント
```python
# Qwenモデルで勾配チェックポイントを有効化
model.qwen.gradient_checkpointing_enable()

# カスタムチェックポイント実装
def custom_forward_with_checkpoint(module, *args):
    return checkpoint(module, *args, use_reentrant=False)
```

#### 混合精度学習
```python
# 自動混合精度の設定
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast(dtype=torch.float16):
    outputs = model(**inputs)
    loss = compute_loss(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 7. 推論の最適化

#### バッチ推論の並列化
```python
def batch_inference(images, instructions, batch_size=4):
    results = []
    
    # データローダーの作成
    dataset = InferenceDataset(images, instructions)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    with torch.no_grad():
        for batch in dataloader:
            # 並列推論
            outputs = model.generate_with_masks(**batch)
            results.extend(process_outputs(outputs))
    
    return results
```

#### キャッシングとメモリ管理
```python
# KVキャッシュの管理
past_key_values = None
for token in generate_tokens():
    outputs, past_key_values = model.forward_with_cache(
        token, 
        past_key_values=past_key_values
    )
```

## パフォーマンス指標

### 推論速度
- 単一画像（512x512）: ~200ms on RTX 3090
- バッチサイズ4: ~600ms on RTX 3090
- トークン生成速度: ~30 tokens/sec

### メモリ使用量
- モデルロード: ~12GB (fp16)
- 推論時（バッチサイズ1）: +2GB
- 学習時（バッチサイズ4）: ~20GB

### 精度指標（期待値）
- RefCOCO val: mIoU ~75%
- RefCOCO+ val: mIoU ~70%
- VQA accuracy: ~80%

## トラブルシューティング

### よくある問題と解決策

1. **CUDA Out of Memory**
   - バッチサイズを削減
   - 勾配累積ステップを増加
   - 画像解像度を下げる

2. **SAMチェックポイントのロード失敗**
   - 正しいconfig pathを指定
   - チェックポイントの整合性を確認

3. **生成品質の問題**
   - 温度パラメータの調整
   - プロンプトエンジニアリング
   - ファインチューニングの追加

## 今後の技術的改善案

1. **Flash Attention 2の統合**
2. **量子化による軽量化**
3. **分散学習のサポート**
4. **オンライン学習機能**