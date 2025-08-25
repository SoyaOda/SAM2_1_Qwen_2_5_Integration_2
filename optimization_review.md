# コード効率化レビュー結果

## 1. デバッグログの問題

### 現状の問題点
- `logger.debug()`が約40箇所以上存在
- 特に`save_visualization`メソッド内に多数のデバッグログ
- 本番環境でも常に文字列フォーマットが実行される（ログレベルに関わらず）

### 最適化案
```python
# 現在のコード（非効率）
logger.debug(f"[DEBUG] Batch keys: {batch.keys()}")
logger.debug(f"[DEBUG] batch['orig_hw'] type: {type(batch['orig_hw'])}")

# 改善案（条件付きログ）
if logger.isEnabledFor(logging.DEBUG):
    logger.debug(f"[DEBUG] Batch keys: {batch.keys()}")
```

### 影響
- 文字列フォーマットのオーバーヘッド削減
- 特に大きなテンソルやリストの文字列化を回避

## 2. テンソル操作の非効率性

### 問題のあるパターン

#### a) 不要な`.detach().cpu().numpy()`チェーン
```python
# 現在（1299行目）
pred_mask_np = torch.sigmoid(pred_mask_processed).squeeze().detach().cpu().numpy()

# 改善案
with torch.no_grad():  # 勾配計算を無効化
    pred_mask_np = torch.sigmoid(pred_mask_processed).squeeze().cpu().numpy()
```

#### b) 重複する形状変換
```python
# 現在のコード（複数箇所で同じ処理）
if pred_mask.dim() == 2:
    pred_mask = pred_mask.unsqueeze(0).unsqueeze(0)
elif pred_mask.dim() == 3:
    if pred_mask.shape[0] == 1:
        pred_mask = pred_mask.unsqueeze(0)
    else:
        pred_mask = pred_mask.unsqueeze(1)

# 改善案（関数化）
def ensure_4d_mask(mask):
    """マスクを[B, C, H, W]形式に統一"""
    if mask.dim() == 2:
        return mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        return mask.unsqueeze(0) if mask.shape[0] == 1 else mask.unsqueeze(1)
    return mask
```

## 3. 可視化処理の最適化

### 現状の問題点
- 毎回SAM画像全体をdenormalizeしている
- 使わない場合でも`original_image_np`の復元処理を実行

### 改善案
```python
# 元画像が必要な場合のみ復元
if original_image_np is None and self.visualize:
    # SAM画像から復元処理
    ...
```

## 4. メモリリークの可能性

### 問題箇所
```python
# save_visualization内（複数のnumpy配列作成）
pred_colored = np.zeros_like(original_image_np, dtype=np.float32)
gt_colored = np.zeros_like(original_image_np, dtype=np.float32)
compare_colored = np.zeros_like(original_image_np, dtype=np.float32)
```

### 改善案
```python
# 一つの配列を再利用
overlay = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
# 赤チャンネルに予測マスクを設定
overlay[:, :, 0] = (pred_binary * 255).astype(np.uint8)
```

## 5. データローダーの効率性

### src/data/dataset.py の問題点

#### a) 毎回PIL画像の変換
```python
# 現在（HybridDataset.__getitem__）
if isinstance(image_qwen, torch.Tensor):
    # 正規化を元に戻す処理が毎回実行
    qwen_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    qwen_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    ...
```

### 改善案
```python
# クラス変数として定義
class HybridDataset:
    QWEN_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    QWEN_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
```

## 6. 不要な変数定義

### 削除可能な変数
```python
# minimal_train.py
self.sam_transforms = SAM2Transforms(...)  # 推論時のみ使用？

# save_visualization内
quality_score = 1.0  # 使われていない
loss_weight = 1.0    # 使われていない
```

## 7. compute_loss の最適化

### 現状の問題
- 毎回SAMサイズ（1024）を変数定義
- アサーションが本番でも実行される

### 改善案
```python
class MinimalTrainer:
    SAM_SIZE = 1024  # クラス定数として定義
    
    def compute_loss(self, ...):
        # アサーションをデバッグモードのみに
        if self.debug:
            assert pred_mask_sam.shape == (self.SAM_SIZE, self.SAM_SIZE)
```

## 8. バッチ処理の効率化

### 現状の問題
```python
# collators.pyで個別に処理
for f in features:
    if 'mask_labels' in f:
        mask_labels.append(f['mask_labels'])
```

### 改善案
```python
# リスト内包表記を使用
mask_labels = [f.get('mask_labels', torch.zeros(1, 1024, 1024)) for f in features]
```

## 推定される改善効果

1. **処理速度**: 約10-15%の高速化
   - デバッグログのオーバーヘッド削減
   - テンソル操作の最適化

2. **メモリ使用量**: 約20-30%削減
   - 不要な中間変数の削除
   - numpy配列の再利用

3. **GPU利用効率**: 改善
   - 不要な`.detach()`の削除
   - バッチ処理の最適化

## 実装優先度

1. **高優先度**（すぐに実装すべき）
   - デバッグログの条件付き実行
   - テンソル定数のクラス変数化
   - 不要な変数の削除

2. **中優先度**（パフォーマンステスト後）
   - 形状変換の関数化
   - numpy配列の再利用

3. **低優先度**（必要に応じて）
   - バッチ処理の最適化
   - PIL画像変換の最適化