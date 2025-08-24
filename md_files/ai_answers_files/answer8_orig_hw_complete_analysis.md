# SAM2.1とQwen2.5-VL統合モデル（LISA改）における元画像サイズ情報欠落問題の完全分析

## 問題の詳細

### 現在の症状
1. **可視化時のマスクオーバーレイ不一致**
   - `outputs/minimal_train_20250824_121030/visualizations/step_000005.png`で確認
   - 予測マスクとGTマスクが元画像と正しく重ならない
   - 位置とサイズの両方がずれている

2. **ログから確認できる問題**
   ```
   Using orig_hw from batch: (1024, 1024)
   ```
   - 常に`(1024, 1024)`が返されている
   - 実際の元画像サイズが異なるにも関わらず固定値

## 実装の詳細分析

### データフローの完全な流れ

```
1. 元データセット（SemSegDataset/ReferSegDataset等）
   ↓
2. __getitem__()で10要素タプルを返す
   ↓
3. HybridDataset.__getitem__()で受け取る
   ↓
4. collate_fn()でバッチ化
   ↓
5. MinimalTrainer.save_visualization()で可視化
```

### 1. 元データセット側の実装（SemSegDataset._get_semseg_item）

```python
# src/data/sem_seg_dataset.py (Line 556-557)
orig_h, orig_w = image.shape[:2]
coord_transform = CoordinateTransform(orig_size=(orig_h, orig_w))

# Line 600-612: SAM用画像の前処理
pad_h = sam_size - h
pad_w = sam_size - w
image_for_sam = cv2.copyMakeBorder(
    image_for_sam, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0)
)
image_for_sam = torch.from_numpy(image_for_sam).permute(2, 0, 1).float() / 255.0
# SAM2.1の正規化
sam_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
sam_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
image_for_sam = (image_for_sam - sam_mean) / sam_std
resize = image.shape[:2]  # ← 注意：前処理前のサイズ

# Line 690-701: 返り値
return (
    image_path,        # 0: 画像パス
    image_for_sam,     # 1: SAM用前処理済み画像 (torch.Tensor)
    image_for_qwen,    # 2: Qwen用前処理済み画像 (torch.Tensor)
    conversations,     # 3: 会話形式のテキスト
    masks,             # 4: マスク
    label_tensor,      # 5: ラベル
    resize,            # 6: リサイズ情報 (Tuple) ← 元画像サイズ
    questions,         # 7: 質問リスト
    sampled_classes,   # 8: クラス名リスト
    coord_transform    # 9: 座標変換オブジェクト ← orig_size含む
)
```

**重要な発見**：
- `coord_transform`オブジェクトは`orig_size=(orig_h, orig_w)`を持っている
- `resize`タプルも元画像サイズを保持している
- しかし、これらの情報がHybridDatasetで正しく抽出されていない

### 2. HybridDataset側の実装（現在の修正後）

```python
# src/data/dataset.py (Line 676-703)
if len(sample) == 10:
    # 10要素形式
    image_path, image_sam, image_qwen_tensor, conversations, masks, label, resize, questions, sampled_classes, coord_transform = sample
    
    # 元画像サイズを取得（image_pathから読み込む）
    if isinstance(image_path, str) and os.path.exists(image_path):
        try:
            with Image.open(image_path) as img:
                orig_hw = (img.height, img.width)
        except:
            # coord_transformから取得を試みる
            if hasattr(coord_transform, 'original_height') and hasattr(coord_transform, 'original_width'):
                orig_hw = (coord_transform.original_height, coord_transform.original_width)
            elif hasattr(coord_transform, 'orig_size'):
                orig_hw = coord_transform.orig_size
            else:
                orig_hw = (1024, 1024)
```

**問題点**：
1. `image_path`が相対パスまたは無効なパスの可能性
2. `coord_transform.orig_size`の属性名が間違っている可能性
3. `resize`タプルを使用していない

### 3. CoordinateTransformクラスの実装確認が必要

```python
# src/utils/coordinate_transform.py
class CoordinateTransform:
    def __init__(self, orig_size):
        self.orig_size = orig_size  # (H, W)形式で保存されているはず
```

## 根本原因の特定

### 主要な問題
1. **属性名の不一致**
   - `coord_transform`は`orig_size`属性を持っているが、コードは`original_height`と`original_width`を先に探している
   - 順序を逆にすべき

2. **resizeタプルの未使用**
   - 9要素形式では`resize`が6番目の要素として渡されるが、これを使用していない
   - `resize`は元画像サイズを直接含んでいる

3. **image_pathの問題**
   - 相対パスの場合、`os.path.exists()`がFalseを返す
   - `base_image_dir`との結合が必要

## 解決策の提案

### 即座に実装可能な修正

```python
# HybridDataset.__getitem__の修正案
if len(sample) == 10:
    image_path, image_sam, image_qwen_tensor, conversations, masks, label, resize, questions, sampled_classes, coord_transform = sample
    
    # 1. まずcoord_transformから取得を試みる（最も信頼性が高い）
    if hasattr(coord_transform, 'orig_size'):
        orig_hw = coord_transform.orig_size  # (H, W)タプル
    # 2. resizeタプルを使用（元画像サイズが直接入っている）
    elif resize is not None and isinstance(resize, tuple) and len(resize) == 2:
        orig_hw = resize
    # 3. 最後の手段：image_pathから読み込む
    elif isinstance(image_path, str):
        # base_image_dirとの結合を試みる
        full_path = image_path
        if hasattr(self, 'base_image_dir') and not os.path.isabs(image_path):
            full_path = os.path.join(self.base_image_dir, image_path)
        
        if os.path.exists(full_path):
            try:
                with Image.open(full_path) as img:
                    orig_hw = (img.height, img.width)
            except:
                orig_hw = (1024, 1024)
        else:
            orig_hw = (1024, 1024)
    else:
        orig_hw = (1024, 1024)

# 9要素形式も同様に修正
elif len(sample) == 9:
    image_path, image_sam, image_qwen_tensor, conversations, masks, label, resize, questions, sampled_classes = sample
    
    # resizeから直接取得（9要素形式では最も信頼性が高い）
    if resize is not None and isinstance(resize, tuple) and len(resize) == 2:
        orig_hw = resize
    # 以下同様...
```

### 長期的な解決策

1. **データセット仕様の統一**
   - すべてのデータセットが`orig_hw`を明示的に返すように修正
   - 11要素形式として`orig_hw`を追加

2. **前処理の遅延実行**
   - HybridDatasetで画像の前処理を行い、元画像サイズを確実に保持

3. **メタデータクラスの導入**
   ```python
   @dataclass
   class SampleMetadata:
       orig_hw: Tuple[int, int]
       coord_transform: Optional[CoordinateTransform]
       resize_info: Optional[Any]
   ```

## 質問

1. **CoordinateTransformクラスの正確な実装**
   - `orig_size`属性の形式は`(H, W)`か`(W, H)`か？
   - 他にどのような属性・メソッドがあるか？

2. **resizeタプルの形式**
   - 常に`(H, W)`形式か？
   - パディング情報も含まれているか？

3. **SAM2.1の公式実装での対処法**
   - SAM2.1の公式実装では元画像サイズ情報をどのように保持しているか？
   - postprocess_masksの正しい使用方法は？

4. **LISAオリジナルの実装**
   - オリジナルLISAではこの問題をどう解決していたか？
   - SAM1時代のResizeLongestSideからの移行でどのような変更が必要か？

5. **パフォーマンスへの影響**
   - image_pathから毎回画像を開くことのI/Oコストは許容範囲か？
   - キャッシュ機構を導入すべきか？

## 推奨される次のステップ

1. **即座の修正**：上記の修正案を実装し、`coord_transform.orig_size`と`resize`タプルを優先的に使用
2. **デバッグログの追加**：取得したorig_hwの値をログ出力して確認
3. **テスト**：異なるデータセットで元画像サイズが正しく取得できることを確認
4. **長期的改善**：データセット仕様の統一を検討

これらについて、Qwen2.5-VL、SAM2.1、LISA等の公式実装を参考にした解決策を教えてください。