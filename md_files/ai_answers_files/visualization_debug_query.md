# Visualization Debug Query

## 現状の問題
学習中のVisualizationが正しく作成されていない。具体的には：

1. **エラー1（修正済み）**: `NameError: name 'F' is not defined` 
   - `torch.nn.functional as F`のインポートが抜けていた → 修正済み

2. **問題2（未解決）**: `orig_hw`が常に`(1024, 1024)`になっている
   - ログ出力：`Using orig_hw from batch: (1024, 1024)`
   - 期待値：元画像の実際のサイズ（例：`(480, 640)`）

## コードの現状

### データセット側（src/data/dataset.py - HybridDataset.__getitem__）
```python
# 返り値の構築
return {
    'orig_hw': orig_hw,  # SAM後処理用の元画像サイズ（必ず設定）
    'original_image': image_pil,  # 可視化用の元画像（PIL形式）
    # ... 他のフィールド
}
```

### Collator側（src/data/collators.py - MultiModalDataCollator）
```python
# orig_hwとoriginal_imagesの処理方法が不明
```

### 可視化側（minimal_train.py - save_visualization）
```python
# orig_hwを取得（リスト形式に対応）
if 'orig_hw' in batch and batch['orig_hw'] is not None:
    # ...処理
```

## 調査が必要な点

1. **MultiModalDataCollatorでの`orig_hw`の扱い**
   - HybridDatasetから返される`orig_hw`がCollatorでどのように処理されているか？
   - バッチ化の際に正しく保持されているか？

2. **`original_images`の扱い**
   - PIL画像のリストとして正しく渡されているか？
   - Collatorでドロップされていないか？

3. **SemSegDatasetでの`orig_hw`設定**
   - 実際の画像サイズが正しく記録されているか？
   - coord_transformオブジェクトのorig_sizeは使われているか？

## 質問

1. PyTorchのDataLoaderとcustom collate_fnを使用する際、非テンソルデータ（`orig_hw`のタプル、`original_image`のPIL画像）を正しくバッチに含める標準的な方法は？

2. 学習時の可視化で、以下の3つの座標系を正しく統一する方法：
   - 元画像の座標系（例：640x480）
   - SAM入力の座標系（1024x1024、パディング付き）
   - モデル出力の座標系（可変、例：448x448）

3. デバッグのため、DataLoaderから取得したbatchの全フィールドとその型・形状を詳細に出力する効率的な方法は？

これらについて、Qwen2.5-VL、SAM2.1、LISA等の公式実装を参考にした解決策を教えてください。