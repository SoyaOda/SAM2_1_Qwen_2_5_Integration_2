# SAM2.1の公式postprocess実装について

## 質問背景
LISA改（Qwen2.5-VL + SAM2.1統合モデル）で位置ずれ問題を解決するため、SAMのpostprocess機能を実装しようとしています。

## 現在の実装
独自に`sam_postprocess_mask`関数を実装しました：
```python
def sam_postprocess_mask(mask, orig_size, input_size_after_resize=None, encoder_img_size=1024):
    """
    SAM標準のpostprocess: パディング除去と元解像度への復元
    """
    # 1. ResizeLongestSide後のサイズを計算
    # 2. パディングを除去（右下パディング前提）
    # 3. 元の解像度にリサイズ（bilinear, align_corners=False）
```

## 質問事項

### 1. SAM2.1に公式のpostprocess関数はありますか？
- `sam2`パッケージ内に標準的なpostprocess機能があるはずですが、具体的にどのモジュール/クラスにありますか？
- 例：`sam2.utils.transforms`、`sam2.predictor`、`sam2.postprocessing`など？

### 2. 公式実装の使用方法
もし公式実装がある場合：
- インポート方法：`from sam2.??? import ???`
- 使用例：
```python
# マスクのpostprocess
postprocessed_mask = official_postprocess(
    masks=predicted_masks,
    input_size=???,
    original_size=???
)
```

### 3. SAM2.1のSamPredictor内部での処理
`SamPredictor`クラスでは内部的にpostprocessを行っているはずです：
- `predict`メソッドの中でどのようにpostprocessを呼んでいますか？
- 特に`postprocess_masks`メソッドや類似の機能はありますか？

### 4. 正確な実装詳細
公式実装では以下をどう処理していますか：
- ResizeLongestSide変換の逆変換
- パディングの除去（右下パディングの仮定）
- 元解像度への復元（interpolation設定）
- バッチ処理対応

## 期待する回答
1. **公式postprocess関数の正確な場所とインポート方法**
2. **使用例のコードスニペット**
3. **独自実装 vs 公式実装のメリット・デメリット**

## 参考情報
- SAM2.1を使用（sam2-hiera-large）
- 画像は1024x1024にリサイズ+右下パディング
- マスクは256x256 → 1024x1024 → 元解像度という流れ

これらについて、SAM2.1の公式実装を参考にした正確な情報を教えてください。