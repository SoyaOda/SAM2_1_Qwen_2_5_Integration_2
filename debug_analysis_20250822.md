# SAM2.1 + Qwen2.5-VL 統合モデル デバッグ分析レポート
日付: 2025-08-22

## 問題の概要
学習時の予測マスクに格子状のスリットパターンとランダムな幾何学的模様が現れる問題。
推論時は正常に動作しているため、学習時特有の問題と考えられる。

## 発見された主要な問題点

### 1. SAM画像前処理の不一致

#### 学習時（src/data/dataset.py）
```python
def preprocess_sam_image(image: Image.Image, target_size: Optional[int] = None) -> torch.Tensor:
    # 1. 最長辺を1024にリサイズ（アスペクト比保持）
    # 2. 1024x1024にパディング（黒で埋める）
    # 3. 正規化: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
```

#### 推論時（minimal_train.py内の評価、test_inference_v2.py）
```python
# SAM用の高解像度画像を準備
sam_image = np.array(image.resize((1024, 1024)))  # アスペクト比を無視して直接リサイズ
sam_image_tensor = torch.from_numpy(sam_image).permute(2, 0, 1).float() / 255.0  # 単純な0-1正規化
```

### 2. 正規化の不一致
- **学習時**: SAM2.1標準の正規化（ImageNet正規化）を使用
- **推論時**: 単純な0-1正規化のみ

### 3. リサイズ方法の違い
- **学習時**: アスペクト比を保持して最長辺を1024にし、パディングで正方形に
- **推論時**: 直接1024x1024にリサイズ（アスペクト比が歪む可能性）

## SAM ViTフローの分析

### 学習時のフロー（src/models/lisa_model.py）
```python
if sam_images is not None:
    with torch.no_grad():  # SAM ImageEncoderは凍結
        sam_encoder_dtype = next(self.sam_image_encoder.parameters()).dtype
        sam_images_converted = sam_images.to(dtype=sam_encoder_dtype)
        backbone_out = self.sam_image_encoder(sam_images_converted)
```

- SAM ImageEncoderは凍結されており、学習されていない
- dtype変換は行われているが、前処理の不一致により入力分布が異なる

## 可視化コードの分析

### 予測マスクの処理（minimal_train.py）
```python
# マスクを取得（バッチの最初のサンプル）
pred_mask = outputs.mask_logits[0]
if isinstance(pred_mask, list):
    pred_mask = pred_mask[0]

# 予測マスクをシグモイドで確率に変換
pred_mask_np = torch.sigmoid(pred_mask).detach().cpu().numpy()
if pred_mask_np.ndim > 2:
    pred_mask_np = pred_mask_np.squeeze()

# 可視化時のリサイズ
pred_mask_resized = cv2.resize(pred_mask_np, (image_np.shape[1], image_np.shape[0]), 
                               interpolation=cv2.INTER_LINEAR)
```

可視化処理自体は正常で、格子状パターンは実際の予測結果を反映していると考えられる。

## 根本原因の推定

1. **主要因**: SAM画像の前処理不一致
   - 学習時はImageNet正規化、推論時は0-1正規化
   - SAM ImageEncoderは事前学習済みモデルであり、ImageNet正規化を前提としている
   - 不適切な正規化により、SAM特徴抽出が機能していない可能性

2. **副次的要因**: アスペクト比の扱い
   - 学習時はパディング、推論時は直接リサイズ
   - これによりSAM特徴の空間的な対応関係が崩れる可能性

## 推奨される修正

### 優先度1: 推論時のSAM画像前処理を学習時と統一
```python
# minimal_train.pyの評価部分を修正
from src.data.dataset import preprocess_sam_image

# 修正前
sam_image = np.array(image.resize((1024, 1024)))
sam_image_tensor = torch.from_numpy(sam_image).permute(2, 0, 1).float() / 255.0

# 修正後
sam_image_tensor = preprocess_sam_image(image, target_size=1024)
```

### 優先度2: デバッグログを追加
SAM特徴の統計値をログ出力し、学習時と推論時の差異を確認

### 優先度3: SAM2.1の公式実装との整合性確認
公式のSAM2.1がどのような前処理を期待しているか再確認