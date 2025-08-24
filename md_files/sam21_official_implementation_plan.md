# SAM2.1公式実装への移行計画

## 1. 現状の問題分析

### 1.1 可視化の位置ずれ問題
- **症状**: 可視化においてPredicted Maskの位置がずれている
  - 横長画像: 下側にスペースが発生
  - 縦長画像: 右側にスペースが発生
- **原因**: アスペクト比を考慮しない直接リサイズによる歪み

### 1.2 現在の実装状況
- **前処理**: `preprocess_sam_image`で1024x1024に直接リサイズ（SAM2公式仕様準拠）
- **正規化**: ImageNet標準（mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]）
- **可視化**: 自前実装でF.interpolateを使用

## 2. SAM2.1公式実装の要点

### 2.1 SAM2Transforms クラス
```python
# 公式実装の特徴
- resolution: 1024（デフォルト）
- 前処理: Resize((resolution, resolution)) + Normalize(mean, std)
- 後処理: postprocess_masks(masks, orig_hw)
  - 連結成分補正（オプション）
  - F.interpolate(masks, orig_hw, mode="bilinear", align_corners=False)
```

### 2.2 重要な公式メソッド
1. **transform_coords**: 座標変換（絶対座標→正規化座標）
2. **transform_boxes**: ボックス座標変換
3. **postprocess_masks**: マスクの後処理（リサイズ＋補正）

## 3. 移行計画

### Phase 1: SAM2Transforms統合（優先度: 高）

#### 3.1 前処理の統一
**現状**:
```python
# src/data/dataset.py
def preprocess_sam_image(image, target_size=1024):
    image = image.resize((target_size, target_size), Image.LANCZOS)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)
```

**改善案**:
```python
from sam2.utils.transforms import SAM2Transforms

class HybridDataset:
    def __init__(self, ...):
        self.sam_transforms = SAM2Transforms(
            resolution=1024,
            mask_threshold=0.0,
            max_hole_area=0.0,
            max_sprinkle_area=0.0
        )
    
    def __getitem__(self, idx):
        # SAM用画像処理
        sam_images = self.sam_transforms(image_pil)
        # 元画像サイズを保存（後処理で必要）
        orig_hw = image_pil.size[::-1]  # (H, W)
```

#### 3.2 マスク後処理の統一
**現状の問題**:
- 可視化時に独自のF.interpolate実装
- アスペクト比の考慮が不完全

**改善案**:
```python
# minimal_train.py - 可視化関数
def save_visualization_step(self, ...):
    # マスクの後処理に公式postprocess_masksを使用
    pred_mask_processed = self.sam_transforms.postprocess_masks(
        pred_mask_logits,  # [B, C, 256, 256]
        orig_hw  # 元画像サイズ
    )
```

### Phase 2: 座標変換の統一（優先度: 中）

#### 3.3 プロンプト座標の処理
```python
# ポイントプロンプトの変換
point_coords = self.sam_transforms.transform_coords(
    coords=point_coords,
    normalize=True,  # 絶対座標の場合
    orig_hw=orig_hw
)

# ボックスプロンプトの変換
box_coords = self.sam_transforms.transform_boxes(
    boxes=boxes,
    normalize=True,
    orig_hw=orig_hw
)
```

### Phase 3: 推論パイプラインの最適化（優先度: 低）

#### 3.4 SAM2ImagePredictorの活用検討
- 推論専用のヘルパークラス
- 学習時は直接使用不可（勾配計算が必要）
- 推論評価時のみ使用を検討

## 4. 実装手順

### Step 1: SAM2Transformsのインポートと初期化
1. `src/data/dataset.py`にSAM2Transformsを導入
2. `preprocess_sam_image`関数をSAM2Transforms呼び出しに置換
3. 元画像サイズ（orig_hw）の保持機構を追加

### Step 2: 可視化の修正
1. `minimal_train.py`の可視化関数を修正
2. `postprocess_masks`を使用してマスクを元サイズに復元
3. アスペクト比を保持した正しい表示

### Step 3: テストと検証
1. 様々なアスペクト比の画像でテスト
2. マスクの位置ずれが解消されていることを確認
3. 学習・推論の精度が維持されていることを確認

### Step 4: コード整理
1. 不要な自前実装の削除
2. コメントとドキュメントの更新
3. デバッグログのクリーンアップ

## 5. 期待される効果

1. **位置ずれ問題の解決**: 公式postprocess_masksによる正確な座標復元
2. **保守性向上**: SAM2公式実装への準拠により将来のアップデートに対応しやすい
3. **コード簡潔化**: 自前実装の削減によるコードベースの簡素化
4. **精度向上の可能性**: 連結成分補正などの公式最適化の恩恵

## 6. 注意事項

1. **後方互換性**: 既存のチェックポイントとの互換性を維持
2. **性能測定**: 変更前後でのベンチマーク実施
3. **段階的移行**: 一度に全て変更せず、段階的に移行
4. **テスト充実**: 各段階で十分なテストを実施

## 7. 参考資料

- SAM2公式実装: `sam2/utils/transforms.py`
- 過去のAI回答: `md_files/ai_answers_files/`
- 公式ドキュメント: [SAM2 GitHub](https://github.com/facebookresearch/sam2)