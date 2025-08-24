# 学習時の可視化における予測マスクのパディング問題に関するAI Query

## 現在の実装状況と改善内容

### 1. 実装済みの修正内容

#### 1.1 座標系管理システムの実装
- **SamSquareMeta**クラスを作成し、SAM前処理のメタデータを管理
  - アスペクト比保持+右下パディング方式の実装
  - 元画像サイズ、リサイズ後サイズ、パディング情報を記録
  - 逆変換を可能にするメタデータ構造

```python
@dataclass
class SamSquareMeta:
    raw_h: int  # 元画像の高さ
    raw_w: int  # 元画像の幅
    side: int = 1024  # SAMの入力サイズ
    scale: float = 1.0  # リサイズスケール
    new_h: int = 0  # リサイズ後の高さ
    new_w: int = 0  # リサイズ後の幅
    pad_top: int = 0  # 上パディング（通常0）
    pad_left: int = 0  # 左パディング（通常0）
    pad_bottom: int = 0  # 下パディング
    pad_right: int = 0  # 右パディング
```

#### 1.2 SAM前処理の修正
- **SemSegDataset**を修正し、正しいSAM前処理を適用
  - アスペクト比保持+パディング方式（ストレッチしない）
  - apply_sam_transform_to_image/apply_sam_transform_to_mask関数を使用
  - sam_metaを10番目の要素として返す

#### 1.3 損失計算の統一
- **compute_loss**メソッドをSAM座標系（1024x1024）で統一
  - 予測マスクをbilinearで1024x1024にアップサンプル
  - GTマスクもSAM座標系に合わせる（既に1024x1024のはず）
  - nearest-exactモードを使用（PyTorchのnearestの既知の問題を回避）

#### 1.4 可視化の修正
- **save_visualization**メソッドを原寸ベースに修正
  - orig_hwを使用して元画像サイズを取得
  - パディングを考慮した逆変換を実装
  - 元画像、予測マスク、GTマスクをすべて原寸で表示

### 2. 現在の問題

#### 2.1 症状
- **GTマスクと画像の位相は揃った** ✅
- **推論時の可視化は正常** ✅
- **学習時の予測マスクのみ位相がずれる** ❌
  - サイズは正しい（元画像サイズになっている）
  - 下側と右側にパディング様のスペースがあり、マスクが左上に偏っている
  - つまり、マスクの有効領域が画像全体ではなく左上の一部に圧縮されている

#### 2.2 原因の推定
予測マスクの生成過程で以下の問題が発生している可能性：

1. **LISA_Model.forward**での問題
   - 予測マスクのアップサンプル時に誤った目標サイズを使用
   - 現在のコード（1111行目付近）：
   ```python
   # Upscale mask to original image size
   if image_grid_thw is not None:
       H_grid_raw = int(image_grid_thw[i, 1].item())
       W_grid_raw = int(image_grid_thw[i, 2].item())
       orig_h = H_grid_raw * 14
       orig_w = W_grid_raw * 14
   ```
   - これはQwenの画像サイズ（448x448程度）を計算している可能性

2. **座標系の混乱**
   - 予測マスクは392x392で生成される（Qwenベースのサイズ）
   - これを1024x1024にアップサンプルすべきだが、448x448程度にアップサンプルしている
   - save_visualizationで1024x1024から元サイズに戻す際、パディング領域が残る

### 3. コード分析

#### 3.1 データフロー
```
1. 入力画像（例：640x480）
   ↓
2. SAM前処理
   - アスペクト比保持リサイズ: 640x480 → 1024x768
   - 右下パディング: 1024x768 → 1024x1024
   - sam_metaに変換情報を記録
   ↓
3. Qwen前処理  
   - アスペクト比保持リサイズ: 640x480 → 448x336
   - 14の倍数パディング: 448x336 → 448x336（変更なし）
   ↓
4. モデル推論
   - SAM MaskDecoder: 256x256の低解像度マスク生成
   - アップサンプル: 256x256 → ???（ここが問題）
   ↓
5. 損失計算
   - 予測マスク: 392x392（なぜか小さい）
   - GTマスク: 1024x1024
   - 両方を1024x1024に統一して計算
   ↓
6. 可視化
   - 予測マスク: 1024x1024 → 640x480（パディング除去）
   - GTマスク: 1024x1024 → 640x480（パディング除去）
   - 問題: 予測マスクの有効領域が左上に偏っている
```

#### 3.2 問題の核心
- **LISA_Model.forward**で、予測マスクのアップサンプル目標サイズが間違っている
- image_grid_thwから計算される`orig_h, orig_w`はQwenの入力サイズ（448x336など）
- これをSAMサイズ（1024x1024）にすべき

### 4. 質問

1. **SAM2.1の公式実装では、MaskDecoderの出力をどのサイズにアップサンプルすべきか？**
   - 256x256 → 1024x1024（SAM入力サイズ）が正しい？
   - それとも元画像サイズに直接アップサンプル？

2. **Qwen2.5-VLとSAM2.1を統合する際のベストプラクティスは？**
   - 予測マスクは常にSAM座標系（1024x1024）で生成すべき？
   - それともQwen座標系で生成して後で変換？

3. **LISAの公式実装での扱いは？**
   - LISAではどの座標系で予測マスクを生成している？
   - 学習時と推論時で異なる処理をしている？

### 5. 修正案

#### Option A: 予測マスクを常にSAM座標系（1024x1024）で生成
```python
# LISA_Model.forwardの修正
mask_logit = F.interpolate(
    low_res_masks,
    size=(1024, 1024),  # 常にSAMサイズ
    mode='bilinear',
    align_corners=False
)
```

#### Option B: sam_metaを使用して正しいサイズを計算
```python
# バッチにsam_metaを含める
if 'sam_meta' in batch and batch['sam_meta'] is not None:
    sam_meta = batch['sam_meta'][i]
    target_h = sam_meta.side  # 1024
    target_w = sam_meta.side  # 1024
    mask_logit = F.interpolate(
        low_res_masks,
        size=(target_h, target_w),
        mode='bilinear',
        align_corners=False
    )
```

#### Option C: 動的にSAMサイズを取得
```python
# sam_imagesのサイズから動的に取得
if sam_images is not None:
    target_h, target_w = sam_images.shape[-2:]  # 1024, 1024
    mask_logit = F.interpolate(
        low_res_masks,
        size=(target_h, target_w),
        mode='bilinear',
        align_corners=False
    )
```

### 6. 期待する回答

1. **即座に修正すべき箇所**
   - LISA_Model.forwardのマスクアップサンプル部分
   - 正しい目標サイズの計算方法

2. **ベストプラクティス**
   - Qwen2.5-VL、SAM2.1、LISAの公式実装での座標系管理方法
   - 学習時と推論時で統一すべき処理

3. **将来的な改善点**
   - より堅牢な座標系管理システムの設計
   - デバッグしやすいログ出力の追加

これらについて、Qwen2.5-VL、SAM2.1、LISA等の公式実装を参考にした解決策を教えてください。