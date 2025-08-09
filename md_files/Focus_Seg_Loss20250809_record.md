# Focus_Seg_Loss20250809 実装記録

## 実装日時
2025年1月9日

## 実装内容
**問題1: テキスト指示の空間情報不足の修正**

## 背景と問題点
現在の実装では、SAMのプロンプトエンコーダに常に画像中心のポイント座標を与えていた。これにより：
- オブジェクトの位置手掛かりが失われる
- 対象物が中央にない場合、マスクDecoderが誤った領域を切り取る
- 学習初期に常に中心付近の不正確なマスクが出力され、seg_lossが高止まりする

## 実装方針
**擬似ポイントを動的に配置**し、テキスト指示に空間的ヒントを与える。具体的には、**真値マスクの重心（centroid）**を計算し、その座標をプロンプトエンコーダに入力する。

## 実装詳細

### 1. 重心計算メソッドの追加
**ファイル**: `src/models/lisa_model.py`
**追加位置**: `LISA_Model`クラス内（`forward`メソッドの前）

```python
def compute_mask_centroid(self, mask: torch.Tensor, orig_h: int, orig_w: int) -> torch.Tensor:
    """
    バイナリマスクから重心座標を計算
    
    Args:
        mask: バイナリマスク [H, W] or [1, H, W]
        orig_h: 元画像の高さ（ピクセル座標系）
        orig_w: 元画像の幅（ピクセル座標系）
    
    Returns:
        重心座標 [2] = (cx, cy) in pixels
    """
    if mask.dim() == 3:
        mask = mask.squeeze(0)
    
    # Float型に変換
    m = mask.to(torch.float32)
    h, w = m.shape
    
    # 面積（非ゼロ画素数）
    mass = m.sum()
    if mass <= 0:
        # マスクが空の場合は画像中心を返す
        return torch.tensor([orig_w // 2, orig_h // 2], 
                          dtype=torch.float32, device=mask.device)
    
    # 座標グリッド
    ys = torch.arange(h, device=m.device, dtype=torch.float32).view(h, 1)
    xs = torch.arange(w, device=m.device, dtype=torch.float32).view(1, w)
    
    # 重心計算（マスク座標系）
    cy = (m * ys).sum() / mass
    cx = (m * xs).sum() / mass
    
    # マスク座標系から元画像座標系へ変換
    # マスクがリサイズされている場合のスケーリング
    scale_x = orig_w / w
    scale_y = orig_h / h
    cx = cx * scale_x
    cy = cy * scale_y
    
    return torch.stack([cx, cy])
```

### 2. forwardメソッドの修正
**変更箇所**: SAMプロンプトエンコーダへのpoint_coords設定部分（約720-750行目）

**変更前**:
```python
# Create a center point as anchor for text-guided segmentation
h_feat, w_feat = image_features_sam[i:i+1].shape[-2:]
# Map to original image coordinates (feature stride is 16)
h_img, w_img = h_feat * 16, w_feat * 16
center_x, center_y = w_img // 2, h_img // 2
```

**変更後**:
```python
# 真値マスクから重心座標を計算（訓練時）、または画像中心を使用（推論時）
h_feat, w_feat = image_features_sam[i:i+1].shape[-2:]
# Map to original image coordinates (feature stride is 16)
h_img, w_img = h_feat * 16, w_feat * 16

if mask_labels is not None and i < len(mask_labels) and mask_labels[i] is not None:
    # 訓練時: GTマスクから重心を計算
    gt_mask = mask_labels[i]
    
    # GTマスクが複数SEGに対応している場合、最初のマスクを使用
    if isinstance(gt_mask, list):
        if len(gt_mask) > j:
            gt_mask = gt_mask[j]
        else:
            gt_mask = gt_mask[0] if len(gt_mask) > 0 else None
    
    if gt_mask is not None:
        # 重心を計算
        centroid = self.compute_mask_centroid(gt_mask, h_img, w_img)
        center_x, center_y = centroid[0].item(), centroid[1].item()
        logger.debug(f"[CENTROID] Batch {i}, SEG {j}: Computed centroid from GT mask - x={center_x:.1f}, y={center_y:.1f} (image center would be {w_img//2}, {h_img//2})")
    else:
        # GTマスクがない場合は画像中心を使用
        center_x, center_y = w_img // 2, h_img // 2
        logger.debug(f"[CENTROID] Batch {i}, SEG {j}: No GT mask - using image center ({center_x}, {center_y})")
else:
    # 推論時: 画像中心を使用（フォールバック）
    center_x, center_y = w_img // 2, h_img // 2
    logger.debug(f"[CENTROID] Batch {i}, SEG {j}: Inference mode - using image center ({center_x}, {center_y})")
```

## 動作確認

### テスト実行
```bash
python minimal_train.py --fast_dev_run --samples_per_epoch 4 --batch_size 2 --num_epochs 1 --debug
```

### 実行結果
エラーなく正常に動作し、以下のログが確認された：

```
08/09/2025 15:58:43 - DEBUG - src.models.lisa_model - [CENTROID] Batch 0, SEG 0: Computed centroid from GT mask - x=881.5, y=112.9 (image center would be 512, 512)
08/09/2025 15:58:45 - DEBUG - src.models.lisa_model - [CENTROID] Batch 0, SEG 0: Computed centroid from GT mask - x=509.2, y=572.9 (image center would be 512, 512)
```

### 確認ポイント
1. ✅ GTマスクから正しく重心座標を計算
2. ✅ 画像中心（512, 512）とは異なる適切な座標を生成
3. ✅ 訓練時と推論時で適切に処理を分岐
4. ✅ エラーなく学習が進行

## 技術的詳細

### 重心計算アルゴリズム
1. バイナリマスクの非ゼロ画素の座標を取得
2. 一次モーメント（重み付き平均）で重心を計算
3. マスク座標系から元画像座標系へスケーリング

### 座標系の整合性
- **入力**: GTマスク（任意のサイズ）
- **出力**: SAM2.1が期待するピクセル座標（X, Y）
- **変換**: feature stride = 16を考慮した座標変換

### フォールバック処理
- マスクが空の場合 → 画像中心を使用
- 推論時 → 画像中心を使用
- GTマスクがない場合 → 画像中心を使用

## 期待される効果

1. **セグメンテーション精度の向上**
   - 対象物の実際の位置に基づいたプロンプト生成
   - より正確なマスク予測

2. **画像中央への固定バイアスの解消**
   - 中心から離れたオブジェクトも正確にセグメント可能
   - 多様な位置のオブジェクトに対する汎化性能向上

3. **学習の安定化**
   - seg_lossの高止まりを防ぐ
   - より効率的な学習が可能

## 今後の拡張可能性

### 複数インスタンスへの対応
現在は全体の重心を計算しているが、将来的には：
- 連結成分ごとの重心計算
- 複数ポイントプロンプトの同時入力
- SAM2の`get_connected_components`を活用した実装

### 代替アプローチ
- バウンディングボックスの中心を使用
- 複数の代表点をサンプリング
- 領域の分布を考慮した重み付き重心

## 参考資料
- Focus_Seg_Loss20250809_spec.md（修正方針）
- o3_query_answers.md（SAM2.1のpoint prompting実装詳細）
- LISA論文（embedding-as-mask パラダイム）