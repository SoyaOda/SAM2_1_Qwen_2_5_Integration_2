# LISA改 マスク位置オフセット問題の診断と解決

## 問題の概要
LISA改（Qwen2.5-VL + SAM2.1統合モデル）において、予測マスクが正しいオブジェクトを検出しているものの、**位置が上方向にずれている**問題が発生しています。

例：芝生を検出対象としているが、実際の芝生より上の位置にマスクが生成される

## 現在のモデル構造

### 画像処理パイプライン
```
入力画像 → 2つの異なる解像度で処理：
1. Qwen経路: 448×448 (spatial_merge_size=2でパッチマージ)
2. SAM経路: 1024×1024 (高解像度維持)
```

### 特徴融合プロセス
```python
# Qwen特徴とSAM特徴の融合
1. Qwen特徴: [B, 256, H_q, W_q] (H_q, W_q は可変、通常14×14や16×16)
2. SAM特徴: [B, 256, 64, 64] (固定サイズ)
3. Qwen特徴をSAMと同じ64×64にバイリニア補間でリサイズ
4. Sigma-Add Fusion: SAM特徴 + β * Qwen特徴（リサイズ済み）
```

### 観測されたログ（step 5の例）
```
[SAM ViT] Processing SAM images with shape: torch.Size([2, 3, 1024, 1024])
[SAM ViT] sam_image_embedding shape: torch.Size([2, 256, 64, 64])
[Feature Fusion] Qwen features shape before resize: torch.Size([2, 256, 14, 14])
[Feature Fusion] Qwen features shape after resize: torch.Size([2, 256, 64, 64])
[Feature Fusion] beta_scaled value: 0.5000
[Feature Fusion] fused_image_embedding shape: torch.Size([2, 256, 64, 64])
[SAM Decoder] low_res_masks shape: torch.Size([1, 1, 256, 256])
[SAM Decoder] Upsampled mask shape: torch.Size([1, 1, 1024, 1024])
```

## 位置ずれの原因候補

### 1. 座標系アライメントの問題
**Qwen特徴の14×14を64×64にリサイズする際の位置情報の歪み**
- 14×14の各セルが約4.57×4.57のSAMピクセルに対応
- バイリニア補間により、境界付近の位置情報が曖昧になる
- 特に上下左右の端での誤差が累積

### 2. Spatial Mergeの影響
**Qwen2.5-VLのspatial_merge_size=2による位置情報の粗視化**
- 2×2パッチの平均化により、細かい位置情報が失われる
- 元の28×28グリッドが14×14に縮小される際、位置精度が低下
- マージ処理が上方向にバイアスを持つ可能性

### 3. パディング戦略の不一致
**異なる前処理による座標系のミスマッチ**
- Qwen: 448×448でcenter-crop/padding
- SAM: 1024×1024でresize+padding
- 両者のパディング位置が異なり、オブジェクトの相対位置がずれる

### 4. 位置エンコーディングの不整合
**異なる解像度での位置埋め込みの非互換性**
- Qwenの位置埋め込み: 448×448を想定
- SAMの位置埋め込み: 1024×1024を想定
- 融合時に位置情報の解釈が異なる

## 実装の確認

### 現在のマスク次元処理コード
```python
# minimal_train.py line 1202-1206
if pred_mask_np.ndim == 3 and pred_mask_np.shape[0] == 1:
    pred_mask_np = pred_mask_np[0]  # [1, H, W] -> [H, W]
elif pred_mask_np.ndim > 2:
    logger.warning(f"[Visualization] Unexpected mask shape: {pred_mask_np.shape}, squeezing")
    pred_mask_np = pred_mask_np.squeeze()
```

**質問1: この実装は正しいですか？**
→ はい、正しいです。SAMが出力する[1, H, W]形式を適切に[H, W]に変換しています。

## 質問事項

### 1. 位置ずれの根本原因は何でしょうか？
特に以下の点について教えてください：
- Qwen特徴のリサイズ（14×14→64×64）が位置情報を歪めているか
- spatial_merge_size=2の影響で上方向にバイアスが生じるか
- 異なる解像度での前処理による座標系のミスマッチか

### 2. 解決方法について
以下のアプローチのどれが有効でしょうか：
- **A. 位置補正の追加**: リサイズ後に位置オフセットを補正する変換を追加
- **B. 特徴アライメント改善**: Qwen特徴を直接64×64で抽出（spatial_mergeを調整）
- **C. 統一前処理**: QwenとSAMで同じ前処理パイプラインを使用
- **D. 位置埋め込みの再学習**: 融合後の特徴に対して新しい位置埋め込みを学習

### 3. デバッグ方法について
位置ずれを定量的に測定し、修正を検証するために：
- 特定のピクセル座標でのアクティベーションを追跡すべきか
- グリッド状のテスト画像で位置マッピングを可視化すべきか
- 中間特徴マップの空間対応を確認すべきか

## 期待する回答
1. 位置ずれの最も可能性の高い原因
2. 推奨される解決アプローチ（実装の具体例付き）
3. 問題を検証・修正するためのデバッグ戦略

特に、**Qwen2.5-VLのspatial_merge**と**SAM2.1の高解像度処理**を統合する際のベストプラクティスについて、具体的な実装指針をいただければ幸いです。

## 追加情報
- 4分割マスク問題は解決済み（SAMのmulti-mask出力を適切に処理）
- 学習時のDiceスコア: 0.3-0.97（良好）
- 可視化時のDiceスコア: 0.01-0.03（非常に低い）
- 位置ずれは一貫して上方向

これらについて、Qwen2.5-VL、SAM2.1、LISA等の公式実装を参考にした解決策を教えてください。