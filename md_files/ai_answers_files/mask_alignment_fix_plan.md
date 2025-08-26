# マスク位置ずれ修正方針書

## 問題の概要
Visualizationで保存される画像において、生成されたマスクとGTマスクが元画像と位置が合わない問題が発生している。

## 原因分析

### 1. データセット別の現状分析

#### SemSegDataset (src/data/sem_seg_dataset.py)
- **問題あり**: 画像とマスクで異なる前処理を適用
  - 画像: アスペクト比維持リサイズ → パディング（448x448 for Qwen, 1024x1024 for SAM）
  - マスク: 元のラベルサイズのまま使用、HybridDatasetで単純リサイズ

#### ReasonSegDataset (src/data/reason_seg_dataset.py)
- **問題あり**: 画像とマスクで異なる前処理
  - SAM画像: アスペクト比維持リサイズ → パディング（1024x1024）
  - マスク: 元画像サイズのまま返却
  - Qwen画像: 単純リサイズ（448x448、アスペクト比無視）

#### VQADataset (src/data/vqa_dataset.py)
- **軽微な問題**: 
  - SAM画像: 単純リサイズ（1024x1024、アスペクト比無視）
  - Qwen画像: 単純リサイズ（448x448、アスペクト比無視）
  - マスクは使用しないため直接的な影響なし

### 2. 共通処理での問題

#### HybridDataset (src/data/dataset.py)
- `preprocess_mask`関数: 単純に1024x1024にリサイズ（アスペクト比無視）
- 画像とマスクの座標系が不一致

## 修正方針

### 優先度1: データセットでのマスク前処理統一

#### 1. SemSegDataset修正
```python
# マスクも画像と同じ変換を適用
# 1. アスペクト比維持リサイズ
# 2. 同じパディング位置
```

#### 2. ReasonSegDataset修正
```python
# マスクをSAM画像と同じ変換で処理
# get_mask_from_json後にリサイズとパディング
```

### 優先度2: 共通関数の改善

#### 1. preprocess_mask_with_aspect_ratio関数の追加
- アスペクト比を維持したリサイズ
- 画像と同じパディング戦略

#### 2. HybridDatasetでの統一処理
- 新しいpreprocess_mask関数を使用
- orig_hwを活用した正確な変換

### 優先度3: 学習・可視化の改善

#### 1. compute_lossの簡略化
- 前処理で統一されるため、リサイズ処理を削除

#### 2. save_visualizationの改善
- パディング領域を考慮した可視化

## 実装順序

1. **修正方針の保存とGit操作**
   - このドキュメントを保存
   - git add . && git push

2. **データセットの修正**
   - SemSegDataset: マスクに画像と同じ変換を適用
   - ReasonSegDataset: マスクをSAM画像に合わせる
   - VQADataset: SAM画像をアスペクト比維持に変更（オプション）

3. **共通関数の追加**
   - preprocess_mask_with_aspect_ratio関数を実装
   - 既存のpreprocess_maskは互換性のため残す

4. **HybridDatasetの修正**
   - 新しいマスク前処理関数を使用
   - orig_hwを確実に伝達

5. **学習・可視化コードの最適化**
   - 不要なリサイズ処理を削除
   - デバッグログのクリーンアップ

## 期待される効果

1. **座標系の一致**: 画像とマスクが完全に同じ座標系で処理される
2. **学習の改善**: 正確な位置対応によりloss計算が改善
3. **可視化の修正**: GTマスクと予測マスクが正しく重なる
4. **コードの簡潔化**: 不要なリサイズ処理が削除される

## テスト方法

1. 修正後、minimal_train.pyで短時間学習を実行
2. visualizationsフォルダの画像を確認
3. GTマスクと予測マスクの位置が一致することを検証
4. IoU/Dice係数の改善を確認