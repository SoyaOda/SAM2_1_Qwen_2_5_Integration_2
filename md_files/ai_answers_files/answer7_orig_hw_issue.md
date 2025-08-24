# SAM2.1とQwen2.5-VL統合モデル（LISA改）における元画像サイズ情報の欠落問題

## 現在の問題

SAM2.1とQwen2.5-VL統合モデル（LISA改）において、学習時の可視化で予測マスクとGTマスクが元画像と正しく重ならない問題が発生しています。

### 具体的な症状
1. `orig_hw`（元画像サイズ）が常に`(1024, 1024)`となっている
2. 実際の元画像サイズが失われているため、`postprocess_masks`が正しく動作しない
3. 可視化時にマスクと元画像のオーバーレイが一致しない

## 現在の実装詳細

### データフロー
```
SemSegDataset/ReferSegDataset（元データセット）
    ↓
    9要素または10要素のタプルを返す：
    - image_path: 画像パス（文字列）
    - image_sam: 前処理済みSAMテンソル (3, 1024, 1024)
    - image_qwen_tensor: 前処理済みQwenテンソル
    - conversations: 会話データ
    - masks: GTマスク
    - その他のメタデータ
    ↓
HybridDataset.__getitem__()
    ↓
    元画像サイズを取得しようとするが失敗
    ↓
collate_fn
    ↓
MinimalTrainer.save_visualization()
```

### 問題の根本原因

#### 1. データセット側の前処理
元データセット（SemSegDataset等）は、既に前処理済みのテンソルを返しています：
- `image_sam`: SAM2Transformsまたは類似の前処理で1024×1024に変換済み
- `image_qwen_tensor`: Qwen用に前処理済み
- **元画像サイズ情報が保存されていない**

#### 2. HybridDatasetでの復元試行（現在の実装）
```python
# 9要素形式と10要素形式の処理
if len(sample) == 9 or len(sample) == 10:
    # 元画像サイズを取得しようとする
    if isinstance(image_path, str) and os.path.exists(image_path):
        try:
            with Image.open(image_path) as img:
                orig_hw = (img.height, img.width)
        except:
            # フォールバック処理...
            orig_hw = (1024, 1024)
```

問題点：
- `image_path`が相対パスまたは無効なパスの可能性
- 画像ファイルが移動/削除されている可能性
- coord_transformやresizeオブジェクトが元画像サイズを持っていない

#### 3. SAM2のpostprocess_masks
```python
# SAM2公式の後処理
pred_mask_processed = self.sam_transforms.postprocess_masks(
    pred_mask.float(),
    orig_hw  # ここが(1024, 1024)だと正しく復元できない
)
```

SAM2は正方形ストレッチ（アスペクト比無視）で1024×1024にリサイズするため、元画像サイズが分からないと正しく復元できません。

## 質問

### 1. データセット設計について
- SemSegDataset、ReferSegDataset等の元データセットを修正して、元画像サイズ情報を保持させるべきか？
- それとも、HybridDataset側で画像を再読み込みして元サイズを取得すべきか？

### 2. 座標変換オブジェクトについて
- coord_transform（10要素形式）やresize（9要素形式）オブジェクトの正確な仕様は？
- これらのオブジェクトに元画像サイズ情報を持たせる標準的な方法は？

### 3. 最適な解決策
以下の選択肢のうち、どれが最適か：

**Option A: データセット側を修正**
- SemSegDataset等を修正して、タプルに`orig_hw`を追加（11要素形式にする）
- メリット：確実に元画像サイズが取得できる
- デメリット：既存のデータセットクラスすべてを修正する必要がある

**Option B: 画像パスからの再読み込み**
- HybridDatasetで`base_image_dir`を使って完全パスを構築し、画像を再度開く
- メリット：既存データセットの修正不要
- デメリット：I/Oオーバーヘッド、パス解決の複雑さ

**Option C: 前処理を遅延実行**
- データセットは生画像（PIL Image）を返し、HybridDatasetで前処理を実行
- メリット：元画像サイズが確実に取得でき、前処理の一元管理が可能
- デメリット：大規模な設計変更が必要

**Option D: GTマスクのサイズから推定**
- GTマスクの元サイズ（前処理前）から元画像サイズを推定
- メリット：追加情報不要
- デメリット：マスクも前処理されている場合は使えない

### 4. LISAオリジナルの実装
- オリジナルLISAではこの問題をどう解決していたか？
- SAM1のResizeLongestSide + パディング方式では、パディング情報から元サイズを逆算できたが、SAM2の正方形ストレッチではどうすべきか？

## 現在のコードベース情報

### 関連ファイル
- `src/data/dataset.py`: HybridDataset（統合データセット）
- `src/data/sem_seg_dataset.py`: Semantic Segmentationデータセット
- `src/data/refer_seg_dataset.py`: Referring Segmentationデータセット
- `src/data/reason_seg_dataset.py`: Reasoning Segmentationデータセット
- `src/data/vqa_dataset.py`: VQAデータセット
- `src/utils/coordinate_transform.py`: 座標変換ユーティリティ
- `minimal_train.py`: 学習スクリプト（可視化処理含む）

### SAM2Transformsの仕様
- 入力：PIL画像またはnumpy配列
- 処理：正方形リサイズ（1024×1024、アスペクト比無視）+ ImageNet正規化
- postprocess_masks：(B, N, 256, 256) → (B, N, H_orig, W_orig)
- **H_orig, W_origは変換前の元画像サイズが必要**

## 期待する回答

1. **推奨される解決策**（Option A〜Dのどれか、または新しい提案）
2. **具体的な実装方法**（コード例付き）
3. **既存コードへの影響を最小化する方法**
4. **パフォーマンスとメモリ使用量の考慮**
5. **デバッグとテストの方法**

これらの点について、SAM2.1とQwen2.5-VLの公式実装、およびLISAの設計思想に基づいた最適な解決策を教えてください。特に、大規模データセットでの学習時のI/O効率とメモリ効率を考慮した実装方法を提案してください。