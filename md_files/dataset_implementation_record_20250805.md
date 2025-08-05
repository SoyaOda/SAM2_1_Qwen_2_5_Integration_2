# データセット・前処理実装記録 (2025年8月5日)

## 概要

LISA改プロジェクトにおけるデータセットと前処理の実装記録。Qwen2.5-VL-3BとSAM2.1のデュアルストリーム処理に対応したデータパイプラインの構築と、Gemma形式からmessages形式への移行について記録する。

## 実装の経緯と重要な変更

### 1. Gemma-3形式からmessages形式への完全移行

**背景**: 
- 当初Gemma-3形式（`<start_of_turn>/<end_of_turn>`）を使用
- Qwen2.5-VLの`apply_chat_template`との不整合が発生

**解決策**:
```python
# 旧形式（Gemma-3）
conv.append_message("user", "Segment the mountain.")
conv.append_message("model", "Sure, it is [SEG].")
conversation = conv.get_prompt()  # "<start_of_turn>user\n..."

# 新形式（messages）
messages = [
    {
        "role": "user",
        "content": "Segment the mountain."
    },
    {
        "role": "assistant", 
        "content": "Sure, it is <SEG>."
    }
]
conversations.append(messages)
```

**利点**:
- モデル非依存の中間表現
- `apply_chat_template`による自動変換
- 将来的な他モデルへの移植性向上

### 2. SEGトークンの統一（[SEG] → <SEG>）

**問題**: 
- constants.pyでは`<SEG>`定義
- ANSWER_LISTでは`[SEG]`使用
- 不統一によるトークン認識エラーの可能性

**解決**:
```python
# constants.py
DEFAULT_SEG_TOKEN = "<SEG>"

ANSWER_LIST = [
    "It is <SEG>.",
    "Sure, <SEG>.",
    "Sure, it is <SEG>.",
    "Sure, the segmentation result is <SEG>.",
    "<SEG>.",
]
```

### 3. デュアル解像度処理の実装

**要件**:
- Qwen用: 448×448（14の倍数パディング）
- SAM用: 1024×1024（最長辺基準リサイズ）

**実装**:
```python
# Qwen用画像前処理
h, w = image.shape[:2]
scale = 448 / max(h, w)
new_h = int(h * scale)
new_w = int(w * scale)

# 縮小時はINTER_AREA、拡大時はINTER_CUBIC
interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
image_for_qwen = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

# 14の倍数にパディング
pad_h = (-new_h) % 14
pad_w = (-new_w) % 14
```

### 4. apply_chat_templateの統一使用

**O3からの推奨事項**:
1. 画像とテキストを同時に処理
2. 学習・推論で共通使用
3. 前処理の不一致を防止

**実装**:
```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_pil},
            {"type": "text", "text": text_prompt}
        ]
    }
]

qwen_processed = self.qwen_processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
)
```

### 5. 可視化の改善（縞模様問題の解決）

**問題**: 
- apply_chat_template後の正規化済み画像が縞模様に表示
- 人間には理解不可能な表示

**原因**:
- 正規化により画素値が-2.5〜+2.5の範囲
- matplotlib表示時に0-1範囲外がクリップ

**解決**:
```python
# 元画像を別途保持
return {
    'pixel_values': image_qwen,      # 前処理済み（モデル用）
    'original_image': image_pil,     # 元画像（可視化用）
    ...
}
```

### 6. マルチスケール損失の削除

**背景**:
- 当初1024×1024の高解像度損失を実装
- SAM画像を現状使用していないため計算コストのみ増加

**変更**:
```python
# 削除: compute_multiscale_segmentation_loss関数
# 削除: LISALossクラスのmulti_scale_loss_weights引数
# シンプルな単一スケール損失に統一
```

## データセット構成

### サポートされるデータセット

1. **セマンティックセグメンテーション**
   - ADE20K
   - COCOStuff  
   - Mapillary
   - PACO LVIS
   - Pascal Part

2. **参照セグメンテーション**
   - RefCOCO/RefCOCO+/RefCOCOg

3. **推論セグメンテーション**
   - ReasonSeg

4. **VQA（Visual Question Answering）**
   - LLaVA Instruct 150K

### データフロー

```
原画像
  ├─→ Qwen前処理（448×448）
  │     └─→ apply_chat_template → pixel_values
  └─→ SAM前処理（1024×1024）
        └─→ 正規化 → sam_images
```

## collate_fn実装の詳細

**バッチ処理**:
- テキストシーケンスの動的パディング
- マスクの有無による条件分岐
- image_grid_thwの適切な処理

**メモリ効率**:
- original_imagesはCPUメモリに保持
- GPUには前処理済みテンソルのみ転送

## 今後の課題と改善点

1. **SAM画像の活用**
   - 現在未使用のSAM用画像ストリームの活用
   - 高解像度セグメンテーションの実装

2. **学習時の最適化**
   - 元画像保持の条件分岐（学習/評価モード）
   - メモリ使用量の更なる削減

3. **データ拡張**
   - RandomResizedCropなどの拡張手法
   - MixUpやCutMixの実装検討

## 実装ファイル一覧

- `src/data/dataset.py`: 統合データセットとcollate_fn
- `src/data/sem_seg_dataset.py`: セマンティックセグメンテーション
- `src/data/refer_seg_dataset.py`: 参照セグメンテーション
- `src/data/reason_seg_dataset.py`: 推論セグメンテーション
- `src/data/vqa_dataset.py`: VQAデータセット
- `src/data/conversation.py`: 会話テンプレート管理
- `src/data/constants.py`: 定数定義
- `tests/visualize_preprocessing.py`: 前処理可視化ツール

## 参考資料

- O3クエリ結果: `md_files/current/o3_img_processing.md`
- 高解像度機能仕様: `md_files/past/high_res_features_spec.md`
- 実装記録: `md_files/implementation_record_20250804.md`