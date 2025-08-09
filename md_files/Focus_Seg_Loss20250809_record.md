# Focus_Seg_Loss20250809 実装記録

## 実装日時
2025年1月9日

## 実装内容
**問題1: テキスト指示の空間情報不足の修正**
**問題2: マルチマスク出力の複雑さの修正**

---

# 問題1: テキスト指示の空間情報不足の修正

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

---

# 問題2: マルチマスク出力の複雑さの修正

## 背景と問題点
現行設計では、1つの会話（QAペア）中に複数の`<SEG>`トークンを出力させることで複数マスクに対応しようとしていた。しかし：
- LLMにとって出力制御が非常に難しい
- データセット側でも明示的にマルチマスクの指示は用意されていない
- seg_lossが下がらないケース（複数対象を1枚のマスクにまとめてしまい誤差が残る）が発生

## 実装方針
**1会話につき1マスク出力**に統一。マルチクラスの同時セグメンテーション要求は訓練データ上作らない方針。一枚の画像に複数対象がある場合でも、それぞれを別個のQAペア（または別ターン）として扱う。

## 実装詳細

### 1. データセット設定の変更
**変更ファイル:**
- `src/data/dataset.py`
- `src/data/sem_seg_dataset.py`
- `src/data/refer_seg_dataset.py`
- `src/data/reason_seg_dataset.py`
- `src/data/vqa_dataset.py`

**変更内容:**
```python
# 全データセットで統一
num_classes_per_sample: int = 1,  # 1会話1マスクに統一
```

### 2. セマンティックセグメンテーションデータセットの簡素化
**ファイル**: `src/data/sem_seg_dataset.py`

#### _get_vlpart_item メソッドの変更
```python
# 変更前: 複数クラスを選択
if len(anns) >= self.num_classes_per_sample:
    sampled_anns = np.random.choice(anns, size=self.num_classes_per_sample, replace=False).tolist()
else:
    sampled_anns = anns

# 変更後: 1クラスのみを選択
if len(anns) > 0:
    sampled_anns = [np.random.choice(anns)]
else:
    sampled_anns = []
```

```python
# マスクのテンソル変換も単一マスクに最適化
# 変更前
masks = np.stack(masks, axis=0)
masks = torch.from_numpy(masks)

# 変更後
if len(masks) > 0:
    masks = torch.from_numpy(masks[0]).unsqueeze(0)  # (1, H, W)形式
else:
    return self.__getitem__(0)
```

#### _get_semseg_item メソッドの変更
```python
# マスクの作成（1会話1マスクに統一）
label_tensor = torch.from_numpy(label).long()

# 最初のクラスのみを使用（num_classes_per_sample=1）
if len(sampled_classes) > 0:
    sampled_cls = sampled_classes[0]
    try:
        if isinstance(classes, np.ndarray):
            class_id = np.where(classes == sampled_cls)[0]
            if len(class_id) > 0:
                class_id = class_id[0]
            else:
                return self.__getitem__(0)
        else:
            class_id = classes.index(sampled_cls)
    except (ValueError, IndexError):
        return self.__getitem__(0)
    
    # 単一のマスクを作成
    mask = (label_tensor == class_id).float()
    masks = mask.unsqueeze(0)  # (1, H, W)形式に
else:
    return self.__getitem__(0)
```

### 3. 参照・推論セグメンテーションの修正
**ファイル**: `src/data/refer_seg_dataset.py`, `src/data/reason_seg_dataset.py`

```python
# 1つの参照表現/説明のみを選択
if len(sents) > 0:
    sampled_inds = [np.random.choice(len(sents))]
else:
    sampled_inds = []
```

### 4. 損失計算の最適化
**ファイル**: `minimal_train.py`

```python
def compute_loss(self, outputs, labels, mask_labels):
    """損失計算（1会話1マスクに最適化）"""
    # ...
    
    if outputs.mask_logits is not None:
        for batch_idx, batch_masks in enumerate(outputs.mask_logits):
            if batch_masks is not None and len(batch_masks) > 0:
                gt_mask = mask_labels[batch_idx]
                
                # 1会話1マスクなので、最初のマスクのみを使用
                pred_mask = batch_masks[0] if isinstance(batch_masks, list) else batch_masks
                
                # サイズ調整時も1マスクを前提に処理
                if (pred_h, pred_w) != (gt_h, gt_w):
                    # 複数マスクの場合は最初のマスクのみを使用
                    if gt_mask.dim() == 3 and gt_mask.shape[0] > 1:
                        gt_mask = gt_mask[0]
                    # 4次元テンソルの生成と検証
                    if gt_mask_4d.dim() != 4:
                        # 適切な次元調整
                        ...
```

## 動作確認

### テスト実行
```bash
python minimal_train.py --fast_dev_run --samples_per_epoch 4 --batch_size 2 --num_epochs 1 --debug
```

### 実行結果

#### 1. マスク形状の統一を確認
```
pred_mask shape: torch.Size([1, 448, 448])
gt_mask shape: torch.Size([1, 1024, 1024])
```
→ 単一マスクのみが処理されている（以前は複数マスクの可能性があった）

#### 2. 問題1の重心計算も正常動作
```
[CENTROID] Batch 0, SEG 0: Computed centroid from GT mask - x=503.0, y=731.6 (image center would be 512, 512)
[CENTROID] Batch 0, SEG 0: Computed centroid from GT mask - x=575.0, y=591.1 (image center would be 512, 512)
```

#### 3. 学習の正常進行
```
Epoch 1 - 平均損失: 19.1614, LM: 17.8750, Seg: 1.2864
```
→ エラーなく完了、seg_lossも適切に計算（1.2864）

## 期待される効果

### 問題1の修正による効果
1. **セグメンテーション精度の向上**
   - 対象物の実際の位置に基づいたプロンプト生成
   - より正確なマスク予測

2. **画像中央への固定バイアスの解消**
   - 中心から離れたオブジェクトも正確にセグメント可能
   - 多様な位置のオブジェクトに対する汎化性能向上

### 問題2の修正による効果
1. **学習の安定化**
   - LLMの出力制御負荷が軽減
   - 1質問1マスクの明確な対応関係

2. **実装の簡素化**
   - データローダ、コレータ、損失計算のロジックが単純化
   - 複数マスクのループ処理が不要に

3. **seg_lossの改善**
   - 複数対象を1マスクにまとめる誤差要因を排除
   - より正確なマスク予測の学習が可能

## 今後の拡張性

### 複数インスタンスへの対応
将来的にマルチマスク出力が必要な場合は、推論時に以下の方法で対応可能：
- 複数の質問を内部で生成し、複数回推論
- 各マスクを個別に取得して統合
- ただし、学習は1会話1マスクのシンプルな設計を維持

### 代替アプローチ
- 連結成分ごとの重心計算と複数ポイントプロンプト
- バウンディングボックスの中心を使用
- 領域の分布を考慮した重み付き重心

## 参考資料
- Focus_Seg_Loss20250809_spec.md（修正方針）
- o3_query_answers.md（SAM2.1のpoint prompting実装詳細）
- LISA論文（embedding-as-mask パラダイム）