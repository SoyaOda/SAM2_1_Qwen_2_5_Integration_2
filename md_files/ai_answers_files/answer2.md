以下は、**「LISA改（Qwen2.5‑VL + SAM2.1）で“マスクが常に上方向にずれる”」**現象の**原因診断 → 優先度順の対処 → 再現・検証のデバッグ手順**です。要点だけ先にまとめ、その後に実装例を示します。

---

## 結論（TL;DR）

* **最も可能性が高い原因は、Qwen系前処理のレターボックス（リサイズ＋パディング）と、SAM2系前処理のレターボックスが一致していないこと**による**座標系ギャップ**です。
  SAMの公式パイプラインは **「Longest side を 1024 にリサイズ → 右端・下端にのみパディング（F.pad）」** という**左上原点固定（top‑left anchor）**の手順を取り、さらに**postprocess でこのパディングを除去して元解像度へ復元**します。学習時の可視化で**この復元（postprocess）を省く／別のレターボックス手順を混在**させると、**一方向（典型的に上側）へ系統的なオフセット**が出ます。([Minibatch AI][1])
* **Qwen2.5‑VLの視覚トークンは PatchMerge（2×2）で 1/4 に減る**（例：28×28→14×14）。14×14 を 64×64 に**非整数倍率（×4.571…）でバイリニア補間**してから SAM の 64×64 と足し込むだけだと、\*\*“どの画素中心に合わせるか”のズレ（align\_corners の約束事）\*\*で目立つ方向オフセットが出やすいです。\*\*補間の規約をSAM側と合わせる（align\_corners=False）\*\*か、**厳密なアフィン変換で座標を一致**させてから融合してください。([PyTorch Forums][2])
* **推論で正常に見える理由**は、多くのユーティリティが**低解像マスクを postprocess で元サイズに戻し、パディングを除去してから表示**しているため。学習時の可視化も**必ず同じ postprocess**を通すべきです。([Dss Solutions][3], [Minibatch AI][1])

---

## 1) もっとも可能性の高い根本原因

### (A) レターボックス（Resize＋Pad）の不一致

* **SAM2 の標準**

  1. **ResizeLongestSide(1024)**（アスペクト比維持）
  2. **右端・下端のみパディング**（`F.pad(x, (0, pad_w, 0, pad_h))`）
  3. **mask の postprocess**で「**まず 1024 にアップサンプル → パディング分をクロップ → 元画像解像度へ再リサイズ**」
     という**一貫した座標変換**になっています（`SamPredictor` 相当）。([Minibatch AI][1])
* **Qwen2.5‑VL 側**
  ダイナミック解像度で **`image_grid_thw=[T,H,W]`** を持ち、ViT は **`spatial_merge_size=2`** で 2×2 マージ（28×28→14×14）。Qwen の Processor は **独自にパディングやサイズ調整**を行うため、**SAM とは pad 方向/量 が一致しない**ことがあります（例：両側パディング or センタリング）。([Hugging Face][4], [VLLM Documentation][5])
  → **SAM（左上固定）**と**Qwen（センター寄せ等）**の**起点の違い**が、**上方向への恒常的ずれ**として現れやすいです。

### (B) 非整数倍率アップサンプリングと align\_corners 規約

* 14→64 は**非整数倍率**。`F.interpolate` の\*\*`align_corners=True/False`\*\*の扱い次第で、**格子中心の取り方**が変わり、**一方向へ半ピクセル〜数ピクセル相当の系統誤差**が生まれます。**SAM の postprocess は bilinear + `align_corners=False` が前提**なので、**Qwen→64 の補間も同じ規約に統一**してください。([Minibatch AI][1], [PyTorch Forums][2])

### (C) “可視化だけ” postprocess を省いている

* SAM 系の実装では**低解像（例：256×256）で出た logits を postprocess で 1024 → 元解像度へ戻す**のが標準。**学習時の可視化がここを省く**と、**レターボックス座標**のまま上書き表示され **上方向／左方向などにずれて見える**のが定番パターンです。([Dss Solutions][3])

> 注：**Qwen 側の PatchMerge**（`spatial_merge_size=2`）自体に**方向バイアス**は設計上ありません（等方的に 2×2 マージ）。ズレの“方向性”が一貫して上方向に出るのは、**前処理のパディング起点（top‑left 固定 vs センター）や補間規約の不一致**が原因のことがほとんどです。([Hugging Face][6])

---

## 2) すぐ効く対策（推奨順）

### ✅ 対策1：**前処理を SAM 準拠に統一**（最も堅い）

* **両経路とも**、まず **SAM の `ResizeLongestSide(1024)`→右/下パディング**で\*\*同じ“レターボックス画像（1024×1024）”\*\*を作る。
* Qwen にはこの 1024×1024 を**さらに 448×448 に（top‑left を保ったまま）縮小**して与える＝**パディング規約を同じにする**。
* 学習時の可視化も\*\*SAM の postprocess（パディング除去＋元解像度復元）\*\*で表示する。([Minibatch AI][1], [Dss Solutions][3])

**理由**：SAM の仕様は**右/下のみ pad**が既定（`F.pad(..., (0,pad_w,0,pad_h))`）。ここに Qwen の画像も合わせれば**座標原点が一致**します。([Minibatch AI][1])

### ✅ 対策2：**Qwen 特徴を SAM 座標へワープ**（座標合わせの厳密法）

* オリジナル→Qwen と、オリジナル→SAM の**二つのレターボックス写像**から、
  **Qwen 座標 → SAM 座標**の**アフィン**（実質は**並進＋等方スケール**）を導出：

  $$
  \mathbf{x}_{\text{SAM}} = k \, (\mathbf{x}_{\text{Qwen}} - \mathbf{t}_\text{Q}) + \mathbf{t}_\text{S},\quad 
  k=\frac{s_\text{SAM}}{s_\text{Q}}
  $$

  （$s$：各系のスケール、$\mathbf{t}$：各系のパディングオフセット）
* このアフィンを `affine_grid` + `grid_sample`（`align_corners=False`）で適用し、**Qwen の 2D 特徴（14×14 → 64×64）を SAM の 64×64 に厳密位置合わせ**してから**Sigma‑Add**する。
* **補間は bilinear + `align_corners=False`**（SAM の postprocess と整合）。([Minibatch AI][1], [PyTorch Forums][2])

### ✅ 対策3：**補間規約と postprocess を統一**

* **学習中の可視化パス**でも、**SAM と同じ postprocess**（1024 へアップ→pad 除去→元解像度）を**毎回適用**。([Dss Solutions][3])
* **Qwen→64 補間**は常に `mode="bilinear", align_corners=False`。
  （`nearest` は座標ずれの既知問題報告があり非推奨、どうしても使う場合は注意）([GitHub][7], [PyTorch Forums][2])

---

## 3) 実装例（抜粋）

### (A) **SAM 準拠レターボックスで統一**（簡潔・堅牢）

```python
import torch
import torch.nn.functional as F

def resize_longest_side(image, target):
    # image: (H0, W0, 3) numpy or torch
    H0, W0 = image.shape[:2]
    s = float(target) / max(H0, W0)
    H1, W1 = int(round(H0 * s)), int(round(W0 * s))
    # resize with bilinear; align_corners=False
    img = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).float()  # 1,C,H,W
    img = F.interpolate(img, size=(H1, W1), mode="bilinear", align_corners=False, antialias=True)
    # pad right/bottom only
    pad_h, pad_w = target - H1, target - W1
    img = F.pad(img, (0, pad_w, 0, pad_h))  # (left,right,top,bottom)
    return img.squeeze(0), s, (0, 0)  # offset(top,left) = (0,0)
```

* **SAM 入力**：上の 1024×1024 をそのまま使用。
* **Qwen 入力**：得られた 1024×1024 を\*\*448×448へ縮小（top‑left 起点維持）\*\*して渡す。
* こうすれば**両経路の原点とスケールが一致**し、**上方向オフセット**は解消します。
  （SAM 標準の手順：Longest side 1024 → pad right/bottom、postprocess で pad 除去・元解像度復元。原理は SAM の docstring/実装にも明記）([Minibatch AI][1])

### (B) **Qwen 特徴を SAM 座標へアフィンでワープ**（座標厳密合わせ）

```python
def warp_qwen_to_sam(qwen_feat_2d,  # (B,C,Hq,Wq), 例: (B,256,14,14)
                     H0, W0,        # original image size
                     qwen_target=448, sam_target=1024):
    # compute SAM letterbox
    sS = sam_target / max(H0, W0)
    HS, WS = int(round(H0 * sS)), int(round(W0 * sS))
    tS = (0.0, 0.0)  # pad top,left are zero in SAM (right/bottom only)

    # compute Qwen letterbox (ここは現在使っているQwen前処理規約を反映すること)
    sQ = qwen_target / max(H0, W0)
    HQ, WQ = int(round(H0 * sQ)), int(round(W0 * sQ))
    # 例: センター寄せなら top,left = (target - size)/2、SAM式なら (0,0)
    topQ  = (qwen_target - HQ) / 2.0
    leftQ = (qwen_target - WQ) / 2.0
    tQ = (topQ, leftQ)

    # Qwen座標 -> SAM座標 のアフィン（y,x）
    k = sS / sQ
    # 連続座標での 2x3 行列（行は y, x の順）
    A = torch.tensor([[k, 0,  (tS[0] - k * tQ[0]) / (qwen_target/2) * 2],
                      [0,  k, (tS[1] - k * tQ[1]) / (qwen_target/2) * 2]], 
                     dtype=qwen_feat_2d.dtype, device=qwen_feat_2d.device)  # 正規化[-1,1]系

    B = qwen_feat_2d.size(0)
    theta = A.unsqueeze(0).repeat(B,1,1)
    grid = torch.nn.functional.affine_grid(theta, 
                size=(B, qwen_feat_2d.size(1), 64, 64), 
                align_corners=False)
    qwen_on_sam = torch.nn.functional.grid_sample(qwen_feat_2d, grid, 
                         mode="bilinear", align_corners=False)
    return qwen_on_sam  # (B,C,64,64)
```

* **Qwen 側のパディング規約**（top/left が 0 なのか、センター寄せなのか）をここに正確に入れるのがコツ。
* こうして得た `qwen_on_sam` を **`fused = sam_embed + beta * qwen_on_sam`** で加算。
* **補間は bilinear + `align_corners=False`** に統一。([PyTorch Forums][2])

### (C) **学習時可視化も postprocess を使う**（表示の一貫化）

```python
# SAMの low-res mask → まず 1024へ（bilinear, align_corners=False）→ pad除去 → 元解像度へ
def sam_postprocess_masks(low_res_masks, input_hw_after_resize, orig_hw, encoder_img_size=1024):
    m = F.interpolate(low_res_masks, (encoder_img_size, encoder_img_size),
                      mode="bilinear", align_corners=False)
    H_in, W_in = input_hw_after_resize   # ResizeLongestSide 後のサイズ
    m = m[..., :H_in, :W_in]             # 右/下パディング除去（SAM流儀）
    m = F.interpolate(m, orig_hw, mode="bilinear", align_corners=False)
    return m
```

* **これは SAM 実装の postprocess と同等の流れ**です（1024化→pad除去→元解像度へ）。**学習ログ可視化も必ずこれを通す**。([Minibatch AI][1])

---

## 4) デバッグ戦略（位置ずれを“定量化”して潰す）

1. **レターボックスの内部変数をログ**

   * SAM 側：`H1, W1, pad_h, pad_w`（ResizeLongestSide 直後のサイズと pad 量）
   * Qwen 側：`Hq, Wq` と **top/left/right/bottom の pad 量**
     → **左右と上下の pad が一致しているか**をまず確認。SAM は**右/下のみ pad**が前提です。([Minibatch AI][1])

2. **チェッカーボード/ドット画像での対応テスト**

   * 原画像の**既知座標の点**（例：上端から y=80px）をマーキング → 両経路それぞれの**64×64**上で**どのセルに来るか**を可視化。
   * これで\*\*“上方向〇px相当の系統オフセット”\*\*が数値化できます（期待は 0）。

3. **相互相関で最適シフトを推定**

   * `xcorr = correlate2d(mask_pred, mask_gt)` のピーク位置から\*\*(dy, dx)\*\* を測定し、**補正量が一定**なら**レターボックス／補間規約**が原因と判断。

4. **align\_corners の A/B テスト**

   * Qwen→64 補間で `align_corners=True/False` を切替 → **ずれ方向が反転/減衰**するなら補間規約の不一致が主因。**最終的には SAM と同じ False に固定**。([PyTorch Forums][2])

---

## 5) よくある質問への回答

### Q1. 「14×14→64×64のリサイズ」が直接の犯人？

**単独ではありません**が、**非整数倍率補間＋規約不一致**はズレを増幅します。**align\_corners=False**に揃え、\*\*座標合わせ（対策1 or 2）\*\*を併用してください。([PyTorch Forums][2])

### Q2. PatchMerge（spatial\_merge\_size=2）が上方向バイアスを持つ？

**設計上そのようなバイアスは想定されていません**。Qwen の `vision_config` には **`spatial_merge_size: 2`** が明示（3B の `out_hidden_size=2048` も記載）。方向性のあるズレは**前処理差／補間規約差**が主因です。([Hugging Face][6])

### Q3. 推論で正常に見えるのはなぜ？

**SAM の postprocess（pad 除去＋元解像度復元）**が**必ず通る**ため。**学習時可視化**でも**同じ postprocess**を適用してください。([Dss Solutions][3])

---

## 6) 参考（一次情報）

* **SAM/SAM2 の前処理と postprocess の流れ**

  * Longest side を 1024 にリサイズ → **右/下パディング** → **low‑res マスクを 1024 にアップ → pad 除去 → 元解像度へ**。実装イメージ・コード断片。 ([Minibatch AI][1])
  * SAM/SAM2 の記事やチュートリアルでも **postprocess で元解像度へ戻す**前提が繰り返し説明。([Dss Solutions][3], [Ultralytics Docs][8])
* **Qwen2.5‑VL**

  * 公式ドキュメント（ダイナミック解像度・MRoPE・動画対応）と `vision_config`（**`out_hidden_size=2048`**, **`spatial_merge_size=2`**）。([Hugging Face][4])
  * `image_grid_thw`（各画像の `[T,H,W]` グリッド情報）を用いた動的解像度の扱い（実装/仕様周辺）。([VLLM Documentation][5])
* **補間規約（PyTorch）**

  * `align_corners=True/False` の意味と可視的な差。**SAM は False** を前提に利用。([PyTorch Forums][2])
  * `nearest` 補間のズレ報告（既知 issue）。([GitHub][7])

---

### 最後に（実運用の指針）

1. \*\*前処理の統一（対策1）\*\*をまず適用し、**学習時可視化でも必ず postprocess**。
2. まだ僅かにズレる場合は \*\*対策2（ワープで厳密座標合わせ）\*\*を足す。
3. **補間は bilinear + `align_corners=False`** で統一。
4. Qwen の 14×14 は**必ず PatchMerge 後の格子**として扱い、**14→64 のアップサンプリングは一旦の近似**と捉える。根本は**座標系を同じにすること**です（原点・スケール・pad 方向）。([Hugging Face][6], [Minibatch AI][1])

この順に直すと、**「常に上にずれる」系統誤差は解消**し、**学習時の可視化 Dice**も**推論時と整合**してきます。

[1]: https://minibatchai.com/2023/04/26/SAM-Fluids.html "Turbulent flow segmentation with SAM | Minibatch AI"
[2]: https://discuss.pytorch.org/t/what-we-should-use-align-corners-false/22663?utm_source=chatgpt.com "What we should use align_corners = False - vision"
[3]: https://dsssolutions.com/2024/08/22/train-fine-tune-segment-anything-2-sam-2-in-60-lines-of-code/?utm_source=chatgpt.com "Train/Fine-Tune Segment Anything 2 (SAM 2) in 60 Lines ..."
[4]: https://huggingface.co/docs/transformers/en/model_doc/qwen2_5_vl?utm_source=chatgpt.com "Qwen2.5-VL"
[5]: https://docs.vllm.ai/en/v0.10.1/api/vllm/model_executor/models/qwen2_5_vl.html?utm_source=chatgpt.com "vllm.model_executor.models.qwen2_5_vl"
[6]: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/acbce9a6c80e81356680a632166380c8cc1d27ec/config.json?utm_source=chatgpt.com "config.json · Qwen/Qwen2.5-VL-3B-Instruct at ..."
[7]: https://github.com/pytorch/pytorch/issues/34808?utm_source=chatgpt.com "Using torch.nn.functional.interpolate with 'nearest' mode ..."
[8]: https://docs.ultralytics.com/reference/models/sam/predict/?utm_source=chatgpt.com "Reference for ultralytics/models/sam/predict.py"
