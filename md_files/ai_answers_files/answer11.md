了解。添付の4枚（学習中2枚・推論2枚）を前提に、\*\*「学習中の可視化だけ位置がズレる」\*\*原因を特定するための徹底レビューと、再現性のある修正案をまとめました。
（リポジトリは `SoyaOda/SAM2_1_Qwen_2_5_Integration_2` 前提ですが、ここでは画像と一般的な実装パターンから切り分けています。）

---

# まず観察結果（添付画像から読める事実）

* **学習ステップ図①（step\_000005.png）**

  * 表示テキスト：`Orig HW: (480, 640) | SAM: 1024x1024 (stretched)`
  * しかし左上の「Original Image」は **shape: (336, 448, 3)** と書かれており、**Orig HWと一致していない**。
  * 下段の `GT Overlay` で**下側が真っ黒に帯状に切れている**（＝オーバーレイ時に画像とマスクのサイズ/アスペクトが不一致）。
  * `Pred shape: (480, 640) | GT shape: (480, 640)` とあり、**損失計算直前ではサイズは揃っていそう**だが、**表示（可視化）時にズレている**。

* **学習ステップ図②（step\_000010.png）**

  * 「Original Image」は **(448, 308, 3)** と縦長。
  * ヘッダには `Pred shape: (1024,1024) | GT shape: (1024,1024) | Orig HW: (1024,1024)` とあり、**可視化は1024平方前提**。
  * 下段 `Prediction Overlay` が**途中で不自然に切れている**（左半分だけ赤、右が黒っぽい）＝やはり**オーバーレイの座標合わせが壊れている**。

* **推論図（truck\_step5.png / dog\_step5.png）**

  * 画像とマスクが**きれいに一致**している。
    ➜ **推論パイプラインの幾何変換は正しく、ズレは「学習時の可視化」限定**と判断。

---

# 結論（原因の核心）

> **学習時の可視化で使っている “ベース画像” の座標系（解像度/アスペクト）と、GT/予測マスクの座標系が一致していない。**
> 具体的には、
>
> * `Original Image` に **「RAW画像（例: 336×448）」** を使い、
> * `GT` と `Pred` は **「学習用に一旦リサイズ（例: 480×640 や 1024×1024 へ STRETCH）」** した座標系のまま、
> * **逆変換（inverse transform）や正しいリサイズ** をせずに**重ねている**ため、帯やズレが出ている。

さらに細かい“やらかしがち”ポイント（**優先度順**）：

1. **ベース画像の選択ミス**

   * 可視化時に **RAW画像** を使っているのに、**GT/Pred は前処理後サイズ**（480×640 or 1024×1024）で重ねている。
   * あるいはその逆で、**前処理後画像**に**RAWサイズのGT**を重ねている。
     → いずれも**片方をもう一方へ厳密に合わせる処理が欠落**。

2. **SAM2.1の正方形リサイズ（stretch）を “戻していない”**

   * SAM2.1は**パディングせず縦横別スケール (sx≠sy) で平方化**します。
   * \*\*可視化時は `sx = 1024/Wraw`、`sy = 1024/Hraw` を用いた“非等方の逆リサイズ”\*\*が必要。
   * 等方スケール（LongestSide系）や単純な`cv2.resize(..., (Wraw, Hraw))`だけだと**見かけ上合っても位置が流れる**。

3. **`cv2.resize` のサイズ引数の順序ミス**

   * OpenCVは `(width, height)` 順。`(H, W)` を渡すと**天地/左右が崩れる**。
   * その結果が**下端の黒帯**や**左右に寄る**現象として現れやすい。

4. **補間設定ミス**

   * **マスクは `INTER_NEAREST`** で戻すべきところを **`INTER_LINEAR`** で戻すと、**境界がにじみ、当たり判定がずれる**。
   * Dice/IoU は小さく見積もられ、可視化も“にじんでズレた”印象になる。

5. **EXIFの回転未反映（頻度は低いが要確認）**

   * 画像だけ自動回転され、マスクは回っていないケース。
   * PIL読み込み→`ImageOps.exif_transpose` を使わないと稀に発生。

---

# 影響範囲の評価

* **損失計算そのもの**は、表示ログから見ると **`Pred` と `GT` のテンソルサイズが一致**しているので**致命的ではない**可能性が高い（`IoU: 0.144` 等が一応出ている）。
* ただし、**GTを作る前処理とPredを合わせる前処理が“完全一致”していなければ、Lossは理論上もズレ**ます（特にSAM2.1の非等方平方化を跨ぐ場合）。
* よって、**可視化修正だけでなく、Loss前の座標系も必ずアサートで検証**してください。

---

# いますぐ入れるべき**検証ログ（アサート）**

学習ループ内（損失計算直前）で、**1ステップに1回で良い**ので次を出す：

```python
# 例：学習バッチから1サンプルだけ可視化対象を抜粋
raw = batch["raw_image_np"]             # H0,W0,3  : 前処理前
sam = batch["sam_square_np"]            # 1024,1024: SAM2.1入力
gt  = batch["gt_mask_np"]               # Hgt,Wgt  : 損失と同じ座標系
pred= pred_mask_np                      # Hpred,Wpred: 損失と同

print("RAW:", raw.shape, "SAM:", sam.shape, "GT:", gt.shape, "PRED:", pred.shape)
assert gt.shape == pred.shape, "GTとPredはLoss前に同形であるべき"

# 可視化直前に「重ね合わせ対象のベース画像」と「マスク」を必ず同じHWへ合わせる
vis_base = raw                          # 可視化はRAWに統一するのが安全
gt_on_raw   = resize_mask_to_raw(gt,   raw.shape[:2], meta)   # ★後述の関数
pred_on_raw = resize_mask_to_raw(pred, raw.shape[:2], meta)   # ★

assert gt_on_raw.shape == raw.shape[:2]
assert pred_on_raw.shape == raw.shape[:2]
```

`meta` は前処理で記録した **(sx, sy, 逆変換が可能な情報)**。

---

# **決定打：幾何情報を“往復”できるように一元化**（実装テンプレ）

SAM2.1の**正方形ストレッチ**に合わせ、**前処理→学習→可視化/Loss**を同一の幾何で往復させます。

```python
# transforms/geometry.py
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class SquareStretchMeta:
    raw_h: int
    raw_w: int
    sam_side: int  # 1024
    sx: float      # sam_side / raw_w
    sy: float      # sam_side / raw_h

def square_stretch_image(img: np.ndarray, sam_side: int) -> tuple[np.ndarray, SquareStretchMeta]:
    h, w = img.shape[:2]
    sx, sy = sam_side / w, sam_side / h
    img_sq = cv2.resize(img, (sam_side, sam_side), interpolation=cv2.INTER_LINEAR)
    return img_sq, SquareStretchMeta(h, w, sam_side, sx, sy)

def mask_from_raw_to_square(mask_raw: np.ndarray, meta: SquareStretchMeta) -> np.ndarray:
    # Raw(H0,W0) -> SAM(1024,1024)
    return cv2.resize(mask_raw.astype(np.uint8), (meta.sam_side, meta.sam_side), interpolation=cv2.INTER_NEAREST)

def mask_from_square_to_raw(mask_sq: np.ndarray, meta: SquareStretchMeta) -> np.ndarray:
    # SAM(1024,1024) -> Raw(H0,W0)
    return cv2.resize(mask_sq.astype(np.uint8), (meta.raw_w, meta.raw_h), interpolation=cv2.INTER_NEAREST)
```

**データローダ**（`__getitem__`）では：

```python
img_raw = read_image(...)                # (H0,W0,3) EXIF補正済み
img_sq, meta = square_stretch_image(img_raw, sam_side=1024)

# GTがRAW座標系で来るなら
gt_raw = load_gt_mask_raw(...)
gt_sq  = mask_from_raw_to_square(gt_raw, meta)

# モデル入力: img_sq -> tensor化
# 損失: pred_sq vs gt_sq （同一座標系）でOK
sample = {
    "raw_image_np": img_raw,
    "sam_square_np": img_sq,
    "gt_mask_sq": gt_sq,
    "meta": meta,
}
```

**学習ループ**では：

```python
# pred_sq: (1024,1024) か、出力をF.interpolateで1024へ
loss = dice_bce_loss(pred_sq, gt_sq)

# 可視化：RAW座標へ“必ず”戻す
pred_raw = mask_from_square_to_raw((pred_sq > 0.5).astype(np.uint8), meta)
gt_raw   = mask_from_square_to_raw(gt_sq, meta)

vis_pred = overlay_mask(sample["raw_image_np"], pred_raw)
vis_gt   = overlay_mask(sample["raw_image_np"], gt_raw)
```

**OpenCVの引数順**は `(width,height)` に注意。
**マスクは常に `INTER_NEAREST`**。

---

# 既存コードを最小修正で直すチェックリスト

1. **可視化で使う“ベース画像”を “GT/Predと同じ座標系” に統一**

   * RAWで見せたいなら、**GT/PredをRAWへ逆変換してから**重ねる。
   * いまは `Original Image shape` と `GT shape` が一致していない（例: 336×448 vs 480×640）。

2. **SAM2.1のstretchを考慮した“非等方逆リサイズ”を必ず適用**

   * 1024へ行くとき `sx, sy` を記録 → 戻すとき `(Wraw, Hraw)` へ `INTER_NEAREST` で。

3. **`cv2.resize` の引数順と補間を点検**

   * `cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)` になっているか？

4. **Loss前の形一致をアサート**

   * `pred_sq.shape == gt_sq.shape`（または同じRAW）を強制。
   * どちらの座標でLossを取るかを**プロジェクト全体で統一**（推奨：**SAM平方座標**）。

5. **EXIF回転対策**

   * 画像読込直後に `ImageOps.exif_transpose` 相当の処理を入れ、マスクも同じ回転に合わせる。

---

# 追加でやると幸せになるデバッグ可視化

可視化パネルを**4列**にする（学習時）：

1. **RAW画像**
2. **SAM入力画像（1024×1024）**
3. **GT（RAWに逆変換してオーバーレイ）**
4. **Pred（RAWに逆変換してオーバーレイ）**

これで**どこでズレたか**が一目瞭然になります。

---

# まとめ（短縮）

* **原因**：学習時の可視化で、**画像とマスクの座標系（解像度/アスペクト）が一致していない**。SAM2.1の**正方形ストレッチ**を**逆変換せず**に重ねていることが主因。
* **損失**は一見サイズ整合しているが、**幾何が完全一致しているか**のアサートを入れて担保すべき。
* **修正**：前処理で `sx, sy` を記録 → **GT/Predを可視化・Loss時に同じ座標系へ統一**。`cv2.resize` の引数順・補間設定 (`INTER_NEAREST`) を厳守。
* **推論はOK**なので、\*\*学習可視化（visualize系関数）\*\*のみを中心に修正すれば解決します。

必要なら `minimal_train.py` / `src/` の可視化関数（`visualize_*`）を特定して、上記テンプレを直接当て込む形で**パッチ化**します。
