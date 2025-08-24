以下は、**SAM2.1 の `SAM2Transforms.postprocess_masks()` の正確な入出力仕様**と、**リサイズ検証・ぼやけ対策**までを、公式実装と一次情報のコードから要点整理したものです。引用元は Meta 公式の `facebookresearch/sam2`（`sam2/utils/transforms.py`）およびそれを直接利用している公式／準公式チュートリアル・ドキュメントです。([Hugging Face][1], [PyTorch Docs][2], [OpenVINO Documentation][3])

---

## 1) 入力形状（`SAM2Transforms.postprocess_masks(masks, orig_hw)`）

* **期待されるテンソル形状**：**4次元 `B × C × H × W`**

  * **`B`**：バッチ（画像）数
  * **`C`**：出力マスクのチャンネル数（= 候補マスク数。`multimask_output=True` なら通常 4、`False` なら 1）
  * **`H × W`**：**低解像度マスクの空間サイズ**（H=W=**256** が標準：SAM2 のマスクデコーダ出力）
  * 公式実装は内部で `mask_flat = masks.flatten(0, 1).unsqueeze(1)` を用いて **\[B×C, 1, H, W]** に変換し、連結成分の穴埋めなどを行っています。つまり **B 次元と C 次元の両方が前提**で、**3次元 `[C, H, W]` は非想定**です。([Hugging Face][1])
* **`orig_hw`**：**出力のターゲット解像度**を表す **(H\_orig, W\_orig)** のタプル（**順序は高さ→幅**）。ここを (W,H) と取り違えると縦横が入れ替わり、リサイズが破綻します。([PyTorch Docs][2])
* **低解像度（256×256）での前提**：OpenVINO の公式ノートブックや各種実装が、**`low_res_masks` の空間サイズは 256×256** と明記しています（SAM2 の標準デコーダ出力）。([OpenVINO Documentation][3])

> 参考：`SAM2ImagePredictor` のドキュメント／実装も、**`low_res_masks` を `postprocess_masks()` に通してから**二値化する流れを示し、**入出力形状を `B×C×H×W`** と記述しています。([Hugging Face][4])

---

## 2) 出力形状

* **戻り値のテンソル形状**：**`B × C × H_orig × W_orig`**

  * **B と C は維持**され、空間次元のみ `orig_hw` に合わせて拡大されます。
  * 拡大には **`F.interpolate(..., mode="bilinear", align_corners=False)`** が使われます（公式実装）。**二値化はこの後段**で行うのが正道です。([Hugging Face][1])
* 公式・準公式のサンプルでも、**出力（ロジット）を `> threshold` で二値化**するのは postprocess の後。例：PyTorch TensorRT チュートリアルでは `masks = predictor._transforms.postprocess_masks(...); masks = (masks > 0.0)` の順番です。([PyTorch Docs][2])

---

## 3) 「256→元解像度」リサイズが正しいか確認するチェックリスト

**A. 形状・次元を前後で厳密に検証**

```python
# low_res_masks: 期待は [B, C, 256, 256]
assert low_res_masks.ndim == 4
assert low_res_masks.shape[-1] == 256 and low_res_masks.shape[-2] == 256

# postprocess
masks_up = sam2_trans.postprocess_masks(low_res_masks, orig_hw)  # -> [B, C, H_orig, W_orig]
assert masks_up.shape[0] == low_res_masks.shape[0]   # B 一致
assert masks_up.shape[1] == low_res_masks.shape[1]   # C 一致
assert tuple(masks_up.shape[-2:]) == tuple(orig_hw)  # 空間次元が orig_hw に一致
```

* **`orig_hw` は (H, W)**。順序を誤ると縦横が逆になります（よくあるバグ）。([PyTorch Docs][2])

**B. リサイズの“規約”が一致しているか**

* \*\*`align_corners=False`（公式）\*\*になっていること。
* **ロジットのままアップサンプリング**し、**最後に閾値で二値化**していること（先に二値化してから拡大するとギザギザ・崩れが出やすい）。([Hugging Face][1], [PyTorch Docs][2])

**C. 画素値レンジ・dtype**

* `low_res_masks` は **float ロジット**（通常 `float32`/`bfloat16`）。**`uint8` や `bool` で補間**していないかを確認。
* 公式実装は `masks = masks.float()` と明示してから処理します。([Hugging Face][1])

**D. 可視化の一致**

* **postprocess 後のマスク**を使って可視化（学習時のデバッグでも同じ経路にする）。**任意の独自リサイズで代用しない**。公式チュートリアル類も postprocess の出力をそのまま可視化へ渡します。([PyTorch Docs][2])

---

## 4) 「マスクが小さい／ぼやける」主因と対策

| 症状            | ありがちな原因                                                | 対策                                                                                                    |
| ------------- | ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| **全体に一回り小さい** | `orig_hw` を **(W,H)** で渡している（縦横逆）／`align_corners=True` | \*\*`orig_hw=(H,W)`**で渡す。**`align_corners=False`\*\*に統一（公式どおり）。([PyTorch Docs][2], [Hugging Face][1]) |
| **エッジがギザギザ**  | **二値化してから**アップサンプリング                                   | **ロジットのまま補間→最後に閾値**の順にする（公式の順序）。([PyTorch Docs][2])                                                   |
| **濃度が薄い／滲む**  | `uint8`/`bool` のまま補間、または画像正規化をマスクに誤適用                  | **`masks.float()`** で補間。マスクに **画像用 Normalize をかけない**。([Hugging Face][1])                              |
| **一部だけ欠ける**   | 連結成分の穴埋め未適用 or CUDA 拡張無しで想定外                           | `max_hole_area`/`max_sprinkle_area` を設定。CUDA 拡張が無い場合は自動でスキップされる（挙動は仕様）。([Hugging Face][1])            |
| **過度に平滑**     | 低解像（256）から直接 4K などへ巨大拡大                                | まず `1024×1024` へ（※SAM2 前処理解像度）→ `orig_hw` の順で段階的に可視化検証。公式も `1024→orig` の流れを前提。([PyTorch Docs][2])     |

> **要点**：\*\*postprocess は“ロジット → bilinear（align\_corners=False）→ 二値化”\*\*が原則です。これを崩すと「小さく見える／ぼやける」原因になります。([Hugging Face][1], [PyTorch Docs][2])

---

## 公式実装の根拠（該当行）

* **`SAM2Transforms.postprocess_masks` 本体**（`masks.float()` → 連結成分補正（任意） → **`F.interpolate(..., orig_hw, mode="bilinear", align_corners=False)`** → return）。([Hugging Face][1])
* **公式チュートリアル**（Torch‑TensorRT）：`masks = predictor._transforms.postprocess_masks(out["low_res_masks"], orig_hw)` の直後に **`(masks > 0.0)`**（= 二値化）。**postprocess の後に二値化**している。([PyTorch Docs][2])
* **OpenVINO ノートブック**：`low_res_masks` は **H=W=256** の前処理前マスクで、**postprocess 後に元解像度へ**。さらに **`multimask_output=False`** で **1 マスク運用**例。([OpenVINO Documentation][3])
* **`SAM2ImagePredictor` の返却仕様**：**`masks: B×C×H_orig×W_orig`**, **`low_res_masks: B×C×256×256`**, **`iou_predictions: B×C`** の記述。([Hugging Face][5])

---

## 実務向けコード断片（安全版）

```python
from sam2.utils.transforms import SAM2Transforms

# 例：1024 正方形前処理と閾値 0.0
sam2_trans = SAM2Transforms(resolution=1024, mask_threshold=0.0)

# ---- 低解像マスクの postprocess ----
# low_res_masks: [B, C, 256, 256]（SAM2のマスクデコーダ出力）
# orig_hw: (H_orig, W_orig)  ← 高さ→幅 の順！
masks_up = sam2_trans.postprocess_masks(low_res_masks, orig_hw)  # -> [B, C, H_orig, W_orig]

# 推論表示／評価時（二値化は最後）
binary = (masks_up > 0.0).to(torch.uint8)   # align_corners=False で補間後に閾値

# デバッグ用の形状チェック
assert low_res_masks.ndim == 4 and masks_up.ndim == 4
assert masks_up.shape[:2] == low_res_masks.shape[:2]
assert tuple(masks_up.shape[-2:]) == tuple(orig_hw)
```

---

## 付記：Qwen / LISA との統合に関する注意

* **前処理の統一**：SAM2 は**正方形リサイズ前提**（`Resize((res,res))`）。Qwen 側がアスペクト比維持の letterbox を使う場合は、**融合前に座標変換**または **同一の正方形画像から両経路テンソルを作る**運用が堅実です。座標不一致は**系統的な位置ずれ**の主因になります。
* **可視化も postprocess 出力で**：学習時の可視化でも\*\*必ず postprocess の出力（元解像度）\*\*を使い、独自 `interpolate` で代用しないこと。([PyTorch Docs][2])

---

### まとめ

* **入力**：`postprocess_masks` は **`[B, C, 256, 256]`** を前提（B と C を必ず持つ 4D）。
* **出力**：**`[B, C, H_orig, W_orig]`**（B/C を保持、空間のみ拡大）。
* **手順**：**ロジットのまま bilinear（`align_corners=False`）で拡大 → 最後に閾値二値化**（公式と同じ）。
* **“小さい/ぼやけ”対策**：`orig_hw` の順序、`align_corners=False`、dtype（float）と二値化の順番を厳守。

これらはすべて **SAM2 公式コードの `SAM2Transforms` 実装**および公式チュートリアルで確認できる内容です。([Hugging Face][1], [PyTorch Docs][2], [OpenVINO Documentation][3])

[1]: https://huggingface.co/spaces/fffiloni/SAM2-Image-Predictor/blob/main/sam2/utils/transforms.py "sam2/utils/transforms.py · fffiloni/SAM2-Image-Predictor at main"
[2]: https://docs.pytorch.org/TensorRT/tutorials/_rendered_examples/dynamo/torch_export_sam2.html?utm_source=chatgpt.com "Compiling SAM2 using the dynamo backend"
[3]: https://docs.openvino.ai/2024/notebooks/segment-anything-2-image-with-output.html?utm_source=chatgpt.com "Object masks from prompts with SAM2 and OpenVINO for ..."
[4]: https://huggingface.co/spaces/fffiloni/SAM2-Image-Predictor/blob/main/sam2/sam2_image_predictor.py?utm_source=chatgpt.com "fffiloni/SAM2-Image-Predictor at main"
[5]: https://huggingface.co/spaces/fffiloni/SAM2-Video-Predictor/blame/01bbb6ccc5c1344294b90cc0537ba89c6693b072/sam2/sam2_image_predictor.py?utm_source=chatgpt.com "sam2/sam2_image_predictor.py · fffiloni/SAM2-Video- ..."
