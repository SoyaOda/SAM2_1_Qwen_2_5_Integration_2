### 全体像 — 「**低解像度で推論 → 高解像度でマスク**」を破綻させない３レイヤ設計

```
Qwen (448×448, patch=14px) ──┐          ┌─ GT mask (任意解像度)
                             │token/box  │    ▲
                             ▼           │    │
   ┌─▶ ① 低解像度 Loss (optional)       │    │      ②座標変換
   │          │                         │    │
   │          ▼                         │    │
   │     SAM Prompt (pts/box) ─────────▶ SAM2.1 (1024×1024) ──▶ ③ 高解像度 Loss
   │                                     ▲
   └─────────────────────────────────────┘
```

---

## 1  Qwen 448 × 448 と高解像 GT マスクのズレ対策

| 課題                                               | ベストプラクティス                                                                                                                           |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| **位置ずれ**：Qwen の patch 座標 (448) → SAM の 1024 でマスク | *リサイズ & パディングのスケール係数を保存*<br>`sx = 1024 / w_qwen_no_pad`, `sy = 1024 / h_qwen_no_pad` を使い **点 / box** を座標射影してから `SAMPromptEncoder` へ |
| **教師マスクの扱い**                                     | ① *補助損失*：GT を **nearest で 448 に down‑sample** し、Qwen が返す coarse mask に BCE / Dice<br>② *主損失*：1024 マスク(SAM出力) と **元解像 GT** を比較       |
| **どちらを合わせるか**                                    | *推論*: マスクは 1024 を維持し可視化品質を確保<br>*学習*: **両解像度で２段 Loss**（coarse＋fine）を推奨 ― OTAS も 1/8 マップを mask‐refine で高解像へ上げている ([arXiv][1])        |

### コード断片（座標射影）

```python
# img_q: 448 包含 pad, (C, Hq, Wq)
# 1) scale & offsetを保存
scale = 448 / max(orig_h, orig_w)
left_pad  = (448 - orig_w*scale) / 2
top_pad   = (448 - orig_h*scale) / 2

def qwen2sam(x_q, y_q):
    x_orig = (x_q - left_pad) / scale
    y_orig = (y_q - top_pad)  / scale
    x_sam  = x_orig * (1024 / orig_w)
    y_sam  = y_orig * (1024 / orig_h)
    return x_sam, y_sam
```

---

## 2  損失計算時の解像度合わせ

| ステージ                                     | 推奨解像度                                             | 処理                                                    | 理由                       |
| ---------------------------------------- | ------------------------------------------------- | ----------------------------------------------------- | ------------------------ |
| **低解像補助**                                | 448                                               | `GT↓= F.interpolate(GT.float(), 448, mode='nearest')` | coarse mask で Qwen を直接監督 |
| **高解像主損失**                               | 1024                                              | `pred = pred.float();  loss = Dice(pred, GT_high)`    | 境界細部を維持した IoU/Dice が得やすい |
| **`F.interpolate(..., mode='nearest')`** | 最近傍指定は整数ラベル保持が community 推奨 ([PyTorch Forums][2]) |                                                       |                          |

---

## 3  Qwen 448 → SAM 1024 へ解像度ギャップを埋める３手法

| 手法                          | 処理量 | 精度  | 実装ポイント                                                                             |
| --------------------------- | --- | --- | ---------------------------------------------------------------------------------- |
| **A. 直接スケール (最小構成)**        | ★   | ◯   | 上記 `qwen2sam()` で pts/box を変換、`SAMPromptEncoder` へ                                 |
| **B. ２段マスク (Sa2VA 流)**      | ★★  | ★★  | Qwen patch Token → *instruction token* → SAM high‑res decoderへ ([arXiv][3])        |
| **C. Mask‑refine (OTAS 流)** | ★★★ | ★★★ | Qwen coarse mask (1/8,1/16) を **SAM2.1 mask refinement**ネットで upsample ([arXiv][1]) |

> **実践**：まず *A* で動かし、精度不足を感じたら *B/C* を足すのが安全。

---

## 4  448×448 画像の作成 & `image_grid_thw`

* **アスペクト維持＋ゼロパディング (14 px 倍数)**

  ```python
  img = cv2.resize(orig, (round(w*scale), round(h*scale)), interpolation=cv2.INTER_AREA)
  pad_h = (-img.shape[0]) % 14; pad_w = (-img.shape[1]) % 14
  img = cv2.copyMakeBorder(img,0,pad_h,0,pad_w,cv2.BORDER_CONSTANT, value=0)
  ```
* **Processor 使用**

  ```python
  prompt = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
  inputs = proc(text=[prompt], images=[img], return_tensors="pt")
  # これで `pixel_values (B,3,H,W)` と `image_grid_thw=[[1,H/14,W/14]]` が同梱される :contentReference[oaicite:4]{index=4}
  ```
* 手動前処理の場合は

  ```python
  grid = torch.tensor([[1, H//14, W//14]], dtype=torch.long)
  model(pixel_values=img_t, image_grid_thw=grid, input_ids=ids)
  ```

---

## 5  データ型とメモリ最適化

| データ                    | dtype (CPU)      | dtype (GPU)                                    | 備考                  |
| ---------------------- | ---------------- | ---------------------------------------------- | ------------------- |
| **画像**                 | `uint8`          | `float16` (Qwen) / `float32` (SAM, fine‑tune時) |                     |
| **マスク (interpolate前)** | —                | `float32`                                      |                     |
| **マスク (後 / 保存)**       | `uint8` / `bool` | `bool`                                         | `bool` で BCE/Dice 可 |

---

## 6  参考コード（簡略 LISA ループ）

```python
# --- 前処理 ---
sam_img  = resize_longest(orig, 1024)     # black pad
qwen_img = resize_and_pad(orig, 448, 14)  # keep ratio

# --- Qwen 推論 (box/coord) ---
q_in   = proc(text=[prompt], images=[qwen_img], return_tensors="pt").to(dev)
q_out  = qwen.generate(**q_in, max_new_tokens=32)
pts448 = parse_points_from_llm(q_out)     # e.g. [[x,y],...]

# --- 座標変換 ---
pts1024 = [qwen2sam(x,y) for x,y in pts448]

# --- SAM 推論 ---
sam_in  = sam_preprocess(sam_img)         # mean/std norm
masks   = sam.predict(point_coords=pts1024)  # (M,1024,1024)

# --- 損失 ---
gt_low  = F.interpolate(gt_mask.float(), size=(448,448), mode="nearest")
loss    = λ1*Dice(qwen_mask, gt_low) + λ2*Dice(masks, gt_mask_high)
```

---

### まとめチェックリスト

| ✅                                             | 項目 |
| --------------------------------------------- | -- |
| `image_grid_thw` は `[1,H/14,W/14]` を Qwen に渡す |    |
| GT マスクは *nearest* で up/down サンプリング            |    |
| Qwen→SAM 座標変換は **(scale, pad) を保存して按分**       |    |
| multi‑scale Loss：448 補助 + 1024 主損失            |    |
| 画像 `float16`、マスクスコア `float32` で VRAM を節約      |    |

以上を守れば **LISA 改（Qwen2.5‑VL＋SAM2.1）** での解像度ギャップやマスクズレを最小限に抑えつつ、高精度かつ軽量なトレーニング／推論パイプラインを構築できます。

[1]: https://arxiv.org/html/2507.08851v1 "OTAS: Open-vocabulary Token Alignment for Outdoor Segmentation"
[2]: https://discuss.pytorch.org/t/resizing-semantic-segmentation-labels/143246 "Resizing semantic segmentation labels - vision - PyTorch Forums"
[3]: https://arxiv.org/abs/2501.04001 "[2501.04001] Sa2VA: Marrying SAM2 with LLaVA for Dense Grounded Understanding of Images and Videos"
