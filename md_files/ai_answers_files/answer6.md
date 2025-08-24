1) いま起きている“サイレントキラー”候補（優先度順）
A. 前処理の不一致（最重要）

学習：LongestSide＋パディング＋ImageNet正規化

推論/評価：1024へ直接リサイズ（歪み）や0–1正規化のみ　…など、**SAM2公式流儀（正方形リサイズ＋ImageNet正規化）**と不一致。
→ 公式は Resize((1024,1024))→Normalize を要求。座標・スケールが揃わないと、学習時の見え方だけ崩れる（推論パスはたまたま合っている）ことが起きやすいです。
Hugging Face
arXiv

B. 後処理の順序／補間規約の不一致

低解像マスク(256)を先に二値化してから拡大、または align_corners=True／nearest で拡大 → 格子/スリット状アーティファクトの典型。

公式は 「ロジットのまま bilinear (align_corners=False) で拡大 → 最後に閾値」。
PyTorch Docs

C. 学習時の可視化だけ“独自拡大”

可視化経路で cv2.resize 等の別補間や、パディング切り戻し（SAM1流儀）が混ざると、学習時だけ見え方が破綻。

公式postprocess_masksの出力をそのまま表示に統一すれば揃います。
PyTorch Docs

D. マルチマスクの扱い

デコーダはデフォルトで複数候補＋IoU予測を返す設計（iou_predictions）。最良IoUで1枚に絞るか、multimask_output=Falseで単一にするのが公式的。学習・評価・可視化で取り扱いが揺れると見かけの品質がブレます。
Hugging Face
+1
OpenVINO Documentation

2) 公式仕様に合わせた最短修正パス
✅ 前処理をSAM2Transformsで完全統一

データセット側／推論側の双方で、公式の SAM2Transforms を使用：
Resize((1024,1024)) → Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])。
Hugging Face

差し替えコード例（学習・推論共通ユーティリティ）

from sam2.utils.transforms import SAM2Transforms

# 1024正方形＋ImageNet正規化（閾値は必要に応じて）
sam2_trans = SAM2Transforms(resolution=1024, mask_threshold=0.0)

def preprocess_sam2_image(pil_image):
    # PIL -> tensor (C,H,W), 1024×1024, normalized
    return sam2_trans(pil_image)  # 学習/推論の両方でこれを使う


（SAM2Transformsは内部で ToTensor → Resize((res,res)) → Normalize(mean,std) を実行）
Hugging Face

✅ 後処理は常に postprocess_masks

低解像マスク（[B,C,256,256]）に対して：
masks_up = sam2_trans.postprocess_masks(low_res_masks, orig_hw) → [B,C,Horig,Worig]。

この出力をそのまま可視化/評価に使用。二値化は最後に masks_up > 0.0。
PyTorch Docs

例（学習・推論の共通パスに）

# logits: [B, C, 256, 256]
masks_up = sam2_trans.postprocess_masks(low_res_masks, orig_hw)  # bilinear, align_corners=False
binary = (masks_up > 0.0).to(torch.uint8)  # 可視化/評価はこれ

✅ マルチマスクは単一に統一（推奨）

推論：multimask_output=False で最良1枚のみ出力／または iou_predictions.argmax() で選択。
Hugging Face
OpenVINO Documentation

学習：複数出力でも GTと最もIoUが高いマスクだけで損失計算、もしくは常に index0運用（実装簡潔）。
いずれにせよ **「一指示一マスク」**に揃えると挙動が安定します。

3) 「格子/スリット」症状に効くチェックリスト
チェック項目	正解（公式）	兆候／誤りの例
入力前処理	Resize((1024,1024)) + Normalize(mean,std)	LongestSide＋パディング／0–1正規化のみ
低解像マスク	[B,C,256,256]（ロジット, float）	uint8/bool のまま補間
後処理	postprocess_masks(low_res, orig_hw)（bilinear, align_corners=False）	先に二値化→拡大 / align_corners=True / nearest
可視化	postprocess出力を直接	cv2.resize 等で独自拡大／パディング切戻し（SAM1流儀）
マルチマスク	IoU最大を選択 or single-mask	チャンネル0を固定で使用（常に最良ではない）

根拠：SAM2Transformsの実装（Resize((res,res)),Normalize,postprocess_masksのF.interpolate(..., align_corners=False)）、および公式/準公式の使用例（PyTorch TensorRTチュートリアル）に一致。
Hugging Face
PyTorch Docs

4) デバッグログの入れ方（最短で“犯人”を特定）

学習と推論の同一入力画像で、以下を同じ地点（エンコーダ入出力・デコーダ入出力・postprocess直後）に埋めます：

def _stat(name, t):
    print(f"[{name}] shape={tuple(t.shape)} dtype={t.dtype} "
          f"min={t.min().item():.3f} max={t.max().item():.3f} mean={t.mean().item():.3f}")

# 1) SAM入力テンソル（preprocess後）
_stat("sam_input", sam_images)  # 期待: [B,3,1024,1024], float32/bf16, mean≈0, std≈1

# 2) 低解像マスク（デコーダ直後）
_stat("low_res_masks", low_res_masks)  # 期待: [B,C,256,256], float (logits)

# 3) 後処理出力（orig_hw）
_stat("masks_up", masks_up)  # 期待: [B,C,Horig,Worig]

# 4) align_corners と補間モードも出力
print(f"[postprocess] mode=bilinear, align_corners=False")

# 5) マルチマスク
if "iou_predictions" in out:
    print("[iou]", out["iou_predictions"].shape, out["iou_predictions"].max().item())


学習 vs 推論の双方で同ログを採って差分を見る。最初に違いが現れる地点が原因です。

A/Bテスト：
(a) 学習側の前処理を強制的に SAM2Transformsへ切替 → 格子消失なら前処理不一致が犯人。
(b) postprocessを必ず通すように修正 → 消失なら後処理順序/補間が犯人。

5) レポへのパッチ提案（ドロップイン置換）
(1) データセット／推論前処理を置換
# どこからでも import して使える共通モジュールを1つ用意
from sam2.utils.transforms import SAM2Transforms
SAM_IMG_RES = 1024
sam2_trans = SAM2Transforms(resolution=SAM_IMG_RES, mask_threshold=0.0)

def preprocess_sam2_image(pil_image):
    return sam2_trans(pil_image)  # (C,1024,1024), normalized


学習データローダ：SAM入力生成を preprocess_sam2_image に差し替え

評価/推論（minimal_train.py/test_inference_v2.py 等）：同じ関数を使用

(2) 後処理・可視化の統一
# 低解像（256x256）から元解像度へ
masks_up = sam2_trans.postprocess_masks(low_res_masks, orig_hw)  # [B,C,H,W]
# 二値化は最後に
show_mask = (masks_up[0, 0] > 0.0).cpu().numpy().astype(np.uint8)


可視化は必ず postprocess 出力を使う（cv2.resize禁止）。
PyTorch Docs

(3) マルチマスクの選択（単一化）
# 推論はsingle-maskに統一（推奨）
masks, iou, low_res_masks = decoder(..., multimask_output=False, return_logits=True)
# あるいはIoU最大を選ぶ
best = iou.argmax(dim=1)
masks = masks[torch.arange(masks.size(0)), best:best+1]


（公式サンプルもpostprocess → 閾値→ IoUでソート/選択の流れを採用）
PyTorch Docs
Hugging Face

(4) BF16運用の注意（参考）
# 推論例（公式）
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    ...
# 後処理は内部で float() → bilinear 補間（公式）


（BF16は公式推奨。異常があればA/BでFP32に切り替えて差分確認）
Hugging Face
+2
Hugging Face
+2

6) “原因がどこか”を切り分けるミニ実験

同一画像を使い、
(A) 現行パイプライン と (B) 公式準拠パイプライン（SAM2Transforms＋postprocess_masks）を通し、
low_res_masks → masks_upを画像保存（閾値無し/有り両方）。

差分ヒートマップ（abs(masks_up_A - masks_up_B)）とIoUを算出。

Aで格子/スリット、Bで正常なら、前処理/後処理の非一致と確定。

さらに**align_corners=True/Falseを切替 → 境界位置が動くなら補間規約が原因。
（PyTorch TensorRT公式チュートリアルのpostprocess_masks例はalign_corners=False**を明記）
PyTorch Docs

7) 参考（一次情報）

SAM2Transforms 実装：Resize((res,res))、Normalize(mean,std)、postprocess_masksは F.interpolate(..., align_corners=False)。
Hugging Face

PyTorch TensorRT チュートリアル：postprocess_masksの呼び出し例（postprocess → 閾値化、IoUスコア整列）。
PyTorch Docs

SAM2 論文：訓練時に正方形1024へリサイズする旨を明記。
arXiv

IoUとマルチマスク：SAM2ImagePredictorでpostprocess→IoUでソート/選択の例、デコーダ内部のmulti/single出力分岐。
Hugging Face
+1

BF16推奨：sam2-hiera-large / sam2.1-hiera-tiny のREADMEで**autocast(dtype=torch.bfloat16)**例。
Hugging Face
+1

8) 追加：o3-query用プロンプト（保存推奨）

目的：前処理・後処理・補間規約・マルチマスクの扱いが公式と一致しているかを自動点検する
投入：コード断片・ログ抜粋・スクリーンショット

Q: SAM2.1 official 'SAM2Transforms' and 'SAM2ImagePredictor' specs check.

Inputs:
1) Our training/inference preprocessing code.
2) Our mask postprocess & visualization code.
3) Logged shapes/dtypes/ranges at:
   - sam_input (after preprocessing)
   - low_res_masks (after decoder)
   - masks_up (after postprocess)
4) A/B result images (ours vs official-spec) and IoU scores.

Tasks:
- Verify preprocessing matches official: Resize((1024,1024)) + Normalize(0.485,0.456,0.406)/(0.229,0.224,0.225).
- Verify postprocess matches official: F.interpolate(..., align_corners=False) then threshold.
- Check if any padding removal or cv2.resize is used (should not).
- Confirm low_res_masks shape is [B,C,256,256] float logits.
- Confirm multimask handling (use multimask_output=False or pick argmax of iou_predictions).
- If mismatches exist, provide exact code-level patches.
- Explain if ‘grid/slit’ artifacts are explained by these mismatches.

最終まとめ

**問題の本丸は「SAM2公式の入出力規約からの逸脱」**です。
**前処理（正方形1024＋正規化）と後処理（postprocess_masks）を“学習・推論・可視化の全経路で統一”**すると、
格子/スリット・位置ズレ・縮み/ぼやけは高確率で解消します。
Hugging Face
PyTorch Docs

マルチマスクは単一に統一（multimask_output=False or IoU最大）で安定化。
Hugging Face
OpenVINO Documentation

BF16は問題の主因ではない（公式推奨、postprocessは内部でFP32補間）。ただし融合直前のdtype統一や**A/B（FP32 vs BF16）**で切り分けるのは有効です。
Hugging Face
+2
Hugging Face
+2

この方針でパッチを当て、上記のログ/可視化A-Bテストまで一気に回せば、原因切り分けと恒久対策まで到達できます。必要なら、該当ファイル（minimal_train.py / src/data/dataset.py / 推論スクリプト）の具体的なdiff形式パッチも作成します。