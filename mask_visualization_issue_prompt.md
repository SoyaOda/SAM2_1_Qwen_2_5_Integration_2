# LISA改 (Qwen2.5-VL + SAM2.1) マスク可視化の4分割問題について

## 背景
LISA改という、Qwen2.5-VL（3Bモデル）とSAM2.1（Hiera-Large）を統合したマルチモーダルセグメンテーションモデルを開発しています。LISAの設計思想に基づき、VLMの言語理解能力とSAMの精密なセグメンテーション能力を組み合わせています。

## モデルアーキテクチャ

### 全体構造
```
入力画像 → 2つの経路で処理：
1. Qwen2.5-VL経路：448×448解像度 → Vision Encoder → 2048次元特徴
2. SAM2.1経路：1024×1024解像度 → SAM Image Encoder → 256次元特徴
```

### 主要コンポーネント

1. **Qwen2.5-VL (3B)**
   - Vision Encoder: ViT with PatchMerge (spatial_merge_size=2)
   - 出力: 2048次元のLLM投影済み特徴
   - LoRA適用 (r=8, alpha=32)

2. **SAM2.1 (Hiera-Large)**
   - Image Encoder: 256チャンネル出力
   - Mask Decoder: LoRA適用 (r=8, alpha=64)
   - Prompt Encoder: 凍結

3. **Token-FPN**
   - Qwenの中間層特徴を抽出（4層: 6, 13, 20, 27）
   - 各層を256次元に投影
   - FPN構造でマルチスケール特徴を生成

4. **特徴融合**
   - Qwen特徴とSAM特徴をSigma-Add Fusion（学習可能なβパラメータ）で融合

### パラメータ統計
- 総パラメータ数: 3,988,465,588
- 学習可能パラメータ: 9,395,970 (0.24%)
  - Token-FPN: 4,462,848 (47.5%)
  - Qwen LoRA: 2,506,752 (26.7%)
  - Text Prompt Projector: 1,181,952 (12.6%)
  - Image Adapter: 1,180,928 (12.6%)
  - SAM LoRA: 61,440 (0.7%)

## 観測されている問題

### 訓練時の可視化（step_000010.png）
- **現象**: Predicted Maskが4つの正方形に分割されている
- **詳細**: 左上の正方形のみが適切にマスク生成されており、他の3つの領域は不活性
- **画像サイズ**: 448×448（可視化時）

### 推論評価時（inference_results/）
- **現象**: 正常にマスクが生成されている
- **詳細**: 統一された単一のマスクとして出力
- **画像サイズ**: 元画像サイズを維持

## コード詳細

### マスク取得処理（訓練時の可視化）
```python
# minimal_train.py line 1169-1174
pred_mask = outputs.mask_logits[0]
if isinstance(pred_mask, list):
    pred_mask = pred_mask[0]
pred_mask_np = torch.sigmoid(pred_mask).detach().cpu().numpy()
if pred_mask_np.ndim > 2:
    pred_mask_np = pred_mask_np.squeeze()
```

### マスク取得処理（推論時）
```python
# minimal_train.py line 1601-1608
if hasattr(outputs, 'mask_logits') and outputs.mask_logits is not None:
    if len(outputs.mask_logits) > 0 and outputs.mask_logits[0] is not None:
        mask_logit = outputs.mask_logits[0]
        if isinstance(mask_logit, list):
            mask_logit = mask_logit[0]
        pred_mask = torch.sigmoid(mask_logit).detach().cpu().numpy()
        if pred_mask_np.ndim > 2:
            pred_mask_np = pred_mask_np.squeeze()
```

### SAM Mask Decoderへの入力
```python
# lisa_model.py - SAM decoderへの入力
# image_embeddings: [B, 256, 64, 64] (SAM encoder出力をQwen特徴と融合済み)
# prompt_embeds: テキストプロンプトから生成された256次元ベクトル
# sparse_prompt_embeddings と dense_prompt_embeddings として入力
```

## 質問

1. **この4分割の現象は正常な学習初期の挙動でしょうか？**
   - Token-FPNが4層の特徴を抽出しているため、各層が独立して活性化している？
   - SAM2.1のマルチスケール処理の影響？
   - それとも実装上の問題？

2. **なぜ左上の正方形だけが活性化しているのでしょうか？**
   - 最低解像度の特徴が最初に学習される？
   - 位置埋め込みの問題？

3. **推論時には正常に見える理由は？**
   - 後処理で統合されている？
   - 可視化方法の違い？

4. **改善方法はありますか？**
   - Token-FPNの出力を異なる方法で統合すべき？
   - SAM Decoderへの入力方法を変更すべき？
   - 学習率やウォームアップの調整？

## 追加情報
- 使用しているデータセット: LISA dataset (ADE20K, COCO-Stuff, RefCOCO, VQA, ReasonSeg)
- 学習設定: batch_size=2, gradient_accumulation=16, learning_rate=1e-3
- 1エポック目の10ステップ時点での観測

## 期待する回答
この現象が正常な学習プロセスの一部なのか、それとも修正が必要な実装上の問題なのかを判断したいです。特に、SAM2.1とToken-FPNを組み合わせた際の期待される挙動について、専門的な知見をいただければ幸いです。