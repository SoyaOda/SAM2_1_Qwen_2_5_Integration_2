# Focus_Seg_Loss20250810 修正記録

## 概要
SEG LOSSが下がらない問題に対して実施したデータ整合性の修正と検証結果をまとめる。

作成日: 2025年8月10日

## 1. 問題の背景

### 1.1 発見された問題
minimal_train.pyを使用した学習において、以下の問題が確認された：
- SEG LOSSが1.3前後から下がらない
- Dice scoreが0.001-0.003程度と極めて低い
- 学習が進行してもセグメンテーション性能が改善しない

### 1.2 疑われた原因
データセット関連のScriptに以下の問題がある可能性：
1. ランダムサンプリング機能により異なるサンプルのGT Maskがラベルとして使われている
2. テキストのラベルがcopyやcloneで入力と同じものを使用している
3. 前処理の整合性が取れていない

## 2. 実施した修正

### 2.1 データセットクラスの修正

#### SemSegDataset (src/data/sem_seg_dataset.py)
**修正内容**：
- `__getitem__`メソッドで一貫したサンプル取得を保証
- conversationsとmasksが同じサンプルから生成されることを確認
- 1会話1マスクに統一（num_classes_per_sample=1）

**主要コード**：
```python
# 650-663行目: 会話形式の生成
conversations = []
for i in range(len(questions)):
    messages = [
        {"role": "user", "content": questions[i]},
        {"role": "assistant", "content": answers[i]}
    ]
    conversations.append(messages)

# 684行目: 対応するクラスのマスク生成
mask = (label_tensor == class_id).float()
masks = mask.unsqueeze(0)  # (1, H, W)形式
```

### 2.2 HybridDataset (src/data/dataset.py)
**修正内容**：
- messagesの正しい構築（user/assistantのrole設定）
- 画像とテキストの適切な組み合わせ
- SEGトークンの確実な配置

**主要コード**：
```python
# 864-884行目: メッセージ形式の構築
if conversation_messages:
    messages = []
    for msg in conversation_messages:
        if msg["role"] == "user":
            user_message = {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_pil},
                    {"type": "text", "text": msg["content"]}
                ]
            }
            messages.append(user_message)
        else:
            assistant_message = {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": msg["content"]}
                ]
            }
            messages.append(assistant_message)
```

### 2.3 MultiModalDataCollator (src/data/collators.py)
**修正内容**：
- test_a1準拠のバッチ処理
- 2D pixel_values形式の適切な処理
- image_grid_thwの正しい計算
- original_imagesの保持（可視化用）

**特徴**：
- Qwen2.5-VLの動的解像度に対応
- パッチ形式（N_patches, D_v）の適切な処理
- グリッド次元の正確な計算

### 2.4 minimal_train.pyの可視化機能追加
**追加機能**：
- `--visualize`フラグで可視化の有効/無効を制御
- `--visualize_steps`で可視化間隔を指定
- 6パネル可視化（入力画像、GTマスク、予測マスク、オーバーレイ、メトリクス）
- JSONファイルでメトリクスとテキスト情報を保存

## 3. 検証結果

### 3.1 データ整合性の確認
**確認項目**：
- ✅ 質問と回答のペアが正しく対応
- ✅ マスクが対応するクラスから生成
- ✅ user/assistantのroleが適切に設定
- ✅ SEGトークンが回答に含まれる
- ✅ ラベルマスキングが適切（masked_tokens: 293-319、unmasked_tokens: 4-5）

### 3.2 可視化による確認
**サンプル例（step_000200.json）**：
```json
{
  "text": {
    "user": "What celestial object can be identified in the picture...",
    "assistant": "Sure, <SEG>. In the picture, the moon can be seen...",
    "masked_tokens": 319,
    "unmasked_tokens": 5
  }
}
```

**サンプル例（step_001500.json）**：
```json
{
  "text": {
    "user": "Please provide a segmentation mask for the stairs.",
    "assistant": "Sure, the segmentation result is <SEG>.",
    "masked_tokens": 293,
    "unmasked_tokens": 4
  }
}
```

### 3.3 解決された問題
1. **ランダムサンプリング問題**: 各`__getitem__`で一貫したサンプル取得を実現
2. **コピー/クローン問題**: assistantの回答に適切な内容とSEGトークン配置
3. **前処理の整合性**: 画像、テキスト、マスクが同一サンプルから生成

## 4. 残存する課題

### 4.1 SEG LOSSが下がらない問題
データの整合性は確認できたが、SEG LOSSが依然として下がらない。

**考えられる原因**：
1. **モデルアーキテクチャの問題**
   - VLM→SAM接続部分の実装
   - SEGトークンの埋め込み処理
   - TextPromptProjectorの初期化

2. **学習設定の問題**
   - 学習率の調整が必要
   - seg_loss_weightのバランス
   - 勾配の流れ方

3. **SAM側の問題**
   - point promptingの実装
   - マスクデコーダーのLoRA適用
   - 座標変換の精度

## 5. 次のステップ

### 5.1 A-1テスト（SAM側のオラクル・プロンプト過学習）
Focus_Seg_Loss20250810_spec.mdに従い、以下を実施：
- VLMを迂回してSAM2.1単体の学習能力を検証
- GTの重心点を直接SAMに入力
- 1画像1マスクの極小データで過学習
- 期待値：Dice ≥ 0.95、mIoU ≥ 0.9

### 5.2 A-2テスト（VLM→SAM接続テスト）
- SEGトークンの隠れ表現→SAM Promptの接続検証
- 埋め込みの寸法/正規化/座標変換の確認

## 6. 実装の使用方法

### 6.1 可視化機能付き訓練
```bash
python minimal_train.py \
    --dataset_types "sem_seg||refer_seg||vqa||reason_seg" \
    --sample_rates "9,3,3,1" \
    --samples_per_epoch 1000 \
    --batch_size 2 \
    --gradient_accumulation_steps 16 \
    --num_epochs 3 \
    --seg_loss_weight 3.0 \
    --adapter_lr 5e-3 \
    --seg_token_lr 1e-3 \
    --lora_lr 1.5e-4 \
    --save_steps 2000 \
    --visualize \
    --visualize_steps 100
```

### 6.2 可視化出力
- 保存先：`outputs/minimal_train_{timestamp}/visualizations/`
- PNG画像：6パネル可視化
- JSONファイル：メトリクスとテキスト情報

## 7. まとめ

データ前処理とラベルの整合性については徹底的に修正・検証を行い、問題がないことを確認した。
しかし、SEG LOSSが下がらない根本的な問題は解決していない。
次はFocus_Seg_Loss20250810_spec.mdに従い、段階的なテストを実施してモデルアーキテクチャや学習設定の問題を特定する必要がある。

## 8. 参照ファイル

### 修正したファイル
- src/data/sem_seg_dataset.py
- src/data/dataset.py  
- src/data/collators.py
- minimal_train.py

### 関連ドキュメント
- md_files/current/Focus_Seg_Loss20250810_spec.md
- outputs/minimal_train_20250810_165603/visualizations/

## 9. 技術的詳細

### 9.1 Qwen2.5-VLの動的解像度対応
- min_pixels: 112896、max_pixels: 7225344
- 14の倍数にパディング
- image_grid_thw形式：[T, H_grid, W_grid]

### 9.2 ラベルマスキング戦略
- システムメッセージ：-100でマスク
- ユーザーメッセージ：-100でマスク  
- アシスタントメッセージ：学習対象（SEGトークン周辺のみ）

### 9.3 座標変換システム
- CoordinateTransform使用
- オリジナル座標→Qwen座標→SAM座標の変換
- 重心ベースのpoint prompting準備済み