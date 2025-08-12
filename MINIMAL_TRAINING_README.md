# LISA改 ミニマルトレーニングガイド

## 概要

このドキュメントは、Qwen2.5-VLとSAM2.1を統合したLISA改モデルのトレーニング方法を説明します。

### 主な改善点（2025年1月版）

1. **加算アプローチによる埋め込み融合**
   - SAMの位置情報を100%保持しつつLLM情報を追加
   - 学習可能なβパラメータによる自動最適化

2. **アライメントステージ（ステージ0）**
   - LLM埋め込みをSAM埋め込みに事前調整
   - seg_lossの初期値を大幅に改善

3. **アライメントチェックポイントのキャッシュ**
   - 一度実行したアライメントを再利用可能
   - 開発効率の大幅向上

## クイックスタート

### 最小構成での動作確認

```bash
# 5サンプルで動作確認（約1分）
python minimal_train.py \
  --samples_per_epoch 5 \
  --batch_size 1 \
  --num_epochs 1 \
  --fast_dev_run

# アライメント付き（推奨）
python minimal_train.py \
  --samples_per_epoch 5 \
  --batch_size 1 \
  --num_epochs 1 \
  --fast_dev_run \
  --align_steps 10
```

### 標準的な学習設定

```bash
# 中規模学習（数時間）
python minimal_train.py \
  --samples_per_epoch 1000 \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --num_epochs 5 \
  --align_steps 100 \
  --use_cached_alignment \
  --save_steps 100 \
  --visualize \
  --visualize_steps 20
```

## アライメントステージ機能

### 基本的な使い方

```bash
# 初回：アライメントを実行（約30分/2000ステップ）
python minimal_train.py --align_steps 2000 --samples_per_epoch 500

# 2回目以降：キャッシュを使用（アライメントをスキップ）
python minimal_train.py --align_steps 2000 --use_cached_alignment --samples_per_epoch 500
```

### アライメントチェックポイントの管理

アライメント完了後、以下のファイルが自動生成されます：

```
checkpoints/alignment/
├── align_2000.pt      # モデルの重み
└── align_2000.json    # メタデータ（人間が読める形式）
```

### 高度な使い方

```bash
# 特定のチェックポイントを指定
python minimal_train.py \
  --alignment_checkpoint "checkpoints/alignment/align_2000.pt" \
  --samples_per_epoch 1000

# キャッシュを無視して再実行
python minimal_train.py \
  --align_steps 2000 \
  --force_realign \
  --samples_per_epoch 500
```

## データセット設定

### 単一データセット

```bash
# セマンティックセグメンテーションのみ
python minimal_train.py --dataset_types "sem_seg" --sample_rates "1.0"

# Referring Segmentationのみ
python minimal_train.py --dataset_types "refer_seg" --sample_rates "1.0"

# VQAのみ
python minimal_train.py --dataset_types "vqa" --sample_rates "1.0"

# Reasoning Segmentationのみ
python minimal_train.py --dataset_types "reason_seg" --sample_rates "1.0"
```

### 複数データセット

```bash
# セグメンテーション特化（VQA除外）
python minimal_train.py \
  --dataset_types "sem_seg||refer_seg||reason_seg" \
  --sample_rates "6,3,1" \
  --samples_per_epoch 1000

# 全データセット（デフォルト）
python minimal_train.py \
  --dataset_types "sem_seg||refer_seg||vqa||reason_seg" \
  --sample_rates "9,3,3,1" \
  --samples_per_epoch 5000
```

## 学習パラメータ調整

### メモリ効率重視

```bash
# GPUメモリ12GB以下向け
python minimal_train.py \
  --batch_size 1 \
  --gradient_accumulation_steps 16 \
  --samples_per_epoch 500 \
  --align_steps 50 \
  --use_cached_alignment
```

### 学習率の調整

```bash
# 高学習率（初期実験）
python minimal_train.py \
  --adapter_lr 5e-3 \
  --lora_lr 5e-4 \
  --seg_token_lr 1e-4 \
  --samples_per_epoch 200

# 低学習率（ファインチューニング）
python minimal_train.py \
  --adapter_lr 1e-4 \
  --lora_lr 1e-5 \
  --seg_token_lr 5e-6 \
  --samples_per_epoch 1000
```

### LoRA設定

```bash
# 小ランク（メモリ節約）
python minimal_train.py --lora_r 4 --lora_alpha 16

# 標準（デフォルト）
python minimal_train.py --lora_r 8 --lora_alpha 32

# 大ランク（表現力重視）
python minimal_train.py --lora_r 16 --lora_alpha 64
```

## 本格的な学習パターン

### 短期実験（1-2時間）

```bash
python minimal_train.py \
  --samples_per_epoch 500 \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --num_epochs 3 \
  --align_steps 100 \
  --use_cached_alignment \
  --save_steps 100 \
  --visualize \
  --visualize_steps 20
```

### 中期学習（半日）

```bash
python minimal_train.py \
  --samples_per_epoch 2000 \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_epochs 5 \
  --align_steps 500 \
  --use_cached_alignment \
  --save_steps 200 \
  --adapter_lr 1e-3 \
  --lora_lr 1e-4 \
  --seg_token_lr 5e-5
```

### 本番学習（1日以上）

```bash
python minimal_train.py \
  --samples_per_epoch 10000 \
  --batch_size 4 \
  --gradient_accumulation_steps 8 \
  --num_epochs 10 \
  --align_steps 2000 \
  --use_cached_alignment \
  --save_steps 500 \
  --adapter_lr 5e-4 \
  --lora_lr 5e-5 \
  --seg_token_lr 1e-5 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --seg_loss_weight 1.0 \
  --use_wandb \
  --wandb_project "lisa-kai-production"
```

## モニタリングとデバッグ

### 可視化機能

```bash
# 可視化を有効化
python minimal_train.py \
  --visualize \
  --visualize_steps 10 \
  --samples_per_epoch 100
```

可視化結果は以下に保存されます：
- `outputs/minimal_train_YYYYMMDD_HHMMSS/visualizations/`
- 各ステップでマスク予測の比較画像を生成
- Dice Score、IoU、損失値を表示

### WandB統合

```bash
# WandBでモニタリング
python minimal_train.py \
  --use_wandb \
  --wandb_project "lisa-kai-experiment" \
  --samples_per_epoch 1000
```

### デバッグモード

```bash
# 詳細ログ出力
python minimal_train.py \
  --debug \
  --samples_per_epoch 10 \
  --batch_size 1
```

## 出力ファイル構造

```
outputs/minimal_train_YYYYMMDD_HHMMSS/
├── config.json                    # 学習設定
├── loss_curves.png                # 損失曲線グラフ
├── combined_loss_curves.png       # 統合損失グラフ
├── loss_history.json              # 損失履歴データ
├── checkpoints/
│   ├── best/                     # ベストモデル
│   ├── epoch_1/                  # エポック終了時
│   ├── step_100/                 # 定期保存
│   └── final/                    # 最終モデル
└── visualizations/               # 可視化結果（--visualize時）
    ├── step_000005.png
    ├── step_000005.json
    └── ...
```

## トラブルシューティング

### メモリ不足エラー

```bash
# バッチサイズを小さくする
--batch_size 1 --gradient_accumulation_steps 16

# LoRAランクを下げる
--lora_r 4 --lora_alpha 16

# サンプル数を減らす
--samples_per_epoch 100
```

### seg_lossが下がらない

```bash
# アライメントステップを増やす
--align_steps 500

# 学習率を調整
--adapter_lr 5e-3 --lora_lr 5e-4

# SAM側のLoRAランクを上げる（lisa_config.pyで設定）
sam_lora_r=16
```

### アライメントチェックポイントのリセット

```bash
# キャッシュを削除
rm -rf checkpoints/alignment/

# または強制再実行
--force_realign
```

## パラメータ説明

| パラメータ | デフォルト | 説明 |
|---------|----------|------|
| `--samples_per_epoch` | 10 | 1エポックあたりのサンプル数 |
| `--batch_size` | 4 | バッチサイズ |
| `--gradient_accumulation_steps` | 4 | 勾配累積ステップ数 |
| `--num_epochs` | 3 | エポック数 |
| `--align_steps` | 0 | アライメントステップ数（0=無効） |
| `--use_cached_alignment` | False | キャッシュされたアライメントを使用 |
| `--force_realign` | False | キャッシュを無視して再アライメント |
| `--adapter_lr` | 1e-3 | アダプター学習率 |
| `--lora_lr` | 1e-4 | LoRA学習率 |
| `--seg_token_lr` | 5e-5 | SEGトークン学習率 |
| `--lora_r` | 8 | LoRAランク |
| `--lora_alpha` | 32 | LoRAアルファ |
| `--seg_loss_weight` | 1.0 | セグメンテーション損失の重み |
| `--visualize` | False | 可視化を有効化 |
| `--visualize_steps` | 5 | 可視化間隔（ステップ） |

## 推奨される学習フロー

1. **初期実験**（動作確認）
   ```bash
   python minimal_train.py --samples_per_epoch 10 --fast_dev_run
   ```

2. **アライメント実行**（初回のみ）
   ```bash
   python minimal_train.py --align_steps 500 --samples_per_epoch 100
   ```

3. **パラメータ探索**（キャッシュ利用）
   ```bash
   for lr in 1e-3 5e-4 1e-4; do
     python minimal_train.py \
       --align_steps 500 \
       --use_cached_alignment \
       --adapter_lr $lr \
       --samples_per_epoch 500 \
       --num_epochs 3
   done
   ```

4. **本番学習**
   ```bash
   python minimal_train.py \
     --align_steps 2000 \
     --use_cached_alignment \
     --samples_per_epoch 10000 \
     --num_epochs 10 \
     --use_wandb
   ```

## 技術的詳細

### 加算アプローチの実装

```python
# 位置埋め込みを100%保持しつつLLM情報を追加
e_pos = sparse_embeddings[:, 0, :]  # SAMの位置埋め込み
beta_scaled = torch.sigmoid(self.prompt_beta)  # 学習可能な係数
e_add = e_pos + beta_scaled * llm_embed  # 加算融合
sparse_embeddings[:, 0, :] = e_add
```

### アライメントステージのロス

```python
# LLM埋め込みをSAM埋め込みに近づける
align_loss = MSE(e_llm, e_pos.detach())
```

## 更新履歴

- **2025.01.11**: 加算アプローチとアライメントステージを実装
- **2025.01.11**: アライメントチェックポイントのキャッシュ機能を追加
- **2025.01.11**: β値のモニタリング機能を追加

## 関連ファイル

- `minimal_train.py`: メイン学習スクリプト
- `src/models/lisa_model.py`: モデル実装（加算アプローチ）
- `src/config.py`: 設定クラス
- `test_residual_addition.py`: 実装テストスクリプト
- `md_files/current/residual_addition_pre_alignment20250811.md`: 技術仕様書