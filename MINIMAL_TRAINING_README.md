# LISA改 ミニマルトレーニングガイド

## 概要

このドキュメントでは、LISA改（Qwen2.5-VL-3B + SAM2.1）モデルを実データで訓練するためのミニマルな実装について説明します。

## 実装内容

### `minimal_train.py`

実データを使用した最小限のトレーニングスクリプトです。

**主な特徴:**
- セマンティックセグメンテーション（ADE20K）データセットに特化
- LoRAによる効率的なファインチューニング
- 勾配累積とメモリ最適化
- チェックポイント保存機能
- WandB対応（オプション）

### 主要コンポーネント

1. **モデル構成**
   - Qwen2.5-VL-3B: ビジョン・言語理解
   - SAM2.1: 高精度セグメンテーション
   - <SEG>トークン: セグメンテーション指示

2. **学習可能パラメータ**
   - LoRAアダプター（r=8, α=16）
   - 画像特徴アダプター
   - テキストプロンプト射影層
   - <SEG>トークン埋め込み

3. **損失関数**
   - 言語モデリング損失（CrossEntropy）
   - セグメンテーション損失（BCE + Dice）

## 実行方法

### 1. 環境準備

```bash
# 必要なライブラリのインストール
pip install -r requirements.txt

# SAM2.1チェックポイントのダウンロード
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -P checkpoints/
```

### 2. データセットの準備

ADE20Kデータセットを以下の構造で配置：

```
/path/to/your/data/
├── ADEChallengeData2016/
│   ├── images/
│   │   ├── training/
│   │   └── validation/
│   └── annotations/
│       ├── training/
│       └── validation/
```

### 3. トレーニングの実行

```bash
# 基本的な実行（デフォルトのデータセットパスを使用）
python minimal_train.py \
    --samples_per_epoch 1000 \
    --batch_size 4 \
    --num_epochs 3

# カスタムデータパスを指定する場合
python minimal_train.py \
    --data_dir /path/to/your/data \
    --samples_per_epoch 1000 \
    --batch_size 4 \
    --num_epochs 3
```

## パラメータ説明

### データ関連
- `--data_dir`: データセットのベースディレクトリ（省略時はLISAConfigのデフォルト値を使用）
- `--samples_per_epoch`: 1エポックあたりのサンプル数（デフォルト: 1000）

### 訓練設定
- `--batch_size`: バッチサイズ（デフォルト: 4）
- `--num_epochs`: エポック数（デフォルト: 3）
- `--warmup_ratio`: ウォームアップ比率（デフォルト: 0.1）

### 学習率
- `--adapter_lr`: アダプター学習率（デフォルト: 1e-3）
- `--lora_lr`: LoRA学習率（デフォルト: 1e-4）
- `--seg_token_lr`: SEGトークン学習率（デフォルト: 5e-5）

### LoRA設定
- `--lora_r`: LoRAランク（デフォルト: 8）
- `--lora_alpha`: LoRAアルファ（デフォルト: 16）

### その他
- `--save_steps`: チェックポイント保存間隔（デフォルト: 100）
- `--use_wandb`: WandB使用フラグ
- `--wandb_project`: WandBプロジェクト名

## 出力構造

```
outputs/
└── minimal_train_YYYYMMDD_HHMMSS/
    ├── config.json          # 訓練設定
    ├── checkpoints/
    │   ├── best/           # ベストモデル
    │   ├── epoch_1/        # エポック終了時
    │   ├── step_100/       # ステップチェックポイント
    │   └── final/          # 最終モデル
    └── logs/               # ログファイル
```

## メモリ要件

- GPU: 最小16GB VRAM推奨（バッチサイズ4の場合）
- RAM: 32GB以上推奨

## トラブルシューティング

### OOMエラーの場合
```bash
# バッチサイズを小さくする
--batch_size 2

# LoRAランクを小さくする
--lora_r 4 --lora_alpha 8
```

### データセットが見つからない場合
- `--data_dir`パスを確認
- データセット構造が正しいか確認

## 次のステップ

1. **マルチタスク訓練**: 他のデータセット（RefCOCO、VQA等）を追加
2. **評価スクリプト**: mIoUやcIoUの計算
3. **推論最適化**: バッチ推論やストリーミング対応

## 参考情報

- [LISA論文](https://arxiv.org/abs/2308.00692)
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
- [SAM2.1](https://github.com/facebookresearch/sam2)

## 注意事項

- このスクリプトは実データでの動作を前提としています
- GPUメモリに応じてバッチサイズを調整してください
- 初回実行時はモデルのダウンロードに時間がかかります