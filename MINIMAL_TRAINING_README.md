# LISA改 ミニマルトレーニングガイド

## 概要

このドキュメントでは、LISA改（Qwen2.5-VL-3B + SAM2.1）モデルを実データで訓練するためのミニマルな実装について説明します。

## 実装内容

### `minimal_train.py`

実データを使用した最小限のトレーニングスクリプトです。

**主な特徴:**
- マルチデータセット対応（セマンティック/参照セグメンテーション、VQA、推論セグメンテーション）
- LoRAによる効率的なファインチューニング
- 勾配累積とメモリ最適化（4ステップ）
- チェックポイント保存機能
- WandB対応（オプション）
- Loss推移の可視化機能
- O3推奨の最適化実装済み

### 主要コンポーネント

1. **モデル構成**
   - Qwen2.5-VL-3B: ビジョン・言語理解
   - SAM2.1: 高精度セグメンテーション
   - <SEG>トークン: セグメンテーション指示

2. **学習可能パラメータ**
   - LoRAアダプター（r=8, α=32）
   - 画像特徴アダプター
   - テキストプロンプト射影層（2層MLP + LayerNorm）
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

データセットを以下の構造で配置：

```
/path/to/your/data/
├── ADEChallengeData2016/             # セマンティックセグメンテーション
│   ├── images/
│   └── annotations/
├── coco/                            # COCOStuff
│   ├── train2017/
│   └── stuffthingmaps_trainval2017/
├── refcoco/                         # 参照セグメンテーション
│   ├── refcoco/
│   ├── refcoco+/
│   └── refcocog/
├── llava_instruct_150k.json        # VQAデータ
└── reason_seg/                      # 推論セグメンテーション
```

#### 対応データセット
- **sem_seg**: ADE20K (20,210サンプル), COCOStuff (118,287サンプル)
- **refer_seg**: RefCOCO/RefCOCO+/RefCOCOg (合計55,885サンプル)
- **vqa**: LLaVA Instruct 150k (157,712サンプル)
- **reason_seg**: ReasonSeg (239サンプル)

### 3. トレーニングの実行

```bash
# 基本的な実行（セマンティックセグメンテーションのみ）
python minimal_train.py \
    --dataset_types sem_seg \
    --samples_per_epoch 1000 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_epochs 3

# マルチデータセットでの訓練（セマンティック + 参照セグメンテーション）
python minimal_train.py \
    --dataset_types "sem_seg||refer_seg" \
    --sample_rates "7,3" \
    --samples_per_epoch 1000 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_epochs 3

# 全データセットタイプを使用した大規模訓練
python minimal_train.py \
    --dataset_types "sem_seg||refer_seg||vqa||reason_seg" \
    --sample_rates "9,3,3,1" \
    --samples_per_epoch 10000 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_epochs 10 \
    --save_steps 500 \
    --use_wandb \
    --wandb_project lisa-kai-full

# デバッグモードでの実行
python minimal_train.py \
    --dataset_types "sem_seg||refer_seg" \
    --sample_rates "5,5" \
    --samples_per_epoch 10 \
    --batch_size 1 \
    --num_epochs 1 \
    --save_steps 5 \
    --debug
```

## パラメータ説明

### データ関連
- `--data_dir`: データセットのベースディレクトリ（省略時はLISAConfigのデフォルト値を使用）
- `--samples_per_epoch`: 1エポックあたりのサンプル数（デフォルト: 1000）
- `--dataset_types`: データセットタイプ（||で区切る）（デフォルト: sem_seg）
- `--sample_rates`: 各データセットのサンプルレート（,で区切る）（デフォルト: 1.0）

### 訓練設定
- `--batch_size`: バッチサイズ（デフォルト: 4）
- `--gradient_accumulation_steps`: 勾配累積ステップ数（デフォルト: 4）
- `--num_epochs`: エポック数（デフォルト: 3）
- `--warmup_ratio`: ウォームアップ比率（デフォルト: 0.1）

### 学習率
- `--adapter_lr`: アダプター学習率（デフォルト: 1e-3）
- `--lora_lr`: LoRA学習率（デフォルト: 1e-4）
- `--seg_token_lr`: SEGトークン学習率（デフォルト: 5e-5）

### LoRA設定
- `--lora_r`: LoRAランク（デフォルト: 8）
- `--lora_alpha`: LoRAアルファ（デフォルト: 32）

### その他
- `--save_steps`: チェックポイント保存間隔（デフォルト: 100）
- `--use_wandb`: WandB使用フラグ
- `--wandb_project`: WandBプロジェクト名
- `--debug`: デバッグモード有効化

## WandB統合

### 自動設定
`--use_wandb`フラグを使用すると、自動的にWandBが有効になります：
```bash
python minimal_train.py --use_wandb --wandb_project my-project
```

APIキーは自動的に設定されます（環境変数で上書き可能）。

### 手動設定（推奨）
```bash
# 環境変数で設定
export WANDB_API_KEY=your_api_key_here

# または .env ファイルを作成
cp .env.example .env
# .env ファイルを編集してAPIキーを設定
```

## 出力構造

```
outputs/
└── minimal_train_YYYYMMDD_HHMMSS/
    ├── config.json          # 訓練設定
    ├── loss_history.json    # Loss履歴
    ├── loss_curves.png      # Lossグラフ（4分割）
    ├── combined_loss_curves.png  # 統合Lossグラフ
    └── checkpoints/
        ├── best/           # ベストモデル
        ├── epoch_1/        # エポック終了時
        ├── step_100/       # ステップチェックポイント
        └── final/          # 最終モデル
```

## メモリ要件

- GPU: 最小16GB VRAM推奨（バッチサイズ2 + Gradient Accumulation 4の場合）
- RAM: 32GB以上推奨

### メモリ使用量の目安
- バッチサイズ1: ~12GB VRAM
- バッチサイズ2: ~16GB VRAM
- バッチサイズ4: ~24GB VRAM

## トラブルシューティング

### OOMエラーの場合
```bash
# バッチサイズを小さくし、Gradient Accumulationで補う
--batch_size 1 --gradient_accumulation_steps 8

# LoRAランクを小さくする
--lora_r 4 --lora_alpha 16
```

### データセットが見つからない場合
- `--data_dir`パスを確認
- データセット構造が正しいか確認

## 次のステップ

1. **評価スクリプト**: mIoUやcIoUの計算
2. **推論最適化**: バッチ推論やストリーミング対応
3. **EdgeLoss実装**: エッジ認識精度の向上

## O3推奨の最適化実装済み

1. **Gradient Accumulation**: 実効バッチサイズの増加
2. **LoRA α=32**: 学習初期の安定性向上
3. **2層MLP + LayerNorm**: マッチング精度の向上
4. **トークン数上限チェック**: RoPE制限（2048トークン）への対応
5. **Loss推移可視化**: 詳細な学習状況のモニタリング

## 参考情報

- [LISA論文](https://arxiv.org/abs/2308.00692)
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
- [SAM2.1](https://github.com/facebookresearch/sam2)

## 注意事項

- このスクリプトは実データでの動作を前提としています
- GPUメモリに応じてバッチサイズを調整してください
- 初回実行時はモデルのダウンロードに時間がかかります