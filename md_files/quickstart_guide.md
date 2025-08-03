# LISA改 クイックスタートガイド

## インストール

### 1. 環境準備

```bash
# Python 3.9以上が必要
python --version

# CUDAの確認（11.8以上推奨）
nvidia-smi
```

### 2. リポジトリのクローン

```bash
git clone https://github.com/SoyaOda/SAM2_1_Qwen_2_5_Integration_2.git
cd SAM2_1_Qwen_2_5_Integration_2
```

### 3. 仮想環境の作成（推奨）

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# または
venv\Scripts\activate  # Windows
```

### 4. 依存関係のインストール

```bash
# 基本的な依存関係
pip install -r requirements.txt

# SAM2のインストール
pip install git+https://github.com/facebookresearch/sam2.git

# Flash Attention（オプション、高速化用）
pip install flash-attn --no-build-isolation
```

### 5. モデルのダウンロード

```bash
# ディレクトリ作成
mkdir -p checkpoints

# SAM2.1チェックポイントのダウンロード
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# Qwen2.5-VL-3Bは初回実行時に自動ダウンロード
```

## 基本的な使い方

### 1. 単一画像の推論

```bash
# CLIから実行
./scripts/run_inference.sh "path/to/image.jpg" "Please segment the cat in this image"

# Pythonスクリプトから実行
python inference.py \
    --model_path output/lisa_kai_experiment/final_model \
    --image path/to/image.jpg \
    --instruction "Please segment the cat in this image" \
    --output_dir ./results
```

### 2. Pythonコードでの使用

```python
from inference import LISAInference
from PIL import Image

# 推論パイプラインの初期化
pipeline = LISAInference(
    model_path="output/lisa_kai_experiment/final_model",
    device="cuda"
)

# 画像とインストラクションを準備
image = Image.open("path/to/image.jpg")
instruction = "Please segment all the apples in this image"

# 推論実行
text_response, mask = pipeline.generate_response(
    image=image,
    instruction=instruction,
    max_new_tokens=100,
    temperature=0.7
)

print(f"Response: {text_response}")
if mask is not None:
    print(f"Mask shape: {mask.shape}")
    # マスクを保存
    import numpy as np
    np.save("segmentation_mask.npy", mask)
```

### 3. バッチ推論

```python
# JSONファイルを準備
# inference_samples.json:
[
    {
        "id": "sample1",
        "image": "/path/to/image1.jpg",
        "instruction": "Please segment the dog"
    },
    {
        "id": "sample2",
        "image": "/path/to/image2.jpg",
        "instruction": "Where is the red car?"
    }
]

# バッチ推論の実行
./scripts/run_batch_inference.sh inference_samples.json
```

## 学習の実行

### 1. データセットの準備

```bash
# RefCOCOデータセットの例
mkdir -p data/refcoco
# データセットをダウンロードして配置
```

### 2. 学習設定の作成

```json
// configs/train_config.json
{
    "output_dir": "output/my_experiment",
    "num_train_epochs": 10,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 1e-4,
    "warmup_steps": 500,
    "logging_steps": 50,
    "save_steps": 1000,
    "eval_steps": 500,
    "use_lora": true,
    "lora_r": 16,
    "lora_alpha": 32,
    "train_datasets": ["refcoco"],
    "dataset_weights": [1.0]
}
```

### 3. 学習の開始

```bash
./scripts/train_lisa.sh

# または直接実行
python train.py --config configs/train_config.json
```

## よくある使用例

### 例1: 物体のセグメンテーション

```python
# 単一オブジェクト
instruction = "Please segment the apple on the table"

# 複数オブジェクト
instruction = "Can you highlight all the cars in this street scene?"

# 属性付きセグメンテーション
instruction = "Show me the person wearing a red shirt"
```

### 例2: インタラクティブな使用

```python
import gradio as gr
from inference import LISAInference

# Gradioアプリの作成
pipeline = LISAInference("path/to/model")

def segment_image(image, instruction):
    text, mask = pipeline.generate_response(image, instruction)
    # マスクのオーバーレイ作成
    if mask is not None:
        # 可視化コード
        pass
    return text, visualization

interface = gr.Interface(
    fn=segment_image,
    inputs=[
        gr.Image(type="pil"),
        gr.Textbox(placeholder="Enter instruction...")
    ],
    outputs=[
        gr.Textbox(label="Response"),
        gr.Image(label="Segmentation")
    ]
)

interface.launch()
```

### 例3: カスタムデータセットでの学習

```python
from src.data.datasets import CustomDataset
from train import train_model

# カスタムデータセットの作成
class MyDataset(CustomDataset):
    def __getitem__(self, idx):
        # データの読み込みと前処理
        return {
            'pixel_values': image_tensor,
            'input_ids': input_ids,
            'labels': labels,
            'mask_labels': mask
        }

# 学習の実行
train_model(
    model=model,
    train_dataset=MyDataset(),
    output_dir="./my_model"
)
```

## トラブルシューティング

### CUDA Out of Memory

```python
# バッチサイズを減らす
config.per_device_train_batch_size = 1
config.gradient_accumulation_steps = 16

# 画像サイズを減らす
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    min_pixels=256*256,
    max_pixels=512*512
)
```

### 推論が遅い

```python
# Flash Attentionを有効化
config.use_flash_attention = True

# バッチ推論を使用
batch_size = 8  # GPUメモリに応じて調整
```

### セグメンテーションが生成されない

```python
# プロンプトを明確にする
instruction = "Please segment the <specific object> in this image"

# 温度を調整
temperature = 0.3  # より決定的な出力
```

## 次のステップ

1. [APIリファレンス](api_reference.md)で詳細な使い方を確認
2. [技術詳細](technical_details.md)でアーキテクチャを理解
3. [実装サマリー](implementation_summary.md)で全体像を把握

## サポート

問題が発生した場合：
1. GitHubのIssuesで報告
2. ログファイルを確認（`output/logs/`）
3. 軽量テストを実行して環境を確認

```bash
python tests/lightweight_integration_test.py
```