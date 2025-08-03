# LISA改 APIリファレンス

## モデルクラス

### LISA_Model

メインのモデルクラス。Qwen2.5-VLとSAM2.1を統合します。

```python
class LISA_Model(nn.Module):
    def __init__(
        self,
        config: LISAConfig,
        qwen_model: Optional[Qwen2_5_VLForConditionalGeneration] = None,
        sam_predictor: Optional[SAM2ImagePredictor] = None,
        tokenizer: Optional[AutoTokenizer] = None,
    )
```

#### パラメータ
- `config`: モデル設定（LISAConfig）
- `qwen_model`: 事前ロード済みのQwenモデル（オプション）
- `sam_predictor`: 事前ロード済みのSAM predictor（オプション）
- `tokenizer`: 事前ロード済みのtokenizer（オプション）

#### メソッド

##### forward()
```python
def forward(
    self,
    input_ids: torch.Tensor,
    pixel_values: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    mask_labels: Optional[List[torch.Tensor]] = None,
    return_dict: bool = True,
) -> Union[LISAModelOutput, Tuple]
```

標準的なforward pass。学習時に使用。

##### generate_with_masks()
```python
def generate_with_masks(
    self,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    do_sample: bool = True,
    top_p: float = 0.9,
    **kwargs
) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]
```

テキスト生成とマスク生成を同時に実行。

**戻り値**:
```python
{
    'generated_ids': torch.Tensor,  # 生成されたトークンID
    'generated_text': List[str],    # デコードされたテキスト
    'masks': List[List[torch.Tensor]],  # 生成されたマスク
    'seg_positions': List[List[int]]  # SEGトークンの位置
}
```

##### set_tokenizer()
```python
def set_tokenizer(self, tokenizer: AutoTokenizer, seg_token: str = "<SEG>")
```

トークナイザーを設定し、SEGトークンを追加。

##### save_pretrained()
```python
def save_pretrained(self, save_directory: str)
```

モデルとアダプターを保存。

##### from_pretrained()
```python
@classmethod
def from_pretrained(cls, load_directory: str, **kwargs)
```

保存されたモデルをロード。

## アダプタークラス

### ImageFeatureAdapter

```python
class ImageFeatureAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256)
```

Qwenのビジョン特徴量をSAM形式に変換。

#### パラメータ
- `in_dim`: 入力次元（Qwenのビジョン特徴量次元）
- `out_dim`: 出力次元（SAMの画像埋め込み次元）

### TextPromptProjector

```python
class TextPromptProjector(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int = 256, 
        use_mlp: bool = False
    )
```

テキストの隠れ状態をSAMプロンプトに変換。

#### パラメータ
- `in_dim`: 入力次元（Qwenの隠れ状態次元）
- `out_dim`: 出力次元（SAMのプロンプト次元）
- `use_mlp`: MLPを使用するか（False: 線形変換のみ）

## 設定クラス

### LISAConfig

```python
@dataclass
class LISAConfig:
    # Model paths
    qwen_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    sam_model_name: str = "facebook/sam2-hiera-large"
    
    # Model dimensions
    qwen_hidden_size: Optional[int] = None  # 自動検出
    qwen_vision_hidden_size: Optional[int] = None  # 自動検出
    sam_image_embedding_dim: int = 256
    sam_prompt_dim: int = 256
    
    # Adapter settings
    image_adapter_type: str = "linear"
    text_prompt_out_dim: int = 256
    
    # Training settings
    freeze_qwen: bool = True
    freeze_sam: bool = True
    train_seg_token: bool = True
    
    # Special tokens
    seg_token: str = "<SEG>"
    
    # Device settings
    device_map: str = "auto"
    torch_dtype: torch.dtype = torch.float16
    use_flash_attention: bool = False
```

## データセットクラス

### RefCOCODataset

```python
class RefCOCODataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        dataset_name: str = "refcoco",
        processor: Any = None,
        tokenizer: Any = None,
        max_length: int = 512,
        seg_token: str = "<SEG>"
    )
```

Referring segmentationデータセット。

### VQADataset

```python
class VQADataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        processor: Any = None,
        tokenizer: Any = None,
        max_length: int = 512
    )
```

Visual Question Answeringデータセット。

### ImageCaptionDataset

```python
class ImageCaptionDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        processor: Any = None,
        tokenizer: Any = None,
        max_length: int = 512,
        use_seg_token: bool = False,
        seg_token: str = "<SEG>"
    )
```

画像キャプションデータセット。

## コレータークラス

### MultiModalDataCollator

```python
class MultiModalDataCollator:
    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 512,
        padding: str = "longest",
        return_tensors: str = "pt"
    )
```

マルチモーダルデータのバッチ処理。

## 推論クラス

### LISAInference

```python
class LISAInference:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16
    )
```

推論パイプライン。

#### メソッド

##### generate_response()
```python
def generate_response(
    self,
    image: Union[str, Path, Image.Image],
    instruction: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    do_sample: bool = True
) -> Tuple[str, Optional[np.ndarray]]
```

単一画像に対する推論を実行。

**戻り値**:
- `text_response`: 生成されたテキスト
- `mask`: セグメンテーションマスク（なければNone）

##### visualize_result()
```python
def visualize_result(
    self,
    image: Union[str, Path, Image.Image],
    text_response: str,
    mask: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
)
```

推論結果を可視化。

### BatchInference

```python
class BatchInference:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        batch_size: int = 4,
        num_workers: int = 4
    )
```

バッチ推論パイプライン。

#### メソッド

##### process_batch()
```python
def process_batch(
    self,
    input_file: str,
    output_dir: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7
) -> List[Dict]
```

JSONファイルから複数の画像を処理。

## ユーティリティ関数

### prepare_tokenizer_for_lisa()

```python
def prepare_tokenizer_for_lisa(
    model_name: str,
    seg_token: str = "<SEG>",
    padding_side: str = "right"
) -> AutoTokenizer
```

LISA用にトークナイザーを準備。

### create_optimizer()

```python
def create_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    optimizer_type: str = "adamw"
) -> torch.optim.Optimizer
```

最適化器を作成。

### compute_metrics()

```python
def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    metric_type: str = "accuracy"
) -> Dict[str, float]
```

評価メトリクスを計算。

## 使用例

### 基本的な推論

```python
from src.models import LISA_Model
from src.config import LISAConfig
from src.utils import prepare_tokenizer_for_lisa

# モデルの初期化
config = LISAConfig()
model = LISA_Model(config)
tokenizer = prepare_tokenizer_for_lisa(config.qwen_model_name)
model.set_tokenizer(tokenizer)

# 推論
with torch.no_grad():
    outputs = model.generate_with_masks(
        input_ids=input_ids,
        pixel_values=pixel_values,
        max_new_tokens=100
    )

print(f"Generated text: {outputs['generated_text'][0]}")
print(f"Number of masks: {len(outputs['masks'][0])}")
```

### 学習ループ

```python
from train import train_model

# 学習の実行
train_model(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    output_dir="./output",
    num_epochs=10,
    batch_size=4,
    learning_rate=1e-4
)
```

### バッチ推論

```python
from batch_inference import BatchInference

# バッチ推論の実行
inferencer = BatchInference(
    model_path="./output/final_model",
    batch_size=8
)

results = inferencer.process_batch(
    input_file="./data/test_samples.json",
    output_dir="./results"
)
```