# LISA改 (LISA-Kai) - Integration of Qwen2.5-VL and SAM2.1

This project implements LISA改, an advanced multimodal model that integrates Qwen2.5-VL-3B for vision-language understanding with SAM2.1 for high-precision segmentation.

## Overview

LISA改 builds upon the LISA (Language Instructed Segmentation Assistant) architecture, combining:
- **Qwen2.5-VL-3B**: A powerful vision-language model for deep image understanding
- **SAM2.1**: State-of-the-art segmentation model from Meta
- **Learned Adapters**: Efficient connection between VLM and segmentation modules

The model can understand complex language instructions and generate precise segmentation masks for specified objects in images.

## Features

- 🎯 **Reasoning Segmentation**: Segment objects based on complex language instructions
- 🔄 **Unified Architecture**: Single model for both language and segmentation tasks
- 🚀 **Efficient Training**: LoRA and adapter-based fine-tuning
- 💡 **Special Token <SEG>**: Seamless integration of segmentation in language generation

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd SAM2_1_Qwen_2_5_Integration_2
```

2. Install dependencies:
```bash
python setup.py
```

This will install all required packages including:
- PyTorch 2.5.1+
- Transformers 4.49.0+
- SAM2 1.1.0+
- PEFT for LoRA

## Training and Inference Scripts

### Training Script: `minimal_train.py`

The main training script supports multi-dataset training with flexible configuration options.

#### Basic Usage

```bash
# Simple training run
python minimal_train.py --num_epochs 3 --batch_size 4

# Training with inference evaluation
python minimal_train.py \
    --num_epochs 3 \
    --run_inference_eval \
    --inference_eval_samples 3 \
    --save_interval 500
```

#### Key Features

- **Multi-dataset Support**: Trains on ADE20K, COCO-Stuff, RefCOCO, VQA, and ReasonSeg simultaneously
- **Flexible Learning Configuration**: Individual control over each component's learning/freezing
- **Checkpoint Management**: Automatic saving, best model tracking, and training resumption
- **Inference Evaluation**: Automatic inference on benchmark images during checkpoint saves
- **Visualization**: Real-time loss plots and segmentation visualizations

#### Important Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--num_epochs` | Number of training epochs | 3 |
| `--batch_size` | Training batch size | 4 |
| `--lora_r` | LoRA rank for Qwen | 8 |
| `--sam_lora_r` | LoRA rank for SAM | 8 |
| `--adapter_lr` | Learning rate for adapters | 1e-3 |
| `--lora_lr` | Learning rate for LoRA | 1e-4 |
| `--run_inference_eval` | Enable inference evaluation | False |
| `--save_interval` | Checkpoint save interval (steps) | 500 |
| `--wandb` | Enable Weights & Biases logging | False |

#### Advanced Usage

```bash
# Resume training from checkpoint
python minimal_train.py \
    --checkpoint_dir outputs/minimal_train_20250814_100000/checkpoints/step_5000 \
    --resume_training \
    --num_epochs 5

# Large-scale training with all features
python minimal_train.py \
    --num_epochs 20 \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --samples_per_epoch 50000 \
    --lora_r 32 \
    --sam_lora_r 16 \
    --run_inference_eval \
    --wandb \
    --wandb_project "LISA_KAI"
```

### Inference Script: `test_inference_v2.py`

The inference script loads trained checkpoints and performs segmentation on test images.

#### Basic Usage

```bash
# Use latest checkpoint
python test_inference_v2.py

# Specify checkpoint directory
python test_inference_v2.py \
    --checkpoint-dir outputs/minimal_train_20250814_100000/checkpoints/best
```

#### Features

- **Automatic Checkpoint Loading**: Supports both new and legacy checkpoint formats
- **Benchmark Evaluation**: Tests on standard SAM benchmark images (truck, groceries, dog)
- **Visualization**: Generates comparison images showing original, mask, and overlay
- **Statistics**: Computes and saves mask statistics (mean, min, max, positive pixels)

#### Output Structure

```
outputs/inference_results_v2/
├── truck_result.png         # Visualization for truck image
├── groceries_result.png     # Visualization for groceries image
├── dog_result.png           # Visualization for dog image
└── inference_results.json   # Numerical results and statistics
```

### Training Output Structure

When training with `--run_inference_eval`, the following structure is created:

```
outputs/minimal_train_YYYYMMDD_HHMMSS/
├── checkpoints/
│   ├── step_500/
│   │   ├── inference_results/    # Inference evaluation at step 500
│   │   │   ├── truck_step500.png
│   │   │   ├── groceries_step500.png
│   │   │   └── evaluation_results.json
│   │   └── [checkpoint files]
│   └── best/
│       └── [best model checkpoint]
├── visualizations/               # Training sample visualizations
├── inference_cache/              # Cached benchmark images
└── loss_history.png             # Training loss plot
```

## Model Configuration

The model behavior is controlled by `src/config.py`:

```python
@dataclass
class LISAConfig:
    # Model selection
    qwen_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    sam_model_name: str = "facebook/sam2.1-hiera-large"
    
    # LoRA configuration
    lora_r: int = 8              # Qwen LoRA rank
    sam_lora_r: int = 8          # SAM LoRA rank
    
    # Training control (what to freeze)
    freeze_qwen_lora: bool = False       # False = train Qwen LoRA
    freeze_seg_token: bool = False       # False = train SEG token
    freeze_sam_mask_decoder_base: bool = True  # True = use SAM LoRA
    
    # Feature Fusion configuration
    fusion_type: str = "sigma_add"       # "sigma_add" or "cross_attention"
    fusion_num_heads: int = 8            # Attention heads for Cross-Attention
    fusion_dropout: float = 0.1          # Dropout rate for Cross-Attention
    fusion_use_gate: bool = True         # Enable gating mechanism for Cross-Attention
    
    # Architecture choices
    use_token_fpn: bool = True   # Use Token-FPN for multi-scale features
```

## Best Practices

### 1. Quick Experimentation
```bash
# Small-scale test to verify setup
python minimal_train.py \
    --num_epochs 1 \
    --samples_per_epoch 100 \
    --batch_size 2 \
    --run_inference_eval
```

### 2. Hyperparameter Tuning
```bash
# Test different LoRA ranks
for r in 4 8 16; do
    python minimal_train.py \
        --lora_r $r \
        --sam_lora_r $r \
        --num_epochs 3 \
        --wandb_run_name "lora_r_${r}"
done
```

### 3. Production Training
```bash
# Full training with monitoring
python minimal_train.py \
    --num_epochs 20 \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --samples_per_epoch 50000 \
    --lora_r 32 \
    --sam_lora_r 16 \
    --adapter_lr 5e-4 \
    --lora_lr 5e-5 \
    --run_inference_eval \
    --save_interval 1000 \
    --wandb
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `--batch_size 1`
- Use gradient accumulation: `--gradient_accumulation_steps 8`
- Reduce LoRA rank: `--lora_r 4 --sam_lora_r 4`

### Checkpoint Loading Issues
The inference script automatically handles both new and legacy checkpoint formats. If issues persist:
```python
from src.utils.checkpoint_io import load_lisa_checkpoint
result = load_lisa_checkpoint("path/to/checkpoint", device='cuda')
```

### Dataset Loading Errors
- Specify data directory: `--data_dir /path/to/LISA-dataset/data/dataset`
- Clear cache: `rm -rf src/data/__pycache__`

## Project Structure

```
SAM2_1_Qwen_2_5_Integration_2/
├── src/
│   ├── config.py           # Model configuration
│   ├── models/
│   │   ├── adapters.py     # ImageFeatureAdapter, TextPromptProjector
│   │   ├── lisa_model.py   # Main LISA改 model
│   │   └── losses.py       # Loss functions
│   ├── data/
│   │   ├── hybrid_dataset.py  # Multi-dataset loader
│   │   └── dataset_cache.py   # Efficient dataset caching
│   └── utils/
│       ├── checkpoint_io.py   # Checkpoint save/load utilities
│       └── model_utils.py     # Model initialization helpers
│       ├── tokenizer_utils.py  # Tokenizer utilities
│       └── lora_utils.py       # LoRA configuration
├── tests/
│   ├── test_adapters.py    # Adapter module tests
│   └── test_lisa_integration.py  # Integration tests
└── setup.py                # Setup script
```

## Model Architecture

### Key Components

1. **Vision Encoder**: Qwen2.5-VL's ViT processes input images
2. **Language Decoder**: Qwen's LLM handles text generation and reasoning
3. **Image Feature Adapter**: Transforms Qwen vision features for SAM (D_v → 256)
4. **Text Prompt Projector**: Projects <SEG> hidden states to SAM prompts (D_l → 256)
5. **Feature Fusion Module**: Cross-Attention or Sigma-Add fusion between Qwen and SAM features
6. **SAM Mask Decoder**: Generates segmentation masks from prompts

### Feature Fusion Methods

LISA改 supports two fusion methods for combining Qwen and SAM visual features:

#### 1. Cross-Attention Fusion (Recommended)
- **Architecture**: Multi-head attention with SAM features as Query, Qwen features as Key/Value
- **Parameters**: 8 attention heads, 256-dim embeddings, dropout=0.1
- **Gating Mechanism**: Dynamic mixing between fused and original features
- **Advantages**: Superior accuracy for reasoning segmentation, spatially-aware fusion
- **Use Case**: Best for accuracy-critical applications with sufficient GPU resources

#### 2. Sigma-Add Fusion (Lightweight Alternative)
- **Architecture**: Simple weighted addition with learnable scalar β
- **Parameters**: Single trainable parameter (β)
- **Advantages**: Minimal computation overhead, extremely stable
- **Use Case**: Resource-constrained environments or rapid prototyping

### Fusion Configuration

Configure fusion in `src/config.py`:

```python
# Fusion configuration
fusion_type: str = "sigma_add"  # Options: "sigma_add", "cross_attention"
fusion_num_heads: int = 8       # Number of attention heads for Cross-Attention
fusion_dropout: float = 0.1     # Dropout rate for Cross-Attention
fusion_use_gate: bool = True    # Enable gating mechanism for Cross-Attention

# Recommended settings based on GPU resources:
# - GPU 16GB+: fusion_type="cross_attention" (best accuracy)
# - GPU 8-16GB: fusion_type="cross_attention" with smaller batch_size
# - GPU <8GB: fusion_type="sigma_add" (lightest computation)
```

### Information Flow

```
Image → Qwen ViT → Image Adapter → Qwen Features (16×16)
                                          ↓
                                    Upsample to 64×64
                                          ↓
Image → SAM Encoder → SAM Features (64×64)
                            ↓
                    [Cross-Attention Fusion]
                            ↓
Text → Qwen LLM → <SEG> → Text Projector → SAM Prompt
                                          ↓
                                    SAM Mask Decoder → Segmentation Mask
```

## Usage

### Basic Example

```python
from src.config import LISAConfig
from src.models import LISA_Model
from src.utils import prepare_tokenizer_for_lisa

# Initialize configuration
config = LISAConfig()

# Create model
model = LISA_Model(config)

# Prepare tokenizer with <SEG> token
tokenizer = prepare_tokenizer_for_lisa()
model.set_tokenizer(tokenizer)

# Configure for training
from src.utils import configure_lisa_for_training
training_info = configure_lisa_for_training(model, config, tokenizer)
```

### Training Configuration

The model uses efficient training strategies:
- Qwen and SAM base models are frozen by default
- LoRA applied to cross-attention layers
- Only adapters and <SEG> embedding are fully trainable

### Loss Functions

Combined loss for multitask learning:
- **Language Loss**: Cross-entropy for text generation
- **Segmentation Loss**: BCE + Dice loss for masks

## Testing

Run the test suite:

```bash
# Test adapter modules
python tests/test_adapters.py

# Test full integration (requires models)
python tests/test_lisa_integration.py

# Test Cross-Attention fusion
python test_cross_attention_fusion.py

# Advanced fusion tests with visualization
python test_fusion_advanced.py
```

### Testing Fusion Methods

Compare different fusion configurations:

```bash
# Test Cross-Attention with default settings
python test_cross_attention_fusion.py

# Compare Cross-Attention vs Sigma-Add performance
python test_fusion_advanced.py

# The tests will generate:
# - attention_vis.png: Attention weight heatmap
# - mask_output_*.png: Segmentation results for each fusion type
# - convergence_comparison.png: Training convergence comparison
```

## Requirements

- GPU: NVIDIA GPU with 12GB+ VRAM (for 3B model)
- Memory: 20GB+ RAM for model loading
- Disk: ~15GB for model weights

## Future Work

- Implement streaming generation with mask output
- Add training scripts for downstream tasks
- Support for video segmentation using SAM2's temporal features
- Integration with FoodLMM for specialized applications

## Citation

This work builds upon:
- LISA: Reasoning Segmentation via Large Language Model
- Qwen2.5-VL Technical Report
- SAM2: Segment Anything Model 2

## License

This project follows the licenses of its component models:
- Qwen2.5-VL: Apache 2.0
- SAM2: Apache 2.0