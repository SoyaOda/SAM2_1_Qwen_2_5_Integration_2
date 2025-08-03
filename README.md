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

## Project Structure

```
SAM2_1_Qwen_2_5_Integration_2/
├── src/
│   ├── config.py           # Model configuration
│   ├── models/
│   │   ├── adapters.py     # ImageFeatureAdapter, TextPromptProjector
│   │   ├── lisa_model.py   # Main LISA改 model
│   │   └── losses.py       # Loss functions
│   └── utils/
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
5. **SAM Mask Decoder**: Generates segmentation masks from prompts

### Information Flow

```
Image → Qwen ViT → Image Adapter → SAM Features
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