#!/usr/bin/env python3
"""
SAM LoRA機能のテスト
"""

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

print("=" * 80)
print("SAM LoRA機能テスト")
print("=" * 80)

# 1. 設定（SAM LoRAを有効化）
from src.config import LISAConfig

config = LISAConfig(
    qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    sam_model_name="./checkpoints/sam2.1_hiera_large.pt",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype="auto",
    freeze_qwen=True,
    freeze_sam=True,
    train_seg_token=True,
    sam_lora_r=4,  # SAM LoRAを有効化
    sam_lora_alpha=16,
    sam_lora_dropout=0.1
)

print(f"\n設定:")
print(f"  SAM LoRA r: {config.sam_lora_r}")
print(f"  SAM LoRA alpha: {config.sam_lora_alpha}")
print(f"  SAM LoRA dropout: {config.sam_lora_dropout}")

# 2. モデルのセットアップ
print("\nモデルのセットアップ...")
from src.models.lisa_model import LISA_Model
from src.utils.tokenizer_utils import prepare_tokenizer_for_lisa
from transformers import AutoProcessor

tokenizer = prepare_tokenizer_for_lisa(config.qwen_model_name, config.seg_token)
processor = AutoProcessor.from_pretrained(config.qwen_model_name)
if config.seg_token not in processor.tokenizer.get_vocab():
    processor.tokenizer.add_special_tokens({"additional_special_tokens": [config.seg_token]})

model = LISA_Model(config)
model.set_tokenizer(tokenizer)

# 3. SAM LoRAの適用
print("\nSAM LoRAの適用...")
model.add_sam_lora(
    lora_r=config.sam_lora_r,
    lora_alpha=config.sam_lora_alpha,
    lora_dropout=config.sam_lora_dropout
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 4. パラメータの確認
print("\nパラメータ統計:")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# 各コンポーネントの学習可能パラメータ数
qwen_trainable = sum(p.numel() for n, p in model.qwen.named_parameters() if p.requires_grad)
sam_trainable = sum(p.numel() for n, p in model.sam_mask_decoder.named_parameters() if p.requires_grad)
adapter_trainable = sum(p.numel() for n, p in model.image_adapter.named_parameters() if p.requires_grad)
text_proj_trainable = sum(p.numel() for n, p in model.text_prompt_proj.named_parameters() if p.requires_grad)
high_res_trainable = sum(p.numel() for n, p in model.high_res_generator.named_parameters() if p.requires_grad)

print(f"  総パラメータ数: {total_params:,}")
print(f"  学習可能パラメータ数: {trainable_params:,}")
print(f"  学習可能率: {100 * trainable_params / total_params:.2f}%")
print(f"\n  コンポーネント別学習可能パラメータ:")
print(f"    Qwen: {qwen_trainable:,}")
print(f"    SAM MaskDecoder: {sam_trainable:,}")
print(f"    Image Adapter: {adapter_trainable:,}")
print(f"    Text Projector: {text_proj_trainable:,}")
print(f"    High-Res Generator: {high_res_trainable:,}")

# 5. 簡単な推論テスト
print("\n推論テスト...")
from PIL import Image
import numpy as np

# テスト画像
image = Image.new('RGB', (224, 224), color=(128, 128, 128))
pixels = np.array(image)
pixels[50:150, 50:150] = [255, 0, 0]  # 赤い四角
image = Image.fromarray(pixels.astype(np.uint8))

text = "Segment the red square. <SEG>"

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": text}
    ]
}]

processed = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
)

input_ids = processed['input_ids'].to(device)
pixel_values = processed['pixel_values'].to(device)
attention_mask = processed['attention_mask'].to(device)
image_grid_thw = processed.get('image_grid_thw', torch.tensor([[1, 16, 16]], device=device)).to(device)

# グラウンドトゥルースマスク
gt_mask = torch.zeros(1, 224, 224, device=device)
gt_mask[0, 50:150, 50:150] = 1.0

labels = input_ids.clone()
labels[:, :10] = -100

# Forward pass
model.eval()
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        labels=labels,
        mask_labels=[gt_mask],
        image_grid_thw=image_grid_thw
    )
    
    print("  ✅ Forward pass成功")
    
    # 損失計算
    if outputs.mask_logits and outputs.mask_logits[0]:
        import torch.nn as nn
        pred_mask = outputs.mask_logits[0][0]
        
        # サイズ調整
        if pred_mask.shape[-2:] != (224, 224):
            pred_mask = nn.functional.interpolate(
                pred_mask.unsqueeze(0) if pred_mask.dim() == 2 else pred_mask,
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            )
        
        while pred_mask.dim() > 2:
            pred_mask = pred_mask.squeeze(0)
        
        # BCE損失
        gt_mask_2d = gt_mask.squeeze(0) if gt_mask.dim() > 2 else gt_mask
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            pred_mask, gt_mask_2d.float()
        )
        
        # Dice損失
        pred_sigmoid = torch.sigmoid(pred_mask)
        intersection = (pred_sigmoid * gt_mask_2d).sum()
        dice = 2 * intersection / (pred_sigmoid.sum() + gt_mask_2d.sum() + 1e-8)
        dice_loss = 1 - dice
        
        seg_loss = bce_loss + dice_loss
        
        print(f"\n  損失値:")
        print(f"    BCE Loss: {bce_loss.item():.4f}")
        print(f"    Dice Loss: {dice_loss.item():.4f}")
        print(f"    SEG Loss: {seg_loss.item():.4f}")

print("\n✅ SAM LoRA機能は正常に動作しています！")