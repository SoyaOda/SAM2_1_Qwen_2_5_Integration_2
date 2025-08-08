#!/usr/bin/env python3
"""
デバッグ用の最小限のテスト
"""

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

print("=" * 80)
print("クイックデバッグテスト")
print("=" * 80)

# 1. 設定の確認
print("\n1. 設定の初期化")
from src.config import LISAConfig

config = LISAConfig(
    qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    sam_model_name="./checkpoints/sam2.1_hiera_large.pt",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype="auto",
    freeze_qwen=True,
    freeze_sam=True,
    train_seg_token=True,
)
print(f"Device: {config.device_map}")
print(f"SAM LoRA r: {config.sam_lora_r}")

# 2. トークナイザーの準備
print("\n2. トークナイザーの準備")
from src.utils.tokenizer_utils import prepare_tokenizer_for_lisa
tokenizer = prepare_tokenizer_for_lisa(config.qwen_model_name, config.seg_token)
print(f"Tokenizer vocab size: {len(tokenizer)}")

# 3. プロセッサーの準備
print("\n3. プロセッサーの準備")
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(config.qwen_model_name)
if config.seg_token not in processor.tokenizer.get_vocab():
    processor.tokenizer.add_special_tokens({"additional_special_tokens": [config.seg_token]})
    print("ProcessorにSEGトークン追加")

# 4. モデルのロード（ステップごとに確認）
print("\n4. モデルのロード開始")
from src.models.lisa_model import LISA_Model

print("  4.1. LISA_Modelインスタンス作成...")
model = LISA_Model(config)
print("  4.2. LISA_Modelインスタンス作成完了")

print("  4.3. set_tokenizer呼び出し...")
model.set_tokenizer(tokenizer)
print("  4.4. set_tokenizer完了")

print("  4.5. デバイスへの移動...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print("  4.6. デバイスへの移動完了")

# 5. SAM LoRAのテスト（オプション）
print("\n5. SAM LoRAテスト")
if config.sam_lora_r > 0:
    print(f"  SAM LoRAを適用 (r={config.sam_lora_r})")
    model.add_sam_lora(
        lora_r=config.sam_lora_r,
        lora_alpha=config.sam_lora_alpha,
        lora_dropout=config.sam_lora_dropout
    )
else:
    print("  SAM LoRA無効 (r=0)")

# 6. 簡単な推論テスト
print("\n6. 簡単な推論テスト")
from PIL import Image
import numpy as np

# ダミー画像作成
image = Image.new('RGB', (224, 224), color=(128, 128, 128))
text = "Test image with <SEG>"

# メッセージ形式
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": text}
    ]
}]

print("  6.1. apply_chat_template...")
processed = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
)
print("  6.2. apply_chat_template完了")

# 入力を準備
input_ids = processed['input_ids'].to(device)
pixel_values = processed['pixel_values'].to(device)
attention_mask = processed['attention_mask'].to(device)

print(f"  Input shapes:")
print(f"    input_ids: {input_ids.shape}")
print(f"    pixel_values: {pixel_values.shape}")
print(f"    attention_mask: {attention_mask.shape}")

# SEGトークンの確認
seg_token_id = tokenizer.convert_tokens_to_ids(config.seg_token)
seg_positions = (input_ids == seg_token_id).nonzero(as_tuple=True)
if len(seg_positions[1]) > 0:
    print(f"  ✅ SEGトークン検出（位置: {seg_positions[1].tolist()}）")
else:
    print(f"  ❌ SEGトークン未検出")

print("\n7. Forward pass...")
model.eval()
with torch.no_grad():
    # image_grid_thw
    image_grid_thw = processed.get('image_grid_thw')
    if image_grid_thw is None:
        image_grid_thw = torch.tensor([[1, 16, 16]], dtype=torch.long, device=device)
    else:
        image_grid_thw = image_grid_thw.to(device)
    
    print(f"  image_grid_thw: {image_grid_thw}")
    
    # ラベル（ダミー）
    labels = input_ids.clone()
    labels[:, :10] = -100
    
    # Forward
    print("  実行中...")
    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        labels=labels,
        mask_labels=None,
        image_grid_thw=image_grid_thw
    )
    print("  Forward pass完了！")
    
    if outputs.logits is not None:
        print(f"  Logits shape: {outputs.logits.shape}")
    if outputs.mask_logits:
        print(f"  Mask logits数: {len(outputs.mask_logits)}")

print("\n✅ テスト完了！モデルは正常に動作しています。")