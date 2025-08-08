#!/usr/bin/env python3
"""
シンプルなオーバーフィッティングテスト
デバッグ出力付きで問題を特定
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
from PIL import Image
import logging

sys.path.append(str(Path(__file__).parent))

from src.config import LISAConfig
from src.models.lisa_model import LISA_Model
from src.utils.tokenizer_utils import prepare_tokenizer_for_lisa
from transformers import AutoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_simple_overfitting():
    """シンプルなオーバーフィッティングテスト"""
    
    print("=" * 80)
    print("シンプルなオーバーフィッティングテスト")
    print("=" * 80)
    
    # 設定
    config = LISAConfig(
        qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        sam_model_name="./checkpoints/sam2.1_hiera_large.pt",
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype="auto",
        freeze_qwen=False,
        freeze_sam=False,
        train_seg_token=True,
        segmentation_loss_weight=10.0
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # モデル準備
    print("\n1. モデルの準備")
    tokenizer = prepare_tokenizer_for_lisa(config.qwen_model_name, config.seg_token)
    processor = AutoProcessor.from_pretrained(config.qwen_model_name)
    
    # ProcessorにもSEGトークン追加
    if config.seg_token not in processor.tokenizer.get_vocab():
        processor.tokenizer.add_special_tokens({"additional_special_tokens": [config.seg_token]})
        print(f"ProcessorにSEGトークン追加完了")
    
    seg_token_id = tokenizer.convert_tokens_to_ids(config.seg_token)
    print(f"SEGトークンID: {seg_token_id}")
    
    model = LISA_Model(config)
    model.set_tokenizer(tokenizer)
    model = model.to(device)
    model.train()
    
    # オプティマイザ
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # 固定データ作成
    print("\n2. 固定データの作成")
    
    # 赤い四角形のある画像
    image = Image.new('RGB', (448, 448), color=(128, 128, 128))
    pixels = np.array(image)
    pixels[150:300, 150:300] = [255, 0, 0]  # 赤い四角
    image = Image.fromarray(pixels.astype(np.uint8))
    
    # グラウンドトゥルースマスク
    gt_mask = torch.zeros(1, 448, 448, device=device)
    gt_mask[0, 150:300, 150:300] = 1.0
    
    # テキストプロンプト
    text = "Segment the red square in this image. <SEG>"
    
    # メッセージ形式で処理
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text}
            ]
        }
    ]
    
    # apply_chat_template
    processed = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )
    
    input_ids = processed['input_ids'].to(device)
    attention_mask = processed['attention_mask'].to(device)
    pixel_values = processed['pixel_values'].to(device)
    
    # image_grid_thw
    if 'image_grid_thw' in processed:
        image_grid_thw = processed['image_grid_thw'].to(device)
    else:
        image_grid_thw = torch.tensor([[1, 32, 32]], dtype=torch.long, device=device)
    
    # ラベル作成
    labels = input_ids.clone()
    labels[:, :10] = -100  # 最初の部分をマスク
    
    # SEGトークンの位置を確認
    seg_positions = (input_ids == seg_token_id).nonzero(as_tuple=True)
    if len(seg_positions[1]) > 0:
        print(f"✅ SEGトークンが入力に含まれています（位置: {seg_positions[1].tolist()}）")
    else:
        print(f"❌ SEGトークンが入力に含まれていません！")
        return
    
    # 訓練ループ
    print("\n3. 訓練開始（100ステップ）")
    
    seg_loss_history = []
    
    for step in range(100):
        # Forward
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            labels=labels,
            mask_labels=[gt_mask],
            image_grid_thw=image_grid_thw
        )
        
        # 損失計算
        vocab_size = outputs.logits.size(-1)
        lm_loss = nn.functional.cross_entropy(
            outputs.logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=-100
        )
        
        seg_loss = torch.tensor(2.0, device=device)  # デフォルト値
        
        if outputs.mask_logits and len(outputs.mask_logits) > 0:
            if outputs.mask_logits[0] is not None and len(outputs.mask_logits[0]) > 0:
                pred_mask = outputs.mask_logits[0][0]  # 最初のマスク
                
                # デバッグ出力（最初のステップのみ）
                if step == 0:
                    print(f"  pred_mask shape: {pred_mask.shape}")
                    print(f"  gt_mask shape: {gt_mask.shape}")
                
                # サイズ調整
                if pred_mask.shape[-2:] != gt_mask.shape[-2:]:
                    pred_mask = nn.functional.interpolate(
                        pred_mask.unsqueeze(0) if pred_mask.dim() == 2 else pred_mask,
                        size=gt_mask.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                # 次元を揃える（両方を2Dにする）
                while pred_mask.dim() > 2:
                    pred_mask = pred_mask.squeeze(0)
                while gt_mask.dim() > 2:
                    gt_mask_squeezed = gt_mask.squeeze(0)
                else:
                    gt_mask_squeezed = gt_mask
                
                # BCE損失
                bce_loss = nn.functional.binary_cross_entropy_with_logits(
                    pred_mask,
                    gt_mask_squeezed.float()
                )
                
                # Dice損失
                pred_sigmoid = torch.sigmoid(pred_mask)
                intersection = (pred_sigmoid * gt_mask_squeezed).sum()
                dice = 2 * intersection / (pred_sigmoid.sum() + gt_mask_squeezed.sum() + 1e-8)
                dice_loss = 1 - dice
                
                seg_loss = bce_loss + dice_loss
        
        total_loss = lm_loss + config.segmentation_loss_weight * seg_loss
        
        seg_loss_history.append(seg_loss.item())
        
        if step % 10 == 0:
            print(f"Step {step:3d}: SEG Loss = {seg_loss.item():.4f}, LM Loss = {lm_loss.item():.4f}")
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        
        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
    
    # 結果の分析
    print("\n" + "=" * 80)
    print("診断結果")
    print("=" * 80)
    
    initial_seg = seg_loss_history[0] if seg_loss_history else 2.0
    final_seg = seg_loss_history[-1] if seg_loss_history else 2.0
    min_seg = min(seg_loss_history) if seg_loss_history else 2.0
    
    print(f"SEG Loss: 初期値 {initial_seg:.4f} → 最終値 {final_seg:.4f} (最小値 {min_seg:.4f})")
    
    reduction = (initial_seg - final_seg) / initial_seg * 100 if initial_seg > 0 else 0
    
    if reduction > 50:
        print(f"✅ SEG Lossが{reduction:.1f}%減少 → 設計は正常に機能しています！")
    elif reduction > 20:
        print(f"⚠️ SEG Lossが{reduction:.1f}%減少 → ある程度機能していますが、改善の余地があります")
    else:
        print(f"❌ SEG Lossの減少が{reduction:.1f}%のみ → 根本的な問題があります")
        print("\n考えられる原因:")
        print("1. マスクデコーダへの特徴伝播に問題")
        print("2. アダプターの初期化が不適切")
        print("3. 学習率が低すぎる")

if __name__ == "__main__":
    test_simple_overfitting()