#!/usr/bin/env python3
"""
最小限のSEG Loss検証テスト
- メモリ効率的
- デバッグ出力付き
- 勾配無効化オプション付き
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
from PIL import Image
import gc

sys.path.append(str(Path(__file__).parent))

from src.config import LISAConfig
from src.models.lisa_model import LISA_Model
from src.utils.tokenizer_utils import prepare_tokenizer_for_lisa
from transformers import AutoProcessor

def test_minimal_seg():
    """最小限のSEG Loss検証"""
    
    print("=" * 80)
    print("最小限のSEG Loss検証テスト")
    print("=" * 80)
    
    # メモリクリア
    gc.collect()
    torch.cuda.empty_cache()
    
    # 設定（メモリ節約）
    config = LISAConfig(
        qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        sam_model_name="./checkpoints/sam2.1_hiera_large.pt",
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype="bfloat16",  # メモリ節約のためbf16使用
        freeze_qwen=True,  # Qwenは凍結
        freeze_sam=True,   # SAMも凍結
        train_seg_token=True,
        segmentation_loss_weight=1.0
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
    
    seg_token_id = tokenizer.convert_tokens_to_ids(config.seg_token)
    print(f"SEGトークンID: {seg_token_id}")
    
    # モデルロード（メモリ効率的に）
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        model = LISA_Model(config)
        model.set_tokenizer(tokenizer)
        model = model.to(device)
        model.eval()  # まず評価モードで動作確認
    
    print("モデルロード完了")
    
    # テストデータ作成
    print("\n2. テストデータの作成")
    
    # 小さい画像で検証（224x224）
    image_size = 224
    image = Image.new('RGB', (image_size, image_size), color=(128, 128, 128))
    pixels = np.array(image)
    pixels[50:150, 50:150] = [255, 0, 0]  # 赤い四角
    image = Image.fromarray(pixels.astype(np.uint8))
    
    # マスク
    gt_mask = torch.zeros(1, image_size, image_size, device=device)
    gt_mask[0, 50:150, 50:150] = 1.0
    
    # テキスト
    text = "Segment the red square. <SEG>"
    
    # メッセージ形式
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text}
            ]
        }
    ]
    
    # 処理
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
    
    # image_grid_thw
    if 'image_grid_thw' in processed:
        image_grid_thw = processed['image_grid_thw'].to(device)
    else:
        image_grid_thw = torch.tensor([[1, 16, 16]], dtype=torch.long, device=device)
    
    # ラベル
    labels = input_ids.clone()
    labels[:, :10] = -100
    
    # SEGトークン確認
    seg_positions = (input_ids == seg_token_id).nonzero(as_tuple=True)
    if len(seg_positions[1]) > 0:
        print(f"✅ SEGトークン検出（位置: {seg_positions[1].tolist()}）")
    else:
        print(f"❌ SEGトークン未検出")
        return
    
    # 推論のみ（勾配なし）で動作確認
    print("\n3. 推論テスト（勾配なし）")
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            print("Forward pass開始...")
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                labels=labels,
                mask_labels=[gt_mask],
                image_grid_thw=image_grid_thw
            )
            print("Forward pass完了")
            
            # SEG Loss計算
            seg_loss = torch.tensor(2.0, device=device)
            
            if outputs.mask_logits and len(outputs.mask_logits) > 0:
                if outputs.mask_logits[0] is not None and len(outputs.mask_logits[0]) > 0:
                    pred_mask = outputs.mask_logits[0][0]
                    
                    print(f"予測マスク shape: {pred_mask.shape}")
                    print(f"GTマスク shape: {gt_mask.shape}")
                    
                    # サイズ調整
                    if pred_mask.shape[-2:] != gt_mask.shape[-2:]:
                        pred_mask = nn.functional.interpolate(
                            pred_mask.unsqueeze(0) if pred_mask.dim() == 2 else pred_mask,
                            size=gt_mask.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        )
                    
                    # 次元調整
                    while pred_mask.dim() > 2:
                        pred_mask = pred_mask.squeeze(0)
                    while gt_mask.dim() > 2:
                        gt_mask = gt_mask.squeeze(0)
                    
                    # 損失計算
                    bce_loss = nn.functional.binary_cross_entropy_with_logits(
                        pred_mask, gt_mask.float()
                    )
                    
                    pred_sigmoid = torch.sigmoid(pred_mask)
                    intersection = (pred_sigmoid * gt_mask).sum()
                    dice = 2 * intersection / (pred_sigmoid.sum() + gt_mask.sum() + 1e-8)
                    dice_loss = 1 - dice
                    
                    seg_loss = bce_loss + dice_loss
                    
                    print(f"\n損失値:")
                    print(f"  BCE Loss: {bce_loss.item():.4f}")
                    print(f"  Dice Loss: {dice_loss.item():.4f}")
                    print(f"  SEG Loss: {seg_loss.item():.4f}")
                else:
                    print("❌ マスクが生成されませんでした")
            else:
                print("❌ mask_logitsが空です")
    
    # 簡単な学習テスト（オプション）
    print("\n4. 学習可能性テスト（5ステップ）")
    
    # アダプターのみ学習可能に
    for param in model.parameters():
        param.requires_grad = False
    for param in model.image_adapter.parameters():
        param.requires_grad = True
    for param in model.text_prompt_proj.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.Adam([
        {'params': model.image_adapter.parameters(), 'lr': 1e-3},
        {'params': model.text_prompt_proj.parameters(), 'lr': 1e-3}
    ])
    
    model.train()
    seg_losses = []
    
    for step in range(5):
        print(f"\nStep {step + 1}/5")
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                labels=labels,
                mask_labels=[gt_mask],
                image_grid_thw=image_grid_thw
            )
            
            # 簡易損失計算
            if outputs.mask_logits and outputs.mask_logits[0]:
                pred_mask = outputs.mask_logits[0][0]
                
                # サイズ調整（必要な場合）
                if pred_mask.shape[-2:] != (image_size, image_size):
                    pred_mask = nn.functional.interpolate(
                        pred_mask.unsqueeze(0) if pred_mask.dim() == 2 else pred_mask,
                        size=(image_size, image_size),
                        mode='bilinear',
                        align_corners=False
                    )
                
                while pred_mask.dim() > 2:
                    pred_mask = pred_mask.squeeze(0)
                
                # 損失
                gt_mask_2d = gt_mask.squeeze(0) if gt_mask.dim() > 2 else gt_mask
                loss = nn.functional.binary_cross_entropy_with_logits(
                    pred_mask, gt_mask_2d.float()
                )
                
                seg_losses.append(loss.item())
                print(f"  Loss: {loss.item():.4f}")
                
                # バックプロパゲーション
                optimizer.zero_grad()
                loss.backward()
                
                # 勾配クリッピング
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
    
    # 結果分析
    print("\n" + "=" * 80)
    print("診断結果")
    print("=" * 80)
    
    if len(seg_losses) >= 2:
        improvement = (seg_losses[0] - seg_losses[-1]) / seg_losses[0] * 100
        print(f"SEG Loss: {seg_losses[0]:.4f} → {seg_losses[-1]:.4f} ({improvement:.1f}%改善)")
        
        if improvement > 10:
            print("✅ SEG Lossが下がる可能性があります")
        else:
            print("⚠️ SEG Lossの下がりが小さいです。学習率調整が必要かもしれません")
    else:
        print("❌ 学習テストが完了しませんでした")

if __name__ == "__main__":
    test_minimal_seg()