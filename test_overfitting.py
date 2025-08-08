#!/usr/bin/env python3
"""
オーバーフィッティングテスト
1つのサンプルでSEG Lossが下がるポテンシャルを確認
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SingleSampleDataset(Dataset):
    """単一サンプルを繰り返すデータセット"""
    def __init__(self, processor, tokenizer, seg_token_id, device):
        self.processor = processor
        self.tokenizer = tokenizer
        self.seg_token_id = seg_token_id
        self.device = device
        
        # 固定のダミーデータを作成
        self.image = Image.new('RGB', (448, 448), color=(128, 128, 128))
        
        # 中央に赤い四角を描画（ターゲットマスク用）
        pixels = np.array(self.image)
        pixels[150:300, 150:300] = [255, 0, 0]
        self.image = Image.fromarray(pixels.astype(np.uint8))
        
        # グラウンドトゥルースマスク（赤い部分だけ1）
        self.mask = torch.zeros(1, 448, 448)
        self.mask[0, 150:300, 150:300] = 1.0
        
        # プロンプト
        self.text = "Segment the red square in this image. <SEG>"
        
    def __len__(self):
        return 100  # 100回同じサンプルを返す
    
    def __getitem__(self, idx):
        # メッセージ形式で処理
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self.image},
                    {"type": "text", "text": self.text}
                ]
            }
        ]
        
        # apply_chat_template
        processed = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        input_ids = processed['input_ids'].squeeze(0)
        attention_mask = processed['attention_mask'].squeeze(0)
        pixel_values = processed['pixel_values'].squeeze(0)
        
        # image_grid_thw
        if 'image_grid_thw' in processed:
            image_grid_thw = processed['image_grid_thw'].squeeze(0)
        else:
            image_grid_thw = torch.tensor([1, 32, 32], dtype=torch.long)
        
        # ラベル作成
        labels = input_ids.clone()
        labels[:10] = -100  # 最初の部分をマスク
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'mask_labels': [self.mask],
            'image_grid_thw': image_grid_thw
        }

def test_overfitting():
    """オーバーフィッティングテストのメイン関数"""
    
    print("=" * 80)
    print("オーバーフィッティングテスト開始")
    print("1つのサンプルでSEG Lossが下がるか確認")
    print("=" * 80)
    
    # 設定
    config = LISAConfig(
        qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        sam_model_name="./checkpoints/sam2.1_hiera_large.pt",
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype="auto",
        freeze_qwen=False,  # 全パラメータ学習可能
        freeze_sam=False,   # 全パラメータ学習可能
        train_seg_token=True,
        segmentation_loss_weight=10.0  # SEGを重視
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデル準備
    print("\n1. モデルの準備")
    tokenizer = prepare_tokenizer_for_lisa(config.qwen_model_name, config.seg_token)
    processor = AutoProcessor.from_pretrained(config.qwen_model_name)
    
    # ProcessorにもSEGトークン追加
    if config.seg_token not in processor.tokenizer.get_vocab():
        processor.tokenizer.add_special_tokens({"additional_special_tokens": [config.seg_token]})
    
    seg_token_id = tokenizer.convert_tokens_to_ids(config.seg_token)
    print(f"SEGトークンID: {seg_token_id}")
    
    model = LISA_Model(config)
    model.set_tokenizer(tokenizer)
    model = model.to(device)
    
    # データセット
    print("\n2. データセット準備")
    dataset = SingleSampleDataset(processor, tokenizer, seg_token_id, device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # オプティマイザ（高学習率）
    # SEGトークン埋め込みを含む全パラメータを学習
    optimizer = torch.optim.AdamW([
        {'params': model.image_adapter.parameters(), 'lr': 1e-2},
        {'params': model.text_prompt_proj.parameters(), 'lr': 1e-2},
        {'params': model.high_res_generator.parameters(), 'lr': 1e-2},
        {'params': model.qwen.parameters(), 'lr': 1e-3}  # Qwen全体（SEGトークン含む）
    ])
    
    # 訓練ループ
    print("\n3. 訓練開始（同じサンプルで100イテレーション）")
    model.train()
    
    loss_history = []
    seg_loss_history = []
    lm_loss_history = []
    
    for epoch in range(10):  # 10エポック
        print(f"\nEpoch {epoch+1}/10")
        
        for i, batch in enumerate(dataloader):
            if i >= 10:  # 各エポック10ステップ
                break
            
            # デバイスに移動
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward
            outputs = model(
                input_ids=batch['input_ids'],
                pixel_values=batch['pixel_values'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
                mask_labels=batch['mask_labels'],
                image_grid_thw=batch['image_grid_thw']
            )
            
            # 損失計算
            vocab_size = outputs.logits.size(-1)
            lm_loss = nn.functional.cross_entropy(
                outputs.logits.view(-1, vocab_size),
                batch['labels'].view(-1),
                ignore_index=-100
            )
            
            seg_loss = 0.0
            seg_count = 0
            
            if outputs.mask_logits:
                for batch_idx, batch_masks in enumerate(outputs.mask_logits):
                    if batch_masks is not None and len(batch_masks) > 0:
                        gt_mask = batch['mask_labels'][batch_idx].to(device)
                        
                        for pred_mask in batch_masks:
                            # サイズ調整
                            if pred_mask.shape[-2:] != gt_mask.shape[-2:]:
                                pred_mask = nn.functional.interpolate(
                                    pred_mask.unsqueeze(0),
                                    size=gt_mask.shape[-2:],
                                    mode='bilinear',
                                    align_corners=False
                                ).squeeze(0)
                            
                            # BCE損失
                            # pred_maskとgt_maskの次元を揃える
                            if pred_mask.dim() == 3 and pred_mask.shape[0] == 1:
                                pred_mask = pred_mask.squeeze(0)
                            if gt_mask.dim() == 3 and gt_mask.shape[0] == 1:
                                gt_mask = gt_mask.squeeze(0)
                            
                            bce_loss = nn.functional.binary_cross_entropy_with_logits(
                                pred_mask,
                                gt_mask.float()
                            )
                            
                            # Dice損失
                            pred_sigmoid = torch.sigmoid(pred_mask)
                            intersection = (pred_sigmoid * gt_mask).sum()
                            dice = 2 * intersection / (pred_sigmoid.sum() + gt_mask.sum() + 1e-8)
                            dice_loss = 1 - dice
                            
                            seg_loss += bce_loss + dice_loss
                            seg_count += 1
            
            if seg_count > 0:
                seg_loss = seg_loss / seg_count
            else:
                print(f"  ⚠️ SEGトークンが検出されませんでした")
                seg_loss = torch.tensor(2.0).to(device)  # ダミー値
            
            total_loss = lm_loss + config.segmentation_loss_weight * seg_loss
            
            # 記録
            loss_history.append(total_loss.item())
            seg_loss_history.append(seg_loss.item() if isinstance(seg_loss, torch.Tensor) else seg_loss)
            lm_loss_history.append(lm_loss.item())
            
            print(f"  Step {i+1}: Total={total_loss:.4f}, LM={lm_loss:.4f}, SEG={seg_loss:.4f}")
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    
    # 結果の可視化
    print("\n4. 結果の分析")
    
    # プロット
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(seg_loss_history)
    plt.title('SEG Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(lm_loss_history)
    plt.title('LM Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(loss_history)
    plt.title('Total Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('overfitting_test.png')
    print("結果を overfitting_test.png に保存しました")
    
    # 分析
    print("\n" + "=" * 80)
    print("診断結果")
    print("=" * 80)
    
    initial_seg = seg_loss_history[0] if seg_loss_history else 2.0
    final_seg = seg_loss_history[-1] if seg_loss_history else 2.0
    
    print(f"SEG Loss: {initial_seg:.4f} → {final_seg:.4f}")
    
    if final_seg < initial_seg * 0.5:
        print("✅ SEG Lossが50%以上減少 → 設計は正常に機能しています！")
    elif final_seg < initial_seg * 0.8:
        print("⚠️ SEG Lossが20%程度減少 → 学習率やアーキテクチャの調整が必要")
    else:
        print("❌ SEG Lossがほとんど下がらない → 根本的な問題があります")
        print("\n考えられる原因:")
        print("1. SEGトークンが正しく処理されていない")
        print("2. マスクデコーダへの特徴伝播に問題")
        print("3. 勾配が正しく伝播していない")
    
    # SEGトークンの検出確認
    if all(s == 2.0 for s in seg_loss_history[-10:]):
        print("\n⚠️ 警告: SEGトークンが全く検出されていません")
        print("ProcessorのトークナイザーにSEGトークンが正しく追加されているか確認してください")

if __name__ == "__main__":
    test_overfitting()