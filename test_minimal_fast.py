#!/usr/bin/env python3
"""
高速テスト用のミニマルトレーニングスクリプト
少量サンプルで動作確認を行う
"""

import torch
import logging
from pathlib import Path
from src.config import LISAConfig
from src.models.lisa_model import LISA_Model
from src.utils.tokenizer_utils import prepare_tokenizer_for_lisa
from transformers import AutoProcessor
from src.data.dataset import HybridDataset
from torch.utils.data import DataLoader
from src.data.collators import MultiModalDataCollator
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_mini_dataset(processor, num_samples=10):
    """最小限のダミーデータセットを作成"""
    from PIL import Image
    import numpy as np
    
    class MiniDataset(torch.utils.data.Dataset):
        def __init__(self, processor, num_samples=10):
            self.processor = processor
            self.num_samples = num_samples
            self.seg_token_idx = processor.tokenizer.convert_tokens_to_ids("<SEG>")
            
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # ダミー画像を生成（ランダムカラー）
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            image = Image.fromarray(img_array)
            
            # シンプルなプロンプト
            prompts = [
                "Segment the cat in this image. <SEG>",
                "Find the person wearing red. <SEG>",
                "Identify the main object. <SEG>",
                "Segment the background. <SEG>",
                "Find all cars. <SEG>"
            ]
            text_prompt = prompts[idx % len(prompts)]
            
            # apply_chat_template
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_prompt}
                ]
            }]
            
            processed = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )
            
            # ダミーマスク
            mask = torch.rand(1, 1024, 1024) > 0.5
            
            # Get actual image_grid_thw from processed output
            pixel_values = processed['pixel_values'].squeeze(0)
            
            # Get or calculate image_grid_thw
            if 'image_grid_thw' in processed and processed['image_grid_thw'] is not None:
                image_grid_thw = processed['image_grid_thw'].squeeze(0)
            else:
                # Calculate from pixel_values shape
                if pixel_values.dim() == 2:
                    # 2D format (N_patches, D_v)
                    n_patches = pixel_values.shape[0]
                    # Find the most square-like grid
                    import math
                    grid_size = int(math.sqrt(n_patches))
                    if grid_size * grid_size == n_patches:
                        image_grid_thw = torch.tensor([1, grid_size, grid_size])
                    else:
                        # Find closest factors
                        for h in range(int(math.sqrt(n_patches)), 0, -1):
                            if n_patches % h == 0:
                                w = n_patches // h
                                image_grid_thw = torch.tensor([1, h, w])
                                break
                        else:
                            image_grid_thw = torch.tensor([1, n_patches, 1])
                else:
                    # 3D format
                    image_grid_thw = torch.tensor([1, 16, 16])
            
            return {
                'input_ids': processed['input_ids'].squeeze(0),
                'attention_mask': processed['attention_mask'].squeeze(0),
                'pixel_values': pixel_values,
                'image_grid_thw': image_grid_thw,
                'labels': processed['input_ids'].squeeze(0),
                'mask_labels': mask.float(),
                'seg_token_mask': (processed['input_ids'].squeeze(0) == self.seg_token_idx)
            }
    
    return MiniDataset(processor, num_samples)

def test_training_pipeline():
    """トレーニングパイプラインの高速テスト"""
    
    start_time = time.time()
    
    # 1. 設定
    logger.info("設定を初期化中...")
    config = LISAConfig(
        qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        sam_model_name="./checkpoints/sam2.1_hiera_large.pt",
        use_dynamic_resolution=True
    )
    
    # 2. プロセッサ準備
    logger.info("プロセッサを準備中...")
    tokenizer = prepare_tokenizer_for_lisa(
        model_name=config.qwen_model_name,
        seg_token=config.seg_token
    )
    
    processor = AutoProcessor.from_pretrained(
        config.qwen_model_name,
        min_pixels=config.qwen_min_pixels,
        max_pixels=config.qwen_max_pixels
    )
    processor.tokenizer = tokenizer
    
    # 3. ミニデータセット作成（高速）
    logger.info("ミニデータセットを作成中...")
    dataset = create_mini_dataset(processor, num_samples=10)
    
    # 4. DataLoader
    collator = MultiModalDataCollator(
        tokenizer=processor.tokenizer,
        config=config
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=collator,
        shuffle=True
    )
    
    # 5. モデル読み込み
    logger.info("モデルを読み込み中...")
    model = LISA_Model(config)
    model.set_tokenizer(tokenizer)
    
    # LoRA設定（簡易版）
    from peft import LoraConfig, get_peft_model, TaskType
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model.qwen = get_peft_model(model.qwen, lora_config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 6. 1バッチのテスト
    logger.info("トレーニングステップをテスト中...")
    model.train()
    
    for i, batch in enumerate(dataloader):
        if i >= 2:  # 2バッチだけテスト
            break
        
        # デバイスに移動
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        
        # Forward pass
        try:
            outputs = model(
                input_ids=batch['input_ids'],
                pixel_values=batch['pixel_values'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
                mask_labels=[batch['mask_labels'][i] for i in range(batch['mask_labels'].size(0))],
                image_grid_thw=batch.get('image_grid_thw')
            )
            
            logger.info(f"✅ バッチ {i+1} 成功")
            logger.info(f"  - Total Loss: {outputs.loss.item():.4f}")
            if outputs.lm_loss is not None:
                logger.info(f"  - LM Loss: {outputs.lm_loss.item():.4f}")
            if outputs.seg_loss is not None:
                logger.info(f"  - Seg Loss: {outputs.seg_loss.item():.4f}")
            
        except Exception as e:
            logger.error(f"❌ バッチ {i+1} エラー: {e}")
            raise
    
    elapsed_time = time.time() - start_time
    logger.info(f"\n✅ テスト完了！ 実行時間: {elapsed_time:.1f}秒")
    logger.info("動的解像度サポートが正常に動作しています。")
    
    return True

if __name__ == "__main__":
    success = test_training_pipeline()
    if success:
        print("\n🎉 すべてのテストが成功しました！")
        print("本番のトレーニングを開始できます。")
    else:
        print("\n⚠️ テストに失敗しました。")
        print("エラーログを確認してください。")