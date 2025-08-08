#!/usr/bin/env python3
"""
バッチ処理でのトークン選択テスト
"""
import torch
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from src.models.lisa_model import LISA_Model
from src.config import LISAConfig
from src.data.dataset import HybridDataset
from src.data.collators import MultiModalDataCollator
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from torch.utils.data import DataLoader
import logging

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_batch_processing():
    """バッチ処理でのトークン選択テスト"""
    
    print("=" * 80)
    print("バッチ処理トークン選択テスト")
    print("=" * 80)
    
    # 1. モデル準備
    print("\n1. モデル初期化...")
    config = LISAConfig()
    
    # Qwenモデルを先に読み込み
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.qwen_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    
    # LISA_Model初期化
    model = LISA_Model(
        config=config,
        qwen_model=qwen_model,
        sam_predictor=None
    )
    model.eval()
    
    # 設定確認
    print("\n設定確認:")
    if hasattr(model.qwen.config, 'image_feature_select_strategy'):
        print(f"  model.qwen.config.image_feature_select_strategy = {model.qwen.config.image_feature_select_strategy}")
    
    # 2. データセット準備
    print("\n2. データセット準備...")
    from src.utils import prepare_tokenizer_for_lisa
    
    tokenizer = prepare_tokenizer_for_lisa(
        model_name=config.qwen_model_name,
        seg_token=config.seg_token
    )
    
    processor = AutoProcessor.from_pretrained(
        config.qwen_model_name,
        min_pixels=config.qwen_min_pixels,
        max_pixels=config.qwen_max_pixels
    )
    
    # ミニデータセットを作成
    dataset = HybridDataset(
        base_image_dir=config.dataset_base_dir,
        qwen_processor=processor,
        samples_per_epoch=10,
        dataset='sem_seg',
        sample_rate=[1.0],
        qwen_image_size=config.qwen_image_size,
        sam_image_size=config.sam_image_size,
    )
    
    # DataLoader作成
    collator = MultiModalDataCollator(
        tokenizer=tokenizer,
        max_length=config.model_max_length,
        config=config
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collator,
        num_workers=0
    )
    
    # 3. バッチ処理テスト
    print("\n3. バッチ処理テスト...")
    
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}:")
        
        # デバイスに移動
        device = next(model.parameters()).device
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        
        B = batch['pixel_values'].shape[0]
        print(f"  バッチサイズ: {B}")
        print(f"  pixel_values shape: {batch['pixel_values'].shape}")
        
        if 'image_grid_thw' in batch and batch['image_grid_thw'] is not None:
            for i in range(B):
                grid = batch['image_grid_thw'][i]
                H_raw, W_raw = int(grid[1].item()), int(grid[2].item())
                raw_patches = H_raw * W_raw
                expected_tokens = raw_patches // 4
                print(f"  Sample {i}: {H_raw}×{W_raw} = {raw_patches} patches → {expected_tokens} tokens expected")
        
        # get_image_featuresを直接呼び出し
        with torch.no_grad():
            try:
                # extract_vision_featuresを呼び出し
                vision_features = model.extract_vision_features(
                    batch['pixel_values'],
                    batch.get('image_grid_thw')
                )
                
                print(f"  vision_features shape: {vision_features.shape}")
                
                if vision_features.shape[0] == B:
                    print(f"  ✓ バッチ処理成功")
                else:
                    print(f"  ✗ バッチサイズ不一致")
                
            except Exception as e:
                print(f"  ✗ エラー発生: {e}")
                
                # エラー時の詳細情報
                image_embeds = model.qwen.model.get_image_features(
                    batch['pixel_values'],
                    batch.get('image_grid_thw')
                )
                if isinstance(image_embeds, tuple):
                    image_embeds = image_embeds[0]
                
                print(f"  get_image_features raw output shape: {image_embeds.shape}")
                print(f"  Total tokens: {image_embeds.shape[0]}")
                print(f"  Batch size: {B}")
                print(f"  Tokens per sample (if equal): {image_embeds.shape[0] / B if B > 0 else 0}")
        
        # 最初のバッチのみテスト
        break
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_batch_processing()