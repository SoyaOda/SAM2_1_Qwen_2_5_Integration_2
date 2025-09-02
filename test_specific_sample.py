#!/usr/bin/env python3
"""
特定のエラーサンプル(ADE20k #6949)の詳細を確認するテストスクリプト
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor

# プロジェクトルートをPythonパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.dataset import HybridDataset
from src.data.sem_seg_dataset import SemSegDataset

def test_specific_sample():
    """特定サンプルの詳細を確認"""
    
    print("="*60)
    print("特定サンプル (ADE20k #6949) の詳細確認")
    print("="*60)
    
    # プロセッサの初期化
    processor = AutoProcessor.from_pretrained('Qwen/Qwen2-VL-2B-Instruct')
    tokenizer = processor.tokenizer
    
    # SEGトークンを追加
    tokenizer.add_tokens(["<SEG>"])
    seg_token_idx = tokenizer.convert_tokens_to_ids("<SEG>")
    print(f"<SEG>トークンID: {seg_token_idx}")
    
    # SemSegDatasetを直接作成
    sem_seg_dataset = SemSegDataset(
        base_image_dir="/mnt/h/download/LISA-dataset/data/dataset",
        tokenizer=tokenizer,
        vision_tower=None,
        samples_per_epoch=10000,
        precision="bf16",
        image_size=1024,
        num_classes_per_sample=3,
        exclude_val=False,
        sem_seg_data="ade20k||train",
    )
    
    print(f"SemSegDatasetサイズ: {len(sem_seg_dataset)}")
    
    # サンプル6949を取得
    try:
        sample_idx = 6949
        sample = sem_seg_dataset[sample_idx]
        
        print(f"\n### サンプル {sample_idx} の詳細 ###")
        print(f"サンプルのタイプ: {type(sample)}")
        print(f"サンプルの要素数: {len(sample) if isinstance(sample, tuple) else 'N/A'}")
        
        if isinstance(sample, tuple) and len(sample) >= 9:
            image_path = sample[0]
            image_sam = sample[1]
            image_qwen = sample[2]
            conversation_messages = sample[3]
            masks = sample[4]
            label = sample[5]
            resize = sample[6]
            questions = sample[7]
            sampled_classes = sample[8]
            
            print(f"\n[データ詳細]")
            print(f"画像パス: {image_path}")
            print(f"画像パス存在: {os.path.exists(image_path) if image_path else False}")
            
            # SAM画像
            print(f"\nSAM画像:")
            print(f"  タイプ: {type(image_sam)}")
            if isinstance(image_sam, torch.Tensor):
                print(f"  形状: {image_sam.shape}")
                print(f"  dtype: {image_sam.dtype}")
                print(f"  requires_grad: {image_sam.requires_grad}")
            
            # Qwen画像
            print(f"\nQwen画像:")
            print(f"  タイプ: {type(image_qwen)}")
            if isinstance(image_qwen, torch.Tensor):
                print(f"  形状: {image_qwen.shape}")
                print(f"  dtype: {image_qwen.dtype}")
            
            # マスク
            print(f"\nマスク:")
            print(f"  タイプ: {type(masks)}")
            if masks is not None:
                if isinstance(masks, torch.Tensor):
                    print(f"  形状: {masks.shape}")
                    print(f"  dtype: {masks.dtype}")
                    print(f"  最小値/最大値: {masks.min().item()}/{masks.max().item()}")
                    print(f"  非ゼロ要素数: {(masks > 0).sum().item()}")
                    print(f"  requires_grad: {masks.requires_grad}")
                elif isinstance(masks, np.ndarray):
                    print(f"  形状: {masks.shape}")
                    print(f"  dtype: {masks.dtype}")
                    print(f"  最小値/最大値: {masks.min()}/{masks.max()}")
                    print(f"  非ゼロ要素数: {(masks > 0).sum()}")
            else:
                print("  マスクはNone!")
            
            # 会話メッセージ
            print(f"\n会話メッセージ:")
            print(f"  タイプ: {type(conversation_messages)}")
            if conversation_messages:
                if isinstance(conversation_messages, list):
                    print(f"  長さ: {len(conversation_messages)}")
                    if len(conversation_messages) > 0:
                        first_msg = conversation_messages[0]
                        if isinstance(first_msg, dict):
                            print(f"  最初のメッセージ: {first_msg}")
                        elif isinstance(first_msg, list) and len(first_msg) > 0:
                            print(f"  最初のメッセージ (ネスト): {first_msg[0] if first_msg else 'empty'}")
            
            # ラベル
            print(f"\nラベル:")
            print(f"  タイプ: {type(label)}")
            if isinstance(label, torch.Tensor):
                print(f"  値: {label.item() if label.dim() == 0 else label}")
            
            # その他
            print(f"\nその他:")
            print(f"  resize: {resize}")
            print(f"  questions: {questions}")
            print(f"  sampled_classes: {sampled_classes}")
            
            # セグメンテーションタスクかどうか判定
            is_seg_sample = masks is not None and (
                isinstance(masks, (torch.Tensor, np.ndarray)) and 
                (masks.sum() > 0 if hasattr(masks, 'sum') else True)
            )
            print(f"\nセグメンテーションタスク判定: {is_seg_sample}")
            
        # HybridDatasetでも確認
        print("\n" + "="*60)
        print("HybridDatasetでの処理確認")
        print("="*60)
        
        hybrid_dataset = HybridDataset(
            qwen_processor=processor,
            base_image_dir="/mnt/h/download/LISA-dataset/data/dataset",
            samples_per_epoch=1,
            sem_seg_data="ade20k||train",
            refer_seg_data=None,
            vqa_data=None,
            reason_seg_data=None,
            sample_rate=[1.0],
        )
        
        # 強制的に特定のサンプルを取得する方法を模擬
        # (実際にはランダムサンプリングなので、デバッグ用)
        print("\nHybridDatasetで処理してみる...")
        
        # 直接SemSegDatasetのサンプルを処理
        if len(hybrid_dataset.all_datasets) > 0:
            sem_dataset = hybrid_dataset.all_datasets[0]
            if isinstance(sem_dataset, SemSegDataset):
                raw_sample = sem_dataset[sample_idx]
                print(f"Raw sample type: {type(raw_sample)}, length: {len(raw_sample) if isinstance(raw_sample, tuple) else 'N/A'}")
                
                # HybridDatasetの__getitem__の処理を一部模擬
                if isinstance(raw_sample, tuple) and len(raw_sample) >= 9:
                    masks = raw_sample[4]
                    is_seg = masks is not None and (
                        isinstance(masks, (torch.Tensor, np.ndarray)) and 
                        (masks.sum() > 0 if hasattr(masks, 'sum') else True)
                    )
                    print(f"HybridDatasetでのセグメンテーション判定: {is_seg}")
                    
    except Exception as e:
        print(f"\nエラーが発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_specific_sample()