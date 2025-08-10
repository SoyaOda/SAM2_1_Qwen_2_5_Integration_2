#!/usr/bin/env python3
"""
テキスト処理とラベルマスキングのテストスクリプト
HybridDatasetが正しくuser/assistantを分離してラベルマスキングしているかを確認
"""

import json
import os
import sys
import torch
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.dataset import HybridDataset
from transformers import AutoProcessor

def test_text_processing():
    """テキスト処理とラベルマスキングのテスト"""
    
    print("=" * 80)
    print("テキスト処理とラベルマスキングのテスト")
    print("=" * 80)
    
    # プロセッサの初期化
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    
    # データセットの初期化
    print("\n1. HybridDatasetを初期化中...")
    dataset = HybridDataset(
        base_image_dir="/mnt/h/download/LISA-dataset/dataset",
        qwen_processor=processor,
        samples_per_epoch=10,
        dataset="sem_seg||refer_seg||reason_seg||vqa",
        sample_rate=[1, 1, 1, 1],
    )
    
    print(f"   データセット数: {len(dataset.all_datasets)}")
    
    # 各データセットタイプからサンプルを取得
    dataset_names = ["sem_seg", "refer_seg", "reason_seg", "vqa"]
    
    for i, dataset_name in enumerate(dataset_names):
        if i >= len(dataset.all_datasets):
            continue
            
        print(f"\n2. {dataset_name}データセットのテスト")
        print("-" * 40)
        
        # サンプルを取得
        sample = dataset[i]
        
        # input_idsとlabelsを確認
        input_ids = sample['input_ids']
        labels = sample['labels']
        tokenizer = processor.tokenizer
        
        # トークンをデコード
        tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
        
        # ラベルマスキングの確認
        print(f"   入力長: {len(input_ids)}")
        print(f"   ラベル長: {len(labels)}")
        
        # マスクされているトークンとされていないトークンを確認
        masked_count = (labels == -100).sum().item()
        unmasked_count = (labels != -100).sum().item()
        
        print(f"   マスクされたトークン数: {masked_count}")
        print(f"   学習対象トークン数: {unmasked_count}")
        
        # 最初の数トークンを表示
        print("\n   最初の20トークン:")
        for j in range(min(20, len(tokens))):
            token = tokens[j]
            label = labels[j].item()
            mask_status = "MASKED" if label == -100 else f"LEARN({label})"
            print(f"     {j:3d}: {token:20s} -> {mask_status}")
        
        # アシスタント応答部分を探す
        assistant_start = -1
        for j, token in enumerate(tokens):
            if "assistant" in token.lower():
                assistant_start = j
                break
        
        if assistant_start >= 0:
            print(f"\n   アシスタント応答開始位置: {assistant_start}")
            print("   アシスタント応答付近のトークン:")
            start = max(0, assistant_start - 2)
            end = min(len(tokens), assistant_start + 10)
            for j in range(start, end):
                token = tokens[j]
                label = labels[j].item()
                mask_status = "MASKED" if label == -100 else f"LEARN({label})"
                marker = " <-- assistant" if j == assistant_start else ""
                print(f"     {j:3d}: {token:20s} -> {mask_status}{marker}")
        
        # <SEG>トークンを探す
        seg_token_id = tokenizer.convert_tokens_to_ids("<SEG>")
        seg_positions = (input_ids == seg_token_id).nonzero(as_tuple=True)[0].tolist()
        
        if seg_positions:
            print(f"\n   <SEG>トークン位置: {seg_positions}")
            for pos in seg_positions:
                print(f"   <SEG>トークン周辺 (位置 {pos}):")
                start = max(0, pos - 3)
                end = min(len(tokens), pos + 4)
                for j in range(start, end):
                    token = tokens[j]
                    label = labels[j].item()
                    mask_status = "MASKED" if label == -100 else f"LEARN({label})"
                    marker = " <-- SEG" if j == pos else ""
                    print(f"     {j:3d}: {token:20s} -> {mask_status}{marker}")
        
        # text_promptの確認
        if 'text_prompt' in sample and sample['text_prompt']:
            print(f"\n   text_prompt: {sample['text_prompt'][:100]}...")
        
        print("\n" + "=" * 40)

if __name__ == "__main__":
    test_text_processing()