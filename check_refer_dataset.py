#!/usr/bin/env python3
"""
ReferSegデータセットの内容を確認するスクリプト
「orange」に関連するサンプルを探す
"""

import os
import sys
import random
import numpy as np
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent))

from src.data.refer_seg_dataset import ReferSegDataset
from transformers import AutoProcessor

def check_refer_seg_dataset():
    """ReferSegデータセットの内容を確認"""
    
    # プロセッサの初期化
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    
    # データセットの初期化
    print("ReferSegDataセットを初期化中...")
    dataset = ReferSegDataset(
        base_image_dir="/mnt/h/download/LISA-dataset/dataset",
        tokenizer=processor.tokenizer,
        vision_tower=None,
        samples_per_epoch=100,
        precision="bf16",
        image_size=448,
        num_classes_per_sample=1,
        exclude_val=False,
        refer_seg_data="refcoco||refcoco+||refcocog",
    )
    
    print(f"データセット数: {len(dataset)}")
    
    # 「orange」を含むサンプルを探す
    orange_samples = []
    
    print("\n'orange'を含むサンプルを探しています...")
    
    # 最初の100サンプルをチェック
    for i in range(min(100, len(dataset))):
        try:
            sample = dataset[i]
            
            # サンプルの形式を確認
            if len(sample) >= 9:
                image_path = sample[0]
                conversations = sample[3]
                questions = sample[7] if len(sample) > 7 else []
                sampled_classes = sample[8] if len(sample) > 8 else []
                
                # conversations内のテキストを確認
                if isinstance(conversations, list):
                    for conv in conversations:
                        if isinstance(conv, list):
                            for msg in conv:
                                if isinstance(msg, dict) and 'content' in msg:
                                    content = msg['content'].lower()
                                    if 'orange' in content:
                                        orange_samples.append({
                                            'index': i,
                                            'image_path': image_path,
                                            'content': msg['content'],
                                            'questions': questions,
                                            'sampled_classes': sampled_classes
                                        })
                                        break
                
                # questionsも確認
                if questions:
                    for q in questions:
                        if isinstance(q, str) and 'orange' in q.lower():
                            orange_samples.append({
                                'index': i,
                                'image_path': image_path,
                                'question': q,
                                'sampled_classes': sampled_classes
                            })
                            break
                
                # sampled_classesも確認
                if sampled_classes:
                    for cls in sampled_classes:
                        if isinstance(cls, str) and 'orange' in cls.lower():
                            orange_samples.append({
                                'index': i,
                                'image_path': image_path,
                                'class': cls,
                                'questions': questions
                            })
                            break
                            
        except Exception as e:
            print(f"サンプル {i} でエラー: {e}")
            continue
    
    # 結果を表示
    print(f"\n'orange'を含むサンプル数: {len(orange_samples)}")
    
    if orange_samples:
        print("\n最初の5つのサンプル:")
        for i, sample in enumerate(orange_samples[:5]):
            print(f"\n--- サンプル {i+1} (index: {sample['index']}) ---")
            print(f"画像パス: {sample.get('image_path', 'N/A')}")
            if 'content' in sample:
                print(f"コンテンツ: {sample['content']}")
            if 'question' in sample:
                print(f"質問: {sample['question']}")
            if 'class' in sample:
                print(f"クラス: {sample['class']}")
            if 'sampled_classes' in sample:
                print(f"サンプルクラス: {sample.get('sampled_classes', [])}")
            if 'questions' in sample:
                print(f"質問: {sample.get('questions', [])}")
    
    # 固定シードで特定のサンプルを確認
    print("\n\n固定シード(42)で特定のサンプルを確認:")
    random.seed(42)
    np.random.seed(42)
    
    sample = dataset[0]  # データセット内部でランダムサンプリング
    
    if len(sample) >= 9:
        image_path = sample[0]
        conversations = sample[3]
        questions = sample[7] if len(sample) > 7 else []
        sampled_classes = sample[8] if len(sample) > 8 else []
        
        print(f"画像パス: {image_path}")
        print(f"質問: {questions}")
        print(f"クラス: {sampled_classes}")
        
        if isinstance(conversations, list) and len(conversations) > 0:
            print(f"会話内容:")
            for conv in conversations:
                if isinstance(conv, list):
                    for msg in conv:
                        if isinstance(msg, dict):
                            print(f"  {msg.get('role', 'unknown')}: {msg.get('content', '')}")

if __name__ == "__main__":
    check_refer_seg_dataset()