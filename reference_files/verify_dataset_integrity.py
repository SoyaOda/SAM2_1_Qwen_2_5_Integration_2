#!/usr/bin/env python3
"""
第1節：データセット構築と完全性の検証 (HybridDataset)

HybridDatasetが各サブデータセットを正しく処理し、意図した通りの学習サンプルを生成しているかを
目視で確認するための検証スクリプト。

論理的根拠:
- データパイプラインの欠陥は最も一般的かつ致命的なエラー源
- 画像とテキストのペア不整合、セグメンテーションマスクの間違い、対話ターンの不適切な形式を検出
- 各データソースの「意味的意図」が最終的な学習サンプル形式で正しく保持されているかを保証
"""

import argparse
import os
import sys
import random
import json
from collections import defaultdict
from datetime import datetime

import torch
from torchvision.transforms import ToPILImage
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # バックエンドを非対話型に設定
import matplotlib.pyplot as plt
import numpy as np

# プロジェクトのルートディレクトリをsys.pathに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_config():
    """動的設定読み込み"""
    config_candidates = ['config_linux', 'config_small_test']
    
    for config_name in config_candidates:
        try:
            config_module = __import__(config_name)
            print(f"✓ 設定ファイルを使用: {config_name}")
            return config_module
        except ImportError:
            continue
    
    raise ImportError("利用可能な設定ファイルが見つかりません")

def parse_args():
    parser = argparse.ArgumentParser(description="Verify HybridDataset Integrity")
    parser.add_argument(
        "--datasets", 
        nargs='+', 
        default=["sem_seg", "refer_seg", "vqa", "reason_seg"],
        help="List of sub-dataset names to inspect"
    )
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to inspect per dataset")
    return parser.parse_args()

def save_comparison_image(image_tensor, mask_tensor, dataset_name, sample_idx, 
                         image_path, conversations, questions, class_names, session_timestamp):
    """比較画像とメタデータを保存（画面表示なし）- 学習時と同じ処理"""
    try:
        item_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # ミリ秒まで
        base_filename = f"{dataset_name}_{sample_idx}_{item_timestamp}"
        
        # セッション用のサブディレクトリ作成
        output_dir = os.path.join("verification_output", f"session_{session_timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 画像処理（元画像から直接読み込み）
        try:
            # 元画像を直接読み込み（最も確実な方法）
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"元画像読み込みエラー: {e}")
            # フォールバック: SAM前処理済み画像の逆変換
            pil_converter = ToPILImage()
            
            if image_tensor.dim() == 4:
                image_tensor = image_tensor[0]
            
            # SAM前処理の逆変換（正規化とパディングを元に戻す）
            pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
            pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
            
            # 正規化を逆変換
            image_denorm = image_tensor * pixel_std + pixel_mean
            image_denorm = torch.clamp(image_denorm / 255.0, 0, 1)
            
            # PIL画像に変換（1024x1024のまま）
            image = pil_converter(image_denorm.cpu())
        
        # マスク処理 - 画像と同じサイズにリサイズ
        mask = None
        if mask_tensor is not None:
            if isinstance(mask_tensor, list) and len(mask_tensor) > 0:
                mask_tensor = mask_tensor[0]
            
            if isinstance(mask_tensor, torch.Tensor):
                mask = mask_tensor.cpu().numpy()
            
            if mask is not None and mask.ndim == 3:
                if mask.shape[0] == 1:
                    mask = mask.squeeze(0)
                elif mask.shape[-1] == 1:
                    mask = mask.squeeze(-1)
                else:
                    mask = mask[0]
            
            # マスクを画像と同じサイズにリサイズ（シンプルな方法）
            if mask is not None and mask.ndim == 2:
                # 画像のサイズを取得
                img_width, img_height = image.size
                
                # マスクをPILでリサイズ
                mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
                mask_resized = mask_pil.resize((img_width, img_height), Image.NEAREST)
                mask = np.array(mask_resized) / 255.0
        
        # 比較画像作成（画面表示なし）
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # SAM処理済み画像
        axes[0].imshow(image)
        axes[0].set_title(f"{dataset_name} - SAM Processed Image\n({image.size[0]}x{image.size[1]})")
        axes[0].axis('off')
        
        # マスクのみ
        if mask is not None and mask.ndim == 2:
            axes[1].imshow(mask, cmap='gray')
            axes[1].set_title(f"{dataset_name} - Resized Mask\n({mask.shape[1]}x{mask.shape[0]})")
        else:
            axes[1].text(0.5, 0.5, 'No Mask\n(VQA Task)', ha='center', va='center', 
                        transform=axes[1].transAxes, fontsize=12)
            axes[1].set_title(f"{dataset_name} - No Mask")
        axes[1].axis('off')
        
        # マスク付き画像
        axes[2].imshow(image)
        if mask is not None and mask.ndim == 2:
            # マスクと画像のサイズが一致することを確認
            if mask.shape[:2] == (image.size[1], image.size[0]):
                mask_colored = np.zeros((*mask.shape, 4))
                mask_colored[mask > 0.1] = [1, 0, 0, 0.6]  # 赤色半透明
                axes[2].imshow(mask_colored)
                axes[2].set_title(f"{dataset_name} - Synchronized Overlay")
            else:
                axes[2].set_title(f"{dataset_name} - Size Mismatch: img{image.size} vs mask{mask.shape}")
        else:
            axes[2].set_title(f"{dataset_name} - Image Only")
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # 画像保存（画面表示なし）
        image_filename = os.path.join(output_dir, f"{base_filename}_comparison.png")
        plt.savefig(image_filename, dpi=150, bbox_inches='tight')
        plt.close()  # メモリリークを防ぐためにfigureを閉じる
        
        # メタデータ保存
        metadata = {
            "dataset_name": dataset_name,
            "sample_idx": sample_idx,
            "session_timestamp": session_timestamp,
            "item_timestamp": item_timestamp,
            "original_image_path": image_path,
            "conversations": conversations,
            "questions": questions,
            "class_names": class_names,
            "comparison_image": image_filename,
            "processing_note": "Simple resize: mask resized to match processed image size",
            "image_size": f"{image.size[0]}x{image.size[1]}",
            "mask_size": f"{mask.shape[1]}x{mask.shape[0]}" if mask is not None else "None"
        }
        
        metadata_filename = os.path.join(output_dir, f"{base_filename}_metadata.json")
        import json
        with open(metadata_filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        return image_filename, metadata_filename
        
    except Exception as e:
        print(f"保存エラー: {e}")
        import traceback
        traceback.print_exc()
        return None, None



def main():
    args = parse_args()
    config = get_config()
    
    # セッションタイムスタンプの生成
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("🔍 LISA-Gemma Dataset Integrity Verification")
    print("=" * 60)
    print(f"🚀 検証セッション開始: {session_timestamp}")
    print(f"📁 出力ディレクトリ: verification_output/session_{session_timestamp}/")
    print("=" * 60)
    
    # 個別データセットクラスを使用した検証
    try:
        from utils.sem_seg_dataset import SemSegDataset
        from utils.refer_seg_dataset import ReferSegDataset
        from utils.vqa_dataset import VQADataset
        from utils.reason_seg_dataset import ReasonSegDataset
        
        print(f"📦 個別データセット初期化中...")
        
        # データセット設定
        dataset_configs = []
        if "sem_seg" in args.datasets:
            dataset_configs.append(("sem_seg", SemSegDataset, {
                "base_image_dir": config.DATASET_BASE_DIR,
                "tokenizer": None,
                "samples_per_epoch": 50,
                "sem_seg_data": getattr(config, 'SEM_SEG_DATA', "ade20k")
            }))
        
        if "refer_seg" in args.datasets:
            dataset_configs.append(("refer_seg", ReferSegDataset, {
                "base_image_dir": config.DATASET_BASE_DIR,
                "tokenizer": None,
                "samples_per_epoch": 50,
                "refer_seg_data": getattr(config, 'REFER_SEG_DATA', "refcoco")
            }))
        
        if "vqa" in args.datasets:
            dataset_configs.append(("vqa", VQADataset, {
                "base_image_dir": config.DATASET_BASE_DIR,
                "tokenizer": None,
                "samples_per_epoch": 50,
                "vqa_data": getattr(config, 'VQA_DATA', "llava_instruct_150k")
            }))
        
        if "reason_seg" in args.datasets:
            dataset_configs.append(("reason_seg", ReasonSegDataset, {
                "base_image_dir": config.DATASET_BASE_DIR,
                "tokenizer": None,
                "samples_per_epoch": 50,
                "reason_seg_data": getattr(config, 'REASON_SEG_DATA', "ReasonSeg|train")
            }))
        
        # 各データセットを初期化して検証
        print("\n🔬 データセット検証開始")
        
        for dataset_name, dataset_class, dataset_kwargs in dataset_configs:
            print(f"\n===== {dataset_name.upper()} データセット検証 =====")
            
            try:
                # データセット初期化
                dataset = dataset_class(**dataset_kwargs)
                print(f"✅ {dataset_name} 初期化完了 (サンプル数: {len(dataset)})")
                
                # サンプル検証
                for i in range(min(args.num_samples, len(dataset))):
                    print(f"\n--- Sample {i+1}/{args.num_samples} ---")
                    
                    try:
                        sample = dataset[i]
                        
                        # データ構造確認
                        if isinstance(sample, dict):
                            print(f"📦 辞書形式: {list(sample.keys())}")
                            
                            # 会話データ
                            if 'conversations' in sample:
                                conversations = sample['conversations']
                                print(f"💬 会話ターン数: {len(conversations)}")
                                for j, turn in enumerate(conversations[:2]):
                                    if isinstance(turn, dict):
                                        print(f"  Turn {j+1}: {turn.get('from')} -> {turn.get('value', '')[:100]}...")
                            
                            # 画像データ
                            image_tensor = sample.get('image') or sample.get('images')
                            if image_tensor is not None:
                                print(f"🖼️ 画像: {image_tensor.shape}, {image_tensor.dtype}")
                            
                            # マスクデータ
                            mask_tensor = sample.get('masks') or sample.get('mask')
                            if mask_tensor is not None:
                                if isinstance(mask_tensor, list):
                                    print(f"🎯 マスク: リスト({len(mask_tensor)}個)")
                                    if len(mask_tensor) > 0:
                                        print(f"    最初のマスク: {mask_tensor[0].shape if hasattr(mask_tensor[0], 'shape') else type(mask_tensor[0])}")
                                else:
                                    print(f"🎯 マスク: {mask_tensor.shape}, {mask_tensor.dtype}")
                            
                            # 可視化
                            if image_tensor is not None and mask_tensor is not None:
                                save_comparison_image(
                                    image_tensor, mask_tensor,
                                    dataset_name, i,
                                    image_path=sample.get('image_path', ''),
                                    conversations=conversations,
                                    questions=sample.get('questions'),
                                    class_names=sample.get('sampled_classes'),
                                    session_timestamp=session_timestamp
                                )
                                print(f"📸 {dataset_name} Sample {i+1} - 比較画像を保存しました")
                        
                        elif isinstance(sample, (list, tuple)):
                            print(f"📦 tuple形式: {len(sample)}要素")
                            for j, element in enumerate(sample):
                                if isinstance(element, torch.Tensor):
                                    print(f"  [{j}]: Tensor {element.shape}")
                                elif isinstance(element, str):
                                    print(f"  [{j}]: String ('{element[:50]}...')")
                                elif isinstance(element, list):
                                    print(f"  [{j}]: List({len(element)}個)")
                                elif isinstance(element, tuple):
                                    print(f"  [{j}]: tuple")
                                else:
                                    print(f"  [{j}]: {type(element)}")
                            
                            # 会話データの確認（タプル形式の場合）
                            conversation_element = None
                            if len(sample) > 3 and isinstance(sample[3], list):
                                conversation_element = sample[3]
                            
                            if conversation_element is not None:
                                print(f"💬 会話ターン数: {len(conversation_element)}")
                                for j, turn in enumerate(conversation_element[:2]):
                                    if isinstance(turn, str):
                                        print(f"  Turn {j+1}: {turn[:100]}...")
                                    else:
                                        print(f"  Turn {j+1}: {type(turn)}")
                            
                            # 可視化と保存（タプル形式）
                            try:
                                # 要素の抽出
                                image_path = sample[0] if len(sample) > 0 and isinstance(sample[0], str) else ""
                                sam_image = sample[1] if len(sample) > 1 and isinstance(sample[1], torch.Tensor) else None
                                gemma_image = sample[2] if len(sample) > 2 and isinstance(sample[2], torch.Tensor) else None
                                conversations = sample[3] if len(sample) > 3 and isinstance(sample[3], list) else None
                                masks = sample[4] if len(sample) > 4 and isinstance(sample[4], torch.Tensor) else None
                                questions = sample[7] if len(sample) > 7 and isinstance(sample[7], list) else None
                                sampled_classes = sample[8] if len(sample) > 8 and isinstance(sample[8], list) else None
                                
                                # SAM画像（要素1）とマスク（要素4）で可視化・保存
                                if sam_image is not None and masks is not None:
                                    # マスクが空でないかチェック
                                    if masks.numel() > 0:  # マスクにデータがある場合
                                        # 比較画像の保存
                                        save_comparison_image(
                                            sam_image, masks, 
                                            dataset_name, i,
                                            image_path=image_path,
                                            conversations=conversations,
                                            questions=questions,
                                            class_names=sampled_classes,
                                            session_timestamp=session_timestamp
                                        )
                                        print(f"📸 {dataset_name} Sample {i+1} - 比較画像を保存しました")
                                    else:
                                        # マスクが空の場合（VQAなど）
                                        save_comparison_image(
                                            sam_image, None, 
                                            dataset_name, i,
                                            image_path=image_path,
                                            conversations=conversations,
                                            questions=questions,
                                            class_names=sampled_classes,
                                            session_timestamp=session_timestamp
                                        )
                                        print(f"📸 {dataset_name} Sample {i+1} - マスクが空のため、画像のみ保存（VQAタスクなど）")
                                elif sam_image is not None:
                                    # マスクがない場合でも画像は保存
                                    save_comparison_image(
                                        sam_image, None, 
                                        dataset_name, i,
                                        image_path=image_path,
                                        conversations=conversations,
                                        questions=questions,
                                        class_names=sampled_classes,
                                        session_timestamp=session_timestamp
                                    )
                                    print(f"📸 {dataset_name} Sample {i+1} - マスクデータがないため、画像のみ保存")
                                else:
                                    print("⚠️ 可視化用データが不足")
                                    
                            except Exception as viz_e:
                                print(f"❌ 可視化・保存エラー: {viz_e}")
                        
                        print(f"  ✅ サンプル {i+1} 検証完了")
                        
                    except Exception as e:
                        print(f"  ❌ サンプル {i+1} 検証失敗: {e}")
                        import traceback
                        traceback.print_exc()
                
            except Exception as e:
                print(f"❌ {dataset_name} データセット初期化失敗: {e}")
                import traceback
                traceback.print_exc()
        
        # セッション情報のサマリー作成
        session_dir = os.path.join("verification_output", f"session_{session_timestamp}")
        session_summary = {
            "session_info": {
                "session_timestamp": session_timestamp,
                "start_time": session_timestamp,
                "end_time": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "datasets_verified": args.datasets,
                "samples_per_dataset": args.num_samples,
                "config_used": os.path.basename(config_path) if 'config_path' in locals() else "config_linux.py"
            },
            "verification_results": "See individual metadata files for detailed results"
        }
        
        # セッションサマリーを保存
        summary_file = os.path.join(session_dir, "session_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(session_summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n🎉 第1節検証完了: データセット構築と完全性の検証")
        print(f"📁 保存された比較画像とメタデータ: verification_output/session_{session_timestamp}/")
        print(f"📋 セッションサマリー: {summary_file}")
        print(f"🕐 セッション終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"❌ 検証失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 