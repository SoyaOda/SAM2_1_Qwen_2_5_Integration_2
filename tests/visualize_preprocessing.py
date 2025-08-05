#!/usr/bin/env python3
"""
データセット前処理の可視化テストスクリプト
各データセットから前処理後の画像（Qwen用）、マスク、テキストを可視化
SAM画像は現在使用されていないため、Qwen画像のみ表示
"""

import os
import sys
import argparse
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
import cv2
import json

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import HybridDataset, setup_seg_token
from src.data.collate_fn import lisa_collate_fn
from src.config import LISAConfig
from transformers import AutoProcessor
from torch.utils.data import DataLoader


def denormalize_image(image_tensor, mean, std):
    """正規化された画像を元に戻す"""
    if isinstance(mean, list):
        mean = torch.tensor(mean).view(3, 1, 1)
    if isinstance(std, list):
        std = torch.tensor(std).view(3, 1, 1)
    
    image = image_tensor.clone()
    image = image * std + mean
    image = torch.clamp(image, 0, 1)
    return image


def tensor_to_numpy(tensor):
    """テンソルをnumpy配列に変換（表示用）"""
    if tensor.dim() == 3:  # C, H, W
        tensor = tensor.permute(1, 2, 0)  # H, W, C
    return tensor.cpu().numpy()


def visualize_sample(sample, dataset_name, sample_idx, save_dir):
    """1つのサンプルを可視化（現在の仕様に合わせて簡略化）"""
    # データの取り出し
    pixel_values = sample.get('pixel_values')  # Qwen用画像（前処理済み）
    original_image = sample.get('original_image')  # 元画像（PIL形式）
    ground_truth_mask = sample.get('ground_truth_mask')
    text_prompt = sample.get('text_prompt', 'No text prompt')
    has_mask = sample.get('has_mask', False)
    
    # 図の作成（画像とマスクのみ）
    fig = plt.figure(figsize=(10, 5))
    
    # 1. Qwen用画像（448x448）
    ax1 = plt.subplot(1, 3, 1)
    if original_image is not None:
        # PIL画像をnumpy配列に変換
        if hasattr(original_image, 'mode'):  # PIL画像の場合
            qwen_img_np = np.array(original_image)
        else:
            # 元画像がない場合は、前処理済み画像を逆正規化して表示
            qwen_mean = [0.48145466, 0.4578275, 0.40821073]
            qwen_std = [0.26862954, 0.26130258, 0.27577711]
            qwen_img = denormalize_image(pixel_values, qwen_mean, qwen_std)
            qwen_img_np = tensor_to_numpy(qwen_img)
        ax1.imshow(qwen_img_np)
        ax1.set_title(f'Original Image\n{qwen_img_np.shape[0]}x{qwen_img_np.shape[1]}')
    else:
        # 元画像がない場合のフォールバック
        ax1.text(0.5, 0.5, 'No Original Image', ha='center', va='center', fontsize=20)
        ax1.set_title('Original Image\n(Not Available)')
    ax1.axis('off')
    
    # 2. マスク（存在する場合）
    ax2 = plt.subplot(1, 3, 2)
    if has_mask and ground_truth_mask is not None:
        if ground_truth_mask.dim() == 3:  # バッチ次元がある場合
            mask = ground_truth_mask[0]
        else:
            mask = ground_truth_mask
        mask_np = mask.cpu().numpy()
        ax2.imshow(mask_np, cmap='viridis')
        ax2.set_title(f'Ground Truth Mask\n{mask_np.shape[0]}x{mask_np.shape[1]}')
    else:
        ax2.text(0.5, 0.5, 'No Mask', ha='center', va='center', fontsize=20)
        ax2.set_title('Ground Truth Mask\n(Not Available)')
    ax2.axis('off')
    
    # 3. マスクを元画像に重ねて表示
    ax3 = plt.subplot(1, 3, 3)
    if original_image is not None and has_mask and ground_truth_mask is not None:
        qwen_img_with_mask = qwen_img_np.copy()
        if ground_truth_mask.dim() == 3:
            mask = ground_truth_mask[0]
        else:
            mask = ground_truth_mask
        mask_np = mask.cpu().numpy()
        
        # マスクのサイズをQwen画像に合わせてリサイズ
        if mask_np.shape != qwen_img_with_mask.shape[:2]:
            mask_np = cv2.resize(mask_np.astype(np.float32), 
                                (qwen_img_with_mask.shape[1], qwen_img_with_mask.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
        
        # マスクを色付きオーバーレイとして追加
        mask_colored = np.zeros_like(qwen_img_with_mask)
        mask_colored[:, :, 0] = mask_np * 0.5  # 赤チャンネル
        qwen_img_with_mask = qwen_img_with_mask * 0.7 + mask_colored * 0.3
        
        ax3.imshow(qwen_img_with_mask)
        ax3.set_title('Original Image + Mask Overlay')
    else:
        ax3.text(0.5, 0.5, 'No Overlay', ha='center', va='center', fontsize=20)
        ax3.set_title('Mask Overlay\n(Not Available)')
    ax3.axis('off')
    
    plt.suptitle(f'{dataset_name} - Sample {sample_idx}', fontsize=16)
    plt.tight_layout()
    
    # 画像の保存
    save_path = os.path.join(save_dir, f'{dataset_name}_sample_{sample_idx}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 画像保存完了: {save_path}")
    
    # テキスト情報をJSONファイルに保存
    info_data = {
        "dataset": dataset_name,
        "sample_index": sample_idx,
        "has_mask": has_mask,
        "text_prompt": text_prompt,
        "image_shape": list(pixel_values.shape) if pixel_values is not None else None,
        "mask_shape": list(ground_truth_mask.shape) if has_mask and ground_truth_mask is not None else None,
    }
    
    if has_mask and ground_truth_mask is not None:
        unique_values = torch.unique(ground_truth_mask)
        info_data["mask_unique_values"] = unique_values.tolist()
        info_data["mask_num_unique_values"] = len(unique_values)
    
    # JSONファイルに保存
    json_path = os.path.join(save_dir, f'{dataset_name}_sample_{sample_idx}_info.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(info_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 情報保存完了: {json_path}")
    
    # テキストプロンプトを別ファイルに保存
    text_path = os.path.join(save_dir, f'{dataset_name}_sample_{sample_idx}_prompt.txt')
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(text_prompt)
    
    print(f"✓ プロンプト保存完了: {text_path}")


def test_dataset_visualization(dataset_type, num_samples=5):
    """特定のデータセットタイプの可視化テスト"""
    print(f"\n=== {dataset_type} データセットの可視化テスト ===")
    
    # 設定の読み込み
    config = LISAConfig()
    
    # プロセッサの準備
    print("1. Qwen2.5-VLプロセッサの準備...")
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        trust_remote_code=True
    )
    
    # [SEG]トークンのセットアップ
    seg_token_idx = setup_seg_token(processor.tokenizer, "[SEG]")
    print(f"   ✓ SEGトークンID: {seg_token_idx}")
    
    # データセットの作成
    print(f"2. {dataset_type} データセットの作成...")
    try:
        dataset = HybridDataset(
            base_image_dir=config.dataset_base_dir,
            qwen_processor=processor,
            samples_per_epoch=num_samples * 10,  # 多めに設定してランダム性を確保
            dataset=dataset_type,
            sample_rate=[1.0],
            qwen_image_size=config.qwen_image_size,
            sam_image_size=config.sam_image_size,
        )
        print(f"   ✓ データセット作成成功")
    except Exception as e:
        print(f"   ✗ データセット作成エラー: {e}")
        return
    
    # 保存ディレクトリの作成
    save_dir = f"visualization_results/{dataset_type}"
    os.makedirs(save_dir, exist_ok=True)
    
    # サンプルの可視化
    print(f"3. サンプルの可視化（{num_samples}個）...")
    success_count = 0
    for i in range(min(num_samples, len(dataset))):
        try:
            sample = dataset[i]
            visualize_sample(sample, dataset_type, i, save_dir)
            success_count += 1
        except Exception as e:
            print(f"   ✗ サンプル {i} の可視化エラー: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n可視化結果: {success_count}/{num_samples} サンプル成功")
    print(f"保存先: {save_dir}")
    print(f"  - 画像: *_sample_*.png")
    print(f"  - 情報: *_sample_*_info.json")
    print(f"  - プロンプト: *_sample_*_prompt.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="データセット前処理の可視化")
    parser.add_argument("--dataset", type=str, default="sem_seg",
                        choices=["all", "sem_seg", "refer_seg", "vqa", "reason_seg"],
                        help="可視化するデータセットタイプ（'all'ですべてのデータセット）")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="各データセットから可視化するサンプル数")
    
    args = parser.parse_args()
    
    if args.dataset == "all":
        # すべてのデータセットタイプをテスト
        dataset_types = ["sem_seg", "refer_seg", "vqa", "reason_seg"]
        for dataset_type in dataset_types:
            try:
                test_dataset_visualization(dataset_type, args.num_samples)
            except Exception as e:
                print(f"\n{dataset_type} のテストでエラー: {e}")
                continue
    else:
        test_dataset_visualization(args.dataset, args.num_samples)
    
    print("\n=== 可視化テスト完了 ===")