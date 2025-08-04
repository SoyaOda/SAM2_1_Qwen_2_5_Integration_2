"""
データセット読み込みテスト
各データセットクラスの基本的な動作確認
"""
import os
import sys
import torch
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import LISAConfig
from src.data.dataset import HybridDataset, preprocess_qwen_image, preprocess_sam_image
from src.data.sem_seg_dataset import SemSegDataset
from src.data.refer_seg_dataset import ReferSegDataset
from src.data.vqa_dataset import VQADataset
from src.data.reason_seg_dataset import ReasonSegDataset
from transformers import AutoProcessor, AutoTokenizer
from PIL import Image
import numpy as np


def test_dataset_loading():
    """データセット読み込みのテスト"""
    print("=== データセット読み込みテスト開始 ===\n")
    
    # 設定
    config = LISAConfig()
    
    # Qwen2.5-VLのプロセッサとトークナイザを準備
    print("1. Qwen2.5-VLプロセッサの準備...")
    try:
        processor = AutoProcessor.from_pretrained(config.qwen_model_name)
        tokenizer = AutoTokenizer.from_pretrained(config.qwen_model_name)
        
        # SEGトークンの追加
        if config.seg_token not in tokenizer.get_vocab():
            tokenizer.add_tokens([config.seg_token])
            print(f"   ✓ {config.seg_token}トークンを追加しました")
        
        seg_token_idx = tokenizer.convert_tokens_to_ids(config.seg_token)
        print(f"   ✓ プロセッサとトークナイザの準備完了")
        print(f"   ✓ SEGトークンID: {seg_token_idx}")
    except Exception as e:
        print(f"   ✗ エラー: {e}")
        return
    
    # テスト用のダミー画像を作成
    print("\n2. テスト用ダミー画像の作成...")
    dummy_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    dummy_image_path = "/tmp/test_image.jpg"
    dummy_image.save(dummy_image_path)
    print(f"   ✓ ダミー画像を作成: {dummy_image_path}")
    
    # 各データセットクラスの基本動作確認
    print("\n3. 各データセットクラスの動作確認...")
    
    # 3.1 画像前処理関数のテスト
    print("\n3.1 画像前処理関数のテスト")
    try:
        qwen_image = preprocess_qwen_image(dummy_image, processor, config.qwen_image_size)
        sam_image = preprocess_sam_image(dummy_image, config.sam_image_size)
        print(f"   ✓ Qwen画像前処理: shape={qwen_image.shape}")
        print(f"   ✓ SAM画像前処理: shape={sam_image.shape}")
    except Exception as e:
        print(f"   ✗ 画像前処理エラー: {e}")
    
    # 3.2 HybridDatasetの作成（実際のデータなしでのテスト）
    print("\n3.2 HybridDatasetの初期化テスト")
    try:
        # データセットパスの存在確認
        dataset_base_dir = Path(config.dataset_base_dir)
        if not dataset_base_dir.exists():
            print(f"   ⚠ データセットディレクトリが見つかりません: {dataset_base_dir}")
            print("   ⚠ 実際のデータセットをテストする場合は、適切なパスを設定してください")
        else:
            print(f"   ✓ データセットディレクトリを確認: {dataset_base_dir}")
            
            # 実際にHybridDatasetを作成してみる（エラーが出ても良い）
            try:
                hybrid_dataset = HybridDataset(
                    base_image_dir=config.dataset_base_dir,
                    qwen_processor=processor,
                    samples_per_epoch=500 * 8 * 2 * 10,
                    precision="fp32",
                    qwen_image_size=config.qwen_image_size,
                    sam_image_size=config.sam_image_size,
                    exclude_val=False,
                    dataset="sem_seg||refer_seg||vqa||reason_seg",
                    sample_rate=[9, 3, 3, 1],
                    sem_seg_data=config.sem_seg_data,
                    refer_seg_data=config.refer_seg_data,
                    vqa_data=config.vqa_data,
                    reason_seg_data=config.reason_seg_data
                )
                print(f"   ✓ HybridDatasetの作成成功")
                print(f"   ✓ データセット数: {len(hybrid_dataset.all_datasets)}")
                
                # 最初のサンプルを取得してみる
                if len(hybrid_dataset) > 0:
                    try:
                        sample = hybrid_dataset[0]
                        print("\n   サンプルデータの構造:")
                        for key, value in sample.items():
                            if isinstance(value, torch.Tensor):
                                print(f"     - {key}: {value.shape}")
                            elif value is None:
                                print(f"     - {key}: None")
                            else:
                                print(f"     - {key}: {type(value)}")
                    except Exception as e:
                        print(f"   ⚠ サンプル取得エラー: {e}")
            except Exception as e:
                print(f"   ⚠ HybridDataset作成エラー: {e}")
                print("   ⚠ データセットが正しく配置されていない可能性があります")
    except Exception as e:
        print(f"   ✗ エラー: {e}")
    
    # 後処理
    if os.path.exists(dummy_image_path):
        os.remove(dummy_image_path)
        print(f"\n✓ テスト用画像を削除しました")
    
    print("\n=== テスト完了 ===")


if __name__ == "__main__":
    test_dataset_loading()