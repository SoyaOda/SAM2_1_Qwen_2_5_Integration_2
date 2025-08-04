"""
データセットとモデルの統合テスト
HybridDatasetからLISA改モデルへのデータフローを確認
"""
import os
import sys
import torch
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import LISAConfig
from src.data.dataset import HybridDataset
from src.models import LISA_Model
from transformers import AutoProcessor, AutoTokenizer
import warnings
warnings.filterwarnings("ignore")


def test_dataset_model_integration():
    """データセットとモデルの統合テスト"""
    print("=== データセット・モデル統合テスト ===\n")
    
    # 設定
    config = LISAConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    
    # 1. モデルとプロセッサの準備
    print("\n1. モデルとプロセッサの準備...")
    try:
        # プロセッサとトークナイザの準備
        processor = AutoProcessor.from_pretrained(config.qwen_model_name)
        tokenizer = AutoTokenizer.from_pretrained(config.qwen_model_name)
        
        # SEGトークンの追加
        if config.seg_token not in tokenizer.get_vocab():
            tokenizer.add_tokens([config.seg_token])
        seg_token_idx = tokenizer.convert_tokens_to_ids(config.seg_token)
        
        # モデルの作成
        model = LISA_Model(config)
        model = model.to(device)
        model.eval()
        print("   ✓ LISA改モデルの作成成功")
        print(f"   ✓ SEGトークンID: {seg_token_idx}")
    except Exception as e:
        print(f"   ✗ モデル作成エラー: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. HybridDatasetの準備（VQAのみでテスト）
    print("\n2. HybridDatasetの準備...")
    try:
        # VQAデータセットのみでテスト
        hybrid_dataset = HybridDataset(
            base_image_dir=config.dataset_base_dir,
            qwen_processor=processor,
            samples_per_epoch=10,  # 少ないサンプル数でテスト
            precision="fp32",
            qwen_image_size=config.qwen_image_size,
            sam_image_size=config.sam_image_size,
            exclude_val=False,
            dataset="vqa",  # VQAのみ
            sample_rate=[1],
            vqa_data=config.vqa_data
        )
        print(f"   ✓ HybridDataset作成成功")
        print(f"   ✓ データセット数: {len(hybrid_dataset.all_datasets)}")
        print(f"   ✓ 総サンプル数: {len(hybrid_dataset)}")
    except Exception as e:
        print(f"   ✗ データセット作成エラー: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. サンプルデータの取得とフォワードパス
    print("\n3. サンプルデータの取得とフォワードパス...")
    try:
        # 最初のサンプルを取得
        sample = hybrid_dataset[0]
        print("   ✓ サンプル取得成功")
        
        # サンプルの構造を確認
        print("\n   サンプルデータの構造:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"     - {key}: {value.shape}, dtype={value.dtype}")
            elif value is None:
                print(f"     - {key}: None")
            else:
                print(f"     - {key}: {type(value)}")
        
        # バッチ化（1サンプルのバッチ）
        batch = {}
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.unsqueeze(0).to(device)
            else:
                batch[key] = value
        
        # フォワードパス
        print("\n   フォワードパステスト...")
        with torch.no_grad():
            try:
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    pixel_values=batch['pixel_values'],
                    sam_images=batch['sam_images'],
                    image_grid_thw=batch['image_grid_thw']
                )
                print("   ✓ フォワードパス成功")
                
                # 出力の確認
                if hasattr(outputs, 'language_logits'):
                    print(f"     - 言語出力shape: {outputs.language_logits.shape}")
                if hasattr(outputs, 'seg_logits'):
                    print(f"     - セグメンテーション出力shape: {outputs.seg_logits.shape if outputs.seg_logits is not None else 'None'}")
                
            except Exception as e:
                print(f"   ✗ フォワードパスエラー: {e}")
                import traceback
                traceback.print_exc()
        
    except Exception as e:
        print(f"   ✗ サンプル処理エラー: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. データセットのバッチ処理テスト
    print("\n4. データセットのバッチ処理テスト...")
    try:
        # DataLoaderの作成
        from torch.utils.data import DataLoader
        from src.data.collate_fn import get_collate_fn
        
        collate_fn = get_collate_fn(tokenizer)
        dataloader = DataLoader(
            hybrid_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        # バッチの取得
        for i, batch in enumerate(dataloader):
            if i >= 1:  # 最初のバッチのみテスト
                break
            
            print("   ✓ バッチ取得成功")
            print("\n   バッチデータの構造:")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"     - {key}: {value.shape}, dtype={value.dtype}")
                elif isinstance(value, list):
                    print(f"     - {key}: list of {len(value)} items")
                else:
                    print(f"     - {key}: {type(value)}")
                    
    except Exception as e:
        print(f"   ✗ バッチ処理エラー: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== 統合テスト完了 ===")


if __name__ == "__main__":
    test_dataset_model_integration()