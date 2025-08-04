"""
全データセットタイプの統合テスト
RefCOCO, ADE20K, ReasonSeg, VQAを含む包括的なテスト
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
from transformers import AutoProcessor, AutoTokenizer
import warnings
warnings.filterwarnings("ignore")


def test_all_datasets():
    """全データセットタイプのテスト"""
    print("=== 全データセット統合テスト ===\n")
    
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
        seg_token_idx = tokenizer.convert_tokens_to_ids(config.seg_token)
        print(f"   ✓ SEGトークンID: {seg_token_idx}")
    except Exception as e:
        print(f"   ✗ エラー: {e}")
        return
    
    # 各データセットタイプをテスト
    dataset_configs = [
        {
            "name": "Semantic Segmentation (ADE20K)",
            "dataset": "sem_seg",
            "sample_rate": [1],
            "sem_seg_data": config.sem_seg_data,
            "samples_per_epoch": 5
        },
        {
            "name": "Referring Segmentation (RefCOCO)",
            "dataset": "refer_seg",
            "sample_rate": [1],
            "refer_seg_data": config.refer_seg_data,
            "samples_per_epoch": 5
        },
        {
            "name": "VQA",
            "dataset": "vqa",
            "sample_rate": [1],
            "vqa_data": config.vqa_data,
            "samples_per_epoch": 5
        },
        {
            "name": "Reasoning Segmentation",
            "dataset": "reason_seg",
            "sample_rate": [1],
            "reason_seg_data": config.reason_seg_data,
            "samples_per_epoch": 5
        },
        {
            "name": "Mixed Dataset (All Types)",
            "dataset": "sem_seg||refer_seg||vqa||reason_seg",
            "sample_rate": [3, 3, 3, 1],
            "sem_seg_data": config.sem_seg_data,
            "refer_seg_data": config.refer_seg_data,
            "vqa_data": config.vqa_data,
            "reason_seg_data": config.reason_seg_data,
            "samples_per_epoch": 10
        }
    ]
    
    for i, ds_config in enumerate(dataset_configs):
        print(f"\n{i+2}. {ds_config['name']}のテスト...")
        
        try:
            # データセット固有の引数を準備
            kwargs = {
                "base_image_dir": config.dataset_base_dir,
                "qwen_processor": processor,
                "samples_per_epoch": ds_config["samples_per_epoch"],
                "precision": "fp32",
                "qwen_image_size": config.qwen_image_size,
                "sam_image_size": config.sam_image_size,
                "exclude_val": False,
                "dataset": ds_config["dataset"],
                "sample_rate": ds_config["sample_rate"]
            }
            
            # データセット固有のパラメータを追加
            if "sem_seg_data" in ds_config:
                kwargs["sem_seg_data"] = ds_config["sem_seg_data"]
            if "refer_seg_data" in ds_config:
                kwargs["refer_seg_data"] = ds_config["refer_seg_data"]
            if "vqa_data" in ds_config:
                kwargs["vqa_data"] = ds_config["vqa_data"]
            if "reason_seg_data" in ds_config:
                kwargs["reason_seg_data"] = ds_config["reason_seg_data"]
                
            # HybridDatasetの作成
            hybrid_dataset = HybridDataset(**kwargs)
            
            print(f"   ✓ データセット作成成功")
            print(f"   - データセット数: {len(hybrid_dataset.all_datasets)}")
            print(f"   - 総サンプル数: {len(hybrid_dataset)}")
            
            # 最初のサンプルを取得してテスト
            if len(hybrid_dataset) > 0:
                try:
                    sample = hybrid_dataset[0]
                    print(f"   ✓ サンプル取得成功")
                    
                    # 必須フィールドの確認
                    required_fields = ['input_ids', 'labels', 'attention_mask', 'pixel_values', 
                                     'sam_images', 'ground_truth_mask']
                    missing_fields = []
                    for field in required_fields:
                        if field not in sample:
                            missing_fields.append(field)
                    
                    if missing_fields:
                        print(f"   ⚠ 不足フィールド: {missing_fields}")
                    else:
                        print(f"   ✓ 全必須フィールドが存在")
                        
                    # image_grid_thwの確認
                    if 'image_grid_thw' in sample and sample['image_grid_thw'] is not None:
                        print(f"   ✓ image_grid_thw: {sample['image_grid_thw'].shape}")
                    else:
                        print(f"   ⚠ image_grid_thwが不足")
                        
                except Exception as e:
                    print(f"   ✗ サンプル取得エラー: {e}")
                    import traceback
                    traceback.print_exc()
                    
        except Exception as e:
            print(f"   ✗ データセット作成エラー: {e}")
            import traceback
            traceback.print_exc()
    
    # DataLoaderでのバッチ処理テスト
    print("\n8. DataLoaderバッチ処理テスト（混合データセット）...")
    try:
        from torch.utils.data import DataLoader
        from src.data.collate_fn import get_collate_fn
        
        # 混合データセットでテスト
        hybrid_dataset = HybridDataset(
            base_image_dir=config.dataset_base_dir,
            qwen_processor=processor,
            samples_per_epoch=8,
            precision="fp32",
            qwen_image_size=config.qwen_image_size,
            sam_image_size=config.sam_image_size,
            exclude_val=False,
            dataset="sem_seg||refer_seg||vqa",
            sample_rate=[1, 1, 1],
            sem_seg_data=config.sem_seg_data,
            refer_seg_data=config.refer_seg_data,
            vqa_data=config.vqa_data
        )
        
        collate_fn = get_collate_fn(tokenizer)
        dataloader = DataLoader(
            hybrid_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        # 最初のバッチを取得
        for batch in dataloader:
            print("   ✓ バッチ取得成功")
            print(f"   - バッチサイズ: {batch['input_ids'].shape[0]}")
            print(f"   - input_ids shape: {batch['input_ids'].shape}")
            print(f"   - pixel_values shape: {batch['pixel_values'].shape}")
            print(f"   - sam_images shape: {batch['sam_images'].shape}")
            print(f"   - image_grid_thw shape: {batch['image_grid_thw'].shape}")
            break
            
    except Exception as e:
        print(f"   ✗ バッチ処理エラー: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== テスト完了 ===")


if __name__ == "__main__":
    test_all_datasets()