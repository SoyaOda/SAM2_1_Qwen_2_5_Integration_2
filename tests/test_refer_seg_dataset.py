"""
Referring Segmentationデータセット単体テスト
"""
import os
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import LISAConfig
from src.data.refer_seg_dataset import ReferSegDataset
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings("ignore")


def test_refer_seg_dataset():
    """Referring Segmentationデータセットの単体テスト"""
    print("=== Referring Segmentationデータセットテスト ===\n")
    
    # 設定
    config = LISAConfig()
    
    # トークナイザの準備
    print("1. トークナイザの準備...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.qwen_model_name)
        
        # SEGトークンの追加
        if config.seg_token not in tokenizer.get_vocab():
            tokenizer.add_tokens([config.seg_token])
        seg_token_idx = tokenizer.convert_tokens_to_ids(config.seg_token)
        print(f"   ✓ SEGトークンID: {seg_token_idx}")
    except Exception as e:
        print(f"   ✗ エラー: {e}")
        return
    
    # ReferSegDatasetの初期化
    print("\n2. ReferSegDatasetの初期化...")
    try:
        refer_seg_dataset = ReferSegDataset(
            base_image_dir=config.dataset_base_dir,
            tokenizer=tokenizer,
            vision_tower=None,
            samples_per_epoch=10,
            precision="fp32",
            image_size=config.qwen_image_size,
            num_classes_per_sample=3,
            exclude_val=False,
            refer_seg_data="refcoco"  # RefCOCOのみでテスト
        )
        print(f"   ✓ データセット初期化成功")
        print(f"   ✓ 有効なデータセット: {refer_seg_dataset.refer_seg_datas}")
    except Exception as e:
        print(f"   ✗ 初期化エラー: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # サンプル取得テスト
    print("\n3. サンプル取得テスト...")
    try:
        sample = refer_seg_dataset[0]
        print(f"   ✓ サンプル取得成功")
        print(f"   - サンプル要素数: {len(sample)}")
        
        # 各要素の確認
        print("\n   サンプル内容:")
        print(f"   - 画像パス: {sample[0]}")
        print(f"   - SAM画像shape: {sample[1].shape if hasattr(sample[1], 'shape') else type(sample[1])}")
        print(f"   - Qwen画像shape: {sample[2].shape if hasattr(sample[2], 'shape') else type(sample[2])}")
        print(f"   - 会話数: {len(sample[3]) if sample[3] else 0}")
        print(f"   - マスクshape: {sample[4].shape if hasattr(sample[4], 'shape') else type(sample[4])}")
        print(f"   - ラベルshape: {sample[5].shape if hasattr(sample[5], 'shape') else type(sample[5])}")
        print(f"   - リサイズ情報: {sample[6]}")
        print(f"   - 質問数: {len(sample[7]) if sample[7] else 0}")
        print(f"   - クラス数: {len(sample[8]) if sample[8] else 0}")
    except Exception as e:
        print(f"   ✗ サンプル取得エラー: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== テスト完了 ===")


if __name__ == "__main__":
    test_refer_seg_dataset()