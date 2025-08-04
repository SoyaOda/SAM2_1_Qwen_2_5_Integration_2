"""
シンプルなデータセット読み込みテスト
"""
import os
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import LISAConfig
from src.data.vqa_dataset import VQADataset
from transformers import AutoProcessor, AutoTokenizer


def test_vqa_dataset():
    """VQAデータセットの基本的な読み込みテスト"""
    print("=== VQAデータセット読み込みテスト ===\n")
    
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
        print(f"   ✓ SEGトークンID: {seg_token_idx}")
    except Exception as e:
        print(f"   ✗ エラー: {e}")
        return
    
    # VQAデータセットのパスを確認
    print("\n2. VQAデータセットの確認...")
    vqa_path = Path(config.dataset_base_dir) / "llava_dataset" / "llava_instruct_150k.json"
    if vqa_path.exists():
        print(f"   ✓ VQAデータセットを確認: {vqa_path}")
        
        # VQADatasetの初期化
        try:
            vqa_dataset = VQADataset(
                base_image_dir=config.dataset_base_dir,
                image_size=896,  # 元の画像サイズ
                tokenizer=tokenizer,
                precision="fp32",
                samples_per_epoch=1000  # 少ないサンプル数でテスト
            )
            print(f"   ✓ VQADataset初期化成功")
            print(f"   ✓ サンプル数: {len(vqa_dataset)}")
            
            # 最初のサンプルを取得
            if len(vqa_dataset) > 0:
                print("\n3. サンプルデータの取得...")
                try:
                    sample = vqa_dataset[0]
                    print("   ✓ サンプル取得成功")
                    
                    # サンプルの構造を確認
                    if isinstance(sample, tuple) and len(sample) == 5:
                        image_path, image, conversations, _, _ = sample
                        print(f"   - 画像パス: {image_path}")
                        print(f"   - 画像shape: {image.shape if hasattr(image, 'shape') else type(image)}")
                        print(f"   - 会話数: {len(conversations) if conversations else 0}")
                        if conversations:
                            print(f"   - 最初の会話: {conversations[0][:100]}...")
                    else:
                        print(f"   - サンプル形式: {type(sample)}, 要素数: {len(sample) if isinstance(sample, (tuple, list)) else 'N/A'}")
                except Exception as e:
                    print(f"   ✗ サンプル取得エラー: {e}")
                    import traceback
                    traceback.print_exc()
        except Exception as e:
            print(f"   ✗ VQADataset初期化エラー: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"   ✗ VQAデータセットが見つかりません: {vqa_path}")
    
    print("\n=== テスト完了 ===")


if __name__ == "__main__":
    test_vqa_dataset()