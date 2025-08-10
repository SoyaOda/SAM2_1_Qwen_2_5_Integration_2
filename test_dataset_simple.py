#!/usr/bin/env python3
"""
データセット初期化の問題を特定するシンプルなテストスクリプト
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import LISAConfig

def test_dataset_path():
    """データセットパスが正しいか確認"""
    
    lisa_config = LISAConfig()
    print(f"Dataset base directory: {lisa_config.dataset_base_dir}")
    
    # パスが存在するか確認
    base_path = Path(lisa_config.dataset_base_dir)
    if base_path.exists():
        print(f"✓ Directory exists: {base_path}")
        
        # サブディレクトリを確認
        subdirs = ['ade20k', 'cocostuff', 'mapillary', 'coco', 'vlpart']
        for subdir in subdirs:
            subdir_path = base_path / subdir
            if subdir_path.exists():
                print(f"  ✓ {subdir} exists")
            else:
                print(f"  ✗ {subdir} NOT found")
    else:
        print(f"✗ Directory NOT found: {base_path}")
        print("\n代替パスを探索中...")
        
        # 代替パスを試す
        alt_paths = [
            Path("/home/oda/VLM_RAG_MultiModal_Dataset"),
            Path("/mnt/h/download/LISA-dataset/data/dataset"),
            Path("./data/dataset"),
        ]
        
        for alt_path in alt_paths:
            if alt_path.exists():
                print(f"  Found: {alt_path}")
                # サブディレクトリを確認
                subdirs = ['ade20k', 'cocostuff', 'mapillary', 'coco', 'vlpart']
                existing_subdirs = []
                for subdir in subdirs:
                    if (alt_path / subdir).exists():
                        existing_subdirs.append(subdir)
                if existing_subdirs:
                    print(f"    Available datasets: {', '.join(existing_subdirs)}")

if __name__ == "__main__":
    test_dataset_path()