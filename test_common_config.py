#!/usr/bin/env python3
"""
テストスクリプト共通設定
すべてのテストで同じデータセット・パラメータを使用
"""

import torch
import numpy as np
import random

# ========== 共通パラメータ ==========
# シード値（固定）
SEED = 42

# 学習パラメータ
NUM_STEPS = 100  # 総ステップ数
BATCH_SIZE = 2    # バッチサイズ
LEARNING_RATE = 1e-4  # 学習率
NUM_FIXED_SAMPLES = 10  # 固定サンプル数（各テストで同じ100個）

# データセット設定
DATASET_TYPE = "sem_seg"  # 使用するデータセットタイプ
SAMPLE_RATE = [1.0]  # サンプリング比率

# 可視化設定
VIS_INTERVAL = 10  # 可視化を保存する間隔（ステップ）
LOG_INTERVAL = 10  # ログ出力間隔

# 評価設定
EVAL_THRESHOLD = 0.5  # seg loss改善の判定閾値（初期値の何倍か）


def set_random_seed(seed=SEED):
    """全ての乱数シードを固定"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic設定
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


class FixedSampleDataset(torch.utils.data.Dataset):
    """固定サンプルデータセット（全テスト共通）"""
    
    def __init__(self, base_dataset, num_samples=NUM_FIXED_SAMPLES, seed=SEED):
        self.base_dataset = base_dataset
        self.num_samples = num_samples
        
        # シード固定してサンプルを取得
        set_random_seed(seed)
        
        # 固定サンプルを取得
        self.fixed_samples = []
        print(f"固定サンプルを{num_samples}個準備中（seed={seed}）...")
        for i in range(num_samples):
            # base_datasetのランダム性も制御するため、
            # 各サンプル取得前にもシードを再設定
            np.random.seed(seed + i)
            random.seed(seed + i)
            sample = base_dataset[i]
            self.fixed_samples.append(sample)
            if (i + 1) % 10 == 0:
                print(f"  サンプル{i+1}/{num_samples}を取得")
        print(f"✅ {num_samples}個の固定サンプル準備完了")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 固定サンプルを順番に返す
        return self.fixed_samples[idx % self.num_samples]


def save_test_config(output_dir, test_name, additional_config=None):
    """テスト設定を保存"""
    import json
    from pathlib import Path
    
    config = {
        'test_name': test_name,
        'seed': SEED,
        'num_steps': NUM_STEPS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'num_fixed_samples': NUM_FIXED_SAMPLES,
        'dataset_type': DATASET_TYPE,
        'sample_rate': SAMPLE_RATE,
        'vis_interval': VIS_INTERVAL,
        'log_interval': LOG_INTERVAL,
        'eval_threshold': EVAL_THRESHOLD,
    }
    
    # 追加の設定があれば追加
    if additional_config:
        config.update(additional_config)
    
    # 保存
    config_path = Path(output_dir) / 'test_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Test config saved to {config_path}")
    return config


def create_standard_output_structure(output_dir):
    """標準的な出力ディレクトリ構造を作成"""
    from pathlib import Path
    
    output_dir = Path(output_dir)
    
    # サブディレクトリを作成
    (output_dir / 'visualizations').mkdir(parents=True, exist_ok=True)
    (output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (output_dir / 'logs').mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory structure created at {output_dir}")
    return output_dir