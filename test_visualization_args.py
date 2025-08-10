#!/usr/bin/env python3
"""可視化引数のテスト"""

import subprocess
import sys

# 可視化ありでテスト
print("=" * 60)
print("可視化ありでテスト（5ステップごと）")
print("=" * 60)
cmd = [
    sys.executable, "minimal_train.py",
    "--samples_per_epoch", "2",
    "--batch_size", "1",
    "--num_epochs", "1",
    "--fast_dev_run",
    "--visualize",  # 可視化を有効化
    "--visualize_steps", "2"  # 2ステップごとに可視化
]
subprocess.run(cmd)

print("\n" + "=" * 60)
print("可視化なしでテスト")
print("=" * 60)
cmd = [
    sys.executable, "minimal_train.py",
    "--samples_per_epoch", "2",
    "--batch_size", "1",
    "--num_epochs", "1",
    "--fast_dev_run"
    # --visualizeを指定しないので可視化は無効
]
subprocess.run(cmd)