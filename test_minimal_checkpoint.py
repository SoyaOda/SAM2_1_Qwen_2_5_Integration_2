#!/usr/bin/env python3
"""
Simplified test for checkpoint save/load between minimal_train.py and test_inference_v2.py
"""

import sys
import torch
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent))

def main():
    print("Testing checkpoint system...")
    
    # 既存のチェックポイントを探す
    outputs_dir = Path("outputs")
    checkpoint_found = False
    
    if outputs_dir.exists():
        for train_dir in sorted(outputs_dir.glob("minimal_train_*"), reverse=True):
            checkpoint_dir = train_dir / "checkpoints"
            if checkpoint_dir.exists():
                for ckpt in checkpoint_dir.glob("*/"):
                    if ckpt.is_dir() and (ckpt / "config.pt").exists():
                        print(f"\nFound checkpoint: {ckpt}")
                        
                        # test_inference_v2でロード試行
                        from test_inference_v2 import InferenceRunnerV2
                        
                        try:
                            print("Loading with InferenceRunnerV2...")
                            runner = InferenceRunnerV2(str(ckpt))
                            
                            # 基本検証
                            assert hasattr(runner.model, 'qwen'), "Missing qwen"
                            assert hasattr(runner.model, 'sam_mask_decoder'), "Missing sam_mask_decoder"
                            assert runner.seg_id is not None, "SEG token ID not set"
                            
                            print(f"✅ Successfully loaded checkpoint!")
                            print(f"   - Model device: {runner.device}")
                            print(f"   - SEG token: {runner.seg_token} (ID: {runner.seg_id})")
                            print(f"   - Config LoRA rank: {runner.config.lora_r}")
                            
                            checkpoint_found = True
                            break
                            
                        except Exception as e:
                            print(f"❌ Failed to load: {e}")
                            return False
                            
                if checkpoint_found:
                    break
    
    if checkpoint_found:
        print("\n" + "="*60)
        print("✅ CHECKPOINT SYSTEM WORKING!")
        print("minimal_train.py saves checkpoints that test_inference_v2.py can load.")
        print("="*60)
        return True
    else:
        print("\n❌ No valid checkpoints found to test.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)