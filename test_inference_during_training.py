#!/usr/bin/env python3
"""
Test inference evaluation during checkpoint saving
"""

import sys
import os
import tempfile
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from minimal_train import MinimalTrainer
from src.config import LISAConfig
from types import SimpleNamespace

def test_inference_eval():
    """推論評価機能のテスト"""
    
    print("Testing inference evaluation during checkpoint saving...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 設定を作成
        config = SimpleNamespace(
            # LISAConfig関連
            lora_r=4,
            lora_alpha=32,
            sam_lora_r=4,
            freeze_qwen_lora=False,
            freeze_seg_token=False,
            freeze_sam_mask_decoder_base=True,
            use_token_fpn=True,
            
            # データセット設定
            data_dir=None,
            samples_per_epoch=10,
            dataset_types='sem_seg||refer_seg||vqa||reason_seg',
            sample_rates='9,3,3,1',
            
            # 訓練設定
            batch_size=1,
            gradient_accumulation_steps=1,
            num_epochs=1,
            warmup_ratio=0.1,
            
            # 学習率設定
            adapter_lr=1e-3,
            lora_lr=1e-4,
            seg_token_lr=5e-5,
            weight_decay=0.01,
            
            # 損失設定
            seg_loss_weight=1.0,
            
            # アライメント設定
            align_steps=0,
            use_cached_alignment=False,
            alignment_checkpoint=None,
            force_realign=False,
            
            # 保存・ログ設定
            save_interval=1,  # すぐに保存してテスト
            visualize=False,
            visualize_steps=5,
            wandb=False,
            
            # 推論評価設定
            run_inference_eval=True,  # 推論評価を有効化
            inference_eval_samples=2,  # 2サンプルのみテスト
            
            # 推論設定
            inference_only=False,
            inference_checkpoint=None,
            checkpoint_dir=None,
            resume_training=False,
        )
        
        # LISAConfigを作成
        lisa_config = LISAConfig(
            qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
            sam_model_name="facebook/sam2.1-hiera-large",
            lora_r=config.lora_r,
            sam_lora_r=config.sam_lora_r,
            freeze_qwen_lora=config.freeze_qwen_lora,
            freeze_seg_token=config.freeze_seg_token,
            freeze_sam_mask_decoder_base=config.freeze_sam_mask_decoder_base,
            use_token_fpn=config.use_token_fpn,
        )
        
        # トレーナーを初期化
        trainer = MinimalTrainer(config)
        trainer.lisa_config = lisa_config
        trainer.output_dir = Path(temp_dir) / "test_output"
        trainer.output_dir.mkdir(parents=True, exist_ok=True)
        trainer.checkpoint_dir = trainer.output_dir / "checkpoints"
        trainer.checkpoint_dir.mkdir(exist_ok=True)
        
        print("\nSetting up model...")
        trainer.setup_model_and_data()
        trainer.setup_optimizer_and_scheduler()
        
        print("\nSaving checkpoint with inference evaluation...")
        trainer.save_checkpoint("test_inference")
        
        # 結果を確認
        checkpoint_path = trainer.checkpoint_dir / "test_inference"
        inference_dir = checkpoint_path / "inference_results"
        
        if inference_dir.exists():
            print(f"\n✅ Inference results saved to: {inference_dir}")
            
            # 保存されたファイルを確認
            for file in inference_dir.glob("*"):
                print(f"  - {file.name}")
            
            # JSONファイルが存在するか確認
            json_file = inference_dir / "evaluation_results.json"
            if json_file.exists():
                import json
                with open(json_file, 'r') as f:
                    results = json.load(f)
                print(f"\n📊 Evaluation results: {len(results)} samples processed")
                for result in results:
                    print(f"  - {result['name']}: mean={result['mask_mean']:.3f}")
            
            return True
        else:
            print(f"\n❌ No inference results found at {inference_dir}")
            return False

if __name__ == "__main__":
    try:
        success = test_inference_eval()
        if success:
            print("\n✅ Inference evaluation during training works correctly!")
            sys.exit(0)
        else:
            print("\n❌ Inference evaluation test failed")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)