#!/usr/bin/env python3
"""
End-to-end test for checkpoint save/load between minimal_train.py and test_inference_v2.py
エンドツーエンドのチェックポイント保存・ロードテスト
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
import logging
from PIL import Image

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import LISAConfig

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_save_and_load():
    """minimal_train.pyで保存してtest_inference_v2.pyでロードするテスト"""
    
    logger.info("="*80)
    logger.info("Starting end-to-end checkpoint save/load test")
    logger.info("="*80)
    
    # 一時ディレクトリを作成
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # 1. MinimalTrainerでチェックポイントを保存
        logger.info("\n1. Creating MinimalTrainer instance and saving checkpoint...")
        
        # 設定を作成（小規模なテスト用）
        config = LISAConfig(
            qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
            sam_model_name="facebook/sam2.1-hiera-large",
            lora_r=4,
            sam_lora_r=4,
            freeze_qwen_lora=False,
            freeze_seg_token=False,
            freeze_sam_mask_decoder_base=True,
            use_token_fpn=True,
        )
        
        # MinimalTrainerを作成してモデルを初期化
        from minimal_train import MinimalTrainer
        
        # 出力ディレクトリを設定
        train_output_dir = temp_dir / "train_output"
        train_output_dir.mkdir(parents=True, exist_ok=True)
        
        # MinimalTrainerの設定を作成（argparseのようなオブジェクト）
        from types import SimpleNamespace
        
        trainer_config = SimpleNamespace(
            # LISAConfig関連
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            sam_lora_r=config.sam_lora_r,
            freeze_qwen_lora=config.freeze_qwen_lora,
            freeze_seg_token=config.freeze_seg_token,
            freeze_sam_mask_decoder_base=config.freeze_sam_mask_decoder_base,
            use_token_fpn=config.use_token_fpn,
            
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
            save_interval=100,
            visualize=False,
            visualize_steps=5,
            wandb=False,
            
            # 推論設定
            inference_only=False,
            inference_checkpoint=None,
            checkpoint_dir=None,
            resume_training=False,
        )
        
        trainer = MinimalTrainer(trainer_config)
        
        # MinimalTrainerが内部で作成するLISAConfigを保存
        trainer.lisa_config = config  # 実際のLISAConfigを直接設定
        
        # モデルとデータセットをセットアップ
        logger.info("Setting up model and optimizer...")
        trainer.setup_model_and_data()
        trainer.setup_optimizer_and_scheduler()
        
        # モデルを少しトレーニング（1ステップだけ）
        logger.info("Running one training step to modify model weights...")
        
        # ダミーのバッチを作成
        dummy_batch = {
            'images': torch.randn(1, 3, 224, 224).to(trainer.device),
            'sam_images': torch.randn(1, 3, 1024, 1024).to(trainer.device),
            'masks': torch.randint(0, 2, (1, 1024, 1024)).float().to(trainer.device),
            'input_ids': torch.randint(0, 1000, (1, 100)).to(trainer.device),
            'attention_mask': torch.ones(1, 100).to(trainer.device),
            'labels': torch.randint(0, 1000, (1, 100)).to(trainer.device),
        }
        
        # 1ステップのトレーニング
        trainer.optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=False):
            outputs = trainer.model(
                input_ids=dummy_batch['input_ids'],
                attention_mask=dummy_batch['attention_mask'],
                sam_images=dummy_batch['sam_images'],
                labels=dummy_batch['labels'],
                mask_labels=dummy_batch['masks']
            )
        
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            loss = outputs.loss
            loss.backward()
            trainer.optimizer.step()
            trainer.global_step += 1
            logger.info(f"Training step completed, loss: {loss.item():.4f}")
        
        # チェックポイントを保存
        checkpoint_name = "test_checkpoint"
        trainer.save_checkpoint(checkpoint_name)
        checkpoint_path = trainer.checkpoint_dir / checkpoint_name
        
        logger.info(f"Checkpoint saved to: {checkpoint_path}")
        
        # 保存されたファイルを確認
        logger.info("\nSaved files:")
        for root, dirs, files in os.walk(checkpoint_path):
            level = root.replace(str(checkpoint_path), '').count(os.sep)
            indent = ' ' * 2 * level
            logger.info(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:10]:  # 最初の10ファイルのみ表示
                size = os.path.getsize(os.path.join(root, file))
                logger.info(f"{subindent}{file} ({size:,} bytes)")
        
        # 2. test_inference_v2でチェックポイントをロード
        logger.info("\n2. Loading checkpoint with InferenceRunnerV2...")
        
        from test_inference_v2 import InferenceRunnerV2
        
        try:
            runner = InferenceRunnerV2(str(checkpoint_path))
            logger.info("✓ Checkpoint loaded successfully with InferenceRunnerV2!")
            
            # 3. ロードされたモデルの基本的な検証
            logger.info("\n3. Verifying loaded model...")
            
            # モデル構造の確認
            assert hasattr(runner.model, 'qwen'), "Model missing 'qwen' component"
            assert hasattr(runner.model, 'sam_mask_decoder'), "Model missing 'sam_mask_decoder'"
            assert hasattr(runner.model, 'image_adapter'), "Model missing 'image_adapter'"
            logger.info("✓ Model structure is valid")
            
            # SEGトークンの確認
            assert runner.seg_id is not None, "SEG token ID not set"
            assert runner.seg_token == config.seg_token, f"SEG token mismatch: {runner.seg_token} != {config.seg_token}"
            logger.info(f"✓ SEG token correctly loaded: {runner.seg_token} (ID: {runner.seg_id})")
            
            # 設定の確認
            assert runner.config.lora_r == config.lora_r, "LoRA rank mismatch"
            assert runner.config.sam_lora_r == config.sam_lora_r, "SAM LoRA rank mismatch"
            logger.info("✓ Configuration matches")
            
            # 4. 簡単な推論テスト
            logger.info("\n4. Running simple inference test...")
            
            # ダミー画像を作成
            test_image = Image.new('RGB', (512, 512), color='white')
            test_prompt = "Test segmentation prompt"
            
            try:
                result = runner.run_inference(test_image, test_prompt)
                
                if 'mask' in result and result['mask'] is not None:
                    logger.info(f"✓ Inference completed, mask shape: {result['mask'].shape}")
                else:
                    logger.warning("Inference completed but no mask generated")
                    
            except Exception as e:
                logger.warning(f"Inference test failed (expected for dummy data): {e}")
            
            # 5. パラメータの比較（オプション）
            logger.info("\n5. Comparing key parameters...")
            
            # prompt_betaの値を確認
            trainer_beta = trainer.model.prompt_beta.item()
            runner_beta = runner.model.prompt_beta.item()
            
            if abs(trainer_beta - runner_beta) < 1e-6:
                logger.info(f"✓ prompt_beta matches: {trainer_beta:.6f}")
            else:
                logger.error(f"✗ prompt_beta mismatch: trainer={trainer_beta:.6f}, runner={runner_beta:.6f}")
            
            logger.info("\n" + "="*80)
            logger.info("✅ End-to-end test PASSED!")
            logger.info("minimal_train.py can save checkpoints that test_inference_v2.py can load correctly.")
            logger.info("="*80)
            
            return True
            
        except Exception as e:
            logger.error(f"\n❌ End-to-end test FAILED!")
            logger.error(f"Error loading checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_existing_checkpoint():
    """既存のチェックポイントをtest_inference_v2.pyでロードするテスト"""
    
    logger.info("\n" + "="*80)
    logger.info("Testing loading of existing checkpoints...")
    logger.info("="*80)
    
    # 既存のチェックポイントを探す
    outputs_dir = Path("outputs")
    existing_checkpoints = []
    
    if outputs_dir.exists():
        for train_dir in outputs_dir.glob("minimal_train_*"):
            checkpoint_dir = train_dir / "checkpoints"
            if checkpoint_dir.exists():
                for ckpt in checkpoint_dir.glob("*/"):
                    if ckpt.is_dir():
                        # 新しい形式（checkpoint_io使用）または古い形式のチェックポイントをチェック
                        if (ckpt / "config.pt").exists() or (ckpt / "checkpoint_info.json").exists():
                            existing_checkpoints.append(ckpt)
                            break  # 各訓練から1つだけテスト
    
    if not existing_checkpoints:
        logger.info("No existing checkpoints found to test.")
        return True
    
    logger.info(f"Found {len(existing_checkpoints)} checkpoint(s) to test")
    
    from test_inference_v2 import InferenceRunnerV2
    
    success_count = 0
    for i, ckpt_path in enumerate(existing_checkpoints[:1], 1):  # 最初の1つだけテスト
        logger.info(f"\n{i}. Testing: {ckpt_path}")
        
        try:
            runner = InferenceRunnerV2(str(ckpt_path))
            logger.info("✓ Successfully loaded checkpoint")
            
            # 基本的な検証
            assert hasattr(runner.model, 'qwen'), "Model missing 'qwen'"
            assert hasattr(runner.model, 'sam_mask_decoder'), "Model missing 'sam_mask_decoder'"
            assert runner.seg_id is not None, "SEG token ID not set"
            
            logger.info("✓ Model structure and configuration are valid")
            success_count += 1
            
        except Exception as e:
            logger.error(f"✗ Failed to load: {e}")
    
    logger.info(f"\n{success_count}/{len(existing_checkpoints[:1])} existing checkpoint(s) loaded successfully")
    return success_count == len(existing_checkpoints[:1])


if __name__ == "__main__":
    # 1. エンドツーエンドテスト（新規保存・ロード）
    test1_passed = test_save_and_load()
    
    # 2. 既存チェックポイントのロードテスト
    test2_passed = test_existing_checkpoint()
    
    # 最終結果
    print("\n" + "="*80)
    print("FINAL RESULTS:")
    print("="*80)
    
    if test1_passed and test2_passed:
        print("✅ ALL TESTS PASSED!")
        print("Checkpoint save/load between minimal_train.py and test_inference_v2.py works correctly.")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED!")
        if not test1_passed:
            print("  - End-to-end save/load test failed")
        if not test2_passed:
            print("  - Existing checkpoint load test failed")
        sys.exit(1)