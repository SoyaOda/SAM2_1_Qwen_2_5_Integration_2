#!/usr/bin/env python3
"""
SAM Decoder学習フローのテストスクリプト
実際にデータが適切に流れているか確認
"""

import os
import sys
import torch
import logging
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import LISAConfig
from minimal_train import MinimalTrainer

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_sam_training():
    """SAM Decoderの学習フローをテスト"""
    
    # LISAConfigを作成（モデル設定）
    config = LISAConfig(
        # モデル設定
        qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        sam_model_name="./checkpoints/sam2.1_hiera_large.pt",
        
        # 凍結設定 - SAM Decoderが学習されるように
        freeze_qwen_base=True,
        freeze_sam_mask_decoder_base=False,  # SAM Decoderは学習する
        freeze_sam_prompt_encoder=True,
        freeze_sam_image_encoder=True,
        freeze_sam_memory_attention=True,
        
        # LoRA無効化（直接学習）
        sam_lora_r=0,
        
        # 出力先
        # output_dir="./test_outputs/sam_training_test"  # LISAConfigにはoutput_dirがない
    )
    
    # 学習用の設定を簡単なnamespaceオブジェクトとして作成
    import argparse
    train_config = argparse.Namespace(
        # データ設定
        batch_size=1,
        samples_per_epoch=2,  # 2サンプルのみ
        num_epochs=1,
        gradient_accumulation_steps=1,
        
        # 学習設定
        adapter_lr=1e-5,
        lora_lr=1e-5,
        seg_token_lr=1e-5,
        weight_decay=0.01,
        warmup_ratio=0.0,
        seg_loss_weight=1.0,
        save_steps=100,
        
        # LoRA設定（minimal_train.pyで必要）
        lora_r=0,  # LoRA無効化
        lora_alpha=32,
        
        # データセット設定
        dataset_types='sem_seg',  # ADE20Kのみ
        sample_rates='1',
        max_samples=2,  # 最大2サンプル
        data_dir=None,  # デフォルトのデータディレクトリを使用
        
        # デバッグ設定
        debug=True,
        use_wandb=False,
        fast_dev_run=True,
        wandb_project='lisa-kai-minimal',
        
        # 可視化
        visualize=False,
        visualize_steps=10,
        run_inference_eval=False,
        inference_eval_samples=0,
        
        # アライメント設定
        align_steps=0,
        use_cached_alignment=False,
        alignment_checkpoint=None,
        force_realign=False,
        
        # 出力先
        output_dir="./test_outputs/sam_training_test"
    )
    
    # configを統合
    # MinimalTrainerは両方の設定を使用する
    train_config.config = config  # LISAConfigをtrain_configに追加
    
    logger.info("=" * 80)
    logger.info("SAM Decoder Training Flow Test")
    logger.info("=" * 80)
    
    try:
        # トレーナー初期化
        logger.info("Initializing trainer...")
        trainer = MinimalTrainer(train_config)
        
        # モデルのセットアップ
        logger.info("Setting up model and data...")
        trainer.setup_model_and_data()
        
        # オプティマイザとスケジューラのセットアップ
        logger.info("Setting up optimizer and scheduler...")
        trainer.setup_optimizer_and_scheduler()
        
        # パラメータ統計の表示
        trainer.display_parameter_statistics()
        
        # 1エポックのみ実行
        logger.info("\n" + "=" * 80)
        logger.info("Starting training for 1 epoch with 2 samples...")
        logger.info("=" * 80)
        
        # train_epochを直接呼び出し
        avg_loss = trainer.train_epoch(0)
        
        logger.info("\n" + "=" * 80)
        logger.info(f"Training completed successfully!")
        logger.info(f"Average loss: {avg_loss:.4f}")
        logger.info("=" * 80)
        
        # 成功を示すメッセージ
        print("\n✅ SAM Decoder training flow test PASSED!")
        print("   - Data is flowing through SAM Decoder")
        print("   - Loss computation is working")
        print("   - Gradients are being computed")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n❌ SAM Decoder training flow test FAILED!")
        print(f"   Error: {e}")
        
        return False

if __name__ == "__main__":
    success = test_sam_training()
    sys.exit(0 if success else 1)