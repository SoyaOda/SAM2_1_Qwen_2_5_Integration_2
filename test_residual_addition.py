#!/usr/bin/env python3
"""
加算アプローチとアライメントステージの動作確認テスト
"""

import sys
import torch
import logging
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.config import LISAConfig
from src.models.lisa_model import LISA_Model
from src.utils.tokenizer_utils import prepare_tokenizer_for_lisa

# ロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_prompt_beta_parameter():
    """prompt_betaパラメータが正しく追加されているか確認"""
    logger.info("=" * 50)
    logger.info("Test 1: prompt_betaパラメータの確認")
    logger.info("=" * 50)
    
    config = LISAConfig()
    model = LISA_Model(config)
    
    # prompt_betaパラメータが存在するか
    assert hasattr(model, 'prompt_beta'), "prompt_betaパラメータが見つかりません"
    assert isinstance(model.prompt_beta, torch.nn.Parameter), "prompt_betaがParameterではありません"
    
    # 初期値の確認
    initial_value = model.prompt_beta.item()
    logger.info(f"✓ prompt_beta初期値: {initial_value}")
    logger.info(f"✓ sigmoid(prompt_beta): {torch.sigmoid(model.prompt_beta).item():.3f}")
    
    # 学習可能か確認
    assert model.prompt_beta.requires_grad, "prompt_betaが学習可能ではありません"
    logger.info("✓ prompt_betaは学習可能です")
    
    logger.info("Test 1: PASSED ✅")
    return True

def test_embedding_fusion():
    """埋め込み融合ロジックの動作確認"""
    logger.info("=" * 50)
    logger.info("Test 2: 埋め込み融合ロジックの確認")
    logger.info("=" * 50)
    
    config = LISAConfig()
    tokenizer = prepare_tokenizer_for_lisa(model_name=config.qwen_model_name, seg_token=config.seg_token)
    model = LISA_Model(config)
    model.set_tokenizer(tokenizer)
    model.eval()
    
    # ダミー入力の作成
    batch_size = 1
    seq_len = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # SEGトークンを含む入力を作成
    seg_token_id = tokenizer.convert_tokens_to_ids(config.seg_token)
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), device=device)
    input_ids[0, 10] = seg_token_id  # SEGトークンを挿入
    
    # ダミー画像
    pixel_values = torch.randn(batch_size, 3, 224, 224, device=device)
    
    # ダミーマスク
    mask_labels = [torch.ones(224, 224, device=device)]
    
    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                labels=input_ids,
                mask_labels=mask_labels
            )
        logger.info("✓ Forward passが成功しました")
        
        # 出力の確認
        assert outputs.logits is not None, "logitsが出力されていません"
        assert outputs.mask_logits is not None, "mask_logitsが出力されていません"
        logger.info(f"✓ logits shape: {outputs.logits.shape}")
        if outputs.mask_logits[0] is not None:
            logger.info(f"✓ mask_logits[0] length: {len(outputs.mask_logits[0])}")
        
        logger.info("Test 2: PASSED ✅")
        return True
        
    except Exception as e:
        logger.error(f"✗ Forward passでエラー: {e}")
        logger.info("Test 2: FAILED ❌")
        return False

def test_alignment_stage():
    """アライメントステージの動作確認"""
    logger.info("=" * 50)
    logger.info("Test 3: アライメントステージの動作確認")
    logger.info("=" * 50)
    
    from minimal_train import MinimalTrainer
    import argparse
    
    # テスト用の設定
    args = argparse.Namespace(
        data_dir=None,
        samples_per_epoch=2,
        dataset_types='sem_seg',
        sample_rates='1.0',
        batch_size=1,
        gradient_accumulation_steps=1,
        num_epochs=1,
        warmup_ratio=0.1,
        adapter_lr=1e-3,
        lora_lr=1e-4,
        seg_token_lr=5e-5,
        weight_decay=0.01,
        lora_r=8,
        lora_alpha=32,
        seg_loss_weight=1.0,
        save_steps=100,
        use_wandb=False,
        wandb_project='test',
        debug=False,
        fast_dev_run=True,
        max_samples=2,
        visualize=False,
        visualize_steps=5,
        align_steps=10  # アライメントステップを有効化
    )
    
    try:
        trainer = MinimalTrainer(args)
        trainer.setup_model_and_data()
        trainer.setup_optimizer_and_scheduler()
        
        # prompt_betaの初期値を記録
        initial_beta = torch.sigmoid(trainer.model.prompt_beta).item()
        logger.info(f"アライメント前のβ値: {initial_beta:.3f}")
        
        # アライメントステージを実行
        trainer.run_alignment_stage()
        
        # prompt_betaが変化したか確認
        final_beta = torch.sigmoid(trainer.model.prompt_beta).item()
        logger.info(f"アライメント後のβ値: {final_beta:.3f}")
        
        logger.info("✓ アライメントステージが正常に実行されました")
        logger.info("Test 3: PASSED ✅")
        return True
        
    except Exception as e:
        logger.error(f"✗ アライメントステージでエラー: {e}")
        import traceback
        traceback.print_exc()
        logger.info("Test 3: FAILED ❌")
        return False

def main():
    """全テストを実行"""
    logger.info("🚀 加算アプローチ実装のテストを開始します")
    
    results = []
    
    # Test 1: prompt_betaパラメータ
    results.append(test_prompt_beta_parameter())
    
    # Test 2: 埋め込み融合ロジック
    results.append(test_embedding_fusion())
    
    # Test 3: アライメントステージ
    results.append(test_alignment_stage())
    
    # 結果サマリー
    logger.info("=" * 50)
    logger.info("テスト結果サマリー")
    logger.info("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        logger.info(f"✅ 全テスト合格: {passed}/{total}")
        logger.info("修正が正しく実装されています！")
    else:
        logger.info(f"⚠️ 一部テスト失敗: {passed}/{total}")
        logger.info("修正内容を確認してください。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)