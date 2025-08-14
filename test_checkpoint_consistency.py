#!/usr/bin/env python3
"""
Test checkpoint save/load consistency
チェックポイント保存・ロード機能の整合性テスト
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any
import tempfile
import shutil
import logging

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import LISAConfig
from src.models.lisa_model import LISA_Model
from src.utils.checkpoint_io import save_lisa_checkpoint, load_lisa_checkpoint
from transformers import AutoProcessor

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_model_and_processor():
    """テスト用のモデルとプロセッサを作成"""
    logger.info("Creating test model and processor...")
    
    # 設定を作成
    config = LISAConfig(
        qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        sam_model_name="facebook/sam2.1-hiera-large",
        lora_r=4,  # 小さなLoRAランクでテスト
        sam_lora_r=4,
        freeze_qwen_lora=False,  # LoRAを学習可能にする
        freeze_seg_token=False,  # SEGトークンを学習可能にする
        freeze_sam_mask_decoder_base=True,  # SAM LoRAを有効にする
        use_token_fpn=True,  # Token-FPNを使用
    )
    
    # プロセッサを初期化
    processor = AutoProcessor.from_pretrained(
        config.qwen_model_name,
        min_pixels=224*224,
        max_pixels=1024*1024,
        max_length=8192
    )
    
    # SEGトークンを追加
    if config.seg_token not in processor.tokenizer.get_vocab():
        processor.tokenizer.add_tokens([config.seg_token], special_tokens=True)
    
    # モデルを初期化（小規模なテストのため軽量化）
    logger.info("Initializing LISA model...")
    model = LISA_Model(config)
    model.set_tokenizer(processor.tokenizer, seg_token=config.seg_token)
    
    return model, processor, config


def compare_model_states(model1: LISA_Model, model2: LISA_Model) -> Dict[str, Any]:
    """2つのモデルの状態を比較"""
    differences = []
    
    # 主要コンポーネントの重みを比較
    components_to_check = [
        ('image_adapter', model1.image_adapter, model2.image_adapter),
        ('text_prompt_proj', model1.text_prompt_proj, model2.text_prompt_proj),
        ('prompt_beta', model1.prompt_beta, model2.prompt_beta),
    ]
    
    # Token-FPNまたはHighResGeneratorをチェック
    if hasattr(model1, 'token_fpn') and hasattr(model2, 'token_fpn'):
        components_to_check.append(('token_fpn', model1.token_fpn, model2.token_fpn))
    elif hasattr(model1, 'high_res_generator') and hasattr(model2, 'high_res_generator'):
        components_to_check.append(('high_res_generator', model1.high_res_generator, model2.high_res_generator))
    
    # SEGトークン埋め込みをチェック
    if hasattr(model1, 'seg_token_embedding') and hasattr(model2, 'seg_token_embedding'):
        if model1.seg_token_embedding is not None and model2.seg_token_embedding is not None:
            components_to_check.append(('seg_token_embedding', model1.seg_token_embedding, model2.seg_token_embedding))
    
    for name, comp1, comp2 in components_to_check:
        logger.info(f"Checking {name}...")
        
        # パラメータの場合
        if isinstance(comp1, torch.nn.Parameter) and isinstance(comp2, torch.nn.Parameter):
            # 同じデバイスに移動してから比較
            comp1_cpu = comp1.detach().cpu()
            comp2_cpu = comp2.detach().cpu()
            if not torch.allclose(comp1_cpu, comp2_cpu, atol=1e-6):
                differences.append(f"{name}: Parameter values differ")
                max_diff = torch.max(torch.abs(comp1_cpu - comp2_cpu)).item()
                differences.append(f"  Max difference: {max_diff}")
        # モジュールの場合
        else:
            state1 = comp1.state_dict()
            state2 = comp2.state_dict()
            
            if set(state1.keys()) != set(state2.keys()):
                differences.append(f"{name}: State dict keys differ")
                differences.append(f"  Keys in model1 but not model2: {set(state1.keys()) - set(state2.keys())}")
                differences.append(f"  Keys in model2 but not model1: {set(state2.keys()) - set(state1.keys())}")
            else:
                for key in state1.keys():
                    # 同じデバイスに移動してから比較
                    tensor1_cpu = state1[key].detach().cpu()
                    tensor2_cpu = state2[key].detach().cpu()
                    if not torch.allclose(tensor1_cpu, tensor2_cpu, atol=1e-6):
                        differences.append(f"{name}.{key}: Values differ")
                        max_diff = torch.max(torch.abs(tensor1_cpu - tensor2_cpu)).item()
                        differences.append(f"  Max difference: {max_diff}")
    
    # SAM LoRAの重みをチェック
    logger.info("Checking SAM LoRA weights...")
    sam_lora_params1 = {}
    sam_lora_params2 = {}
    
    for name, param in model1.sam_mask_decoder.named_parameters():
        if 'lora_' in name.lower():
            sam_lora_params1[name] = param
    
    for name, param in model2.sam_mask_decoder.named_parameters():
        if 'lora_' in name.lower():
            sam_lora_params2[name] = param
    
    if sam_lora_params1 or sam_lora_params2:
        if set(sam_lora_params1.keys()) != set(sam_lora_params2.keys()):
            differences.append("SAM LoRA: Parameter keys differ")
        else:
            for key in sam_lora_params1.keys():
                # 同じデバイスに移動してから比較
                param1_cpu = sam_lora_params1[key].detach().cpu()
                param2_cpu = sam_lora_params2[key].detach().cpu()
                if not torch.allclose(param1_cpu, param2_cpu, atol=1e-6):
                    differences.append(f"SAM LoRA.{key}: Values differ")
    
    return {
        'has_differences': len(differences) > 0,
        'differences': differences,
        'num_differences': len(differences)
    }


def test_checkpoint_save_load():
    """チェックポイント保存・ロードのテスト"""
    logger.info("="*80)
    logger.info("Starting checkpoint save/load test")
    logger.info("="*80)
    
    # 一時ディレクトリを作成
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "test_checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. テストモデルを作成
        logger.info("\n1. Creating original model...")
        model1, processor, config = create_test_model_and_processor()
        
        # 一部の重みをランダムに変更（テスト用）
        logger.info("Modifying some weights for testing...")
        with torch.no_grad():
            # prompt_betaを変更
            model1.prompt_beta.data = torch.tensor(0.5)
            
            # image_adapterの一部を変更
            for param in model1.image_adapter.parameters():
                param.data += torch.randn_like(param) * 0.01
                break  # 最初のパラメータのみ変更
        
        # 2. チェックポイントを保存
        logger.info(f"\n2. Saving checkpoint to {checkpoint_dir}...")
        
        # ダミーのオプティマイザとスケジューラを作成
        optimizer = torch.optim.AdamW(model1.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        
        save_lisa_checkpoint(
            checkpoint_dir=str(checkpoint_dir),
            model=model1,
            processor=processor,
            config=config,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=1000,
            best_loss=0.5,
            epoch=5,
            additional_info={'test': 'checkpoint_consistency'}
        )
        
        # 保存されたファイルをリスト
        logger.info("\nSaved files:")
        for root, dirs, files in os.walk(checkpoint_dir):
            level = root.replace(str(checkpoint_dir), '').count(os.sep)
            indent = ' ' * 2 * level
            logger.info(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                size = os.path.getsize(os.path.join(root, file))
                logger.info(f"{subindent}{file} ({size:,} bytes)")
        
        # 3. チェックポイントをロード
        logger.info(f"\n3. Loading checkpoint from {checkpoint_dir}...")
        result = load_lisa_checkpoint(
            checkpoint_dir=str(checkpoint_dir),
            device='cpu'  # CPUでテスト
        )
        
        model2 = result['model']
        processor2 = result['processor']
        config2 = result['config']
        training_state = result['training_state']
        
        # 4. モデルの状態を比較
        logger.info("\n4. Comparing model states...")
        comparison = compare_model_states(model1, model2)
        
        if comparison['has_differences']:
            logger.error(f"Found {comparison['num_differences']} differences:")
            for diff in comparison['differences']:
                logger.error(f"  - {diff}")
        else:
            logger.info("✓ All model states match perfectly!")
        
        # 5. 設定を比較
        logger.info("\n5. Comparing configurations...")
        config1_dict = vars(config)
        config2_dict = vars(config2)
        
        config_diffs = []
        for key in config1_dict.keys():
            if key not in config2_dict:
                config_diffs.append(f"Key '{key}' missing in loaded config")
            elif config1_dict[key] != config2_dict[key]:
                config_diffs.append(f"Key '{key}': {config1_dict[key]} != {config2_dict[key]}")
        
        if config_diffs:
            logger.error(f"Config differences found:")
            for diff in config_diffs:
                logger.error(f"  - {diff}")
        else:
            logger.info("✓ Configurations match!")
        
        # 6. トレーニング状態を確認
        logger.info("\n6. Checking training state...")
        logger.info(f"  Global step: {training_state.get('global_step', 'N/A')}")
        logger.info(f"  Best loss: {training_state.get('best_loss', 'N/A')}")
        logger.info(f"  Epoch: {training_state.get('epoch', 'N/A')}")
        
        # 7. プロセッサのトークナイザを確認
        logger.info("\n7. Checking processor/tokenizer...")
        seg_token_id1 = processor.tokenizer.convert_tokens_to_ids(config.seg_token)
        seg_token_id2 = processor2.tokenizer.convert_tokens_to_ids(config.seg_token)
        
        if seg_token_id1 == seg_token_id2:
            logger.info(f"✓ SEG token ID matches: {seg_token_id1}")
        else:
            logger.error(f"SEG token ID mismatch: {seg_token_id1} != {seg_token_id2}")
        
        # 8. サマリー
        logger.info("\n" + "="*80)
        logger.info("Test Summary:")
        
        all_passed = (
            not comparison['has_differences'] and
            not config_diffs and
            seg_token_id1 == seg_token_id2 and
            training_state.get('global_step') == 1000
        )
        
        if all_passed:
            logger.info("✅ All tests PASSED! Checkpoint save/load is working correctly.")
        else:
            logger.error("❌ Some tests FAILED. Please check the errors above.")
        
        logger.info("="*80)
        
        return all_passed


def test_backward_compatibility():
    """既存のチェックポイントとの互換性をテスト"""
    logger.info("\n" + "="*80)
    logger.info("Testing backward compatibility with existing checkpoints...")
    logger.info("="*80)
    
    # 既存のチェックポイントディレクトリを探す
    existing_checkpoints = []
    outputs_dir = Path("outputs")
    
    if outputs_dir.exists():
        for train_dir in outputs_dir.glob("minimal_train_*"):
            checkpoint_dir = train_dir / "checkpoints"
            if checkpoint_dir.exists():
                for ckpt in checkpoint_dir.glob("*/"):
                    if ckpt.is_dir() and (ckpt / "config.pt").exists():
                        existing_checkpoints.append(ckpt)
    
    if not existing_checkpoints:
        logger.info("No existing checkpoints found to test.")
        return True
    
    logger.info(f"Found {len(existing_checkpoints)} existing checkpoints to test:")
    for ckpt in existing_checkpoints[:3]:  # 最初の3つのみテスト
        logger.info(f"  - {ckpt}")
    
    # 各チェックポイントをロード
    success_count = 0
    for i, ckpt_dir in enumerate(existing_checkpoints[:3], 1):
        logger.info(f"\n{i}. Testing: {ckpt_dir}")
        
        try:
            # 新しい方法でロード
            result = load_lisa_checkpoint(
                checkpoint_dir=str(ckpt_dir),
                device='cpu',
                strict=False  # 互換性のため厳密チェックを無効化
            )
            
            model = result['model']
            logger.info(f"  ✓ Successfully loaded checkpoint")
            success_count += 1
            
            # 基本的な検証
            assert hasattr(model, 'qwen'), "Model missing 'qwen' component"
            assert hasattr(model, 'sam_mask_decoder'), "Model missing 'sam_mask_decoder'"
            assert hasattr(model, 'image_adapter'), "Model missing 'image_adapter'"
            logger.info(f"  ✓ Model structure is valid")
            
        except Exception as e:
            logger.error(f"  ✗ Failed to load: {e}")
    
    logger.info(f"\nBackward compatibility: {success_count}/{min(3, len(existing_checkpoints))} checkpoints loaded successfully")
    return success_count == min(3, len(existing_checkpoints))


if __name__ == "__main__":
    # メインテストを実行
    logger.info("Running checkpoint consistency tests...\n")
    
    # 1. 保存・ロードの整合性テスト
    test1_passed = test_checkpoint_save_load()
    
    # 2. 既存チェックポイントとの互換性テスト
    test2_passed = test_backward_compatibility()
    
    # 最終結果
    logger.info("\n" + "="*80)
    logger.info("FINAL RESULTS:")
    logger.info("="*80)
    
    if test1_passed and test2_passed:
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("The checkpoint save/load system is working correctly.")
        sys.exit(0)
    else:
        logger.error("❌ SOME TESTS FAILED!")
        if not test1_passed:
            logger.error("  - Save/Load consistency test failed")
        if not test2_passed:
            logger.error("  - Backward compatibility test failed")
        sys.exit(1)