#!/usr/bin/env python3
"""
簡易版：Test A2でSAM ImageEncoderが呼ばれているかを確認
"""

import os
import sys
import torch
import logging
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# ログレベルを設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 特定のロガーをDEBUGレベルに設定
lisa_logger = logging.getLogger('src.models.lisa_model')
lisa_logger.setLevel(logging.DEBUG)

def check_sam_usage_in_test_a2():
    """Test A2の実際のコードをシミュレートして、SAM ImageEncoderが呼ばれているか確認"""
    
    # Test A2の実際の実行ログを検証
    logger.info("=== Checking SAM usage in Test A2 ===")
    
    # test_a2のコードの問題を特定
    logger.info("Test A2のコードをレビューします...")
    
    # test_a2_unfreeze_sam.pyの重要部分を確認
    with open('test_a2_unfreeze_sam.py', 'r') as f:
        content = f.read()
    
    # SAM画像生成のコードが含まれているか確認
    has_sam_generation = 'sam_images' in content and 'sam_image_size' in content
    logger.info(f"✓ SAM画像生成コード存在: {has_sam_generation}")
    
    # forwardメソッドにsam_imagesが渡されているか確認
    has_sam_forward = 'sam_images=sam_images_batch' in content
    logger.info(f"✓ forwardにSAM画像を渡すコード存在: {has_sam_forward}")
    
    # LISA_Modelのforwardメソッドを確認
    logger.info("LISA_Modelのforwardメソッドを確認...")
    
    with open('src/models/lisa_model.py', 'r') as f:
        lisa_content = f.read()
    
    # SAM ImageEncoderを呼び出すコードが存在するか
    has_sam_encoder_call = 'self.sam_image_encoder(' in lisa_content
    logger.info(f"✓ SAM ImageEncoder呼び出しコード存在: {has_sam_encoder_call}")
    
    # SAM画像の条件チェック
    has_sam_condition = 'if sam_images is not None:' in lisa_content
    logger.info(f"✓ SAM画像条件チェック存在: {has_sam_condition}")
    
    # デバッグログの存在確認
    has_debug_logs = '[SAM ViT]' in lisa_content
    logger.info(f"✓ SAMデバッグログ存在: {has_debug_logs}")
    
    # 問題の特定
    logger.info("\n=== 問題の特定 ===")
    
    if not (has_sam_generation and has_sam_forward and has_sam_encoder_call and has_sam_condition):
        logger.error("❌ 基本的なSAM統合コードに問題があります")
        return False
    
    # Test A2の出力結果から問題を推測
    logger.info("Test A2の結果分析:")
    logger.info("- マスク生成精度が悪い = SAM ImageEncoderが使われていない可能性")
    logger.info("- 考えられる原因:")
    logger.info("  1. SAM画像が正しく生成されていない")
    logger.info("  2. forwardメソッドでSAM分岐に入っていない")
    logger.info("  3. SAM ImageEncoderの初期化に問題がある")
    logger.info("  4. debug logsが出力されていない = SAM処理が実行されていない")
    
    return True

def propose_debug_solution():
    """デバッグソリューションを提案"""
    logger.info("\n=== デバッグソリューション ===")
    
    solutions = [
        "1. test_a2実行時のログ出力を詳細にチェック",
        "2. LISA_Modelの初期化でSAM ImageEncoderが正しく設定されているか確認",
        "3. forwardメソッドでSAM画像が実際に渡されているか確認",
        "4. [SAM ViT]のデバッグログが出力されているか確認",
        "5. SAM画像のデータ型・サイズ・デバイスが適切か確認"
    ]
    
    for solution in solutions:
        logger.info(solution)
    
    logger.info("\n=== 推奨アクション ===")
    logger.info("A. test_a2にデバッグログを追加して再実行")
    logger.info("B. LISA_Modelの初期化段階でSAM ImageEncoderの存在を確認")
    logger.info("C. forwardメソッドの先頭でSAM画像の受信を確認")

if __name__ == "__main__":
    success = check_sam_usage_in_test_a2()
    if success:
        propose_debug_solution()