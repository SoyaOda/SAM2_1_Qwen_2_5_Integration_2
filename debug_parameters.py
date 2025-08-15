#!/usr/bin/env python3
"""
Debug script to identify all trainable parameters
"""
import torch
from src.config import LISAConfig
from src.models.lisa_model import LISA_Model
from src.utils.model_utils import display_parameter_statistics
import logging

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(message)s')

def debug_parameters():
    # Sigma-Add設定でモデルを初期化
    config = LISAConfig(
        fusion_type="sigma_add"
    )
    
    print("\n" + "="*80)
    print("SIGMA-ADD FUSION MODE")
    print("="*80)
    model = LISA_Model(config)
    
    
    # model_utilsの統計表示を使用
    display_parameter_statistics(model)
    
    print("\n" + "="*80)
    print("DETAILED FUSION PARAMETERS:")
    print("="*80)
    
    fusion_params = []
    for name, param in model.named_parameters():
        if 'beta' in name.lower() or 'fusion' in name.lower():
            fusion_params.append((name, param.shape, param.requires_grad, param.numel()))
            print(f"{name:60} shape={param.shape} trainable={param.requires_grad}")
    
    # Cross-Attention設定でモデルを初期化
    print("\n" + "="*80)
    print("CROSS-ATTENTION FUSION MODE")
    print("="*80)
    config = LISAConfig(
        fusion_type="cross_attention"
    )
    model = LISA_Model(config)
    
    # model_utilsの統計表示を使用
    display_parameter_statistics(model)
    
    print("\n" + "="*80)
    print("DETAILED FUSION PARAMETERS:")
    print("="*80)
    
    for name, param in model.named_parameters():
        if 'fusion' in name.lower() or 'cross' in name.lower() or 'gate' in name.lower():
            print(f"{name:60} shape={param.shape} trainable={param.requires_grad}")

if __name__ == "__main__":
    debug_parameters()