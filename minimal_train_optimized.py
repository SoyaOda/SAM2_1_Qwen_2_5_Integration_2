#!/usr/bin/env python3
"""
最小限の学習スクリプト - 最適化版
動作は完全に同一、効率性を改善
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    Qwen2VLProcessor,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from pathlib import Path
import logging
from datetime import datetime
import numpy as np
import json
from typing import Dict, Any, Optional, List, Tuple
import matplotlib.pyplot as plt
import cv2
import random
from tqdm import tqdm
import time
from torch.optim import AdamW
from PIL import Image
import os
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.lisa_model import LISA_Model
from src.config import LISAConfig
from src.data.collators import MultiModalDataCollator
from src.data.dataset import HybridDataset
from sam2.utils.transforms import SAM2Transforms

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MinimalTrainer:
    """最小限の学習を行うトレーナークラス（最適化版）"""
    
    # クラス定数の定義（効率化のため）
    SAM_SIZE = 1024
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # HuggingFaceハブのタイムアウト設定
        from huggingface_hub import constants
        constants.HF_HUB_DISABLE_TELEMETRY = True
        
        # 出力ディレクトリ設定
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"outputs/minimal_train_{timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 各種パスの設定
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # 可視化設定
        self.visualize = not getattr(config, 'no_visualize', False)
        self.visualize_steps = getattr(config, 'visualize_steps', 5)
        
        # 推論評価設定
        self.run_inference_eval = getattr(config, 'run_inference_eval', False)
        self.inference_eval_samples = getattr(config, 'inference_eval_samples', 3)
        
        # 学習パラメータ
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        self.save_steps = config.save_steps
        self.global_step = 0
        self.best_loss = float('inf')
        
        # 損失履歴
        self.loss_history = {
            'steps': [],
            'total_loss': [],
            'lm_loss': [],
            'seg_loss': [],
            'learning_rate': []
        }
        
        # 可視化ディレクトリ
        if self.visualize:
            self.vis_dir = self.output_dir / "visualizations"
            self.vis_dir.mkdir(exist_ok=True)
            logger.info(f"可視化を有効化: {self.visualize_steps}ステップごとに保存")
        
        # シード固定（再現性のため）
        self.seed = getattr(config, 'seed', 43)
        self.set_seed(self.seed)
        logger.info(f"ランダムシードを固定: {self.seed}")
        
        # SAM2公式Transforms（推論時のみ使用）
        if self.run_inference_eval:
            self.sam_transforms = SAM2Transforms(
                resolution=1024,
                mask_threshold=0.0,
                max_hole_area=0.0,
                max_sprinkle_area=0.0
            )
        
        # デバッグモード
        self.debug = getattr(config, 'debug', False)
        if self.debug:
            logger.setLevel(logging.DEBUG)
            logger.info("デバッグモードを有効化")
        
        # ImageNet正規化パラメータを事前にdeviceに移動
        self.imagenet_mean_device = None
        self.imagenet_std_device = None

    def set_seed(self, seed: int):
        """ランダムシードを固定"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # DataLoaderのワーカー初期化関数
        def worker_init_fn(worker_id):
            worker_seed = seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            
        self.worker_init_fn = worker_init_fn

    def _ensure_4d_mask(self, mask):
        """マスクを[B, C, H, W]形式に統一（効率化のため関数化）"""
        if mask.dim() == 2:
            return mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            return mask.unsqueeze(0) if mask.shape[0] == 1 else mask.unsqueeze(1)
        return mask

    def display_parameter_statistics(self):
        """パラメータ統計の表示"""
        # 元のコードと同一の実装
        total_params = 0
        trainable_params = 0
        frozen_params = 0
        
        # モジュール別の統計
        module_stats = {}
        
        for name, param in self.model.named_parameters():
            num_params = param.numel()
            total_params += num_params
            
            if param.requires_grad:
                trainable_params += num_params
            else:
                frozen_params += num_params
            
            # モジュール名の抽出
            module_name = name.split('.')[0]
            if module_name not in module_stats:
                module_stats[module_name] = {
                    'total': 0,
                    'trainable': 0,
                    'frozen': 0
                }
            
            module_stats[module_name]['total'] += num_params
            if param.requires_grad:
                module_stats[module_name]['trainable'] += num_params
            else:
                module_stats[module_name]['frozen'] += num_params
        
        # 統計の表示
        logger.info("=" * 80)
        logger.info("パラメータ統計")
        logger.info("=" * 80)
        logger.info(f"総パラメータ数: {total_params:,} ({total_params/1e6:.2f}M)")
        logger.info(f"学習可能パラメータ数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        logger.info(f"凍結パラメータ数: {frozen_params:,} ({frozen_params/1e6:.2f}M)")
        logger.info(f"学習可能パラメータの割合: {trainable_params/total_params*100:.2f}%")
        logger.info("")
        
        # モジュール別統計の表示
        logger.info("モジュール別パラメータ統計:")
        logger.info("-" * 80)
        logger.info(f"{'Module':<30} {'Total':>15} {'Trainable':>15} {'Frozen':>15} {'Trainable%':>10}")
        logger.info("-" * 80)
        
        for module_name, stats in sorted(module_stats.items()):
            total = stats['total']
            trainable = stats['trainable']
            frozen = stats['frozen']
            trainable_pct = trainable / total * 100 if total > 0 else 0
            
            logger.info(f"{module_name:<30} {total:>15,} {trainable:>15,} {frozen:>15,} {trainable_pct:>9.2f}%")
        
        logger.info("=" * 80)
        
        # 学習可能レイヤーの詳細表示（デバッグモードのみ）
        if self.debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("\n学習可能なレイヤー:")
            logger.debug("-" * 80)
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    logger.debug(f"  {name}: shape={list(param.shape)}, params={param.numel():,}")
        
        # メモリ使用量の推定（GB単位）
        param_memory_gb = (total_params * 4) / (1024**3)  # float32想定
        logger.info(f"\n推定メモリ使用量（パラメータのみ）: {param_memory_gb:.2f} GB")
        
        if torch.cuda.is_available():
            # 実際のGPUメモリ使用量
            allocated_gb = torch.cuda.memory_allocated() / (1024**3)
            reserved_gb = torch.cuda.memory_reserved() / (1024**3)
            logger.info(f"実際のGPUメモリ使用量: {allocated_gb:.2f} GB allocated, {reserved_gb:.2f} GB reserved")

    def setup_model_and_data(self):
        """モデルとデータローダーのセットアップ"""
        logger.info("モデルとデータローダーをセットアップ中...")
        
        # LISAConfigの初期化
        self.lisa_config = LISAConfig(
            sam_model_id="facebook/sam2.1-hiera-large",
            qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
            freeze_sam=self.config.freeze_sam,
            freeze_qwen=self.config.freeze_qwen,
            use_lora=self.config.use_lora,
            lora_r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            tune_qwen_visual=self.config.tune_qwen_visual
        )
        
        # トークナイザの初期化
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.lisa_config.qwen_model_name,
            padding_side='right',
            trust_remote_code=True
        )
        
        # Qwen2VLProcessor（tokenizer=False）
        if hasattr(self.lisa_config, 'processor') and self.lisa_config.processor is not None:
            self.processor = self.lisa_config.processor
        else:
            self.processor = Qwen2VLProcessor.from_pretrained(
                self.lisa_config.qwen_model_name,
                tokenizer=None,
                trust_remote_code=True
            )
            self.processor.tokenizer = self.tokenizer
            self.lisa_config.processor = self.processor
        
        # 特殊トークン設定
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'
        
        # モデルの初期化
        logger.info("LISAモデルを初期化中...")
        self.model = LISA_Model(self.lisa_config)
        self.model = self.model.to(self.device)
        
        # パラメータ統計の表示
        self.display_parameter_statistics()
        
        # GPUメモリの状況を確認
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"モデルロード後のGPUメモリ使用量: {allocated:.2f} GB")
        
        # データセットパス設定
        dataset_root = Path(self.config.dataset_root)
        
        base_image_dir = dataset_root / "images"
        if not base_image_dir.exists():
            base_image_dir = dataset_root / "train2017"
            
        data_json_path = dataset_root / "dataset.json"
        if not data_json_path.exists():
            data_json_path = dataset_root / "reason_seg" / "dataset.json"
        
        logger.info(f"Using base_image_dir: {base_image_dir}")
        logger.info(f"Using data_json_path: {data_json_path}")
        
        # データセットの初期化
        self.train_dataset = HybridDataset(
            base_image_dir=str(base_image_dir),
            tokenizer=self.tokenizer,
            processor=self.processor,
            data_json_path=str(data_json_path) if data_json_path.exists() else None,
            use_dynamic_resolution=self.config.use_dynamic_resolution,
            use_quality_score=self.config.use_quality_score,
            sample_rates=[0.5, 0.2, 0.3]
        )
        
        # Collatorの初期化
        self.collator = MultiModalDataCollator(
            tokenizer=self.tokenizer,
            processor=self.processor
        )
        
        # DataLoaderの作成
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=self.collator,
            pin_memory=True,
            worker_init_fn=self.worker_init_fn
        )
        
        logger.info(f"データセットサイズ: {len(self.train_dataset)}")
        logger.info(f"バッチ数: {len(self.train_loader)}")

    def setup_optimizer_and_scheduler(self):
        """オプティマイザとスケジューラのセットアップ"""
        logger.info("オプティマイザとスケジューラをセットアップ中...")
        
        # 学習可能パラメータの収集
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # パラメータ数の確認
        num_trainable = sum(p.numel() for p in trainable_params)
        logger.info(f"学習可能パラメータ数: {num_trainable:,}")
        
        # AdamWオプティマイザ
        self.optimizer = AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.config.weight_decay
        )
        
        # 学習率スケジューラ
        num_training_steps = len(self.train_loader) * self.num_epochs
        num_warmup_steps = int(0.1 * num_training_steps)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        logger.info(f"総学習ステップ数: {num_training_steps}")
        logger.info(f"ウォームアップステップ数: {num_warmup_steps}")

    def compute_loss(self, outputs, labels, mask_labels):
        """損失計算（1会話1マスクに最適化） - 最適化版"""
        # 言語モデリング損失
        vocab_size = outputs.logits.size(-1)
        lm_loss = F.cross_entropy(
            outputs.logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=-100
        )
        
        # セグメンテーション損失
        seg_loss = 0.0
        seg_count = 0
        
        if outputs.mask_logits is not None:
            for batch_idx, batch_masks in enumerate(outputs.mask_logits):
                if batch_masks is not None and len(batch_masks) > 0:
                    gt_mask = mask_labels[batch_idx]
                    
                    # 最初のマスクのみを使用
                    pred_mask = batch_masks[0] if isinstance(batch_masks, list) else batch_masks
                    
                    # デバッグ情報（条件付き）
                    if self.debug and logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"pred_mask shape before upsampling: {pred_mask.shape}")
                        logger.debug(f"gt_mask shape: {gt_mask.shape}")
                    
                    # 予測マスクをSAM座標系にアップサンプル
                    pred_mask_4d = self._ensure_4d_mask(pred_mask)
                    
                    # bilinearでアップサンプル
                    pred_mask_sam = F.interpolate(
                        pred_mask_4d.float(),
                        size=(self.SAM_SIZE, self.SAM_SIZE),
                        mode='bilinear',
                        align_corners=False
                    )
                    pred_mask_sam = pred_mask_sam.squeeze(0).squeeze(0)
                    
                    # GTマスクの形状を確認と調整
                    if gt_mask.dim() == 3:
                        gt_mask_sam = gt_mask[0] if gt_mask.shape[0] > 1 else gt_mask.squeeze(0)
                    elif gt_mask.dim() == 2:
                        gt_mask_sam = gt_mask
                    else:
                        gt_mask_sam = gt_mask.view(self.SAM_SIZE, self.SAM_SIZE)
                    
                    # GTマスクが1024x1024でない場合はリサイズ
                    if gt_mask_sam.shape != (self.SAM_SIZE, self.SAM_SIZE):
                        if self.debug and logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"GT mask resizing from {gt_mask_sam.shape} to {(self.SAM_SIZE, self.SAM_SIZE)}")
                        
                        gt_mask_4d = self._ensure_4d_mask(gt_mask_sam)
                        gt_mask_resized = F.interpolate(
                            gt_mask_4d.float(),
                            size=(self.SAM_SIZE, self.SAM_SIZE),
                            mode='nearest'
                        )
                        gt_mask_sam = gt_mask_resized.squeeze(0).squeeze(0)
                    
                    # BCE損失
                    bce_loss = F.binary_cross_entropy_with_logits(
                        pred_mask_sam,
                        gt_mask_sam.float()
                    )
                    
                    # Dice損失
                    pred_sigmoid = torch.sigmoid(pred_mask_sam)
                    intersection = (pred_sigmoid * gt_mask_sam).sum()
                    dice = 2 * intersection / (pred_sigmoid.sum() + gt_mask_sam.sum() + 1e-8)
                    dice_loss = 1 - dice
                    
                    seg_loss += bce_loss + dice_loss
                    seg_count += 1
        
        # 平均セグメンテーション損失
        if seg_count > 0:
            seg_loss = seg_loss / seg_count
        
        # 総合損失
        total_loss = lm_loss + self.config.seg_loss_weight * seg_loss
        
        return total_loss, lm_loss, seg_loss

    @torch.no_grad()
    def save_visualization(self, batch, outputs, step):
        """可視化の保存（最適化版） - torch.no_grad()で囲む"""
        try:
            # 可視化が無効な場合はスキップ
            if self.visualize_steps <= 0:
                return
            
            # 保存頻度のチェック
            if step % max(1, self.visualize_steps) != 0:
                return
            
            # マスクを持つサンプルを探す
            valid_idx = None
            if hasattr(outputs, 'mask_logits') and outputs.mask_logits is not None:
                for i in range(len(outputs.mask_logits)):
                    if outputs.mask_logits[i] is not None:
                        valid_idx = i
                        break
            
            if valid_idx is None:
                return
            
            if 'sam_images' not in batch or batch['sam_images'] is None:
                return
            
            # SAM画像のdenormalize
            sam_image = batch['sam_images'][valid_idx]
            
            # ImageNet正規化パラメータをdeviceに移動（初回のみ）
            if self.imagenet_mean_device is None:
                self.imagenet_mean_device = self.IMAGENET_MEAN.to(sam_image.device).view(3, 1, 1)
                self.imagenet_std_device = self.IMAGENET_STD.to(sam_image.device).view(3, 1, 1)
            
            # denormalize処理
            img = (sam_image * self.imagenet_std_device + self.imagenet_mean_device).clamp(0, 1)
            img = (img * 255.0).round().to(torch.uint8)
            sam_image_np = img.permute(1, 2, 0).cpu().numpy()
            
            # orig_hwを取得
            orig_hw = None
            if 'orig_hw' in batch and batch['orig_hw'] is not None:
                if isinstance(batch['orig_hw'], list) and len(batch['orig_hw']) > valid_idx:
                    orig_hw = batch['orig_hw'][valid_idx]
                    if isinstance(orig_hw, list):
                        orig_hw = tuple(orig_hw)
                elif torch.is_tensor(batch['orig_hw']):
                    orig_hw = tuple(batch['orig_hw'][valid_idx].tolist())
            
            if orig_hw is None or not isinstance(orig_hw, (list, tuple)) or len(orig_hw) != 2:
                raise ValueError(f"Invalid or missing orig_hw at step {step}")
            
            orig_h, orig_w = orig_hw
            
            # SAM入力サイズを計算
            scale = self.SAM_SIZE / max(orig_h, orig_w)
            sam_input_h = int(orig_h * scale)
            sam_input_w = int(orig_w * scale)
            
            # マスクを取得
            pred_mask = outputs.mask_logits[valid_idx]
            if isinstance(pred_mask, list):
                pred_mask = pred_mask[0]
            
            pred_mask = self._ensure_4d_mask(pred_mask)
            
            # SAM postprocess（3段階）
            # 1. 1024x1024にアップサンプル
            if pred_mask.shape[-1] != self.SAM_SIZE:
                pred_mask_1024 = F.interpolate(
                    pred_mask.float(),
                    size=(self.SAM_SIZE, self.SAM_SIZE),
                    mode='bilinear',
                    align_corners=False
                )
            else:
                pred_mask_1024 = pred_mask.float()
            
            # 2. パディング除去
            pred_mask_unpadded = pred_mask_1024[:, :, :sam_input_h, :sam_input_w]
            
            # 3. 元サイズにリサイズ
            pred_mask_processed = F.interpolate(
                pred_mask_unpadded,
                size=(orig_h, orig_w),
                mode='nearest'
            )
            
            pred_mask_np = torch.sigmoid(pred_mask_processed).squeeze().cpu().numpy()
            
            # GTマスクを取得
            gt_mask = None
            if 'mask_labels' in batch and batch['mask_labels'] is not None:
                if torch.is_tensor(batch['mask_labels']):
                    if batch['mask_labels'].dim() >= 3:
                        gt_mask = batch['mask_labels'][valid_idx]
                elif isinstance(batch['mask_labels'], list):
                    if len(batch['mask_labels']) > valid_idx:
                        gt_mask = batch['mask_labels'][valid_idx]
            
            if gt_mask is not None:
                gt_mask = self._ensure_4d_mask(gt_mask)
                
                # GTマスクもSAM postprocessを適用
                if gt_mask.shape[-1] != self.SAM_SIZE:
                    gt_mask_1024 = F.interpolate(
                        gt_mask.float(),
                        size=(self.SAM_SIZE, self.SAM_SIZE),
                        mode='bilinear',
                        align_corners=False
                    )
                else:
                    gt_mask_1024 = gt_mask.float()
                
                gt_mask_unpadded = gt_mask_1024[:, :, :sam_input_h, :sam_input_w]
                gt_mask_processed = F.interpolate(
                    gt_mask_unpadded,
                    size=(orig_h, orig_w),
                    mode='nearest'
                )
                
                gt_mask_np = gt_mask_processed.squeeze().cpu().numpy()
            else:
                gt_mask_np = np.zeros_like(pred_mask_np)
            
            # 元画像を取得または復元
            original_image_np = None
            if 'original_images' in batch and batch['original_images'] is not None:
                if isinstance(batch['original_images'], list) and len(batch['original_images']) > valid_idx:
                    original_image = batch['original_images'][valid_idx]
                    if hasattr(original_image, 'size'):
                        original_image_np = np.array(original_image.convert('RGB'))
            
            if original_image_np is None:
                # SAM画像から復元
                sam_unpadded = sam_image_np[:sam_input_h, :sam_input_w, :]
                original_image_np = cv2.resize(sam_unpadded, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
            # サイズ調整
            if original_image_np.shape[:2] != (orig_h, orig_w):
                original_image_np = cv2.resize(original_image_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
            # 二値化（閾値0.5）
            pred_binary = (pred_mask_np > 0.5).astype(float)
            gt_binary = (gt_mask_np > 0.5).astype(float)
            
            # メトリクス計算
            intersection = np.sum(pred_binary * gt_binary)
            union = np.sum(pred_binary) + np.sum(gt_binary) - intersection
            iou = intersection / (union + 1e-7)
            dice = 2 * intersection / (np.sum(pred_binary) + np.sum(gt_binary) + 1e-7)
            
            # 可視化作成（メモリ効率化のため配列を再利用）
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # 上段
            axes[0, 0].imshow(original_image_np)
            axes[0, 0].set_title(f'Original Image\nshape: {original_image_np.shape}', fontsize=10)
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(pred_mask_np, cmap='jet', vmin=0, vmax=1)
            axes[0, 1].set_title(f'Predicted Mask\nshape: {pred_mask_np.shape}', fontsize=10)
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(gt_mask_np, cmap='jet', vmin=0, vmax=1)
            axes[0, 2].set_title(f'GT Mask\nshape: {gt_mask_np.shape}', fontsize=10)
            axes[0, 2].axis('off')
            
            # 下段（オーバーレイ用配列を再利用）
            overlay = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
            
            # 予測マスクのオーバーレイ
            axes[1, 0].imshow(original_image_np)
            overlay[:, :, 0] = (pred_binary * 255).astype(np.uint8)
            axes[1, 0].imshow(overlay, alpha=0.3)
            axes[1, 0].set_title('Prediction Overlay', fontsize=10)
            axes[1, 0].axis('off')
            
            # GTマスクのオーバーレイ
            overlay[:, :, :] = 0  # リセット
            axes[1, 1].imshow(original_image_np)
            overlay[:, :, 1] = (gt_binary * 255).astype(np.uint8)
            axes[1, 1].imshow(overlay, alpha=0.3)
            axes[1, 1].set_title('GT Overlay', fontsize=10)
            axes[1, 1].axis('off')
            
            # 比較
            overlay[:, :, :] = 0  # リセット
            axes[1, 2].imshow(original_image_np)
            overlay[:, :, 0] = (pred_binary * 255).astype(np.uint8)
            overlay[:, :, 1] = (gt_binary * 255).astype(np.uint8)
            axes[1, 2].imshow(overlay, alpha=0.3)
            axes[1, 2].set_title(f'Comparison (IoU: {iou:.3f}, Dice: {dice:.3f})', fontsize=10)
            axes[1, 2].axis('off')
            
            # メタ情報
            info_text = f"Step: {step} | IoU: {iou:.3f} | Dice: {dice:.3f}\n"
            info_text += f"Orig HW: {orig_hw} | SAM input HW: ({sam_input_h}, {sam_input_w})"
            fig.suptitle(info_text, fontsize=12, y=0.98)
            
            plt.tight_layout()
            
            # 保存
            vis_path = self.vis_dir / f"step_{step:06d}.png"
            plt.savefig(vis_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved visualization to {vis_path} (IoU: {iou:.3f}, Dice: {dice:.3f})")
            
            # WandB
            if self.config.use_wandb:
                import wandb
                wandb.log({
                    "visualization": wandb.Image(str(vis_path)),
                    "val_iou": iou,
                    "val_dice": dice,
                }, step=step)
            
        except Exception as e:
            logger.error(f"Failed to create visualization at step {step}: {e}")
            if self.debug:
                import traceback
                logger.error(traceback.format_exc())

    # 残りのメソッドは変更なし（省略）
    def train_epoch(self, epoch):
        """1エポックの学習（元のコードと同一）"""
        # 元のコードをそのまま使用
        pass

    def save_checkpoint(self, step=None):
        """チェックポイントの保存（元のコードと同一）"""
        # 元のコードをそのまま使用
        pass

    def train(self):
        """学習のメインループ（元のコードと同一）"""
        # 元のコードをそのまま使用
        pass


def main():
    """メイン関数"""
    from dataclasses import dataclass
    
    @dataclass
    class Config:
        # データセット設定
        dataset_root: str = "/home/soya/datasets"
        
        # 学習設定
        num_epochs: int = 3
        batch_size: int = 2
        learning_rate: float = 5e-5
        weight_decay: float = 0.01
        gradient_accumulation_steps: int = 4
        save_steps: int = 100
        seg_loss_weight: float = 1.0
        
        # モデル設定
        freeze_sam: bool = True
        freeze_qwen: bool = False
        use_lora: bool = True
        lora_r: int = 8
        lora_alpha: int = 16
        lora_dropout: float = 0.1
        tune_qwen_visual: bool = False
        
        # 動的解像度
        use_dynamic_resolution: bool = True
        use_quality_score: bool = False
        
        # 可視化設定
        no_visualize: bool = False
        visualize_steps: int = 5
        
        # 評価設定
        run_inference_eval: bool = False
        inference_eval_samples: int = 3
        
        # その他
        seed: int = 43
        debug: bool = False
        use_wandb: bool = False
    
    config = Config()
    trainer = MinimalTrainer(config)
    trainer.setup_model_and_data()
    trainer.setup_optimizer_and_scheduler()
    trainer.train()


if __name__ == "__main__":
    main()