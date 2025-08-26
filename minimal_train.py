#!/usr/bin/env python3
"""
実データを使用したミニマルなトレーニングスクリプト
セマンティックセグメンテーションデータセット（ADE20K）に絞って実装
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
import logging
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm
import numpy as np
import random
import wandb
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # バックエンドを設定
import cv2
from PIL import Image

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent))

from src.config import LISAConfig
from src.models import LISA_Model
from src.utils import prepare_tokenizer_for_lisa
from src.data.dataset import HybridDataset
from src.data.collators import MultiModalDataCollator
from transformers import AutoProcessor, get_linear_schedule_with_warmup
from sam2.utils.transforms import SAM2Transforms

# ロギング設定
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.DEBUG if '--debug' in sys.argv else logging.INFO
)
logger = logging.getLogger(__name__)


class MinimalTrainer:
    """最小限の学習を行うトレーナークラス（最適化版）"""
    
    # クラス定数（効率化のため）
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
        
        # シード固定
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
        
        # ImageNet正規化パラメータ（デバイス移動後にキャッシュ）
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
        
        # 学習可能レイヤーの詳細（デバッグモードのみ）
        if self.debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("\n学習可能なレイヤー:")
            logger.debug("-" * 80)
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    logger.debug(f"  {name}: shape={list(param.shape)}, params={param.numel():,}")
        
        # メモリ使用量の推定
        param_memory_gb = (total_params * 4) / (1024**3)
        logger.info(f"\n推定メモリ使用量（パラメータのみ）: {param_memory_gb:.2f} GB")
        
        if torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated() / (1024**3)
            reserved_gb = torch.cuda.memory_reserved() / (1024**3)
            logger.info(f"実際のGPUメモリ使用量: {allocated_gb:.2f} GB allocated, {reserved_gb:.2f} GB reserved")

    def setup_model_and_data(self):
        """モデルとデータローダーのセットアップ"""
        logger.info("モデルとデータローダーをセットアップ中...")
        
        # transformersから必要なクラスをインポート
        from transformers import AutoTokenizer, Qwen2VLProcessor
        from src.utils.tokenizer_utils import prepare_tokenizer_for_lisa
        
        # LISAConfigの初期化 - 実際のLISAConfigパラメータのみ使用
        self.lisa_config = LISAConfig(
            sam_model_name="facebook/sam2.1-hiera-large",
            qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
            # LoRA設定はLISAConfig内で定義済み
            lora_r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            # freeze設定もLISAConfig内の正しいパラメータ名を使用
            freeze_qwen_base=self.config.freeze_qwen,
            freeze_sam_mask_decoder_base=self.config.freeze_sam,
            # 視覚モデルのチューニング設定
            freeze_qwen_lora=not self.config.use_lora  # use_loraがFalseならfreeze
        )
        
        # トークナイザーの初期化（SEGトークン付き）
        logger.info("トークナイザーとSEGトークンを準備中...")
        self.tokenizer = prepare_tokenizer_for_lisa(
            model_name=self.lisa_config.qwen_model_name,
            seg_token=self.lisa_config.seg_token,
            padding_side='right'  # 学習時はright
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
        
        # ProcessorのトークナイザーにもSEGトークンを追加
        if self.lisa_config.seg_token not in self.processor.tokenizer.get_vocab():
            logger.info(f"ProcessorのトークナイザーにSEGトークンを追加")
            self.processor.tokenizer.add_special_tokens({"additional_special_tokens": [self.lisa_config.seg_token]})
            logger.info(f"Processor語彙サイズ: {len(self.processor.tokenizer)}")
        
        # モデルの初期化
        logger.info("LISAモデルを初期化中...")
        self.model = LISA_Model(self.lisa_config)
        
        # 重要: トークナイザーを設定してSEGトークンの埋め込みをリサイズ
        logger.info("トークナイザーを設定してSEGトークンの埋め込みをリサイズ")
        self.model.set_tokenizer(self.tokenizer, seg_token=self.lisa_config.seg_token)
        
        # SEGトークンIDが正しく設定されたか確認
        if self.model.seg_token_id is not None:
            logger.info(f"✅ モデルのSEGトークンIDが設定されました: {self.model.seg_token_id}")
        else:
            logger.error(f"❌ モデルのSEGトークンIDの設定に失敗しました")
        
        # モデルをデバイスに移動
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
        
        # base_image_dirの設定（HybridDatasetの要求に合わせて）
        base_image_dir = dataset_root / "images"
        if not base_image_dir.exists():
            base_image_dir = dataset_root / "train2017"
        if not base_image_dir.exists():
            # さらにフォールバック
            base_image_dir = dataset_root
            
        logger.info(f"Using base_image_dir: {base_image_dir}")
        
        # データセットの初期化 - HybridDatasetの正しいパラメータ名を使用
        # sample_ratesを文字列からfloatのリストに変換
        if hasattr(self.config, 'sample_rates') and isinstance(self.config.sample_rates, str):
            sample_rates = [float(x) for x in self.config.sample_rates.split(',')]
        else:
            sample_rates = getattr(self.config, 'sample_rates', [1.0])
        
        # dataset_typesも処理
        if hasattr(self.config, 'dataset_types') and isinstance(self.config.dataset_types, str):
            dataset_types = self.config.dataset_types.split('||')
        else:
            dataset_types = getattr(self.config, 'dataset_types', ['reason_seg'])
        
        self.train_dataset = HybridDataset(
            base_image_dir=str(base_image_dir),
            qwen_processor=self.processor,  # tokenizerではなくqwen_processorを使用
            samples_per_epoch=getattr(self.config, 'samples_per_epoch', 500 * 8 * 2 * 10),
            dataset='||'.join(dataset_types),
            sample_rate=sample_rates,
            qwen_image_size=self.lisa_config.qwen_image_size,
            sam_image_size=self.lisa_config.sam_image_size,
        )
        
        # Collatorの初期化
        self.collator = MultiModalDataCollator(
            tokenizer=self.tokenizer,
            max_length=self.lisa_config.model_max_length,
            config=self.lisa_config  # configを渡す
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

    def get_alignment_checkpoint_path(self):
        """アライメントチェックポイントのパスを取得"""
        checkpoint_name = f"alignment_checkpoint_{self.lisa_config.qwen_model_name.replace('/', '_')}.pt"
        return self.checkpoint_dir / checkpoint_name

    def save_alignment_checkpoint(self, alignment_state):
        """アライメントチェックポイントの保存"""
        checkpoint_path = self.get_alignment_checkpoint_path()
        
        checkpoint = {
            'alignment_state': alignment_state,
            'config': {
                'qwen_model_name': self.lisa_config.qwen_model_name,
                'sam_model_id': self.lisa_config.sam_model_id,
            },
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"アライメントチェックポイントを保存: {checkpoint_path}")
        
        # パラメータ値の詳細を保存（デバッグ用）
        if self.debug and logger.isEnabledFor(logging.DEBUG):
            debug_info = {}
            for name, tensor in alignment_state.items():
                if isinstance(tensor, torch.Tensor):
                    debug_info[name] = {
                        'shape': list(tensor.shape),
                        'mean': float(tensor.mean()),
                        'std': float(tensor.std()),
                        'min': float(tensor.min()),
                        'max': float(tensor.max())
                    }
            
            debug_path = checkpoint_path.with_suffix('.debug.json')
            with open(debug_path, 'w') as f:
                json.dump(debug_info, f, indent=2)
            logger.debug(f"デバッグ情報を保存: {debug_path}")

    def _apply_alignment_freeze_settings(self):
        """アライメント段階のフリーズ設定を適用"""
        # Qwenの全パラメータをフリーズ
        for name, param in self.model.qwen_model.named_parameters():
            if param.requires_grad:
                param.requires_grad = False
                if self.debug and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"[Align] Frozen: {name}")
        
        # SAMは既にフリーズされているはず
        for name, param in self.model.sam_model.named_parameters():
            if param.requires_grad:
                param.requires_grad = False
                if self.debug and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"[Align] Frozen SAM: {name}")

    def _restore_training_freeze_settings(self):
        """通常学習用のフリーズ設定に戻す"""
        # LoRAパラメータのフリーズ解除
        if self.config.use_lora:
            for name, param in self.model.qwen_model.named_parameters():
                if 'lora' in name.lower():
                    param.requires_grad = True
                    if self.debug and logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"[Training] Unfrozen LoRA: {name}")
        
        # ビジュアルエンコーダー
        if self.config.tune_qwen_visual:
            for name, param in self.model.qwen_model.visual.named_parameters():
                param.requires_grad = True
                if self.debug and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"[Training] Unfrozen Visual: {name}")

    def load_alignment_checkpoint(self):
        """アライメントチェックポイントのロード"""
        checkpoint_path = self.get_alignment_checkpoint_path()
        
        if not checkpoint_path.exists():
            logger.warning(f"チェックポイントが見つかりません: {checkpoint_path}")
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            alignment_state = checkpoint['alignment_state']
            
            # パラメータの復元
            for name, param in self.model.qwen_model.named_parameters():
                if name in alignment_state:
                    param.data.copy_(alignment_state[name].to(self.device))
                    if self.debug and logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Restored: {name}")
            
            # SEGトークン埋め込みの復元
            if 'seg_token_embedding' in alignment_state:
                self.model.seg_token_embedding.data.copy_(
                    alignment_state['seg_token_embedding'].to(self.device)
                )
            
            # プロンプトプロジェクターとベータの復元
            if hasattr(self.model, 'text_prompt_projector') and 'text_prompt_projector.weight' in alignment_state:
                self.model.text_prompt_projector.weight.data.copy_(
                    alignment_state['text_prompt_projector.weight'].to(self.device)
                )
                self.model.text_prompt_projector.bias.data.copy_(
                    alignment_state['text_prompt_projector.bias'].to(self.device)
                )
            
            if hasattr(self.model, 'prompt_beta') and 'prompt_beta' in alignment_state:
                self.model.prompt_beta.data.copy_(
                    alignment_state['prompt_beta'].to(self.device)
                )
            
            logger.info(f"アライメントチェックポイントをロード: {checkpoint_path}")
            logger.info(f"タイムスタンプ: {checkpoint.get('timestamp', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"チェックポイントのロードに失敗: {e}")
            return False

    def check_alignment_cache(self):
        """アライメントキャッシュの存在確認"""
        checkpoint_path = self.get_alignment_checkpoint_path()
        
        if checkpoint_path.exists():
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                timestamp = checkpoint.get('timestamp', 'unknown')
                logger.info(f"アライメントキャッシュが見つかりました")
                logger.info(f"  パス: {checkpoint_path}")
                logger.info(f"  作成日時: {timestamp}")
                
                # configの互換性チェック
                saved_config = checkpoint.get('config', {})
                if saved_config.get('qwen_model_name') != self.lisa_config.qwen_model_name:
                    logger.warning("モデル名が異なります。キャッシュは使用できません")
                    return False
                
                return True
            except Exception as e:
                logger.warning(f"キャッシュの確認に失敗: {e}")
                return False
        
        return False

    def run_alignment_stage(self):
        """アライメント段階の実行（最適化版）"""
        logger.info("=" * 80)
        logger.info("アライメント段階を開始")
        logger.info("=" * 80)
        
        # キャッシュの確認
        if self.check_alignment_cache():
            user_input = input("アライメントキャッシュが見つかりました。使用しますか？ (y/n): ")
            if user_input.lower() == 'y':
                if self.load_alignment_checkpoint():
                    logger.info("キャッシュからアライメントをロードしました")
                    self._restore_training_freeze_settings()
                    return
                else:
                    logger.warning("キャッシュのロードに失敗。アライメントを再実行します")
        
        # アライメント用のフリーズ設定
        self._apply_alignment_freeze_settings()
        
        # アライメント用のパラメータだけを収集
        alignment_params = []
        
        # SEGトークン埋め込み
        alignment_params.append(self.model.seg_token_embedding)
        
        # プロンプトプロジェクター
        if hasattr(self.model, 'text_prompt_projector'):
            alignment_params.extend(self.model.text_prompt_projector.parameters())
        
        # prompt_beta
        if hasattr(self.model, 'prompt_beta'):
            alignment_params.append(self.model.prompt_beta)
        
        # QwenのLoRAパラメータ（アライメント時も学習）
        if self.config.use_lora:
            for name, param in self.model.qwen_model.named_parameters():
                if 'lora' in name.lower() and param.requires_grad:
                    alignment_params.append(param)
                    if self.debug and logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"[Align] Added Qwen LoRA param: {name}")
        
        # その他のrequires_grad=Trueのパラメータ
        for name, param in self.model.named_parameters():
            if param.requires_grad and param not in alignment_params:
                if 'seg_token' in name:
                    alignment_params.append(param)
                    if self.debug and logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"[Align] Added SEG token param: {name}")
                elif 'text_prompt_projector' in name or 'prompt_projector' in name:
                    alignment_params.append(param)
                    if self.debug and logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"[Align] Added TextPromptProjector param: {name}")
                elif 'prompt_beta' in name:
                    alignment_params.append(param)
                    if self.debug and logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"[Align] Added prompt_beta param: {name}")
                else:
                    if self.debug and logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"[Align] Skipped param: {name} (requires_grad={param.requires_grad})")
        
        if not alignment_params:
            logger.warning("アライメント用のパラメータが見つかりません")
            self._restore_training_freeze_settings()
            return
        
        # アライメント用オプティマイザ
        align_optimizer = AdamW(
            alignment_params,
            lr=self.config.learning_rate * 2,  # やや高めの学習率
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        # アライメント実行
        logger.info(f"アライメントパラメータ数: {len(alignment_params)}")
        logger.info("100ステップのアライメントを実行中...")
        
        self.model.train()
        align_steps = 100
        align_losses = []
        
        # プログレスバー
        align_pbar = tqdm(total=align_steps, desc="Alignment", ncols=100)
        
        for step in range(align_steps):
            try:
                batch = next(iter(self.train_loader))
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
            
            # デバイスに移動
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                pixel_values=batch['pixel_values'],
                sam_images=batch.get('sam_images'),
                labels=batch['labels'],
                attention_mask=batch['attention_mask'],
                image_grid_thw=batch.get('image_grid_thw')
            )
            
            # 損失計算（言語損失のみ）
            lm_loss = outputs.loss
            
            # Backward
            lm_loss.backward()
            
            # 勾配クリップ
            torch.nn.utils.clip_grad_norm_(alignment_params, max_norm=1.0)
            
            # パラメータ更新
            align_optimizer.step()
            align_optimizer.zero_grad()
            
            # 記録
            align_losses.append(lm_loss.item())
            
            # プログレスバー更新
            align_pbar.set_postfix({'loss': f'{lm_loss.item():.4f}'})
            align_pbar.update(1)
            
            # 定期的なログ
            if (step + 1) % 20 == 0:
                avg_loss = np.mean(align_losses[-20:])
                logger.info(f"Alignment Step {step+1}/{align_steps}, Avg Loss: {avg_loss:.4f}")
        
        align_pbar.close()
        
        # アライメント結果の保存
        logger.info("アライメント完了。状態を保存中...")
        
        alignment_state = {}
        
        # Qwenのパラメータを保存
        for name, param in self.model.qwen_model.named_parameters():
            if 'lora' in name.lower():
                alignment_state[name] = param.data.cpu()
        
        # SEGトークン埋め込み
        if hasattr(self.model, 'seg_token_embedding'):
            alignment_state['seg_token_embedding'] = self.model.seg_token_embedding.data.cpu()
        
        # プロンプトプロジェクター
        if hasattr(self.model, 'text_prompt_projector'):
            alignment_state['text_prompt_projector.weight'] = self.model.text_prompt_projector.weight.data.cpu()
            alignment_state['text_prompt_projector.bias'] = self.model.text_prompt_projector.bias.data.cpu()
        
        # prompt_beta
        if hasattr(self.model, 'prompt_beta'):
            alignment_state['prompt_beta'] = self.model.prompt_beta.data.cpu()
        
        # チェックポイント保存
        self.save_alignment_checkpoint(alignment_state)
        
        # 通常学習用の設定に戻す
        self._restore_training_freeze_settings()
        
        logger.info("アライメント段階完了")
        logger.info(f"最終アライメント損失: {np.mean(align_losses[-10:]):.4f}")
        logger.info("=" * 80)

    def compute_loss(self, outputs, labels, mask_labels):
        """損失計算（最適化版）"""
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
                    pred_mask = batch_masks[0] if isinstance(batch_masks, list) else batch_masks
                    
                    # デバッグ情報（条件付き）
                    if self.debug and logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"pred_mask shape before upsampling: {pred_mask.shape}")
                        logger.debug(f"gt_mask shape: {gt_mask.shape}")
                    
                    # 予測マスクをSAM座標系にアップサンプル
                    pred_mask_4d = self._ensure_4d_mask(pred_mask)
                    
                    pred_mask_sam = F.interpolate(
                        pred_mask_4d.float(),
                        size=(self.SAM_SIZE, self.SAM_SIZE),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).squeeze(0)
                    
                    # GTマスクの形状調整
                    if gt_mask.dim() == 3:
                        gt_mask_sam = gt_mask[0] if gt_mask.shape[0] > 1 else gt_mask.squeeze(0)
                    elif gt_mask.dim() == 2:
                        gt_mask_sam = gt_mask
                    else:
                        gt_mask_sam = gt_mask.view(self.SAM_SIZE, self.SAM_SIZE)
                    
                    # GTマスクのリサイズ（必要な場合）
                    if gt_mask_sam.shape != (self.SAM_SIZE, self.SAM_SIZE):
                        if self.debug and logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"GT mask resizing from {gt_mask_sam.shape} to {(self.SAM_SIZE, self.SAM_SIZE)}")
                        
                        gt_mask_4d = self._ensure_4d_mask(gt_mask_sam)
                        gt_mask_sam = F.interpolate(
                            gt_mask_4d.float(),
                            size=(self.SAM_SIZE, self.SAM_SIZE),
                            mode='nearest'
                        ).squeeze(0).squeeze(0)
                    
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

    def train_epoch(self, epoch):
        """1エポックの学習"""
        self.model.train()
        epoch_loss = 0
        epoch_lm_loss = 0
        epoch_seg_loss = 0
        
        # プログレスバー
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}", ncols=100)
        
        for batch_idx, batch in enumerate(pbar):
            # メモリ使用量の記録（デバッグモードのみ）
            if self.debug and logger.isEnabledFor(logging.DEBUG) and batch_idx == 0:
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    logger.debug(f"GPU memory before batch: {allocated:.2f} GB")
            
            # デバイスに移動
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # トークン数の計算（デバッグモードのみ）
            if self.debug and logger.isEnabledFor(logging.DEBUG):
                total_tokens = 0
                if 'pixel_values' in batch and batch['pixel_values'] is not None:
                    if batch['pixel_values'].dim() == 2:
                        img_tokens = batch['pixel_values'].shape[0]
                    else:
                        img_tokens = batch['pixel_values'].shape[0] * 256
                    total_tokens += img_tokens
                
                txt_tokens = batch['input_ids'].shape[1]
                total_tokens += txt_tokens
                
                logger.debug(f"Batch {batch_idx}: img_tokens={img_tokens if 'img_tokens' in locals() else 0}, txt_tokens={txt_tokens}, total={total_tokens}")
                
                if 'pixel_values' in batch:
                    logger.debug(f"pixel_values shape: {batch['pixel_values'].shape}")
                logger.debug(f"input_ids shape: {batch['input_ids'].shape}")
                
                if 'sam_images' in batch:
                    if batch['sam_images'] is not None:
                        logger.debug(f"sam_images shape: {batch['sam_images'].shape}")
                
                if 'image_grid_thw' in batch:
                    logger.debug(f"image_grid_thw: {batch['image_grid_thw']}")
            
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                pixel_values=batch['pixel_values'],
                sam_images=batch.get('sam_images'),
                labels=batch['labels'],
                attention_mask=batch['attention_mask'],
                image_grid_thw=batch.get('image_grid_thw')
            )
            
            # 損失計算
            mask_labels = batch.get('mask_labels', None)
            if mask_labels is not None:
                total_loss, lm_loss, seg_loss = self.compute_loss(
                    outputs, batch['labels'], mask_labels
                )
            else:
                total_loss = outputs.loss
                lm_loss = outputs.loss
                seg_loss = torch.tensor(0.0)
            
            # 勾配累積で正規化
            loss = total_loss / self.gradient_accumulation_steps
            loss.backward()
            
            # 勾配累積
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # 勾配クリップ
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # オプティマイザステップ
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # グローバルステップを更新
                self.global_step += 1
                
                # 損失の記録
                self.loss_history['steps'].append(self.global_step)
                self.loss_history['total_loss'].append(total_loss.item())
                self.loss_history['lm_loss'].append(lm_loss.item())
                self.loss_history['seg_loss'].append(seg_loss.item() if torch.is_tensor(seg_loss) else seg_loss)
                self.loss_history['learning_rate'].append(self.scheduler.get_last_lr()[0])
                
                # WandBログ（重要！）
                if self.config.use_wandb:
                    import wandb
                    wandb.log({
                        'train/loss': total_loss.item(),
                        'train/lm_loss': lm_loss.item(),
                        'train/seg_loss': seg_loss.item() if torch.is_tensor(seg_loss) else seg_loss,
                        'train/learning_rate': self.scheduler.get_last_lr()[0],
                        'train/epoch': epoch,
                        'train/step': self.global_step,
                        'train/beta': self.model.prompt_beta.item() if hasattr(self.model, 'prompt_beta') else 0.0,
                        'train/image_fusion_beta': self.model.image_fusion_beta.item() if hasattr(self.model, 'image_fusion_beta') else 0.0
                    }, step=self.global_step)
                
                # チェックポイント保存
                if self.global_step % self.save_steps == 0:
                    self.save_checkpoint(self.global_step)
                
                # 可視化の保存（条件付き）
                if self.visualize and mask_labels is not None and self.global_step % self.visualize_steps == 0:
                    if self.debug and logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Calling save_visualization at step {self.global_step}")
                    self.save_visualization(batch, outputs, self.global_step)
            
            # 累積損失の計算
            epoch_loss += total_loss.item()
            epoch_lm_loss += lm_loss.item()
            epoch_seg_loss += seg_loss.item() if torch.is_tensor(seg_loss) else seg_loss
            
            # プログレスバー更新
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'lm': f"{lm_loss.item():.4f}",
                'seg': f"{seg_loss.item() if torch.is_tensor(seg_loss) else seg_loss:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        # エポック平均損失
        num_batches = len(self.train_loader)
        avg_loss = epoch_loss / num_batches
        avg_lm_loss = epoch_lm_loss / num_batches  
        avg_seg_loss = epoch_seg_loss / num_batches
        
        logger.info(f"Epoch {epoch+1}/{self.num_epochs} - Avg Loss: {avg_loss:.4f} "
                   f"(LM: {avg_lm_loss:.4f}, Seg: {avg_seg_loss:.4f})")
        
        # WandBにエポック平均もログ
        if self.config.use_wandb:
            import wandb
            wandb.log({
                'epoch/avg_loss': avg_loss,
                'epoch/avg_lm_loss': avg_lm_loss,
                'epoch/avg_seg_loss': avg_seg_loss,
                'epoch/number': epoch + 1
            }, step=self.global_step)
        
        return avg_loss

    @torch.no_grad()
    def save_visualization(self, batch, outputs, step):
        """可視化の保存（最適化版）"""
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
            if pred_mask.shape[-1] != self.SAM_SIZE:
                pred_mask_1024 = F.interpolate(
                    pred_mask.float(),
                    size=(self.SAM_SIZE, self.SAM_SIZE),
                    mode='bilinear',
                    align_corners=False
                )
            else:
                pred_mask_1024 = pred_mask.float()
            
            pred_mask_unpadded = pred_mask_1024[:, :, :sam_input_h, :sam_input_w]
            pred_mask_processed = F.interpolate(
                pred_mask_unpadded,
                size=(orig_h, orig_w),
                mode='nearest'
            )
            
            pred_mask_np = torch.sigmoid(pred_mask_processed).squeeze().cpu().numpy()
            
            # GTマスク処理
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
            
            # 元画像取得または復元
            original_image_np = None
            if 'original_images' in batch and batch['original_images'] is not None:
                if isinstance(batch['original_images'], list) and len(batch['original_images']) > valid_idx:
                    original_image = batch['original_images'][valid_idx]
                    if hasattr(original_image, 'size'):
                        original_image_np = np.array(original_image.convert('RGB'))
            
            if original_image_np is None:
                sam_unpadded = sam_image_np[:sam_input_h, :sam_input_w, :]
                original_image_np = cv2.resize(sam_unpadded, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
            if original_image_np.shape[:2] != (orig_h, orig_w):
                original_image_np = cv2.resize(original_image_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
            # 二値化
            pred_binary = (pred_mask_np > 0.5).astype(float)
            gt_binary = (gt_mask_np > 0.5).astype(float)
            
            # メトリクス計算
            intersection = np.sum(pred_binary * gt_binary)
            union = np.sum(pred_binary) + np.sum(gt_binary) - intersection
            iou = intersection / (union + 1e-7)
            dice = 2 * intersection / (np.sum(pred_binary) + np.sum(gt_binary) + 1e-7)
            
            # 可視化作成（メモリ効率化）
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
            
            # 下段（配列再利用）
            overlay = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
            
            axes[1, 0].imshow(original_image_np)
            overlay[:, :, 0] = (pred_binary * 255).astype(np.uint8)
            axes[1, 0].imshow(overlay, alpha=0.3)
            axes[1, 0].set_title('Prediction Overlay', fontsize=10)
            axes[1, 0].axis('off')
            
            overlay[:, :, :] = 0
            axes[1, 1].imshow(original_image_np)
            overlay[:, :, 1] = (gt_binary * 255).astype(np.uint8)
            axes[1, 1].imshow(overlay, alpha=0.3)
            axes[1, 1].set_title('GT Overlay', fontsize=10)
            axes[1, 1].axis('off')
            
            overlay[:, :, :] = 0
            axes[1, 2].imshow(original_image_np)
            overlay[:, :, 0] = (pred_binary * 255).astype(np.uint8)
            overlay[:, :, 1] = (gt_binary * 255).astype(np.uint8)
            axes[1, 2].imshow(overlay, alpha=0.3)
            axes[1, 2].set_title(f'Comparison (IoU: {iou:.3f}, Dice: {dice:.3f})', fontsize=10)
            axes[1, 2].axis('off')
            
            info_text = f"Step: {step} | IoU: {iou:.3f} | Dice: {dice:.3f}\n"
            info_text += f"Orig HW: {orig_hw} | SAM input HW: ({sam_input_h}, {sam_input_w})"
            fig.suptitle(info_text, fontsize=12, y=0.98)
            
            plt.tight_layout()
            
            vis_path = self.vis_dir / f"step_{step:06d}.png"
            plt.savefig(vis_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved visualization to {vis_path} (IoU: {iou:.3f}, Dice: {dice:.3f})")
            
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

    def save_checkpoint(self, step=None):
        """チェックポイントの保存"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'config': self.config,
            'loss_history': self.loss_history
        }
        
        if step is not None:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
        else:
            checkpoint_path = self.checkpoint_dir / "checkpoint_latest.pt"
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"チェックポイントを保存: {checkpoint_path}")
        
        # ベストモデルの保存
        current_loss = np.mean(self.loss_history['total_loss'][-100:]) if len(self.loss_history['total_loss']) > 0 else float('inf')
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"ベストモデルを更新: {best_path} (loss: {current_loss:.4f})")

    def run_inference_evaluation(self):
        """推論評価の実行"""
        if not self.run_inference_eval:
            return
        
        logger.info("推論評価を実行中...")
        
        # サンプル画像の取得
        sample_images = self.get_sample_images_for_inference()
        
        if not sample_images:
            logger.warning("推論用のサンプル画像が見つかりません")
            return
        
        self.model.eval()
        results = []
        
        for img_path in sample_images[:self.inference_eval_samples]:
            try:
                result = self.run_single_inference(img_path)
                if result:
                    results.append(result)
                    self.visualize_inference_result(result, img_path)
            except Exception as e:
                logger.error(f"推論エラー: {img_path}: {e}")
        
        # 統計の表示
        if results:
            avg_iou = np.mean([r['iou'] for r in results if 'iou' in r])
            logger.info(f"推論評価完了 - 平均IoU: {avg_iou:.4f}")

    def get_sample_images_for_inference(self):
        """推論用のサンプル画像を取得"""
        sample_images = []
        
        # データセットから数枚選択
        if hasattr(self, 'train_dataset'):
            dataset_size = len(self.train_dataset)
            indices = np.random.choice(dataset_size, 
                                     min(self.inference_eval_samples, dataset_size), 
                                     replace=False)
            
            for idx in indices:
                try:
                    sample = self.train_dataset[idx]
                    if 'image_path' in sample and sample['image_path']:
                        sample_images.append(sample['image_path'])
                except:
                    continue
        
        return sample_images

    def run_single_inference(self, image_path):
        """単一画像での推論実行"""
        # 画像読み込み
        image = Image.open(image_path).convert('RGB')
        orig_size = image.size  # (width, height)
        
        # プロンプト生成
        prompt_text = "Please segment the main object in this image. <SEG>"
        
        # メッセージ形式
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]
        
        # 画像処理
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        
        # テキスト処理
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # プロセッサで処理
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        # SAM用画像処理
        sam_image = self.sam_transforms(image)
        sam_image = sam_image.unsqueeze(0).to(self.device)
        
        # デバイスに移動
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v 
                 for k, v in inputs.items()}
        
        # 推論
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs['input_ids'],
                pixel_values=inputs['pixel_values'],
                sam_images=sam_image,
                attention_mask=inputs['attention_mask'],
                image_grid_thw=inputs.get('image_grid_thw')
            )
        
        # マスク取得
        if outputs.mask_logits and outputs.mask_logits[0] is not None:
            mask_logit = outputs.mask_logits[0][0] if isinstance(outputs.mask_logits[0], list) else outputs.mask_logits[0]
            
            # postprocess
            if mask_logit.dim() == 2:
                mask_logit = mask_logit.unsqueeze(0).unsqueeze(0)
            
            # SAM座標系から元サイズへ
            scale = 1024 / max(orig_size)
            input_h = int(orig_size[1] * scale)
            input_w = int(orig_size[0] * scale)
            
            if mask_logit.shape[-1] != 1024:
                mask_logit = F.interpolate(mask_logit.float(), size=(1024, 1024), 
                                          mode='bilinear', align_corners=False)
            
            mask_logit_unpadded = mask_logit[:, :, :input_h, :input_w]
            mask_logit_processed = F.interpolate(
                mask_logit_unpadded,
                size=(orig_size[1], orig_size[0]),
                mode='nearest'
            )
            
            pred_mask = mask_logit_processed[0].cpu().numpy()
            
            # 二値化
            pred_mask = torch.sigmoid(mask_logit_processed[0]).cpu().numpy()
            pred_binary = (pred_mask > 0.5).astype(float)
            
            return {
                'image': np.array(image),
                'mask': pred_binary,
                'mask_prob': pred_mask
            }
        
        return None

    def visualize_inference_result(self, result, image_path):
        """推論結果の可視化"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 元画像
        axes[0].imshow(result['image'])
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        # 予測マスク
        axes[1].imshow(result['mask_prob'].squeeze(), cmap='jet', vmin=0, vmax=1)
        axes[1].set_title('Predicted Mask (Probability)')
        axes[1].axis('off')
        
        # オーバーレイ
        axes[2].imshow(result['image'])
        overlay = np.zeros_like(result['image'])
        overlay[:, :, 0] = result['mask'].squeeze() * 255
        axes[2].imshow(overlay, alpha=0.3)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # 保存
        filename = Path(image_path).stem
        save_path = self.vis_dir / f"inference_{filename}.png"
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info(f"推論結果を保存: {save_path}")

    def plot_loss_history(self):
        """損失履歴のプロット"""
        if not self.loss_history['steps']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 総合損失
        axes[0, 0].plot(self.loss_history['steps'], self.loss_history['total_loss'])
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # 言語損失
        axes[0, 1].plot(self.loss_history['steps'], self.loss_history['lm_loss'])
        axes[0, 1].set_title('Language Modeling Loss')
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
        
        # セグメンテーション損失
        axes[1, 0].plot(self.loss_history['steps'], self.loss_history['seg_loss'])
        axes[1, 0].set_title('Segmentation Loss')
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)
        
        # 学習率
        axes[1, 1].plot(self.loss_history['steps'], self.loss_history['learning_rate'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # 保存
        plot_path = self.output_dir / 'loss_history.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info(f"損失履歴プロットを保存: {plot_path}")
        
        # JSONでも保存
        json_path = self.output_dir / 'loss_history.json'
        with open(json_path, 'w') as f:
            json.dump(self.loss_history, f, indent=2)

    def train(self):
        """学習のメインループ"""
        logger.info("=" * 80)
        logger.info("学習を開始")
        logger.info("=" * 80)
        
        # アライメント段階（一時的に無効化 - qwen_modelアクセスエラー回避）
        # if not getattr(self.config, 'skip_alignment', False):
        #     self.run_alignment_stage()
        
        # 学習ループ
        for epoch in range(self.num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{self.num_epochs}")
            logger.info("-" * 40)
            
            # エポック学習
            avg_loss = self.train_epoch(epoch)
            
            # チェックポイント保存
            self.save_checkpoint()
            
            # 推論評価
            if (epoch + 1) % max(1, self.num_epochs // 3) == 0:
                self.run_inference_evaluation()
        
        # 最終チェックポイント
        self.save_checkpoint()
        
        # 損失履歴のプロット
        self.plot_loss_history()
        
        logger.info("=" * 80)
        logger.info("学習完了")
        logger.info(f"出力ディレクトリ: {self.output_dir}")
        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="LISA改ミニマルトレーニング")
    
    # データ設定
    parser.add_argument('--data_dir', type=str, default=None,
                       help='データセットのベースディレクトリ（Noneの場合はLISAConfigのデフォルトを使用）')
    parser.add_argument('--dataset_root', type=str, default='/home/soya/datasets',
                       help='データセットのルートディレクトリ')
    parser.add_argument('--samples_per_epoch', type=int, default=10,
                       help='1エポックあたりのサンプル数')
    parser.add_argument('--dataset_types', type=str, default='sem_seg||refer_seg||vqa||reason_seg',
                       help='データセットタイプ（||で区切る）: sem_seg||refer_seg||vqa||reason_seg')
    parser.add_argument('--sample_rates', type=str, default='9,3,3,1',
                       help='各データセットのサンプルレート（,で区切る）例: 9,3,3,1')
    
    # 訓練設定
    parser.add_argument('--batch_size', type=int, default=4,
                       help='バッチサイズ')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='勾配累積ステップ数（実効バッチサイズ = batch_size * gradient_accumulation_steps）')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='エポック数')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                       help='ウォームアップ比率')
    
    # 学習率設定
    parser.add_argument('--adapter_lr', type=float, default=1e-3,
                       help='アダプター学習率')
    parser.add_argument('--lora_lr', type=float, default=1e-4,
                       help='LoRA学習率')
    parser.add_argument('--seg_token_lr', type=float, default=5e-5,
                       help='SEGトークン学習率')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='基本学習率')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    
    # LoRA設定
    parser.add_argument('--lora_r', type=int, default=8,
                       help='LoRAランク')
    parser.add_argument('--lora_alpha', type=int, default=32,
                       help='LoRAアルファ（32推奨：学習初期の発散を抑制）')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                       help='LoRAドロップアウト率')
    
    # モデル設定
    parser.add_argument('--freeze_sam', action='store_true',
                       help='SAMモデルを凍結（デフォルト）')
    parser.add_argument('--no_freeze_sam', dest='freeze_sam', action='store_false',
                       help='SAMモデルを凍結しない')
    parser.set_defaults(freeze_sam=True)
    
    parser.add_argument('--freeze_qwen', action='store_true',
                       help='Qwenベースモデルを凍結')
    parser.add_argument('--no_freeze_qwen', dest='freeze_qwen', action='store_false',
                       help='Qwenベースモデルを凍結しない（デフォルト）')
    parser.set_defaults(freeze_qwen=False)
    
    parser.add_argument('--use_lora', action='store_true',
                       help='LoRAを使用（デフォルト）')
    parser.add_argument('--no_use_lora', dest='use_lora', action='store_false',
                       help='LoRAを使用しない')
    parser.set_defaults(use_lora=True)
    
    parser.add_argument('--tune_qwen_visual', action='store_true',
                       help='Qwenビジュアルエンコーダーをチューニング')
    
    parser.add_argument('--use_dynamic_resolution', action='store_true',
                       help='動的解像度を使用（デフォルト）')
    parser.add_argument('--no_dynamic_resolution', dest='use_dynamic_resolution', action='store_false',
                       help='動的解像度を無効化')
    parser.set_defaults(use_dynamic_resolution=True)
    
    parser.add_argument('--use_quality_score', action='store_true',
                       help='品質スコアを使用')
    
    # 損失設定
    parser.add_argument('--seg_loss_weight', type=float, default=1.0,
                       help='セグメンテーション損失の重み')
    
    # アライメントステージ設定
    parser.add_argument('--align_steps', type=int, default=0,
                       help='埋め込みアライメントステップ数（0の場合は実行しない）')
    parser.add_argument('--use_cached_alignment', action='store_true',
                       help='既存のアライメントチェックポイントを自動検出して使用')
    parser.add_argument('--alignment_checkpoint', type=str, default=None,
                       help='読み込むアライメントチェックポイントのパス')
    parser.add_argument('--force_realign', action='store_true',
                       help='キャッシュを無視して再アライメント')
    parser.add_argument('--skip_alignment', action='store_true',
                       help='アライメントステージをスキップ')
    
    # その他
    parser.add_argument('--save_steps', type=int, default=100,
                       help='チェックポイント保存間隔')
    parser.add_argument('--use_wandb', action='store_true',
                       help='WandBを使用')
    parser.add_argument('--wandb_project', type=str, default='lisa-kai-minimal',
                       help='WandBプロジェクト名')
    parser.add_argument('--debug', action='store_true',
                       help='デバッグモードを有効化')
    parser.add_argument('--fast_dev_run', action='store_true',
                       help='高速開発モード（少量データで動作確認）')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='各データセットから読み込む最大サンプル数（開発時用）')
    
    # 可視化設定
    parser.add_argument('--no_visualize', action='store_true',
                       help='訓練中の可視化を無効化（デフォルトは有効）')
    parser.add_argument('--visualize_steps', type=int, default=5,
                       help='可視化の間隔（ステップ数）')
    parser.add_argument('--run_inference_eval', action='store_true',
                       help='チェックポイント保存時に推論評価を実行')
    parser.add_argument('--inference_eval_samples', type=int, default=3,
                       help='推論評価で使用するサンプル数（最大3）')
    
    # シード設定
    parser.add_argument('--seed', type=int, default=42,
                       help='ランダムシード（再現性のため）')
    
    args = parser.parse_args()
    
    # データセット設定の検証
    valid_dataset_types = {'sem_seg', 'refer_seg', 'vqa', 'reason_seg'}
    dataset_types = args.dataset_types.split('||')
    for dt in dataset_types:
        if dt not in valid_dataset_types:
            raise ValueError(f"無効なデータセットタイプ: {dt}. 有効なタイプ: {valid_dataset_types}")
    
    # WandBの初期化
    if args.use_wandb:
        # 環境変数からAPIキーを設定（なければデフォルトを使用）
        import os
        if 'WANDB_API_KEY' not in os.environ:
            os.environ['WANDB_API_KEY'] = 'a389f0d40902815f4eaf9f4dd9e298b722db36e9'
        
        wandb.init(project=args.wandb_project, config=vars(args))
    
    # 訓練の実行
    trainer = MinimalTrainer(args)
    trainer.setup_model_and_data()
    trainer.setup_optimizer_and_scheduler()
    
    # アライメントステージの実行（オプション）
    if hasattr(trainer, 'run_alignment_stage') and hasattr(trainer.config, 'align_steps') and trainer.config.align_steps > 0:
        trainer.run_alignment_stage()
    
    # 本学習の実行
    trainer.train()
    
    # WandBの終了
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
    
# 使用例:
# python minimal_train.py --dataset_types sem_seg --samples_per_epoch 1000
# python minimal_train.py --dataset_types "sem_seg||refer_seg" --sample_rates "7,3" --samples_per_epoch 1000
# python minimal_train.py --dataset_types "sem_seg||refer_seg||vqa||reason_seg" --sample_rates "9,3,3,1" --samples_per_epoch 10000