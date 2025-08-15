#!/usr/bin/env python3
"""
実データを使用したミニマルなトレーニングスクリプト
セマンティックセグメンテーションデータセット（ADE20K）に絞って実装
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm
import numpy as np
import wandb
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # バックエンドを設定

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent))

from src.config import LISAConfig
from src.models import LISA_Model
from src.utils import prepare_tokenizer_for_lisa
from src.data.dataset import HybridDataset
from src.data.collators import MultiModalDataCollator
from transformers import AutoProcessor, get_linear_schedule_with_warmup

# ロギング設定
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.DEBUG if '--debug' in sys.argv else logging.INFO
)
logger = logging.getLogger(__name__)


class MinimalTrainer:
    """ミニマルなトレーニングクラス"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_loss = float('inf')
        self.global_step = 0
        
        # 可視化設定
        self.visualize = getattr(config, 'visualize', False)
        self.visualize_steps = getattr(config, 'visualize_steps', 5)
        
        # Loss履歴を記録
        self.loss_history = {
            'steps': [],
            'total_loss': [],
            'lm_loss': [],
            'seg_loss': [],
            'learning_rate': []
        }
        
        # 保存ディレクトリの設定
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"outputs/minimal_train_{timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # チェックポイントディレクトリ
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # 可視化ディレクトリ（可視化が有効な場合のみ）
        if self.visualize:
            self.vis_dir = self.output_dir / 'visualizations'
            self.vis_dir.mkdir(exist_ok=True)
            logger.info(f"可視化を有効化: {self.visualize_steps}ステップごとに保存")
        
        # 設定を保存
        with open(self.output_dir / "config.json", 'w') as f:
            json.dump(vars(config), f, indent=2)
    

    def setup_model_and_data(self):
        """モデルとデータセットのセットアップ"""
        logger.info("モデルとデータセットのセットアップを開始")
        
        # LISAConfig作成
        # すべての設定はconfig.pyで一元管理
        # ここでは環境依存の設定（デバイス、モデルパス）のみオーバーライド
        self.lisa_config = LISAConfig(
            # 環境依存の設定のみオーバーライド
            qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
            sam_model_name="./checkpoints/sam2.1_hiera_large.pt",
            device_map=str(self.device),
            torch_dtype="auto",
            use_flash_attention=False,
            # その他すべての設定（LoRA、学習率、損失重み等）はconfig.pyから自動的に読み込まれる
        )
        
        # トークナイザーとプロセッサの準備
        logger.info("トークナイザーとプロセッサの準備")
        self.tokenizer = prepare_tokenizer_for_lisa(
            model_name=self.lisa_config.qwen_model_name,
            seg_token=self.lisa_config.seg_token
        )
        # AutoProcessorに動的解像度の設定を追加
        if self.lisa_config.use_dynamic_resolution:
            logger.info(f"動的解像度モード有効: min_pixels={self.lisa_config.qwen_min_pixels}, max_pixels={self.lisa_config.qwen_max_pixels}")
            self.processor = AutoProcessor.from_pretrained(
                self.lisa_config.qwen_model_name,
                min_pixels=self.lisa_config.qwen_min_pixels,
                max_pixels=self.lisa_config.qwen_max_pixels
            )
        else:
            logger.info("固定解像度モード")
            self.processor = AutoProcessor.from_pretrained(self.lisa_config.qwen_model_name)
        
        # 重要: ProcessorのトークナイザーにもSEGトークンを追加
        if self.lisa_config.seg_token not in self.processor.tokenizer.get_vocab():
            logger.info(f"ProcessorのトークナイザーにSEGトークンを追加")
            self.processor.tokenizer.add_special_tokens({"additional_special_tokens": [self.lisa_config.seg_token]})
            logger.info(f"Processor語彙サイズ: {len(self.processor.tokenizer)}")
        
        # モデルの読み込み
        logger.info("LISA改モデルの読み込み")
        self.model = LISA_Model(self.lisa_config)
        
        # 融合タイプのログ出力
        if hasattr(self.model, 'fusion_type'):
            fusion_type = self.model.fusion_type
            logger.info(f"特徴融合モード: {fusion_type}")
            if fusion_type == 'sigma_add':
                if hasattr(self.model, 'image_fusion_beta'):
                    beta_value = torch.sigmoid(self.model.image_fusion_beta).item()
                    logger.info(f"  - Sigma-Add融合 (学習可能パラメータ: image_fusion_beta={beta_value:.4f})")
            elif fusion_type == 'cross_attention':
                fusion_params = sum(p.numel() for p in self.model.image_fusion.parameters())
                logger.info(f"  - Cross-Attention融合 (パラメータ数: {fusion_params:,})")
        
        # 重要: LoRA適用前にトークナイザーを設定してresize_token_embeddingsを実行
        logger.info("トークナイザーを設定してSEGトークンの埋め込みをリサイズ")
        self.model.set_tokenizer(self.tokenizer)
        
        # Qwen LoRAはLISA_Model内で設定されるため、ここでは何もしない
        # config.freeze_qwen_lora=FalseならLISA_Model.__init__でLoRAが適用される
        
        # SAM MaskDecoderへのLoRA適用（設定されている場合）
        if hasattr(self.lisa_config, 'sam_lora_r') and self.lisa_config.sam_lora_r > 0:
            logger.info(f"SAM MaskDecoderにLoRAを適用 (r={self.lisa_config.sam_lora_r})")
            self.model.add_sam_lora(
                lora_r=self.lisa_config.sam_lora_r,
                lora_alpha=self.lisa_config.sam_lora_alpha,
                lora_dropout=self.lisa_config.sam_lora_dropout
            )
        
        # デバイスに移動
        self.model = self.model.to(self.device)
        
        # パラメータ統計の詳細表示
        from src.utils.model_utils import display_parameter_statistics
        display_parameter_statistics(self.model, logger_name=__name__)
        
        # データセットの作成
        logger.info("データセットの作成")
        # LISAConfigのデータセットパスを使用
        data_dir = self.lisa_config.dataset_base_dir if self.config.data_dir is None else self.config.data_dir
        logger.info(f"データセットディレクトリ: {data_dir}")
        
        # データセットタイプとサンプルレートの設定
        dataset_types = self.config.dataset_types.split('||')
        sample_rates = [float(x) for x in self.config.sample_rates.split(',')]
        
        # 長さが一致しない場合は均等に分配
        if len(sample_rates) != len(dataset_types):
            logger.warning(f"sample_rates数({len(sample_rates)})とdataset_types数({len(dataset_types)})が一致しません")
            sample_rates = [1.0 / len(dataset_types)] * len(dataset_types)
            logger.info(f"均等なサンプルレートを使用: {sample_rates}")
        
        logger.info(f"データセットタイプ: {dataset_types}")
        logger.info(f"サンプルレート: {sample_rates}")
        
        self.train_dataset = HybridDataset(
            base_image_dir=data_dir,
            qwen_processor=self.processor,
            samples_per_epoch=self.config.samples_per_epoch,
            dataset='||'.join(dataset_types),
            sample_rate=sample_rates,
            qwen_image_size=self.lisa_config.qwen_image_size,
            sam_image_size=self.lisa_config.sam_image_size,
        )
        
        # DataLoaderの作成
        self.collator = MultiModalDataCollator(
            tokenizer=self.tokenizer,
            max_length=self.lisa_config.model_max_length,  # 動的解像度対応の最大長
            config=self.lisa_config  # configを渡す
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.collator,
            num_workers=0,  # Set to 0 for debugging to avoid duplicate messages
            pin_memory=True
        )
        
        logger.info(f"データセットサイズ: {len(self.train_dataset)}")
        logger.info(f"バッチ数: {len(self.train_loader)}")
    
    def setup_optimizer_and_scheduler(self):
        """オプティマイザとスケジューラのセットアップ"""
        # パラメータグループの作成
        adapter_params = []
        lora_params = []
        seg_token_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "adapter" in name or "prompt_proj" in name or "prompt_beta" in name:
                    adapter_params.append(param)
                elif "lora" in name:
                    lora_params.append(param)
                elif "word_embeddings" in name:
                    seg_token_params.append(param)
        
        # パラメータグループ（学習率はLISAConfigから取得）
        param_groups = [
            {"params": adapter_params, "lr": self.lisa_config.adapter_lr},
            {"params": lora_params, "lr": self.lisa_config.lora_lr},
            {"params": seg_token_params, "lr": self.lisa_config.seg_token_lr}
        ]
        
        # オプティマイザ（weight_decayもLISAConfigから取得）
        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.lisa_config.weight_decay
        )
        
        # スケジューラ
        # Gradient Accumulationを考慮した実際の最適化ステップ数
        # 切り上げ処理で最後のバッチも含める
        import math
        num_training_steps = math.ceil(len(self.train_loader) / self.config.gradient_accumulation_steps) * self.config.num_epochs
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        logger.info(f"総ステップ数: {num_training_steps}")
        logger.info(f"ウォームアップステップ数: {num_warmup_steps}")

    def get_alignment_checkpoint_path(self):
        """アライメントチェックポイントのパスを取得"""
        if self.config.alignment_checkpoint:
            return Path(self.config.alignment_checkpoint)
        
        # 自動生成パス: checkpoints/alignment/align_{steps}.pt
        align_dir = Path("checkpoints/alignment")
        align_dir.mkdir(parents=True, exist_ok=True)
        return align_dir / f"align_{self.config.align_steps}.pt"
    
    def save_alignment_checkpoint(self):
        """アライメント完了後の状態を保存（最適化版）"""
        checkpoint_path = self.get_alignment_checkpoint_path()
        
        # アライメントで学習したパラメータのみを保存
        alignment_state = {}
        
        # 1. Qwen LoRAパラメータ
        for name, param in self.model.named_parameters():
            if "qwen" in name and "lora" in name:
                alignment_state[name] = param.data.cpu()
        
        # 2. SEGトークン埋め込み（個別管理）
        if hasattr(self.model, 'seg_token_embedding') and self.model.seg_token_embedding is not None:
            alignment_state['seg_token_embedding'] = self.model.seg_token_embedding.data.cpu()
        
        # 3. TextPromptProjector
        alignment_state['text_prompt_proj_state'] = self.model.text_prompt_proj.state_dict()
        
        # 4. prompt_beta
        if hasattr(self.model, 'prompt_beta'):
            alignment_state['prompt_beta'] = self.model.prompt_beta.data.cpu()
        
        checkpoint = {
            'alignment_state': alignment_state,  # 学習済みパラメータのみ
            'alignment_steps': self.config.align_steps,
            'beta_value': torch.sigmoid(self.model.prompt_beta).item() if hasattr(self.model, 'prompt_beta') else None,
            'timestamp': datetime.now().isoformat(),
            'seg_token_id': self.model.seg_token_id if hasattr(self.model, 'seg_token_id') else None,
            'config': {
                'lora_lr': self.lisa_config.lora_lr,
                'seg_token_lr': self.lisa_config.seg_token_lr,
                'adapter_lr': self.lisa_config.adapter_lr,
            }
        }
        
        torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=True)
        
        # ファイルサイズを確認
        file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        logger.info(f"アライメントチェックポイントを保存: {checkpoint_path} ({file_size_mb:.1f} MB)")
        
        # メタデータファイルも保存（人間が読める形式）
        meta_path = checkpoint_path.with_suffix('.json')
        with open(meta_path, 'w') as f:
            json.dump({
                'alignment_steps': self.config.align_steps,
                'beta_value': checkpoint['beta_value'],
                'timestamp': checkpoint['timestamp'],
                'seg_token_id': checkpoint['seg_token_id'],
                'config': checkpoint['config'],
                'file_size_mb': file_size_mb
            }, f, indent=2)

    def _apply_alignment_freeze_settings(self):
        """アライメントステージ専用の凍結設定を適用"""
        logger.info("アライメントステージ用の凍結設定を適用中...")
        
        frozen_count = 0
        unfrozen_count = 0
        
        for name, param in self.model.named_parameters():
            should_train = False
            
            # アライメントステージで学習対象となるパラメータを判定
            # 1. QwenのLoRAパラメータ（Configに従う）
            if "qwen" in name and "lora" in name and not self.lisa_config.freeze_qwen_lora:
                should_train = True
            # 2. SEGトークンEmbedding（Configに従う）
            # seg_token_embeddingは個別のパラメータとして管理されている
            elif "seg_token_embedding" in name and not self.lisa_config.freeze_seg_token:
                should_train = True
            # 3. TextPromptProjector（Configに従う）
            elif "text_prompt_proj" in name and not self.lisa_config.freeze_text_prompt_projector:
                should_train = True
            # 4. 融合β（Configに従う）
            elif "prompt_beta" in name and not self.lisa_config.freeze_prompt_beta:
                should_train = True
            # それ以外は全て凍結（Configに関係なく）
            else:
                should_train = False
            
            # パラメータの学習可能フラグを設定
            param.requires_grad = should_train
            
            if should_train:
                unfrozen_count += 1
                logger.debug(f"[Align] Unfrozen: {name}")
            else:
                frozen_count += 1
                logger.debug(f"[Align] Frozen: {name}")
        
        logger.info(f"アライメントステージ設定完了: 学習可能={unfrozen_count:,}, 凍結={frozen_count:,}")
        
        # 主要コンポーネントの状態をログ出力
        logger.info("アライメントステージの主要コンポーネント状態:")
        logger.info(f"  - Qwen LoRA: {'学習可能' if not self.lisa_config.freeze_qwen_lora else '凍結'}")
        logger.info(f"  - SEG Token: {'学習可能' if not self.lisa_config.freeze_seg_token else '凍結'}")
        logger.info(f"  - Text Prompt Projector: {'学習可能' if not self.lisa_config.freeze_text_prompt_projector else '凍結'}")
        logger.info(f"  - Prompt Beta: {'学習可能' if not self.lisa_config.freeze_prompt_beta else '凍結'}")
        logger.info("  - SAM PromptEncoder: 凍結（強制）")
        logger.info("  - SAM MaskDecoder: 凍結（強制）")
        logger.info("  - SAM ImageEncoder: 凍結（強制）")
        logger.info("  - Image Adapter: 凍結（強制）")
        logger.info("  - Token-FPN: 凍結（強制）")
    
    def _restore_training_freeze_settings(self):
        """アライメント後に通常の学習用凍結設定に戻す"""
        logger.info("通常学習用の凍結設定に復元中...")
        
        # 通常の凍結設定を再適用（lisa_model.pyの_setup_freeze_settingsと同等）
        for name, param in self.model.named_parameters():
            should_train = False
            
            # 通常の学習設定に従って判定
            if "qwen" in name and "lora" in name and not self.lisa_config.freeze_qwen_lora:
                should_train = True
            elif "word_embeddings" in name and not self.lisa_config.freeze_seg_token:
                should_train = True
            elif "sam" in name and "lora" in name and not self.lisa_config.freeze_sam_lora:
                should_train = True
            elif "image_adapter" in name and not self.lisa_config.freeze_image_adapter:
                should_train = True
            elif "text_prompt_proj" in name and not self.lisa_config.freeze_text_prompt_projector:
                should_train = True
            elif "token_fpn" in name and not self.lisa_config.freeze_token_fpn:
                should_train = True
            elif "prompt_beta" in name and not self.lisa_config.freeze_prompt_beta:
                should_train = True
            # 明示的な凍結設定
            elif "sam" in name and "image_encoder" in name and self.lisa_config.freeze_sam_image_encoder:
                should_train = False
            elif "sam" in name and "memory" in name and self.lisa_config.freeze_sam_memory_attention:
                should_train = False
            elif "sam" in name and "mask_decoder" in name and self.lisa_config.freeze_sam_mask_decoder_base:
                should_train = False
            elif "sam" in name and "prompt_encoder" in name and self.lisa_config.freeze_sam_prompt_encoder:
                should_train = False
            
            param.requires_grad = should_train
        
        logger.info("通常学習用設定の復元が完了しました")
    
    def load_alignment_checkpoint(self, checkpoint_path):
        """保存済みアライメントチェックポイントを読み込み（最適化版）"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 新形式と旧形式の両方に対応
        if 'alignment_state' in checkpoint:
            # 新形式：最適化された保存形式
            alignment_state = checkpoint['alignment_state']
            
            # 1. Qwen LoRAパラメータを復元
            for name, param in self.model.named_parameters():
                if name in alignment_state:
                    param.data.copy_(alignment_state[name].to(self.device))
            
            # 2. SEGトークン埋め込みを復元
            if 'seg_token_embedding' in alignment_state:
                if hasattr(self.model, 'seg_token_embedding'):
                    self.model.seg_token_embedding.data.copy_(
                        alignment_state['seg_token_embedding'].to(self.device)
                    )
            
            # 3. TextPromptProjectorを復元
            if 'text_prompt_proj_state' in alignment_state:
                self.model.text_prompt_proj.load_state_dict(alignment_state['text_prompt_proj_state'])
            
            # 4. prompt_betaを復元
            if 'prompt_beta' in alignment_state and hasattr(self.model, 'prompt_beta'):
                self.model.prompt_beta.data.copy_(
                    alignment_state['prompt_beta'].to(self.device)
                )
            
            # SEGトークンIDを復元
            if 'seg_token_id' in checkpoint:
                self.model.seg_token_id = checkpoint['seg_token_id']
                
        elif 'model_state_dict' in checkpoint:
            # 旧形式：全体のstate_dictを保存している場合（後方互換性）
            logger.warning("旧形式のアライメントチェックポイントを検出。新形式への移行を推奨します。")
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            raise ValueError(f"不明なチェックポイント形式: {checkpoint_path}")
        
        logger.info(f"アライメントチェックポイントを読み込みました: {checkpoint_path}")
        logger.info(f"  - アライメントステップ: {checkpoint['alignment_steps']}")
        if checkpoint.get('beta_value'):
            logger.info(f"  - Beta値: {checkpoint['beta_value']:.4f}")
        logger.info(f"  - 保存日時: {checkpoint['timestamp']}")
        
        return checkpoint
    
    def check_alignment_cache(self):
        """既存のアライメントチェックポイントをチェック"""
        if self.config.force_realign:
            logger.info("--force_realignが指定されたため、再アライメントを実行します")
            return False
        
        checkpoint_path = self.get_alignment_checkpoint_path()
        
        if checkpoint_path.exists():
            # メタデータを確認
            meta_path = checkpoint_path.with_suffix('.json')
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                logger.info(f"既存のアライメントチェックポイントを検出:")
                logger.info(f"  - ステップ数: {meta['alignment_steps']}")
                logger.info(f"  - Beta値: {meta.get('beta_value', 'N/A')}")
                logger.info(f"  - 作成日時: {meta['timestamp']}")
            return True
        
        return False
    
    def run_alignment_stage(self):
        """埋め込みアライメントステージの実行"""
        if self.config.align_steps <= 0:
            return
        
        # キャッシュされたアライメントをチェック
        if self.config.use_cached_alignment and self.check_alignment_cache():
            checkpoint_path = self.get_alignment_checkpoint_path()
            logger.info(f"キャッシュされたアライメントを使用します: {checkpoint_path}")
            self.load_alignment_checkpoint(checkpoint_path)
            return
        
        logger.info(f"ステージ0: 埋め込みアライメントを{self.config.align_steps}ステップ実行します")
        
        # アライメントステージ専用の凍結設定を適用
        self._apply_alignment_freeze_settings()
        
        self.model.train()
        
        # アライメント用のオプティマイザを構築
        logger.info("アライメント用オプティマイザを構築中...")
        align_params = []
        total_params_checked = 0
        for name, param in self.model.named_parameters():
            total_params_checked += 1
            if not param.requires_grad:
                continue
            
            # アライメントステージで学習対象となるパラメータのみを追加
            # 1. QwenのLoRAパラメータ（Configに従う）
            if "qwen" in name and "lora" in name and not self.lisa_config.freeze_qwen_lora:
                align_params.append({"params": param, "lr": self.lisa_config.lora_lr})
                logger.debug(f"[Align] Added Qwen LoRA param: {name}")
            # 2. SEGトークンEmbedding（Configに従う）
            # seg_token_embeddingは個別のパラメータとして管理されている
            elif "seg_token_embedding" in name and not self.lisa_config.freeze_seg_token:
                align_params.append({"params": param, "lr": self.lisa_config.seg_token_lr})
                logger.debug(f"[Align] Added SEG token param: {name}")
            # 3. TextPromptProjector（Configに従う）
            elif "text_prompt_proj" in name and not self.lisa_config.freeze_text_prompt_projector:
                align_params.append({"params": param, "lr": self.lisa_config.adapter_lr})
                logger.debug(f"[Align] Added TextPromptProjector param: {name}")
            # 4. 融合β（Configに従う）
            elif "prompt_beta" in name and not self.lisa_config.freeze_prompt_beta:
                align_params.append({"params": param, "lr": self.lisa_config.adapter_lr})
                logger.debug(f"[Align] Added prompt_beta param: {name}")
            else:
                # アライメントステージでは学習対象外
                logger.debug(f"[Align] Skipped param: {name} (requires_grad={param.requires_grad})")
        
        logger.info(f"アライメント用パラメータ検索完了: {total_params_checked}個のパラメータをチェック")
        logger.info(f"アライメント対象パラメータ数: {len(align_params)}")
        
        if not align_params:
            logger.warning("アライメント対象のパラメータがありません")
            return
        
        logger.info("AdamWオプティマイザを作成中...")
        align_optimizer = torch.optim.AdamW(align_params)
        logger.info("オプティマイザ作成完了")
        
        # データローダーのイテレータ
        logger.info("データローダーのイテレータを作成中...")
        data_iter = iter(self.train_loader)
        logger.info("イテレータ作成完了")
        
        # アライメント学習ループ
        logger.info(f"アライメント学習ループ開始: {self.config.align_steps}ステップ")
        for step in range(self.config.align_steps):
            if step % 50 == 0 and step > 0:  # 50ステップごとに進捗表示
                logger.info(f"アライメントステップ {step}/{self.config.align_steps}")
            
            # バッチ取得
            try:
                if step == 0:
                    logger.info("最初のバッチを取得中...")
                batch = next(data_iter)
                if step == 0:
                    logger.info(f"バッチ取得成功: keys={list(batch.keys())}")
            except StopIteration:
                logger.info("データローダーを再初期化")
                data_iter = iter(self.train_loader)
                batch = next(data_iter)
            
            # デバイスへ転送
            if step == 0:
                logger.info("デバイスへバッチデータを転送中...")
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)
            if step == 0:
                logger.info("デバイス転送完了")
            
            # フォワードパス
            forward_kwargs = {
                'input_ids': batch['input_ids'],
                'pixel_values': batch['pixel_values'],
                'attention_mask': batch['attention_mask'],
                'labels': batch['labels'],
                'mask_labels': batch['mask_labels']
            }
            
            # image_grid_thwがある場合は追加
            if 'image_grid_thw' in batch and batch['image_grid_thw'] is not None:
                forward_kwargs['image_grid_thw'] = batch['image_grid_thw']
            
            if step == 0:
                logger.info("モデルのフォワードパスを実行中...")
                logger.info(f"  input_ids shape: {batch['input_ids'].shape}")
                logger.info(f"  pixel_values shape: {batch['pixel_values'].shape}")
                if 'image_grid_thw' in batch and batch['image_grid_thw'] is not None:
                    logger.info(f"  image_grid_thw: {batch['image_grid_thw']}")
            
            outputs = self.model(**forward_kwargs)
            
            if step == 0:
                logger.info("フォワードパス完了")
            
            # LLM埋め込みとSAM埋め込みを取得してMSEロスを計算
            if step == 0:
                logger.info("アライメント損失の計算開始...")
            
            align_loss = torch.tensor(0.0, device=self.device)
            align_count = 0
            
            if step == 0:
                logger.info(f"seg_token_positions: {outputs.seg_token_positions if hasattr(outputs, 'seg_token_positions') else 'Not found'}")
            
            if hasattr(outputs, 'seg_token_positions') and outputs.seg_token_positions is not None:
                if step == 0:
                    logger.info(f"seg_token_positions値: {outputs.seg_token_positions}")
                
                # language_hidden_statesを取得
                if not hasattr(outputs, 'language_hidden_states'):
                    if step == 0:
                        logger.warning("outputs.language_hidden_statesが存在しません")
                    continue
                    
                hidden_states = outputs.language_hidden_states
                if step == 0:
                    logger.info(f"hidden_states shape: {hidden_states.shape if hidden_states is not None else 'None'}")
                
                for i, seg_list in enumerate(outputs.seg_token_positions):
                    if len(seg_list) == 0:
                        continue
                    
                    if step == 0 and i == 0:
                        logger.info(f"Sample {i}: SEGトークン位置 = {seg_list}")
                    
                    # 最初のSEGトークン位置のみ使用
                    j = seg_list[0]
                    
                    # LLM埋め込みを取得（dtypeを統一）
                    e_llm = self.model.text_prompt_proj(hidden_states[i, j])
                    
                    # GTマスクから重心を計算してSAM埋め込みを取得
                    if batch['mask_labels'][i] is None:
                        continue
                    
                    gt_mask = batch['mask_labels'][i]
                    if isinstance(gt_mask, list):
                        gt_mask = gt_mask[0] if len(gt_mask) > 0 else None
                    
                    if gt_mask is None:
                        continue
                    
                    # 重心計算
                    orig_h = batch['pixel_values'].shape[-2] if batch['pixel_values'] is not None else 1024
                    orig_w = batch['pixel_values'].shape[-1] if batch['pixel_values'] is not None else 1024
                    
                    if step == 0 and i == 0:
                        logger.info(f"マスク重心計算中...")
                    
                    center = self.model.compute_mask_centroid(gt_mask, orig_h, orig_w)
                    cx, cy = int(center[0].item()), int(center[1].item())
                    
                    # SAM PromptEncoderから埋め込みを取得
                    point = torch.tensor([[cx, cy]], dtype=torch.float32, device=self.device)
                    point_label = torch.tensor([1], dtype=torch.int32, device=self.device)
                    
                    sparse_embeddings, _ = self.model.sam_prompt_encoder(
                        points=(point.unsqueeze(0), point_label.unsqueeze(0)),
                        boxes=None,
                        masks=None
                    )
                    
                    if sparse_embeddings.shape[1] == 0:
                        continue
                    
                    # SAM埋め込みをdetachして勾配を流さない
                    e_pos = sparse_embeddings[0, 0, :].detach()
                    
                    # MSEロス計算（dtypeを統一）
                    align_loss += torch.nn.functional.mse_loss(e_llm.float(), e_pos.float())
                    align_count += 1
            
            if step == 0:
                logger.info(f"アライメント損失計算完了: align_count={align_count}")
            
            if align_count > 0:
                align_loss = align_loss / align_count
                
                if step == 0:
                    logger.info(f"損失値: {align_loss.item():.6f}")
                    logger.info("逆伝播開始...")
                
                # 逆伝播と最適化
                align_optimizer.zero_grad()
                align_loss.backward()
                
                if step == 0:
                    logger.info("逆伝播完了、最適化ステップ実行...")
                    
                align_optimizer.step()
                
                if step == 0:
                    logger.info("最適化ステップ完了")
                
                if (step + 1) % 10 == 0:  # 10ステップごとに表示（100から変更）
                    logger.info(f"[Align Stage] Step {step+1}/{self.config.align_steps}, align_loss={align_loss.item():.6f}")
            else:
                if step == 0:
                    logger.warning(f"アライメント損失が計算されませんでした (align_count=0)")
        
        # モデル全体の勾配をリセット
        self.model.zero_grad(set_to_none=True)
        
        logger.info("ステージ0完了。LLMの<SEG>埋め込みを事前調整しました。")
        
        # アライメントチェックポイントを保存
        self.save_alignment_checkpoint()
        
        # 通常の学習用凍結設定に復元
        self._restore_training_freeze_settings()
    
    def compute_loss(self, outputs, labels, mask_labels):
        """損失計算（1会話1マスクに最適化）"""
        # 言語モデリング損失
        vocab_size = outputs.logits.size(-1)
        lm_loss = nn.functional.cross_entropy(
            outputs.logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=-100
        )
        
        # セグメンテーション損失（1会話1マスクに統一）
        seg_loss = 0.0
        seg_count = 0
        
        if outputs.mask_logits is not None:
            for batch_idx, batch_masks in enumerate(outputs.mask_logits):
                if batch_masks is not None and len(batch_masks) > 0:
                    gt_mask = mask_labels[batch_idx]
                    
                    # 1会話1マスクなので、最初のマスクのみを使用
                    pred_mask = batch_masks[0] if isinstance(batch_masks, list) else batch_masks
                    
                    # デバッグ情報
                    logger.debug(f"pred_mask shape: {pred_mask.shape}")
                    logger.debug(f"gt_mask shape: {gt_mask.shape}")
                    
                    # マスクの次元を確認して適切に処理
                    # pred_maskとgt_maskの形状を合わせる
                    if pred_mask.dim() == 2:
                        pred_h, pred_w = pred_mask.shape
                    else:
                        pred_h, pred_w = pred_mask.shape[-2:]
                        
                    if gt_mask.dim() == 2:
                        gt_h, gt_w = gt_mask.shape
                    else:
                        gt_h, gt_w = gt_mask.shape[-2:]
                    
                    # サイズが異なる場合はリサイズ
                    if (pred_h, pred_w) != (gt_h, gt_w):
                        # gt_maskを(B, C, H, W)形式に変換
                        if gt_mask.dim() == 2:
                            gt_mask_4d = gt_mask.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
                        elif gt_mask.dim() == 3:
                            # 最初のマスクのみを使用（1会話1マスク）
                            if gt_mask.shape[0] > 1:
                                gt_mask = gt_mask[0]  # 複数マスクの場合は最初のみ
                                gt_mask_4d = gt_mask.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
                            else:
                                gt_mask_4d = gt_mask.unsqueeze(1)  # (1, H, W) -> (1, 1, H, W)
                        else:
                            gt_mask_4d = gt_mask
                        
                        # 4次元であることを確認
                        if gt_mask_4d.dim() != 4:
                            logger.debug(f"gt_mask_4d shape before fix: {gt_mask_4d.shape}")
                            if gt_mask_4d.dim() == 3:
                                gt_mask_4d = gt_mask_4d.unsqueeze(0)
                            elif gt_mask_4d.dim() == 2:
                                gt_mask_4d = gt_mask_4d.unsqueeze(0).unsqueeze(0)
                        
                        # float型に変換してinterpolate
                        gt_mask_4d = gt_mask_4d.float()
                        gt_mask_resized = nn.functional.interpolate(
                            gt_mask_4d,
                            size=(pred_h, pred_w),
                            mode='nearest'
                        )
                        
                        # 元の次元に戻す（(1, 1, H, W) -> (H, W)）
                        gt_mask_resized = gt_mask_resized.squeeze(0).squeeze(0)
                    else:
                        gt_mask_resized = gt_mask
                        # 複数マスクの場合は最初のマスクのみを使用
                        if gt_mask_resized.dim() == 3 and gt_mask_resized.shape[0] > 1:
                            gt_mask_resized = gt_mask_resized[0]
                    
                    # pred_maskも適切な形状に
                    if pred_mask.dim() > 2:
                        pred_mask = pred_mask.squeeze(0)
                    
                    # BCE損失
                    bce_loss = nn.functional.binary_cross_entropy_with_logits(
                        pred_mask,
                        gt_mask_resized.float()
                    )
                    
                    # Dice損失
                    pred_sigmoid = torch.sigmoid(pred_mask)
                    intersection = (pred_sigmoid * gt_mask_resized).sum()
                    dice = 2 * intersection / (pred_sigmoid.sum() + gt_mask_resized.sum() + 1e-8)
                    dice_loss = 1 - dice
                    
                    seg_loss += bce_loss + dice_loss
                    seg_count += 1
        
        # 平均セグメンテーション損失
        if seg_count > 0:
            seg_loss = seg_loss / seg_count
        
        # 総合損失（損失重みもLISAConfigから取得）
        total_loss = lm_loss + self.lisa_config.segmentation_loss_weight * seg_loss
        
        return total_loss, lm_loss, seg_loss
    
    def train_epoch(self, epoch):
        """1エポックの学習"""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_lm_loss = 0.0
        epoch_seg_loss = 0.0
        
        # Gradient Accumulation用の変数
        accumulation_steps = self.config.gradient_accumulation_steps
        accumulated_loss = 0.0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # デバイスに移動
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # デバッグ: トークン数を確認
            if batch_idx == 0 or self.config.debug:
                img_tokens = batch['pixel_values'].shape[1] if batch['pixel_values'].dim() == 3 else 0
                txt_tokens = batch['input_ids'].shape[1]
                total_tokens = img_tokens + txt_tokens
                logger.debug(f"Batch {batch_idx}: img_tokens={img_tokens}, txt_tokens={txt_tokens}, total={total_tokens}")
                if batch['pixel_values'].dim() == 3:
                    logger.debug(f"pixel_values shape: {batch['pixel_values'].shape}")
                logger.debug(f"input_ids shape: {batch['input_ids'].shape}")
                if 'image_grid_thw' in batch and batch['image_grid_thw'] is not None:
                    logger.debug(f"image_grid_thw: {batch['image_grid_thw']}")
                
                # データセットタイプ別の統計（初回のみ）
                if batch_idx == 0 and hasattr(self.train_dataset, 'dataset_indices'):
                    dataset_stats = {}
                    for i, ds in enumerate(self.train_dataset.all_datasets):
                        ds_name = type(ds).__name__
                        dataset_stats[ds_name] = 0
                    
                    # 現在のエポックのサンプル分布を計算
                    for idx in self.train_dataset.dataset_indices[:self.config.samples_per_epoch]:
                        ds_name = type(self.train_dataset.all_datasets[idx[0]]).__name__
                        dataset_stats[ds_name] += 1
                    
                    logger.info("データセットサンプル分布:")
                    for ds_name, count in dataset_stats.items():
                        percentage = (count / self.config.samples_per_epoch) * 100
                        logger.info(f"  {ds_name}: {count} ({percentage:.1f}%)")
            
            # Forward pass
            forward_kwargs = {
                'input_ids': batch['input_ids'],
                'pixel_values': batch['pixel_values'],
                'attention_mask': batch['attention_mask'],
                'labels': batch['labels'],
                'mask_labels': [batch['mask_labels'][i] for i in range(batch['mask_labels'].size(0))]
            }
            
            # image_grid_thwがある場合は追加
            if 'image_grid_thw' in batch:
                forward_kwargs['image_grid_thw'] = batch['image_grid_thw']
            
            outputs = self.model(**forward_kwargs)
            
            # 損失計算
            total_loss, lm_loss, seg_loss = self.compute_loss(
                outputs, batch['labels'], batch['mask_labels']
            )
            
            # Gradient Accumulationを考慮したloss
            # 各ミニバッチの損失を累積ステップ数で割る
            scaled_loss = total_loss / accumulation_steps
            
            # Backward pass
            scaled_loss.backward()
            accumulated_loss += total_loss.item()
            
            # Gradient Accumulation: 指定ステップごとに最適化
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                # 勾配クリッピング
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # 最適化ステップ
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # 累積損失をリセット
                accumulated_loss = 0.0
            
            # 損失の記録
            epoch_loss += total_loss.item()
            epoch_lm_loss += lm_loss.item()
            epoch_seg_loss += seg_loss.item() if isinstance(seg_loss, torch.Tensor) else seg_loss
            
            # βパラメータの値を取得
            prompt_beta_value = torch.sigmoid(self.model.prompt_beta).item() if hasattr(self.model, 'prompt_beta') else 0.0
            
            # Fusion betaの値を取得（Sigma-Add融合の場合）
            fusion_info = {}
            if hasattr(self.model, 'fusion_type'):
                fusion_info['fusion'] = self.model.fusion_type[:2]  # 'cr' for cross_attention, 'si' for sigma_add
                if self.model.fusion_type == 'sigma_add' and hasattr(self.model, 'image_fusion_beta'):
                    fusion_beta = torch.sigmoid(self.model.image_fusion_beta).item()
                    fusion_info['fβ'] = f"{fusion_beta:.3f}"
            
            # プログレスバーの更新
            postfix_dict = {
                'loss': f"{total_loss.item():.4f}",
                'lm': f"{lm_loss.item():.4f}",
                'seg': f"{seg_loss:.4f}" if isinstance(seg_loss, torch.Tensor) else f"{seg_loss:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
                'pβ': f"{prompt_beta_value:.3f}"  # prompt beta
            }
            postfix_dict.update(fusion_info)
            progress_bar.set_postfix(postfix_dict)
            
            # WandBログ（使用する場合）
            if self.config.use_wandb:
                wandb.log({
                    'train/loss': total_loss.item(),
                    'train/lm_loss': lm_loss.item(),
                    'train/seg_loss': seg_loss if isinstance(seg_loss, float) else seg_loss.item(),
                    'train/learning_rate': self.scheduler.get_last_lr()[0],
                    'train/epoch': epoch,
                    'train/step': self.global_step
                })
            
            # Loss履歴を記録
            self.loss_history['steps'].append(self.global_step)
            self.loss_history['total_loss'].append(total_loss.item())
            self.loss_history['lm_loss'].append(lm_loss.item())
            self.loss_history['seg_loss'].append(seg_loss.item() if isinstance(seg_loss, torch.Tensor) else seg_loss)
            self.loss_history['learning_rate'].append(self.scheduler.get_last_lr()[0])
            
            self.global_step += 1
            
            # 定期的な可視化を保存
            if 'mask_labels' in batch and batch['mask_labels'] is not None:
                self.save_visualization(batch, outputs, self.global_step)
            
            # 定期的なチェックポイント保存
            if self.global_step % self.config.save_steps == 0:
                self.save_checkpoint(f"step_{self.global_step}")
        
        # エポック平均
        avg_loss = epoch_loss / len(self.train_loader)
        avg_lm_loss = epoch_lm_loss / len(self.train_loader)
        avg_seg_loss = epoch_seg_loss / len(self.train_loader)
        
        logger.info(f"Epoch {epoch+1} - 平均損失: {avg_loss:.4f}, LM: {avg_lm_loss:.4f}, Seg: {avg_seg_loss:.4f}")
        
        # エポック終了時に可視化を保存
        self.plot_loss_history()
        
        return avg_loss
    
    def save_visualization(self, batch, outputs, step):
        """訓練中の予測を可視化（バッチの最初のサンプルのみ）"""
        # 可視化が無効の場合はスキップ
        if not self.visualize:
            return
            
        # 指定されたステップ間隔で可視化
        if step % self.visualize_steps != 0:
            return
        
        # マスクがない場合はスキップ
        if outputs.mask_logits is None or len(outputs.mask_logits) == 0:
            return
            
        if outputs.mask_logits[0] is None:
            return
        
        try:
            import cv2
            from PIL import Image
            import numpy as np
            import json
            
            # original_imagesがcollatorから渡されている場合はそれを使用
            if 'original_images' in batch and batch['original_images'] is not None and len(batch['original_images']) > 0:
                original_img = batch['original_images'][0]
                if isinstance(original_img, Image.Image):
                    image_np = np.array(original_img)
                elif isinstance(original_img, torch.Tensor):
                    img_tensor = original_img
                    if img_tensor.dim() == 4:
                        img_tensor = img_tensor[0]
                    if img_tensor.shape[0] == 3:
                        image_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                    else:
                        image_np = img_tensor.cpu().numpy()
                    if image_np.max() <= 1.0:
                        image_np = (image_np * 255).astype(np.uint8)
                else:
                    logger.debug(f"Skipping visualization at step {step} (unsupported image format)")
                    return
            else:
                # pixel_valuesから復元を試みる
                pixel_values = batch['pixel_values'][0].cpu()
                
                # パッチ形式の場合はスキップ
                if pixel_values.dim() == 2:
                    logger.debug(f"Skipping visualization at step {step} (patch format, no original_images)")
                    return
                
                # 3D形式の場合
                if pixel_values.dim() == 3:
                    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
                    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
                    image = pixel_values * std + mean
                    image = torch.clamp(image, 0, 1)
                    image_np = image.permute(1, 2, 0).numpy()
                else:
                    logger.debug(f"Skipping visualization at step {step} (unsupported pixel format)")
                    return
            
            # マスクを取得（バッチの最初のサンプル）
            pred_mask = outputs.mask_logits[0]
            if isinstance(pred_mask, list):
                pred_mask = pred_mask[0]
            
            # 予測マスクをシグモイドで確率に変換
            pred_mask_np = torch.sigmoid(pred_mask).detach().cpu().numpy()
            if pred_mask_np.ndim > 2:
                pred_mask_np = pred_mask_np.squeeze()
            
            # GTマスクを取得
            gt_mask = batch['mask_labels'][0]
            gt_mask_np = gt_mask.detach().cpu().numpy()
            if gt_mask_np.ndim > 2:
                gt_mask_np = gt_mask_np[0] if gt_mask_np.shape[0] > 0 else gt_mask_np.squeeze()
            
            # テキスト情報を取得
            input_ids = batch['input_ids'][0]
            labels = batch.get('labels', torch.full_like(input_ids, -100))[0]
            
            # テキストをデコード
            try:
                full_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
                
                # ユーザー入力とアシスタント応答を分離
                user_start_marker = "<|im_start|>user"
                assistant_start_marker = "<|im_start|>assistant"
                assistant_end_marker = "<|im_end|>"
                
                user_text = ""
                assistant_text = ""
                
                if user_start_marker in full_text:
                    user_start = full_text.index(user_start_marker) + len(user_start_marker)
                    if assistant_start_marker in full_text:
                        user_end = full_text.index(assistant_start_marker)
                        user_text = full_text[user_start:user_end].strip()
                
                if assistant_start_marker in full_text:
                    assistant_start = full_text.index(assistant_start_marker) + len(assistant_start_marker)
                    remaining_text = full_text[assistant_start:]
                    if assistant_end_marker in remaining_text:
                        assistant_end = remaining_text.index(assistant_end_marker)
                        assistant_text = remaining_text[:assistant_end].strip()
                    else:
                        assistant_text = remaining_text.strip()
                
                # <|im_end|>を削除
                user_text = user_text.replace("<|im_end|>", "").strip()
                
                # vision部分を簡潔に表示
                if "<|vision_start|>" in user_text and "<|vision_end|>" in user_text:
                    vision_start = user_text.index("<|vision_start|>")
                    vision_end = user_text.index("<|vision_end|>") + len("<|vision_end|>")
                    vision_content = user_text[vision_start:vision_end]
                    image_pad_count = vision_content.count("<|image_pad|>")
                    simplified_vision = f"<|vision_start|>[{image_pad_count} image patches]<|vision_end|>"
                    user_text = user_text[:vision_start] + simplified_vision + user_text[vision_end:]
                
                # ラベルマスキング情報
                masked_count = (labels == -100).sum().item()
                unmasked_count = (labels != -100).sum().item()
                
            except Exception as e:
                user_text = f"[Decoding Error: {e}]"
                assistant_text = ""
                masked_count = 0
                unmasked_count = 0
            
            # 可視化
            fig = plt.figure(figsize=(20, 10))
            
            # 上段: 画像、GTマスク、予測マスク
            ax1 = plt.subplot(2, 3, 1)
            ax1.imshow(image_np)
            ax1.set_title(f'Input Image (Step {step})', fontsize=12, fontweight='bold')
            ax1.axis('off')
            
            ax2 = plt.subplot(2, 3, 2)
            ax2.imshow(gt_mask_np, cmap='gray')
            ax2.set_title('GT Mask', fontsize=12, fontweight='bold')
            ax2.axis('off')
            
            ax3 = plt.subplot(2, 3, 3)
            ax3.imshow(pred_mask_np, cmap='gray', vmin=0, vmax=1)
            ax3.set_title('Predicted Mask', fontsize=12, fontweight='bold')
            ax3.axis('off')
            
            # 下段: オーバーレイ比較
            ax4 = plt.subplot(2, 3, 4)
            ax4.imshow(image_np)
            gt_mask_resized = cv2.resize(gt_mask_np, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_overlay = np.zeros_like(image_np)
            mask_overlay[:, :, 0] = gt_mask_resized * 255
            ax4.imshow(mask_overlay, alpha=0.3)
            ax4.set_title('Image + GT Mask', fontsize=12, fontweight='bold')
            ax4.axis('off')
            
            ax5 = plt.subplot(2, 3, 5)
            ax5.imshow(image_np)
            pred_mask_resized = cv2.resize(pred_mask_np, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_LINEAR)
            pred_overlay = np.zeros_like(image_np)
            pred_overlay[:, :, 1] = pred_mask_resized * 255
            ax5.imshow(pred_overlay, alpha=0.3)
            ax5.set_title('Image + Pred Mask', fontsize=12, fontweight='bold')
            ax5.axis('off')
            
            # メトリクス表示
            ax6 = plt.subplot(2, 3, 6)
            ax6.axis('off')
            
            # Dice scoreとIoUを計算
            pred_binary = (pred_mask_resized > 0.5).astype(np.float32)
            gt_binary = (gt_mask_resized > 0.5).astype(np.float32)
            intersection = (pred_binary * gt_binary).sum()
            union = pred_binary.sum() + gt_binary.sum() - intersection
            iou = intersection / (union + 1e-6)
            dice = 2 * intersection / (pred_binary.sum() + gt_binary.sum() + 1e-6)
            
            # 損失情報も追加
            lm_loss = outputs.loss.item() if hasattr(outputs, 'loss') else 0.0
            
            info_text = f"Step: {step}\n"
            info_text += f"Dice Score: {dice:.4f}\n"
            info_text += f"IoU: {iou:.4f}\n"
            info_text += f"LM Loss: {lm_loss:.4f}\n"
            info_text += f"Mask Size: {gt_mask_np.shape}\n"
            info_text += f"Image Size: {image_np.shape[:2]}\n"
            info_text += f"Label Masking: {masked_count} masked, {unmasked_count} unmasked\n\n"
            if len(user_text) > 200:
                info_text += f"User: {user_text[:200]}...\n\n"
            else:
                info_text += f"User: {user_text}\n\n"
            if len(assistant_text) > 100:
                info_text += f"Assistant: {assistant_text[:100]}..."
            else:
                info_text += f"Assistant: {assistant_text}"
            ax6.text(0.05, 0.95, info_text, transform=ax6.transAxes, 
                    fontsize=9, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    wrap=True, family='monospace')
            ax6.set_title('Metrics & Text', fontsize=12, fontweight='bold')
            
            plt.suptitle(f'Training Visualization - Step {step}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # 可視化ディレクトリを使用（既に作成済み）
            save_path = self.vis_dir / f'step_{step:06d}.png'
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            # JSON形式でも情報を保存
            vis_info = {
                'step': int(step),
                'metrics': {
                    'dice_score': float(dice),
                    'iou': float(iou),
                    'lm_loss': float(lm_loss),
                },
                'text': {
                    'user': user_text,
                    'assistant': assistant_text,
                    'masked_tokens': int(masked_count),
                    'unmasked_tokens': int(unmasked_count),
                },
                'shapes': {
                    'gt_mask': list(gt_mask_np.shape),
                    'pred_mask': list(pred_mask_np.shape),
                    'image': list(image_np.shape),
                }
            }
            
            json_path = self.vis_dir / f'step_{step:06d}.json'
            with open(json_path, 'w') as f:
                json.dump(vis_info, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Saved visualization to {save_path} (Dice: {dice:.4f}, IoU: {iou:.4f})")
            
        except Exception as e:
            logger.warning(f"Failed to save visualization at step {step}: {e}")
    
    def save_checkpoint(self, name="best"):
        """チェックポイントの保存（新しいcheckpoint_io使用）"""
        from src.utils.checkpoint_io import save_lisa_checkpoint, build_checkpoint_dir
        
        # チェックポイントディレクトリを構築
        if name == "best" or name == "final":
            # best/finalは既存のディレクトリ構造を使用
            save_path = self.checkpoint_dir / name
        else:
            # それ以外（epoch_X, step_X）も既存構造を維持
            save_path = self.checkpoint_dir / name
        
        # 新しい保存関数を使用
        save_lisa_checkpoint(
            checkpoint_dir=str(save_path),
            model=self.model,
            processor=self.processor,
            config=self.lisa_config,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=None,  # AMPを使用していない場合
            global_step=self.global_step,
            best_loss=self.best_loss,
            epoch=getattr(self, 'current_epoch', 0),
            additional_info={
                'save_name': name,
                'total_steps': getattr(self, 'total_steps', self.global_step),
                'learning_rate': self.scheduler.get_last_lr()[0] if self.scheduler else getattr(self.config, 'learning_rate', 1e-4),
            }
        )
        
        logger.info(f"チェックポイント保存完了: {save_path}")
        
        # チェックポイント保存時に推論評価を実行（フラグがTrueの場合）
        if getattr(self.config, 'run_inference_eval', False):
            try:
                self.run_inference_evaluation(save_path, name)
            except Exception as e:
                logger.warning(f"推論評価の実行に失敗: {e}")
    
    def run_inference_evaluation(self, checkpoint_path, checkpoint_name):
        """チェックポイント保存時に推論評価を実行
        
        test_inference_v2.pyと同じサンプル画像を使用して推論を実行し、
        結果を可視化してチェックポイントディレクトリに保存します。
        
        Args:
            checkpoint_path: チェックポイントのパス
            checkpoint_name: チェックポイントの名前（best, final, step_X など）
        
        Note:
            - --run_inference_eval フラグで有効化
            - --inference_eval_samples でサンプル数を指定（デフォルト3、最大3）
            - 結果は checkpoint_path/inference_results/ に保存
            - 各ステップでの推論性能の変化を追跡可能
        """
        logger.info(f"🔍 チェックポイント {checkpoint_name} の推論評価を開始...")
        
        # 推論結果を保存するディレクトリ
        inference_dir = checkpoint_path / "inference_results"
        inference_dir.mkdir(exist_ok=True)
        
        # モデルを評価モードに
        self.model.eval()
        
        # サンプル画像をダウンロードまたは使用
        sample_images = self.get_sample_images_for_inference()
        
        results = []
        with torch.no_grad():
            for idx, img_data in enumerate(sample_images):
                try:
                    # 推論を実行
                    result = self.run_single_inference(
                        img_data['image'], 
                        img_data['prompt']
                    )
                    
                    # 結果を可視化して保存
                    if result['mask'] is not None:
                        self.visualize_inference_result(
                            img_data['image'],
                            result['mask'],
                            img_data['name'],
                            img_data['prompt'],
                            inference_dir
                        )
                        
                        # 統計情報を記録
                        mask = result['mask']
                        result_info = {
                            "name": img_data['name'],
                            "prompt": img_data['prompt'],
                            "checkpoint": checkpoint_name,
                            "global_step": self.global_step,
                            "mask_min": float(mask.min()),
                            "mask_max": float(mask.max()),
                            "mask_mean": float(mask.mean()),
                            "positive_pixels": int((mask > 0.5).sum()),
                            "total_pixels": int(mask.size)
                        }
                        results.append(result_info)
                        logger.info(f"  ✓ {img_data['name']}: mean={result_info['mask_mean']:.3f}, positive={result_info['positive_pixels']}/{result_info['total_pixels']}")
                        
                except Exception as e:
                    logger.warning(f"  ✗ {img_data['name']}: 推論失敗 - {e}")
                    continue
        
        # 結果をJSONで保存
        if results:
            results_path = inference_dir / "evaluation_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"  → 評価結果を保存: {results_path}")
        
        # モデルを訓練モードに戻す
        self.model.train()
        logger.info(f"✅ 推論評価完了: {len(results)}/{len(sample_images)} 成功")
    
    def get_sample_images_for_inference(self):
        """推論評価用のサンプル画像を取得"""
        import requests
        from io import BytesIO
        from PIL import Image
        
        # キャッシュディレクトリ
        cache_dir = self.output_dir / "inference_cache"
        cache_dir.mkdir(exist_ok=True)
        
        sample_images = [
            {
                "url": "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/truck.jpg",
                "name": "truck",
                "prompt": "Please segment the truck in the image."
            },
            {
                "url": "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/groceries.jpg",
                "name": "groceries",
                "prompt": "Please segment the fruits on the table."
            },
            {
                "url": "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/dog.jpg",
                "name": "dog",
                "prompt": "Please segment the dog in the image."
            },
        ]
        
        # サンプル数を制限
        max_samples = getattr(self.config, 'inference_eval_samples', 3)
        sample_images = sample_images[:max_samples]
        
        downloaded_images = []
        for img_info in sample_images:
            # キャッシュをチェック
            cache_path = cache_dir / f"{img_info['name']}.jpg"
            
            try:
                if cache_path.exists():
                    # キャッシュから読み込み
                    image = Image.open(cache_path).convert("RGB")
                else:
                    # ダウンロードしてキャッシュに保存
                    response = requests.get(img_info["url"], timeout=10)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content)).convert("RGB")
                    image.save(cache_path)
                
                downloaded_images.append({
                    "image": image,
                    "name": img_info["name"],
                    "prompt": img_info["prompt"]
                })
                
            except Exception as e:
                logger.warning(f"画像 {img_info['name']} の取得に失敗: {e}")
                continue
        
        return downloaded_images
    
    def run_single_inference(self, image, prompt):
        """単一画像に対する推論を実行（test_inference_v2.pyと同様）"""
        from PIL import Image
        import numpy as np
        
        # プロンプトに<SEG>トークンを追加
        if self.lisa_config.seg_token not in prompt:
            prompt = prompt + f" {self.lisa_config.seg_token}"
        
        # メッセージフォーマット
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }]
        
        # テキストプロンプトを生成
        text_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 画像処理（qwen_vl_utilsを使用）
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
        except ImportError:
            # フォールバック
            image_inputs = [image]
            video_inputs = []
        
        # プロセッサで処理
        if video_inputs:
            inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                max_length=8192
            ).to(self.device)
        else:
            inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
                max_length=8192
            ).to(self.device)
        
        # SAM用の高解像度画像を準備
        sam_image = np.array(image.resize((1024, 1024)))
        sam_image_tensor = torch.from_numpy(sam_image).permute(2, 0, 1).float() / 255.0
        sam_image_tensor = sam_image_tensor.unsqueeze(0).to(self.device)
        
        # モデルのforward
        outputs = self.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values=inputs.pixel_values if hasattr(inputs, 'pixel_values') else None,
            image_grid_thw=inputs.image_grid_thw if hasattr(inputs, 'image_grid_thw') else None,
            sam_images=sam_image_tensor,
            labels=None,
            mask_labels=None
        )
        
        # マスクを取得
        pred_mask = None
        if hasattr(outputs, 'pred_masks') and outputs.pred_masks is not None:
            pred_mask = outputs.pred_masks[0].cpu().numpy()
        elif hasattr(outputs, 'mask_logits') and outputs.mask_logits is not None:
            if len(outputs.mask_logits) > 0 and outputs.mask_logits[0] is not None:
                mask_logit = outputs.mask_logits[0]
                if isinstance(mask_logit, list):
                    mask_logit = mask_logit[0]
                pred_mask = torch.sigmoid(mask_logit).detach().cpu().numpy()
                if pred_mask.ndim > 2:
                    pred_mask = pred_mask.squeeze()
        
        if pred_mask is None:
            pred_mask = np.zeros((image.height, image.width))
        
        return {
            "mask": pred_mask,
            "logits": outputs.logits if hasattr(outputs, 'logits') else None
        }
    
    def visualize_inference_result(self, image, mask, name, prompt, output_dir):
        """推論結果を可視化して保存"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 元画像
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # セグメンテーションマスク
        axes[1].imshow(mask, cmap='jet', alpha=0.7)
        axes[1].set_title("Segmentation Mask")
        axes[1].axis('off')
        
        # オーバーレイ
        axes[2].imshow(image)
        axes[2].imshow(mask, cmap='jet', alpha=0.5)
        axes[2].set_title("Overlay")
        axes[2].axis('off')
        
        # プロンプトを表示
        fig.suptitle(f"Step {self.global_step}: {prompt[:50]}...", fontsize=10)
        
        # 保存
        output_path = output_dir / f"{name}_step{self.global_step}.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    def plot_loss_history(self):
        """Loss履歴を可視化して保存"""
        if len(self.loss_history['steps']) == 0:
            return
        
        # 2x2のサブプロットを作成
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Total Loss
        ax = axes[0, 0]
        ax.plot(self.loss_history['steps'], self.loss_history['total_loss'], 'b-', linewidth=2)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Total Loss')
        ax.set_title('Total Loss')
        ax.grid(True, alpha=0.3)
        
        # 2. LM Loss
        ax = axes[0, 1]
        ax.plot(self.loss_history['steps'], self.loss_history['lm_loss'], 'g-', linewidth=2)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Language Model Loss')
        ax.set_title('Language Model Loss')
        ax.grid(True, alpha=0.3)
        
        # 3. Segmentation Loss
        ax = axes[1, 0]
        ax.plot(self.loss_history['steps'], self.loss_history['seg_loss'], 'r-', linewidth=2)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Segmentation Loss')
        ax.set_title('Segmentation Loss')
        ax.grid(True, alpha=0.3)
        
        # 4. Learning Rate
        ax = axes[1, 1]
        ax.plot(self.loss_history['steps'], self.loss_history['learning_rate'], 'm-', linewidth=2)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # 対数スケール
        
        # レイアウト調整
        plt.tight_layout()
        
        # 保存
        save_path = self.output_dir / 'loss_curves.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # ステップごとの詳細グラフも保存
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 全Lossを1つのグラフに
        ax.plot(self.loss_history['steps'], self.loss_history['total_loss'], 'b-', label='Total Loss', linewidth=2)
        ax.plot(self.loss_history['steps'], self.loss_history['lm_loss'], 'g--', label='LM Loss', linewidth=2)
        ax.plot(self.loss_history['steps'], self.loss_history['seg_loss'], 'r:', label='Seg Loss', linewidth=2)
        
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 保存
        save_path = self.output_dir / 'combined_loss_curves.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Loss curves saved to {self.output_dir}")
    
    def train(self):
        """訓練のメインループ"""
        logger.info("訓練開始")
        
        # アライメントステージはmain()で既に実行済みなのでここでは実行しない
        # self.run_alignment_stage()  # コメントアウト: main()で実行済み
        
        logger.info("メイン学習ステージ開始")
        
        for epoch in range(self.config.num_epochs):
            avg_loss = self.train_epoch(epoch)
            
            # ベストモデルの保存
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save_checkpoint("best")
                logger.info(f"ベストモデル更新: 損失 = {self.best_loss:.4f}")
            
            # エポック終了時の保存
            self.save_checkpoint(f"epoch_{epoch+1}")
        
        logger.info("訓練完了")
        logger.info(f"ベスト損失: {self.best_loss:.4f}")
        
        # 最終的な可視化を保存
        self.plot_loss_history()
        
        # Loss履歴をJSONで保存
        import json
        with open(self.output_dir / "loss_history.json", 'w') as f:
            json.dump(self.loss_history, f, indent=2)
        
        # 最終チェックポイント
        self.save_checkpoint("final")


def main():
    parser = argparse.ArgumentParser(description="LISA改ミニマルトレーニング")
    
    # データ設定
    parser.add_argument('--data_dir', type=str, default=None,
                       help='データセットのベースディレクトリ（Noneの場合はLISAConfigのデフォルトを使用）')
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
    
    # 注意: 学習率設定はsrc/config.pyで一元管理されています
    # adapter_lr, lora_lr, seg_token_lr, weight_decayを変更する場合は
    # src/config.pyを直接編集してください
    
    # ========================================================================
    # 重要: すべてのモデル設定と学習パラメータはsrc/config.pyで一元管理されています
    # ========================================================================
    # 以下の設定を変更する場合は、src/config.pyを直接編集してください：
    # - LoRA設定（lora_r, lora_alpha, lora_visual_enabled等）
    # - 学習率（adapter_lr, lora_lr, seg_token_lr）
    # - 損失重み（segmentation_loss_weight）
    # - その他のモデル設定
    
    # アライメントステージ設定
    parser.add_argument('--align_steps', type=int, default=0,
                       help='埋め込みアライメントステップ数（0の場合は実行しない）')
    parser.add_argument('--use_cached_alignment', action='store_true',
                       help='既存のアライメントチェックポイントを自動検出して使用')
    parser.add_argument('--alignment_checkpoint', type=str, default=None,
                       help='読み込むアライメントチェックポイントのパス')
    parser.add_argument('--force_realign', action='store_true',
                       help='キャッシュを無視して再アライメント')
    
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
    parser.add_argument('--visualize', action='store_true',
                       help='訓練中の可視化を有効化')
    parser.add_argument('--visualize_steps', type=int, default=5,
                       help='可視化の間隔（ステップ数）')
    parser.add_argument('--run_inference_eval', action='store_true',
                       help='チェックポイント保存時に推論評価を実行')
    parser.add_argument('--inference_eval_samples', type=int, default=3,
                       help='推論評価で使用するサンプル数（最大3）')
    
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