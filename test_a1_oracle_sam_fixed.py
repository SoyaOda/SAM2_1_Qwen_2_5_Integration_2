#!/usr/bin/env python3
"""
A-1テスト: SAM2.1側の"オラクル・プロンプト"過学習テスト
minimal_train.pyの訓練状況を再現し、現在のデータセット仕様に対応

狙い: マスクデコーダ経路（＋LoRA）が正常に学習できるかを、テキスト抜きでまず切り分ける。
手順:
1. LISA_ModelからSAM部分を取り出し、Qwenを完全凍結
2. GTの重心点を直接SAM2.1のPromptEncoderに入力
3. 1画像1マスクの極小データで学習（オリジナルLISAのランダム性を維持）
4. 期待: 数百〜数千ステップでDice ≳ 0.95 / mIoU ≳ 0.9、BCE+Diceが極小へ
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent))

# 共通設定をインポート
from test_common_config import (
    SEED, NUM_STEPS, BATCH_SIZE, LEARNING_RATE, NUM_FIXED_SAMPLES,
    DATASET_TYPE, SAMPLE_RATE, VIS_INTERVAL, LOG_INTERVAL, EVAL_THRESHOLD,
    set_random_seed, FixedSampleDataset, save_test_config, create_standard_output_structure
)

from src.config import LISAConfig
from src.models import LISA_Model
from src.utils import prepare_tokenizer_for_lisa
from transformers import AutoProcessor
from src.data.dataset import HybridDataset
from src.data.collators import MultiModalDataCollator
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex

# ロギング設定
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO  # INFOレベルに戻す
)
logger = logging.getLogger(__name__)


class OracleSAMTester:
    """LISA_ModelのSAM2.1部分を使用したオラクルプロンプト過学習テスター"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 出力ディレクトリ（共通構造）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = create_standard_output_structure(f"test_outputs/test_a1_{timestamp}")
        
        # メトリクス
        self.dice_metric = BinaryF1Score(threshold=0.5).to(self.device)
        self.iou_metric = BinaryJaccardIndex(threshold=0.5).to(self.device)
        
        # Loss履歴
        self.loss_history = {
            'steps': [],
            'total_loss': [],
            'bce_loss': [],
            'dice_loss': [],
            'dice_score': [],
            'iou_score': []
        }
        
        # 固定サンプル情報（過学習テスト用）
        self.fixed_sample = None
        self.fixed_batch = None
        
        logger.info(f"Output directory: {self.output_dir}")
        
        # テスト設定を保存
        save_test_config(self.output_dir, 'test_a1_oracle_sam', {
            'test_description': 'Oracle SAM prompt test with GT centroids'
        })
    
    def setup_model(self):
        """LISA_Modelのセットアップ（minimal_train.pyと同じ設定）"""
        logger.info("Setting up LISA_Model...")
        
        # LISAConfig作成（minimal_train.pyと同じ）
        self.lisa_config = LISAConfig(
            qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
            sam_model_name="./checkpoints/sam2.1_hiera_large.pt",
            device_map=str(self.device),
            torch_dtype="auto",
            use_flash_attention=False,
            # Training configuration - Test A1: Oracle SAM Fixed
            freeze_qwen_lora=True,        # Qwenは完全凍結（A1）
            freeze_seg_token=True,        # SEGトークンも凍結（A1）
            freeze_sam_lora=True,         # SAMも凍結（A1）
            freeze_image_adapter=True,    # アダプターも凍結（A1）
            freeze_text_prompt_projector=True,  # プロジェクターも凍結（A1）
            freeze_token_fpn=True,        # Token-FPNも凍結（A1）
            freeze_prompt_beta=True       # Betaも凍結（A1）
        )
        
        # トークナイザーとプロセッサの準備
        logger.info("Preparing tokenizer and processor...")
        self.tokenizer = prepare_tokenizer_for_lisa(
            model_name=self.lisa_config.qwen_model_name,
            seg_token=self.lisa_config.seg_token
        )
        
        # ProcessorにもSEGトークンを追加（minimal_train.pyと同じ）
        if self.lisa_config.use_dynamic_resolution:
            self.processor = AutoProcessor.from_pretrained(
                self.lisa_config.qwen_model_name,
                min_pixels=self.lisa_config.qwen_min_pixels,
                max_pixels=self.lisa_config.qwen_max_pixels
            )
        else:
            self.processor = AutoProcessor.from_pretrained(self.lisa_config.qwen_model_name)
        
        if self.lisa_config.seg_token not in self.processor.tokenizer.get_vocab():
            self.processor.tokenizer.add_special_tokens({"additional_special_tokens": [self.lisa_config.seg_token]})
        
        # モデルの読み込み
        logger.info("Loading LISA_Model...")
        self.model = LISA_Model(self.lisa_config)
        
        # トークナイザーを設定
        self.model.set_tokenizer(self.tokenizer)
        
        # パラメータ統計の詳細表示
        from src.utils.model_utils import display_parameter_statistics
        display_parameter_statistics(self.model, logger_name=__name__)
        
        # SAM MaskDecoderへのLoRA適用（freeze_sam_loraがFalseの場合のみ）
        if not self.lisa_config.freeze_sam_lora and hasattr(self.lisa_config, 'sam_lora_r') and self.lisa_config.sam_lora_r > 0:
            logger.info(f"Applying LoRA to SAM MaskDecoder (r={self.lisa_config.sam_lora_r})")
            self.model.add_sam_lora(
                lora_r=self.lisa_config.sam_lora_r,
                lora_alpha=self.lisa_config.sam_lora_alpha,
                lora_dropout=self.lisa_config.sam_lora_dropout
            )
        
        self.model.to(self.device)
        
        # パラメータ数の確認
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def setup_data(self):
        """データセットのセットアップ（すべてのデータセットタイプに対応）"""
        logger.info("Setting up dataset...")
        
        # minimal_train.pyと同じデータセット設定
        data_dir = self.lisa_config.dataset_base_dir
        logger.info(f"Dataset directory: {data_dir}")
        
        # データセットタイプの設定（引数から取得）
        dataset_types = self.config.dataset_types if hasattr(self.config, 'dataset_types') else 'sem_seg|refer_seg|reason_seg|vqa'
        sample_rates = self.config.sample_rates if hasattr(self.config, 'sample_rates') else [9.0, 3.0, 3.0, 1.0]
        
        logger.info(f"Dataset types: {dataset_types}")
        logger.info(f"Sample rates: {sample_rates}")
        
        # ベースのHybridDatasetを作成
        base_dataset = HybridDataset(
            base_image_dir=data_dir,
            qwen_processor=self.processor,
            samples_per_epoch=NUM_FIXED_SAMPLES,  # 共通設定から
            dataset=DATASET_TYPE,  # 共通設定から
            sample_rate=SAMPLE_RATE,  # 共通設定から
            qwen_image_size=self.lisa_config.qwen_image_size,
            sam_image_size=self.lisa_config.sam_image_size,
        )
        
        # 固定サンプル版に変換（共通クラスを使用）
        self.dataset = FixedSampleDataset(base_dataset, num_samples=NUM_FIXED_SAMPLES, seed=SEED)
        
        # DataCollator（minimal_train.pyと同じ）
        self.collator = MultiModalDataCollator(
            tokenizer=self.tokenizer,
            max_length=self.lisa_config.model_max_length,
            config=self.lisa_config
        )
        
        # DataLoader（オリジナルLISAのランダム性を維持）
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,  # HybridDatasetが内部でランダムサンプリング
            collate_fn=self.collator,
            num_workers=0
        )
        
        logger.info(f"Dataset size: {len(self.dataset)}")
    
    def get_fixed_sample(self):
        """最初のサンプルを取得して固定（過学習テスト用）"""
        if self.fixed_batch is None:
            logger.info("Getting first sample for overfitting test...")
            for batch in self.dataloader:
                self.fixed_batch = batch
                break
            
            # バッチの内容を確認
            logger.info(f"Fixed batch keys: {self.fixed_batch.keys()}")
            for key, value in self.fixed_batch.items():
                if torch.is_tensor(value):
                    logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                elif isinstance(value, list):
                    logger.info(f"  {key}: list of {len(value)} items")
        
        return self.fixed_batch
    
    def compute_mask_centroid(self, mask):
        """マスクから重心座標を計算"""
        if mask.dim() == 4:
            mask = mask.squeeze(0).squeeze(0)
        elif mask.dim() == 3:
            mask = mask.squeeze(0)
        
        m = mask.to(torch.float32)
        h, w = m.shape
        
        mass = m.sum()
        if mass <= 0:
            return torch.tensor([w // 2, h // 2], dtype=torch.float32, device=mask.device)
        
        ys = torch.arange(h, device=m.device, dtype=torch.float32).view(h, 1)
        xs = torch.arange(w, device=m.device, dtype=torch.float32).view(1, w)
        
        cy = (m * ys).sum() / mass
        cx = (m * xs).sum() / mass
        
        return torch.stack([cx, cy])
    
    def forward_with_oracle_prompt(self, batch, step):
        """オラクルプロンプト（GT重心）を使用してマスクを生成（minimal_train.pyのforwardを参考）"""
        
        # バッチをデバイスに移動
        pixel_values = batch['pixel_values'].to(self.device)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        mask_labels = batch.get('mask_labels')
        image_grid_thw = batch.get('image_grid_thw')
        
        # GTマスクを取得
        if mask_labels is None:
            logger.warning("No GT masks in batch")
            return None, None
        
        gt_masks = mask_labels.to(self.device)
        
        # Qwenを通して画像特徴を抽出（minimal_train.pyと同じ）
        with torch.no_grad():  # Qwenは凍結
            # Qwenのforward
            qwen_outputs = self.model.qwen(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                image_grid_thw=image_grid_thw,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Vision features extraction
            vision_features = self.model.extract_vision_features(pixel_values, image_grid_thw)
            if len(vision_features.shape) == 2:
                vision_features = vision_features.unsqueeze(0)
        
        # SAM用の画像特徴をアダプタで変換（勾配を流す）
        image_features_sam = self.model.image_adapter(vision_features, image_grid_thw)
        
        # 高解像度特徴の生成
        if self.model.use_token_fpn:
            image_features_sam_fpn, sam_high_res_features = self.model.token_fpn(
                image_features_sam,
                image_grid_thw=image_grid_thw,
                use_hooks=True
            )
            image_features_sam = image_features_sam_fpn
        else:
            sam_high_res_features = self.model.high_res_generator(image_features_sam)
        
        # GTマスクから重心を計算
        B = gt_masks.shape[0]
        all_pred_masks = []
        
        for i in range(B):
            # 各バッチのGTマスク
            gt_mask = gt_masks[i]
            
            # 重心を計算
            if gt_mask.dim() == 3 and gt_mask.shape[0] > 1:
                gt_mask = gt_mask[0]  # 最初のマスクのみ使用
            
            centroid = self.compute_mask_centroid(gt_mask)
            
            # minimal_train.pyのforward内のSAMマスク生成部分を参考
            h_feat, w_feat = image_features_sam[i:i+1].shape[-2:]
            h_img, w_img = h_feat * 16, w_feat * 16  # Feature stride is 16
            
            # 重心座標をSAMのポイント形式に変換
            point_coords = centroid.unsqueeze(0).unsqueeze(0)  # [1, 1, 2]
            point_labels = torch.ones(1, 1, dtype=torch.int32, device=self.device)
            
            # SAMのPromptEncoderを使用
            sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
                points=(point_coords, point_labels),
                boxes=None,
                masks=None,
            )
            
            # Positional encoding
            image_pe = self.model.sam_prompt_encoder.get_dense_pe()
            if image_pe.shape[-2:] != image_features_sam[i:i+1].shape[-2:]:
                image_pe = F.interpolate(
                    image_pe,
                    size=(h_feat, w_feat),
                    mode='bilinear',
                    align_corners=False
                )
            
            # Dense embeddingsのサイズ調整
            if dense_embeddings.shape[-2:] != image_features_sam[i:i+1].shape[-2:]:
                dense_embeddings = F.interpolate(
                    dense_embeddings,
                    size=(h_feat, w_feat),
                    mode='bilinear',
                    align_corners=False
                )
            
            # 高解像度特徴の準備（minimal_train.pyと同じ）
            sam_dtype = next(self.model.sam_mask_decoder.parameters()).dtype
            
            if sam_high_res_features is not None and len(sam_high_res_features) >= 2:
                feat_s0 = sam_high_res_features[0][i:i+1].to(dtype=sam_dtype)
                feat_s1 = sam_high_res_features[1][i:i+1].to(dtype=sam_dtype)
                
                if hasattr(self.model.sam_mask_decoder, 'conv_s0') and hasattr(self.model.sam_mask_decoder, 'conv_s1'):
                    feat_s0 = self.model.sam_mask_decoder.conv_s0(feat_s0)
                    feat_s1 = self.model.sam_mask_decoder.conv_s1(feat_s1)
                else:
                    raise RuntimeError("SAM2.1 MaskDecoder missing conv_s0/conv_s1")
                
                high_res_features = [feat_s0, feat_s1]
            else:
                raise RuntimeError("High-resolution features not available")
            
            # SAM MaskDecoderでマスク生成
            low_res_masks, iou_predictions, sam_tokens_out, obj_score_logits = self.model.sam_mask_decoder(
                image_embeddings=image_features_sam[i:i+1].to(dtype=sam_dtype),
                image_pe=image_pe.to(dtype=sam_dtype),
                sparse_prompt_embeddings=sparse_embeddings.to(dtype=sam_dtype),
                dense_prompt_embeddings=dense_embeddings.to(dtype=sam_dtype),
                multimask_output=False,
                repeat_image=True,
                high_res_features=high_res_features,
            )
            
            # アップサンプリング
            if image_grid_thw is not None and i < image_grid_thw.shape[0]:
                H_grid_raw = int(image_grid_thw[i, 1].item())
                W_grid_raw = int(image_grid_thw[i, 2].item())
                orig_h = H_grid_raw * 14
                orig_w = W_grid_raw * 14
            else:
                orig_h, orig_w = gt_mask.shape[-2:]
            
            pred_mask = F.interpolate(
                low_res_masks,
                size=(orig_h, orig_w),
                mode='bilinear',
                align_corners=False
            )
            
            all_pred_masks.append(pred_mask)
        
        # バッチ化
        pred_masks = torch.cat(all_pred_masks, dim=0)
        
        # GTマスクのサイズを合わせる
        if gt_masks.shape[-2:] != pred_masks.shape[-2:]:
            gt_masks = F.interpolate(
                gt_masks.float(),
                size=pred_masks.shape[-2:],
                mode='nearest'
            )
        
        return pred_masks, gt_masks
    
    def compute_losses(self, pred_masks, gt_masks):
        """損失計算（minimal_train.pyと同じ）"""
        # BCE損失
        bce_loss = F.binary_cross_entropy_with_logits(pred_masks, gt_masks.float())
        
        # Dice損失
        pred_probs = torch.sigmoid(pred_masks)
        smooth = 1e-6
        
        intersection = (pred_probs * gt_masks).sum(dim=(2, 3))
        union = pred_probs.sum(dim=(2, 3)) + gt_masks.sum(dim=(2, 3))
        dice_score = (2 * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice_score.mean()
        
        # 合計損失
        total_loss = bce_loss + dice_loss
        
        return total_loss, bce_loss, dice_loss, dice_score.mean()
    
    def save_visualization(self, batch, pred_masks, gt_masks, step):
        """可視化の保存（テキスト情報も含む）"""
        if step % VIS_INTERVAL != 0:  # 共通設定の間隔を使用
            return
        
        # original_imagesがcollatorから渡されている場合はそれを使用
        if 'original_images' in batch and batch['original_images'] is not None and len(batch['original_images']) > 0:
            # PIL画像をnumpy配列に変換
            from PIL import Image
            import numpy as np
            # リストから最初の画像を取得
            original_img = batch['original_images'][0]
            if isinstance(original_img, Image.Image):
                # PIL画像の場合
                image_np = np.array(original_img)
            elif isinstance(original_img, torch.Tensor):
                # Tensorの場合
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
                # その他の場合、スキップ
                logger.info(f"Skipping visualization at step {step} (unsupported image format)")
                return
        else:
            # original_imageがない場合、pixel_valuesから復元を試みる
            pixel_values = batch['pixel_values'][0].cpu()
            
            # パッチ形式の場合はスキップ
            if pixel_values.dim() == 2:
                logger.info(f"Skipping visualization at step {step} (patch format, no original_images)")
                return
            
            # 3D形式の場合
            if pixel_values.dim() == 3:
                # 正規化を解除
                mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
                std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
                image = pixel_values * std + mean
                image = torch.clamp(image, 0, 1)
                image_np = image.permute(1, 2, 0).numpy()
            else:
                # 可視化をスキップ
                logger.info(f"Skipping visualization at step {step} (unsupported pixel format)")
                return
        
        # マスクを取得
        pred_mask_np = torch.sigmoid(pred_masks[0, 0]).detach().cpu().numpy()
        gt_mask_np = gt_masks[0, 0].detach().cpu().numpy()
        
        # GTマスクから重心を計算して表示
        centroid = self.compute_mask_centroid(gt_masks[0, 0])
        centroid_x, centroid_y = centroid[0].item(), centroid[1].item()
        
        # テキスト情報を取得（トークンIDとラベルをデコード）
        input_ids = batch['input_ids'][0]
        labels = batch.get('labels', torch.full_like(input_ids, -100))[0]  # ラベルも取得
        attention_mask = batch.get('attention_mask', torch.ones_like(input_ids))[0]
        
        # テキストをデコード（特殊トークンの処理を改善）
        try:
            # 全体のテキスト（特殊トークン含む）
            full_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
            
            # ユーザー入力部分とアシスタント応答部分を分離
            user_start_marker = "<|im_start|>user"
            assistant_start_marker = "<|im_start|>assistant"
            assistant_end_marker = "<|im_end|>"
            
            user_text = ""
            assistant_text = ""
            
            # ユーザー入力を抽出
            if user_start_marker in full_text:
                user_start = full_text.index(user_start_marker) + len(user_start_marker)
                if assistant_start_marker in full_text:
                    user_end = full_text.index(assistant_start_marker)
                    user_text = full_text[user_start:user_end].strip()
                else:
                    user_text = full_text[user_start:].strip()
            
            # アシスタント応答を抽出
            if assistant_start_marker in full_text:
                assistant_start = full_text.index(assistant_start_marker) + len(assistant_start_marker)
                # 次の<|im_end|>を探す
                remaining_text = full_text[assistant_start:]
                if assistant_end_marker in remaining_text:
                    assistant_end = remaining_text.index(assistant_end_marker)
                    assistant_text = remaining_text[:assistant_end].strip()
                else:
                    assistant_text = remaining_text.strip()
            
            # <|im_end|>を削除
            user_text = user_text.replace("<|im_end|>", "").strip()
            
            # vision_start/endトークンとimage_padトークンを整理して表示
            if "<|vision_start|>" in user_text and "<|vision_end|>" in user_text:
                # vision部分を簡潔に表示
                vision_start = user_text.index("<|vision_start|>")
                vision_end = user_text.index("<|vision_end|>") + len("<|vision_end|>")
                vision_content = user_text[vision_start:vision_end]
                
                # image_padトークンの数をカウント
                image_pad_count = vision_content.count("<|image_pad|>")
                
                # vision部分を簡潔な表現に置換
                simplified_vision = f"<|vision_start|>[{image_pad_count} image patches]<|vision_end|>"
                user_text = user_text[:vision_start] + simplified_vision + user_text[vision_end:]
            
            # ラベルマスキング情報を取得
            masked_count = (labels == -100).sum().item()
            unmasked_count = (labels != -100).sum().item()
            
            # 表示用テキスト
            display_text = user_text[:300] if len(user_text) > 300 else user_text
            
        except Exception as e:
            display_text = f"[Decoding Error: {e}]"
            user_text = ""
            assistant_text = ""
            masked_count = 0
            unmasked_count = 0
        
        # 可視化（2行のレイアウト）
        fig = plt.figure(figsize=(20, 10))
        
        # 上段: 画像、GTマスク、予測マスク
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(image_np)
        # 重心をプロット
        ax1.plot(centroid_x, centroid_y, 'r*', markersize=15, markeredgewidth=2, 
                markeredgecolor='white', label=f'GT Centroid ({centroid_x:.0f}, {centroid_y:.0f})')
        ax1.set_title(f'Input Image (Step {step})', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.axis('off')
        
        ax2 = plt.subplot(2, 3, 2)
        ax2.imshow(gt_mask_np, cmap='gray')
        ax2.set_title('GT Mask', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        ax3 = plt.subplot(2, 3, 3)
        ax3.imshow(pred_mask_np, cmap='gray', vmin=0, vmax=1)
        ax3.set_title(f'Predicted Mask', fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        # 下段: オーバーレイ比較
        ax4 = plt.subplot(2, 3, 4)
        # 画像にGTマスクをオーバーレイ
        ax4.imshow(image_np)
        # マスクを画像サイズにリサイズ
        import cv2
        gt_mask_resized = cv2.resize(gt_mask_np, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_overlay = np.zeros_like(image_np)
        mask_overlay[:, :, 0] = gt_mask_resized * 255  # 赤でGTマスク
        ax4.imshow(mask_overlay, alpha=0.3)
        ax4.set_title('Image + GT Mask Overlay', fontsize=12, fontweight='bold')
        ax4.axis('off')
        
        ax5 = plt.subplot(2, 3, 5)
        # 画像に予測マスクをオーバーレイ
        ax5.imshow(image_np)
        # 予測マスクも画像サイズにリサイズ
        pred_mask_resized = cv2.resize(pred_mask_np, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_LINEAR)
        pred_overlay = np.zeros_like(image_np)
        pred_overlay[:, :, 1] = pred_mask_resized * 255  # 緑で予測マスク
        ax5.imshow(pred_overlay, alpha=0.3)
        ax5.set_title('Image + Pred Mask Overlay', fontsize=12, fontweight='bold')
        ax5.axis('off')
        
        # メトリクス表示
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # Dice scoreとIoUを計算（リサイズされたマスクで計算）
        pred_binary = (pred_mask_resized > 0.5).astype(np.float32)
        gt_binary = (gt_mask_resized > 0.5).astype(np.float32)
        intersection = (pred_binary * gt_binary).sum()
        union = pred_binary.sum() + gt_binary.sum() - intersection
        iou = intersection / (union + 1e-6)
        dice = 2 * intersection / (pred_binary.sum() + gt_binary.sum() + 1e-6)
        
        # メトリクスとテキスト情報を表示
        info_text = f"Step: {step}\n"
        info_text += f"Dice Score: {dice:.4f}\n"
        info_text += f"IoU: {iou:.4f}\n"
        info_text += f"Centroid: ({centroid_x:.0f}, {centroid_y:.0f})\n"
        info_text += f"Mask Size: {gt_mask_np.shape}\n"
        info_text += f"Image Size: {image_np.shape[:2]}\n"
        info_text += f"Label Masking: {masked_count} masked, {unmasked_count} unmasked\n\n"
        info_text += f"User Input:\n{display_text}\n\n"
        info_text += f"Assistant Response:\n{assistant_text}"
        
        ax6.text(0.05, 0.95, info_text, transform=ax6.transAxes, 
                fontsize=9, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                wrap=True, family='monospace')
        ax6.set_title('Information', fontsize=12, fontweight='bold')
        
        plt.suptitle(f'Oracle SAM Test Visualization - Step {step}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / 'visualizations' / f'step_{step:05d}.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # JSON形式でも情報を保存
        visualization_info = {
            'step': int(step),
            'metrics': {
                'dice_score': float(dice),
                'iou': float(iou),
            },
            'gt_centroid': {
                'x': float(centroid_x),
                'y': float(centroid_y)
            },
            'shapes': {
                'gt_mask': list(gt_mask_np.shape),
                'pred_mask': list(pred_mask_np.shape),
                'image': list(image_np.shape)
            },
            'text_info': {
                'user_input': user_text,
                'assistant_response': assistant_text,  # アシスタント応答を追加
                'full_text_length': len(full_text) if 'full_text' in locals() else 0,
                'tokenized_length': len(input_ids.tolist()),
                'label_masking': {
                    'masked_tokens': masked_count,
                    'unmasked_tokens': unmasked_count,
                    'total_tokens': len(labels.tolist()) if 'labels' in locals() else 0
                }
            },
            'training_info': {
                'loss': self.loss_history['total_loss'][-1] if self.loss_history['total_loss'] else 0,
                'bce_loss': self.loss_history['bce_loss'][-1] if self.loss_history['bce_loss'] else 0,
                'dice_loss': self.loss_history['dice_loss'][-1] if self.loss_history['dice_loss'] else 0,
            }
        }
        
        json_path = self.output_dir / f'visualization_step_{step}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(visualization_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved visualization to {save_path} and {json_path} (Dice: {dice:.4f}, IoU: {iou:.4f})")
    
    def train(self):
        """訓練ループ"""
        logger.info("Starting training...")
        
        # オプティマイザ（minimal_train.pyと同じ設定を参考）
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # 訓練ループ
        self.model.train()
        global_step = 0
        
        # 最初のバッチを固定（過学習テスト）
        fixed_batch = self.get_fixed_sample()
        
        with tqdm(total=self.config.max_steps) as pbar:
            while global_step < self.config.max_steps:
                # 固定バッチを使用（過学習テスト）
                batch = fixed_batch
                
                # オラクルプロンプトでマスク生成
                pred_masks, gt_masks = self.forward_with_oracle_prompt(batch, global_step)
                
                if pred_masks is None:
                    logger.warning(f"Step {global_step}: Failed to generate masks")
                    global_step += 1
                    continue
                
                # 損失計算
                total_loss, bce_loss, dice_loss, dice_score = self.compute_losses(pred_masks, gt_masks)
                
                # バックプロパゲーション
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                # メトリクス計算
                with torch.no_grad():
                    pred_binary = torch.sigmoid(pred_masks) > 0.5
                    dice_metric_score = self.dice_metric(pred_binary, gt_masks)
                    iou_metric_score = self.iou_metric(pred_binary, gt_masks)
                
                # 履歴記録
                self.loss_history['steps'].append(global_step)
                self.loss_history['total_loss'].append(total_loss.item())
                self.loss_history['bce_loss'].append(bce_loss.item())
                self.loss_history['dice_loss'].append(dice_loss.item())
                self.loss_history['dice_score'].append(dice_metric_score.item())
                self.loss_history['iou_score'].append(iou_metric_score.item())
                
                # ログ出力
                if global_step % LOG_INTERVAL == 0:
                    logger.info(
                        f"Step {global_step}: "
                        f"Loss={total_loss:.4f}, "
                        f"BCE={bce_loss:.4f}, "
                        f"Dice Loss={dice_loss:.4f}, "
                        f"Dice Score={dice_metric_score:.4f}, "
                        f"IoU={iou_metric_score:.4f}"
                    )
                
                # 可視化
                self.save_visualization(batch, pred_masks, gt_masks, global_step)
                
                # プログレスバー更新
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f'{total_loss:.4f}',
                    'dice': f'{dice_metric_score:.4f}',
                    'iou': f'{iou_metric_score:.4f}'
                })
                
                global_step += 1
                
                # 早期停止条件（Dice > 0.95）
                if dice_metric_score > 0.95:
                    logger.info(f"Early stopping: Dice score {dice_metric_score:.4f} > 0.95")
                    break
        
        logger.info("Training completed!")
        self.save_results()
    
    def save_results(self):
        """結果の保存"""
        # 損失グラフ
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].plot(self.loss_history['steps'], self.loss_history['total_loss'])
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(self.loss_history['steps'], self.loss_history['bce_loss'])
        axes[0, 1].set_title('BCE Loss')
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(self.loss_history['steps'], self.loss_history['dice_score'])
        axes[1, 0].set_title('Dice Score')
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].grid(True)
        axes[1, 0].set_ylim([0, 1])
        
        axes[1, 1].plot(self.loss_history['steps'], self.loss_history['iou_score'])
        axes[1, 1].set_title('IoU Score')
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].grid(True)
        axes[1, 1].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'logs' / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 履歴をJSONで保存
        with open(self.output_dir / 'logs' / 'training_history.json', 'w') as f:
            json.dump(self.loss_history, f, indent=2)
        
        # 最終結果のサマリー
        final_results = {
            'final_loss': self.loss_history['total_loss'][-1],
            'final_dice': self.loss_history['dice_score'][-1],
            'final_iou': self.loss_history['iou_score'][-1],
            'max_dice': max(self.loss_history['dice_score']),
            'max_iou': max(self.loss_history['iou_score']),
            'total_steps': len(self.loss_history['steps'])
        }
        
        with open(self.output_dir / 'final_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info(f"Results saved to {self.output_dir}")
        logger.info(f"Final Results: {final_results}")


def main():
    """メイン関数"""
    import argparse
    
    # シード固定
    set_random_seed(SEED)
    
    parser = argparse.ArgumentParser(description='A-1 Oracle SAM Test - Enhanced Version with All Datasets')
    parser.add_argument('--num_samples', type=int, default=NUM_FIXED_SAMPLES,
                       help='Number of samples in dataset (for random sampling)')
    parser.add_argument('--max_steps', type=int, default=NUM_STEPS,
                       help='Maximum training steps')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--dataset_types', type=str, default='sem_seg',
                       help='Dataset types separated by || (e.g., "sem_seg||refer_seg||reason_seg||vqa")')
    parser.add_argument('--sample_rates', type=str, default='1.0',
                       help='Sample rates for each dataset type, comma-separated (e.g., "9.0,3.0,3.0,1.0")')
    
    args = parser.parse_args()
    
    # sample_ratesを文字列からリストに変換
    args.sample_rates = [float(x) for x in args.sample_rates.split(',')]
    
    # データセットタイプの数とサンプル率の数が一致するか確認
    num_datasets = len(args.dataset_types.split('||'))
    if len(args.sample_rates) == 1 and num_datasets > 1:
        # 単一の比率が指定された場合、すべてのデータセットに同じ比率を適用
        args.sample_rates = args.sample_rates * num_datasets
    elif len(args.sample_rates) != num_datasets:
        raise ValueError(f"Number of sample rates ({len(args.sample_rates)}) must match number of datasets ({num_datasets})")
    
    logger.info("="*50)
    logger.info("A-1 Oracle SAM Test - Enhanced Version")
    logger.info("Testing with all dataset types and text visualization")
    logger.info("="*50)
    logger.info(f"Configuration:")
    logger.info(f"  Num samples: {args.num_samples}")
    logger.info(f"  Max steps: {args.max_steps}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Weight decay: {args.weight_decay}")
    logger.info(f"  Dataset types: {args.dataset_types}")
    logger.info(f"  Sample rates: {args.sample_rates}")
    
    # テスターの初期化
    tester = OracleSAMTester(args)
    
    # セットアップ
    tester.setup_model()
    tester.setup_data()
    
    # 訓練実行
    tester.train()
    
    logger.info("Test completed successfully!")


if __name__ == "__main__":
    main()