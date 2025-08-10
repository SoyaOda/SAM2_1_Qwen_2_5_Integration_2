#!/usr/bin/env python3
"""
A-1テスト: SAM2.1側の"オラクル・プロンプト"過学習テスト（現在のデータセット仕様対応版）
オリジナルLISA準拠のランダムサンプリングデータセットに対応

狙い: マスクデコーダ経路（＋LoRA）が正常に学習できるかを、テキスト抜きでまず切り分ける。
手順:
1. LISA_ModelからSAM部分を取り出し、Qwenを完全凍結
2. GTの重心点を直接SAM2.1のPromptEncoderに入力
3. 同一画像を繰り返し学習（過学習テスト）
4. 期待: 数百〜数千ステップでDice ≳ 0.95 / mIoU ≳ 0.9、BCE+Diceが極小へ
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import cv2

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent))

from src.config import LISAConfig
from src.models import LISA_Model
from src.utils import prepare_tokenizer_for_lisa
from transformers import AutoProcessor
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex

# ロギング設定
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class SingleSampleDataset(Dataset):
    """同一サンプルを繰り返し返すデータセット（過学習テスト用）"""
    
    def __init__(self, base_dataset, sample_idx=0):
        """
        Args:
            base_dataset: 元のデータセット
            sample_idx: 使用するサンプルのインデックス
        """
        self.base_dataset = base_dataset
        self.sample_idx = sample_idx
        
        # 最初のサンプルを取得して保存
        logger.info(f"Loading sample {sample_idx} from base dataset...")
        self.fixed_sample = base_dataset[sample_idx]
        
        # サンプルの情報をログ出力
        if isinstance(self.fixed_sample, tuple):
            logger.info(f"Fixed sample: tuple with {len(self.fixed_sample)} elements")
            if len(self.fixed_sample) > 0:
                logger.info(f"  Image path: {self.fixed_sample[0]}")
        elif isinstance(self.fixed_sample, dict):
            logger.info(f"Fixed sample: dict with keys {list(self.fixed_sample.keys())}")
    
    def __len__(self):
        return 1000  # 十分な回数の反復を許可
    
    def __getitem__(self, idx):
        # 常に同じサンプルを返す
        return self.fixed_sample


class OracleSAMTester:
    """LISA_ModelのSAM2.1部分を使用したオラクルプロンプト過学習テスター"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 出力ディレクトリ
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"test_outputs/test_a1_{timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        logger.info(f"Output directory: {self.output_dir}")
    
    def setup_model(self):
        """LISA_Modelのセットアップ（SAM部分のみ使用）"""
        logger.info("Setting up LISA_Model...")
        
        # LISAConfig作成
        self.lisa_config = LISAConfig(
            qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
            sam_model_name="./checkpoints/sam2.1_hiera_large.pt",
            device_map=str(self.device),
            torch_dtype="auto",
            freeze_qwen=True,  # Qwenは完全凍結
            freeze_sam=False,   # SAMは学習可能（LoRAで）
            train_seg_token=False,  # SEGトークンは使わない
            use_flash_attention=False
        )
        
        # トークナイザーとプロセッサの準備
        self.tokenizer = prepare_tokenizer_for_lisa(
            model_name=self.lisa_config.qwen_model_name,
            seg_token=self.lisa_config.seg_token
        )
        self.processor = AutoProcessor.from_pretrained(self.lisa_config.qwen_model_name)
        
        # モデルの読み込み
        self.model = LISA_Model(self.lisa_config)
        
        # トークナイザーを設定
        self.model.set_tokenizer(self.tokenizer)
        
        # SAM部分のみLoRAを適用
        if self.config.lora_r > 0:
            logger.info(f"Applying LoRA to SAM MaskDecoder (r={self.config.lora_r})")
            self.model.add_sam_lora(
                lora_r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout
            )
        
        self.model.to(self.device)
        
        # パラメータ数の確認
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def setup_data(self):
        """極小データセットのセットアップ"""
        logger.info("Setting up minimal dataset...")
        
        # 元のデータセットをインポート
        from src.data.sem_seg_dataset import SemSegDataset
        
        # SemSegDatasetを直接使用（シンプルな構造のため）
        base_dataset = SemSegDataset(
            base_image_dir=self.lisa_config.dataset_base_dir,
            tokenizer=self.tokenizer,
            samples_per_epoch=10,  # 少数サンプル
            sem_seg_data="ade20k",  # ADE20Kのみ
            num_classes_per_sample=1  # 1会話1マスク
        )
        
        # 同一サンプルを繰り返すデータセット
        self.dataset = SingleSampleDataset(base_dataset, sample_idx=0)
        
        # 簡易的なコレーター（tupleをdictに変換）
        def simple_collate_fn(batch):
            # batch[0]がtuple形式のサンプル
            sample = batch[0]
            
            # オリジナルLISA形式のtuple (10要素)
            # (image_path, image_for_sam, image_for_qwen, conversations, 
            #  masks, label, resize, questions, sampled_classes, coord_transform)
            
            result = {}
            
            # 画像データ
            result['image_path'] = sample[0]
            result['sam_image'] = sample[1].unsqueeze(0).to(self.device)  # SAM用画像
            result['qwen_image'] = sample[2].unsqueeze(0).to(self.device)  # Qwen用画像
            
            # マスクデータ
            masks = sample[4]
            if masks is not None:
                if not isinstance(masks, torch.Tensor):
                    masks = torch.tensor(masks)
                if masks.dim() == 2:
                    masks = masks.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                elif masks.dim() == 3:
                    masks = masks.unsqueeze(0)  # [1, C, H, W]
                result['gt_masks'] = masks.to(self.device)
            else:
                result['gt_masks'] = None
            
            # その他の情報
            result['conversations'] = sample[3]
            result['resize_info'] = sample[6]
            
            return result
        
        # DataLoader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,  # 過学習テストなので固定順序
            collate_fn=simple_collate_fn,
            num_workers=0
        )
        
        logger.info(f"Dataset size: {len(self.dataset)}")
    
    def compute_mask_centroid(self, mask):
        """マスクから重心座標を計算"""
        # 次元を調整
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
    
    def generate_mask_with_oracle_prompt(self, batch, step):
        """オラクルプロンプト（GT重心）を使用してマスクを生成"""
        
        # デバッグ: 最初のステップでバッチ内容を確認
        if step == 0:
            logger.info(f"Batch keys: {batch.keys()}")
            for key, value in batch.items():
                if torch.is_tensor(value):
                    logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    logger.info(f"  {key}: type={type(value)}")
        
        # SAM用画像とGTマスクを取得
        sam_image = batch['sam_image']  # [1, 3, 1024, 1024]
        gt_mask = batch['gt_masks']  # [1, 1, H, W]
        
        if gt_mask is None:
            logger.warning(f"No GT mask found for step {step}")
            return None
        
        # GTマスクから重心を計算
        centroid = self.compute_mask_centroid(gt_mask)
        
        # SAM用の画像特徴量を生成（LISA_Modelの実装に従う）
        with torch.no_grad():
            # SAM画像を適切な形式に変換（SAM2.1は1024x1024を期待）
            # sam_imageは既に正規化済み
            
            # 簡易的な画像特徴量生成（本来はQwenのビジョンエンコーダを通すが、ここでは簡略化）
            # SAM2.1の画像エンコーダを直接使う代わりに、ダミーの特徴量を生成
            # 注: 実際のLISA_Modelでは、Qwenのビジョン特徴をアダプタで変換している
            
            # ダミーの画像埋め込み（256チャネル、64x64の特徴マップ）
            h_feat, w_feat = 64, 64  # SAM2.1の標準的な特徴マップサイズ
            image_embeddings = torch.randn(1, 256, h_feat, w_feat, device=self.device) * 0.1
            
            # 高解像度特徴の生成（LISA_Modelと同様）
            # stride-4とstride-8の特徴を生成
            feat_s1 = torch.randn(1, 256, h_feat * 2, w_feat * 2, device=self.device) * 0.1  # stride-8
            feat_s0 = torch.randn(1, 256, h_feat * 4, w_feat * 4, device=self.device) * 0.1  # stride-4
            high_res_features = [feat_s0, feat_s1]
            
            # 重心座標を計算（元画像座標系）
            h, w = gt_mask.shape[-2:]
            # centroidは既に計算済み（cx, cy）
            
            # SAM2.1のpoint座標形式に変換
            point_coords = centroid.unsqueeze(0).unsqueeze(0)  # [1, 1, 2]
            
            # ポイントラベル（前景=1）
            point_labels = torch.ones(1, 1, dtype=torch.int32, device=self.device)
        
        # SAMのマスクデコーダを通す（ここは勾配を流す）
        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None
        )
        
        # Get positional encoding
        image_pe = self.model.sam_prompt_encoder.get_dense_pe()
        if image_pe.shape[-2:] != image_embeddings.shape[-2:]:
            image_pe = F.interpolate(
                image_pe,
                size=image_embeddings.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        # SAM MaskDecoderの前に高解像度特徴の前処理
        # conv_s0/conv_s1で256→32/64チャネルに圧縮
        if hasattr(self.model.sam_mask_decoder, 'conv_s0') and hasattr(self.model.sam_mask_decoder, 'conv_s1'):
            feat_s0_compressed = self.model.sam_mask_decoder.conv_s0(high_res_features[0])  # 256 -> 32
            feat_s1_compressed = self.model.sam_mask_decoder.conv_s1(high_res_features[1])  # 256 -> 64
            high_res_features_compressed = [feat_s0_compressed, feat_s1_compressed]
        else:
            high_res_features_compressed = None
        
        # マスクデコーダ（LISA_Model.forwardの実装に従う）
        low_res_masks, iou_predictions, sam_tokens_out, obj_score_logits = self.model.sam_mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,  # 単一マスク出力
            repeat_image=True,  # バッチサイズの不一致を許可
            high_res_features=high_res_features_compressed  # 圧縮済み高解像度特徴
        )
        
        # アップサンプリング
        pred_masks = F.interpolate(
            low_res_masks,
            size=(h, w),
            mode='bilinear',
            align_corners=False
        )
        
        return pred_masks, gt_mask
    
    def compute_losses(self, pred_masks, gt_masks):
        """損失計算"""
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
        """可視化の保存"""
        if step % 50 != 0:
            return
        
        # 元画像を取得（SAM用画像から復元）
        sam_image = batch['sam_image'][0].cpu()  # [3, 1024, 1024]
        
        # 正規化を解除
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        sam_image = sam_image * std + mean
        sam_image = torch.clamp(sam_image, 0, 1)
        
        # numpy配列に変換
        image_np = sam_image.permute(1, 2, 0).numpy()
        
        # マスクを取得
        pred_mask_np = torch.sigmoid(pred_masks[0, 0]).cpu().numpy()
        gt_mask_np = gt_masks[0, 0].cpu().numpy()
        
        # 可視化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 元画像
        axes[0].imshow(image_np)
        axes[0].set_title(f'Input Image (Step {step})')
        axes[0].axis('off')
        
        # GT マスク
        axes[1].imshow(gt_mask_np, cmap='gray')
        axes[1].set_title('GT Mask')
        axes[1].axis('off')
        
        # 予測マスク
        axes[2].imshow(pred_mask_np, cmap='gray')
        axes[2].set_title(f'Predicted Mask')
        axes[2].axis('off')
        
        plt.tight_layout()
        save_path = self.output_dir / f'visualization_step_{step}.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization to {save_path}")
    
    def train(self):
        """訓練ループ"""
        logger.info("Starting training...")
        
        # オプティマイザ
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # 訓練ループ
        self.model.train()
        global_step = 0
        
        with tqdm(total=self.config.max_steps) as pbar:
            while global_step < self.config.max_steps:
                for batch in self.dataloader:
                    if global_step >= self.config.max_steps:
                        break
                    
                    # オラクルプロンプトでマスク生成
                    pred_masks, gt_masks = self.generate_mask_with_oracle_prompt(batch, global_step)
                    
                    if pred_masks is None:
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
                    if global_step % 10 == 0:
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
        
        logger.info("Training completed!")
        
        # 最終結果の保存
        self.save_results()
    
    def save_results(self):
        """結果の保存"""
        # 損失グラフ
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Total Loss
        axes[0, 0].plot(self.loss_history['steps'], self.loss_history['total_loss'])
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # BCE Loss
        axes[0, 1].plot(self.loss_history['steps'], self.loss_history['bce_loss'])
        axes[0, 1].set_title('BCE Loss')
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
        
        # Dice Score
        axes[1, 0].plot(self.loss_history['steps'], self.loss_history['dice_score'])
        axes[1, 0].set_title('Dice Score')
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].grid(True)
        axes[1, 0].set_ylim([0, 1])
        
        # IoU Score
        axes[1, 1].plot(self.loss_history['steps'], self.loss_history['iou_score'])
        axes[1, 1].set_title('IoU Score')
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].grid(True)
        axes[1, 1].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 履歴をJSONで保存
        with open(self.output_dir / 'training_history.json', 'w') as f:
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
    
    parser = argparse.ArgumentParser(description='A-1 Oracle SAM Test')
    parser.add_argument('--num_samples', type=int, default=1,
                       help='Number of samples to use (will repeat the same sample)')
    parser.add_argument('--max_steps', type=int, default=1000,
                       help='Maximum training steps')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--lora_r', type=int, default=8,
                       help='LoRA rank (0 to disable LoRA)')
    parser.add_argument('--lora_alpha', type=int, default=16,
                       help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                       help='LoRA dropout')
    
    args = parser.parse_args()
    
    logger.info("="*50)
    logger.info("A-1 Oracle SAM Test - Current Dataset Version")
    logger.info("="*50)
    logger.info(f"Configuration:")
    logger.info(f"  Num samples: {args.num_samples}")
    logger.info(f"  Max steps: {args.max_steps}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  LoRA r: {args.lora_r}")
    
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