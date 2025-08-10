#!/usr/bin/env python3
"""
A-1テスト: SAM2.1側の"オラクル・プロンプト"過学習テスト（VLMを迂回）
LISA_Modelの実装を使用したバージョン

狙い: マスクデコーダ経路（＋LoRA）が正常に学習できるかを、テキスト抜きでまず切り分ける。
手順:
1. LISA_ModelからSAM部分を取り出し、Qwenを完全凍結
2. GTの重心点を直接SAM2.1のPromptEncoderに入力
3. 1画像1マスクの極小データ（1〜4枚）で学習
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

from src.config import LISAConfig
from src.models import LISA_Model
from src.utils import prepare_tokenizer_for_lisa
from transformers import AutoProcessor
from src.data.dataset import HybridDataset
from src.data.collators import MultiModalDataCollator
from peft import LoraConfig, get_peft_model, TaskType
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex

# ロギング設定
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


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
        
        # トークナイザーとプロセッサの準備（画像処理のみ使用）
        self.tokenizer = prepare_tokenizer_for_lisa(
            model_name=self.lisa_config.qwen_model_name,
            seg_token=self.lisa_config.seg_token
        )
        self.processor = AutoProcessor.from_pretrained(self.lisa_config.qwen_model_name)
        
        # LISA_Modelの読み込み
        logger.info("Loading LISA_Model...")
        self.model = LISA_Model(self.lisa_config)
        self.model.set_tokenizer(self.tokenizer)
        
        # SAM MaskDecoderへのLoRA適用
        if self.config.use_lora and self.config.lora_r > 0:
            logger.info(f"Applying LoRA to SAM MaskDecoder (r={self.config.lora_r})")
            self.apply_lora_to_sam()
        
        # デバイスに移動
        self.model = self.model.to(self.device)
        
        # Qwenパラメータを完全凍結
        for param in self.model.qwen.parameters():
            param.requires_grad = False
        
        # アダプタも凍結（SAMのみ学習）
        for param in self.model.image_adapter.parameters():
            param.requires_grad = False
        for param in self.model.text_prompt_proj.parameters():
            param.requires_grad = False
        if hasattr(self.model, 'high_res_generator'):
            for param in self.model.high_res_generator.parameters():
                param.requires_grad = False
        if hasattr(self.model, 'token_fpn'):
            for param in self.model.token_fpn.parameters():
                param.requires_grad = False
        
        # パラメータ統計
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")
    
    def apply_lora_to_sam(self):
        """SAM MaskDecoderにLoRAを適用"""
        # SAM MaskDecoderの特定レイヤーにLoRAを適用
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=["transformer"],  # MaskDecoderのTransformer部分
            lora_dropout=0.1,
            bias="none",
            modules_to_save=[]
        )
        
        # SAM部分のみLoRAを適用
        from peft import inject_adapter_in_model
        
        # MaskDecoderのTransformerにLoRAを適用
        for name, module in self.model.sam_mask_decoder.named_modules():
            if 'transformer' in name and hasattr(module, 'layers'):
                for layer in module.layers:
                    # self_attnとmlpにLoRAを適用
                    if hasattr(layer, 'self_attn'):
                        for sub_name in ['q_proj', 'v_proj']:
                            if hasattr(layer.self_attn, sub_name):
                                linear = getattr(layer.self_attn, sub_name)
                                # LoRAレイヤーを手動で追加
                                in_features = linear.in_features
                                out_features = linear.out_features
                                
                                # LoRA A/Bマトリックスを作成
                                lora_A = nn.Parameter(torch.randn(self.config.lora_r, in_features) * 0.01)
                                lora_B = nn.Parameter(torch.zeros(out_features, self.config.lora_r))
                                
                                # パラメータとして登録
                                linear.register_parameter(f'lora_A_{sub_name}', lora_A)
                                linear.register_parameter(f'lora_B_{sub_name}', lora_B)
                                
                                # 元の重みは凍結
                                linear.weight.requires_grad = False
                                if linear.bias is not None:
                                    linear.bias.requires_grad = False
    
    def setup_data(self):
        """極小データセットのセットアップ"""
        logger.info("Setting up minimal dataset...")
        
        # データセット作成（sem_segのみ、極小サンプル）
        self.dataset = HybridDataset(
            base_image_dir=self.lisa_config.dataset_base_dir,
            qwen_processor=self.processor,
            samples_per_epoch=self.config.num_samples,  # 1-4サンプル
            dataset='sem_seg',  # ADE20Kのみ
            sample_rate=[1.0],
            qwen_image_size=self.lisa_config.qwen_image_size,
            sam_image_size=self.lisa_config.sam_image_size,
        )
        
        # DataCollator
        self.collator = MultiModalDataCollator(
            tokenizer=self.tokenizer,
            max_length=self.lisa_config.model_max_length,
            config=self.lisa_config
        )
        
        # DataLoader（バッチサイズ1で固定）
        # 重要: 1サンプルのみを繰り返し使用するため、単一のサンプルを明示的に選択
        from torch.utils.data import Subset
        # 最初のサンプルのみを使用するSubsetDatasetを作成
        single_sample_dataset = Subset(self.dataset, [0])
        
        self.dataloader = DataLoader(
            single_sample_dataset,
            batch_size=1,
            shuffle=False,  # 過学習テストなので固定順序
            collate_fn=self.collator,
            num_workers=0
        )
        
        logger.info(f"Dataset size: {len(self.dataset)}")
    
    def compute_mask_centroid(self, mask):
        """マスクから重心座標を計算"""
        # 4次元の場合: [B, C, H, W] -> [H, W]
        if mask.dim() == 4:
            mask = mask.squeeze(0).squeeze(0)
        # 3次元の場合: [C, H, W] -> [H, W]
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
        # デバッグ: バッチのキーを確認
        if step == 0:
            logger.info(f"Batch keys: {batch.keys()}")
            for key, value in batch.items():
                if torch.is_tensor(value):
                    logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                elif isinstance(value, list):
                    logger.info(f"  {key}: list of {len(value)} items")
                else:
                    logger.info(f"  {key}: type={type(value)}")
        
        # バッチから必要なデータを取得
        pixel_values = batch['pixel_values'].to(self.device)  # [1, num_patches, 1176]
        gt_masks = batch['mask_labels']  # List of tensors
        
        # SAM用の画像を準備（pixel_valuesは既にパッチ化されているので、元画像のサイズを推定）
        # minimal_train.pyではimage_grid_thwから元画像サイズを推定している
        if 'image_grid_thw' in batch and batch['image_grid_thw'] is not None:
            # 動的解像度の場合
            image_grid_thw = batch['image_grid_thw']
            H_grid_raw = int(image_grid_thw[0, 1].item())
            W_grid_raw = int(image_grid_thw[0, 2].item())
            # 14px per patch
            orig_h = H_grid_raw * 14
            orig_w = W_grid_raw * 14
        else:
            # 固定解像度の場合（448x448がデフォルト）
            orig_h = orig_w = 448
        
        # 最初のGTマスクを取得
        gt_mask = gt_masks[0] if isinstance(gt_masks, list) else gt_masks
        if isinstance(gt_mask, list):
            gt_mask = gt_mask[0] if len(gt_mask) > 0 else None
        
        if gt_mask is None:
            logger.warning(f"No GT mask found for step {step}")
            return None
        
        gt_mask = gt_mask.to(self.device)
        
        # 複数マスクの場合は最初のマスクのみ使用
        if gt_mask.dim() == 3 and gt_mask.shape[0] > 1:
            gt_mask = gt_mask[0]
        
        # Qwen経由で画像特徴を抽出（ただし勾配は流さない）
        with torch.no_grad():
            # minimal_train.pyと同じように、実際のinput_idsを使う
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Qwenで画像特徴を抽出
            qwen_outputs = self.model.qwen(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                image_grid_thw=batch.get('image_grid_thw'),
                output_hidden_states=True,
                return_dict=True
            )
            
            # ビジョン特徴を抽出
            vision_features = self.model.extract_vision_features(
                pixel_values, 
                batch.get('image_grid_thw')
            )
            
            if len(vision_features.shape) == 2:
                vision_features = vision_features.unsqueeze(0)
            
            # アダプタで変換
            image_features_sam = self.model.image_adapter(vision_features, batch.get('image_grid_thw'))
        
        # 高解像度特徴を生成（勾配あり）
        if self.model.use_token_fpn:
            image_features_sam_fpn, sam_high_res_features = self.model.token_fpn(
                image_features_sam,
                image_grid_thw=batch.get('image_grid_thw'),
                use_hooks=False  # フックは使わない（Qwenは凍結）
            )
            image_features_sam = image_features_sam_fpn
        else:
            sam_high_res_features = self.model.high_res_generator(image_features_sam)
        
        # GTマスクから重心を計算
        h_feat, w_feat = image_features_sam.shape[-2:]
        h_img, w_img = h_feat * 16, w_feat * 16
        
        centroid = self.compute_mask_centroid(gt_mask)
        
        # マスクとSAM特徴のサイズを調整
        if gt_mask.shape[-2:] != (h_img, w_img):
            gt_mask_resized = F.interpolate(
                gt_mask.unsqueeze(0).unsqueeze(0).float(),
                size=(h_img, w_img),
                mode='nearest'
            ).squeeze(0).squeeze(0)
            # 重心も再計算
            centroid = self.compute_mask_centroid(gt_mask_resized)
        else:
            gt_mask_resized = gt_mask
        
        cx, cy = centroid[0].item(), centroid[1].item()
        
        if step % 10 == 0:
            logger.info(f"Step {step}: Centroid at ({cx:.1f}, {cy:.1f}), Image size: {w_img}x{h_img}")
        
        # SAMのPromptEncoderで点プロンプトを生成
        point_coords = torch.tensor([[cx, cy]], dtype=torch.float32, device=self.device).unsqueeze(0)
        point_labels = torch.tensor([1], dtype=torch.int32, device=self.device).unsqueeze(0)
        
        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )
        
        # 密な埋め込みのサイズを調整
        if dense_embeddings.shape[-2:] != image_features_sam.shape[-2:]:
            dense_embeddings = F.interpolate(
                dense_embeddings,
                size=image_features_sam.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        # SAM MaskDecoderでマスクを生成
        image_pe = self.model.sam_prompt_encoder.get_dense_pe()
        if image_pe.shape[-2:] != image_features_sam.shape[-2:]:
            image_pe = F.interpolate(
                image_pe,
                size=image_features_sam.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        # 高解像度特徴の準備
        if sam_high_res_features is not None and len(sam_high_res_features) >= 2:
            feat_s0 = sam_high_res_features[0]
            feat_s1 = sam_high_res_features[1]
            
            # チャネル圧縮（データタイプを統一）
            if hasattr(self.model.sam_mask_decoder, 'conv_s0'):
                # SAM MaskDecoderのdtypeに合わせる
                sam_dtype = next(self.model.sam_mask_decoder.parameters()).dtype
                feat_s0 = self.model.sam_mask_decoder.conv_s0(feat_s0.to(dtype=sam_dtype))
                feat_s1 = self.model.sam_mask_decoder.conv_s1(feat_s1.to(dtype=sam_dtype))
            
            high_res_features = [feat_s0, feat_s1]
        else:
            high_res_features = None
        
        # マスク生成
        low_res_masks, _, _, _ = self.model.sam_mask_decoder(
            image_embeddings=image_features_sam,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=True,
            high_res_features=high_res_features,
        )
        
        # アップサンプリング
        pred_mask = F.interpolate(
            low_res_masks,
            size=(h_img, w_img),
            mode='bilinear',
            align_corners=False
        )
        
        return pred_mask.squeeze(0).squeeze(0), gt_mask_resized
    
    def train_step(self, batch, step):
        """1ステップの学習"""
        # オラクルプロンプトでマスク生成
        result = self.generate_mask_with_oracle_prompt(batch, step)
        
        if result is None:
            return None
        
        pred_logits, gt_mask = result
        
        # 損失計算（サイズを確認して合わせる）
        # gt_maskが4次元の場合はsqueezeする
        if gt_mask.dim() == 4:
            gt_mask = gt_mask.squeeze(0).squeeze(0)
        elif gt_mask.dim() == 3:
            gt_mask = gt_mask.squeeze(0)
        
        bce_loss = F.binary_cross_entropy_with_logits(pred_logits, gt_mask.float())
        
        # Dice損失
        pred_sigmoid = torch.sigmoid(pred_logits)
        intersection = (pred_sigmoid * gt_mask).sum()
        dice = 2 * intersection / (pred_sigmoid.sum() + gt_mask.sum() + 1e-8)
        dice_loss = 1 - dice
        
        # 総合損失
        total_loss = bce_loss + dice_loss
        
        # メトリクス計算
        dice_score = self.dice_metric(pred_sigmoid.unsqueeze(0), gt_mask.unsqueeze(0))
        iou_score = self.iou_metric(pred_sigmoid.unsqueeze(0), gt_mask.unsqueeze(0))
        
        return {
            'total_loss': total_loss,
            'bce_loss': bce_loss,
            'dice_loss': dice_loss,
            'dice_score': dice_score,
            'iou_score': iou_score,
            'pred_mask': pred_sigmoid.detach().cpu().numpy(),
            'gt_mask': gt_mask.detach().cpu().numpy()
        }
    
    def visualize_results(self, step, results, batch):
        """結果の可視化"""
        if step % self.config.vis_interval != 0:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # original_imagesがcollatorから渡されている場合はそれを使用
        # MultiModalDataCollatorは'original_images'（複数形）として渡す
        if 'original_images' in batch and batch['original_images'] is not None and len(batch['original_images']) > 0:
            # PIL画像をnumpy配列に変換
            from PIL import Image
            # リストから最初の画像を取得
            original_img = batch['original_images'][0]
            if isinstance(original_img, Image.Image):
                # PIL画像の場合
                img_np = np.array(original_img)
            elif isinstance(original_img, torch.Tensor):
                # Tensorの場合（不要になるはず）
                img_tensor = original_img
                if img_tensor.dim() == 4:
                    img_tensor = img_tensor[0]
                if img_tensor.shape[0] == 3:
                    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                else:
                    img_np = img_tensor.cpu().numpy()
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
            else:
                # その他の場合
                img_np = np.ones((512, 512, 3), dtype=np.uint8) * 128
            
            axes[0].imshow(img_np)
            axes[0].set_title(f'Input Image ({img_np.shape[0]}x{img_np.shape[1]})')
        else:
            # original_imageがない場合はグレーのプレースホルダー
            if 'image_grid_thw' in batch and batch['image_grid_thw'] is not None:
                H_grid = int(batch['image_grid_thw'][0, 1].item())
                W_grid = int(batch['image_grid_thw'][0, 2].item())
            else:
                H_grid = W_grid = 32
            dummy_image = np.ones((H_grid * 14, W_grid * 14, 3)) * 0.5
            axes[0].imshow(dummy_image, cmap='gray')
            axes[0].set_title(f'Input Image ({H_grid*14}x{W_grid*14})')
        axes[0].axis('off')
        
        # GT マスク
        axes[1].imshow(results['gt_mask'], cmap='gray')
        axes[1].set_title('GT Mask')
        axes[1].axis('off')
        
        # 予測マスク
        axes[2].imshow(results['pred_mask'], cmap='gray')
        axes[2].set_title(f'Predicted Mask (Dice: {results["dice_score"]:.3f})')
        axes[2].axis('off')
        
        plt.tight_layout()
        save_path = self.output_dir / f'visualization_step_{step}.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    def plot_loss_curves(self):
        """Loss曲線の描画"""
        if len(self.loss_history['steps']) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Total Loss
        ax = axes[0, 0]
        ax.plot(self.loss_history['steps'], self.loss_history['total_loss'], 'b-')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Total Loss')
        ax.set_title('Total Loss (BCE + Dice)')
        ax.grid(True, alpha=0.3)
        
        # BCE Loss
        ax = axes[0, 1]
        ax.plot(self.loss_history['steps'], self.loss_history['bce_loss'], 'g-')
        ax.set_xlabel('Steps')
        ax.set_ylabel('BCE Loss')
        ax.set_title('Binary Cross Entropy Loss')
        ax.grid(True, alpha=0.3)
        
        # Dice Score
        ax = axes[1, 0]
        ax.plot(self.loss_history['steps'], self.loss_history['dice_score'], 'r-')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Dice Score')
        ax.set_title('Dice Score (F1)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.95, color='k', linestyle='--', alpha=0.5, label='Target (0.95)')
        ax.legend()
        
        # IoU Score
        ax = axes[1, 1]
        ax.plot(self.loss_history['steps'], self.loss_history['iou_score'], 'm-')
        ax.set_xlabel('Steps')
        ax.set_ylabel('IoU Score')
        ax.set_title('Intersection over Union')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.9, color='k', linestyle='--', alpha=0.5, label='Target (0.9)')
        ax.legend()
        
        plt.tight_layout()
        save_path = self.output_dir / 'loss_curves.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Loss curves saved to {save_path}")
    
    def run_test(self):
        """テストの実行"""
        logger.info("=" * 50)
        logger.info("Starting A-1 Test: Oracle SAM2.1 Overfitting (LISA_Model)")
        logger.info("=" * 50)
        
        # モデルとデータのセットアップ
        self.setup_model()
        self.setup_data()
        
        # オプティマイザ（SAMのLoRAパラメータのみ）
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            logger.warning("No trainable parameters found! Check LoRA setup.")
            # 手動でSAM MaskDecoderのLoRAパラメータを収集
            trainable_params = []
            for name, param in self.model.sam_mask_decoder.named_parameters():
                if 'lora' in name.lower() and param.requires_grad:
                    trainable_params.append(param)
                    logger.info(f"Found LoRA param: {name}")
        
        if len(trainable_params) == 0:
            logger.error("No trainable parameters found even after manual search!")
            return False
        
        optimizer = torch.optim.AdamW(trainable_params, lr=self.config.lr, weight_decay=0)
        logger.info(f"Optimizer created with {len(trainable_params)} trainable parameters")
        
        # 学習ループ
        step = 0
        best_dice = 0.0
        
        self.model.train()
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            
            for batch in tqdm(self.dataloader, desc=f"Epoch {epoch+1}"):
                # Forward pass
                results = self.train_step(batch, step)
                
                if results is None:
                    continue
                
                # Backward pass
                optimizer.zero_grad()
                results['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                
                # 記録
                self.loss_history['steps'].append(step)
                self.loss_history['total_loss'].append(results['total_loss'].item())
                self.loss_history['bce_loss'].append(results['bce_loss'].item())
                self.loss_history['dice_loss'].append(results['dice_loss'].item())
                self.loss_history['dice_score'].append(results['dice_score'].item())
                self.loss_history['iou_score'].append(results['iou_score'].item())
                
                # ベストモデルの更新
                if results['dice_score'] > best_dice:
                    best_dice = results['dice_score']
                    logger.info(f"New best Dice score: {best_dice:.4f}")
                
                # 可視化
                self.visualize_results(step, results, batch)
                
                # ログ出力
                if step % 10 == 0:
                    logger.info(
                        f"Step {step}: Loss={results['total_loss']:.4f}, "
                        f"BCE={results['bce_loss']:.4f}, "
                        f"Dice Loss={results['dice_loss']:.4f}, "
                        f"Dice Score={results['dice_score']:.4f}, "
                        f"IoU={results['iou_score']:.4f}"
                    )
                
                # 早期終了条件
                if results['dice_score'] >= 0.95 and results['iou_score'] >= 0.9:
                    logger.info(f"Target achieved! Dice={results['dice_score']:.4f}, IoU={results['iou_score']:.4f}")
                    break
                
                step += 1
                
                if step >= self.config.max_steps:
                    break
            
            if step >= self.config.max_steps:
                break
        
        # 最終結果
        logger.info("=" * 50)
        logger.info("Test Complete!")
        logger.info(f"Best Dice Score: {best_dice:.4f}")
        if len(self.loss_history['dice_score']) > 0:
            logger.info(f"Final Dice Score: {self.loss_history['dice_score'][-1]:.4f}")
            logger.info(f"Final IoU Score: {self.loss_history['iou_score'][-1]:.4f}")
            logger.info(f"Final Total Loss: {self.loss_history['total_loss'][-1]:.4f}")
        
        # Loss曲線の保存
        self.plot_loss_curves()
        
        # 履歴の保存
        if len(self.loss_history['steps']) > 0:
            with open(self.output_dir / 'test_results.json', 'w') as f:
                json.dump({
                    'config': vars(self.config),
                    'loss_history': self.loss_history,
                    'best_dice': float(best_dice),
                    'final_metrics': {
                        'dice': float(self.loss_history['dice_score'][-1]),
                        'iou': float(self.loss_history['iou_score'][-1]),
                        'total_loss': float(self.loss_history['total_loss'][-1])
                    }
                }, f, indent=2)
        
        # 成功判定
        success = (len(self.loss_history['dice_score']) > 0 and 
                  self.loss_history['dice_score'][-1] >= 0.95 and 
                  self.loss_history['iou_score'][-1] >= 0.9)
        
        if success:
            logger.info("✅ Test PASSED: SAM2.1 in LISA_Model can learn with oracle prompts!")
        else:
            logger.warning("❌ Test FAILED: SAM2.1 learning issues detected")
            logger.warning("Check LoRA configuration, learning rate, or data preprocessing")
        
        return success


def main():
    import argparse
    parser = argparse.ArgumentParser(description="A-1 Test: Oracle SAM2.1 Overfitting with LISA_Model")
    
    # データ設定
    parser.add_argument('--data_dir', type=str, 
                       default='/home/oda/VLM_RAG_MultiModal_Dataset',
                       help='Dataset base directory')
    parser.add_argument('--num_samples', type=int, default=1,
                       help='Number of samples for overfitting (1-4)')
    
    # 学習設定
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--max_steps', type=int, default=500,
                       help='Maximum training steps')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    
    # LoRA設定
    parser.add_argument('--use_lora', action='store_true', default=True,
                       help='Use LoRA for SAM MaskDecoder')
    parser.add_argument('--lora_r', type=int, default=8,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32,
                       help='LoRA alpha')
    
    # その他
    parser.add_argument('--vis_interval', type=int, default=50,
                       help='Visualization interval')
    
    args = parser.parse_args()
    
    # テスト実行
    tester = OracleSAMTester(args)
    success = tester.run_test()
    
    # 終了コード
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()