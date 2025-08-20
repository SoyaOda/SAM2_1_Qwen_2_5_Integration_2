#!/usr/bin/env python3
"""
Test A-3: 段階的ゲーティング（位置優先→テキスト混合）によるseg loss改善テスト

解決策A: 学習初期はSAMの位置付きsparse embeddingを主にし、
学習が進むほどLLM埋め込みの比率を上げる。

期待される結果:
- seg lossが徐々に減少する
- 学習初期は位置情報を保持しつつ、徐々にLLM埋め込みの情報を取り込む
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import logging
from tqdm import tqdm
from typing import Optional, Tuple
import math

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 共通設定をインポート
from test_common_config import (
    SEED, NUM_STEPS, BATCH_SIZE, LEARNING_RATE, NUM_FIXED_SAMPLES,
    DATASET_TYPE, SAMPLE_RATE, VIS_INTERVAL, LOG_INTERVAL, EVAL_THRESHOLD,
    set_random_seed, FixedSampleDataset, save_test_config, create_standard_output_structure
)

from src.models.lisa_model import LISA_Model
from src.config import LISAConfig
from src.data.dataset import HybridDataset
from src.data.collators import MultiModalDataCollator
from src.utils.tokenizer_utils import prepare_tokenizer_for_lisa
from transformers import AutoProcessor

# ロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# パラメータは共通設定から使用
num_steps = NUM_STEPS
batch_size = BATCH_SIZE
learning_rate = LEARNING_RATE
dataset_type = DATASET_TYPE
samples_per_epoch = NUM_FIXED_SAMPLES

# 段階的ゲーティングのパラメータ（test_a3固有）
warmup_steps = 500  # NUM_STEPSの半分でウォームアップ
max_alpha = 0.5     # 最大値を下げて、位置情報をより保持


def compute_alpha_schedule(step: int, warmup_steps: int = 200, max_alpha: float = 0.7) -> float:
    """
    段階的にLLM埋め込みの混合比率を増やすスケジュール
    
    Args:
        step: 現在の学習ステップ
        warmup_steps: ウォームアップステップ数
        max_alpha: 最大混合比率
    
    Returns:
        現在のステップでの混合比率 (0.0 ~ max_alpha)
    """
    if step >= warmup_steps:
        return max_alpha
    
    # コサインスケジュールで滑らかに増加
    progress = step / warmup_steps
    alpha = max_alpha * 0.5 * (1 - math.cos(math.pi * progress))
    return alpha


def save_visualization(output_dir, step, batch, pred_mask, gt_mask, losses, tokenizer, alpha):
    """可視化結果を保存（test_a1スタイル、alpha値も表示）"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 可視化を指定間隔で保存
    if step % VIS_INTERVAL != 0:
        return
    
    # 画像の取得
    image_np = None
    if 'original_images' in batch and batch['original_images'] is not None and len(batch['original_images']) > 0:
        from PIL import Image
        original_img = batch['original_images'][0]
        if isinstance(original_img, Image.Image):
            image_np = np.array(original_img)
    
    if image_np is None and 'pixel_values' in batch and batch['pixel_values'] is not None:
        pixel_values = batch['pixel_values'][0].cpu()
        if pixel_values.dim() == 3:
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
            image = pixel_values * std + mean
            image = torch.clamp(image, 0, 1)
            image_np = image.permute(1, 2, 0).numpy()
    
    if image_np is None:
        return  # 画像がない場合はスキップ
    
    # マスクの準備
    pred_mask_np = torch.sigmoid(pred_mask).detach().cpu().numpy()
    gt_mask_np = gt_mask.cpu().numpy()
    
    # 2x3レイアウトの図を作成
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
    color = 'red' if alpha > 0.3 else 'green'
    ax3.set_title(f'Predicted Mask (α={alpha:.3f})', fontsize=12, fontweight='bold', color=color)
    ax3.axis('off')
    
    # 下段: オーバーレイ比較
    ax4 = plt.subplot(2, 3, 4)
    ax4.imshow(image_np)
    # GTマスクオーバーレイ（赤）
    from skimage.transform import resize
    gt_mask_resized = resize(gt_mask_np, (image_np.shape[0], image_np.shape[1]), order=0)
    mask_overlay = np.zeros_like(image_np)
    mask_overlay[:, :, 0] = gt_mask_resized
    ax4.imshow(mask_overlay, alpha=0.3)
    ax4.set_title('Image + GT Mask', fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    ax5 = plt.subplot(2, 3, 5)
    ax5.imshow(image_np)
    # 予測マスクオーバーレイ（緑）
    pred_mask_resized = resize(pred_mask_np, (image_np.shape[0], image_np.shape[1]), order=1)
    pred_overlay = np.zeros_like(image_np)
    pred_overlay[:, :, 1] = pred_mask_resized
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
    
    # メトリクス情報
    info_text = f"Step: {step}\n"
    info_text += f"α (LLM mix ratio): {alpha:.3f}\n"
    info_text += f"Warmup progress: {min(step/warmup_steps*100, 100):.1f}%\n\n"
    info_text += f"Losses:\n"
    info_text += f"  Total: {losses['total']:.4f}\n"
    info_text += f"  LM: {losses['lm']:.4f}\n"
    info_text += f"  Seg: {losses['seg']:.4f}\n\n"
    info_text += f"Metrics:\n"
    info_text += f"  Dice Score: {dice:.4f}\n"
    info_text += f"  IoU: {iou:.4f}\n\n"
    
    # αの解釈
    if alpha < 0.1:
        info_text += "Mode: Pure SAM position embedding\n"
    elif alpha < 0.3:
        info_text += "Mode: Position-dominant mixing\n"
    elif alpha < max_alpha * 0.8:
        info_text += "Mode: Balanced mixing\n"
    else:
        info_text += "Mode: LLM-dominant mixing\n"
    
    ax6.text(0.05, 0.95, info_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            family='monospace')
    ax6.set_title('Gating Information', fontsize=12, fontweight='bold')
    
    plt.suptitle(f'Gated Embedding Test - Step {step} (α={alpha:.3f})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / 'visualizations' / f'step_{step:05d}.png'
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved visualization to {save_path} (Dice: {dice:.4f}, IoU: {iou:.4f}, α: {alpha:.3f})")


def main():
    # シード固定
    set_random_seed(SEED)
    
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 出力ディレクトリ作成（共通構造）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = create_standard_output_structure(f"test_outputs/test_a3_{timestamp}")
    logger.info(f"Output directory: {output_dir}")
    
    # テスト設定を保存（最初は基本設定のみ）
    save_test_config(output_dir, 'test_a3_gated_embedding', None, {
        'test_description': 'Gradual gating from position to LLM embeddings',
        'warmup_steps': warmup_steps,
        'max_alpha': max_alpha
    })
    
    # 1. モデルとトークナイザの準備
    config = LISAConfig(
        qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        sam_model_name="./checkpoints/sam2.1_hiera_large.pt",
        device_map="cuda",
        torch_dtype="auto",
        use_flash_attention=False,
        # Freeze configuration - Test A3: Gated Embedding
        freeze_qwen_lora=False,         # Qwen LoRA有効 (trainable)
        freeze_seg_token=False,         # SEGトークン学習可能
        freeze_sam_lora=False,          # SAM LoRA有効
        freeze_image_adapter=False,     # アダプター学習可能
        freeze_text_prompt_projector=False,  # プロジェクター学習可能
        freeze_token_fpn=False,         # Token-FPN学習可能
        freeze_prompt_beta=True         # A3: ゲーティングアプローチではalphaを使用、betaは不使用
    )
    
    # プロセッサーの準備（動的解像度対応）
    if config.use_dynamic_resolution:
        processor = AutoProcessor.from_pretrained(
            config.qwen_model_name,
            min_pixels=config.qwen_min_pixels,
            max_pixels=config.qwen_max_pixels
        )
    else:
        processor = AutoProcessor.from_pretrained(config.qwen_model_name)
    
    # Processorのトークナイザにも<SEG>トークン追加
    if config.seg_token not in processor.tokenizer.get_vocab():
        processor.tokenizer.add_special_tokens({"additional_special_tokens": [config.seg_token]})
    
    # カスタムLISAモデルクラス（段階的ゲーティング対応）
    class LISA_Model_Gated(LISA_Model):
        """段階的ゲーティングを実装したLISAモデル"""
        
        def forward_with_gating(
            self,
            input_ids: torch.Tensor,
            pixel_values: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            mask_labels: Optional[list] = None,
            image_grid_thw: Optional[torch.Tensor] = None,
            sam_images: Optional[torch.Tensor] = None,  # SAM画像を追加
            alpha: float = 0.0,  # LLM埋め込みの混合比率
        ):
            """段階的ゲーティングを適用したforward"""
            
            B = input_ids.size(0)
            
            # 1. Qwenモデルの実行
            qwen_outputs = self.qwen(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                image_grid_thw=image_grid_thw,
                output_hidden_states=True,
                return_dict=True
            )
            
            logits = qwen_outputs.logits
            hidden_states = qwen_outputs.hidden_states[-1]
            
            # 2. Vision features（SAM ImageEncoderを含む）
            vision_features = None
            image_features_sam = None
            sam_high_res_features = None
            
            if pixel_values is not None:
                vision_features = self.extract_vision_features(pixel_values, image_grid_thw)
                if len(vision_features.shape) == 2:
                    vision_features = vision_features.unsqueeze(0)
                
                image_features_sam = self.image_adapter(vision_features, image_grid_thw)
                
                # **SAM ImageEncoderを呼び出す**
                if sam_images is not None:
                    # 標準のLISA forwardメソッドと同じSAM処理を適用
                    sam_outputs = self(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        attention_mask=attention_mask,
                        labels=labels,
                        mask_labels=mask_labels,
                        image_grid_thw=image_grid_thw,
                        sam_images=sam_images,
                        return_dict=True
                    )
                    # SAM処理済みの結果を使用
                    return sam_outputs.logits, sam_outputs.mask_logits
                
                if self.use_token_fpn:
                    image_features_sam_fpn, sam_high_res_features = self.token_fpn(
                        image_features_sam,
                        image_grid_thw=image_grid_thw,
                        use_hooks=True
                    )
                    image_features_sam = image_features_sam_fpn
                else:
                    sam_high_res_features = self.high_res_generator(image_features_sam)
            
            # 3. SEGトークン位置の検出
            mask_logits = []
            seg_positions = []
            
            if self.seg_token_id is None:
                return logits, mask_logits
            
            # ラベルまたは入力からSEG位置を取得
            if labels is not None:
                for i in range(B):
                    seg_pos = (labels[i] == self.seg_token_id).nonzero(as_tuple=True)[0]
                    if len(seg_pos) > 0:
                        seg_positions.append(seg_pos.tolist())
                    else:
                        seg_positions.append([])
            else:
                for i in range(B):
                    seg_pos = (input_ids[i] == self.seg_token_id).nonzero(as_tuple=True)[0]
                    if len(seg_pos) > 0:
                        seg_positions.append(seg_pos.tolist())
                    else:
                        seg_positions.append([])
            
            # 4. マスク生成（段階的ゲーティング適用）
            for i in range(B):
                sample_masks = []
                
                if len(seg_positions[i]) > 0 and image_features_sam is not None and i < image_features_sam.shape[0]:
                    # SEG位置の隠れ状態を取得
                    seg_hidden_states = hidden_states[i, seg_positions[i]]
                    
                    # LLM埋め込みを256次元に投影
                    if len(seg_positions[i]) == 1:
                        e_llm = self.text_prompt_proj(seg_hidden_states)
                    else:
                        e_llm = self.text_prompt_proj(seg_hidden_states)
                    
                    for j, llm_embed in enumerate(e_llm if len(seg_positions[i]) > 1 else [e_llm]):
                        try:
                            # 位置エンコーディング
                            image_pe = self.sam_prompt_encoder.get_dense_pe()
                            
                            if image_pe.shape[-2:] != image_features_sam[i:i+1].shape[-2:]:
                                h_feat, w_feat = image_features_sam[i:i+1].shape[-2:]
                                image_pe = F.interpolate(
                                    image_pe,
                                    size=(h_feat, w_feat),
                                    mode='bilinear',
                                    align_corners=False
                                )
                            
                            # 重心座標の計算
                            h_feat, w_feat = image_features_sam[i:i+1].shape[-2:]
                            h_img, w_img = h_feat * 16, w_feat * 16
                            
                            if mask_labels is not None and i < len(mask_labels) and mask_labels[i] is not None:
                                gt_mask = mask_labels[i]
                                if isinstance(gt_mask, list):
                                    gt_mask = gt_mask[j] if len(gt_mask) > j else gt_mask[0]
                                
                                if gt_mask is not None:
                                    centroid = self.compute_mask_centroid(gt_mask, h_img, w_img)
                                    center_x, center_y = centroid[0].item(), centroid[1].item()
                                else:
                                    center_x, center_y = w_img // 2, h_img // 2
                            else:
                                center_x, center_y = w_img // 2, h_img // 2
                            
                            # ポイント座標とラベル
                            point_coords = torch.tensor([[center_x, center_y]], 
                                                      dtype=torch.float32, 
                                                      device=device).unsqueeze(0)
                            point_labels = torch.tensor([1], dtype=torch.int32, 
                                                      device=device).unsqueeze(0)
                            
                            # SAM PromptEncoderで正規の埋め込みを生成
                            sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
                                points=(point_coords, point_labels),
                                boxes=None,
                                masks=None,
                            )
                            
                            # ========== 段階的ゲーティング ==========
                            # 元の位置付き埋め込み
                            e_pos = sparse_embeddings[:, 0, :]  # [1, 256]
                            
                            # 重み付き合成（置換ではなく混合）
                            e_mix = (1 - alpha) * e_pos + alpha * llm_embed.unsqueeze(0)
                            
                            # sparse embeddingsの更新（混合結果を使用）
                            sparse_embeddings[:, 0, :] = e_mix
                            # ==========================================
                            
                            # Dense embeddingsのサイズ調整
                            if dense_embeddings.shape[-2:] != image_features_sam[i:i+1].shape[-2:]:
                                h_feat, w_feat = image_features_sam[i:i+1].shape[-2:]
                                dense_embeddings = F.interpolate(
                                    dense_embeddings,
                                    size=(h_feat, w_feat),
                                    mode='bilinear',
                                    align_corners=False
                                )
                            
                            # 高解像度特徴の準備
                            sam_dtype = next(self.sam_mask_decoder.parameters()).dtype
                            
                            if sam_high_res_features is not None and len(sam_high_res_features) >= 2:
                                feat_s0 = sam_high_res_features[0][i:i+1].to(dtype=sam_dtype)
                                feat_s1 = sam_high_res_features[1][i:i+1].to(dtype=sam_dtype)
                                
                                if hasattr(self.sam_mask_decoder, 'conv_s0') and hasattr(self.sam_mask_decoder, 'conv_s1'):
                                    feat_s0 = self.sam_mask_decoder.conv_s0(feat_s0)
                                    feat_s1 = self.sam_mask_decoder.conv_s1(feat_s1)
                                else:
                                    raise RuntimeError("SAM2.1 MaskDecoder missing conv_s0/conv_s1")
                                
                                high_res_features = [feat_s0, feat_s1]
                            else:
                                raise RuntimeError("High-resolution features not available")
                            
                            # MaskDecoderでマスク生成
                            low_res_masks, _, _, _ = self.sam_mask_decoder(
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
                            elif pixel_values is not None and pixel_values.dim() == 4:
                                orig_h, orig_w = pixel_values.shape[-2:]
                            else:
                                orig_h = h_feat * 16
                                orig_w = w_feat * 16
                            
                            mask_logit = F.interpolate(
                                low_res_masks,
                                size=(orig_h, orig_w),
                                mode='bilinear',
                                align_corners=False
                            )
                            sample_masks.append(mask_logit.squeeze(0))
                            
                        except Exception as e:
                            logger.warning(f"Mask generation failed: {e}")
                            if pixel_values is not None:
                                orig_h, orig_w = pixel_values.shape[-2:]
                                dummy_mask = torch.zeros(1, orig_h, orig_w, device=device, dtype=torch.float32)
                                sample_masks.append(dummy_mask)
                
                mask_logits.append(sample_masks if len(sample_masks) > 0 else None)
            
            return logits, mask_logits
    
    # モデル初期化
    # トークナイザーの準備
    tokenizer = prepare_tokenizer_for_lisa(
        model_name=config.qwen_model_name,
        seg_token=config.seg_token
    )
    
    model = LISA_Model_Gated(config)
    model.set_tokenizer(tokenizer)
    
    # パラメータ統計の詳細表示
    from src.utils.model_utils import display_parameter_statistics
    display_parameter_statistics(model, logger_name=__name__)
    
    # LoRA設定（Qwen側）
    if config.freeze_qwen_base and hasattr(config, 'qwen_lora_r') and config.qwen_lora_r > 0:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=config.qwen_lora_r,
            lora_alpha=config.qwen_lora_alpha,
            target_modules=config.qwen_lora_target_modules,
            lora_dropout=config.qwen_lora_dropout,
            bias=config.qwen_lora_bias,
            task_type="CAUSAL_LM",
        )
        model.qwen = get_peft_model(model.qwen, lora_config)
        logger.info("Applied LoRA to Qwen model")
    
    model = model.to(device)
    model.train()
    
    # 学習対象パラメータの確認
    trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
    logger.info(f"Total trainable parameters: {len(trainable_params)} tensors")
    
    # LISAConfigが設定された後で詳細設定を保存
    total_params = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    save_test_config(output_dir, 'test_a3_gated_embedding', config, {
        'test_description': 'Gradual gating from position to LLM embeddings',
        'warmup_steps': warmup_steps,
        'max_alpha': max_alpha,
        'total_params': total_params,
        'trainable_params': trainable_param_count
    })
    
    # 2. データセット準備（固定サンプル版）
    base_dir = config.dataset_base_dir
    
    # ベースのHybridDatasetを作成
    base_dataset = HybridDataset(
        base_image_dir=base_dir,
        qwen_processor=processor,
        samples_per_epoch=samples_per_epoch,
        dataset=dataset_type,
        sample_rate=[1.0],
        qwen_image_size=config.qwen_image_size,
        sam_image_size=config.sam_image_size,
    )
    
    # 固定サンプル版に変換
    dataset = FixedSampleDataset(base_dataset, num_samples=samples_per_epoch)
    collator = MultiModalDataCollator(
        tokenizer=tokenizer,
        max_length=config.model_max_length,
        config=config
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,  # test_a1と同じ設定（HybridDatasetが内部でランダムサンプリング）
        collate_fn=collator, 
        num_workers=0,
        pin_memory=True
    )
    
    logger.info(f"Dataset loaded: {dataset_type}, total samples={len(dataset)}")
    
    # 3. オプティマイザ準備
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # 損失履歴の記録
    loss_history = []
    alpha_history = []
    
    # 4. 学習ループ
    step = 0
    data_iter = iter(dataloader)
    
    with tqdm(total=num_steps, desc="Training with Gated Embedding") as pbar:
        while step < num_steps:
            # データ取得
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            # デバイスへ転送
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            # **A1の修正を適用: SAM画像を元画像から生成**
            # 元画像（Qwen用）からSAM画像を生成
            if 'pixel_values' in batch and batch['pixel_values'] is not None:
                # original_imagesがあれば優先的に使用
                if 'original_images' in batch and batch['original_images'] is not None:
                    sam_images = []
                    for i, original_img in enumerate(batch['original_images']):
                        from PIL import Image
                        if isinstance(original_img, Image.Image):
                            # PIL画像の場合
                            sam_image = original_img.resize((config.sam_image_size, config.sam_image_size), Image.BICUBIC)
                            sam_array = np.array(sam_image).astype(np.float32) / 255.0
                            sam_tensor = torch.from_numpy(sam_array).permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
                            sam_images.append(sam_tensor)
                        elif isinstance(original_img, torch.Tensor):
                            # Tensorの場合
                            img_tensor = original_img
                            if img_tensor.dim() == 3 and img_tensor.shape[0] == 3:
                                # [C, H, W]形式
                                img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                                if img_np.max() <= 1.0:
                                    img_np = (img_np * 255).astype(np.uint8)
                                pil_image = Image.fromarray(img_np.astype(np.uint8))
                                sam_image = pil_image.resize((config.sam_image_size, config.sam_image_size), Image.BICUBIC)
                                sam_array = np.array(sam_image).astype(np.float32) / 255.0
                                sam_tensor = torch.from_numpy(sam_array).permute(2, 0, 1)
                                sam_images.append(sam_tensor)
                    
                    if sam_images:
                        sam_images_batch = torch.stack(sam_images).to(device)
                    else:
                        sam_images_batch = None
                else:
                    # pixel_valuesから生成を試みる（パッチ形式の場合は生成できない）
                    sam_images = []
                    for i in range(batch['pixel_values'].shape[0]):
                        # Qwen用の画像を取得
                        qwen_image = batch['pixel_values'][i]  # [patches, features]の可能性
                        
                        # RGB形式に変換してPIL Imageに
                        if qwen_image.dim() == 3:
                            # [C, H, W] -> [H, W, C]の順番で変換
                            qwen_np = qwen_image.permute(1, 2, 0).cpu().numpy()
                            # 正規化されている場合は0-255に戻す
                            if qwen_np.max() <= 1.0:
                                qwen_np = (qwen_np * 255).astype(np.uint8)
                            
                            # PIL Imageに変換
                            from PIL import Image
                            pil_image = Image.fromarray(qwen_np.astype(np.uint8))
                            
                            # SAM用にリサイズ
                            sam_image = pil_image.resize((config.sam_image_size, config.sam_image_size), Image.BICUBIC)
                            
                            # Tensorに変換
                            sam_array = np.array(sam_image).astype(np.float32) / 255.0
                            sam_tensor = torch.from_numpy(sam_array).permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
                            sam_images.append(sam_tensor)
                    
                    if sam_images:
                        sam_images_batch = torch.stack(sam_images).to(device)
                    else:
                        sam_images_batch = None
            else:
                sam_images_batch = None
            
            # 現在のalphaを計算
            alpha = compute_alpha_schedule(step, warmup_steps, max_alpha)
            
            # **修正後: SAM画像を含むモデルforward（段階的ゲーティング適用）**
            logits, mask_logits = model.forward_with_gating(
                input_ids=batch['input_ids'],
                pixel_values=batch['pixel_values'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
                mask_labels=batch['mask_labels'],
                image_grid_thw=batch.get('image_grid_thw', None),
                sam_images=sam_images_batch,  # SAM画像を渡す
                alpha=alpha
            )
            
            # 損失計算
            vocab_size = logits.size(-1)
            lm_loss = F.cross_entropy(logits.view(-1, vocab_size), batch['labels'].view(-1), ignore_index=-100)
            
            # セグメンテーション損失
            seg_loss = torch.tensor(0.0, device=device)
            seg_count = 0
            pred_masks_for_vis = []
            gt_masks_for_vis = []
            
            if mask_logits is not None:
                for i, pred_masks in enumerate(mask_logits):
                    if pred_masks is None: 
                        continue
                    
                    if isinstance(pred_masks, list):
                        pred_mask = pred_masks[0] if len(pred_masks) > 0 else None
                    else:
                        pred_mask = pred_masks
                    
                    if pred_mask is None:
                        continue
                    
                    gt_mask = batch['mask_labels'][i]
                    if gt_mask is None:
                        continue
                    
                    # マスクの形状調整
                    if pred_mask.dim() == 2:
                        pred_h, pred_w = pred_mask.shape
                        pred_mask_tensor = pred_mask
                    elif pred_mask.dim() == 3:
                        pred_h, pred_w = pred_mask.shape[-2:]
                        pred_mask_tensor = pred_mask.squeeze(0)
                    else:
                        pred_h, pred_w = pred_mask.shape[-2:]
                        pred_mask_tensor = pred_mask.squeeze(0).squeeze(0)
                    
                    if gt_mask.dim() == 2:
                        gt_h, gt_w = gt_mask.shape
                        gt_mask_for_resize = gt_mask
                    elif gt_mask.dim() == 3:
                        if gt_mask.shape[0] > 1:
                            gt_mask_for_resize = gt_mask[0]
                            gt_h, gt_w = gt_mask_for_resize.shape
                        else:
                            gt_mask_for_resize = gt_mask.squeeze(0)
                            gt_h, gt_w = gt_mask_for_resize.shape
                    else:
                        gt_h, gt_w = gt_mask.shape[-2:]
                        gt_mask_for_resize = gt_mask
                    
                    # サイズ調整
                    if (pred_h, pred_w) != (gt_h, gt_w):
                        if gt_mask_for_resize.dim() == 2:
                            gt_mask_4d = gt_mask_for_resize.unsqueeze(0).unsqueeze(0)
                        elif gt_mask_for_resize.dim() == 3:
                            gt_mask_4d = gt_mask_for_resize.unsqueeze(0)
                        else:
                            gt_mask_4d = gt_mask_for_resize
                        
                        gt_mask_4d = gt_mask_4d.float()
                        gt_mask_resized = F.interpolate(
                            gt_mask_4d,
                            size=(pred_h, pred_w),
                            mode='nearest'
                        )
                        gt_mask_resized = gt_mask_resized.squeeze(0).squeeze(0)
                    else:
                        gt_mask_resized = gt_mask_for_resize.float()
                    
                    # BCE + Dice損失
                    bce = F.binary_cross_entropy_with_logits(pred_mask_tensor, gt_mask_resized)
                    
                    pred_prob = torch.sigmoid(pred_mask_tensor)
                    intersection = (pred_prob * gt_mask_resized).sum()
                    dice = 2 * intersection / (pred_prob.sum() + gt_mask_resized.sum() + 1e-8)
                    dice_loss = 1 - dice
                    
                    seg_loss = seg_loss + (bce + dice_loss)
                    seg_count += 1
                    
                    # 可視化用
                    if len(pred_masks_for_vis) == 0:
                        pred_masks_for_vis.append(pred_mask_tensor)
                        gt_masks_for_vis.append(gt_mask_resized)
            
            if seg_count > 0:
                seg_loss = seg_loss / seg_count
            
            total_loss = lm_loss + seg_loss
            
            # 逆伝播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # 記録
            loss_data = {
                'step': step,
                'total_loss': total_loss.item(),
                'lm_loss': lm_loss.item(),
                'seg_loss': seg_loss.item(),
                'alpha': alpha
            }
            loss_history.append(loss_data)
            alpha_history.append(alpha)
            
            # 可視化
            if len(pred_masks_for_vis) > 0 and len(gt_masks_for_vis) > 0:
                losses_dict = {
                    'total': total_loss.item(),
                    'lm': lm_loss.item(),
                    'seg': seg_loss.item()
                }
                save_visualization(output_dir, step, batch, pred_masks_for_vis[0], 
                                 gt_masks_for_vis[0], losses_dict, tokenizer, alpha)
            
            # ログ出力
            step += 1
            pbar.update(1)
            pbar.set_postfix({
                'total': f"{total_loss.item():.4f}",
                'seg': f"{seg_loss.item():.4f}",
                'α': f"{alpha:.3f}"
            })
            
            if step % LOG_INTERVAL == 0:
                logger.info(f"Step {step}: total={total_loss.item():.4f}, "
                          f"lm={lm_loss.item():.4f}, seg={seg_loss.item():.4f}, α={alpha:.3f}")
    
    logger.info("Training finished.")
    
    # 結果の保存
    results = {
        'config': {
            'freeze_qwen_base': config.freeze_qwen_base,
            'freeze_sam_mask_decoder_base': config.freeze_sam_mask_decoder_base,
            'freeze_seg_token': config.freeze_seg_token,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'num_steps': num_steps,
            'dataset_type': dataset_type,
            'samples_per_epoch': samples_per_epoch,
            'warmup_steps': warmup_steps,
            'max_alpha': max_alpha
        },
        'loss_history': loss_history,
        'alpha_history': alpha_history,
        'final_losses': {
            'total_loss': loss_history[-1]['total_loss'],
            'lm_loss': loss_history[-1]['lm_loss'], 
            'seg_loss': loss_history[-1]['seg_loss']
        }
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}/results.json")
    
    # 損失曲線のプロット
    try:
        import matplotlib.pyplot as plt
        
        steps = [d['step'] for d in loss_history]
        total_losses = [d['total_loss'] for d in loss_history]
        lm_losses = [d['lm_loss'] for d in loss_history]
        seg_losses = [d['seg_loss'] for d in loss_history]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Total loss
        axes[0, 0].plot(steps, total_losses)
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].grid(True)
        
        # LM loss
        axes[0, 1].plot(steps, lm_losses)
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('LM Loss')
        axes[0, 1].set_title('Language Model Loss')
        axes[0, 1].grid(True)
        
        # Seg loss
        axes[1, 0].plot(steps, seg_losses)
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Seg Loss')
        axes[1, 0].set_title('Segmentation Loss')
        axes[1, 0].grid(True)
        
        # Alpha schedule
        axes[1, 1].plot(steps, alpha_history, color='green')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Alpha (LLM mix ratio)')
        axes[1, 1].set_title('Gating Schedule')
        axes[1, 1].grid(True)
        axes[1, 1].axhline(y=max_alpha, color='r', linestyle='--', label=f'Max α={max_alpha}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'loss_curves.png')
        logger.info(f"Loss curves saved to {output_dir}/loss_curves.png")
        
    except ImportError:
        logger.warning("matplotlib not available, skipping plot generation")
    
    # 最終評価
    initial_seg_loss = loss_history[0]['seg_loss']
    final_seg_loss = loss_history[-1]['seg_loss']
    
    if final_seg_loss < initial_seg_loss * EVAL_THRESHOLD:
        logger.info("✅ TEST PASSED: Segmentation loss decreased significantly with gated embedding")
        logger.info(f"   Initial seg_loss: {initial_seg_loss:.4f}")
        logger.info(f"   Final seg_loss: {final_seg_loss:.4f}")
        logger.info(f"   Improvement: {(1 - final_seg_loss/initial_seg_loss)*100:.1f}%")
        logger.info(f"   Final alpha: {alpha_history[-1]:.3f}")
    else:
        logger.warning("⚠️ TEST FAILED: Segmentation loss did not decrease significantly")
        logger.warning(f"   Initial seg_loss: {initial_seg_loss:.4f}")
        logger.warning(f"   Final seg_loss: {final_seg_loss:.4f}")
        logger.warning("   Gated embedding may need parameter tuning")


if __name__ == "__main__":
    main()