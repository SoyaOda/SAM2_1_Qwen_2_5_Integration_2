#!/usr/bin/env python3
"""
A-2テスト: MaskDecoderを学習可能にした統合モデルのセグメンテーション学習検証

目的:
  minimal_train.pyと同等の統合パイプラインで、SAM MaskDecoder/PromptEncoderをfreezeしない場合に
  seg_lossが適切に低下するかを検証する。

手順:
  - LISA_Modelをfreeze_sam=Falseで初期化（SAM部分を全て学習対象に含める）
  - sem_segデータセットから少数サンプルでミニバッチ学習を行い、損失推移を観察
"""
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
from transformers import AutoProcessor
import logging
import matplotlib.pyplot as plt
import numpy as np
import cv2

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
from src.data.dataset import HybridDataset
from src.data.collators import MultiModalDataCollator

# ロガー設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ハイパーパラメータ設定
# パラメータは共通設定から使用
num_steps = NUM_STEPS
batch_size = BATCH_SIZE
learning_rate = LEARNING_RATE
dataset_type = DATASET_TYPE
samples_per_epoch = NUM_FIXED_SAMPLES

def save_visualization(output_dir, step, batch, pred_masks, gt_masks, losses, tokenizer):
    """可視化の保存（test_a1と同様の詳細な可視化）"""
    if step % VIS_INTERVAL != 0:  # 共通設定の間隔を使用
        return
    
    # バッチから最初のサンプルを取得
    if 'original_images' in batch and batch['original_images'] is not None and len(batch['original_images']) > 0:
        # PIL画像をnumpy配列に変換
        from PIL import Image
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
            return
    else:
        # pixel_valuesから復元を試みる
        pixel_values = batch['pixel_values'][0].cpu()
        if pixel_values.dim() == 2:
            return  # パッチ形式はスキップ
        if pixel_values.dim() == 3:
            # 正規化を解除
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
            image = pixel_values * std + mean
            image = torch.clamp(image, 0, 1)
            image_np = image.permute(1, 2, 0).numpy()
            image_np = (image_np * 255).astype(np.uint8)
        else:
            return
    
    # マスクを取得（最初のサンプルのみ）
    if isinstance(pred_masks, list) and len(pred_masks) > 0:
        if isinstance(pred_masks[0], list) and len(pred_masks[0]) > 0:
            pred_mask = pred_masks[0][0]
        else:
            pred_mask = pred_masks[0]
    else:
        pred_mask = pred_masks
    
    if pred_mask.dim() > 2:
        pred_mask_np = torch.sigmoid(pred_mask[0, 0]).detach().cpu().numpy()
    else:
        pred_mask_np = torch.sigmoid(pred_mask).detach().cpu().numpy()
    
    if gt_masks.dim() > 2:
        gt_mask_np = gt_masks[0, 0].detach().cpu().numpy() if gt_masks.dim() == 4 else gt_masks[0].detach().cpu().numpy()
    else:
        gt_mask_np = gt_masks.detach().cpu().numpy()
    
    # 可視化（2行のレイアウト）
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
    ax3.set_title(f'Predicted Mask', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # 下段: オーバーレイ比較
    ax4 = plt.subplot(2, 3, 4)
    ax4.imshow(image_np)
    # マスクを画像サイズにリサイズ
    gt_mask_resized = cv2.resize(gt_mask_np, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_overlay = np.zeros_like(image_np)
    mask_overlay[:, :, 0] = gt_mask_resized * 255  # 赤でGTマスク
    ax4.imshow(mask_overlay, alpha=0.3)
    ax4.set_title('Image + GT Mask Overlay', fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    ax5 = plt.subplot(2, 3, 5)
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
    
    # Dice scoreとIoUを計算
    pred_binary = (pred_mask_resized > 0.5).astype(np.float32)
    gt_binary = (gt_mask_resized > 0.5).astype(np.float32)
    intersection = (pred_binary * gt_binary).sum()
    union = pred_binary.sum() + gt_binary.sum() - intersection
    iou = intersection / (union + 1e-6)
    dice = 2 * intersection / (pred_binary.sum() + gt_binary.sum() + 1e-6)
    
    # メトリクス情報を表示
    info_text = f"Step: {step}\n"
    info_text += f"Total Loss: {losses['total']:.4f}\n"
    info_text += f"LM Loss: {losses['lm']:.4f}\n"
    info_text += f"Seg Loss: {losses['seg']:.4f}\n"
    info_text += f"Dice Score: {dice:.4f}\n"
    info_text += f"IoU: {iou:.4f}\n"
    info_text += f"Mask Size: {gt_mask_np.shape}\n"
    info_text += f"Image Size: {image_np.shape[:2]}\n"
    
    ax6.text(0.05, 0.95, info_text, transform=ax6.transAxes, 
            fontsize=10, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            wrap=True, family='monospace')
    ax6.set_title('Metrics', fontsize=12, fontweight='bold')
    
    plt.suptitle(f'A-2 Test Visualization - Step {step} (SAM Unfrozen)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / 'visualizations' / f'step_{step:05d}.png'
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved visualization to {save_path} (Dice: {dice:.4f}, IoU: {iou:.4f})")

def main():
    # シード固定
    set_random_seed(SEED)
    
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 出力ディレクトリ作成（共通構造）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = create_standard_output_structure(f"test_outputs/test_a2_{timestamp}")
    logger.info(f"Output directory: {output_dir}")
    
    # テスト設定を保存
    save_test_config(output_dir, 'test_a2_unfreeze_sam', {
        'test_description': 'SAM unfrozen with LLM embedding replacement'
    })
    
    # 1. モデルとトークナイザの準備
    config = LISAConfig(
        qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        sam_model_name="./checkpoints/sam2.1_hiera_large.pt",
        device_map="cuda",
        torch_dtype="auto",
        use_flash_attention=False,
        # Freeze configuration - Test A2: Unfreeze SAM
        freeze_qwen_lora=False,         # Qwen LoRA有効 (trainable)
        freeze_seg_token=False,         # SEGトークン学習可能
        sam_lora_r = 8,
        freeze_image_adapter=False,     # アダプター学習可能
        freeze_text_prompt_projector=False,  # プロジェクター学習可能
        freeze_token_fpn=False,         # Token-FPN学習可能
        freeze_prompt_beta=True,        # A2ではBeta無効 (frozen)
        # Freeze settings
        freeze_sam_mask_decoder_base=True,
    )
    processor = None
    if config.use_dynamic_resolution:
        processor = AutoProcessor.from_pretrained(
            config.qwen_model_name,
            min_pixels=config.qwen_min_pixels,
            max_pixels=config.qwen_max_pixels
        )
    else:
        processor = AutoProcessor.from_pretrained(config.qwen_model_name)
    
    # ProcessorのトークナイザにもSEGトークン追加
    if config.seg_token not in processor.tokenizer.get_vocab():
        processor.tokenizer.add_special_tokens({"additional_special_tokens": [config.seg_token]})
    
    # モデル初期化
    # トークナイザーの準備
    tokenizer = prepare_tokenizer_for_lisa(
        model_name=config.qwen_model_name,
        seg_token=config.seg_token
    )
    
    model = LISA_Model(config)
    model.set_tokenizer(tokenizer)
    
    # パラメータ統計の詳細表示
    from src.utils.model_utils import display_parameter_statistics
    display_parameter_statistics(model, logger_name=__name__)
    
    # SAM側はfreeze=Falseなので、LoRAを追加しない（全パラメータを学習）
    # Qwen側にはLoRAを追加
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
    
    # 学習モードに設定
    model.train()
    
    # 学習対象のパラメータを確認
    trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
    logger.info(f"Total trainable parameters: {len(trainable_params)} tensors")
    
    # 最初の10個のパラメータ名を表示
    for name in trainable_params[:10]:
        logger.info(f" - {name}")
    
    # SAM関連のパラメータが学習可能になっているか確認
    sam_trainable = [n for n in trainable_params if 'sam' in n.lower()]
    logger.info(f"SAM trainable parameters: {len(sam_trainable)} tensors")
    
    # 2. データセット準備（固定サンプル版）
    base_dir = config.dataset_base_dir
    
    # ベースのHybridDatasetを作成
    base_dataset = HybridDataset(
        base_image_dir=base_dir,
        qwen_processor=processor,
        samples_per_epoch=NUM_FIXED_SAMPLES,  # 共通設定から
        dataset=DATASET_TYPE,  # 共通設定から
        sample_rate=SAMPLE_RATE,  # 共通設定から
        qwen_image_size=config.qwen_image_size,
        sam_image_size=config.sam_image_size,
    )
    
    # 固定サンプル版に変換（共通クラスとシード使用）
    dataset = FixedSampleDataset(base_dataset, num_samples=NUM_FIXED_SAMPLES, seed=SEED)
    collator = MultiModalDataCollator(
        tokenizer=tokenizer,
        max_length=config.model_max_length,
        config=config
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,  # 固定サンプルなのでシャッフル不要 
        collate_fn=collator, 
        num_workers=0,
        pin_memory=True
    )
    
    logger.info(f"Dataset loaded: {dataset_type}, total samples={len(dataset)}")
    
    # 3. オプティマイザ準備
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # 損失履歴の記録
    loss_history = []
    
    # 4. 学習ループ開始
    step = 0
    data_iter = iter(dataloader)
    
    with tqdm(total=num_steps, desc="Training") as pbar:
        while step < num_steps:
            # データをロード（データセットが終わったら最初から）
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            # バッチからデバイスへ転送
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            # モデルforward実行
            outputs = model(
                input_ids=batch['input_ids'],
                pixel_values=batch['pixel_values'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
                mask_labels=batch['mask_labels'],
                image_grid_thw=batch.get('image_grid_thw', None)
            )
            
            # 損失計算
            # Languageモデリング損失
            vocab_size = outputs.logits.size(-1)
            lm_loss = F.cross_entropy(outputs.logits.view(-1, vocab_size), batch['labels'].view(-1), ignore_index=-100)
            
            # セグメンテーション損失（BCE + Dice）
            seg_loss = torch.tensor(0.0, device=device)
            seg_count = 0
            pred_masks_for_vis = []  # 可視化用に予測マスクを保存
            gt_masks_for_vis = []    # 可視化用にGTマスクを保存
            
            if outputs.mask_logits is not None:
                for i, pred_masks in enumerate(outputs.mask_logits):
                    if pred_masks is None: 
                        continue
                    
                    # リストで来る場合もあるので統一
                    if isinstance(pred_masks, list):
                        pred_mask = pred_masks[0] if len(pred_masks) > 0 else None
                    else:
                        pred_mask = pred_masks
                    
                    if pred_mask is None:
                        continue
                    
                    gt_mask = batch['mask_labels'][i]
                    if gt_mask is None:
                        continue
                    
                    # minimal_train.pyのロジックを適用
                    # pred_maskの形状を取得
                    if pred_mask.dim() == 2:
                        pred_h, pred_w = pred_mask.shape
                        pred_mask_tensor = pred_mask
                    elif pred_mask.dim() == 3:
                        pred_h, pred_w = pred_mask.shape[-2:]
                        pred_mask_tensor = pred_mask.squeeze(0)  # (1, H, W) -> (H, W)
                    else:
                        # 4次元以上の場合
                        pred_h, pred_w = pred_mask.shape[-2:]
                        pred_mask_tensor = pred_mask.squeeze(0).squeeze(0)  # (B, C, H, W) -> (H, W)
                    
                    # gt_maskの形状を取得
                    if gt_mask.dim() == 2:
                        gt_h, gt_w = gt_mask.shape
                        gt_mask_for_resize = gt_mask
                    elif gt_mask.dim() == 3:
                        # 複数マスクの場合は最初のみ使用
                        if gt_mask.shape[0] > 1:
                            gt_mask_for_resize = gt_mask[0]
                            gt_h, gt_w = gt_mask_for_resize.shape
                        else:
                            gt_mask_for_resize = gt_mask.squeeze(0)
                            gt_h, gt_w = gt_mask_for_resize.shape
                    else:
                        gt_h, gt_w = gt_mask.shape[-2:]
                        gt_mask_for_resize = gt_mask
                    
                    # サイズが異なる場合はリサイズ
                    if (pred_h, pred_w) != (gt_h, gt_w):
                        # gt_maskを4次元形式に変換
                        if gt_mask_for_resize.dim() == 2:
                            gt_mask_4d = gt_mask_for_resize.unsqueeze(0).unsqueeze(0)
                        elif gt_mask_for_resize.dim() == 3:
                            gt_mask_4d = gt_mask_for_resize.unsqueeze(0)
                        else:
                            gt_mask_4d = gt_mask_for_resize
                        
                        # float型に変換してinterpolate
                        gt_mask_4d = gt_mask_4d.float()
                        gt_mask_resized = F.interpolate(
                            gt_mask_4d,
                            size=(pred_h, pred_w),
                            mode='nearest'
                        )
                        
                        # 元の次元に戻す（(1, 1, H, W) -> (H, W)）
                        gt_mask_resized = gt_mask_resized.squeeze(0).squeeze(0)
                    else:
                        gt_mask_resized = gt_mask_for_resize.float()
                    
                    # BCE
                    bce = F.binary_cross_entropy_with_logits(pred_mask_tensor, gt_mask_resized)
                    
                    # Dice
                    pred_prob = torch.sigmoid(pred_mask_tensor)
                    intersection = (pred_prob * gt_mask_resized).sum()
                    dice = 2 * intersection / (pred_prob.sum() + gt_mask_resized.sum() + 1e-8)
                    dice_loss = 1 - dice
                    
                    seg_loss = seg_loss + (bce + dice_loss)
                    seg_count += 1
                    
                    # 可視化用に保存（最初のサンプルのみ）
                    if len(pred_masks_for_vis) == 0:
                        pred_masks_for_vis.append(pred_mask_tensor)
                        gt_masks_for_vis.append(gt_mask_resized)
            
            if seg_count > 0:
                seg_loss = seg_loss / seg_count
            
            total_loss = lm_loss + seg_loss
            
            # 逆伝播とオプティマイザステップ
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # 損失の記録
            loss_data = {
                'step': step,
                'total_loss': total_loss.item(),
                'lm_loss': lm_loss.item(),
                'seg_loss': seg_loss.item()
            }
            loss_history.append(loss_data)
            
            # 可視化の保存
            if len(pred_masks_for_vis) > 0 and len(gt_masks_for_vis) > 0:
                losses_dict = {
                    'total': total_loss.item(),
                    'lm': lm_loss.item(),
                    'seg': seg_loss.item()
                }
                save_visualization(output_dir, step, batch, pred_masks_for_vis[0], gt_masks_for_vis[0], losses_dict, tokenizer)
            
            # ログ出力
            step += 1
            pbar.update(1)
            
            if step % LOG_INTERVAL == 0:
                logger.info(f"Step {step}: total_loss={total_loss.item():.4f}, lm_loss={lm_loss.item():.4f}, seg_loss={seg_loss.item():.4f}")
    
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
            'samples_per_epoch': samples_per_epoch
        },
        'loss_history': loss_history,
        'final_losses': {
            'total_loss': loss_history[-1]['total_loss'],
            'lm_loss': loss_history[-1]['lm_loss'], 
            'seg_loss': loss_history[-1]['seg_loss']
        }
    }
    
    # JSONファイルに保存
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}/results.json")
    
    # 損失の推移をプロット（matplotlib利用可能な場合）
    try:
        import matplotlib.pyplot as plt
        
        steps = [d['step'] for d in loss_history]
        total_losses = [d['total_loss'] for d in loss_history]
        lm_losses = [d['lm_loss'] for d in loss_history]
        seg_losses = [d['seg_loss'] for d in loss_history]
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(steps, total_losses)
        plt.xlabel('Step')
        plt.ylabel('Total Loss')
        plt.title('Total Loss')
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(steps, lm_losses)
        plt.xlabel('Step')
        plt.ylabel('LM Loss')
        plt.title('Language Model Loss')
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.plot(steps, seg_losses)
        plt.xlabel('Step')
        plt.ylabel('Seg Loss')
        plt.title('Segmentation Loss')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'loss_curves.png')
        logger.info(f"Loss curves saved to {output_dir}/loss_curves.png")
        
    except ImportError:
        logger.warning("matplotlib not available, skipping plot generation")
    
    # 最終的な損失の評価
    initial_seg_loss = loss_history[0]['seg_loss']
    final_seg_loss = loss_history[-1]['seg_loss']
    
    if final_seg_loss < initial_seg_loss * EVAL_THRESHOLD:  # 指定閾値以上の改善
        logger.info("✅ TEST PASSED: Segmentation loss decreased significantly")
        logger.info(f"   Initial seg_loss: {initial_seg_loss:.4f}")
        logger.info(f"   Final seg_loss: {final_seg_loss:.4f}")
        logger.info(f"   Improvement: {(1 - final_seg_loss/initial_seg_loss)*100:.1f}%")
    else:
        logger.warning("⚠️ TEST FAILED: Segmentation loss did not decrease significantly")
        logger.warning(f"   Initial seg_loss: {initial_seg_loss:.4f}")
        logger.warning(f"   Final seg_loss: {final_seg_loss:.4f}")
        logger.warning("   This suggests the issue is not just with freezing SAM")

if __name__ == "__main__":
    main()