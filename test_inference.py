#!/usr/bin/env python3
"""
学習済みLISA改モデルの推論テスト
Webからサンプル画像をダウンロードして、セグメンテーション推論を実行
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from transformers import AutoTokenizer, AutoProcessor
from typing import Dict, List, Tuple, Optional
import logging

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.lisa_model import LISA_Model
from src.config import LISAConfig

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class InferenceRunner:
    """推論実行クラス"""
    
    def __init__(self, checkpoint_dir: str):
        """
        Args:
            checkpoint_dir: チェックポイントディレクトリ
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # モデルとプロセッサをロード
        self.model, self.processor, self.tokenizer = self.load_model()
        
    def load_model(self):
        """学習済みモデルをロード"""
        logger.info(f"Loading model from {self.checkpoint_dir}")
        
        # 設定をロード
        config_path = self.checkpoint_dir.parent.parent / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # LISAConfigを作成
        config = LISAConfig()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # プロセッサとトークナイザーを初期化
        processor = AutoProcessor.from_pretrained(config.qwen_model_name)
        tokenizer = AutoTokenizer.from_pretrained(config.qwen_model_name)
        
        # SEGトークンを追加
        if config.seg_token not in tokenizer.get_vocab():
            tokenizer.add_tokens([config.seg_token], special_tokens=True)
            logger.info(f"Added {config.seg_token} token to tokenizer")
        
        # モデルを初期化
        model = LISA_Model(
            config=config,
            qwen_model=None,  # 自動でロードされる
            sam_predictor=None,  # 自動でロードされる
            tokenizer=tokenizer
        )
        
        # チェックポイントをロード（個別のファイルから）
        logger.info(f"Loading checkpoints from {self.checkpoint_dir}")
        
        # 各コンポーネントのチェックポイントをロード
        checkpoint_files = {
            'token_fpn': self.checkpoint_dir / "token_fpn.pt",
            'image_adapter': self.checkpoint_dir / "image_adapter.pt",
            'text_prompt_proj': self.checkpoint_dir / "text_prompt_proj.pt",
            'config': self.checkpoint_dir / "config.pt",
        }
        
        # token_fpnをロード
        if checkpoint_files['token_fpn'].exists():
            token_fpn_state = torch.load(checkpoint_files['token_fpn'], map_location=self.device, weights_only=True)
            model.token_fpn.load_state_dict(token_fpn_state)
            logger.info("Loaded token_fpn checkpoint")
        
        # image_adapterをロード
        if checkpoint_files['image_adapter'].exists():
            image_adapter_state = torch.load(checkpoint_files['image_adapter'], map_location=self.device, weights_only=True)
            model.image_adapter.load_state_dict(image_adapter_state)
            logger.info("Loaded image_adapter checkpoint")
        
        # text_prompt_projをロード
        if checkpoint_files['text_prompt_proj'].exists():
            text_prompt_proj_state = torch.load(checkpoint_files['text_prompt_proj'], map_location=self.device, weights_only=True)
            model.text_prompt_proj.load_state_dict(text_prompt_proj_state)
            logger.info("Loaded text_prompt_proj checkpoint")
        
        # configのパラメータをロード（prompt_betaなど）
        if checkpoint_files['config'].exists():
            # config.ptはLISAConfigオブジェクトを含むため、weights_only=Falseが必要
            config_state = torch.load(checkpoint_files['config'], map_location=self.device, weights_only=False)
            if isinstance(config_state, dict) and 'prompt_beta' in config_state:
                model.prompt_beta.data = config_state['prompt_beta']
                logger.info(f"Loaded prompt_beta: {config_state['prompt_beta'].item()}")
            elif hasattr(config_state, 'freeze_prompt_beta'):
                # config_stateがLISAConfigオブジェクトの場合
                logger.info(f"Config object loaded, prompt_beta freeze state: {config_state.freeze_prompt_beta}")
        
        # Qwen LoRAパラメータをロード（もし存在すれば）
        qwen_lora_path = self.checkpoint_dir / "qwen" / "adapter_model.bin"
        if qwen_lora_path.exists():
            qwen_lora_state = torch.load(qwen_lora_path, map_location=self.device)
            # LoRAパラメータをロード
            for name, param in model.qwen.named_parameters():
                if "lora" in name.lower():
                    key = name.replace("base_model.model.", "")
                    if key in qwen_lora_state:
                        param.data = qwen_lora_state[key]
            logger.info("Loaded Qwen LoRA parameters")
        
        model.to(self.device)
        model.eval()
        
        logger.info("Model loaded successfully")
        return model, processor, tokenizer
    
    def download_sample_images(self) -> List[Dict]:
        """サンプル画像をWebからダウンロード"""
        logger.info("Downloading sample images...")
        
        # 様々なソースからのサンプル画像URL
        sample_images = [
            {
                "url": "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/truck.jpg",
                "name": "truck",
                "prompt": "Please segment the <SEG> truck in the image."
            },
            {
                "url": "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/groceries.jpg",
                "name": "groceries",
                "prompt": "Please segment the <SEG> fruits on the table."
            },
            {
                "url": "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/dog.jpg",
                "name": "dog",
                "prompt": "Please segment the <SEG> dog in the image."
            },
            {
                "url": "https://ultralytics.com/images/bus.jpg",
                "name": "bus",
                "prompt": "Please segment the <SEG> bus in the image."
            },
            {
                "url": "https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image01.png",
                "name": "person_bike",
                "prompt": "Please segment the <SEG> person on the bicycle."
            }
        ]
        
        downloaded_images = []
        for img_info in sample_images:
            try:
                response = requests.get(img_info["url"], timeout=10)
                response.raise_for_status()
                
                # PILで画像を開く
                image = Image.open(BytesIO(response.content)).convert("RGB")
                
                downloaded_images.append({
                    "image": image,
                    "name": img_info["name"],
                    "prompt": img_info["prompt"],
                    "url": img_info["url"]
                })
                
                logger.info(f"Downloaded: {img_info['name']} from {img_info['url']}")
                
            except Exception as e:
                logger.warning(f"Failed to download {img_info['name']}: {e}")
                continue
        
        logger.info(f"Successfully downloaded {len(downloaded_images)} images")
        return downloaded_images
    
    @torch.no_grad()
    def run_inference(self, image: Image.Image, prompt: str) -> Dict:
        """単一画像に対して推論を実行"""
        
        # 画像を前処理
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # image_grid_thwを取得（Qwen2.5-VLに必要）
        if hasattr(inputs, 'image_grid_thw'):
            image_grid_thw = inputs.image_grid_thw
        else:
            # image_grid_thwがない場合は計算
            if inputs.pixel_values is not None:
                B, C, H, W = inputs.pixel_values.shape
                # パッチサイズ14で計算
                patch_h = H // 14
                patch_w = W // 14
                image_grid_thw = torch.tensor([[1, patch_h, patch_w]], dtype=torch.long).to(self.device)
            else:
                image_grid_thw = None
        
        # トークナイズ
        text_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # SAM用の高解像度画像を準備
        sam_image = np.array(image.resize((1024, 1024)))
        sam_image_tensor = torch.from_numpy(sam_image).permute(2, 0, 1).float() / 255.0
        sam_image_tensor = sam_image_tensor.unsqueeze(0).to(self.device)
        
        # 推論実行
        outputs = self.model(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
            pixel_values=inputs.pixel_values if hasattr(inputs, 'pixel_values') else None,
            image_grid_thw=image_grid_thw,  # Qwen2.5-VLに必要
            sam_images=sam_image_tensor,  # SAM用の高解像度画像
            labels=None,  # 推論時はラベルなし
            mask_labels=None  # 推論時はマスクラベルなし
        )
        
        # マスクを取得
        if 'pred_masks' in outputs:
            pred_mask = outputs['pred_masks'][0].cpu().numpy()
        else:
            logger.warning("No segmentation mask in output")
            pred_mask = np.zeros((image.height, image.width))
        
        return {
            "mask": pred_mask,
            "logits": outputs.get('logits', None)
        }
    
    def visualize_results(self, image: Image.Image, mask: np.ndarray, name: str, prompt: str, output_dir: Path):
        """結果を可視化して保存"""
        
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
        fig.suptitle(f"Prompt: {prompt}", fontsize=12)
        
        # 保存
        output_path = output_dir / f"{name}_result.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization to {output_path}")
        
    def run(self):
        """メイン実行"""
        
        # 出力ディレクトリを作成
        output_dir = Path("outputs/inference_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # サンプル画像をダウンロード
        sample_images = self.download_sample_images()
        
        if not sample_images:
            logger.error("No images downloaded. Exiting.")
            return
        
        # 各画像に対して推論を実行
        results = []
        for img_data in sample_images:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing: {img_data['name']}")
            logger.info(f"Prompt: {img_data['prompt']}")
            
            try:
                # 推論実行
                output = self.run_inference(img_data['image'], img_data['prompt'])
                
                # 結果を可視化
                self.visualize_results(
                    img_data['image'],
                    output['mask'],
                    img_data['name'],
                    img_data['prompt'],
                    output_dir
                )
                
                # 統計情報を記録
                mask = output['mask']
                result_info = {
                    "name": img_data['name'],
                    "prompt": img_data['prompt'],
                    "mask_shape": mask.shape,
                    "mask_min": float(mask.min()),
                    "mask_max": float(mask.max()),
                    "mask_mean": float(mask.mean()),
                    "positive_pixels": int((mask > 0.5).sum()),
                    "total_pixels": int(mask.size)
                }
                results.append(result_info)
                
                logger.info(f"Mask stats - Min: {result_info['mask_min']:.3f}, "
                          f"Max: {result_info['mask_max']:.3f}, "
                          f"Mean: {result_info['mask_mean']:.3f}")
                logger.info(f"Positive pixels: {result_info['positive_pixels']}/{result_info['total_pixels']} "
                          f"({100*result_info['positive_pixels']/result_info['total_pixels']:.1f}%)")
                
            except Exception as e:
                logger.error(f"Failed to process {img_data['name']}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 結果をJSONで保存
        results_path = output_dir / "inference_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Inference completed. Results saved to {output_dir}")
        logger.info(f"Processed {len(results)}/{len(sample_images)} images successfully")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LISA改モデルの推論テスト")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="outputs/minimal_train_20250812_190145/checkpoints/best",
        help="チェックポイントディレクトリのパス"
    )
    
    args = parser.parse_args()
    
    # 推論を実行
    runner = InferenceRunner(args.checkpoint_dir)
    runner.run()


if __name__ == "__main__":
    main()