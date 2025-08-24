#!/usr/bin/env python3
"""
学習済みLISA改モデルの推論テスト（v2）
Qwen2.5-VLの公式実装に基づいた正しい前処理を実装
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from transformers import AutoProcessor, StoppingCriteria, StoppingCriteriaList
from typing import Dict, List, Tuple, Optional
import logging
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    # Fallback implementation of process_vision_info
    def process_vision_info(messages: List[Dict]) -> Tuple[List, List]:
        """Fallback implementation for process_vision_info"""
        images = []
        videos = []
        for message in messages:
            if 'content' in message:
                for content in message['content']:
                    if content.get('type') == 'image':
                        if 'image' in content:
                            img = content['image']
                            if isinstance(img, Image.Image):
                                images.append(img)
                    elif content.get('type') == 'video':
                        if 'video' in content:
                            videos.append(content['video'])
        return images, videos

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.lisa_model import LISA_Model
from src.config import LISAConfig

# ロギング設定
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StopOnSEG(StoppingCriteria):
    """<SEG>トークンで生成を停止するためのクラス"""
    def __init__(self, seg_id: int):
        self.seg_id = seg_id
    
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> bool:
        # 最後のトークンが<SEG>かチェック
        return input_ids[0, -1].item() == self.seg_id


class InferenceRunnerV2:
    """推論実行クラス（改良版）"""
    
    def __init__(self, checkpoint_dir: str):
        """
        Args:
            checkpoint_dir: チェックポイントディレクトリ
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # モデルとプロセッサをロード
        self.model, self.processor, self.config = self.load_model()
        
        # <SEG>トークンのIDを取得
        self.seg_token = self.config.seg_token
        self.seg_id = self.processor.tokenizer.convert_tokens_to_ids(self.seg_token)
        logger.info(f"SEG token ID: {self.seg_id}")
        
    def load_model(self):
        """学習済みモデルをロード（新しいcheckpoint_ioを使用）"""
        logger.info(f"Loading model from {self.checkpoint_dir}")
        
        # 新しいcheckpoint_ioを使用してロード
        try:
            from src.utils.checkpoint_io import load_lisa_checkpoint
            
            # チェックポイントをロード
            result = load_lisa_checkpoint(
                checkpoint_dir=str(self.checkpoint_dir),
                device=str(self.device),
                strict=False  # 互換性のため厳密チェックを無効化
            )
            
            model = result['model']
            processor = result['processor']
            config = result['config']
            
            # SEGトークンがprocessorに含まれているか確認
            if config.seg_token not in processor.tokenizer.get_vocab():
                processor.tokenizer.add_tokens([config.seg_token], special_tokens=True)
                logger.info(f"Added {config.seg_token} token to tokenizer")
            
            # SEGトークンIDを設定
            model.set_tokenizer(processor.tokenizer, seg_token=config.seg_token)
            logger.info(f"SEG token ID: {model.seg_token_id}")
            
            model.eval()
            logger.info("Model loaded successfully using checkpoint_io")
            
            # 画像トークンIDを確認
            logger.info(f"Image token ID: {model.qwen.config.image_token_id if hasattr(model.qwen.config, 'image_token_id') else 'Not found'}")
            
            return model, processor, config
            
        except Exception as e:
            logger.warning(f"checkpoint_io loading failed: {e}, trying LISA model's load_pretrained")
            
            # フォールバック: LISA_Modelのload_pretrainedメソッドを試す
            try:
                from src.models.lisa_model import LISA_Model
                model = LISA_Model.load_pretrained(
                    str(self.checkpoint_dir),
                    device=str(self.device)
                )
                
                # プロセッサをチェックポイントから読み込み（保存されている場合）
                processor_dir = self.checkpoint_dir / "processor"
                if processor_dir.exists():
                    from transformers import AutoProcessor
                    processor = AutoProcessor.from_pretrained(str(processor_dir))
                    logger.info("Loaded processor from checkpoint")
                else:
                    # プロセッサを初期化
                    processor = AutoProcessor.from_pretrained(
                        model.config.qwen_model_name,
                        min_pixels=224*224,
                        max_pixels=1024*1024,
                        max_length=8192
                    )
                    
                    # SEGトークンを追加
                    if model.config.seg_token not in processor.tokenizer.get_vocab():
                        processor.tokenizer.add_tokens([model.config.seg_token], special_tokens=True)
                        logger.info(f"Added {model.config.seg_token} token to tokenizer")
                
                # SEGトークンIDを設定
                model.set_tokenizer(processor.tokenizer, seg_token=model.config.seg_token)
                logger.info(f"SEG token ID: {model.seg_token_id}")
                
                model.eval()
                logger.info("Model loaded successfully using LISA load_pretrained")
                
                return model, processor, model.config
                
            except Exception as e2:
                logger.error(f"All loading methods failed: {e2}")
                
                # 最終フォールバック: 手動ロード
                logger.info("Attempting manual checkpoint loading...")
                
                # 設定をロード
                config_path = self.checkpoint_dir / "config.pt"
                if not config_path.exists():
                    config_path = self.checkpoint_dir.parent.parent / "config.json"
                    if not config_path.exists():
                        raise FileNotFoundError(f"Config file not found in {self.checkpoint_dir}")
                
                if config_path.suffix == '.pt':
                    import sys
                    sys.path.append('.')
                    from src.config import LISAConfig
                    torch.serialization.add_safe_globals([LISAConfig])
                    config = torch.load(config_path, weights_only=False)
                else:
                    with open(config_path, 'r') as f:
                        config_dict = json.load(f)
                    from src.config import LISAConfig
                    config = LISAConfig()
                    for key, value in config_dict.items():
                        if hasattr(config, key):
                            setattr(config, key, value)
                
                # プロセッサを初期化
                processor = AutoProcessor.from_pretrained(
                    config.qwen_model_name,
                    min_pixels=224*224,
                    max_pixels=1024*1024,
                    max_length=8192
                )
                
                # SEGトークンを追加
                if config.seg_token not in processor.tokenizer.get_vocab():
                    processor.tokenizer.add_tokens([config.seg_token], special_tokens=True)
                    logger.info(f"Added {config.seg_token} token to tokenizer")
                
                # モデルを初期化
                from src.models.lisa_model import LISA_Model
                model = LISA_Model(
                    config=config,
                    qwen_model=None,
                    sam_predictor=None,
                    tokenizer=processor.tokenizer
                )
                
                # チェックポイントをロード
                self._load_checkpoints(model)
                
                # SEGトークンIDを設定
                model.set_tokenizer(processor.tokenizer, seg_token=config.seg_token)
                logger.info(f"SEG token ID: {model.seg_token_id}")
                
                model.to(self.device)
                model.eval()
                
                logger.info("Model loaded successfully using manual method")
                
                return model, processor, config
    
    def _load_checkpoints(self, model):
        """各コンポーネントのチェックポイントをロード"""
        logger.info(f"Loading checkpoints from {self.checkpoint_dir}")
        
        checkpoint_files = {
            'token_fpn': self.checkpoint_dir / "token_fpn.pt",
            'image_adapter': self.checkpoint_dir / "image_adapter.pt",
            'text_prompt_proj': self.checkpoint_dir / "text_prompt_proj.pt",
            'config': self.checkpoint_dir / "config.pt",
            'prompt_beta': self.checkpoint_dir / "prompt_beta.pt",
            'image_fusion_beta': self.checkpoint_dir / "image_fusion_beta.pt",  # Sigma-Add fusion
            'image_fusion_cross_attention': self.checkpoint_dir / "image_fusion_cross_attention.pt",  # Cross-Attention fusion
            'seg_token_embedding': self.checkpoint_dir / "seg_token_embedding.pt",
            'sam_lora': self.checkpoint_dir / "sam_lora.pt",
        }
        
        # Token-FPN
        if checkpoint_files['token_fpn'].exists():
            token_fpn_state = torch.load(checkpoint_files['token_fpn'], map_location=self.device, weights_only=True)
            model.token_fpn.load_state_dict(token_fpn_state)
            logger.info("Loaded token_fpn checkpoint")
        
        # Image Adapter
        if checkpoint_files['image_adapter'].exists():
            image_adapter_state = torch.load(checkpoint_files['image_adapter'], map_location=self.device, weights_only=True)
            model.image_adapter.load_state_dict(image_adapter_state)
            logger.info("Loaded image_adapter checkpoint")
        
        # Text Prompt Projector
        if checkpoint_files['text_prompt_proj'].exists():
            text_prompt_proj_state = torch.load(checkpoint_files['text_prompt_proj'], map_location=self.device, weights_only=True)
            model.text_prompt_proj.load_state_dict(text_prompt_proj_state)
            logger.info("Loaded text_prompt_proj checkpoint")
        
        # Prompt Beta
        if checkpoint_files['prompt_beta'].exists():
            prompt_beta_state = torch.load(checkpoint_files['prompt_beta'], map_location=self.device, weights_only=False)
            if isinstance(prompt_beta_state, dict) and 'prompt_beta' in prompt_beta_state:
                model.prompt_beta.data = prompt_beta_state['prompt_beta']
                logger.info(f"Loaded prompt_beta: {prompt_beta_state['prompt_beta'].item()}")
        elif checkpoint_files['config'].exists():
            # Fallback to old format
            config_state = torch.load(checkpoint_files['config'], map_location=self.device, weights_only=False)
            if isinstance(config_state, dict) and 'prompt_beta' in config_state:
                model.prompt_beta.data = config_state['prompt_beta']
                logger.info(f"Loaded prompt_beta from config: {config_state['prompt_beta'].item()}")
        
        # Image Fusion Beta (for Sigma-Add fusion)
        if checkpoint_files['image_fusion_beta'].exists():
            fusion_beta_state = torch.load(checkpoint_files['image_fusion_beta'], map_location=self.device, weights_only=False)
            if hasattr(model, 'image_fusion_beta') and model.image_fusion_beta is not None:
                if isinstance(fusion_beta_state, dict) and 'image_fusion_beta' in fusion_beta_state:
                    model.image_fusion_beta.data = fusion_beta_state['image_fusion_beta']
                    logger.info(f"Loaded image_fusion_beta: {torch.sigmoid(fusion_beta_state['image_fusion_beta']).item():.4f}")
        
        # Cross-Attention Fusion Module
        if checkpoint_files['image_fusion_cross_attention'].exists():
            if hasattr(model, 'image_fusion') and model.fusion_type == 'cross_attention':
                fusion_state = torch.load(checkpoint_files['image_fusion_cross_attention'], map_location=self.device, weights_only=True)
                model.image_fusion.load_state_dict(fusion_state)
                logger.info("Loaded Cross-Attention fusion module")
        
        # SEG Token Embedding
        if checkpoint_files['seg_token_embedding'].exists():
            seg_embedding_state = torch.load(checkpoint_files['seg_token_embedding'], map_location=self.device, weights_only=False)
            if isinstance(seg_embedding_state, dict) and 'seg_token_embedding' in seg_embedding_state:
                model.seg_token_embedding = nn.Parameter(seg_embedding_state['seg_token_embedding'])
                logger.info("Loaded SEG token embedding")
        
        # SAM LoRA
        if checkpoint_files['sam_lora'].exists():
            sam_lora_state = torch.load(checkpoint_files['sam_lora'], map_location=self.device, weights_only=True)
            # Load SAM LoRA weights
            for name, param in model.sam_mask_decoder.named_parameters():
                if name in sam_lora_state:
                    param.data = sam_lora_state[name]
            logger.info(f"Loaded SAM LoRA parameters: {len(sam_lora_state)} modules")
        
        # Qwen LoRA
        qwen_checkpoint_dir = self.checkpoint_dir / "qwen"
        if qwen_checkpoint_dir.exists():
            # Load Qwen LoRA using PEFT
            from peft import PeftModel
            try:
                model.qwen = PeftModel.from_pretrained(model.qwen, str(qwen_checkpoint_dir))
                logger.info("Loaded Qwen LoRA checkpoint")
            except Exception as e:
                logger.warning(f"Failed to load Qwen LoRA: {e}")
                # Try alternative loading method
                adapter_path = qwen_checkpoint_dir / "adapter_model.safetensors"
                if adapter_path.exists():
                    from safetensors.torch import load_file
                    adapter_weights = load_file(str(adapter_path))
                    # Apply weights to model
                    for name, param in model.qwen.named_parameters():
                        if any(key in name for key in adapter_weights.keys()):
                            matching_key = [k for k in adapter_weights.keys() if k in name][0]
                            param.data = adapter_weights[matching_key]
                    logger.info("Loaded Qwen LoRA checkpoint (alternative method)")
    

    
    @torch.no_grad()
    def run_inference(self, image: Image.Image, prompt: str) -> Dict:
        """単一画像に対して推論を実行（公式実装準拠）"""
        
        # <SEG>トークンをプロンプトに追加（推論時に必要）
        if self.seg_token not in prompt:
            prompt = prompt + f" {self.seg_token}"
        
        # Qwen2.5-VL用のメッセージフォーマットを作成
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }]
        
        # 1) プロセッサのapply_chat_templateを使用（画像プレースホルダが自動挿入される）
        text_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        logger.debug(f"Text with placeholders: {text_prompt[:200]}...")  # 最初の200文字を表示
        
        # 2) 画像の実体を処理（グローバル関数を使用）
        image_inputs, video_inputs = process_vision_info(messages)
        
        # 3) Processorで統合処理（input_ids、pixel_values、image_grid_thwが揃う）
        # videosが空の場合は渡さない
        if video_inputs:
            inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                max_length=8192  # 画像トークンのために十分な長さを確保
            ).to(self.device)
        else:
            inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
                max_length=8192  # 画像トークンのために十分な長さを確保
            ).to(self.device)
        
        # デバッグ: トークン数と特徴数の確認
        if hasattr(self.model.qwen.config, 'image_token_id'):
            image_token_id = self.model.qwen.config.image_token_id
            n_image_tokens = (inputs.input_ids == image_token_id).sum().item()
            if hasattr(inputs, 'image_grid_thw') and inputs.image_grid_thw is not None:
                n_image_features = int(inputs.image_grid_thw[:, 1:].prod(dim=1).sum())
            else:
                n_image_features = 0
            logger.info(f"Image tokens: {n_image_tokens}, Image features: {n_image_features}")
            
            # Qwen2.5-VLでは4:1の比率が正常（merge ratio）
            if n_image_tokens * 4 == n_image_features:
                logger.info("Token-feature ratio is 1:4 (normal for Qwen2.5-VL with merge ratio)")
            elif n_image_tokens != n_image_features and n_image_features > 0:
                logger.warning(f"Token-feature mismatch! Expected 1:4 ratio but got 1:{n_image_features/n_image_tokens:.1f}")
        
        # <SEG>トークンの確認
        n_seg_tokens = (inputs.input_ids == self.seg_id).sum().item()
        logger.info(f"SEG tokens in input: {n_seg_tokens}")
        if n_seg_tokens == 0:
            logger.warning("No SEG token found in input_ids! Check tokenization.")
        
        # SAM用の高解像度画像を準備（SAM2Transformsを使用して学習時と完全統一）
        from sam2.utils.transforms import SAM2Transforms
        
        # SAM2公式Transformsを使用（学習時と同じ）
        sam_transforms = SAM2Transforms(
            resolution=1024,
            mask_threshold=0.0,
            max_hole_area=0.0,
            max_sprinkle_area=0.0
        )
        
        # 元画像サイズを保存（後処理で必要）
        orig_hw = (image.height, image.width)
        logger.debug(f"Original image size (H, W): {orig_hw}")
        
        # SAM2Transformsで前処理（正方形リサイズ + ImageNet正規化）
        sam_image_tensor = sam_transforms(image)  # 3×1024×1024 (float, normalized)
        sam_image_tensor = sam_image_tensor.unsqueeze(0).to(self.device)
        
        # 4) モデルのforward（推論モード）
        try:
            outputs = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pixel_values=inputs.pixel_values if hasattr(inputs, 'pixel_values') else None,
                image_grid_thw=inputs.image_grid_thw if hasattr(inputs, 'image_grid_thw') else None,
                sam_images=sam_image_tensor,
                labels=None,
                mask_labels=None
            )
            
            # マスクを取得（outputsがLISAModelOutputの場合）
            logger.debug(f"Output type: {type(outputs)}")
            logger.debug(f"Output attributes: {dir(outputs) if hasattr(outputs, '__dict__') else 'N/A'}")
            
            if hasattr(outputs, 'pred_masks') and outputs.pred_masks is not None:
                logger.info(f"Found pred_masks with shape: {outputs.pred_masks.shape}")
                pred_mask = outputs.pred_masks[0].cpu().numpy()
            elif isinstance(outputs, dict) and 'pred_masks' in outputs:
                logger.info(f"Found pred_masks in dict with shape: {outputs['pred_masks'].shape}")
                pred_mask = outputs['pred_masks'][0].cpu().numpy()
            else:
                # mask_logitsが存在するか確認
                if hasattr(outputs, 'mask_logits') and outputs.mask_logits is not None:
                    logger.info(f"Found mask_logits: {outputs.mask_logits}")
                    if len(outputs.mask_logits) > 0 and outputs.mask_logits[0] is not None:
                        # 学習スクリプトと同じ処理
                        pred_mask = outputs.mask_logits[0]
                        if isinstance(pred_mask, list):
                            pred_mask = pred_mask[0]
                        
                        # pred_maskの形状を4次元に整形（postprocess_masksの要件）
                        if pred_mask.dim() == 2:
                            pred_mask = pred_mask.unsqueeze(0).unsqueeze(0)  # [H, W] -> [1, 1, H, W]
                        elif pred_mask.dim() == 3:
                            pred_mask = pred_mask.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
                        
                        # SAM2公式のpostprocess_masksで元画像サイズに復元
                        pred_mask_processed = sam_transforms.postprocess_masks(
                            pred_mask.float(),
                            orig_hw
                        )
                        
                        # シグモイドで確率に変換してnumpyに
                        pred_mask = torch.sigmoid(pred_mask_processed).squeeze().detach().cpu().numpy()
                    else:
                        logger.warning("mask_logits is empty")
                        pred_mask = np.zeros((image.height, image.width))
                else:
                    logger.warning("No segmentation mask in output")
                    pred_mask = np.zeros((image.height, image.width))
            
            # logitsも取得
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, dict):
                logits = outputs.get('logits', None)
            else:
                logits = None
            
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            # エラー時はダミーマスクを返す
            pred_mask = np.zeros((image.height, image.width))
            logits = None
        
        return {
            "mask": pred_mask,
            "logits": logits,
            "inputs": inputs,  # デバッグ用
            "orig_hw": orig_hw  # 後処理で使用した元画像サイズ
        }
    
    def run_inference_with_generation(self, image: Image.Image, prompt: str) -> Dict:
        """生成ベースの推論（<SEG>トークンを生成させる）"""
        
        # メッセージフォーマット（<SEG>の生成を促す）
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"{prompt} When ready to segment, output {self.seg_token}."}
            ]
        }]
        
        # 前処理
        text_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        # videosが空の場合は渡さない
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
        
        # <SEG>で停止する条件を設定
        stopping_criteria = StoppingCriteriaList([StopOnSEG(self.seg_id)])
        
        # 生成（<SEG>まで）
        with torch.no_grad():
            outputs = self.model.qwen.generate(
                **inputs,
                max_new_tokens=128,
                stopping_criteria=stopping_criteria,
                output_hidden_states=True,
                return_dict_in_generate=True
            )
        
        # 生成されたテキストを確認
        generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        generated_text = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=False)
        logger.info(f"Generated text: {generated_text}")
        
        # <SEG>トークンの位置を確認
        seg_positions = (generated_ids == self.seg_id).nonzero(as_tuple=True)[0]
        
        if len(seg_positions) > 0:
            logger.info(f"Found {self.seg_token} at position {seg_positions[0].item()}")
            
            # TODO: ここで隠れ状態を取得してSAMに渡す処理を実装
            # last_hidden = outputs.decoder_hidden_states[-1][0, -1]
            # ...
            
        return {
            "generated_text": generated_text,
            "seg_found": len(seg_positions) > 0
        }
    
    def download_sample_images(self) -> List[Dict]:
        """サンプル画像をWebからダウンロード"""
        logger.info("Downloading sample images...")
        
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
        
        downloaded_images = []
        for img_info in sample_images:
            try:
                response = requests.get(img_info["url"], timeout=10)
                response.raise_for_status()
                
                image = Image.open(BytesIO(response.content)).convert("RGB")
                
                downloaded_images.append({
                    "image": image,
                    "name": img_info["name"],
                    "prompt": img_info["prompt"],
                    "url": img_info["url"]
                })
                
                logger.info(f"Downloaded: {img_info['name']}")
                
            except Exception as e:
                logger.warning(f"Failed to download {img_info['name']}: {e}")
                continue
        
        return downloaded_images
    
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
        output_dir = Path("outputs/inference_results_v2")
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
                # 標準推論
                output = self.run_inference(img_data['image'], img_data['prompt'])
                
                # 結果を可視化
                if output['mask'] is not None:
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
                    "mask_shape": mask.shape if mask is not None else None,
                    "mask_min": float(mask.min()) if mask is not None else None,
                    "mask_max": float(mask.max()) if mask is not None else None,
                    "mask_mean": float(mask.mean()) if mask is not None else None,
                    "positive_pixels": int((mask > 0.5).sum()) if mask is not None else 0,
                    "total_pixels": int(mask.size) if mask is not None else 0
                }
                results.append(result_info)
                
                if mask is not None and mask.size > 0:
                    logger.info(f"Mask stats - Min: {result_info['mask_min']:.3f}, "
                              f"Max: {result_info['mask_max']:.3f}, "
                              f"Mean: {result_info['mask_mean']:.3f}")
                    logger.info(f"Positive pixels: {result_info['positive_pixels']}/{result_info['total_pixels']} "
                              f"({100*result_info['positive_pixels']/result_info['total_pixels']:.1f}%)")
                
                # 生成ベースの推論も試す（オプション）
                logger.info("Trying generation-based inference...")
                gen_output = self.run_inference_with_generation(img_data['image'], img_data['prompt'])
                logger.info(f"Generation result: {gen_output}")
                
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
        logger.info(f"Processed {len([r for r in results if r['mask_shape'] is not None])}/{len(sample_images)} images successfully")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LISA改モデルの推論テスト（改良版）")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="outputs/minimal_train_20250812_190145/checkpoints/best",
        help="チェックポイントディレクトリのパス"
    )
    
    args = parser.parse_args()
    
    # 推論を実行
    runner = InferenceRunnerV2(args.checkpoint_dir)
    runner.run()


if __name__ == "__main__":
    main()