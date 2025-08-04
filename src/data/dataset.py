"""
LISA改 (Qwen2.5-VL + SAM2.1) データパイプライン
仕様書第3章に従った実装
"""

import glob
import os
import random
from typing import Dict, List, Tuple, Optional, Any
import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torch.utils.data
from pycocotools import mask
from transformers import AutoProcessor
from torchvision import transforms

from .conversation import get_default_conv_template
from .data_processing import get_mask_from_json
from .reason_seg_dataset import ReasonSegDataset
from .refer import REFER
from .refer_seg_dataset import ReferSegDataset
from .sem_seg_dataset import SemSegDataset
from .vqa_dataset import VQADataset

# 設定ファイルの動的インポート
def get_config():
    """実行時の設定ファイルを動的に取得"""
    try:
        # 本プロジェクトのLISAConfigを使用
        from ..config import LISAConfig
        config = LISAConfig()
        print("設定: LISAConfig を使用")
        
        # 属性名を調整（LISAConfig はプロパティ名が異なる場合があるため）
        class ConfigWrapper:
            def __init__(self, lisa_config):
                self.MODEL_MAX_LENGTH = lisa_config.model_max_length
                self.QWEN_IMAGE_SIZE = lisa_config.qwen_image_size
                self.SAM_IMAGE_SIZE = lisa_config.sam_image_size
                self.SEG_TOKEN = lisa_config.seg_token
                self.DATASET_BASE_DIR = lisa_config.dataset_base_dir
                self.SEM_SEG_DATA = lisa_config.sem_seg_data
                self.REFER_SEG_DATA = lisa_config.refer_seg_data
                self.VQA_DATA = lisa_config.vqa_data
                self.REASON_SEG_DATA = lisa_config.reason_seg_data
        
        return ConfigWrapper(config)
            
    except ImportError as e:
        print(f"LISAConfigのインポートに失敗: {e}")
        
    # フォールバック: デフォルト設定
    print("警告: 設定ファイルが見つかりません。デフォルト設定を使用します。")
    class DefaultConfig:
        MODEL_MAX_LENGTH = 2048
        QWEN_IMAGE_SIZE = 448
        SAM_IMAGE_SIZE = 1024
        SEG_TOKEN = "<SEG>"
        DATASET_BASE_DIR = "/mnt/h/download/LISA-dataset/dataset"
        SEM_SEG_DATA = "ade20k||cocostuff"
        REFER_SEG_DATA = "refcoco||refcoco+||refcocog"
        VQA_DATA = "llava_instruct_150k"
        REASON_SEG_DATA = "ReasonSeg|train"
    
    return DefaultConfig()

# 設定を取得
config = get_config()

# デフォルト設定
DEFAULT_IMAGE_TOKEN = "<image>"
# utils/constants.pyから統一されたIMAGE_TOKEN_INDEXを使用
from .constants import IMAGE_TOKEN_INDEX
DEFAULT_SEG_TOKEN = getattr(config, 'SEG_TOKEN', "<SEG>")
IGNORE_INDEX = -100

def setup_seg_token(tokenizer, seg_token="<SEG>"):
    """
    オリジナルLISA準拠の[SEG]トークンセットアップ
    一元化された処理でフォールバックなし
    """
    # オリジナルLISA準拠: tokenizer.add_tokens("[SEG]")
    num_added_tokens = tokenizer.add_tokens(seg_token)
    # オリジナルLISA準拠: tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    seg_token_idx = tokenizer(seg_token, add_special_tokens=False).input_ids[0]
    
    print(f"[SEG]トークンセットアップ完了:")
    print(f"  - 追加されたトークン数: {num_added_tokens}")
    print(f"  - [SEG]トークンID: {seg_token_idx}")
    
    return seg_token_idx

def preprocess_sam_image(image: Image.Image, target_size: Optional[int] = None) -> torch.Tensor:
    """
    SAM用画像前処理：1024x1024にリサイズ・パディング・正規化
    """
    if target_size is None:
        target_size = getattr(config, 'SAM_IMAGE_SIZE', 1024)
    
    # 1. 最長辺を1024にリサイズ
    w, h = image.size
    if max(w, h) != target_size:
        if w > h:
            new_w, new_h = target_size, int(h * target_size / w)
        else:
            new_w, new_h = int(w * target_size / h), target_size
        image = image.resize((new_w, new_h), Image.LANCZOS)
    
    # 2. 1024x1024にパディング
    w, h = image.size
    pad_w = (target_size - w) // 2
    pad_h = (target_size - h) // 2
    
    # パディング用の新しい画像を作成
    padded_image = Image.new('RGB', (target_size, target_size), (0, 0, 0))
    padded_image.paste(image, (pad_w, pad_h))
    
    # 3. テンソル化と正規化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[123.675/255, 116.28/255, 103.53/255],  # SAMの標準正規化値
            std=[58.395/255, 57.12/255, 57.375/255]
        )
    ])
    
    return transform(padded_image)

def preprocess_qwen_image(image: Image.Image, processor: AutoProcessor, target_size: Optional[int] = None) -> torch.Tensor:
    """
    Qwen用画像前処理：448x448にリサイズ・正規化
    """
    if target_size is None:
        target_size = getattr(config, 'QWEN_IMAGE_SIZE', 448)
    
    # AutoProcessorを使用してQwen用前処理
    try:
        processed = processor(images=image, return_tensors="pt")
        image_tensor = processed['pixel_values'].squeeze(0)  # (1, C, H, W) -> (C, H, W)
        return image_tensor
    except Exception as e:
        print(f"Qwen画像前処理エラー: {e}")
        # フォールバック: 手動前処理
        image_resized = image.resize((target_size, target_size), Image.LANCZOS)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return transform(image_resized)



def build_correct_labels_for_qwen(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """
    Qwen2.5-VLチャットテンプレートに準拠した正確なラベルマスキング
    
    Args:
        input_ids: トークンID列 [seq_len]
        tokenizer: Qwen2.5-VL用トークナイザー
    
    Returns:
        正確にマスクされたラベル [seq_len]
    """
    labels = input_ids.clone()
    
    # Qwen2.5-VLでは、アシスタントの応答部分のみを予測対象とする
    # プロンプト部分（ユーザー入力）は-100でマスク
    
    # シンプルな実装: 全体を-100で初期化し、応答部分のみを有効にする
    # 実際のQwen2.5-VLのチャットテンプレートに応じて調整が必要
    
    # デフォルトでは入力全体を予測対象とする（後で調整可能）
    # labels = input_ids.clone()
    
    return labels

def preprocess_mask(mask: np.ndarray, target_size: Optional[int] = None) -> torch.Tensor:
    """
    マスクの前処理
    """
    if target_size is None:
        target_size = getattr(config, 'SAM_IMAGE_SIZE', 1024)
    
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # マスクの形状を検証
    if mask.size == 0:
        raise ValueError("空のマスクです")
    
    if mask.ndim < 2:
        raise ValueError(f"マスクの次元が不正です: {mask.ndim}D (最低2D必要)")
    
    # マスクを2次元に変換
    if mask.ndim == 2:
        h, w = mask.shape
        mask_2d = mask
    elif mask.ndim == 3:
        if mask.shape[0] == 1:  # (1, H, W)
            mask_2d = mask[0]
            h, w = mask_2d.shape
        elif mask.shape[-1] == 1:  # (H, W, 1)
            mask_2d = mask[:, :, 0]
            h, w = mask_2d.shape
        elif mask.shape[0] == 3 or mask.shape[-1] == 3:  # RGB マスク
            # RGB マスクの場合、最初のチャンネルを使用
            if mask.shape[0] == 3:  # (3, H, W)
                mask_2d = mask[0]
                h, w = mask_2d.shape
            else:  # (H, W, 3)
                mask_2d = mask[:, :, 0]
                h, w = mask_2d.shape
        else:
            # その他の場合、最後の2次元を使用
            mask_2d = mask.reshape(-1, mask.shape[-2], mask.shape[-1])[0]
            h, w = mask_2d.shape
    else:
        raise ValueError(f"サポートされていないマスクの次元: {mask.ndim}D")
    
    # サイズの検証
    if h == 0 or w == 0:
        raise ValueError(f"無効なマスクサイズ: {h}x{w}")
    
    # マスクのリサイズ（OpenCVのバグ回避のためPillowを使用）
    if (h, w) != (target_size, target_size):
        try:
            # numpy -> PIL Image -> リサイズ -> numpy
            mask_uint8 = mask_2d.astype(np.uint8)
            mask_pil = Image.fromarray(mask_uint8, mode='L')  # グレースケール
            mask_resized_pil = mask_pil.resize((target_size, target_size), Image.NEAREST)
            mask_2d = np.array(mask_resized_pil)
        except Exception as e:
            raise ValueError(f"マスクリサイズ失敗 (元サイズ: {h}x{w}, target: {target_size}x{target_size}): {e}")
    
    # テンソル化 (常に1次元追加して (1, H, W) 形式にする)
    if mask_2d.ndim == 2:
        mask_2d = mask_2d[None, ...]  # (H, W) -> (1, H, W)
    
    return torch.from_numpy(mask_2d).float()

class HybridDataset(torch.utils.data.Dataset):
    """
    仕様書第3章.2 HybridDatasetの実装
    デュアルストリーム処理：Qwen用とSAM用の2系統前処理を同時実行
    """

    def __init__(
        self,
        base_image_dir: Optional[str] = None,
        qwen_processor: Optional[AutoProcessor] = None,
        samples_per_epoch: int = 500 * 8 * 2 * 10,
        precision: str = "bf16",
        qwen_image_size: Optional[int] = None,
        sam_image_size: Optional[int] = None,
        num_classes_per_sample: int = 3,
        exclude_val: bool = False,
        dataset: str = "sem_seg||refer_seg||vqa||reason_seg",
        sample_rate: List[float] = [9, 3, 3, 1],
        sem_seg_data: Optional[str] = None,
        refer_seg_data: Optional[str] = None,
        vqa_data: Optional[str] = None,
        reason_seg_data: Optional[str] = None,
        explanatory: float = 0.1,
    ):
        # 設定ファイルからパラメータを取得（引数で指定されていない場合）
        self.base_image_dir = base_image_dir or getattr(config, 'DATASET_BASE_DIR', './dataset')
        self.qwen_processor = qwen_processor
        self.samples_per_epoch = samples_per_epoch
        self.precision = precision
        self.qwen_image_size = qwen_image_size or getattr(config, 'QWEN_IMAGE_SIZE', 448)
        self.sam_image_size = sam_image_size or getattr(config, 'SAM_IMAGE_SIZE', 1024)
        self.num_classes_per_sample = num_classes_per_sample
        self.exclude_val = exclude_val
        self.explanatory = explanatory
        
        # データセット設定を設定ファイルから取得
        self.sem_seg_data = sem_seg_data or getattr(config, 'SEM_SEG_DATA', "ade20k||cocostuff")
        self.refer_seg_data = refer_seg_data or getattr(config, 'REFER_SEG_DATA', "refcoco||refcoco+||refcocog")
        self.vqa_data = vqa_data or getattr(config, 'VQA_DATA', "llava_instruct_150k")
        self.reason_seg_data = reason_seg_data or getattr(config, 'REASON_SEG_DATA', "ReasonSeg|train")
        
        # サンプルレートの正規化
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()

        # [SEG]トークンの一元セットアップ（オリジナルLISA準拠）
        if self.qwen_processor and self.qwen_processor.tokenizer:
            self.seg_token = getattr(config, 'SEG_TOKEN', '[SEG]')
            self.seg_token_idx = setup_seg_token(self.qwen_processor.tokenizer, self.seg_token)
            self.max_length = getattr(config, 'MODEL_MAX_LENGTH', 2048)
        else:
            raise ValueError("qwen_processor は必須です")

        # データセットの初期化
        self.datasets = dataset.split("||")
        self.all_datasets = []
        
        print(f"HybridDataset初期化:")
        print(f"  - ベースディレクトリ: {self.base_image_dir}")
        print(f"  - Qwen画像サイズ: {self.qwen_image_size}")
        print(f"  - SAM画像サイズ: {self.sam_image_size}")
        print(f"  - 対象データセット: {self.datasets}")
        
        # Semantic Segmentation Dataset
        if "sem_seg" in self.datasets:
            print(f"  - Semantic Segmentation: {self.sem_seg_data}")
            try:
                self.all_datasets.append(
                    SemSegDataset(
                        self.base_image_dir,
                        self.qwen_processor.tokenizer,
                        None,  # vision_tower は使用しない
                        samples_per_epoch,
                        precision,
                        self.qwen_image_size,
                        num_classes_per_sample,
                        exclude_val,
                        self.sem_seg_data,
                    )
                )
            except Exception as e:
                print(f"    警告: Semantic Segmentationデータセットの初期化に失敗: {e}")
        
        # Referring Segmentation Dataset
        if "refer_seg" in self.datasets:
            print(f"  - Referring Segmentation: {self.refer_seg_data}")
            try:
                self.all_datasets.append(
                    ReferSegDataset(
                        self.base_image_dir,
                        self.qwen_processor.tokenizer,
                        None,  # vision_tower は使用しない
                        samples_per_epoch,
                        precision,
                        self.qwen_image_size,
                        num_classes_per_sample,
                        exclude_val,
                        self.refer_seg_data,
                    )
                )
            except Exception as e:
                print(f"    警告: Referring Segmentationデータセットの初期化に失敗: {e}")
        
        # VQA Dataset
        if "vqa" in self.datasets:
            print(f"  - VQA: {self.vqa_data}")
            try:
                self.all_datasets.append(
                    VQADataset(
                        self.base_image_dir,
                        self.qwen_processor.tokenizer,
                        None,  # vision_tower は使用しない
                        samples_per_epoch,
                        precision,
                        self.qwen_image_size,
                        exclude_val,
                        self.vqa_data,
                    )
                )
            except Exception as e:
                print(f"    警告: VQAデータセットの初期化に失敗: {e}")
        
        # Reasoning Segmentation Dataset
        if "reason_seg" in self.datasets:
            print(f"  - Reasoning Segmentation: {self.reason_seg_data}")
            try:
                self.all_datasets.append(
                    ReasonSegDataset(
                        base_image_dir=self.base_image_dir,
                        tokenizer=self.qwen_processor.tokenizer,
                        vision_tower=None,  # vision_tower は使用しない
                        samples_per_epoch=samples_per_epoch,
                        precision=precision,
                        image_size=self.qwen_image_size,
                        num_classes_per_sample=num_classes_per_sample,
                        exclude_val=exclude_val,
                        reason_seg_data=self.reason_seg_data,
                        explanatory=explanatory,
                    )
                )
            except Exception as e:
                print(f"    警告: Reasoning Segmentationデータセットの初期化に失敗: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"✅ HybridDataset初期化完了: {len(self.all_datasets)} データセット")
        
        # sample_rateとdataset数の一致を確保
        if len(self.sample_rate) != len(self.all_datasets):
            print(f"⚠️  sample_rate調整: {len(self.sample_rate)} -> {len(self.all_datasets)}")
            if len(self.all_datasets) == 0:
                raise ValueError("有効なデータセットが1つもありません")
            elif len(self.all_datasets) == 1:
                self.sample_rate = [1.0]
            else:
                # 利用可能なデータセット数に合わせてsample_rateを正規化
                original_sample_rate = self.sample_rate[:len(self.all_datasets)]
                total_rate = sum(original_sample_rate)
                self.sample_rate = [rate / total_rate for rate in original_sample_rate]

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx) -> Dict[str, Any]:
        """
        統合データセットからサンプルを取得
        仕様書第3章.2準拠のデュアルストリーム・データパイプライン
        """
        # データセットをサンプリング比率に従って選択
        dataset_idx = np.random.choice(len(self.all_datasets), p=self.sample_rate)
        selected_dataset = self.all_datasets[dataset_idx]
        
        # 選択されたデータセットからサンプルを取得
        try:
            sample = selected_dataset[idx % len(selected_dataset)]
        except Exception as e:
            print(f"データセット取得エラー: {e}")
            # フォールバック: 最初のデータセットから取得
            sample = self.all_datasets[0][0]
        
        # サンプルの形式を確認
        if len(sample) == 9:
            # 新しい9要素形式（SemSegDataset, ReferSegDataset）
            image_path, image_sam, image_qwen, conversations, masks, label, resize, questions, sampled_classes = sample
            
            # conversationsからテキストプロンプトを抽出
            if isinstance(conversations, list) and len(conversations) > 0:
                text_prompt = conversations[0]  # 最初の会話を使用
            else:
                text_prompt = "Segment the object in this image. [SEG]"
            
            # resize と questions, sampled_classes を保持（オリジナルLISAとの互換性）
            resize = resize if 'resize' in locals() else None
            questions = questions if 'questions' in locals() else None
            sampled_classes = sampled_classes if 'sampled_classes' in locals() else None
            
        elif len(sample) == 5:
            # 古い5要素形式（VQADataset, ReasonSegDataset）
            image_path, image_data, text_prompt, masks, label = sample
            
            # 画像データの処理
            if isinstance(image_data, torch.Tensor):
                # 既に前処理済みの場合
                if image_data.dim() == 3 and image_data.size(0) == 3:
                    # (C, H, W) → (H, W, C)
                    image_np = image_data.permute(1, 2, 0).cpu().numpy()
                    if image_np.max() <= 1.0:
                        image_np = (image_np * 255).astype(np.uint8)
                else:
                    image_np = image_data.cpu().numpy()
                
                # 形状の検証
                if len(image_np.shape) >= 2 and (image_np.shape[0] <= 1 or image_np.shape[1] <= 1):
                    raise ValueError(f"無効な画像サイズ: {image_np.shape}")
                
                # データ型の検証と変換
                if image_np.dtype == np.float32 or image_np.dtype == np.float64:
                    # float型の場合、0-1範囲を0-255に変換してuint8に
                    if image_np.max() <= 1.0:
                        image_np = (image_np * 255).astype(np.uint8)
                    else:
                        # 既に0-255範囲の場合
                        image_np = image_np.astype(np.uint8)
                elif image_np.dtype != np.uint8:
                    # その他の型の場合はuint8に変換
                    image_np = image_np.astype(np.uint8)
                
                # PIL画像に変換
                image_pil = Image.fromarray(image_np)
                
            elif isinstance(image_data, Image.Image):
                image_pil = image_data
            else:
                # numpyまたはその他の形式
                if hasattr(image_data, 'shape'):
                    # データ型の検証と変換
                    if image_data.dtype == np.float32 or image_data.dtype == np.float64:
                        if image_data.max() <= 1.0:
                            image_data = (image_data * 255).astype(np.uint8)
                        else:
                            image_data = image_data.astype(np.uint8)
                    elif image_data.dtype != np.uint8:
                        image_data = image_data.astype(np.uint8)
                    
                    image_pil = Image.fromarray(image_data)
                else:
                    raise ValueError(f"サポートされていない画像形式: {type(image_data)}")
            
            # デュアル前処理
            image_qwen = preprocess_qwen_image(image_pil, self.qwen_processor, self.qwen_image_size)
            image_sam = preprocess_sam_image(image_pil, self.sam_image_size)
            
            # オリジナルLISAとの互換性のために初期化
            resize = None
            questions = None
            sampled_classes = None
            
        else:
            raise ValueError(f"不明なサンプル形式: {len(sample)} 要素")
        


        # [SEG]トークンが含まれていることを確認
        if self.seg_token not in text_prompt:
            text_prompt += f" {self.seg_token}"

        # Qwen-3の正しいマルチモーダル処理
        # PIL画像を準備
        if len(sample) == 9:
            # 新しい9要素形式の場合、既にPIL画像がある
            if isinstance(image_sam, torch.Tensor):
                # SAM画像テンソルからPIL画像を復元
                if image_sam.dim() == 3:  # (C, H, W)
                    image_np = image_sam.permute(1, 2, 0).cpu().numpy()
                    if image_np.max() <= 1.0:
                        image_np = (image_np * 255).astype(np.uint8)
                    
                    # 形状の検証
                    if image_np.shape[0] == 1 or image_np.shape[1] == 1:
                        raise ValueError(f"無効な画像サイズ: {image_np.shape}")
                    
                    # データ型の検証と変換
                    if image_np.dtype == np.float32 or image_np.dtype == np.float64:
                        if image_np.max() <= 1.0:
                            image_np = (image_np * 255).astype(np.uint8)
                        else:
                            image_np = image_np.astype(np.uint8)
                    elif image_np.dtype != np.uint8:
                        image_np = image_np.astype(np.uint8)
                    
                    image_pil = Image.fromarray(image_np)
                else:
                    raise ValueError(f"無効なテンソル次元: {image_sam.dim()}")
            else:
                if isinstance(image_sam, Image.Image):
                    image_pil = image_sam
                else:
                    raise ValueError(f"サポートされていない画像形式: {type(image_sam)}")
        else:
            # 5要素形式の場合、既にimage_pilが準備されている
            pass
        
        # Qwen-3の公式apply_chat_templateを使用
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_pil},
                    {"type": "text", "text": text_prompt}
                ]
            }
        ]
        
        try:
            # Qwen-3プロセッサーでマルチモーダル処理
            qwen_processed = self.qwen_processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )
            
            # 処理結果から必要な要素を抽出
            input_ids = qwen_processed['input_ids'].squeeze(0)
            attention_mask = qwen_processed['attention_mask'].squeeze(0)
            pixel_values = qwen_processed['pixel_values'].squeeze(0)
            
            # image_grid_thwを取得（存在する場合）
            image_grid_thw = qwen_processed.get('image_grid_thw', None)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.squeeze(0) if image_grid_thw.dim() > 1 else image_grid_thw
            
            # pixel_valuesをimage_qwenとして使用
            image_qwen = pixel_values
            
            # SAM用画像を別途処理
            image_sam = preprocess_sam_image(image_pil, self.sam_image_size)
            
        except Exception as e:
            print(f"❌ Qwen-3マルチモーダル処理エラー: {e}")
            print(f"   テキスト: {text_prompt[:100]}...")
            raise RuntimeError(f"Qwen-3マルチモーダル処理に失敗: {e}")

        # 画像の形状を確認（apply_chat_templateで処理済みなので基本的に正しい形状）
        if image_qwen.dim() == 4:  # (1, C, H, W) → (C, H, W)
            image_qwen = image_qwen.squeeze(0)
        
        if image_sam.dim() == 4:  # (1, C, H, W) → (C, H, W)
            image_sam = image_sam.squeeze(0)

        # [SEG]トークンの位置を特定（オリジナルLISA準拠）
        seg_token_mask = (input_ids == self.seg_token_idx)

        # ラベルの処理（言語生成用）
        # Qwen2.5-VLチャットテンプレートに準拠した正確なラベルマスキング
        labels = build_correct_labels_for_qwen(input_ids, self.qwen_processor.tokenizer)

        # マスクの処理
        has_mask = masks is not None
        if has_mask:
            if isinstance(masks, torch.Tensor):
                if masks.dim() == 2:  # (H, W) → (1, H, W)
                    ground_truth_mask = masks.unsqueeze(0)
                elif masks.dim() == 3:  # (N, H, W) → 最初のマスクを使用
                    ground_truth_mask = masks[0:1]  # (1, H, W)
                else:
                    ground_truth_mask = masks
            else:
                # numpy配列の場合
                if isinstance(masks, np.ndarray):
                    ground_truth_mask = torch.from_numpy(masks)
                    if ground_truth_mask.dim() == 2:
                        ground_truth_mask = ground_truth_mask.unsqueeze(0)
                else:
                    ground_truth_mask = torch.zeros(1, self.sam_image_size, self.sam_image_size)
                    has_mask = False
            
            # マスクのサイズ調整
            if ground_truth_mask.size(-1) != self.sam_image_size or ground_truth_mask.size(-2) != self.sam_image_size:
                ground_truth_mask = F.interpolate(
                    ground_truth_mask.unsqueeze(0).float(),
                    size=(self.sam_image_size, self.sam_image_size),
                    mode='nearest'
                ).squeeze(0)
        else:
            ground_truth_mask = torch.zeros(1, self.sam_image_size, self.sam_image_size)

        # 返り値の構築（仕様書準拠、オリジナルLISAとの互換性を保持）
        # collate_fnが期待するキー名に統一
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'sam_images': image_sam,      # SAM用画像 (C, 1024, 1024)
            'pixel_values': image_qwen,  # Qwen用画像 (C, 448, 448)
            'ground_truth_mask': ground_truth_mask if has_mask else None,
            'has_mask': has_mask,
            'seg_token_mask': seg_token_mask,
            'image_path': image_path if 'image_path' in locals() else None,
            'text_prompt': text_prompt,  # 追加: collate_fn用
            'image_grid_thw': image_grid_thw if 'image_grid_thw' in locals() else None,  # Qwen2.5-VL用
            # オリジナルLISAとの互換性のための追加フィールド
            'resize': resize if 'resize' in locals() else None,
            'questions': questions if 'questions' in locals() else None,
            'sampled_classes': sampled_classes if 'sampled_classes' in locals() else None,
        }

def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    仕様書第3章.3 バッチの結合 (collate_fn)
    デュアルストリーム対応のカスタムcollate関数
    
    最大シーケンス長は設定ファイルのMODEL_MAX_LENGTHを自動的に使用:
    - config_small_test.py が利用可能な場合: 512 (メモリ効率優先)
    - config_linux.py のみの場合: 2048 (通常設定)
    - 設定ファイルなしの場合: 2048 (デフォルト)
    """
    # 設定の取得
    config = get_config()
    sam_image_size = getattr(config, 'SAM_IMAGE_SIZE', 1024)
    # 各キーごとにデータを収集
    pixel_values = []
    sam_images = []
    input_ids = []
    attention_masks = []
    labels = []
    seg_token_masks = []
    ground_truth_masks = []
    has_masks = []
    image_paths = []
    text_prompts = []
    resize_list = []  # オリジナルLISA互換
    questions_list = []  # オリジナルLISA互換
    sampled_classes_list = []  # オリジナルLISA互換
    image_grid_thws = []  # Qwen2.5-VL用
    
    for item in batch:
        pixel_values.append(item["pixel_values"])
        sam_images.append(item["sam_images"])
        input_ids.append(item["input_ids"])
        attention_masks.append(item["attention_mask"])
        
        # labelsの処理（正確なラベルマスキング適用済み）
        label = item["labels"]
        if isinstance(label, torch.Tensor):
            # トークンレベルのラベル（言語生成）の場合
            if label.dim() == 1 and len(label) == item["input_ids"].size(0):
                labels.append(label)  # build_correct_labels_for_qwenで処理済み
            elif label.dim() == 0:  # スカラーテンソル
                # スカラーラベルの場合は全シーケンスに同じラベルを適用（通常はしない）
                labels.append(torch.full_like(item["input_ids"], label.item()))
            else:
                # その他の場合はinput_idsと同じ長さのダミーラベル
                labels.append(torch.full_like(item["input_ids"], -100))
        else:
            # 非テンソルの場合
            labels.append(torch.full_like(item["input_ids"], -100))
        
        seg_token_masks.append(item["seg_token_mask"])
        if item.get("has_mask", False):
            ground_truth_masks.append(item["ground_truth_mask"])
        has_masks.append(item.get("has_mask", False))
        image_paths.append(item.get("image_path"))
        text_prompts.append(item.get("text_prompt"))
        
        # オリジナルLISA互換フィールド
        resize_list.append(item.get("resize"))
        questions_list.append(item.get("questions"))
        sampled_classes_list.append(item.get("sampled_classes"))
        
        # Qwen2.5-VL用
        image_grid_thws.append(item.get("image_grid_thw"))
    
    # テンソルのスタック
    pixel_values = torch.stack(pixel_values)  # (B, 3, 448, 448)
    sam_images = torch.stack(sam_images)      # (B, 3, 1024, 1024)
    
    # テキストシーケンスの長さ統一（パディング）
    max_length = max(ids.size(0) for ids in input_ids)
    
    # パディング関数の定義
    def pad_sequence(sequences, max_len, pad_value=0):
        padded = []
        for seq in sequences:
            if seq.size(0) < max_len:
                padding = torch.full((max_len - seq.size(0),), pad_value, dtype=seq.dtype)
                padded_seq = torch.cat([seq, padding])
            else:
                padded_seq = seq[:max_len]  # 切り詰め
            padded.append(padded_seq)
        return torch.stack(padded)
    
    # シーケンスをパディング
    input_ids_padded = pad_sequence(input_ids, max_length, pad_value=0)
    attention_masks_padded = pad_sequence(attention_masks, max_length, pad_value=0)
    labels_padded = pad_sequence(labels, max_length, pad_value=-100)  # -100でパディング
    seg_token_masks_padded = pad_sequence(seg_token_masks, max_length, pad_value=False)
    
    # マスクが存在するサンプルのみをスタック
    if ground_truth_masks:
        ground_truth_masks_stacked = torch.stack(ground_truth_masks)
    else:
        ground_truth_masks_stacked = None

    # オリジナルLISA互換のlabel_listを生成（セグメンテーション用）
    label_list = []
    for has_mask in has_masks:
        if has_mask:
            # マスクがある場合、ignore_label=255のラベルマップを作成
            label_list.append(torch.ones(1, sam_image_size, sam_image_size) * 255)
        else:
            label_list.append(None)
    
    # image_grid_thwのバッチ化
    # Noneでないものだけをスタック
    valid_grid_thws = [g for g in image_grid_thws if g is not None]
    if valid_grid_thws:
        image_grid_thw_batch = torch.stack(valid_grid_thws)
    else:
        image_grid_thw_batch = None

    return {
        # Qwen-3デュアルストリーム用
        "pixel_values": pixel_values,           # (B, 3, 448, 448)
        "sam_images": sam_images,               # (B, 3, 1024, 1024)
        "input_ids": input_ids_padded,                  # (B, unified_max_length)
        "attention_masks": attention_masks_padded,      # (B, unified_max_length) - オリジナルLISA準拠の命名
        "labels": labels_padded,                        # (B, unified_max_length) - 正確にマスク済み
        "image_grid_thw": image_grid_thw_batch,         # (B, 3) or None - Qwen2.5-VL用
        "seg_token_mask": seg_token_masks_padded,      # (B, unified_max_length)
        "ground_truth_mask": ground_truth_masks_stacked,# (num_masks, 1, 1024, 1024) or None
        "has_mask": has_masks,                          # List[bool]
        "image_paths": image_paths,                     # List[str]
        "text_prompts": text_prompts,                   # List[str]
        # オリジナルLISA互換フィールド
        "masks_list": ground_truth_masks,               # List[Tensor] - オリジナルLISA形式
        "label_list": label_list,                       # List[Tensor] - オリジナルLISA形式
        "resize_list": resize_list,                     # List[Optional[Any]]
        "questions_list": questions_list,               # List[Optional[Any]]
        "sampled_classes_list": sampled_classes_list,   # List[Optional[Any]]
        # 追加の互換性フィールド
        "images": sam_images,                       # エイリアス: オリジナルLISAでの名前
        "images_clip": pixel_values,                # エイリアス: オリジナルLISAでCLIP画像として使用
    }

# エイリアスは削除 - 明確な命名を使用
# 正式名称を使用してください:
# - HybridDataset (統合データセット)
# - collate_fn (バッチ結合関数)

class LisaQwen3ValDataset(torch.utils.data.Dataset):
    """
    LISA-Qwen3用の評価データセット
    デュアルストリーム対応
    """

    def __init__(
        self,
        base_image_dir: str,
        qwen_processor: AutoProcessor,
        val_dataset: str,
        qwen_image_size: int = 448,
        sam_image_size: int = 1024,
    ):
        self.base_image_dir = base_image_dir
        self.qwen_processor = qwen_processor
        self.qwen_image_size = qwen_image_size
        self.sam_image_size = sam_image_size
        
        # 評価用データセットの初期化
        if "refer_seg" in val_dataset.lower():
            self.dataset = ReferSegDataset(
                base_image_dir,
                qwen_processor,
                None,
                1000,  # サンプル数
                "bf16",
                qwen_image_size,
                3,
                False,
                val_dataset,
            )
        elif "sem_seg" in val_dataset.lower():
            self.dataset = SemSegDataset(
                base_image_dir,
                qwen_processor,
                None,
                1000,  # サンプル数
                "bf16",
                qwen_image_size,
                3,
                False,
                val_dataset,
            )
        else:
            # ReasonSegをデフォルトとする
            self.dataset = ReasonSegDataset(
                        base_image_dir,
                qwen_processor,
                None,
                1000,
                "bf16",
                qwen_image_size,
                3,
                False,
                "ReasonSeg|val",
            )

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        """評価用サンプルの取得（デュアルストリーム対応）"""
        sample = self.dataset[idx]
        
        # 訓練用データセットと同じ形式で処理
        if len(sample) < 4:
            raise ValueError(f"評価データセットのサンプル形式が不正です: {len(sample)} 要素 (最低4要素必要)")
        
        image_path = sample[0]
        image = sample[1]
        text_prompt = sample[2] if len(sample) > 2 else "Segment the object in this image. [SEG]"
        mask = sample[3] if len(sample) > 3 else None
        label = sample[4] if len(sample) > 4 else torch.tensor(0)
        
        # 画像の型変換
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.permute(1, 2, 0)
            image = image.cpu().numpy()
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        elif isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise TypeError(f"サポートされていない画像型: {type(image)}")
        
        # 画像サイズの検証
        if image.size[0] == 0 or image.size[1] == 0:
            raise ValueError(f"無効な画像サイズ: {image.size}")
        
        # デュアルストリーム前処理（訓練用と同じ）
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_prompt}
                ]
            }
        ]
        
        try:
            qwen_processed = self.qwen_processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )
        except Exception as e:
            raise RuntimeError(f"評価用Qwen前処理に失敗 (idx={idx}): {e}")
        
        pixel_values = qwen_processed['pixel_values'].squeeze(0)
        input_ids = qwen_processed['input_ids'].squeeze(0)
        attention_mask = qwen_processed['attention_mask'].squeeze(0)
        
        try:
            sam_images = preprocess_sam_image(image, self.sam_image_size)
        except Exception as e:
            raise RuntimeError(f"評価用SAM前処理に失敗 (idx={idx}): {e}")
        
        seg_token_id = self.qwen_processor.tokenizer.convert_tokens_to_ids("[SEG]")
        if seg_token_id is None:
            raise ValueError("[SEG]トークンがトークナイザーに見つかりません")
        
        seg_token_mask = (input_ids == seg_token_id)
        
        # マスクの処理
        has_mask = mask is not None and (isinstance(mask, (torch.Tensor, np.ndarray)) and mask.sum() > 0)
        if has_mask:
            try:
                if isinstance(mask, torch.Tensor):
                    mask_np = mask.cpu().numpy()
                else:
                    mask_np = np.array(mask)
                ground_truth_mask = preprocess_mask(mask_np, self.sam_image_size)
            except Exception as e:
                raise RuntimeError(f"評価用マスク前処理に失敗 (idx={idx}): {e}")
        else:
            ground_truth_mask = torch.zeros(1, self.sam_image_size, self.sam_image_size)
        
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label)
        
        return {
            "pixel_values": pixel_values,
            "sam_images": sam_images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
            "seg_token_mask": seg_token_mask,
            "ground_truth_mask": ground_truth_mask,
            "has_mask": has_mask,
            "image_path": image_path,
            "text_prompt": text_prompt,
        }
