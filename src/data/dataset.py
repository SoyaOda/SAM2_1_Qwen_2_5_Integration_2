"""
LISA改 (Qwen2.5-VL + SAM2.1) データパイプライン
仕様書第3章に従った実装
"""

import glob
import os
import random
from typing import Dict, List, Tuple, Optional, Any
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import cv2
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)
import torch.nn.functional as F
from PIL import Image
import torch.utils.data
from pycocotools import mask

# キャッシュ機能をインポート
try:
    from .dataset_cache import DatasetCache
except ImportError:
    from dataset_cache import DatasetCache
from transformers import AutoProcessor
from torchvision import transforms
from qwen_vl_utils import process_vision_info
from sam2.utils.transforms import SAM2Transforms

from src.utils.coordinate_transform import CoordinateTransform
from src.utils.resolution_utils import ResolutionBucketManager, QualityScoreCalculator, calculate_image_pad_tokens

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
    オリジナルLISA準拠の<SEG>トークンセットアップ
    一元化された処理でフォールバックなし
    """
    # オリジナルLISA準拠: tokenizer.add_tokens("[SEG]")
    num_added_tokens = tokenizer.add_tokens(seg_token)
    # オリジナルLISA準拠: tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    seg_token_idx = tokenizer(seg_token, add_special_tokens=False).input_ids[0]
    
    print(f"<SEG>トークンセットアップ完了:")
    print(f"  - 追加されたトークン数: {num_added_tokens}")
    print(f"  - <SEG>トークンID: {seg_token_idx}")
    
    return seg_token_idx

def preprocess_sam_image(image: Image.Image, target_size: Optional[int] = None) -> torch.Tensor:
    """
    SAM2公式仕様準拠の画像前処理：1024x1024に直接リサイズ・正規化
    パディングは行わない（SAM2は正方形リサイズを前提）
    
    注: この関数は後方互換性のために残されています。
    新しいコードではSAM2Transformsを直接使用することを推奨します。
    """
    if target_size is None:
        target_size = getattr(config, 'SAM_IMAGE_SIZE', 1024)
    
    # SAM2公式: 直接正方形リサイズ（パディング不要）
    image = image.resize((target_size, target_size), Image.LANCZOS)
    
    # SAM2公式: ImageNet正規化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # SAM2.1公式正規化値
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return transform(image)

def preprocess_qwen_image(image: Image.Image, processor: AutoProcessor, target_size: Optional[int] = None) -> torch.Tensor:
    """
    Qwen用画像前処理：動的解像度対応版
    
    Args:
        image: PIL画像
        processor: Qwen2.5-VLのAutoProcessor（min_pixels/max_pixels設定済み）
        target_size: 互換性のため残すが、動的解像度モードでは無視される
    
    Returns:
        処理済み画像テンソル (C, H, W)
    """
    # configの動的解像度設定を確認
    use_dynamic = getattr(config, 'use_dynamic_resolution', True)
    
    if not use_dynamic:
        # 従来の固定サイズ処理（互換性維持）
        if target_size is None:
            target_size = getattr(config, 'qwen_image_size', 448)
        
        # PIL画像をnumpy配列に変換
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        h, w = image_np.shape[:2]
        
        # アスペクト比を維持してリサイズ
        scale = target_size / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # 縮小時はINTER_AREA、拡大時はINTER_CUBIC
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        resized = cv2.resize(image_np, (new_w, new_h), interpolation=interpolation)
        
        # 14の倍数になるようにパディング
        pad_h = (-new_h) % 14
        pad_w = (-new_w) % 14
        
        # 上下左右に均等にパディング
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        # ゼロパディング（黒色）
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                                    cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        # PIL画像に戻してprocessorに渡す
        padded_pil = Image.fromarray(padded)
        
        # AutoProcessorを使用してQwen用前処理
        try:
            processed = processor(images=padded_pil, return_tensors="pt")
            image_tensor = processed['pixel_values'].squeeze(0)  # (1, C, H, W) -> (C, H, W)
            return image_tensor
        except Exception as e:
            print(f"Qwen画像前処理エラー: {e}")
            # フォールバック: 手動前処理
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])
            return transform(padded_pil)
    
    else:
        # 動的解像度モード
        # processorが自動的にmin_pixels/max_pixelsに基づいてリサイズ
        # 内部でsmart_resizeが呼ばれ、アスペクト比維持・14の倍数調整が行われる
        try:
            # 単純にprocessorに画像を渡すだけで、動的解像度処理が適用される
            processed = processor(images=image, return_tensors="pt")
            image_tensor = processed['pixel_values'].squeeze(0)  # (1, C, H, W) -> (C, H, W)
            
            # 処理後のサイズを確認（デバッグ用）
            _, h, w = image_tensor.shape
            if hasattr(config, 'DEBUG') and config.DEBUG:
                print(f"Dynamic resolution: original {image.size} -> processed {(h, w)} -> patches {(h//14, w//14)}")
            
            return image_tensor
            
        except Exception as e:
            print(f"動的解像度処理エラー: {e}")
            print("固定解像度モードにフォールバック")
            # エラー時は固定解像度にフォールバック
            return preprocess_qwen_image(image, processor, target_size=448)



def build_correct_labels_for_qwen(input_ids: torch.Tensor, tokenizer, is_seg_sample: bool = False) -> torch.Tensor:
    """
    Qwen2.5-VLチャットテンプレートに準拠した正確なラベルマスキング
    
    Args:
        input_ids: トークンID列 [seq_len]
        tokenizer: Qwen2.5-VL用トークナイザー
        is_seg_sample: セグメンテーションサンプルかどうか
    
    Returns:
        正確にマスクされたラベル [seq_len]
    """
    # label_utils.pyのbuild_labels_for_qwen_chatを使用
    from .label_utils import build_labels_for_qwen_chat
    
    seg_token = getattr(config, 'SEG_TOKEN', '<SEG>')
    # セグメンテーションサンプルの場合、<SEG>トークン周辺のみ学習
    seg_only_mode = is_seg_sample
    
    labels = build_labels_for_qwen_chat(
        input_ids=input_ids,
        tokenizer=tokenizer,
        seg_token=seg_token,
        is_seg_sample=is_seg_sample,
        seg_only_mode=seg_only_mode
    )
    
    return labels

def preprocess_mask(mask: np.ndarray, target_size: Optional[int] = None) -> torch.Tensor:
    """
    マスクの前処理（最適化版）
    - F.interpolateでnearest補間を使用（セグメンテーションマスクのベストプラクティス）
    - データ型の適切な管理（float32→nearest補間→bool/uint8）
    """
    if target_size is None:
        target_size = getattr(config, 'SAM_IMAGE_SIZE', 1024)
    
    # numpy配列に変換
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask
    
    # マスクの形状を検証
    if mask_np.size == 0:
        raise ValueError("空のマスクです")
    
    if mask_np.ndim < 2:
        raise ValueError(f"マスクの次元が不正です: {mask_np.ndim}D (最低2D必要)")
    
    # マスクを2次元に変換
    if mask_np.ndim == 2:
        h, w = mask_np.shape
        mask_2d = mask_np
    elif mask_np.ndim == 3:
        if mask_np.shape[0] == 1:  # (1, H, W)
            mask_2d = mask_np[0]
            h, w = mask_2d.shape
        elif mask_np.shape[-1] == 1:  # (H, W, 1)
            mask_2d = mask_np[:, :, 0]
            h, w = mask_2d.shape
        elif mask_np.shape[0] == 3 or mask_np.shape[-1] == 3:  # RGB マスク
            # RGB マスクの場合、最初のチャンネルを使用
            if mask_np.shape[0] == 3:  # (3, H, W)
                mask_2d = mask_np[0]
            else:  # (H, W, 3)
                mask_2d = mask_np[:, :, 0]
            h, w = mask_2d.shape
        else:
            # その他の場合、最初の2次元を使用
            mask_2d = mask_np.reshape(-1, mask_np.shape[-2], mask_np.shape[-1])[0]
            h, w = mask_2d.shape
    else:
        raise ValueError(f"サポートされていないマスクの次元: {mask_np.ndim}D")
    
    # サイズの検証
    if h == 0 or w == 0:
        raise ValueError(f"無効なマスクサイズ: {h}x{w}")
    
    # PyTorchテンソルに変換（F.interpolateはfloat型のみサポート）
    mask_tensor = torch.from_numpy(mask_2d).float()
    
    # バッチ次元とチャンネル次元を追加: (H, W) -> (1, 1, H, W)
    if mask_tensor.ndim == 2:
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
    elif mask_tensor.ndim == 3:
        mask_tensor = mask_tensor.unsqueeze(0)
    
    # リサイズが必要な場合のみ補間
    if (h, w) != (target_size, target_size):
        # nearest補間でリサイズ（セグメンテーションマスクのベストプラクティス）
        mask_resized = F.interpolate(
            mask_tensor,
            size=(target_size, target_size),
            mode='nearest'
        )
    else:
        mask_resized = mask_tensor
    
    # バッチ次元を削除: (1, 1, H, W) -> (1, H, W)
    mask_resized = mask_resized.squeeze(0)
    
    # bool型に変換（メモリ効率化）
    # しきい値0.5で2値化（nearest補間後も念のため）
    mask_bool = mask_resized > 0.5
    
    # float型で返す（後続の処理で必要な場合があるため）
    return mask_bool.float()

def preprocess_mask_with_aspect_ratio(
    mask: np.ndarray, 
    orig_hw: tuple,
    target_size: int = 1024
) -> torch.Tensor:
    """
    アスペクト比を維持したマスクの前処理
    画像と同じ変換を適用して座標系を一致させる
    
    Args:
        mask: 入力マスク (numpy array)
        orig_hw: 元画像のサイズ (height, width)
        target_size: 目標サイズ（デフォルト1024）
    
    Returns:
        前処理済みマスク (torch.Tensor)
    """
    if target_size is None:
        target_size = getattr(config, 'SAM_IMAGE_SIZE', 1024)
    
    # numpy配列に変換
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask
    
    # マスクの形状を検証
    if mask_np.size == 0:
        raise ValueError("空のマスクです")
    
    # 2次元に変換
    if mask_np.ndim == 2:
        mask_2d = mask_np
    elif mask_np.ndim == 3:
        if mask_np.shape[0] == 1:  # (1, H, W)
            mask_2d = mask_np[0]
        elif mask_np.shape[-1] == 1:  # (H, W, 1)
            mask_2d = mask_np[:, :, 0]
        else:
            # 最初のチャンネルを使用
            if mask_np.shape[0] <= 3:  # (C, H, W)
                mask_2d = mask_np[0]
            else:  # (H, W, C)
                mask_2d = mask_np[:, :, 0]
    else:
        raise ValueError(f"サポートされていないマスクの次元: {mask_np.ndim}D")
    
    orig_h, orig_w = orig_hw
    
    # アスペクト比を維持したリサイズ
    scale = target_size / max(orig_h, orig_w)
    new_h = int(orig_h * scale)
    new_w = int(orig_w * scale)
    
    # マスクが元画像サイズと異なる場合は、まず元画像サイズに合わせる
    if mask_2d.shape != (orig_h, orig_w):
        mask_2d = cv2.resize(mask_2d.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    # リサイズ（nearest neighborで）
    mask_resized = cv2.resize(mask_2d.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # パディング（右下に0でパディング）
    padded = np.zeros((target_size, target_size), dtype=np.float32)
    padded[:new_h, :new_w] = mask_resized
    
    # PyTorchテンソルに変換して返す
    mask_tensor = torch.from_numpy(padded).float()
    
    # (1, H, W)形式に
    if mask_tensor.ndim == 2:
        mask_tensor = mask_tensor.unsqueeze(0)
    
    return mask_tensor

class HybridDataset(torch.utils.data.Dataset):
    """
    仕様書第3章.2 HybridDatasetの実装
    デュアルストリーム処理：Qwen用とSAM用の2系統前処理を同時実行
    動的解像度・品質スコア対応版
    """

    def __init__(
        self,
        qwen_processor: AutoProcessor,
        base_image_dir: str,
        samples_per_epoch: int = 500 * 8 * 2 * 10,
        precision: str = "fp32",
        num_classes_per_sample: int = 3,
        exclude_val: bool = False,
        sem_seg_data: str = None,
        refer_seg_data: str = None,
        vqa_data: str = None,
        reason_seg_data: str = None,
        explanatory: float = -1,
        sample_rate: List[float] = [9, 3, 3, 1],
        val_dataset: str = None,
        sam_image_size: int = 1024,
        qwen_image_size: int = 448,  # Qwen2.5-VLの基本解像度
        use_quality_score: bool = False,
        use_dynamic_resolution: bool = False,
        cache_dir: str = None,
        persistent_cache: bool = False,
        seg_token: str = "<SEG>",
    ):
        """
        ハイブリッドデータセット
        
        Args:
            qwen_processor: QwenのプロセッサAutoProcessor.from_pretrained()で取得)
            base_image_dir: ベースとなる画像ディレクトリ
            samples_per_epoch: エポックあたりのサンプル数
            precision: 精度設定
            num_classes_per_sample: サンプルあたりのクラス数
            exclude_val: 検証データを除外するか
            sem_seg_data: セマンティックセグメンテーションデータの設定
            refer_seg_data: 参照セグメンテーションデータの設定
            vqa_data: VQAデータの設定
            reason_seg_data: Reason Segmentationデータの設定
            explanatory: 説明付き回答の確率
            sample_rate: 各データセットのサンプリング比率
            val_dataset: 検証データセット
            sam_image_size: SAMの画像サイズ（1024固定）
            qwen_image_size: Qwenの画像サイズ（448固定）
            use_quality_score: 品質スコア最適化の有効化
            use_dynamic_resolution: 動的解像度モードの有効化
            cache_dir: キャッシュディレクトリ（Noneの場合はメモリキャッシュ）
            persistent_cache: 永続化キャッシュを使用するか
            seg_token: セグメンテーショントークン
        """
        super().__init__()
        
        # プロセッサとパラメータの保存
        self.qwen_processor = qwen_processor
        self.base_image_dir = base_image_dir
        self.samples_per_epoch = samples_per_epoch
        self.precision = precision
        self.num_classes_per_sample = num_classes_per_sample
        self.exclude_val = exclude_val
        self.sam_image_size = sam_image_size
        self.qwen_image_size = qwen_image_size
        self.use_quality_score = use_quality_score
        self.use_dynamic_resolution = use_dynamic_resolution
        
        # データセット設定の保存
        self.sem_seg_data = sem_seg_data
        self.refer_seg_data = refer_seg_data
        self.vqa_data = vqa_data
        self.reason_seg_data = reason_seg_data
        self.explanatory = explanatory
        self.sample_rate = sample_rate
        self.seg_token = seg_token
        
        # <SEG>トークンIDを取得
        self.seg_token_idx = self.qwen_processor.tokenizer.convert_tokens_to_ids(seg_token)
        if self.seg_token_idx is None:
            raise ValueError(f"トークナイザーに{seg_token}トークンが見つかりません")
        
        # SAM2公式Transformsの初期化
        self.sam_transforms = SAM2Transforms(
            resolution=self.sam_image_size,
            mask_threshold=0.0,
            max_hole_area=0.0,
            max_sprinkle_area=0.0
        )
        
        # 品質スコア最適化の初期化
        if self.use_quality_score:
            from ..quality import ImageQualityCalculator
            self.quality_calculator = ImageQualityCalculator()
        else:
            self.quality_calculator = None
        
        # データセットキャッシュの初期化
        if cache_dir and persistent_cache:
            from .dataset_cache import PersistentDatasetCache
            self.cache = PersistentDatasetCache(cache_dir)
        elif cache_dir:
            from .dataset_cache import DatasetCache
            self.cache = DatasetCache(cache_dir=cache_dir)
        else:
            self.cache = None
        
        # データセット初期化
        self.all_datasets = []
        print(f"\n{'='*50}")
        print(f"🚀 HybridDataset初期化開始")
        print(f"{'='*50}")
        print(f"📁 ベースディレクトリ: {base_image_dir}")
        print(f"⚙️  設定:")
        print(f"   - サンプル数/エポック: {samples_per_epoch}")
        print(f"   - 精度: {precision}")
        print(f"   - クラス数/サンプル: {num_classes_per_sample}")
        print(f"   - SAM画像サイズ: {sam_image_size}")
        print(f"   - Qwen画像サイズ: {qwen_image_size}")
        print(f"   - 動的解像度: {use_dynamic_resolution}")
        print(f"   - 品質スコア最適化: {use_quality_score}")
        print(f"   - キャッシュ: {'永続化' if persistent_cache else 'メモリ' if cache_dir else '無効'}")
        print(f"{'='*50}\n")
        
        # 各データセットタイプの初期化
        # 1. Semantic Segmentation
        if sem_seg_data is not None:
            print("📊 Semantic Segmentation データセット:")
            print(f"   設定: {sem_seg_data}")
            try:
                if self.cache:
                    cache_params = {
                        'dataset_type': 'sem_seg',
                        'data': sem_seg_data,
                        'samples_per_epoch': samples_per_epoch,
                        'precision': precision,
                        'image_size': self.sam_image_size,  # 修正: SAM画像サイズを使用
                        'num_classes_per_sample': num_classes_per_sample,
                        'exclude_val': exclude_val
                    }
                    dataset = self.cache.get('sem_seg', cache_params)
                    if dataset is None:
                        dataset = SemSegDataset(
                            self.base_image_dir,
                            self.qwen_processor.tokenizer,
                            None,  # vision_tower は使用しない
                            samples_per_epoch,
                            precision,
                            self.sam_image_size,  # 修正: SAM画像サイズを使用
                            num_classes_per_sample,
                            exclude_val,
                            sem_seg_data,
                        )
                        # キャッシュに保存
                        self.cache.set('sem_seg', cache_params, dataset)
                    # キャッシュから読み込んだ場合も追加
                    self.all_datasets.append(dataset)
                else:
                    # キャッシュを使用しない場合
                    self.all_datasets.append(
                        SemSegDataset(
                            self.base_image_dir,
                            self.qwen_processor.tokenizer,
                            None,  # vision_tower は使用しない
                            samples_per_epoch,
                            precision,
                            self.sam_image_size,  # 修正: SAM画像サイズを使用
                            num_classes_per_sample,
                            exclude_val,
                            sem_seg_data,
                        )
                    )
            except Exception as e:
                print(f"   警告: Semantic Segmentationデータセットの初期化に失敗: {e}")
                import traceback
                traceback.print_exc()
        
        # 2. Referring Segmentation
        if refer_seg_data is not None:
            print("🎯 Referring Segmentation データセット:")
            print(f"   設定: {refer_seg_data}")
            try:
                if self.cache:
                    cache_params = {
                        'dataset_type': 'refer_seg',
                        'data': refer_seg_data,
                        'samples_per_epoch': samples_per_epoch,
                        'precision': precision,
                        'image_size': self.sam_image_size,  # 修正: SAM画像サイズを使用
                        'num_classes_per_sample': num_classes_per_sample,
                        'exclude_val': exclude_val
                    }
                    dataset = self.cache.get('refer_seg', cache_params)
                    if dataset is None:
                        dataset = ReferSegDataset(
                            self.base_image_dir,
                            self.qwen_processor.tokenizer,
                            None,  # vision_tower は使用しない
                            samples_per_epoch,
                            precision,
                            self.sam_image_size,  # 修正: SAM画像サイズを使用
                            num_classes_per_sample,
                            exclude_val,
                            refer_seg_data,
                        )
                        # キャッシュに保存
                        self.cache.set('refer_seg', cache_params, dataset)
                    # キャッシュから読み込んだ場合も追加
                    self.all_datasets.append(dataset)
                else:
                    # キャッシュを使用しない場合
                    self.all_datasets.append(
                        ReferSegDataset(
                            self.base_image_dir,
                            self.qwen_processor.tokenizer,
                            None,  # vision_tower は使用しない
                            samples_per_epoch,
                            precision,
                            self.sam_image_size,  # 修正: SAM画像サイズを使用
                            num_classes_per_sample,
                            exclude_val,
                            refer_seg_data,
                        )
                    )
            except Exception as e:
                print(f"   警告: Referring Segmentationデータセットの初期化に失敗: {e}")
                import traceback
                traceback.print_exc()
        
        # 3. VQA Dataset
        if vqa_data is not None:
            print("💬 VQA データセット:")
            print(f"   設定: {vqa_data}")
            try:
                if self.cache:
                    cache_params = {
                        'dataset_type': 'vqa',
                        'data': vqa_data,
                        'samples_per_epoch': samples_per_epoch,
                        'precision': precision,
                        'image_size': self.sam_image_size,  # 修正: SAM画像サイズを使用（VQAも1024x1024のSAM画像を生成）
                        'exclude_val': exclude_val
                    }
                    dataset = self.cache.get('vqa', cache_params)
                    if dataset is None:
                        dataset = VQADataset(
                            self.base_image_dir,
                            self.qwen_processor.tokenizer,
                            None,  # vision_tower は使用しない
                            samples_per_epoch,
                            precision,
                            self.sam_image_size,  # 修正: SAM画像サイズを使用
                            exclude_val,
                            vqa_data,
                        )
                        # キャッシュに保存
                        self.cache.set('vqa', cache_params, dataset)
                    # キャッシュから読み込んだ場合も追加
                    self.all_datasets.append(dataset)
                else:
                    # キャッシュを使用しない場合
                    self.all_datasets.append(
                        VQADataset(
                            self.base_image_dir,
                            self.qwen_processor.tokenizer,
                            None,  # vision_tower は使用しない
                            samples_per_epoch,
                            precision,
                            self.sam_image_size,  # 修正: SAM画像サイズを使用
                            exclude_val,
                            vqa_data,
                        )
                    )
            except Exception as e:
                print(f"    警告: VQAデータセットの初期化に失敗: {e}")
                import traceback
                traceback.print_exc()
        
        # 4. Reasoning Segmentation
        if reason_seg_data is not None:
            print("🧠 Reasoning Segmentation データセット:")
            print(f"   設定: {reason_seg_data}")
            try:
                if self.cache:
                    cache_params = {
                        'dataset_type': 'reason_seg',
                        'data': reason_seg_data,
                        'samples_per_epoch': samples_per_epoch,
                        'precision': precision,
                        'image_size': self.sam_image_size,  # 修正: SAM画像サイズを使用
                        'num_classes_per_sample': num_classes_per_sample,
                        'exclude_val': exclude_val,
                        'explanatory': explanatory
                    }
                    dataset = self.cache.get('reason_seg', cache_params)
                    if dataset is None:
                        dataset = ReasonSegDataset(
                            base_image_dir=self.base_image_dir,
                            tokenizer=self.qwen_processor.tokenizer,
                            vision_tower=None,  # vision_tower は使用しない
                            samples_per_epoch=samples_per_epoch,
                            precision=precision,
                            image_size=self.sam_image_size,  # 修正: SAM画像サイズを使用
                            num_classes_per_sample=num_classes_per_sample,
                            exclude_val=exclude_val,
                            reason_seg_data=self.reason_seg_data,
                            explanatory=explanatory,
                        )
                        # キャッシュに保存
                        if self.cache:
                            self.cache.set('reason_seg', cache_params, dataset)
                    # キャッシュから読み込んだ場合も追加
                    self.all_datasets.append(dataset)
                else:
                    # キャッシュを使用しない場合
                    self.all_datasets.append(
                        ReasonSegDataset(
                            base_image_dir=self.base_image_dir,
                            tokenizer=self.qwen_processor.tokenizer,
                            vision_tower=None,  # vision_tower は使用しない
                            samples_per_epoch=samples_per_epoch,
                            precision=precision,
                            image_size=self.sam_image_size,  # 修正: SAM画像サイズを使用
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
        else:
            # sample_rateが既に正しい長さの場合も正規化が必要
            total_rate = sum(self.sample_rate)
            if abs(total_rate - 1.0) > 1e-6:  # 浮動小数点の誤差を考慮
                print(f"⚠️  sample_rate正規化: 合計 {total_rate} -> 1.0")
                self.sample_rate = [rate / total_rate for rate in self.sample_rate]
        
        # 最終的なsample_rateを確認
        print(f"📊 最終的なsample_rate: {self.sample_rate} (合計: {sum(self.sample_rate)})")

        # デバッグ用: サンプル追跡用の辞書を作成
        self.debug_sample_info = {}

    def set_sample_rates(self, new_rates):
        """
        サンプル率を動的に変更する（Phase切り替え用）
        O3推奨: Phase 1ではVQAを0にする
        
        Args:
            new_rates: 新しいサンプル率 [SemSeg, ReferSeg, VQA, ReasonSeg]
                      例: Phase 1 = [9, 3, 0, 1], Phase 2 = [9, 3, 3, 1]
        """
        if len(new_rates) != len(self.all_datasets):
            raise ValueError(
                f"サンプル率の長さ({len(new_rates)})がデータセット数({len(self.all_datasets)})と一致しません"
            )
        
        # 正規化
        total_rate = sum(new_rates)
        if total_rate > 0:
            self.sample_rate = [rate / total_rate for rate in new_rates]
        else:
            raise ValueError("サンプル率の合計が0です")
        
        # データセット名とサンプル率を表示
        dataset_names = [type(ds).__name__ for ds in self.all_datasets]
        logger.info("サンプル率を更新:")
        for name, rate in zip(dataset_names, self.sample_rate):
            logger.info(f"  {name}: {rate:.3f}")
    
    def get_current_phase_info(self):
        """現在のPhase情報を取得（デバッグ用）"""
        dataset_names = [type(ds).__name__ for ds in self.all_datasets]
        info = {
            'datasets': dataset_names,
            'sample_rates': self.sample_rate,
            'total_datasets': len(self.all_datasets)
        }
        return info

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx) -> Dict[str, Any]:
        """
        仕様書第3章.2 __getitem__メソッドの処理フロー
        
        デュアルストリーム対応のアイテム取得：
        1. データソースタイプの判定とデータ取得
        2. Qwen2.5-VL用・SAM用のデュアル前処理
        3. apply_chat_templateによるトークン化
        4. 正しいラベルマスキング
        5. 返り値の構築（仕様書準拠、オリジナルLISAとの互換性を保持）
        
        Returns:
            Dict: 以下のキーを含む辞書
                - input_ids: トークン化されたテキスト
                - labels: 言語生成用のラベル（正確にマスク済み）
                - attention_mask: アテンションマスク
                - sam_images: SAM用画像 (C, 1024, 1024)
                - pixel_values: Qwen用画像 (C, 448, 448)
                - ground_truth_mask: グラウンドトゥルースマスク
                - has_mask: マスクの有無
                - seg_token_mask: <SEG>トークンのマスク
                - image_path: 画像パス（オプション）
                - text_prompt: テキストプロンプト（オプション）
                - image_grid_thw: Qwen2.5-VL用のグリッド情報（オプション）
                - resize, questions, sampled_classes: オリジナルLISA互換フィールド
                - original_image: 可視化用の元画像（PIL形式）
                - orig_hw: SAM後処理用の元画像サイズ
        """
        # データセットをサンプリング比率に従って選択
        dataset_idx = np.random.choice(len(self.all_datasets), p=self.sample_rate)
        selected_dataset = self.all_datasets[dataset_idx]
        
        # 選択されたデータセットからランダムにサンプルを取得（idxを無視）
        # オリジナルLISA方式：常にランダムサンプリング
        dataset_size = len(selected_dataset)
        sample_idx = np.random.randint(0, dataset_size)
        sample = selected_dataset[sample_idx]
        
        # デバッグ用: サンプル情報を保存
        self.debug_sample_info[idx] = {
            'dataset_idx': dataset_idx,
            'dataset_name': type(selected_dataset).__name__,
            'sample_idx': sample_idx,
            'dataset_size': dataset_size
        }
        
        # データソースタイプの判定（仕様書第3章.2.1）
        # オリジナルLISAとの互換性：9要素または10要素タプル形式の場合
        if isinstance(sample, tuple) and len(sample) >= 9:
            # SemSegDataset等からの10要素タプル
            image_path = sample[0]
            image_sam = sample[1]          # SAM用前処理済み画像
            image_qwen = sample[2]         # Qwen用前処理済み画像
            conversation_messages = sample[3]  # 会話形式のテキスト
            masks = sample[4]               # マスク
            label = sample[5]               # ラベル
            resize = sample[6]              # リサイズ情報
            questions = sample[7]           # 質問リスト
            sampled_classes = sample[8]    # クラス名リスト
            coord_transform = sample[9] if len(sample) > 9 else None  # 座標変換オブジェクト（10要素の場合のみ）
            
            # デバッグ用: 詳細情報を追加
            self.debug_sample_info[idx]['image_path'] = image_path
            self.debug_sample_info[idx]['has_mask'] = masks is not None
            if conversation_messages and isinstance(conversation_messages, list) and len(conversation_messages) > 0:
                if isinstance(conversation_messages[0], dict) and 'content' in conversation_messages[0]:
                    self.debug_sample_info[idx]['first_message'] = conversation_messages[0]['content'][:100]
            
            # conversation_messagesがList[List[Dict]]形式の場合、最初の要素を使用
            if isinstance(conversation_messages, list) and len(conversation_messages) > 0:
                if isinstance(conversation_messages[0], list):
                    # List[List[Dict]]形式 -> 最初の会話を使用
                    conversation_messages = conversation_messages[0]
            
            # 元画像サイズの取得（coordTransformまたはresizeから）
            if coord_transform is not None and hasattr(coord_transform, 'orig_size'):
                orig_hw = coord_transform.orig_size
            elif resize is not None:
                orig_hw = resize
            else:
                # フォールバック：SAM画像サイズ（1024, 1024）
                orig_hw = (self.sam_image_size, self.sam_image_size)
            
            # 元画像の取得（可視化用）
            # 画像パスから元画像を読み込む
            if image_path and os.path.exists(image_path):
                original_image_pil = Image.open(image_path).convert('RGB')
            else:
                # フォールバック：SAM画像から復元（ただし正規化済み）
                original_image_pil = None
            
            # セグメンテーションタスクかどうか判定
            is_seg_sample = masks is not None and (
                isinstance(masks, (torch.Tensor, np.ndarray)) and 
                (masks.sum() > 0 if hasattr(masks, 'sum') else True)
            )
            
            # messagesがあればtext_promptは不要
            text_prompt = None
            
        # VQADataset等からの4要素タプル
        elif isinstance(sample, tuple) and len(sample) == 4:
            image_path = sample[0]
            image_data = sample[1]  # 画像データ（Tensor/ndarray/PIL）
            text_prompt = sample[2]  # テキストプロンプト
            masks = sample[3] if len(sample) > 3 else None
            label = torch.tensor(0)  # ダミーラベル
            conversation_messages = None
            
            # デバッグ用: 詳細情報を追加
            self.debug_sample_info[idx]['image_path'] = image_path
            self.debug_sample_info[idx]['has_mask'] = masks is not None
            self.debug_sample_info[idx]['text_prompt'] = text_prompt[:100] if text_prompt else None
            
            # セグメンテーションタスクかどうか判定
            is_seg_sample = masks is not None and (
                isinstance(masks, (torch.Tensor, np.ndarray)) and 
                (masks.sum() > 0 if hasattr(masks, 'sum') else True)
            )
            
            # 画像データの処理（仕様書第3章.2.2.1）
            # デュアルストリーム用の前処理
            if isinstance(image_data, torch.Tensor):
                # Tensorの場合
                if image_data.dim() == 3 and image_data.size(0) == 3:
                    # (C, H, W) -> (H, W, C)
                    image_np = image_data.permute(1, 2, 0).cpu().numpy()
                elif image_data.dim() == 2:
                    # グレースケール
                    image_np = image_data.cpu().numpy()
                    image_np = np.stack([image_np] * 3, axis=-1)
                else:
                    image_np = image_data.cpu().numpy()
                
                # データ型の変換
                if image_np.dtype == np.float32 or image_np.dtype == np.float64:
                    if image_np.max() <= 1.0:
                        image_np = (image_np * 255).astype(np.uint8)
                    else:
                        image_np = image_np.astype(np.uint8)
                elif image_np.dtype != np.uint8:
                    image_np = image_np.astype(np.uint8)
                
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
            
            # 元画像を保存（可視化用）
            original_image_pil = image_pil.copy()
            
            # SAM用画像前処理 - SAM2公式Transformsを使用
            # 元画像サイズを保存（後処理で必要）
            orig_hw = (image_pil.height, image_pil.width)
            image_sam = self.sam_transforms(image_pil)
            
            # オリジナルLISAとの互換性のために初期化
            resize = None
            questions = None
            sampled_classes = None
            
            # image_qwenは後でapply_chat_templateで処理されるため、ここではimage_pilを保持
            image_qwen = image_pil
            
        else:
            raise ValueError(f"不明なサンプル形式: {len(sample)} 要素")
        
        # 削除: 元画像サイズを取得（すべてのケースで確実に設定）のフォールバック処理
        # orig_hwは各ケースで既に設定済み

        # conversation_messagesの処理
        if conversation_messages:
            # メッセージ形式から処理する（セグメンテーションタスク）
            # conversation_messagesは既にuser/assistantのroleを持つ
            pass  # 後でmessagesとして使用
        else:
            # text_promptが定義されている場合（VQAなど）
            if 'text_prompt' not in locals():
                text_prompt = "Segment the object in this image. <SEG>"
            # <SEG>トークンが含まれていることを確認
            if self.seg_token not in text_prompt:
                text_prompt += f" {self.seg_token}"

        # PIL画像に変換（apply_chat_templateに必要）
        if isinstance(image_qwen, torch.Tensor):
            # 正規化を元に戻す
            qwen_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
            qwen_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
            image_denorm = image_qwen * qwen_std + qwen_mean
            image_denorm = torch.clamp(image_denorm, 0, 1)
            
            # (C, H, W) -> (H, W, C)
            image_np = image_denorm.permute(1, 2, 0).cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            image_pil_for_qwen = Image.fromarray(image_np)
        else:
            # 既にPIL画像の場合
            image_pil_for_qwen = image_qwen

        # Qwen2.5-VLの正しい処理フロー
        if conversation_messages:
            # メッセージ形式がある場合（セグメンテーションタスク）
            messages = []
            for msg in conversation_messages:
                if msg["role"] == "user":
                    # ユーザーメッセージに画像を追加
                    user_message = {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_pil_for_qwen},
                            {"type": "text", "text": msg["content"]}
                        ]
                    }
                    messages.append(user_message)
                else:
                    # アシスタントメッセージ
                    assistant_message = {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": msg["content"]}
                        ]
                    }
                    messages.append(assistant_message)
        else:
            # text_promptのみの場合（VQAなど）
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_pil_for_qwen},
                        {"type": "text", "text": text_prompt}
                    ]
                }
            ]
        
        # 画像情報を抽出
        image_inputs, video_inputs = process_vision_info(messages)
        
        # テキストテンプレートを生成（学習時はadd_generation_prompt=False）
        text = self.qwen_processor.apply_chat_template(
            messages,
            tokenize=False,  # まずテキストのみ生成
            add_generation_prompt=False  # 学習時はFalse
        )
        
        # processorで画像とテキストを処理
        qwen_processed = self.qwen_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        input_ids = qwen_processed['input_ids'].squeeze(0)
        attention_mask = qwen_processed['attention_mask'].squeeze(0)
        pixel_values = qwen_processed['pixel_values'].squeeze(0)  # 処理済み画像
        
        # 動的解像度モード: image_grid_thwを取得
        if 'image_grid_thw' in qwen_processed and qwen_processed['image_grid_thw'] is not None:
            # apply_chat_template(tokenize=True)で自動的に付与される
            image_grid_thw = qwen_processed['image_grid_thw'].squeeze(0)
        else:
            # フォールバック: pixel_valuesの形状から計算
            if pixel_values.dim() == 2:
                # 2D形式 (N_patches, D_v)の場合
                N_patches = pixel_values.shape[0]
                # O3の回答: H_grid × W_grid × 4 = raw_patch数
                # N_patches = (H_grid × W_grid) (圧縮後のパッチ数)
                # したがって raw_patches = N_patches × 4
                raw_patches = N_patches * 4
                
                # グリッドサイズを推定（正方形に近い形状を仮定）
                import math
                H_grid = W_grid = int(math.sqrt(N_patches))
                # 正確でない場合は調整
                if H_grid * W_grid < N_patches:
                    W_grid = math.ceil(N_patches / H_grid)
                
                image_grid_thw = torch.tensor([1, H_grid, W_grid], dtype=torch.long)
            elif pixel_values.dim() == 3:
                # 3D形式 (C, H, W)の場合（tokenize=Falseの場合）
                _, h, w = pixel_values.shape
                H_grid = h // 14  # パッチサイズ14で割る
                W_grid = w // 14
                image_grid_thw = torch.tensor([1, H_grid, W_grid], dtype=torch.long)
            else:
                # デフォルト値
                image_grid_thw = torch.tensor([1, 32, 32], dtype=torch.long)
        
        # 処理済みのpixel_valuesをimage_qwenとして使用
        image_qwen = pixel_values
        
        # 画像の形状を確認
        if image_qwen.dim() == 4:  # (1, C, H, W) → (C, H, W)
            image_qwen = image_qwen.squeeze(0)
        
        if image_sam.dim() == 4:  # (1, C, H, W) → (C, H, W)
            image_sam = image_sam.squeeze(0)

        # <SEG>トークンの位置を特定（オリジナルLISA準拠）
        seg_token_mask = (input_ids == self.seg_token_idx)

        # ラベルの処理（言語生成用）
        # Qwen2.5-VLチャットテンプレートに準拠した正確なラベルマスキング
        labels = build_correct_labels_for_qwen(input_ids, self.qwen_processor.tokenizer, is_seg_sample)

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
                    # マスクがない場合（VQAなど）
                    ground_truth_mask = None
                    has_mask = False
            
            # すべてのセグメンテーションデータセットは1024x1024のマスクを返すべき
            if has_mask and ground_truth_mask is not None:
                if ground_truth_mask.size(-1) != self.sam_image_size or ground_truth_mask.size(-2) != self.sam_image_size:
                    raise ValueError(
                        f"マスクサイズが不正です: {ground_truth_mask.shape} "
                        f"(期待値: {self.sam_image_size}x{self.sam_image_size})\n"
                        f"データセット: {selected_dataset.__class__.__name__}"
                    )
        else:
            # VQAなどマスクを使用しないタスクの場合
            ground_truth_mask = None

        # 品質スコアの計算（動的解像度モード時）
        quality_score = 1.0  # デフォルト値
        loss_weight = 1.0
        
        if self.use_quality_score and self.use_dynamic_resolution:
            try:
                # 元画像（original_image_pil）を使用して品質計算
                if original_image_pil is not None:
                    quality_score = self.quality_calculator.calculate_quality_score(original_image_pil)
                    loss_weight = self.quality_calculator.get_loss_weight(quality_score)
            except Exception as e:
                # エラー時はデフォルト値を使用
                logger.debug(f"品質スコア計算エラー: {e}")
        
        
        # 返り値の構築（仕様書準拠、オリジナルLISAとの互換性を保持）
        # collate_fnが期待するキー名に統一
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'sam_images': image_sam,      # SAM用画像 (C, 1024, 1024)
            'pixel_values': image_qwen,  # Qwen用画像 (C, 448, 448) - 前処理済み
            'ground_truth_mask': ground_truth_mask if has_mask else None,
            'has_mask': has_mask,
            'seg_token_mask': seg_token_mask,
            'image_path': image_path if 'image_path' in locals() else None,
            'text_prompt': text_prompt if 'text_prompt' in locals() else None,  # 追加: collate_fn用
            'image_grid_thw': image_grid_thw if image_grid_thw is not None else None,  # Qwen2.5-VL用
            # オリジナルLISAとの互換性のための追加フィールド
            'resize': resize if 'resize' in locals() else None,
            'questions': questions if 'questions' in locals() else None,
            'sampled_classes': sampled_classes if 'sampled_classes' in locals() else None,
            'original_image': original_image_pil,  # 可視化用の元画像（PIL形式）
            'orig_hw': orig_hw,  # SAM後処理用の元画像サイズ（必ず設定）
            # 品質スコア関連
            'quality_score': quality_score,
            'loss_weight': loss_weight,
            # デバッグ用
            '_debug_info': self.debug_sample_info.get(idx, {})
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
    original_images = []  # 可視化用の元画像
    orig_hw_list = []  # SAM後処理用の元画像サイズ
    
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
        original_images.append(item.get("original_image"))
        
        # SAM後処理用の元画像サイズを収集
        orig_hw = item.get("orig_hw")
        if orig_hw is not None:
            # タプルまたはリストの形式を統一
            if isinstance(orig_hw, (list, tuple)) and len(orig_hw) == 2:
                orig_hw_list.append(tuple(orig_hw))
            else:
                orig_hw_list.append(None)
        else:
            orig_hw_list.append(None)
    
    # テンソルのスタック
    # pixel_valuesは2D形式 (N_patches, D_v) または3D形式 (C, H, W) の可能性がある
    if pixel_values[0].dim() == 2:
        # 2D形式の場合、パディングしてからスタック
        max_patches = max(pv.shape[0] for pv in pixel_values)
        padded_pixel_values = []
        for pv in pixel_values:
            if pv.shape[0] < max_patches:
                padding = torch.zeros(max_patches - pv.shape[0], pv.shape[1], dtype=pv.dtype)
                pv_padded = torch.cat([pv, padding], dim=0)
            else:
                pv_padded = pv
            padded_pixel_values.append(pv_padded)
        pixel_values = torch.stack(padded_pixel_values)  # (B, N_patches, D_v)
    else:
        # 3D形式の場合はそのままスタック
        pixel_values = torch.stack(pixel_values)  # (B, 3, H, W)
    
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
        "mask_labels": ground_truth_masks_stacked,      # エイリアス: minimal_train.py互換
        "has_mask": has_masks,                          # List[bool]
        "image_paths": image_paths,                     # List[str]
        "text_prompts": text_prompts,                   # List[str]
        "orig_hw": orig_hw_list,                        # List[tuple] - SAM後処理用の元画像サイズ
        # オリジナルLISA互換フィールド
        "masks_list": ground_truth_masks,               # List[Tensor] - オリジナルLISA形式
        "label_list": label_list,                       # List[Tensor] - オリジナルLISA形式
        "resize_list": resize_list,                     # List[Optional[Any]]
        "questions_list": questions_list,               # List[Optional[Any]]
        "sampled_classes_list": sampled_classes_list,   # List[Optional[Any]]
        # 追加の互換性フィールド
        "images": sam_images,                       # エイリアス: オリジナルLISAでの名前
        "images_clip": pixel_values,                # エイリアス: オリジナルLISAでCLIP画像として使用
        "original_images": original_images,          # 可視化用の元画像（PIL形式）
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
        
        # SAM2公式Transformsの初期化
        self.sam_transforms = SAM2Transforms(
            resolution=self.sam_image_size,
            mask_threshold=0.0,
            max_hole_area=0.0,
            max_sprinkle_area=0.0
        )
        
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
        text_prompt = sample[2] if len(sample) > 2 else "Segment the object in this image. <SEG>"
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
        
        # 画像情報を抽出
        image_inputs, video_inputs = process_vision_info(messages)
        
        # テキストテンプレートを生成（評価時はadd_generation_prompt=True）
        text = self.qwen_processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True  # 評価時はTrue
        )
        
        try:
            qwen_processed = self.qwen_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
        except Exception as e:
            raise RuntimeError(f"評価用Qwen前処理に失敗 (idx={idx}): {e}")
        
        pixel_values = qwen_processed['pixel_values'].squeeze(0)
        input_ids = qwen_processed['input_ids'].squeeze(0)
        attention_mask = qwen_processed['attention_mask'].squeeze(0)
        
        try:
            # SAM2公式Transformsを使用（元画像サイズを保持）
            orig_hw = (image.height, image.width)
            sam_images = self.sam_transforms(image)
        except Exception as e:
            raise RuntimeError(f"評価用SAM前処理に失敗 (idx={idx}): {e}")
        
        seg_token_id = self.qwen_processor.tokenizer.convert_tokens_to_ids("[SEG]")
        if seg_token_id is None:
            raise ValueError("<SEG>トークンがトークナイザーに見つかりません")
        
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
