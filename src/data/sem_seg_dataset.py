import glob
import json
import os
import random
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.transforms.geometry import SamSquareMeta, apply_sam_transform_to_image, apply_sam_transform_to_mask
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from transformers import CLIPImageProcessor

# from model.segment_anything.utils.transforms import ResizeLongestSide  # 未使用
from . import conversation as conversation_lib
from .constants import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN, LONG_QUESTION_LIST, 
                       SHORT_QUESTION_LIST, SAM_IMAGE_SIZE, SYSTEM_PROMPT)
from .data_processing import get_mask_from_json

# デフォルト会話テンプレート
default_conversation = conversation_lib.default_conversation


def init_mapillary(base_image_dir):
    """Mapillaryデータセットの初期化"""
    mapillary_data_root = os.path.join(base_image_dir, "mapillary")
    
    if not os.path.exists(mapillary_data_root):
        raise FileNotFoundError(f"Mapillaryディレクトリが見つかりません: {mapillary_data_root}")
    
    config_path = os.path.join(mapillary_data_root, "config_v2.0.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Mapillary設定ファイルが見つかりません: {config_path}")
    
    try:
        with open(config_path) as f:
            mapillary_classes = json.load(f)["labels"]
        mapillary_classes = [x["readable"].lower() for x in mapillary_classes]
        mapillary_classes = np.array(mapillary_classes)
        
        labels_dir = os.path.join(mapillary_data_root, "training", "v2.0", "labels")
        if not os.path.exists(labels_dir):
            raise FileNotFoundError(f"Mapillaryラベルディレクトリが見つかりません: {labels_dir}")
            
        mapillary_labels = sorted(glob.glob(os.path.join(labels_dir, "*.png")))
        
        if len(mapillary_labels) == 0:
            raise ValueError("Mapillaryラベルファイルが見つかりません")
        
        mapillary_images = [
            x.replace(".png", ".jpg").replace("v2.0/labels", "images")
            for x in mapillary_labels
        ]
        
        # 存在確認
        valid_images = []
        valid_labels = []
        for img, label in zip(mapillary_images, mapillary_labels):
            if os.path.exists(img) and os.path.exists(label):
                valid_images.append(img)
                valid_labels.append(label)
        
        if len(valid_images) == 0:
            raise ValueError("Mapillaryデータセットの有効な画像がありません")
        
        print(f"mapillary: {len(valid_images)} サンプル")
        return mapillary_classes, valid_images, valid_labels
    except Exception as e:
        raise RuntimeError(f"Mapillary初期化に失敗しました: {e}") from e


def init_ade20k(base_image_dir):
    """ADE20Kデータセットの初期化"""
    ade_path = os.path.join(base_image_dir, "ade20k")
    
    if not os.path.exists(ade_path):
        raise FileNotFoundError(f"ADE20Kディレクトリが見つかりません: {ade_path}")
    
    # クラスリストの読み込み
    classes_file = os.path.join(os.path.dirname(__file__), "ade20k_classes.json")
    if not os.path.exists(classes_file):
        raise FileNotFoundError(f"ADE20Kクラスファイルが見つかりません: {classes_file}")
    
    try:
        with open(classes_file, "r") as f:
            ade20k_classes = json.load(f)
        ade20k_classes = np.array(ade20k_classes)
        
        images_dir = os.path.join(ade_path, "images", "training")
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"ADE20K画像ディレクトリが見つかりません: {images_dir}")
            
        image_ids = sorted(os.listdir(images_dir))
        ade20k_image_ids = []
        for x in image_ids:
            if x.endswith(".jpg"):
                ade20k_image_ids.append(x[:-4])
                
        ade20k_images = []
        for image_id in ade20k_image_ids:
            ade20k_images.append(
                os.path.join(images_dir, "{}.jpg".format(image_id))
            )
            
        ade20k_labels = [
            x.replace(".jpg", ".png").replace("images", "annotations")
            for x in ade20k_images
        ]
        
        # 存在確認
        valid_images = []
        valid_labels = []
        for img, label in zip(ade20k_images, ade20k_labels):
            if os.path.exists(img) and os.path.exists(label):
                valid_images.append(img)
                valid_labels.append(label)
        
        if len(valid_images) == 0:
            raise ValueError("ADE20Kデータセットの有効な画像がありません")
        
        print(f"ade20k: {len(valid_images)} サンプル")
        return ade20k_classes, valid_images, valid_labels
    except Exception as e:
        raise RuntimeError(f"ADE20K初期化に失敗しました: {e}") from e


def init_cocostuff(base_image_dir):
    """COCOStuffデータセットの初期化"""
    classes_file = os.path.join(os.path.dirname(__file__), "cocostuff_classes.txt")
    if not os.path.exists(classes_file):
        raise FileNotFoundError(f"COCOStuffクラスファイルが見つかりません: {classes_file}")
    
    try:
        cocostuff_classes = []
        with open(classes_file) as f:
            for line in f.readlines()[1:]:
                cocostuff_classes.append(line.strip().split(": ")[-1])
        cocostuff_classes = np.array(cocostuff_classes)

        labels_dir = os.path.join(base_image_dir, "cocostuff", "train2017")
        if not os.path.exists(labels_dir):
            raise FileNotFoundError(f"COCOStuffラベルディレクトリが見つかりません: {labels_dir}")

        cocostuff_labels = glob.glob(os.path.join(labels_dir, "*.png"))
        cocostuff_images = [
            x.replace(".png", ".jpg").replace("cocostuff", "coco") for x in cocostuff_labels
        ]

        # 存在確認
        valid_images = []
        valid_labels = []
        for img, label in zip(cocostuff_images, cocostuff_labels):
            if os.path.exists(img) and os.path.exists(label):
                valid_images.append(img)
                valid_labels.append(label)

        if len(valid_images) == 0:
            raise ValueError("COCOStuffデータセットの有効な画像がありません")
        
        print(f"cocostuff: {len(valid_images)} サンプル")
        return cocostuff_classes, valid_images, valid_labels
    except Exception as e:
        raise RuntimeError(f"COCOStuff初期化に失敗しました: {e}") from e


def init_paco_lvis(base_image_dir):
    """PACO LVISデータセットの初期化"""
    annotations_path = os.path.join(base_image_dir, "vlpart", "paco", "annotations", "paco_lvis_v1", "paco_lvis_v1_train.json")
    
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"PACO LVISアノテーションファイルが見つかりません: {annotations_path}")
    
    try:
        coco_api_paco_lvis = COCO(annotations_path)
        all_classes = coco_api_paco_lvis.loadCats(coco_api_paco_lvis.getCatIds())
        class_map_paco_lvis = {}
        for cat in all_classes:
            cat_split = cat["name"].strip().split(":")
            if len(cat_split) == 1:
                name = cat_split[0].split("_(")[0]
            else:
                assert len(cat_split) == 2
                obj, part = cat_split
                obj = obj.split("_(")[0]
                part = part.split("_(")[0]
                name = (obj, part)
            class_map_paco_lvis[cat["id"]] = name
        img_ids = coco_api_paco_lvis.getImgIds()
        
        if len(img_ids) == 0:
            raise ValueError("PACO LVISデータセットに有効な画像がありません")
        
        if len(class_map_paco_lvis) == 0:
            raise ValueError("PACO LVISデータセットに有効なクラスがありません")
        
        print(f"paco_lvis: {len(img_ids)} サンプル")
        return class_map_paco_lvis, img_ids, coco_api_paco_lvis
    except Exception as e:
        raise RuntimeError(f"PACO LVIS初期化に失敗しました: {e}") from e


def init_pascal_part(base_image_dir):
    """Pascal Partデータセットの初期化"""
    annotations_path = os.path.join(base_image_dir, "vlpart", "pascal_part", "train.json")
    
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"Pascal Partアノテーションファイルが見つかりません: {annotations_path}")
    
    try:
        coco_api_pascal_part = COCO(annotations_path)
        all_classes = coco_api_pascal_part.loadCats(coco_api_pascal_part.getCatIds())
        class_map_pascal_part = {}
        
        # Pascal Partの場合、カテゴリ名に":"区切りはないので、オブジェクト名をそのまま使用
        for cat in all_classes:
            cat_name = cat["name"].strip()
            # パート情報がない場合は汎用的なパート名を割り当て
            name = (cat_name, "part")
            class_map_pascal_part[cat["id"]] = name
            
        img_ids = coco_api_pascal_part.getImgIds()
        
        # データの整合性確認
        if len(img_ids) == 0:
            raise ValueError("Pascal Partデータセットに有効な画像がありません")
        
        if len(class_map_pascal_part) == 0:
            raise ValueError("Pascal Partデータセットに有効なクラスがありません")
        
        # 画像ディレクトリ
        image_dir = os.path.join(base_image_dir, "vlpart", "pascal_part", "VOCdevkit", "VOC2010", "JPEGImages")
        
        # 実際に存在するファイルのみをフィルタリング
        valid_img_ids = []
        missing_count = 0
        
        for img_id in img_ids:
            # 画像情報を取得
            image_info = coco_api_pascal_part.loadImgs([img_id])[0]
            file_name = image_info["file_name"]
            image_path = os.path.join(image_dir, file_name)
            
            # ファイルが存在するかチェック
            if not os.path.exists(image_path):
                missing_count += 1
                continue
            
            # アノテーションの有効性確認
            ann_ids = coco_api_pascal_part.getAnnIds(imgIds=[img_id])
            anns = coco_api_pascal_part.loadAnns(ann_ids)
            # セグメンテーション情報があるアノテーションのみ使用
            valid_anns = [ann for ann in anns if ann.get('segmentation') and len(ann['segmentation']) > 0]
            
            if len(valid_anns) > 0:
                valid_img_ids.append(img_id)
        
        if missing_count > 0:
            print(f"警告: Pascal Partデータセットで{missing_count}個のファイルが見つかりません")
        
        if len(valid_img_ids) == 0:
            raise ValueError("Pascal Partデータセットに有効なセグメンテーションアノテーションがありません")
        
        print(f"pascal_part: {len(valid_img_ids)} サンプル（元データ: {len(img_ids)}、欠落: {missing_count}）")
        return class_map_pascal_part, valid_img_ids, coco_api_pascal_part
        
    except Exception as e:
        raise RuntimeError(f"Pascal Part初期化に失敗しました: {e}") from e


class SemSegDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower=None,  # Original-LISA互換性のため
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 1,  # 1会話1マスクに統一
        exclude_val=False,
        sem_seg_data="ade20k||cocostuff||mapillary||pascal_part||paco_lvis",
    ):
        """初期化
        
        Args:
            base_image_dir: ベースとなる画像ディレクトリ
            tokenizer: トークナイザ
            vision_tower: Original-LISA互換性のため
            samples_per_epoch: エポックあたりのサンプル数
            precision: 精度
            image_size: 画像サイズ
            num_classes_per_sample: サンプルあたりのクラス数
            exclude_val: 検証データを除外するか
            sem_seg_data: セマンティックセグメンテーションデータ
        """
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        # self.transform = ResizeLongestSide(image_size)  # 不要：画像処理は別途実装
        # CLIPImageProcessorは使用しないが、互換性のため残す
        # self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        self.data2list = {}
        self.data2classes = {}

        self.sem_seg_datas = sem_seg_data.split("||")
        valid_datasets = []
        
        for ds in self.sem_seg_datas:
            try:
                if ds == "ade20k":
                    classes, images, labels = init_ade20k(base_image_dir)
                elif ds == "cocostuff":
                    classes, images, labels = init_cocostuff(base_image_dir)
                elif ds == "mapillary":
                    classes, images, labels = init_mapillary(base_image_dir)
                elif ds == "paco_lvis":
                    classes, images, labels = init_paco_lvis(base_image_dir)
                elif ds == "pascal_part":
                    classes, images, labels = init_pascal_part(base_image_dir)
                else:
                    print(f"警告: 不明なデータセット: {ds}")
                    continue
                
                if len(images) > 0:
                    self.data2list[ds] = (images, labels)
                    self.data2classes[ds] = classes
                    valid_datasets.append(ds)
                else:
                    print(f"警告: データセット {ds} に有効なデータがありません")
            except Exception as e:
                print(f"データセット {ds} の初期化でエラー: {e}")
                continue

        if len(valid_datasets) == 0:
            raise ValueError("有効なセマンティックセグメンテーションデータセットが見つかりません")

        self.sem_seg_datas = valid_datasets
        print(f"有効なデータセット: {self.sem_seg_datas}")

        if "cocostuff" in self.sem_seg_datas and "cocostuff" in self.data2classes:
            self.cocostuff_class2index = {
                c: i for i, c in enumerate(self.data2classes["cocostuff"])
            }

    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        # データセットをランダムに選択
        if len(self.sem_seg_datas) == 0:
            raise RuntimeError("利用可能なデータセットがありません")

        ds_idx = random.randint(0, len(self.sem_seg_datas) - 1)
        ds = self.sem_seg_datas[ds_idx]

        try:
            if ds in ["paco_lvis", "pascal_part"]:
                return self._get_vlpart_item(ds)
            else:
                return self._get_semseg_item(ds)
        except Exception as e:
            print(f"データセット {ds} からのアイテム取得でエラー: {e}")
            return self.__getitem__(0)

    def _get_vlpart_item(self, ds):
        """VLPartデータセット（paco_lvis, pascal_part）からアイテムを取得"""
        class_map = self.data2classes[ds]
        img_ids, coco_api = self.data2list[ds]
        
        if len(img_ids) == 0:
            return self.__getitem__(0)
            
        idx = random.randint(0, len(img_ids) - 1)
        img_id = img_ids[idx]
        image_info = coco_api.loadImgs([img_id])[0]
        file_name = image_info["file_name"]
        
        if ds == "pascal_part":
            image_path = os.path.join(self.base_image_dir, "vlpart", "pascal_part", "VOCdevkit", "VOC2010", "JPEGImages", file_name)
        else:  # paco_lvis
            image_path = os.path.join(self.base_image_dir, "coco", file_name)

        # 画像の読み込み（エラーハンドリング付き）
        if not os.path.exists(image_path):
            print(f"警告: 画像ファイルが見つかりません: {image_path}")
            return self.__getitem__(0)
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"警告: 画像の読み込みに失敗しました: {image_path}")
            return self.__getitem__(0)
        
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"警告: 画像の色変換に失敗しました: {image_path}, エラー: {e}")
            return self.__getitem__(0)

        # 元画像サイズを記録
        orig_h, orig_w = image.shape[:2]
        
        # デュアルエンコーダ対応: Qwen用とSAM用の画像前処理
        # Qwen用画像前処理（アスペクト比維持・14の倍数パディング）
        h, w = image.shape[:2]
        scale = 448 / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # 縮小時はINTER_AREA、拡大時はINTER_CUBIC
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        image_for_qwen = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
        
        # 14の倍数にパディング
        pad_h = (-new_h) % 14
        pad_w = (-new_w) % 14
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2
        
        image_for_qwen = cv2.copyMakeBorder(
            image_for_qwen, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        
        # テンソル化と正規化（Qwen2.5-VLの正しいパラメータ）
        image_for_qwen = torch.from_numpy(image_for_qwen).permute(2, 0, 1).float() / 255.0
        # 正規化
        qwen_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        qwen_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        image_for_qwen = (image_for_qwen - qwen_mean) / qwen_std
        
        # SAM用画像前処理（1024x1024）- 正しいアスペクト比保持+パディング方式
        image_for_sam, sam_meta = apply_sam_transform_to_image(image)
        
        # テンソル化と正規化（SAM2.1の正しいパラメータ）
        image_for_sam = torch.from_numpy(image_for_sam).permute(2, 0, 1).float() / 255.0
        # SAM2.1の正規化
        sam_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        sam_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_for_sam = (image_for_sam - sam_mean) / sam_std
        resize = image.shape[:2]

        # アノテーションの取得
        ann_ids = coco_api.getAnnIds(imgIds=[img_id])
        anns = coco_api.loadAnns(ann_ids)
        
        if len(anns) == 0:
            return self.__getitem__(0)

        # クラスとマスクの選択
        # 1会話1マスクに統一 - 1つのアノテーションのみを選択
        if len(anns) > 0:
            sampled_anns = [np.random.choice(anns)]
        else:
            sampled_anns = []

        # マスクとクラス名の作成
        masks = []
        sampled_classes = []
        for ann in sampled_anns:
            try:
                mask_data = coco_api.annToMask(ann)
                # マスクもSAMと同じ変換を適用
                mask_sam = apply_sam_transform_to_mask(mask_data, sam_meta, ignore_value=0)
                masks.append(mask_sam)
                class_name = class_map[ann["category_id"]]
                if isinstance(class_name, tuple):
                    obj, part = class_name
                    if random.random() < 0.5:
                        name = obj + " " + part
                    else:
                        name = "the {} of the {}".format(part, obj)
                else:
                    name = str(class_name)
                sampled_classes.append(name)
            except Exception as e:
                # オリジナルと同様に単純な再帰呼び出し
                return self.__getitem__(0)

        if len(masks) == 0:
            return self.__getitem__(0)

        # 会話形式の生成（messages形式）
        questions = []
        answers = []
        for sampled_cls in sampled_classes:
            question_template = random.choice(self.short_question_list)
            question = question_template.format(class_name=sampled_cls.lower())
            questions.append(question)
            answers.append(random.choice(self.answer_list))

        # messages形式でconversationsを生成
        conversations = []
        for i in range(len(questions)):
            messages = [
                {
                    "role": "user",
                    "content": questions[i]
                },
                {
                    "role": "assistant", 
                    "content": answers[i]
                }
            ]
            conversations.append(messages)

        # マスクをテンソルに変換（1会話1マスクなので単一のマスクのみ）
        if len(masks) > 0:
            masks = torch.from_numpy(masks[0]).unsqueeze(0)  # (1, H, W)形式
        else:
            return self.__getitem__(0)
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        # オリジナルLISA準拠の返り値形式（10要素 - sam_meta追加）
        return (
            image_path,        # 0: 画像パス
            image_for_sam,     # 1: SAM用前処理済み画像 (torch.Tensor)
            image_for_qwen,   # 2: Qwen用前処理済み画像 (torch.Tensor)
            conversations,     # 3: 会話形式のテキスト (List[str])
            masks,             # 4: マスク (torch.Tensor) - SAMと同じ変換適用済み
            label,             # 5: ラベル (torch.Tensor)
            resize,            # 6: リサイズ情報 (Tuple)
            questions,         # 7: 質問リスト (List[str])
            sampled_classes,   # 8: クラス名リスト (List[str])
            sam_meta           # 9: SAM変換メタデータ
        )

    def _get_semseg_item(self, ds):
        """セマンティックセグメンテーションデータセット（ade20k, cocostuff, mapillary）からアイテムを取得"""
        images, labels = self.data2list[ds]
        classes = self.data2classes[ds]
        
        if len(images) == 0:
            return self.__getitem__(0)
            
        idx = random.randint(0, len(images) - 1)
        image_path = images[idx]
        label_path = labels[idx]

        # 画像の読み込み（オリジナルのようにエラーチェック最小限）
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 元画像サイズを記録
        orig_h, orig_w = image.shape[:2]
        
        # ラベルの読み込み
        label = Image.open(label_path)
        label = np.array(label)

        # データセット固有の前処理（オリジナルLISA準拠）
        if ds == "ade20k":
            label[label == 0] = 255
            label -= 1
            label[label == 254] = 255
        elif ds == "cocostuff":
            if hasattr(self, 'cocostuff_class2index'):
                for c, i in self.cocostuff_class2index.items():
                    if "-" in c:
                        label[label == i] = 255
        
        # デュアルエンコーダ対応: Qwen用とSAM用の画像前処理
        # Qwen用画像前処理（アスペクト比維持・14の倍数パディング）
        h, w = image.shape[:2]
        scale = 448 / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # 縮小時はINTER_AREA、拡大時はINTER_CUBIC
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        image_for_qwen = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
        
        # 14の倍数にパディング
        pad_h = (-new_h) % 14
        pad_w = (-new_w) % 14
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2
        
        image_for_qwen = cv2.copyMakeBorder(
            image_for_qwen, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        
        # テンソル化と正規化（Qwen2.5-VLの正しいパラメータ）
        image_for_qwen = torch.from_numpy(image_for_qwen).permute(2, 0, 1).float() / 255.0
        # 正規化
        qwen_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        qwen_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        image_for_qwen = (image_for_qwen - qwen_mean) / qwen_std
        
        # SAM用画像前処理（1024x1024）- 正しいアスペクト比保持+パディング方式
        image_for_sam, sam_meta = apply_sam_transform_to_image(image)
        
        # ラベルも画像と同じ変換を適用
        label_for_sam = apply_sam_transform_to_mask(label, sam_meta, ignore_value=255)
        label = label_for_sam  # 以降の処理のためlabelを更新
        
        # テンソル化と正規化（SAM2.1の正しいパラメータ）
        image_for_sam = torch.from_numpy(image_for_sam).permute(2, 0, 1).float() / 255.0
        # SAM2.1の正規化
        sam_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        sam_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_for_sam = (image_for_sam - sam_mean) / sam_std
        resize = image.shape[:2]

        # クラスの選択とマスクの作成
        unique_labels = np.unique(label).tolist()
        if 255 in unique_labels:
            unique_labels.remove(255)
        if len(unique_labels) == 0:
            return self.__getitem__(0)

        classes_list = [classes[class_id] for class_id in unique_labels if class_id < len(classes)]
        if len(classes_list) >= self.num_classes_per_sample:
            sampled_classes = np.random.choice(classes_list, size=self.num_classes_per_sample, replace=False).tolist()
        else:
            sampled_classes = classes_list

        # 会話形式の生成（messages形式）
        questions = []
        answers = []
        for sampled_cls in sampled_classes:
            question_template = random.choice(self.short_question_list)
            questions.append(question_template.format(class_name=sampled_cls.lower()))
            answers.append(random.choice(self.answer_list))

        # messages形式でconversationsを生成
        conversations = []
        for i in range(len(questions)):
            messages = [
                {
                    "role": "user",
                    "content": questions[i]
                },
                {
                    "role": "assistant", 
                    "content": answers[i]
                }
            ]
            conversations.append(messages)

        # マスクの作成（1会話1マスクに統一）
        label_tensor = torch.from_numpy(label).long()
        
        # 最初のクラスのみを使用（num_classes_per_sample=1）
        if len(sampled_classes) > 0:
            sampled_cls = sampled_classes[0]
            try:
                if isinstance(classes, np.ndarray):
                    class_id = np.where(classes == sampled_cls)[0]
                    if len(class_id) > 0:
                        class_id = class_id[0]
                    else:
                        return self.__getitem__(0)
                else:
                    class_id = classes.index(sampled_cls)
            except (ValueError, IndexError):
                return self.__getitem__(0)
            
            # 単一のマスクを作成
            mask = (label_tensor == class_id).float()
            masks = mask.unsqueeze(0)  # (1, H, W)形式に
        else:
            return self.__getitem__(0)

        # オリジナルLISA準拠の返り値形式（10要素 - sam_meta追加）
        return (
            image_path,        # 0: 画像パス
            image_for_sam,     # 1: SAM用前処理済み画像 (torch.Tensor)
            image_for_qwen,   # 2: Qwen用前処理済み画像 (torch.Tensor)
            conversations,     # 3: 会話形式のテキスト (List[str])
            masks,             # 4: マスク (torch.Tensor) - SAMと同じ変換適用済み
            label_tensor,      # 5: ラベル (torch.Tensor)
            resize,            # 6: リサイズ情報 (Tuple)
            questions,         # 7: 質問リスト (List[str])
            sampled_classes,   # 8: クラス名リスト (List[str])
            sam_meta           # 9: SAM変換メタデータ
        )
