import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask

from model.segment_anything.utils.transforms import ResizeLongestSide
from . import conversation as conversation_lib
from .constants import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN, LONG_QUESTION_LIST, 
                       SHORT_QUESTION_LIST, SAM_IMAGE_SIZE, SYSTEM_PROMPT)
from .grefer import G_REFER
from .refer import REFER

# デフォルト会話テンプレート
default_conversation = conversation_lib.default_conversation.copy()


class ReferSegDataset(torch.utils.data.Dataset):
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
        num_classes_per_sample: int = 3,
        exclude_val=False,
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
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
            refer_seg_data: 参照セグメンテーションデータ
        """
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        # CLIPImageProcessorは使用しないが、互換性のため残す
        # self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        # データセットの存在確認
        DATA_DIR = os.path.join(base_image_dir, "refer_seg")
        if not os.path.exists(DATA_DIR):
            raise FileNotFoundError(f"参照セグメンテーションデータディレクトリが見つかりません: {DATA_DIR}")

        self.refer_seg_ds_list = refer_seg_data.split("||")
        self.refer_seg_data = {}
        
        # 各データセットの初期化
        for ds in self.refer_seg_ds_list:
            if ds == "refcocog":
                splitBy = "umd"
            else:
                splitBy = "unc"

            if ds == "grefcoco":
                refer_api = G_REFER(DATA_DIR, ds, splitBy)
            else:
                refer_api = REFER(DATA_DIR, ds, splitBy)
            
            ref_ids_train = refer_api.getRefIds(split="train")
            if len(ref_ids_train) == 0:
                print(f"警告: データセット {ds} にトレーニングデータが見つかりません")
                continue
                
            images_ids_train = refer_api.getImgIds(ref_ids=ref_ids_train)
            refs_train = refer_api.loadRefs(ref_ids=ref_ids_train)

            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_train)

            # 画像パスの構築と存在確認
            valid_images = []
            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        DATA_DIR, "images/saiapr_tc-12", item["file_name"]
                    )
                else:
                    item["file_name"] = os.path.join(
                        DATA_DIR, "images/mscoco/images/train2014", item["file_name"]
                    )
                
                # ファイルの存在確認
                if os.path.exists(item["file_name"]):
                    valid_images.append(item)
                else:
                    print(f"警告: 画像ファイルが見つかりません: {item['file_name']}")
            
            if len(valid_images) == 0:
                print(f"警告: データセット {ds} に有効な画像が見つかりません")
                continue
                
            refer_seg_ds["images"] = valid_images
            refer_seg_ds["annotations"] = refer_api.Anns

            print(f"データセット {ds} ({splitBy}): {len(refer_seg_ds['images'])} サンプル, {len(refer_seg_ds['annotations'])} アノテーション")

            img2refs = {}
            for ref in refs_train:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [ref]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_data[ds] = refer_seg_ds

        # 有効なデータセットがあるかチェック
        if len(self.refer_seg_data) == 0:
            raise ValueError("有効な参照セグメンテーションデータセットが見つかりません")

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
        ds = random.randint(0, len(self.refer_seg_ds_list) - 1)
        ds = self.refer_seg_ds_list[ds]
        refer_seg_ds = self.refer_seg_data[ds]
        images = refer_seg_ds["images"]
        annotations = refer_seg_ds["annotations"]
        img2refs = refer_seg_ds["img2refs"]
        
        idx = random.randint(0, len(images) - 1)
        image_info = images[idx]
        image_path = image_info["file_name"]
        image_id = image_info["id"]
        refs = img2refs[image_id]
        
        if len(refs) == 0:
            return self.__getitem__(0)

        # 文章とアノテーションIDの収集
        sents = []
        ann_ids = []
        for ref in refs:
            for sent in ref["sentences"]:
                text = sent["sent"]
                sents.append(text)
                ann_ids.append(ref["ann_id"])

        if len(sents) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(list(range(len(sents))), size=self.num_classes_per_sample, replace=False)
        else:
            sampled_inds = list(range(len(sents)))

        sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
        sampled_ann_ids = [ann_ids[ind] for ind in sampled_inds]
        sampled_classes = sampled_sents

        # 画像の読み込み（オリジナルのようにエラーチェック最小限）
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # デュアルエンコーダ対応: Gemma用とSAM用の画像前処理
        # Gemma用画像前処理（896x896）
        image_for_gemma = cv2.resize(image, (896, 896))
        image_for_gemma = torch.from_numpy(image_for_gemma).permute(2, 0, 1).float() / 255.0

        # SAM用の前処理
        image_for_sam = self.transform.apply_image(image)
        resize = image_for_sam.shape[:2]
        
        # テンソル前処理
        image_for_sam = self.preprocess(torch.from_numpy(image_for_sam).permute(2, 0, 1).contiguous())

        # 質問と回答の生成
        questions = []
        answers = []
        for i, text in enumerate(sampled_classes):
            text = text.strip()
            assert len(text.split("||")) == 1
            question_template = random.choice(self.short_question_list)
            # 最初の質問にのみ画像トークンを含める
            if i == 0 and DEFAULT_IMAGE_TOKEN not in question_template:
                question = DEFAULT_IMAGE_TOKEN + "\n" + question_template.format(class_name=text.lower())
            else:
                question = question_template.format(class_name=text.lower())
            questions.append(question)
            answers.append(random.choice(self.answer_list))

        # 会話形式の生成（オリジナルLISA準拠）
        conversations = []
        conv = conversation_lib.default_conversation.copy()

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        # マスクの生成（オリジナルLISA準拠）
        masks = []
        for ann_id in sampled_ann_ids:
            if isinstance(ann_id, list):
                if -1 in ann_id:
                    assert len(ann_id) == 1
                    m = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
                else:
                    m_final = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
                    for ann_id_i in ann_id:
                        ann = annotations[ann_id_i]
                        if len(ann["segmentation"]) == 0:
                            m = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
                        else:
                            if type(ann["segmentation"][0]) == list:  # polygon
                                rle = mask.frPyObjects(ann["segmentation"], image_info["height"], image_info["width"])
                            else:
                                rle = ann["segmentation"]
                                for i in range(len(rle)):
                                    if not isinstance(rle[i]["counts"], bytes):
                                        rle[i]["counts"] = rle[i]["counts"].encode()
                            m = mask.decode(rle)
                            m = np.sum(m, axis=2)
                            m = m.astype(np.uint8)
                        m_final = m_final | m
                    m = m_final
                masks.append(m)
                continue

            ann = annotations[ann_id]
            if len(ann["segmentation"]) == 0:
                m = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
                masks.append(m)
                continue

            if type(ann["segmentation"][0]) == list:  # polygon
                rle = mask.frPyObjects(ann["segmentation"], image_info["height"], image_info["width"])
            else:
                rle = ann["segmentation"]
                for i in range(len(rle)):
                    if not isinstance(rle[i]["counts"], bytes):
                        rle[i]["counts"] = rle[i]["counts"].encode()
            m = mask.decode(rle)
            m = np.sum(m, axis=2)
            m = m.astype(np.uint8)
            masks.append(m)

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        # オリジナルLISA準拠の返り値形式（9要素）
        return (
            image_path,        # 0: 画像パス
            image_for_sam,     # 1: SAM用前処理済み画像 (torch.Tensor)
            image_for_gemma,   # 2: Gemma用前処理済み画像 (torch.Tensor)
            conversations,     # 3: 会話形式のテキスト (List[str])
            masks,             # 4: マスク (torch.Tensor)
            label,             # 5: ラベル (torch.Tensor)
            resize,            # 6: リサイズ情報 (Tuple)
            questions,         # 7: 質問リスト (List[str])
            sampled_classes    # 8: クラス名リスト (List[str])
        )
