import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model.segment_anything.utils.transforms import ResizeLongestSide
from . import conversation as conversation_lib
from .constants import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN, EXPLANATORY_QUESTION_LIST,
                       LONG_QUESTION_LIST, SHORT_QUESTION_LIST, SAM_IMAGE_SIZE, 
                       SAM_PIXEL_MEAN, SAM_PIXEL_STD, DEFAULT_IGNORE_LABEL, DEFAULT_SEG_TOKEN)
from .data_processing import get_mask_from_json


class ReasonSegDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor(SAM_PIXEL_MEAN).view(-1, 1, 1)
    pixel_std = torch.Tensor(SAM_PIXEL_STD).view(-1, 1, 1)
    img_size = SAM_IMAGE_SIZE
    ignore_label = DEFAULT_IGNORE_LABEL

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower=None,  # Original-LISA互換性のため
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "bf16",
        image_size: int = SAM_IMAGE_SIZE,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        reason_seg_data="ReasonSeg|train",
        explanatory=0.1,
    ):
        self.exclude_val = exclude_val
        self.reason_seg_data_name = reason_seg_data
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)

        self.short_question_list = SHORT_QUESTION_LIST
        self.long_question_list = LONG_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        # オリジナルLISA準拠のデータ読み込み
        reason_seg_data_name, splits = reason_seg_data.split("|")
        splits = splits.split("_")
        images = []
        for split in splits:
            images_split = glob.glob(
                os.path.join(base_image_dir, "reason_seg", reason_seg_data_name, split, "*.jpg")
            )
            images.extend(images_split)
        jsons = [path.replace(".jpg", ".json") for path in images]
        self.reason_seg_data = (images, jsons)  # オリジナルと同じタプル形式

        print(f"ReasonSegデータセット '{reason_seg_data_name}' ({splits}): {len(images)} サンプル")

        # 説明データの読み込み（オリジナル準拠）
        if explanatory != -1:
            self.explanatory_question_list = EXPLANATORY_QUESTION_LIST
            self.img_to_explanation = {}
            explanatory_path = os.path.join(
                base_image_dir, "reason_seg", reason_seg_data_name, "explanatory", "train.json"
            )
            if os.path.exists(explanatory_path):
                with open(explanatory_path) as f:
                    items = json.load(f)
                for item in items:
                    img_name = item["image"]
                    self.img_to_explanation[img_name] = {
                        "query": item["query"],
                        "outputs": item["outputs"],
                    }
                print(f"explanatory '{reason_seg_data_name}': {len(self.img_to_explanation)} 説明")
            else:
                print(f"説明ファイルが見つかりません: {explanatory_path}")

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
        images, jsons = self.reason_seg_data
        idx = random.randint(0, len(images) - 1)
        image_path = images[idx]
        json_path = jsons[idx]

        # 画像の読み込み（オリジナルのようにエラーチェック最小限）
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]

        # デュアルエンコーダ対応: Gemma用とSAM用の画像前処理
        # Gemma用画像前処理（896x896）
        image_for_gemma = cv2.resize(image, (896, 896))
        image_for_gemma = torch.from_numpy(image_for_gemma).permute(2, 0, 1).float() / 255.0
        
        # SAM用画像前処理（1024x1024）
        image_for_sam = self.transform.apply_image(image)
        resize = image_for_sam.shape[:2]
        image_for_sam = self.preprocess(torch.from_numpy(image_for_sam).permute(2, 0, 1).contiguous())

        # マスクとテキストの取得（オリジナル準拠）
        mask, sents, is_sentence = get_mask_from_json(json_path, image)
        if len(sents) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(
                list(range(len(sents))), size=self.num_classes_per_sample, replace=False
            )
        else:
            sampled_inds = list(range(len(sents)))
        sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
        sampled_masks = [
            (mask == 1).astype(np.float32) for _ in range(len(sampled_inds))
        ]

        # 説明付き回答の処理（オリジナル準拠）
        image_name = image_path.split("/")[-1]
        choice = 0  # デフォルトは[SEG]トークンのみ
        if self.explanatory != -1 and image_name in self.img_to_explanation:
            if random.random() < self.explanatory:
                choice = 2  # 説明付き回答
            else:
                choice = random.randint(0, 1)  # [SEG]トークンまたは[SEG]+説明

        questions = []
        answers = []
        for text in sampled_sents:
            if is_sentence:
                question_template = random.choice(self.long_question_list)
                questions.append(question_template.format(sent=text))
            else:
                question_template = random.choice(self.short_question_list)
                questions.append(question_template.format(class_name=text.lower()))

            # 説明の追加（オリジナル準拠）
            if self.explanatory != -1 and image_name in self.img_to_explanation:
                if choice == 0:  # [SEG] token
                    answers.append(random.choice(self.answer_list))
                elif choice == 1:  # [SEG] token + text answer
                    answer = self.img_to_explanation[image_name]["outputs"]
                    answer = random.choice(self.answer_list) + " {}".format(answer)
                    questions[-1] = DEFAULT_IMAGE_TOKEN + "\n" + text + " {}".format(
                        random.choice(self.explanatory_question_list)
                    )
                    answers.append(answer)
                elif choice == 2:  # vanilla text answer
                    answer = self.img_to_explanation[image_name]["outputs"]
                    questions[-1] = DEFAULT_IMAGE_TOKEN + "\n" + text
                    answers.append(answer)
                else:
                    raise ValueError("Not implemented yet.")
            else:
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

        # マスクの処理
        if len(sampled_masks) > 0:
            masks = np.stack(sampled_masks, axis=0)
            masks = torch.from_numpy(masks)
        else:
            masks = torch.zeros(1, *ori_size)

        label = torch.ones(ori_size) * self.ignore_label

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
            sampled_sents      # 8: クラス名リスト (List[str])
        )
