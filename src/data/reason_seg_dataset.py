import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# from model.segment_anything.utils.transforms import ResizeLongestSide  # 未使用
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
        num_classes_per_sample: int = 1,  # 1会話1マスクに統一
        exclude_val=False,
        reason_seg_data="reason_seg/ReasonSeg|train",
        explanatory=0.1,
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        # self.transform = ResizeLongestSide(image_size)  # 不要：画像処理は別途実装

        self.short_question_list = SHORT_QUESTION_LIST
        self.long_question_list = LONG_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        # オリジナルLISA準拠のデータ読み込み
        # reason_seg_dataの形式を修正: "reason_seg/ReasonSeg|train" の場合
        print(f"[DEBUG] reason_seg_data入力: {reason_seg_data}")
        
        if "/" in reason_seg_data:
            # "reason_seg/ReasonSeg|train" -> base_path="reason_seg", dataset_splits="ReasonSeg|train"
            base_path, dataset_splits = reason_seg_data.rsplit("/", 1)
            reason_seg_data_name, splits = dataset_splits.split("|")
        else:
            # 旧形式 "ReasonSeg|train" の場合（互換性のため）
            base_path = "reason_seg"
            reason_seg_data_name, splits = reason_seg_data.split("|")
        
        # self.reason_seg_data_nameには元の値を保存（デバッグ用）
        self.reason_seg_data_name = reason_seg_data
        
        print(f"[DEBUG] base_path: {base_path}, reason_seg_data_name: {reason_seg_data_name}, splits: {splits}")
        
        splits = splits.split("_")
        images = []
        for split in splits:
            # パスを正しく構築: base_image_dir/reason_seg/ReasonSeg/train
            search_path = os.path.join(base_image_dir, base_path, reason_seg_data_name, split, "*.jpg")
            print(f"[DEBUG] 検索パス: {search_path}")
            images_split = glob.glob(search_path)
            print(f"[DEBUG] {split}で見つかった画像数: {len(images_split)}")
            images.extend(images_split)
        jsons = [path.replace(".jpg", ".json") for path in images]
        self.reason_seg_data = (images, jsons)  # オリジナルと同じタプル形式

        print(f"ReasonSegデータセット '{reason_seg_data_name}' ({splits}): {len(images)} サンプル")
        if len(images) == 0:
            print(f"⚠️ 警告: 画像が見つかりません！")
            print(f"  検索ディレクトリ: {os.path.join(base_image_dir, base_path, reason_seg_data_name)}")
            # ディレクトリの存在確認
            check_dir = os.path.join(base_image_dir, base_path, reason_seg_data_name)
            if os.path.exists(check_dir):
                print(f"  ディレクトリは存在します: {check_dir}")
                subdirs = os.listdir(check_dir)
                print(f"  サブディレクトリ: {subdirs}")
            else:
                print(f"  ディレクトリが存在しません: {check_dir}")

        # 説明データの読み込み（オリジナル準拠）
        if explanatory != -1:
            self.explanatory_question_list = EXPLANATORY_QUESTION_LIST
            self.img_to_explanation = {}
            explanatory_path = os.path.join(
                base_image_dir, base_path, reason_seg_data_name, "explanatory", "train.json"
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
        
        # デバッグ: 画像リストが空かチェック
        if len(images) == 0:
            print(f"⚠️ ReasonSegDataset: 画像リストが空です！")
            print(f"  reason_seg_data_name: {self.reason_seg_data_name}")
            print(f"  base_image_dir: {self.base_image_dir}")
            # エラーではなく、ダミーデータを返す
            # これにより学習は継続できる
            dummy_image = np.zeros((512, 512, 3), dtype=np.uint8)
            dummy_mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            
            # ダミーの会話データ
            conversations = [[
                {"role": "user", "content": "What is in this image?"},
                {"role": "assistant", "content": "<SEG>"}
            ]]
            
            # SAM用画像前処理
            scale = self.image_size / max(dummy_image.shape[:2])
            new_h = int(dummy_image.shape[0] * scale)
            new_w = int(dummy_image.shape[1] * scale)
            image_for_sam = cv2.resize(dummy_image, (new_w, new_h))
            
            h, w = image_for_sam.shape[:2]
            pad_h = self.image_size - h
            pad_w = self.image_size - w
            image_for_sam = cv2.copyMakeBorder(
                image_for_sam, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
            resize = image_for_sam.shape[:2]
            image_for_sam = self.preprocess(torch.from_numpy(image_for_sam).permute(2, 0, 1).contiguous())
            
            # Qwen用画像前処理
            scale_qwen = 448 / max(dummy_image.shape[:2])
            new_h_qwen = int(dummy_image.shape[0] * scale_qwen)
            new_w_qwen = int(dummy_image.shape[1] * scale_qwen)
            image_for_qwen = cv2.resize(dummy_image, (new_w_qwen, new_h_qwen))
            
            pad_h_qwen = (-new_h_qwen) % 14
            pad_w_qwen = (-new_w_qwen) % 14
            top, bottom = pad_h_qwen // 2, pad_h_qwen - pad_h_qwen // 2
            left, right = pad_w_qwen // 2, pad_w_qwen - pad_w_qwen // 2
            
            image_for_qwen = cv2.copyMakeBorder(
                image_for_qwen, top, bottom, left, right, 
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
            image_for_qwen = torch.from_numpy(image_for_qwen).permute(2, 0, 1).float() / 255.0
            
            masks = torch.zeros(1, self.image_size, self.image_size)
            label = torch.ones((self.image_size, self.image_size)) * self.ignore_label
            
            return (
                "dummy_path",
                image_for_sam,
                image_for_qwen,
                conversations,
                masks,
                label,
                resize,
                ["What is in this image?"],
                ["dummy"]
            )
        
        idx = random.randint(0, len(images) - 1)
        image_path = images[idx]
        json_path = jsons[idx]

        # 画像の読み込み（オリジナルのようにエラーチェック最小限）
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]

        # デュアルエンコーダ対応: Qwen用とSAM用の画像前処理
        # SAM用画像前処理（1024x1024、アスペクト比維持）
        # ResizeLongestSideの代わりにcv2.resizeを使用
        scale = self.image_size / max(image.shape[:2])
        new_h = int(image.shape[0] * scale)
        new_w = int(image.shape[1] * scale)
        image_for_sam = cv2.resize(image, (new_w, new_h))
        
        # パディングして正方形にする
        h, w = image_for_sam.shape[:2]
        pad_h = self.image_size - h
        pad_w = self.image_size - w
        image_for_sam = cv2.copyMakeBorder(
            image_for_sam, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        
        resize = image_for_sam.shape[:2]
        image_for_sam = self.preprocess(torch.from_numpy(image_for_sam).permute(2, 0, 1).contiguous())
        
        # Qwen用画像前処理（448x448、アスペクト比維持に変更）
        # アスペクト比を維持したリサイズ
        scale_qwen = 448 / max(image.shape[:2])
        new_h_qwen = int(image.shape[0] * scale_qwen)
        new_w_qwen = int(image.shape[1] * scale_qwen)
        image_for_qwen = cv2.resize(image, (new_w_qwen, new_h_qwen))
        
        # 14の倍数にパディング
        pad_h_qwen = (-new_h_qwen) % 14
        pad_w_qwen = (-new_w_qwen) % 14
        top, bottom = pad_h_qwen // 2, pad_h_qwen - pad_h_qwen // 2
        left, right = pad_w_qwen // 2, pad_w_qwen - pad_w_qwen // 2
        
        image_for_qwen = cv2.copyMakeBorder(
            image_for_qwen, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        
        image_for_qwen = torch.from_numpy(image_for_qwen).permute(2, 0, 1).float() / 255.0

        # マスクとテキストの取得（オリジナル準拠）
        mask, sents, is_sentence = get_mask_from_json(json_path, image)
        
        # マスクをSAM画像と同じ変換で処理（修正点）
        if mask is not None:
            # アスペクト比維持リサイズ
            mask_resized = cv2.resize(mask.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            # パディング（SAM画像と同じ）
            mask_padded = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            mask_padded[:h, :w] = mask_resized
            mask = mask_padded
        
        # 1会話1マスクに統一 - 1つの説明のみを選択
        if len(sents) > 0:
            sampled_inds = [np.random.choice(len(sents))]
        else:
            sampled_inds = []
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
        # messages形式でconversationsを生成
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

        # マスクの処理（SAMと同じ座標系に統一済み）
        if len(sampled_masks) > 0:
            masks = np.stack(sampled_masks, axis=0)
            masks = torch.from_numpy(masks)
        else:
            # SAMサイズのゼロマスク
            masks = torch.zeros(1, self.image_size, self.image_size)

        label = torch.ones((self.image_size, self.image_size)) * self.ignore_label

        # オリジナルLISA準拠の返り値形式（9要素）
        return (
            image_path,        # 0: 画像パス
            image_for_sam,     # 1: SAM用前処理済み画像 (torch.Tensor)
            image_for_qwen,   # 2: Qwen用前処理済み画像 (torch.Tensor)
            conversations,     # 3: 会話形式のテキスト (List[str])
            masks,             # 4: マスク (torch.Tensor) - SAMと同じ変換適用済み
            label,             # 5: ラベル (torch.Tensor)
            resize,            # 6: リサイズ情報 (Tuple)
            questions,         # 7: 質問リスト (List[str])
            sampled_sents      # 8: クラス名リスト (List[str])
        )
