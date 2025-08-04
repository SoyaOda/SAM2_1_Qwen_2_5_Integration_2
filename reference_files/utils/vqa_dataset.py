import json
import os
import random

import cv2
import torch
import torch.nn.functional as F
from PIL import Image

from model.segment_anything.utils.transforms import ResizeLongestSide
from . import conversation as conversation_lib
from .constants import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN, LONG_QUESTION_LIST, 
                       SHORT_QUESTION_LIST, SAM_IMAGE_SIZE, SAM_PIXEL_MEAN, 
                       SAM_PIXEL_STD, DEFAULT_IGNORE_LABEL)

# 簡単な会話クラス（LLaVAの代替）
class SimpleConversation:
    def __init__(self):
        self.roles = ["USER", "ASSISTANT"]
        self.messages = []
        self.sep = "\n"
        self.sep2 = "</s>"
    
    def copy(self):
        new_conv = SimpleConversation()
        new_conv.messages = self.messages.copy()
        return new_conv
    
    def append_message(self, role, message):
        self.messages.append([role, message])
    
    def get_prompt(self):
        if len(self.messages) == 0:
            return ""
        
        prompt = ""
        for i, (role, message) in enumerate(self.messages):
            if i == 0:
                prompt += f"{role}: {message}"
            else:
                prompt += f"{self.sep}{role}: {message}"
        
        return prompt + self.sep2

# デフォルト会話インスタンス
default_conversation = SimpleConversation()


def preprocess_multimodal(source, mm_use_im_start_end=False):
    """Gemma-3用の簡略化されたマルチモーダル前処理"""
    for sentence in source:
        if DEFAULT_IMAGE_TOKEN in sentence["value"]:
            sentence["value"] = (
                sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
            )
            sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
            sentence["value"] = sentence["value"].strip()
    return source


class VQADataset(torch.utils.data.Dataset):
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
        vqa_data="llava_instruct_150k",
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        
        # Gemma-3では画像変換を簡略化
        self.target_size = image_size

        DATA_DIR = os.path.join(base_image_dir, "llava_dataset")
        self.vqa_image_root = os.path.join(base_image_dir, "coco/train2017")
        
        # VQAデータの解析（複数データセット対応）
        self.vqa_datasets = vqa_data.split("||") if "||" in vqa_data else [vqa_data]
        self.vqa_data = []
        
        for dataset_name in self.vqa_datasets:
            vqa_json_path = os.path.join(DATA_DIR, "{}.json".format(dataset_name))
            if not os.path.exists(vqa_json_path):
                print(f"警告: VQAデータファイルが見つかりません: {vqa_json_path}")
                continue
            
            try:
                with open(vqa_json_path) as f:
                    dataset_data = json.load(f)
                
                if len(dataset_data) == 0:
                    print(f"警告: VQAデータが空です: {vqa_json_path}")
                    continue
                
                self.vqa_data.extend(dataset_data)
                print(f"VQAデータセット '{dataset_name}': {len(dataset_data)} サンプル")
            except Exception as e:
                print(f"警告: VQAデータセット '{dataset_name}' の読み込みに失敗: {e}")
                continue

        if len(self.vqa_data) == 0:
            raise ValueError("有効なVQAデータセットが見つかりません")

        print(f"VQA総サンプル数: {len(self.vqa_data)}")

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
        idx = random.randint(0, len(self.vqa_data) - 1)
        item = self.vqa_data[idx]
        image_path = os.path.join(self.vqa_image_root, item["image"])
        
        # オリジナルのようにエラーチェックなしで直接読み込み
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]

        # デュアルエンコーダ対応: Gemma用とSAM用の画像前処理
        # Gemma用画像前処理（896x896）
        image_for_gemma = cv2.resize(image, (896, 896))
        image_for_gemma = torch.from_numpy(image_for_gemma).permute(2, 0, 1).float() / 255.0
        
        # SAM用画像前処理（1024x1024）
        image_for_sam = cv2.resize(image, (self.img_size, self.img_size))
        image_for_sam = torch.from_numpy(image_for_sam).permute(2, 0, 1).float() / 255.0
        image_for_sam = self.preprocess(image_for_sam)

        conv = conversation_lib.default_conversation.copy()
        source = item["conversations"]
        source = preprocess_multimodal(source, mm_use_im_start_end=False)
        
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        conversations = []
        
        if len(source) > 0 and roles.get(source[0]["from"]) != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles.get(sentence["from"], conv.roles[j % 2])
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

        # VQAデータセットではマスクは不要（オリジナル準拠）
        # 空のマスクではなく、適切なサイズのゼロマスクを作成
        masks = torch.zeros(1, *ori_size)  # (1, H, W) 形式のゼロマスク
        label = torch.ones(ori_size) * self.ignore_label

        # 質問と回答の抽出（Original-LISA-Code準拠）
        questions = conversations  # オリジナルと同様
        sampled_classes = conversations  # オリジナルと同様

        # オリジナルLISA準拠の返り値形式（9要素）
        return (
            image_path,        # 0: 画像パス
            image_for_sam,     # 1: SAM用前処理済み画像 (torch.Tensor)
            image_for_gemma,   # 2: Gemma用前処理済み画像 (torch.Tensor)
            conversations,     # 3: 会話形式のテキスト (List[str])
            masks,             # 4: マスク (torch.Tensor)
            label,             # 5: ラベル (torch.Tensor)
            ori_size,          # 6: リサイズ情報 (Tuple)
            questions,         # 7: 質問リスト (List[str])
            sampled_classes    # 8: クラス名リスト (List[str])
        )
