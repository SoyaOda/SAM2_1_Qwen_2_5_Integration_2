"""
ラベルマスキング用のユーティリティ関数
LISA型アーキテクチャ用の適切なラベル生成
"""
import torch
from typing import List, Tuple, Optional


IGNORE_INDEX = -100


def find_assistant_spans(
    input_ids: List[int], 
    tokenizer,
    im_start_token: str = "<|im_start|>",
    im_end_token: str = "<|im_end|>",
    assistant_token: str = "assistant"
) -> List[Tuple[int, int]]:
    """
    アシスタントの発話範囲を特定する
    
    Args:
        input_ids: トークンIDのリスト
        tokenizer: トークナイザー
        im_start_token: 開始境界トークン
        im_end_token: 終了境界トークン
        assistant_token: アシスタントロールトークン
    
    Returns:
        アシスタント発話の(開始位置, 終了位置)のリスト
    """
    spans = []
    
    # トークンIDを取得
    im_start_id = tokenizer.convert_tokens_to_ids(im_start_token)
    im_end_id = tokenizer.convert_tokens_to_ids(im_end_token)
    assistant_id = tokenizer.convert_tokens_to_ids(assistant_token)
    
    i = 0
    while i < len(input_ids):
        # "<|im_start|>assistant" パターンを探す
        if i < len(input_ids) - 1 and input_ids[i] == im_start_id and input_ids[i + 1] == assistant_id:
            start = i + 2  # "<|im_start|>assistant" の後から
            
            # 対応する "<|im_end|>" を探す
            j = start
            while j < len(input_ids) and input_ids[j] != im_end_id:
                j += 1
            
            if j < len(input_ids):
                spans.append((start, j))  # im_end_tokenは含まない
                i = j + 1
            else:
                i += 1
        else:
            i += 1
    
    return spans


def find_vision_spans(
    input_ids: List[int],
    tokenizer,
    vision_start_token: str = "<|vision_start|>",
    vision_end_token: str = "<|vision_end|>"
) -> List[Tuple[int, int]]:
    """
    画像トークン範囲を特定する
    
    Args:
        input_ids: トークンIDのリスト
        tokenizer: トークナイザー
        vision_start_token: ビジョン開始トークン
        vision_end_token: ビジョン終了トークン
    
    Returns:
        画像領域の(開始位置, 終了位置)のリスト
    """
    spans = []
    
    # トークンIDを取得
    vision_start_id = tokenizer.convert_tokens_to_ids(vision_start_token)
    vision_end_id = tokenizer.convert_tokens_to_ids(vision_end_token)
    
    i = 0
    while i < len(input_ids):
        if input_ids[i] == vision_start_id:
            start = i
            # 対応する vision_end を探す
            j = i + 1
            while j < len(input_ids) and input_ids[j] != vision_end_id:
                j += 1
            
            if j < len(input_ids):
                spans.append((start, j + 1))  # vision_end_tokenを含む
                i = j + 1
            else:
                i += 1
        else:
            i += 1
    
    return spans


def build_labels_for_qwen_chat(
    input_ids: torch.Tensor,
    tokenizer,
    seg_token: Optional[str] = "<SEG>",
    is_seg_sample: bool = False,
    seg_only_mode: bool = False
) -> torch.Tensor:
    """
    Qwen2.5-VLチャットテンプレートに準拠したラベルマスキング
    
    Args:
        input_ids: トークンID列 [seq_len] or [batch_size, seq_len]
        tokenizer: Qwen2.5-VL用トークナイザー
        seg_token: セグメンテーショントークン
        is_seg_sample: セグメンテーションサンプルかどうか
        seg_only_mode: <SEG>トークンのみを学習対象にするか
    
    Returns:
        マスクされたラベル（-100でマスク）
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    
    batch_size, seq_len = input_ids.shape
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    
    for b in range(batch_size):
        ids_list = input_ids[b].tolist()
        
        # 1. アシスタント発話範囲を特定
        assistant_spans = find_assistant_spans(ids_list, tokenizer)
        
        # 2. アシスタント範囲のラベルを有効化
        for start, end in assistant_spans:
            labels[b, start:end] = input_ids[b, start:end]
        
        # 3. 画像トークン範囲を強制的にマスク
        vision_spans = find_vision_spans(ids_list, tokenizer)
        for start, end in vision_spans:
            labels[b, start:end] = IGNORE_INDEX
        
        # 4. セグメンテーションサンプルの特別処理
        if is_seg_sample and seg_only_mode and seg_token:
            seg_token_id = tokenizer.convert_tokens_to_ids(seg_token)
            
            # <SEG>トークンとその前後のみを学習対象にする
            seg_positions = []
            for i, token_id in enumerate(ids_list):
                if token_id == seg_token_id:
                    seg_positions.append(i)
            
            if seg_positions:
                # 一旦アシスタント範囲を全てマスク
                for start, end in assistant_spans:
                    labels[b, start:end] = IGNORE_INDEX
                
                # <SEG>トークンとその前後1-2トークンのみ有効化
                for pos in seg_positions:
                    # <SEG>トークン自体
                    labels[b, pos] = seg_token_id
                    
                    # 前後の句読点や短い定型句も含める（オプション）
                    # 例: "Sure, it is <SEG>." の場合、"it is <SEG>." を学習
                    context_window = 2
                    for offset in range(1, context_window + 1):
                        if pos - offset >= 0:
                            # アシスタント範囲内かチェック
                            for start, end in assistant_spans:
                                if start <= pos - offset < end:
                                    labels[b, pos - offset] = input_ids[b, pos - offset]
                        if pos + offset < seq_len:
                            # アシスタント範囲内かチェック
                            for start, end in assistant_spans:
                                if start <= pos + offset < end:
                                    labels[b, pos + offset] = input_ids[b, pos + offset]
    
    if labels.shape[0] == 1:
        labels = labels.squeeze(0)
    
    return labels


def create_messages_for_seg_task(
    user_content: str,
    assistant_content: str,
    image_pil=None,
    use_minimal_response: bool = True
) -> List[dict]:
    """
    セグメンテーションタスク用のメッセージを作成
    
    Args:
        user_content: ユーザーの質問
        assistant_content: アシスタントの応答
        image_pil: PIL画像（オプション）
        use_minimal_response: 最小応答形式を使うか
    
    Returns:
        messages形式のリスト
    """
    # LISA形式の最小応答
    if use_minimal_response and "<SEG>" in assistant_content:
        # "Sure, it is <SEG>." などを "<SEG>" のみに簡素化
        assistant_content = "<SEG>"
    
    messages = []
    
    # User message
    user_message = {"role": "user", "content": []}
    if image_pil is not None:
        user_message["content"].append({"type": "image", "image": image_pil})
    user_message["content"].append({"type": "text", "text": user_content})
    messages.append(user_message)
    
    # Assistant message
    assistant_message = {
        "role": "assistant",
        "content": [{"type": "text", "text": assistant_content}]
    }
    messages.append(assistant_message)
    
    return messages