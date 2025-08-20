"""
カスタムcollate関数
異なる長さのシーケンスをバッチ処理するためのパディング処理
"""
import torch
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


def lisa_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    LISA改データセット用のcollate関数
    
    異なる長さのシーケンスをパディングしてバッチ化する
    
    Args:
        batch: データセットから返される辞書のリスト
    
    Returns:
        バッチ化されたテンソルの辞書
    """
    # バッチサイズ
    batch_size = len(batch)
    
    # 各要素を分離
    keys = batch[0].keys()
    collated = {}
    
    for key in keys:
        values = [item[key] for item in batch]
        
        if key in ['input_ids', 'labels', 'attention_mask', 'seg_token_mask']:
            # 1Dテンソル: 最大長にパディング
            if all(isinstance(v, torch.Tensor) for v in values):
                # 最大長を取得
                max_len = max(v.size(0) for v in values)
                
                # パディング値を決定
                if key == 'labels':
                    pad_value = -100  # 損失計算で無視される値
                elif key == 'attention_mask':
                    pad_value = 0
                elif key == 'seg_token_mask':
                    pad_value = False
                else:
                    pad_value = 0  # input_idsのパディング（通常はpad_token_id）
                
                # パディング実行
                padded = []
                for v in values:
                    if v.size(0) < max_len:
                        padding = torch.full((max_len - v.size(0),), pad_value, dtype=v.dtype, device=v.device)
                        padded_v = torch.cat([v, padding], dim=0)
                    else:
                        padded_v = v
                    padded.append(padded_v)
                
                collated[key] = torch.stack(padded, dim=0)
            else:
                collated[key] = values
                
        elif key in ['pixel_values', 'sam_images', 'ground_truth_mask', 'image_grid_thw']:
            # 画像テンソル: そのままスタック（サイズは統一されているはず）
            if all(isinstance(v, torch.Tensor) for v in values):
                try:
                    collated[key] = torch.stack(values, dim=0)
                except RuntimeError as e:
                    # サイズが異なる場合のエラー処理
                    logger.warning(f"⚠️ Cannot stack {key} due to size mismatch: {e}")
                    collated[key] = values  # リストのまま返す
            else:
                collated[key] = values
                
        elif key == 'has_mask':
            # ブール値のリスト
            collated[key] = values
            
        else:
            # その他（文字列、Noneなど）: リストのまま
            collated[key] = values
    
    return collated


def get_collate_fn(tokenizer=None):
    """
    collate関数を取得（必要に応じてトークナイザーを渡せる）
    
    Args:
        tokenizer: パディングトークンIDが必要な場合に使用
    
    Returns:
        collate関数
    """
    if tokenizer is not None:
        # トークナイザーが提供された場合、pad_token_idを使用する版を返す
        def collate_with_tokenizer(batch):
            collated = lisa_collate_fn(batch)
            
            # input_idsのパディングをpad_token_idで置き換え
            if 'input_ids' in collated and hasattr(tokenizer, 'pad_token_id'):
                mask = collated['attention_mask'] == 0
                collated['input_ids'][mask] = tokenizer.pad_token_id
                
            return collated
            
        return collate_with_tokenizer
    else:
        return lisa_collate_fn