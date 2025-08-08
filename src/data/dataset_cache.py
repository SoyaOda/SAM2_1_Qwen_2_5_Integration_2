"""
データセットのキャッシュ機能
初回読み込み後にpickleで保存し、2回目以降は高速読み込み
"""

import os
import pickle
import hashlib
from pathlib import Path
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)

class DatasetCache:
    """データセットのキャッシュ管理"""
    
    def __init__(self, cache_dir: str = ".cache/datasets"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, dataset_name: str, params: dict) -> str:
        """データセットとパラメータからキャッシュキーを生成"""
        # パラメータを文字列化してハッシュ化
        param_str = str(sorted(params.items()))
        hash_obj = hashlib.md5(f"{dataset_name}_{param_str}".encode())
        return hash_obj.hexdigest()
    
    def get(self, dataset_name: str, params: dict) -> Optional[Any]:
        """キャッシュからデータセットを取得"""
        cache_key = self._get_cache_key(dataset_name, params)
        cache_file = self.cache_dir / f"{dataset_name}_{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                logger.info(f"キャッシュから {dataset_name} を読み込み中...")
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                logger.info(f"✅ キャッシュから {dataset_name} を高速読み込み完了")
                return data
            except Exception as e:
                logger.warning(f"キャッシュ読み込みエラー: {e}")
                # キャッシュが壊れている場合は削除
                cache_file.unlink(missing_ok=True)
        
        return None
    
    def set(self, dataset_name: str, params: dict, data: Any):
        """データセットをキャッシュに保存"""
        cache_key = self._get_cache_key(dataset_name, params)
        cache_file = self.cache_dir / f"{dataset_name}_{cache_key}.pkl"
        
        try:
            logger.info(f"{dataset_name} をキャッシュに保存中...")
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"✅ {dataset_name} のキャッシュ保存完了")
        except Exception as e:
            logger.warning(f"キャッシュ保存エラー: {e}")
            # 保存に失敗した場合はファイルを削除
            cache_file.unlink(missing_ok=True)
    
    def clear(self):
        """すべてのキャッシュをクリア"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        logger.info("すべてのキャッシュをクリアしました")