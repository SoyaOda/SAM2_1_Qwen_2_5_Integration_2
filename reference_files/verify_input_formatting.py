#!/usr/bin/env python3
"""
第2節：Gemma向けマルチモーダル入力フォーマットの検証
全データセット・全サブタイプ対応版

論理的根拠:
- データが正しく準備されても、モデルが解釈できる形式に変換する過程でエラーが発生すれば学習は失敗
- Gemmaの特殊トークン配置、ラベルマスキング、バッチ処理の検証が必要
- 各データセットのサブタイプごとに異なる処理ロジックを検証する必要がある
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor

# プロジェクトのルートディレクトリをsys.pathに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.dataset import HybridDataset, get_config, collate_fn

def get_all_dataset_configs() -> List[Dict[str, Any]]:
    """
    全データセット・全サブタイプの設定を取得
    verify_dataset_integrity.pyを参考にした完全な設定
    """
    config = get_config()
    
    # 設定ファイルからサブデータセット情報を取得
    dataset_structure = getattr(config, 'DATASET_STRUCTURE', {})
    
    configs = []
    
    # Semantic Segmentation - 複数のサブタイプ
    if 'sem_seg' in dataset_structure:
        sem_seg_datasets = dataset_structure['sem_seg']
        for sub_dataset in sem_seg_datasets.keys():
            configs.append({
                'name': f'sem_seg_{sub_dataset}',
                'dataset_type': 'sem_seg',
                'sub_dataset': sub_dataset,
                'config_attr': 'SEM_SEG_DATA',
                'default_value': sub_dataset
            })
    else:
        # フォールバック
        configs.append({
            'name': 'sem_seg_ade20k',
            'dataset_type': 'sem_seg',
            'sub_dataset': 'ade20k',
            'config_attr': 'SEM_SEG_DATA',
            'default_value': 'ade20k'
        })
    
    # Referring Expression Segmentation - 複数のサブタイプ
    if 'refer_seg' in dataset_structure and 'datasets' in dataset_structure['refer_seg']:
        refer_seg_datasets = dataset_structure['refer_seg']['datasets']
        for sub_dataset in refer_seg_datasets.keys():
            configs.append({
                'name': f'refer_seg_{sub_dataset}',
                'dataset_type': 'refer_seg',
                'sub_dataset': sub_dataset,
                'config_attr': 'REFER_SEG_DATA',
                'default_value': sub_dataset
            })
    else:
        # フォールバック
        for sub_dataset in ['refcoco', 'refcoco+', 'refcocog']:
            configs.append({
                'name': f'refer_seg_{sub_dataset}',
                'dataset_type': 'refer_seg',
                'sub_dataset': sub_dataset,
                'config_attr': 'REFER_SEG_DATA',
                'default_value': sub_dataset
            })
    
    # VQA - 複数のサブタイプ
    if 'vqa' in dataset_structure:
        vqa_datasets = dataset_structure['vqa']
        for sub_dataset in vqa_datasets.keys():
            configs.append({
                'name': f'vqa_{sub_dataset}',
                'dataset_type': 'vqa',
                'sub_dataset': sub_dataset,
                'config_attr': 'VQA_DATA',
                'default_value': sub_dataset
            })
    else:
        # フォールバック
        configs.append({
            'name': 'vqa_llava_instruct_150k',
            'dataset_type': 'vqa',
            'sub_dataset': 'llava_instruct_150k',
            'config_attr': 'VQA_DATA',
            'default_value': 'llava_instruct_150k'
        })
    
    # Reasoning Segmentation - 複数のサブタイプ
    if 'reason_seg' in dataset_structure:
        reason_seg_datasets = dataset_structure['reason_seg']
        for sub_dataset in reason_seg_datasets.keys():
            # train/val分割を考慮
            for split in ['train', 'val']:
                configs.append({
                    'name': f'reason_seg_{sub_dataset}_{split}',
                    'dataset_type': 'reason_seg',
                    'sub_dataset': f'{sub_dataset}|{split}',
                    'config_attr': 'REASON_SEG_DATA',
                    'default_value': f'{sub_dataset}|{split}'
                })
    else:
        # フォールバック
        configs.append({
            'name': 'reason_seg_ReasonSeg_train',
            'dataset_type': 'reason_seg',
            'sub_dataset': 'ReasonSeg|train',
            'config_attr': 'REASON_SEG_DATA',
            'default_value': 'ReasonSeg|train'
        })
    
    return configs

def analyze_sample_tokens(input_ids: torch.Tensor, labels: torch.Tensor, 
                         processor, sample_idx: int = 0) -> Dict[str, Any]:
    """
    サンプルのトークン分析（詳細版）
    """
    input_list = input_ids.tolist()
    labels_list = labels.tolist()
    
    # 基本統計
    seq_len = len(input_list)
    non_pad_tokens = sum(1 for token_id in input_list if token_id != processor.tokenizer.pad_token_id)
    labeled_tokens = sum(1 for label in labels_list if label != -100)
    
    # 特殊トークン位置の検出
    special_tokens = {}
    seg_token_id = getattr(processor.tokenizer, 'seg_token_id', None)
    if seg_token_id is not None:
        seg_positions = [i for i, token_id in enumerate(input_list) if token_id == seg_token_id]
        special_tokens['seg_positions'] = seg_positions
    
    # ラベルマスキング分析
    user_turn_tokens = 0
    model_turn_tokens = 0
    
    # Gemma-3のターン開始/終了トークンを検出
    start_of_turn = processor.tokenizer.convert_tokens_to_ids("<start_of_turn>")
    end_of_turn = processor.tokenizer.convert_tokens_to_ids("<end_of_turn>")
    
    current_turn = None
    for i, (input_id, label) in enumerate(zip(input_list, labels_list)):
        if input_id == start_of_turn and i + 1 < len(input_list):
            # 次のトークンでターンの種類を判定
            next_token = input_list[i + 1]
            user_token = processor.tokenizer.convert_tokens_to_ids("user")
            model_token = processor.tokenizer.convert_tokens_to_ids("model")
            
            if next_token == user_token:
                current_turn = "user"
            elif next_token == model_token:
                current_turn = "model"
        
        if current_turn == "user":
            user_turn_tokens += 1
        elif current_turn == "model":
            model_turn_tokens += 1
    
    return {
        'sample_idx': sample_idx,
        'sequence_length': seq_len,
        'non_pad_tokens': non_pad_tokens,
        'labeled_tokens': labeled_tokens,
        'user_turn_tokens': user_turn_tokens,
        'model_turn_tokens': model_turn_tokens,
        'special_tokens': special_tokens,
        'label_masking_ratio': labeled_tokens / non_pad_tokens if non_pad_tokens > 0 else 0
    }

def verify_label_masking(input_ids: torch.Tensor, labels: torch.Tensor, 
                        processor, sample_idx: int = 0) -> Tuple[bool, Dict[str, Any]]:
    """
    ラベルマスキングの正確性を検証
    """
    input_list = input_ids.tolist()
    labels_list = labels.tolist()
    
    # Gemma-3のターン制御トークン
    start_of_turn = processor.tokenizer.convert_tokens_to_ids("<start_of_turn>")
    end_of_turn = processor.tokenizer.convert_tokens_to_ids("<end_of_turn>")
    user_token = processor.tokenizer.convert_tokens_to_ids("user")
    model_token = processor.tokenizer.convert_tokens_to_ids("model")
    
    errors = []
    current_turn = None
    turn_start_pos = None
    
    for i, (input_id, label) in enumerate(zip(input_list, labels_list)):
        # PADトークンはスキップ
        if input_id == processor.tokenizer.pad_token_id:
            continue
            
        # ターン開始の検出
        if input_id == start_of_turn and i + 1 < len(input_list):
            next_token = input_list[i + 1]
            if next_token == user_token:
                current_turn = "user"
                turn_start_pos = i
            elif next_token == model_token:
                current_turn = "model"
                turn_start_pos = i
        
        # ターン終了の検出
        elif input_id == end_of_turn:
            current_turn = None
            turn_start_pos = None
        
        # ラベルマスキングの検証
        if current_turn == "user":
            # userターンは全て-100でマスクされるべき
            if label != -100:
                token_text = processor.tokenizer.decode([input_id])
                errors.append({
                    'position': i,
                    'turn': current_turn,
                    'token': token_text,
                    'input_id': input_id,
                    'label': label,
                    'expected_label': -100,
                    'error': 'User turn should be masked'
                })
        
        elif current_turn == "model":
            # modelターンでは、プロンプト部分（<start_of_turn>modelから最初の実質的な応答まで）は-100
            # 実際の応答部分はinput_idと同じ値
            if turn_start_pos is not None:
                # <start_of_turn>model の直後から予測開始位置を特定
                model_start = turn_start_pos + 2  # <start_of_turn> + model
                
                if i < model_start:
                    # プロンプト部分は-100であるべき
                    if label != -100:
                        token_text = processor.tokenizer.decode([input_id])
                        errors.append({
                            'position': i,
                            'turn': current_turn,
                            'token': token_text,
                            'input_id': input_id,
                            'label': label,
                            'expected_label': -100,
                            'error': 'Model prompt should be masked'
                        })
                else:
                    # 応答部分はinput_idと同じであるべき
                    if label != input_id:
                        token_text = processor.tokenizer.decode([input_id])
                        errors.append({
                            'position': i,
                            'turn': current_turn,
                            'token': token_text,
                            'input_id': input_id,
                            'label': label,
                            'expected_label': input_id,
                            'error': 'Model response should match input_id'
                        })
    
    is_correct = len(errors) == 0
    
    return is_correct, {
        'sample_idx': sample_idx,
        'is_correct': is_correct,
        'error_count': len(errors),
        'errors': errors
    }

def run_comprehensive_verification(samples_per_dataset: int = 25) -> Dict[str, Any]:
    """
    全データセット・全サブタイプでの包括的検証
    """
    print("=" * 80)
    print("第2節: 全データセット・全サブタイプ対応マルチモーダル入力フォーマット検証")
    print("=" * 80)
    
    # 設定読み込み
    config = get_config()
    print(f"✅ 設定読み込み完了")
    print(f"  モデル: {config.GEMMA_MODEL_ID}")
    print(f"  各データセットサンプル数: {samples_per_dataset}")
    
    # セッション情報
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"🚀 包括的検証セッション開始: {session_id}")
    print("=" * 60)
    
    # Processor初期化
    print("📝 Tokenizer/Processorを初期化中...")
    processor = AutoProcessor.from_pretrained(
        config.GEMMA_MODEL_ID,
        trust_remote_code=True,
        use_fast=False
    )
    
    # 全データセット設定を取得
    dataset_configs = get_all_dataset_configs()
    print(f"📦 検証対象データセット数: {len(dataset_configs)}")
    
    # 検証結果を格納
    verification_results = {
        'session_id': session_id,
        'total_datasets': len(dataset_configs),
        'samples_per_dataset': samples_per_dataset,
        'dataset_results': {},
        'summary': {
            'successful_datasets': 0,
            'failed_datasets': 0,
            'total_samples': 0,
            'successful_samples': 0,
            'total_processing_time': 0
        }
    }
    
    start_time = time.time()
    
    for dataset_config in dataset_configs:
        dataset_name = dataset_config['name']
        dataset_type = dataset_config['dataset_type']
        sub_dataset = dataset_config['sub_dataset']
        
        print(f"\n===== {dataset_name.upper()} 検証開始 =====")
        dataset_start_time = time.time()
        
        try:
            # データセット初期化
            print(f"📦 {dataset_name} データセットを準備中...")
            dataset = HybridDataset(
                base_image_dir=config.DATASET_BASE_DIR,
                gemma_processor=processor,
                samples_per_epoch=samples_per_dataset,
                dataset=dataset_type,
                sample_rate=[1],
                **{dataset_config['config_attr'].lower(): sub_dataset}
            )
            
            # DataLoader作成
            dataloader = DataLoader(
                dataset,
                batch_size=8,
                collate_fn=collate_fn,
                shuffle=False
            )
            
            print(f"✅ {dataset_name} 初期化完了 (総サンプル数: {len(dataset)})")
            
            # サンプル検証
            dataset_results = {
                'dataset_name': dataset_name,
                'dataset_type': dataset_type,
                'sub_dataset': sub_dataset,
                'total_samples': 0,
                'successful_samples': 0,
                'failed_samples': 0,
                'token_statistics': [],
                'label_masking_results': [],
                'processing_time': 0,
                'errors': []
            }
            
            sample_count = 0
            successful_count = 0
            
            for batch_idx, batch in enumerate(dataloader):
                if sample_count >= samples_per_dataset:
                    break
                
                try:
                    input_ids = batch['input_ids']
                    labels = batch['labels']
                    
                    batch_size = input_ids.shape[0]
                    
                    for i in range(batch_size):
                        if sample_count >= samples_per_dataset:
                            break
                        
                        # トークン分析
                        token_analysis = analyze_sample_tokens(
                            input_ids[i], labels[i], processor, sample_count
                        )
                        dataset_results['token_statistics'].append(token_analysis)
                        
                        # ラベルマスキング検証
                        is_correct, masking_result = verify_label_masking(
                            input_ids[i], labels[i], processor, sample_count
                        )
                        dataset_results['label_masking_results'].append(masking_result)
                        
                        if is_correct:
                            successful_count += 1
                        
                        sample_count += 1
                        
                        # 進捗表示（簡潔に）
                        if sample_count % 10 == 0:
                            print(f"  進捗: {sample_count}/{samples_per_dataset} サンプル処理完了")
                
                except Exception as batch_error:
                    error_info = {
                        'batch_idx': batch_idx,
                        'error': str(batch_error),
                        'error_type': type(batch_error).__name__
                    }
                    dataset_results['errors'].append(error_info)
                    print(f"  ❌ バッチ {batch_idx} でエラー: {batch_error}")
            
            # データセット結果のまとめ
            dataset_results['total_samples'] = sample_count
            dataset_results['successful_samples'] = successful_count
            dataset_results['failed_samples'] = sample_count - successful_count
            dataset_results['processing_time'] = time.time() - dataset_start_time
            
            # 統計計算
            if dataset_results['token_statistics']:
                avg_seq_len = sum(s['sequence_length'] for s in dataset_results['token_statistics']) / len(dataset_results['token_statistics'])
                avg_labeled_tokens = sum(s['labeled_tokens'] for s in dataset_results['token_statistics']) / len(dataset_results['token_statistics'])
                seg_detection_rate = sum(1 for s in dataset_results['token_statistics'] if s['special_tokens'].get('seg_positions', [])) / len(dataset_results['token_statistics'])
                
                dataset_results['statistics'] = {
                    'avg_sequence_length': avg_seq_len,
                    'avg_labeled_tokens': avg_labeled_tokens,
                    'seg_detection_rate': seg_detection_rate,
                    'success_rate': successful_count / sample_count if sample_count > 0 else 0
                }
            
            # 結果表示
            success_rate = (successful_count / sample_count * 100) if sample_count > 0 else 0
            print(f"✅ {dataset_name} 検証完了")
            print(f"  処理時間: {dataset_results['processing_time']:.2f}秒")
            print(f"  成功率: {success_rate:.1f}% ({successful_count}/{sample_count})")
            
            if success_rate == 100:
                verification_results['summary']['successful_datasets'] += 1
            else:
                verification_results['summary']['failed_datasets'] += 1
            
            verification_results['dataset_results'][dataset_name] = dataset_results
            verification_results['summary']['total_samples'] += sample_count
            verification_results['summary']['successful_samples'] += successful_count
            
        except Exception as dataset_error:
            print(f"❌ {dataset_name} データセット初期化失敗: {dataset_error}")
            verification_results['summary']['failed_datasets'] += 1
            
            # エラー情報を記録
            error_result = {
                'dataset_name': dataset_name,
                'dataset_type': dataset_type,
                'sub_dataset': sub_dataset,
                'error': str(dataset_error),
                'error_type': type(dataset_error).__name__,
                'processing_time': time.time() - dataset_start_time
            }
            verification_results['dataset_results'][dataset_name] = error_result
    
    # 全体の処理時間
    verification_results['summary']['total_processing_time'] = time.time() - start_time
    
    # 最終結果表示
    print("\n" + "=" * 80)
    print("📊 包括的検証結果サマリー")
    print("=" * 80)
    
    summary = verification_results['summary']
    print(f"処理時間: {summary['total_processing_time']:.2f}秒")
    print(f"成功データセット: {summary['successful_datasets']}/{verification_results['total_datasets']}")
    print(f"総サンプル数: {summary['total_samples']}")
    print(f"成功サンプル数: {summary['successful_samples']}")
    print(f"全体成功率: {(summary['successful_samples']/summary['total_samples']*100):.2f}%" if summary['total_samples'] > 0 else "N/A")
    
    # 結果保存
    output_dir = "verification_output"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"comprehensive_input_formatting_{session_id}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(verification_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n📁 詳細結果を保存: {output_file}")
    
    return verification_results

def parse_args():
    parser = argparse.ArgumentParser(description="Comprehensive Input Formatting Verification")
    parser.add_argument("--samples", type=int, default=25, help="Number of samples per dataset")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 包括的検証を実行
    results = run_comprehensive_verification(samples_per_dataset=args.samples)
    
    # 成功/失敗の判定
    summary = results['summary']
    if summary['successful_datasets'] == results['total_datasets'] and summary['successful_samples'] == summary['total_samples']:
        print("\n🎉 全データセット・全サブタイプで検証が成功しました！")
        return 0
    else:
        print(f"\n⚠️ 一部のデータセットで問題が発見されました。")
        print(f"  失敗データセット: {summary['failed_datasets']}")
        print(f"  失敗サンプル: {summary['total_samples'] - summary['successful_samples']}")
        return 1

if __name__ == "__main__":
    exit(main()) 