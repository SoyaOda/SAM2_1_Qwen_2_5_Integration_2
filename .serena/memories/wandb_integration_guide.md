# WandB統合ガイド

このプロジェクトでWandBが実装されており、他のプロジェクトでも使用可能です。

## 1. 依存関係の追加

### requirements.txtに追加:
```
wandb
```

## 2. 環境設定

### .env.example ファイルを作成:
```bash
# WandB設定
WANDB_API_KEY=your_api_key_here
WANDB_PROJECT=your-project-name
WANDB_ENTITY=your_username_here

# その他の環境変数
CUDA_VISIBLE_DEVICES=0
```

### .envファイル（実際の設定）を作成:
実際のAPIキーを.envファイルに設定します。

## 3. コマンドライン引数の追加

### argparseに以下を追加:
```python
import argparse

parser.add_argument('--use_wandb', action='store_true',
                   help='WandBを使用')
parser.add_argument('--wandb_project', type=str, default='your-project-name',
                   help='WandBプロジェクト名')
```

## 4. WandB初期化コード

### メイン関数での初期化:
```python
import wandb
import os

# WandBの初期化
if args.use_wandb:
    # 環境変数からAPIキーを設定（なければデフォルトを使用）
    if 'WANDB_API_KEY' not in os.environ:
        os.environ['WANDB_API_KEY'] = 'your_default_api_key'
    
    wandb.init(project=args.wandb_project, config=vars(args))

# 訓練終了時
if args.use_wandb:
    wandb.finish()
```

## 5. ログ記録の実装

### 訓練ループでのログ記録:
```python
# 損失のログ記録
if self.config.use_wandb:
    wandb.log({
        'train/loss': total_loss.item(),
        'train/lm_loss': lm_loss.item(),
        'train/seg_loss': seg_loss.item(),
        'step': step
    })

# 評価メトリクスのログ記録
if hasattr(self, 'use_wandb') and self.use_wandb:
    wandb.log({
        'val/iou': iou_score,
        'val/dice': dice_score,
        'val/samples': sample_count,
        'step': self.global_step
    })

# 画像のログ記録
if self.config.use_wandb:
    wandb.log({
        "visualization": wandb.Image(str(image_path)),
        "val_iou": iou,
        "val_dice": dice,
    }, step=step)

# フェーズ情報のログ記録
if hasattr(self, 'use_wandb') and self.use_wandb:
    wandb.log({
        "phase": 1 if freeze else 2,
        "lm_frozen": freeze,
        "lm_params_changed": changed_count,
    })
```

## 6. 実際のAPIキー

プロジェクトで使用されているAPIキー:
```
a389f0d40902815f4eaf9f4dd9e298b722db36e9
```

## 7. 使用方法

### 訓練実行時:
```bash
python train.py --use_wandb --wandb_project your-project-name
```

### 環境変数による設定:
```bash
export WANDB_API_KEY=your_api_key
export WANDB_PROJECT=your-project-name
python train.py --use_wandb
```

## 8. ディレクトリ構造

WandBは自動的に以下のディレクトリを作成します:
```
wandb/
├── latest-run
└── run-YYYYMMDD_HHMMSS-ランダムID/
    └── run-ランダムID.wandb
```

## 9. 注意事項

1. **APIキーの管理**: 実際のプロジェクトではAPIキーを環境変数や.envファイルで管理
2. **プロジェクト名**: 各プロジェクトで適切なプロジェクト名を設定
3. **エラーハンドリング**: WandBが利用できない環境でもスクリプトが動作するようにtry-except文で囲む
4. **依存関係**: requirements.txtにwandbを追加することを忘れずに

## 10. 実装例（完全版）

```python
import wandb
import os
import argparse

def setup_wandb_args(parser):
    """WandB関連の引数を追加"""
    parser.add_argument('--use_wandb', action='store_true',
                       help='WandBを使用')
    parser.add_argument('--wandb_project', type=str, default='your-project',
                       help='WandBプロジェクト名')

def initialize_wandb(args):
    """WandBを初期化"""
    if args.use_wandb:
        # 環境変数からAPIキーを設定（なければデフォルトを使用）
        if 'WANDB_API_KEY' not in os.environ:
            os.environ['WANDB_API_KEY'] = 'a389f0d40902815f4eaf9f4dd9e298b722db36e9'
        
        wandb.init(project=args.wandb_project, config=vars(args))
        return True
    return False

def log_metrics(use_wandb, metrics_dict, step=None):
    """メトリクスをログ記録"""
    if use_wandb:
        try:
            if step is not None:
                metrics_dict['step'] = step
            wandb.log(metrics_dict)
        except ImportError:
            pass

def finish_wandb(use_wandb):
    """WandBセッションを終了"""
    if use_wandb:
        wandb.finish()
```