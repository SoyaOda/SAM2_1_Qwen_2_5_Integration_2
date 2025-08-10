#!/usr/bin/env python3
"""
「orange」セグメンテーション問題を分析
ReferSegデータセットで「orange」という単語がどのように扱われているか確認
"""

import json
import os
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent))

def analyze_orange_issue():
    """オレンジ問題を分析"""
    
    # test_a1の結果を確認
    test_dir = Path("test_outputs/test_a1_20250810_153641")
    
    if test_dir.exists():
        json_file = test_dir / "visualization_step_5.json"
        
        if json_file.exists():
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            print("=" * 80)
            print("ReferSegテスト結果の分析")
            print("=" * 80)
            
            print(f"\nユーザー入力: {data['text_info']['user_input']}")
            print(f"アシスタント応答: {data['text_info']['assistant_response']}")
            
            print(f"\nGT重心座標: ({data['gt_centroid']['x']:.1f}, {data['gt_centroid']['y']:.1f})")
            print(f"画像サイズ: {data['shapes']['image']}")
            print(f"マスクサイズ: {data['shapes']['gt_mask']}")
            
            print(f"\nDice Score: {data['metrics']['dice_score']:.4f}")
            print(f"IoU: {data['metrics']['iou']:.4f}")
            
            # 重心の位置を分析
            x_ratio = data['gt_centroid']['x'] / data['shapes']['image'][1]
            y_ratio = data['gt_centroid']['y'] / data['shapes']['image'][0]
            
            print(f"\n重心の相対位置:")
            print(f"  X: {x_ratio:.2%} (左端から)")
            print(f"  Y: {y_ratio:.2%} (上端から)")
            
            print("\n分析:")
            print("- 質問: 'Where is the orange?' は果物のオレンジを指している可能性")
            print("- しかし、画像にオレンジ色の服を着た人物がいる場合、")
            print("  ReferSegデータセットのアノテーションが曖昧な可能性がある")
            print("- 重心座標から、画像の左上寄り(X:26.5%, Y:54.4%)の物体がセグメントされている")
    
    # ReferCOCOデータセットの特性を説明
    print("\n" + "=" * 80)
    print("ReferCOCOデータセットの特性")
    print("=" * 80)
    
    print("""
ReferCOCOデータセットでは、参照表現（referring expression）に基づいて
物体をセグメンテーションします。以下のような曖昧性が存在する可能性があります：

1. 色の名前と物体名の混同
   - "orange" → 果物のオレンジ vs オレンジ色の物体
   - "apple" → 果物のリンゴ vs Apple製品

2. 文脈依存の解釈
   - 画像内に果物のオレンジがない場合、オレンジ色の物体を指す可能性
   - アノテーターの解釈によって異なる場合がある

3. データセットのアノテーション品質
   - ReferCOCO/RefCOCO+/RefCOCOgは人手でアノテーション
   - 一部のアノテーションに曖昧性や誤りが含まれる可能性

推奨対応:
- 元のReferCOCOデータセットのアノテーションを確認
- 必要に応じて曖昧なサンプルを除外
- またはデータ拡張時に明確な参照表現を使用
""")

if __name__ == "__main__":
    analyze_orange_issue()