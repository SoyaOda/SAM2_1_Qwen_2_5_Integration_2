# Cross-Attention Fusion実装記録 (2025年8月14日)

## 実装概要
md_files/current/sam_vit_cross_attention20250814.mdの仕様に従い、SAM2.1とQwen2.5-VLの視覚特徴をCross-Attentionで融合する機能を実装しました。既存のSigma-Add融合に加えて、より柔軟で強力な特徴融合が可能になりました。

## 主な実装内容

### 1. fusion_layers.pyの作成
新規ファイル`src/models/fusion_layers.py`を作成し、以下のクラスを実装：

- **CrossAttentionFusion**: Multi-head Cross-Attentionによる融合
  - SAM特徴をQuery、Qwen特徴をKey/Valueとして使用
  - 8ヘッド、dropout=0.1のデフォルト設定
  - オプションのゲート機構で融合度合いを動的制御
  - 残差接続とLayerNormで学習安定性を確保

- **SigmaAddFusion**: 既存のSigma-Add融合（後方互換性）
  - 学習可能なβパラメータによる単純な加算融合

- **HybridFusionModule**: 両融合方式を統一インターフェースで提供
  - fusion_typeパラメータで融合方式を切り替え可能
  - 空間特徴の自動フラット化/リシェイプ機能

### 2. LISAConfig拡張
`src/config.py`に以下の設定を追加：
```python
fusion_type: str = "sigma_add"  # Options: "sigma_add", "cross_attention"
fusion_num_heads: int = 8       # Cross-Attentionのヘッド数
fusion_dropout: float = 0.1     # Cross-Attentionのドロップアウト率
fusion_use_gate: bool = True    # ゲート機構の使用有無
```

### 3. LISA_Model修正
`src/models/lisa_model.py`の主な変更点：

#### 初期化部分
- fusion_typeに基づいて適切な融合モジュールを初期化
- Cross-Attention選択時はHybridFusionModuleをcross_attentionモードで初期化
- Sigma-Add選択時は既存のimage_fusion_betaパラメータも保持（後方互換性）

#### forward部分
- 融合処理をfusion_typeに応じて分岐
- Cross-Attention: HybridFusionModuleのcross_attention融合を実行
- Sigma-Add: 既存の加算融合を実行（HybridFusionModule経由）

#### save/load部分
- Cross-Attention: state_dictをimage_fusion_cross_attention.ptとして保存
- Sigma-Add: image_fusion_betaをimage_fusion_beta.ptとして保存

### 4. テストコード
- `test_cross_attention_fusion.py`: フルモデルでの統合テスト
- `test_cross_attention_simple.py`: 融合モジュール単体の軽量テスト

## 技術的詳細

### Cross-Attentionのアーキテクチャ
1. SAM特徴（64×64）をQuery、Qwen特徴をKey/Valueとして使用
2. Qwen特徴が32×32の場合、バイリニア補間で64×64にアップサンプル
3. Multi-head Attention（8ヘッド）で位置ごとに重み計算
4. 残差接続 + LayerNormで安定性確保
5. ゲート機構で元特徴と融合特徴をチャネル毎に重み付け

### パラメータ数の比較
- Cross-Attention: 395,008パラメータ
  - Multi-head Attention層
  - ゲート用の全結合層
- Sigma-Add: 1パラメータ（βのみ）

### メモリ使用量
- Attention行列: 64×64×64×64×8ヘッド = 約128MB（float32、バッチサイズ1）
- 計算効率はヘッド並列化で最適化

## 利点と特徴

### Cross-Attentionの利点
1. **位置適応的な融合**: 各空間位置で必要な特徴を選択的に取り込み
2. **学習可能な対応関係**: SAMとQwen特徴間の対応を自動学習
3. **柔軟性**: 物体境界では詳細重視、内部では意味重視など動的調整

### Sigma-Addとの使い分け
- **Cross-Attention**: 高精度が必要な場合、計算リソースが十分な場合
- **Sigma-Add**: 軽量な融合、後方互換性が必要な場合

## 今後の改善案
1. Attention可視化による融合パターンの分析
2. ヘッド数やゲート機構の最適化
3. 異なる解像度での融合戦略の検討
4. 学習時の収束速度比較実験