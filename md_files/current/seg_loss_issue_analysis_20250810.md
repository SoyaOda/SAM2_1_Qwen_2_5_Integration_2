# SEG LOSS問題分析レポート (2025/08/10)

## 概要
minimal_train.pyでSEG LOSSが下がらない問題について、A-1テスト（SAM2.1オラクルプロンプト過学習テスト）を実施し、根本原因を特定しました。

## 実施したテスト
### A-1テスト: SAM2.1側の"オラクル・プロンプト"過学習テスト
- **目的**: マスクデコーダ経路（＋LoRA）が正常に学習できるかを、VLMを迂回して検証
- **手法**: GTマスクの重心点を直接SAM2.1のPromptEncoderに入力
- **期待値**: Dice ≥ 0.95, IoU ≥ 0.9（数百〜数千ステップ）
- **実際の結果**: 
  - 最高Dice Score: 0.7609
  - 最終Dice Score: 0.0185
  - 学習が不安定で目標未達成

## 発見された問題点

### 1. 致命的問題: HybridDatasetのランダム選択バグ
**重要度: ★★★★★（最優先）**

#### 問題内容
`src/data/dataset.py`のHybridDataset.__getitem__メソッドが、インデックスに関係なく毎回ランダムにデータを選択している。

```python
# 問題のコード (dataset.py:656行目)
def __getitem__(self, idx):
    dataset_idx = np.random.choice(len(self.all_datasets), p=self.sample_rate)
    selected_dataset = self.all_datasets[dataset_idx]
```

#### 影響
- **同じインデックスでも毎回異なるサンプルが返される**
- `--num_samples 1`でも毎エポック異なるデータで学習
- 過学習すらできない状態
- **これがSEG LOSSが下がらない根本原因**

#### 証拠
- Step 0: 木のようなマスク
- Step 50: 全く異なる形状のマスク（下部の白い領域）
- 同じデータで繰り返し学習できていない

#### 修正案
```python
def __getitem__(self, idx):
    # ランダム選択ではなく、インデックスベースで決定的に選択
    dataset_idx = idx % len(self.all_datasets)
    selected_dataset = self.all_datasets[dataset_idx]
    sample_idx = idx // len(self.all_datasets) % len(selected_dataset)
    sample = selected_dataset[sample_idx]
```

### 2. MultiModalDataCollatorの問題
**重要度: ★★★☆☆（中程度）**

#### 問題内容
- `original_image`がHybridDatasetから返されているが、collatorを通過後にバッチに含まれない
- 可視化のために必要な元画像が取得できない

#### 影響
- デバッグが困難
- 入力画像と予測結果の対応が確認できない

#### 修正案
MultiModalDataCollatorが`original_images`を正しく渡すようにする（既に実装はあるが動作していない）

### 3. LoRA学習の問題
**重要度: ★★☆☆☆（低〜中程度）**

#### 問題内容
- 学習可能パラメータが902個のみ（全体の0.002%）
- 学習が不安定（最高0.76→最終0.018）

#### 可能な原因
- LoRAのランクが小さすぎる（r=8）
- 学習率が不適切（1e-3）
- LoRAの適用箇所が限定的

## 優先順位付きTODOリスト

### 優先度1（即座に対応）
1. **HybridDatasetのランダム選択バグを修正**
   - dataset.pyの__getitem__メソッドを決定的な選択に変更
   - 同じインデックスで同じサンプルが返るように修正

### 優先度2（次に対応）
2. **修正版でA-1テストを再実行**
   - 同一サンプルでの過学習が可能か確認
   - 目標: Dice ≥ 0.95, IoU ≥ 0.9

3. **MultiModalDataCollatorの修正**
   - original_imagesが正しくバッチに含まれるよう修正
   - 可視化の改善

### 優先度3（その後対応）
4. **A-2テスト: VLM→SAM接続テスト**
   - Focus_Seg_Loss20250810_spec.mdに従って実装
   - <SEG>トークン経由の接続を検証

5. **LoRA設定の最適化**
   - ランクを増やす（r=16 or 32）
   - 学習率の調整（1e-4 or 1e-5）
   - 適用箇所の拡大

## 結論
**HybridDatasetのランダム選択バグが、SEG LOSSが下がらない根本原因である可能性が極めて高い。**
このバグにより、モデルは毎回異なるデータで学習しており、パターンを学習できない状態にある。
この問題を修正することで、SEG LOSSの改善が期待できる。

## 次のアクション
1. HybridDatasetの修正を最優先で実施
2. 修正後、A-1テストで過学習能力を確認
3. 過学習が確認できたら、minimal_train.pyで本格的な学習を再開