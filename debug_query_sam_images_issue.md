# o3-query プロンプト: SAM画像データフロー問題

## 問題の概要
- 統合モデル(LISA改: Qwen2.5-VL + SAM2.1)で学習時の予測マスクが格子状パターンを示す
- 推論時は正常に動作
- ログから学習時に`sam_images: None`、推論時に`sam_images: torch.Size([1, 3, 1024, 1024])`

## 技術詳細

### データフロー
1. **データセット (`HybridDataset.__getitem__`)**:
   - SAM用画像を `preprocess_sam_image()` で1024x1024に前処理
   - 返り値に `'sam_images': image_sam` を含める

2. **データコレーター (`MultiModalDataCollator.__call__`)**:
   - バッチ化処理でsam_imagesを処理するコードを追加済み
   - しかし学習時にNoneになる問題が継続

3. **モデル(`LISA_Model.forward`)**:
   - `sam_images`パラメータでSAM ImageEncoderに渡す
   - Noneの場合はQwen特徴のみでマスク生成→格子状パターン

### 症状パターン
- 学習開始時: sam_imagesが正常に渡される場合もある
- 学習途中: sam_imagesがNoneになる（データローダーの問題？）
- 推論時: 常に正常にsam_imagesが渡される

### 実装差異
- **学習**: DataLoader + MultiModalDataCollator経由
- **推論**: 直接SAM画像準備（コレーター経由なし）

## 質問

1. **PyTorchのDataLoaderとカスタムcollate_fn使用時の、バッチ内での特定フィールドが間欠的にNoneになる原因**:
   - メモリ問題、型変換エラー、例外の無視などの可能性
   - デバッグ方法とベストプラクティス

2. **マルチモーダルデータ（画像+テキスト）のバッチ処理における一般的な落とし穴**:
   - 動的解像度対応時の画像テンソル処理
   - 異なるサイズの画像のバッチ化時の問題

3. **SAM2.1 + VLM統合時の高解像度画像フローの設計パターン**:
   - VLM用低解像度画像とSAM用高解像度画像の並行処理
   - データローダーでの効率的な処理方法

4. **格子状マスクパターンの技術的説明**:
   - SAM特徴なしでVLM特徴のみでマスク生成した場合の典型的な失敗モード
   - Token-FPNやマスクデコーダーでの特徴不整合時の症状

これらについて、Qwen2.5-VL、SAM2.1、LISA等の公式実装を参考にした解決策を教えてください。