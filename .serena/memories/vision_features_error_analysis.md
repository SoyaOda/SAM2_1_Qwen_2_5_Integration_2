# Vision Features Error Analysis

## エラーの詳細
- 期待される形状: [B, N_patches, 1280] (verification_summary.mdより)
- 実際の形状: (288, 2048)
- 288 = 24x12 = image_grid_thw の積
- 2048 = Qwenの言語隠れ次元

## o3_spec1.mdとの相違点

### 1. 視覚特徴次元
- o3_spec1.md: "例えばDv=768" (94行目)
- verification_summary.md: 実際は1280
- エラーで見える値: 2048（言語次元）

### 2. 視覚特徴の取得方法
o3_spec1.mdでは以下を想定：
```python
if hasattr(self.qwen, "vision_tower"):
    vision_feats = self.qwen.vision_tower[0](pixel_values)
```

しかし実際のQwen2.5-VLでは：
- `vision_tower`属性は存在しない
- `visual`属性を使用する必要がある
- grid_thwパラメータが必須

## 結論
これは「o3_spec1.mdの想定と異なる」ケースです。
o3_spec1.mdは擬似コードとして書かれており、実際のHuggingFace実装とは異なります。

## 修正方針
1. Qwen2.5-VLの実際の実装に合わせてextract_vision_featuresを修正
2. vision_tower → visual への変更
3. 適切な次元の視覚特徴を取得する