# Qwen2.5-VLとSAM2.1統合モデルの修正方針

## Q1. Qwen2.5-VLのViT出力取得方法 (現状確認)

現在の統合モデルでは、**Qwen2.5-VL**の画像エンコーダ（Vision Transformer）の出力を**言語モデルと統合する前の中間層**からフックしています。具体的には、**Qwen2.5-VLのVision Transformerの最終層出力**（全パッチの埋め込みベクトル）をフックして取得し、それをSAM側へ渡しています。これは\*\*「merger前」\*\*、すなわちQwenの視覚特徴がLLM部分に統合される直前の段階の出力を使っていることになります。

Qwen2.5-VLのViTは画像をパッチに分割し、**ウィンドウAttention**や**動的解像度**対応を備えた構造になっており、入力画像サイズに応じた**ネイティブ解像度の特徴マップ**を出力します。したがって、たとえば入力画像を\$1024\times1024\$にリサイズすれば、**\$64\times64\$のパッチ特徴**（各パッチに対するD次元埋め込みベクトル）を出力として得られます。このViT出力は、Qwen内部ではテキストと結合される前段階で利用されますが、本統合では**そのViT出力をフック**して取り出し、SAM側の画像特徴として使用しています。

**実装面**では、HuggingFace版Qwen2.5-VLモデルをロードし、`model.model.vision_tower`（または同等のVision Encoder）部分の**最終ブロックにforwardフック**を登録する方法が考えられます。以下に簡単な例を示します：

```python
# Qwen2.5-VLモデル読み込み (7B-Illustrationモデルを例)
from transformers import Qwen2_5_VLForConditionalGeneration
qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# Vision Transformerの最終層出力をフックする
def catch_vit_output(module, input, output):
    # output: (batch, num_patches+1, dim) のテンソル（CLSトークン+パッチ特徴）
    qwen_model.visual_features = output

# モデル内部のVision Transformer最終ブロックにフック登録
vit_encoder = qwen_model.model.vision_tower  # Qwenの視覚エンコーダ
vit_encoder.blocks[-1].register_forward_hook(catch_vit_output)
```

上記のようにしておけば、画像入力を含む推論時に`qwen_model(..., pixel_values=image_tensor, ...)`を呼ぶと、自動的に`qwen_model.visual_features`にViT出力が格納されます。**CLSトークン**を含む場合は除去し、残りのパッチ埋め込みを**空間マップ状にリシェイプ**して使用します（例：`features = qwen_model.visual_features[:,1:,:]` を取り出し、サイズ\$(B,64\times64,D)\$を\$(B,D,64,64)\$に変換）。

> **補足:** 現状のコードベースでは、おそらく上記のように**Vision Transformerの最終層出力**をフックして取得しています（ユーザーメッセージからも「中間層からのhook／merger前」と推測されています）。このアプローチにより、Qwenの視覚特徴を自由に取り出し可能です。

## Q2. SAM2.1の`high_res_features`入力改善方針

**現状**、SAM2.1側のマスクデコーダに渡す`high_res_features`には**ダミーのランダムテンソル**を使用しています。これは統合初期段階で、Qwenの特徴だけでは高解像度なマスク生成に不十分であったための一時措置です。今後はこの部分を改善し、**Qwen2.5-VLからの特徴のみで補完する**か、あるいは\*\*軽量な補完CNN（例えばFPN: Feature Pyramid Network）\*\*を追加して高解像度特徴を生成することが望まれます。

提案としては**両アプローチを組み合わせる形**で、**QwenのViT特徴に軽量FPNを適用して高解像度特徴を生成**する方法が最適と考えられます。これにより、追加で巨大なモデルを増やすことなく（SAMのViTを丸ごと併用すると計算コストが大きい）、既存のQwen視覚特徴から**マスク境界を精細化するための特徴マップ**を作り出せます。

具体的な方針:

* **(A) 軽量FPNモジュールの追加:** Qwen2.5-VLのViT出力（\$64\times64\$空間解像度の特徴マップ）に対し、**転置畳み込み（ConvTranspose2d）やアップサンプリング＋畳み込み**から成る小規模ネットワークを適用し、逐次的に解像度を2倍ずつアップサンプルします。例えば、\$64\times64\$の特徴を**2倍アップ**して\$128\times128\$に、さらにもう一度2倍にして\*\*\$256\times256\$の高解像度特徴**を得ます（元画像1024pxの1/4スケールに相当）。このときチャンネル数も適度に圧縮します（例：Qwen出力次元\$D\$が大きい場合は、1x1畳み込みでまず\$256\$次元程度に縮小）。得られた複数解像度の特徴マップ（例えば\$128^2\$と\$256^2\$の2層分）を**リストとして`high_res_features`にセット\*\*し、SAM2.1のマスクデコーダに渡します。こうすることで、マスクデコーダは低解像度の画像埋め込み`image_embeddings`に加えて、高解像度のディテール情報を得られ、マスク出力の精度が向上します。

* **(B) SAM2.1既存ViTの活用 (今回は非効率なので採用しない):** 基本的には上記(A)でQwenの特徴のみから高解像度特徴を生成する方針ですが、性能や精度上どうしても不足がある場合には**SAM2.1の画像Encoderを併用**するオプションも検討します。例えば、Qwenで推定したバウンディングボックスや粗マスクを**プロンプト**として、SAM2.1のViT＋マスクデコーダに渡し細密なマスクを得る、という**段階的パイプライン**も可能です。ただしこの場合、モデル内で2つのViTを動かすことになるため計算コストが大きく、**統合モデルとしては非効率**です。そこで、どうしてもQwen単独の特徴で補完できない細部についてのみSAM ViTを使う（例えば微小な対象や特殊な質感の場合）といった工夫で最小限の併用に留めるのが良いでしょう。基本方針としては**まずQwen+軽量FPNで完結**させ、必要なら最後の手段でSAM ViTを使用する形が望ましいです。

以上を踏まえて、**修正実装の概要**を以下に示します。

### 修正実装の具体例

1. **QwenのViT出力取得** – 前項Q1で述べたフック機構により、\$B\times(64\*64)\times D\$サイズの視覚特徴テンソル`vit_feats`を取得済みとします。これを\$(B, D, 64, 64)\$にリシェイプし、必要ならCLSトークン分を除去済みとします。

2. **高解像度特徴生成モジュール** – 以下のようなPyTorchモジュールを設計します（簡易な例）:

   ```python
   import torch.nn as nn

   class HighResFeatureGenerator(nn.Module):
       def __init__(self, in_channels, mid_channels=256, out_channels=256):
           super().__init__()
           # 1x1 convでチャネル圧縮
           self.reduce_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
           # 転置畳み込みで2倍アップサンプル x 2段
           self.up_conv1 = nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=2, stride=2)
           self.up_conv2 = nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=2, stride=2)
           # 活性化関数・正規化（適宜）
           self.bn1 = nn.BatchNorm2d(mid_channels)
           self.bn2 = nn.BatchNorm2d(out_channels)
           self.act = nn.GELU()  # 例としてGELU
       def forward(self, x):
           x = self.reduce_conv(x)            # (B, D, 64, 64) -> (B, midC, 64, 64)
           x = self.act(x)
           x = self.bn1(self.up_conv1(x))    # -> (B, midC, 128, 128)
           x = self.act(x)
           x = self.bn2(self.up_conv2(x))    # -> (B, outC, 256, 256)
           x = self.act(x)
           # アップサンプル過程で中間解像度も保持したければreturnする
           return x
   ```

   上記では2段のConvTranspose2dで\$64\rightarrow128\rightarrow256\$とアップサンプルし、最終出力を`out_channels`次元（例では256次元）に揃えています。BatchNormや活性化は必要に応じて挿入します。

3. **SAM2.1マスクデコーダへの接続** – FPNモジュールの出力を、SAMのマスクデコーダ呼び出し時に`high_res_features`引数として渡します。Ultralytics版SAM2実装では以下のようにリストで特徴を渡す仕様になっています:

   ```python
   # vit_feats_map: (B, D, 64, 64) Qwen視覚特徴をリシェイプしたもの
   highres_gen = HighResFeatureGenerator(in_channels=D)
   feat_256 = highres_gen(vit_feats_map)             # (B, 256, 256, 256) 高解像度特徴
   # 必要ならfeat_128も生成できるようモジュールを拡張し、ここでは省略

   # マスクデコーダへの入力準備
   image_embeddings = vit_feats_map  # (B, D, 64, 64)をそのまま低解像度埋め込みとする
   high_res_feats_list = [feat_256]  # 今回1レベルのみ。中間レベルもあれば [feat_128, feat_256] のように。

   # SAM2.1のmask_decoderを呼ぶ（例：Ultralytics版Pseudoコード）
   low_res_masks, iou_scores, _, _ = sam_model.sam_mask_decoder(
       image_embeddings=image_embeddings,
       image_pe=sam_model.sam_prompt_encoder.get_dense_pe(),  # 位置エンコーディング
       sparse_prompt_embeddings=point_embeds,    # 点やボックスのプロンプト埋め込み（あれば）
       dense_prompt_embeddings=mask_embed,       # マスクプロンプト埋め込み（あれば）
       high_res_features=[ lvl.unsqueeze(0) for lvl in high_res_feats_list ]  # リスト各特徴をバッチ次元追加
   )
   ```

   上記のように、**`high_res_features`にリスト形式で特徴マップを渡す**ことで、マスクデコーダがそれらを使用して最終マスクを精細化します。Ultralyticsの実装では内部でリストの各レベル特徴に対しアップサンプリングしたマスクを適用しているため、我々の生成した特徴も同様に活用されます。

4. **形状・次元の検証** – 実装後は、**出力マスクのサイズ**や**IoUスコア**の挙動を確認します。特に`high_res_features`を正しく渡せていない場合、出力マスクが荒いままだったりエラーが発生する可能性があります。Ultralytics版SAM2では`high_res_feats`として複数レベルの特徴を内部的に保持しているため、**対応する解像度・チャネル数**を合わせることが重要です（例えば、SAM2.1のデフォルトでは256次元特徴を使っている可能性が高いので、上記でout\_channels=256と設定）。また、不要になった**ランダムテンソルの仮埋め込み部分**はすべて削除し、異常時には明示的に例外を投げるよう修正します（ダミー入力でエラーを隠蔽しないようにする）。

以上の実装方針により、**Qwen2.5-VLの視覚的な理解力**（物体の位置・大局情報）と**SAM2.1の高精細なマスク生成能力**を組み合わせた統合モデルを構築できます。Qwen側ViT出力のみでも、FPN経由で高解像度特徴を補完することで十分高精度なセグメンテーションが可能と期待されます。それでも不足する特殊なケースでは、オプションとして**SAM2.1のViTを限定的に併用**する拡張も念頭に置き、柔軟に対処できる設計にします。

最後に、本修正後のモデルは**FoodLMM改**のような領域特化ファインチューニングに備えて、トークナイザやLoRA適用なども含め**形状整合性**を確認しつつテストを重ねます。これにより、最新VLM(Qwen2.5-VL)と最新SAM(SAM2.1)が深く統合された強力な基盤モデルを実現できるでしょう。

**参考資料:** Qwen2.5-VLのViT構造と動的解像度処理、SAM2.1のマスクデコーダにおける`high_res_features`利用コード。これらを踏まえ、上記の修正実装を進めました。
