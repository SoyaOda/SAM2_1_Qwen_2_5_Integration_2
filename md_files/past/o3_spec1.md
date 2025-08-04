
# Qwen2.5-VL-3B と SAM2.1 統合モデル「LISA改」実装ガイド

以下では、Qwen2.5-VL-3B（Vision-Languageモデル）と Segment Anything Model 2.1 (SAM2.1) を統合したマルチモーダルモデル「LISA改」の詳細設計と実装スクリプトを示します。モデル各部の接続形状やトークナイザ拡張、LoRAによる軽量微調整、損失関数設計について、可能な限り具体的かつ一意に記述します。

## 前提: モデルとライブラリの準備

LISA改の実装は **PyTorch** および **HuggingFace Transformers** をベースとします。また、一部 **Meta Segment Anything 2.1** の公式実装（`facebookresearch/sam2`）を利用します。GPU環境はA100 8枚まで想定し、大規模モデルには4-bit量子化 + LoRA (QLoRA)も視野に入れます。

まず必要ライブラリをインポートし、Qwen2.5-VL-3BおよびSAM2.1モデルを読み込みます。

```python
!pip install transformers==4.51.3 accelerate
!pip install sam2  # Hugging Face版SAM2.1ライブラリ (推論用ラッパ)
!pip install peft   # LoRA実装ライブラリ
```

* **Qwen2.5-VL-3B**はHuggingFace Hubの`Qwen/Qwen2.5-VL-3B-Instruct`からロードします。このモデルは視覚エンコーダViTとテキストデコーダLLMを内包した、画像→テキスト生成が可能な事前学習済みモデルです。
* **SAM2.1**はHuggingFace上の`facebook/sam2.1-hiera-large`チェックポイントを使用します。これはMetaのSegment Anything Model 2 (画像・動画対応)の大規模版で、SAM2.0から視覚的に紛らわしい物体や小物体、被遮蔽物体への性能改善がされています。公式実装を用いてマスクデコーダ部分のみを抽出します。

```python
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Qwen2.5-VL-3B モデルとトークナイザのロード
qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype=torch.float16, device_map="auto"
)
qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# SAM2.1モデルのロード（画像エンコーダ+マスクデコーダ含む）
sam_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large", device_map="auto")
sam_model = sam_predictor.model  # SAM2モデル本体
sam_mask_decoder = sam_model.mask_decoder      # Mask Decoderモジュール
sam_prompt_encoder = sam_model.prompt_encoder  # Prompt Encoderモジュール (ポイント・ボックス用)
sam_image_encoder = sam_model.image_encoder    # 画像エンコーダ (ViT) - 今回は使用しない
```

上記コードにより、`qwen_model`にQwen2.5-VL-3Bがロードされ、`sam_mask_decoder`にSAM2.1のマスクデコーダ部分が取得されました。Qwenモデルには**画像エンコーダ**と**テキストデコーダ**が含まれ、SAMモデルには**画像エンコーダ（凍結予定）**・**プロンプトエンコーダ**・**マスクデコーダ**が含まれます。次節以降でこれらを統合していきます。

## 特殊トークン<SEG>の追加とトークナイザ拡張

LISA改では、テキストデコーダが\*\*<SEG>\*\*という新しい特殊トークンを出力できるように語彙を拡張します。<SEG>トークンは「ここでセグメンテーションマスクを生成する」という指示をモデル内部に与える特殊シンボルです。これにより、LLMが<SEG>を出力した箇所で、その隠れベクトルを使ってマスク生成モジュールを呼び出せるようになります。

HuggingFaceのトークナイザを拡張することで<SEG>を新規追加し、モデルの埋め込み行列を拡張します。以下のコードで、トークナイザに追加し、モデルの単語埋め込み行列をリサイズしています。

```python
# 特殊トークンの追加
special_tokens = {"additional_special_tokens": ["<SEG>"]}
qwen_tokenizer.add_special_tokens(special_tokens)
new_vocab_size = len(qwen_tokenizer)
qwen_model.resize_token_embeddings(new_vocab_size)
seg_token_id = qwen_tokenizer.convert_tokens_to_ids("<SEG>")
print(f"<SEG> token ID: {seg_token_id}, vocab_size: {new_vocab_size}")
```

* `tokenizer.add_special_tokens`により、語彙に<SEG>が追加されます。既存モデルの埋め込み行列サイズと出力層行列も自動で1増やされます（Qwenモデルでは入力エンコーディングと出力ロジット投影で同じ埋め込みを共有しています）。
* 新規トークンの初期埋め込みベクトルはランダム初期化されます（HuggingFaceの`resize_token_embeddings`デフォルト動作）。学習により<SEG>埋め込みは適切な意味表現を獲得します。

## 統合モデルのアーキテクチャ設計

LISA改モデルは、Qwen2.5-VLの**視覚・言語理解能力**とSAM2.1の**高精度マスク生成能力**を組み合わせた構造です。大まかな情報フローは以下の通りです：

1. **画像エンコーダ (ViT)**: 入力画像をパッチ特徴にエンコードします（Qwen既存のViTを使用）。
   - **実装注**: Qwen2.5-VL-3Bでは、視覚処理後にmergerモジュールで言語空間への射影が行われます。純粋な視覚特徴（1280次元）を取得するには、merger前でフックを使用する必要があります。
2. **言語デコーダ (LLM)**: ユーザのテキスト指示を読み取り、画像特徴とクロスアテンションして推論を行います。必要に応じ<SEG>トークンを出力し、マスク生成の指示ポイントを作ります。
3. **隠れベクトル抽出**: デコーダが<SEG>を出力した位置の隠れ状態ベクトルを取得し、小さな**テキストプロンプト射影モジュール**で次元変換します。これがマスク用プロンプトベクトルp\_text (例えば256次元)になります。
4. **画像特徴アダプタ**: ViTの出力特徴マップをSAMマスクデコーダの期待する形状・次元（チャネル数256程度）に射影・整形します。
   - **実装注**: 入力は[B, N_patches, 1280]の形状で、これを[B, 256, H, W]に変換します。
5. **マスクデコーダ**: SAM2.1のMask Decoderに、画像特徴マップとテキスト由来プロンプトベクトルp\_textを入力し、対象領域の**低解像度マスク**を推論します。さらに元画像サイズへ補間して出力マスクを得ます。
6. **テキスト出力**: 必要に応じ、モデルはマスク後にテキスト応答を続けて生成できます（例: 「リンゴをハイライトしました。」など）。

以上を実現するため、以下の追加モジュールと結合部を実装します。

### 画像特徴アダプタ (Vision Feature Adapter)

QwenのViT出力は元々の視覚特徴次元\$D\_v\$（Qwen2.5-VL-3Bでは実際には1280）ですが、SAMマスクデコーダは約256次元の特徴マップを想定しています。そこで**画像特徴アダプタ**として全結合層（1層線形変換）を用意し、各パッチ特徴ベクトルを\$256\$次元に射影します。また、トークン列を2次元グリッドに並べ替えてCNN的なマスク処理に備えます。

```python
import torch.nn as nn

class ImageFeatureAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
    def forward(self, vis_feats):
        # vis_feats: Tensor [B, N_patch, D_v] (バッチ×パッチ数×特徴次元)
        B, N, Dv = vis_feats.shape
        feat = self.proj(vis_feats)  # 射影: [B, N, out_dim]
        # 空間次元にreshape (ここではNが正方数になる想定)
        H = W = int(N ** 0.5)
        feat_2d = feat.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # [B, out_dim, H, W]
        return feat_2d

# Qwen視覚エンコーダ出力次元はモデル設定に依存（3Bモデルでは実際にはDv=1280）
Dv = qwen_model.config.vision_config.hidden_size  # Qwen視覚隠れ次元
image_adapter = ImageFeatureAdapter(in_dim=Dv, out_dim=256)
```

* **入力**: Qwen ViTのパッチ埋め込み列（\[B, N\_patches, D\_v]）。例えば画像448×448の場合N\_patches=1024（14×14パッチ→32×32=1024トークン）、それがQwen内部で2×2圧縮されているなら実質N=256 (16×16)になります。本実装では動的に\$N\$から平方サイズ\$H=\sqrt{N}\$を算出しています。
* **処理**: `nn.Linear`で各特徴ベクトルをD\_v→256次元に変換しています（バイアス付き全結合）。射影後、`view`と`permute`で\[B,256,H,W]のテンソルに並べ替えています。この形状がSAMのマスクデコーダへの画像特徴入力フォーマットです。
* **出力**: 空間次元付き特徴マップ \[B, 256, H, W]。例えば224×224画像ならH=W=16、448×448ならH=W=32といったサイズになります。

なお、Qwen2.5-VLでは**動的解像度**対応により、入力画像サイズに応じてパッチ数Nが可変です。最大で16384トークン（例えば896×896ピクセル相当）まで処理可能です。高解像度画像ではH,Wが大きくなり、細かな特徴を保持できます。LISA改ではトレーニング時に適切な解像度を設定し、このアダプタが出力する特徴マップサイズもそれに合わせて変化します。

### テキストプロンプト射影 (Text Prompt Projector)

LLMデコーダが生成した<SEG>トークンの隠れ状態ベクトル（次元\$D\_l\$、Qwen-3Bでは実際には2048次元）を、SAMマスクデコーダ用のプロンプトベクトルに変換する小型MLPです。ここでは単純に1層の線形変換で次元を256に落とし込みます。必要に応じReLUなどを挟む2層MLPに拡張も可能ですが、まずは線形で設計します。

```python
class TextPromptProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
    def forward(self, text_hidden_state):
        # text_hidden_state: Tensor [B, D_l]
        return self.fc(text_hidden_state)

Dl = qwen_model.config.hidden_size  # Qwen言語隠れ次元 (実際: 2048)
text_prompt_proj = TextPromptProjector(in_dim=Dl, out_dim=256)
```

* **入力**: LLMデコーダ最終レイヤから取り出した、<SEG>トークン位置の隠れベクトル \[B, D\_l]。
* **処理**: 線形変換で\$D\_l \to 256\$へマッピングします。これによって得られるベクトルが **テキストプロンプト埋め込み p\_text** です。
* **出力**: \[B, 256] の潜在ベクトル。これはSAMのマスクデコーダ内部でマスク出力を導く**クエリトークン**として働きます。

この手法は、LISA論文で提案された **“embedding-as-mask”** パラダイムに基づいており、テキストから直接マスクを得るためにLLM隠れベクトルを使用するものです。<SEG>トークンの隠れ状態に対象物の意味的・位置的情報が詰まっていると仮定し、それをそのままセグメンテーションの鍵として用います。

### LISA改統合モデルクラス

上記パーツを統合し、ひとつのモデルクラスとして実装します。これにより、**エンドツーエンドで画像を入力し、テキスト＋マスクを出力**できるようになります。

```python
class LISA_Model(nn.Module):
    def __init__(self, qwen_model, sam_mask_decoder, image_adapter, text_prompt_proj):
        super().__init__()
        self.qwen = qwen_model                  # Qwen2.5-VL-3B (ViT + LLMデコーダ)
        self.sam_mask_decoder = sam_mask_decoder  # SAM2.1 Mask Decoder (Frozen)
        self.image_adapter = image_adapter      # 画像特徴次元変換
        self.text_prompt_proj = text_prompt_proj  # テキストembedding変換
        # Freeze SAM mask decoder parameters by default
        for param in self.sam_mask_decoder.parameters():
            param.requires_grad = False

    def forward(self, input_ids, pixel_values, attention_mask=None, labels=None, mask_labels=None):
        B = input_ids.size(0)
        # 1. Qwenモデルでテキストデコード（隠れ状態取得）
        # Qwenのvisionエンコーダがpixel_valuesから画像パッチembeddingを生成し、テキストとクロスアテンションする
        outputs = self.qwen(input_ids=input_ids, attention_mask=attention_mask,
                             pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
        # Qwen出力
        logits = outputs.logits  # [B, seq_len, vocab_size] テキスト次トークン予測分布
        hidden_states = outputs.hidden_states[-1]  # 最終層隠れ状態 [B, seq_len, D_l]
        
        # 2. 画像特徴を抽出（QwenのViT出力を利用）
        # Qwenモデル内部からVision Transformer出力を取得
        # （HuggingFace実装によって方法が異なる可能性あり。ここではattributesを直接参照）
        if hasattr(self.qwen, "vision_tower"):
            vision_feats = self.qwen.vision_tower[0](pixel_values)  # [B, N_patch, D_v]
        else:
            # なければQwenモデルを全体で動かした後、内部から引き出す処理（擬似コード）
            vision_feats = outputs.vision_hidden_states  # 仮: HFモデルが画像埋め込み出力を提供する場合
        
        # 【実際の実装】Qwen2.5-VL-3Bでは上記の方法が使えないため、以下のようにフックを使用：
        # visual_module = self.qwen.model.visual
        # def capture_features(module, input, output):
        #     features_dict['before_merger'] = input[0]  # merger前の1280次元特徴を取得
        # hook = visual_module.merger.register_forward_hook(capture_features)
        # _ = visual_module(pixel_values, grid_thw=image_grid_thw)
        # vision_feats = features_dict['before_merger']  # [B, N_patches, 1280]
        
        # 3. 画像特徴をSAM入力形式に変換
        image_features = self.image_adapter(vision_feats)  # [B, 256, H_feat, W_feat]
        
        # 4. <SEG>トークンの隠れ状態を取得し、プロンプト埋め込みに変換
        if labels is not None:
            # 教師あり学習時: labelsから<SEG>位置を特定
            # 各サンプルについて<SEG>トークンのインデックスを探す
            seg_indices = []
            for i in range(B):
                seg_pos = (labels[i] == seg_token_id).nonzero(as_tuple=True)[0]
                if len(seg_pos) > 0:
                    seg_indices.append(seg_pos[0].item())
                else:
                    seg_indices.append(None)
        else:
            # 推論時: 出力シーケンス中の最後のトークンが<SEG>と仮定
            seg_indices = []
            for i in range(B):
                seg_indices.append((input_ids[i] == seg_token_id).nonzero(as_tuple=True)[0][-1].item()
                                    if seg_token_id in input_ids[i] else None)
        
        # <SEG>ベクトルからマスクを予測
        mask_logits = []
        for i, seg_idx in enumerate(seg_indices):
            if seg_idx is None:
                mask_logits.append(None)  # このサンプルはマスク出力なし
            else:
                # 対応する隠れ状態ベクトル取得
                seg_hidden = hidden_states[i, seg_idx, :]        # [D_l]
                prompt_embed = self.text_prompt_proj(seg_hidden)  # [256]
                # 5. SAMのマスクデコーダでマスク生成
                # プロンプトembeddingをマスクデコーダに入力（point/boxは無いのでテキストembeddingのみ）
                mask_out = self.sam_mask_decoder(
                    image_embeddings=image_features[i].unsqueeze(0),  # [1,256,H_feat,W_feat]
                    # SAM2.1の実装に応じて適切にパラメータを与える
                    # 例えば prompt_embed を query として渡すようなインターフェースを仮定:
                    prompt_embeddings=prompt_embed.view(1,1,-1)  # [1,1,256]
                )
                low_res_mask = mask_out["masks"] if isinstance(mask_out, dict) else mask_out  # [1, 1, H_feat*4, W_feat*4] 等
                # 6. マスクを元画像サイズにアップサンプリング
                orig_h, orig_w = pixel_values.shape[2], pixel_values.shape[3]
                mask_full = torch.nn.functional.interpolate(low_res_mask, size=(orig_h, orig_w), mode="bilinear")
                mask_logits.append(mask_full[0])  # [orig_h, orig_w]
        # リストをTensorに (存在しない場合はNoneのまま)
        # mask_logits: list長B, 要素shape=[orig_h, orig_w] or None
        return logits, mask_logits
```

**（コード説明）**:

* `LISA_Model`クラスは、コンストラクタで事前読み込み済みの`qwen_model`とSAMの`mask_decoder`、および上で定義した`image_adapter`と`text_prompt_proj`を受け取って内部に保持します。SAMのマスクデコーダは基本凍結とし、`requires_grad=False`に設定しています（後述のLoRA適用箇所以外は学習しない）。
* `forward`メソッドでは入力として**画像ピクセルtensor**（`pixel_values`）と**テキストトークン列**（`input_ids`および`attention_mask`）、さらに学習時は正解ラベル`labels`（テキスト部分）および`mask_labels`（マスク画像）を受け取る設計です。
* **ステップ1**: `self.qwen(...)`を呼び出し、Qwen2.5-VLモデルに画像とテキストを同時に与えてデコードします。`output_hidden_states=True`により全層の隠れ状態が得られ、`outputs.hidden_states[-1]`が**最終層隠れ状態列**となります（形状\[B, seq\_len, D\_l]）。また`outputs.logits`からテキストの次トークン予測分布も得られます。
* Qwen内部で画像はクロスアテンションに利用されます。Qwen2.5-VLは**物体の位置をJSON形式で出力する**能力さえ持つため（例えばバウンディングボックスを安定的にテキスト出力できる）、クロスアテンションを通じてLLM内部にローカライズ情報が反映されます。今回その能力をさらに活用し、座標ではなく**内部embeddingを直接利用**してセグメンテーションにつなげています。
* **ステップ2**: `vision_feats = self.qwen.vision_tower[0](pixel_values)` の部分では、Qwenモデル内の視覚エンコーダ（vision tower）を直接呼び出しています。HuggingFace版Qwenでは、おそらく`model.vision_tower`（または`vision_model`）という属性があり、それがViTモジュールになっています。ここではそれを仮定し、pixel\_valuesから**画像特徴トークン列**を取得しています（形状\[B, N\_patch, D\_v]）。
  *注:* Transformers実装によっては直接vision出力を得るインターフェースが無い可能性もあります。その場合は、`outputs.hidden_states`内に視覚トークンの埋め込みを含めて取得するか、モデル内部から属性を直接参照する方法が必要です。上記コードでは簡略化のため`vision_tower`を直接呼んでいます。
  
  **実装上の注意**: Qwen2.5-VL-3Bの実際の実装では、`model.model.visual()`を直接呼ぶと言語空間に射影された2048次元の特徴が返されます。純粋な視覚特徴（1280次元）を取得するには、mergerモジュールの前でフックを使用する必要があります。
* **ステップ3**: 取得した`vision_feats`を`image_adapter`に通し、\[B,256,H\_feat,W\_feat]の特徴マップを得ます。これがSAMデコーダへの画像エンコーダ出力に相当します。例えば448×448画像ならH\_feat=W\_feat=32程度、256チャネルのマップです。
* **ステップ4**: <SEG>トークン位置の隠れ状態を取り出します。学習時（labelsあり）と推論時で処理が異なりますが、基本的に各バッチごとにシーケンス中の<SEG>インデックスを検出しています。

  * 学習時には、教師ラベル`labels`中に明示的に<SEG>が含まれているので、それを検索します。上記では簡略化のため各シーケンスで最初に出現した位置を`seg_indices`に集めています（実際のデータでは通常1つだけ含まれる想定）。
  * 推論時には、モデル生成途中で<SEG>を出力した時点でマスク生成に移行します。ここでは仮に「入力シーケンス中最後の<SEG>」を検出しています。実際の対話プロンプト構造によっては、ユーザ入力に含まれる<SEG>等は無い前提なので、生成出力のみを見る形です。
* **ステップ5**: 各サンプルについて、<SEG>ベクトルを`text_prompt_proj`に通し、256次元の`prompt_embed`を得ます。その後、SAMマスクデコーダ`self.sam_mask_decoder`に画像特徴`image_features`と`prompt_embed`を与えてマスク予測を行います。実際のSAM2.1デコーダ実装では、`forward`の引数が例えば:

  ```python
  mask_out = sam_mask_decoder(image_embeddings=..., image_pe=..., sparse_prompt_embeddings=..., dense_prompt_embeddings=...)
  ```

  等になっている可能性があります。ここでは、テキストembeddingを**dense\_prompt\_embeddings**として1個だけ与えるイメージです（点やボックスと同様にembeddingとして扱う）。SAM2.1では**曖昧性に応じ複数マスク出力**も可能ですが、LISA改では基本1つのプロンプトembeddingにつき1つのマスクを得る設計です。
  *注:* `sam_mask_decoder`は内部で256×256の低解像度マスクを予測します。出力は辞書で`"masks"`キーにTensorが入り、形状は\[B, num\_masks, 256, 256]です（SAM2.1の場合、画像エンコーダからのストライドによりますが、基本256のマスク解像度）。上記では簡略化のため`mask_out`から直接`low_res_mask`を取得しています。
* **ステップ6**: 得られた低解像度マスク`low_res_mask`を、元の画像サイズ(`orig_h, orig_w`)に`F.interpolate`で双線形アップサンプリングしています。これにより最終出力マスク`mask_full`を生成します。SAMではこのアップサンプリング時に画像エンコーダの高解像度特徴を利用してエッジを高精細化する処理がありますが、ここでは簡易に補間のみ行っています（実アプリケーションではSAMのpostprocessing関数を呼ぶことも可能です）。

`forward`の戻り値は `(logits, mask_logits)` のタプルとしています。`logits`は通常のテキスト次単語予測用、`mask_logits`はリスト形式でバッチ内各サンプルのマスク予測結果を格納しています（マスクが無い場合は`None`）。

**注意点**: 上記実装では、簡潔さのためバッチループで各サンプルのマスクを計算しています。本来はベクトル化して一度にマスクデコーダに投げることも可能です（prompt embeddingsをまとめて渡し、マスクをまとめて得る）。しかしSAMデコーダが複数プロンプトを同時処理できるか不明なため、安全のためループしています。最適化の際には並列計算も検討します。

### パラメータ凍結と LoRA 適用

LISA改では、**事前学習済みの知識を極力活かす**ために大部分のパラメータを凍結し、ごく一部のみ学習させます。特にQwenの言語デコーダは数十億パラメータ規模のため、全更新は非現実的です。そこでLow-Rank Adaptation (LoRA) を用いて一部重みに微小な学習可能パラメータを追加します。

* **Qwen2.5-VL 側**: テキストデコーダ内部の**画像クロスアテンション層**にLoRAを挿入します。具体的には、各Transformerブロックのマルチヘッド注意機構で**キー/バリュー投影行列**に対し、rankの低い補正行列を学習させます。これにより、画像特徴との融合の最適化だけを微調整できます。併せて、新規追加した<SEG>トークンの埋め込み（word embeddingの末尾1行）も学習対象とします。また視覚エンコーダ（ViT）は原則凍結します（SAMのマスク精度に影響するため、ここではQwenのViTを固定し特徴抽出器として扱う）。
  
  **注**: 実際のQwen2.5-VL-3Bモデルの次元は以下の通りです：
  - 言語隠れ次元 (D_l): 2048
  - 視覚隠れ次元 (D_v): 1280
* **SAM2.1 側**: マスクデコーダとプロンプトエンコーダは原則凍結します。これにより、SA-1Bなどで学習された汎用的なセグメント知識を維持します。ただし、Qwenからの特徴分布に適応させるため、**マスクデコーダ内のクロスアテンションにもLoRA**を挿入可能です。必要に応じてMask DecoderのQ,K,V投影に小さなLoRAを加え、微調整できるようにします。初期段階ではSAMデコーダを完全凍結とし、後で性能に応じてLoRAを検討します。
* **アダプタ部**: 画像特徴アダプタとテキストプロンプト射影は**新規追加モジュール**であり、パラメータ数はごく小さいため、**全て学習対象**とします。

以上をコードで設定します。

```python
lisa_model = LISA_Model(qwen_model, sam_mask_decoder, image_adapter, text_prompt_proj)

# Qwenモデル全体を一旦凍結
for param in lisa_model.qwen.parameters():
    param.requires_grad = False
# SAMマスクデコーダはコンストラクタで既に凍結済み

# 特殊トークン<SEG>の埋め込みのみ学習可に
word_embeddings = lisa_model.qwen.get_input_embeddings()  # Qwenの単語埋め込み層
word_embeddings.weight[seg_token_id].requires_grad = True

# LoRAをQwenのクロスアテンションに適用 (HuggingFace PEFT を利用)
from peft import LoraConfig, get_peft_model
# 例えば、Qwenデコーダブロック内で画像に対するAttentionモジュール名が "cross_attn" を含むと仮定
lora_config = LoraConfig(
    r=8, lora_alpha=32, target_modules=["cross_attn", "cross_attention"], lora_dropout=0.1,
    bias="none", task_type="CAUSAL_LM"
)
lisa_model.qwen = get_peft_model(lisa_model.qwen, lora_config)

# 画像アダプタとテキスト射影モジュールは全パラメータを学習可に
for param in lisa_model.image_adapter.parameters():
    param.requires_grad = True
for param in lisa_model.text_prompt_proj.parameters():
    param.requires_grad = True

print("Trainable parameters:",
      sum(p.numel() for p in lisa_model.parameters() if p.requires_grad))
```

**（コード説明）**:

* `lisa_model.qwen.parameters()`の全てを`requires_grad=False`に設定し、Qwenの既存パラメータを固定します。これにはViTとデコーダ全層が含まれます。次に、embedding層から<SEG>トークンの重みだけを取り出し、`requires_grad=True`にセットしています。これで<SEG>のベクトルはランダム初期化から学習可能です。
* HuggingFaceのPEFTライブラリを用いて、LoRAを適用しています。`LoraConfig`で`target_modules`に調整対象モジュール名の一部を指定します。ここでは仮にQwenデコーダ内のクロスアテンション層名が`"cross_attn"`等を含むと想定し、その重みにLoRAを挿入しています（実際のQwen実装に合わせ適宜変更）。`r=8`や`lora_alpha=32`はLoRAのボトルネック次元・スケーリング因子の例です。`get_peft_model`によりQwenモデルがラップされ、内部にLoRAパラメータが追加されます。これらLoRAパラメータ（極小サイズ）だけが学習可能になります。
* `image_adapter`と`text_prompt_proj`は元々学習対象（requires\_grad=True）なのでそのままですが、念のため明示的にTrueに設定しています。これらはパラメータ数が非常に少ない（例えばLinear 768→256で約20万パラメータ）ため、フルチューニングして問題ありません。
* 最後に`print`で学習対象パラメータ数を出力しています。Qwenデコーダ約30億+ViT数億に対し、LoRA(数百万以下)＋Adapter数十万＋<SEG> embedding(数千)のみになるため、大幅に削減されているはずです。例えばLoRA 8rank×2(=16)×（クロスアテンション重みサイズ）程度となります。

以上により、**Qwenの言語知識・対話能力**や**SAMの汎用セグメント知識**を保ちつつ、両者の橋渡しとなる部分のみを学習する体制を整えました。特に、LLMの文法生成能力や世界知識はそのままに、画像と応答の連携部分（どこで<SEG>を出すか、どのembeddingを生成するか）のみ最適化できます。これは小さなデータでも学習しやすく、また大規模モデル(Qwen-72B等)でもQLoRAを組み合わせることで現実的な学習が可能です。

## マルチタスク学習と損失関数設計

LISA改は**テキスト応答タスク**と**セグメンテーション出力タスク**を統一的に学習します。したがって損失関数も両者を組み合わせた形になります。ここでは各損失の定義と統合について述べます。

* **言語生成損失 (\$\mathcal{L}\_{LM}\$)**: 画像キャプション生成やVQA回答など、テキストが正解で与えられるタスクに対しては、標準的な次元交差エントロピー損失を用います。モデル出力`logits`と教師系列`labels`を比較し、正解トークンの対数尤度を最大化します。実装上は`torch.nn.functional.cross_entropy`を使用しますが、`labels`中で無視すべき部分（例: ユーザ発話）は`-100`にマスクして計算します。
* **セグメンテーション損失 (\$\mathcal{L}\_{seg}\$)**: 画像中の対象物マスクが教師として与えられる場合、マスク予測とGTマスクを比較する損失を定義します。ここでは**ピクセル単位のBinary Cross Entropy (BCE)** と**Dice損失**の組み合わせを用います。

  * BCE損失は、各ピクセルの予測確率\$\hat{M}(x,y)\$と真値\$M^\*(x,y)\in{0,1}\$について計算します。実装上は`F.binary_cross_entropy_with_logits`を使い、`mask_logits`（ロジット）と`mask_labels`（0/1のfloatテンソル）から直接計算できます（内部でシグモイド適用を含む）。
  * Dice損失は、予測マスクとGTマスクのIoUに基づく指標です。softな微分可能版として、\$\text{Dice} = 1 - \frac{2\sum \hat{M}*{bin} M^\* + \epsilon}{\sum \hat{M}*{bin} + \sum M^\* + \epsilon}\$を用います（\$\hat{M}\_{bin}\$は\$\hat{M}\$のシグモイド出力をしきい値0.5適用した二値化）。学習時は確率のまま計算するソフトDice損失を用いることもあります。下記ではシグモイド出力に対して計算しています。
* **統合損失**: 最終的な損失は \$\mathcal{L} = \mathcal{L}*{LM} + \lambda \mathcal{L}*{seg}\$ で与えます。\$\lambda\$はタスク間の重みで、初期は1に設定し学習の進み具合で調整します。これにより、言語タスクとセグメントタスクの勾配がバランスよくモデルを更新します。

コードで損失計算を実装すると次のようになります。

```python
import torch.nn.functional as F

def compute_loss(logits, labels, mask_logits=None, mask_labels=None):
    # 言語損失: 自動回帰言語モデルのクロスエントロピー
    vocab_size = logits.size(-1)
    # labels中の-100は無視される
    lm_loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100)
    
    # セグメンテーション損失（あれば）
    if mask_logits is not None and mask_labels is not None:
        # BCE損失（with logits版）
        bce = F.binary_cross_entropy_with_logits(mask_logits, mask_labels.float())
        # Dice損失
        pred_mask = torch.sigmoid(mask_logits)
        # flatten
        B = mask_labels.size(0)
        pred_flat = pred_mask.view(B, -1)
        true_flat = mask_labels.view(B, -1)
        # soft dice
        intersection = (pred_flat * true_flat).sum(dim=1)
        dice = 1 - (2. * intersection + 1e-5) / (pred_flat.sum(dim=1) + true_flat.sum(dim=1) + 1e-5)
        dice = dice.mean()
        seg_loss = bce + dice
    else:
        seg_loss = 0.0
    
    total_loss = lm_loss + seg_loss
    return total_loss, lm_loss, seg_loss
```

**（コード説明）**:

* `F.cross_entropy`は内部で`log_softmax`を取って-平均してくれるので、明示的にログ確率計算は不要です。`ignore_index=-100`としており、これが設定された場所（ユーザ発話部分など）は損失計算から除外されます。教師データの用意として、例えば`labels = [-100, -100, ..., token_id,...]`のように、アシスタントの発話部分のみトークンIDを持ち、それ以前は-100にしておきます。
* `mask_logits`と`mask_labels`はいずれも\[B, H\_img, W\_img]サイズのテンソルを想定しています（1画像1マスクの場合）。BCEは画素単位の交差エントロピーを計算し、Diceは上記コードでsoft版を計算しています。`1e-5`はゼロ除算防止の微小項です。
* 出力として`total_loss`に加え、デバッグ用に`lm_loss`と`seg_loss`も返しています。

**データセットと学習**: LISA改の学習には多様なデータを混合使用します。例えば、

* COCOなどの**画像キャプション**データ（画像＋キャプション文）→ 言語損失のみ適用。
* VQAv2などの**視覚質問応答**データ（画像＋質問文＋テキスト回答）→ 言語損失のみ。
* RefCOCOなどの**参照セグメンテーション**データ（画像＋指示文＋対象物マスク）→ 言語＋マスク損失。指示文に対しモデル出力は「<SEG>」（＋必要ならテキスト説明）となるよう教師信号を与え、対応する`mask_labels`を提供。
* LISA論文で新規収集した**暗黙的指示によるセグメンテーション**データ（約1000件の画像-指示-マスク）→ 同様に学習。複雑な推論を要する指示にも<SEG>出力で対応できるようにする。

これらをバッチごとに混ぜつつ、`mask_labels`の有無で上記のように損失計算を切り替えます。例えばDataLoaderから来るデータに`mask_labels`が含まれない場合は自動的に`seg_loss=0`になります。適切にシャッフルしながら学習させることで、モデルは**テキスト応答**と**マスク予測**の両方の能力を習得します。

## 実行時の挙動と応答フォーマット

学習を経たLISA改モデルの推論時の動きを確認します。以下に想定シナリオとモデルの挙動を示します：

* **セグメンテーション指示への応答**: ユーザが画像と「この画像のリンゴを全てマスクで示してください」と入力した場合、モデル内部ではテキスト指示を解析し、リンゴに対応する<SEG>トークンを適切なタイミングで出力します。<SEG>の隠れベクトルからSAMマスクデコーダがリンゴ領域を推定し、バイナリマスク画像を生成します。モデルはそれをアップサンプルして出力し、併せて「リンゴをマスクしました。」等のテキストを続けて生成することも可能です。最終的な出力はテキストと画像（マスク）の組になります。
* **質問応答への応答**: ユーザが画像と「この料理に写っている材料は何？」と質問した場合、モデルはセグメントモードに入らず、通常のテキスト応答のみ行います（例えば「トマトとバジルが使われています。」など）。<SEG>は出力されません。これはデータ上そのように学習されているためで、モデルが不要な<SEG>を出すことはありません。
* **複合的な指示への応答**: ユーザが「この料理に何が何個写っていますか？できれば指し示してください。」と依頼した場合、モデルはまず**カウント結果をテキストで**答え（例：「リンゴは3個あります。」）、その後<SEG>を出力して**3つのリンゴをマスク**することが期待されます。LISA改は一つのシーケンス内でテキストとマスクを組み合わせた回答が可能です。実際にはテキスト→<SEG>→テキスト…というシーケンスを作り出し、各<SEG>ごとに対応マスクを生成します。学習データ上でそのような出力例も与えておけばモデルは対応可能です。

推論処理では、生成中に<SEG>トークンが出力されたら、一時的にテキスト生成を中断しマスクデコーダにかける、といった処理フローになります。実装上は、`model.generate()`のループ中で<SEG>を検知したら`model.forward`を利用してマスクを得る、といったカスタム制御が必要になります。しかしこの挙動はUI側で組んでもよい部分なので、モデル自体は単に<SEG>を含むシーケンスとマスクテンソルを返す設計にしています。

最後に、LISA改の**性能ポテンシャル**について触れます。Qwen2.5-VLは既に**高度な物体認識とローカライゼーション能力**を持っています。SAM2.1は**あらゆる物体を高精度に切り抜く**汎用セグメンターです。本モデルはこれらを組み合わせ、指示された対象を正確に理解・定位し、ピクセルレベルでハイライトするというユニークな能力を持ちます。将来的にFoodLMM改のような「料理の量を見積もる」タスクにおいても、まずLISA改が料理領域を正確に抽出し、その上で量推定を行うという形で活用できます。そのため本設計では、深い画像理解とセグメント出力の両立に重点を置きました。特に、モデル内部で<SEG>埋め込みを介して視覚とテキストのブリッジを行う構造は、LISAが提唱した新たな枠組みであり、マルチモーダルモデルにおける**推論セグメンテーション**の可能性を広げるものです。

以上、Qwen2.5-VL-3BとSAM2.1の統合モデル「LISA改」の詳細設計と実装スクリプトを示しました。各モジュールの入出力shape、接続方法、学習戦略について曖昧さなく記述しましたので、これを基に実装・学習を進めれば、目的とする高精度なセグメンテーション対応マルチモーダルモデルが構築できるはずです。

**参考文献**: Qwen2.5-VL 技術報告、LISA: Reasoning Segmentation via LLM、SAM2.1 発表ブログ、Segment Anything 論文など。
