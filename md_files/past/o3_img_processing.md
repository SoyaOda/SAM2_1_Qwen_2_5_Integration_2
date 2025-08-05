了解しました。Qwen2.5-VLとSAM2.1を統合したマルチモーダルモデル「LISA改」における解像度の整合性問題、およびマスク生成・損失計算のベストプラクティスについて、LISA・SAM2・Qwen2.5-VLの最新の公式実装・論文・技術報告を踏まえて、具体的な設計指針をまとめてお伝えします。

調査が完了次第、各論点（解像度変換、位置ズレ防止、マスク補間、損失関数設計）について曖昧さのない形で提案いたします。しばらくお待ちください。


# Qwen2.5-VL-3B と SAM2.1 統合モデル「LISA改」実装ガイド

以下では、Qwen2.5-VL-3B（Vision-Languageモデル）と Segment Anything Model 2.1 (SAM2.1) を統合したマルチモーダルモデル「LISA改」の詳細設計と実装手順を解説します。モデル各部の入出力形状やトークナイザ拡張、LoRAによる軽量微調整、損失関数設計について、**ベストプラクティスに基づき曖昧さなく**記述します。また、**解像度差の扱い**や**マスク損失計算**といった重要ポイントについても具体的な対処法を示します。参考として、LISAや最新のマルチモーダルセグメンテーション手法の知見も踏まえています。

## モデルとライブラリの準備

**環境:** 実装は PyTorch と HuggingFace Transformersを使用します。視覚セグメントにはMeta Segment Anything 2.1 (SAM2.1) の公式実装を利用します。GPUはA100クラスを想定し、大規模モデルでは4-bit量子化＋LoRA（QLoRA）も検討します。

まず必要ライブラリをインストールし、Qwen2.5-VL-3BおよびSAM2.1モデルを読み込みます。Qwen2.5-VL-3BはHuggingFace Hub上の `"Qwen/Qwen2.5-VL-3B-Instruct"` から、SAM2.1は `"facebook/sam2.1-hiera-large"` チェックポイントを使います。

```python
!pip install transformers==4.51.3 accelerate
!pip install sam2    # Hugging Face版SAM2.1ライブラリ (推論用ユーティリティ)
!pip install peft    # LoRA実装ライブラリ

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Qwen2.5-VL-3B モデルとトークナイザのロード（FP16, 自動デバイス割当）
qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype=torch.float16, device_map="auto"
)
qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# SAM2.1モデルのロード（画像エンコーダ+マスクデコーダ含む）
sam_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large", device_map="auto")
sam_model = sam_predictor.model  # SAM2.1モデル本体
sam_mask_decoder = sam_model.mask_decoder      # マスクデコーダモジュール
sam_prompt_encoder = sam_model.prompt_encoder  # プロンプトエンコーダモジュール（ポイント・ボックス用）
sam_image_encoder = sam_model.image_encoder    # 画像エンコーダ (ViT) - 今回は使用しない（後述）
```

上記コードで**Qwen2.5-VL-3B**（3億パラメータ・マルチモーダルLLM）が`qwen_model`にロードされます。このモデルは視覚エンコーダViTと言語デコーダLLMを内包し、画像入力からテキストを生成できるよう指示調整されています。Qwen2.5-VLは画像中の物体や文字、レイアウトなど多様な視覚情報を解析でき、物体の位置をバウンディングボックスやポイントで正確にローカライズし、座標をJSON形式で安定出力する能力も備えています。一方**SAM2.1**はMeta社の汎用セグメンテーションモデルの第2版で、大規模データで学習された\*\*「何でもセグメント」**モデルです。SAM2.1（hierarchical large版）は、SAM2.0で課題であった**視覚的に紛らわしい物体の識別や極小物体・被遮蔽物体のマスク生成\*\*を強化しており、画像・動画双方で安定したセグメンテーション性能を示します。また、SAM2.1では開発者向けにファインチューニング用のトレーニングコードも公開されました。

以上で、Qwenモデル（画像と言語の統合モデル）とSAMモデル（高精度セグメントモデル）の読み込み準備が完了しました。次に、両者を統合するための特殊トークン設定とモジュール設計について説明します。

## 特殊トークンの追加とトークナイザ拡張

**特殊トークン `<SEG>` の導入:** LISA改では、テキストデコーダが特定の箇所で「ここでセグメンテーションマスクを生成する」指示を出すために、新しい特殊トークン `<SEG>` を語彙に追加します。このトークンはモデル内部で「セグメント出力の指示」を担うシンボルであり、LLMがこれを出力した位置で対応する**マスク生成処理**がトリガーされます。LISA論文では、この仕組みを “**embedding-as-mask**” パラダイムと呼び、LLMの隠れベクトルをそのままマスク生成に利用する革新的手法として提案しています。

HuggingFaceのトークナイザを拡張して `<SEG>` を追加し、モデルの埋め込み行列をリサイズします。以下のコードでトークナイザ拡張とモデル埋め込みの調整を行います。

```python
# 特殊トークンの追加
special_tokens = {"additional_special_tokens": ["<SEG>"]}
qwen_tokenizer.add_special_tokens(special_tokens)
new_vocab_size = len(qwen_tokenizer)
qwen_model.resize_token_embeddings(new_vocab_size)
seg_token_id = qwen_tokenizer.convert_tokens_to_ids("<SEG>")
print(f"<SEG> token ID: {seg_token_id}, vocab_size: {new_vocab_size}")
```

* `tokenizer.add_special_tokens` により、語彙に `<SEG>` が追加されます。これによって**入力エンコーディングとデコーダ出力**の両方で `<SEG>` を扱えるようになります。既存モデルの単語埋め込み行列と出力ロジット投影行列も自動でサイズ調整され、新規トークンに対応します（Qwenでは入力埋め込みと出力層が共有されています）。
* 新規トークンの初期埋め込みベクトルはランダム値で初期化されます（`resize_token_embeddings` のデフォルト動作）。このベクトルは学習を通じて適切な意味表現（「ここでマスク生成」という概念）を獲得していきます。

以上の設定により、LLMがマスク生成を要する場面で `<SEG>` をシーケンス中に出力可能となりました。次に、このトークン出力を受けて実際にマスクを生成するためのモデルアーキテクチャ統合を解説します。

## 統合モデルのアーキテクチャ設計

LISA改モデルは、Qwen2.5-VLの高度な視覚理解・言語生成能力と、SAM2.1の高精度なピクセル単位マスク生成能力を組み合わせた**エンドツーエンド**構造です。この節では、両モデルをどのように接続し、どの解像度で特徴やマスクを扱うかなど、設計上のポイントを詳述します。

**情報フロー:** LISA改における画像～テキスト～マスク生成の流れは次のようになります（図式化すると LISA 論文 Figure 3 のような構成）:

1. **画像エンコーダ (ViT)** – Qwenモデル内の視覚エンコーダが入力画像をパッチ埋め込み列にエンコードします。例えば448×448の画像なら \$N\_{\text{patch}} \approx 32 \times 32 = 1024\$ トークン（patchサイズ14の場合）となります（解像度は動的に変更可能で、より大きな画像も処理可能）。この出力は高次元 (\$D\_v\$次元) の視覚特徴列です。
2. **言語デコーダ (LLM)** – テキスト指示と画像特徴に基づき、Qwenのデコーダがクロスアテンションを通して推論を行います。ユーザからの質問や指示に答えるテキストを逐次生成し、必要に応じて上記の `<SEG>` トークンを出力します。例えば「画像中のリンゴをマスクで示して下さい」という指示に対し、モデルは「かしこまりました、対象は <SEG> です。」のように `<SEG>` を含む応答を内部生成します。
3. **隠れベクトル抽出** – デコーダが `<SEG>` を出力した位置の**最終層隠れ状態ベクトル**（サイズ \$D\_l\$）を取り出します。このベクトルには直前までのコンテキスト（画像・テキスト文脈）が反映され、ユーザが指示した対象の**意味的・位置的情報**がエンコードされていると期待されます。
4. **テキストプロンプト射影** – 抽出した隠れベクトルに小型の射影モジュール（全結合層）を適用し、SAMのマスクデコーダが扱いやすい低次元（例えば256次元）の**マスクプロンプト埋め込み** \$p\_{\text{text}}\$ に変換します。これはマスク生成の**クエリ**の役割を果たします。
5. **画像特徴アダプタ** – Qwen視覚エンコーダの出力特徴列を、SAMマスクデコーダへの入力形式に合わせます。具体的には、全結合層で特徴次元を圧縮し（\$D\_v \to 256\$程度）、さらにトークン列を2次元の特徴マップ \[B,256,H\_feat,W\_feat] にreshapeします。これにより、SAMデコーダにとって馴染みのある空間レイアウトの**画像エンベディング**を構築します。例えば448×448入力なら \$H\_{\text{feat}}=W\_{\text{feat}}=32\$ のグリッドになります。
6. **マスクデコーダ** – SAM2.1の Mask Decoder に、(5)の画像特徴マップと(4)のプロンプト埋め込み \$p\_{\text{text}}\$ を入力し、対象領域の**低解像度マスク**を推論させます。SAMでは通常、256×256程度の粗いマスクが出力されます（元画像1024の場合）。LISA改ではQwen特徴が元より低解像度ですが、Mask Decoderが学習で補完することを期待します。出力されたマスクはその後**元の画像解像度**にアップサンプリングされ、最終マスク出力となります。
7. **テキスト応答出力** – 場合によっては、モデルはマスク生成後に追加のテキスト解答を続けて生成することもできます（例えば「リンゴをハイライトしました。」などの説明文）。ただし学習データの形式によりますので、本設計ではまず単一ターン内で `<SEG>` の出力とそれに続くテキストを扱えるようにします。

以上が全体フローです。それでは各構成要素について順に具体的に設計・実装を見ていきます。

### 画像特徴アダプタ (Vision Feature Adapter)

QwenのViTが出力する特徴ベクトル列は次元 \$D\_v\$（例えば768や1024）ですが、SAMのマスクデコーダは約256次元程度のチャネルを想定しています。また、QwenのViT出力は1次元列 (B,N\_patches,D\_v) であるのに対し、SAMでは2次元の特徴マップ (チャネル×高さ×幅) として扱います。そのため**画像特徴アダプタ**として**1層の線形層**を用意し、各パッチ特徴を所望の次元に射影した上で、空間グリッドにreshapeします。

```python
import torch.nn as nn

class ImageFeatureAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
    def forward(self, vis_feats):
        # vis_feats: Tensor [B, N_patch, D_v] (バッチ, パッチ数, 特徴次元)
        B, N, Dv = vis_feats.shape
        feat = self.proj(vis_feats)  # 射影: [B, N, out_dim]
        # パッチ列を空間次元にreshape (N が正方数になる前提)
        H = W = int(N ** 0.5)
        feat_2d = feat.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # [B, out_dim, H, W]
        return feat_2d

# Qwen視覚エンコーダ出力次元を取得（3Bモデルでは例えば D_v=768）
Dv = qwen_model.config.vision_config.hidden_size
image_adapter = ImageFeatureAdapter(in_dim=Dv, out_dim=256)
```

* **入力:** Qwen ViTのパッチ埋め込み列 `[B, N, D_v]`。ここで \$N=H\_{feat}\times W\_{feat}\$ はパッチ数です。Qwen2.5-VLは動的解像度に対応しており、入力画像サイズに応じて \$N\$ が変化します（最大で\$16384\$トークン＝画像896×896相当まで可能）。例えば224×224画像なら\$N=14\times14=196\$、448×448なら\$32\times32=1024\$です。
* **処理:** `nn.Linear` により各パッチベクトルを \$D\_v\$→256次元に射影します（バイアス付き全結合）。これによりトークンごとの特徴量次元を圧縮し、SAMのマスクデコーダと次元整合させます。次に、`view`と`permute`で `[B,256,H_{feat},W_{feat}]` のテンソルに形状変換します。この際 \$H\_{feat}=\sqrt{N}\$（\$N\$は正方に近い数と仮定）となります。例えば448×448入力なら \$H\_{feat}=W\_{feat}=32\$ です。
* **出力:** 空間次元付きの**画像特徴マップ** `[B,256,H_{feat},W_{feat}]`。この出力がSAMデコーダへの画像エンベディング入力となります。

**解像度ギャップへの対処:** Qwenの画像エンコーダはデフォルトでは**入力画像をリサイズ**して処理します。上記例では最大448×448ですが、元画像がそれ以上の場合は縮小されます。これにより**高解像度GTマスク**との間に位置ズレが生じる可能性があります。ベストプラクティスとして、**モデル内で扱う解像度とGTマスクの解像度を対応付ける工夫**が必要です。具体的には、**画像のアスペクト比を維持したリサイズ・パディング**を行い、モデルへの入力画像と元画像との対応関係を保持します。例えば元が1280×720の画像でも、短辺基準で448に縮小し長辺はパディング、といった処理で**同一座標系**にマッピングします。そうすることで、後段のマスクをアップサンプリングしても元画像上で正しく重ねられます。

なお、SAM2.1のマスクデコーダは本来**画像エンコーダ出力**（通常64×64グリッド）と対応する位置エンコーディングを期待しています。今回はQwen特徴を利用するため、解像度32×32程度と粗くなりますが、**モデルが学習で補完**してくれることを狙います。必要に応じて、SAMの位置エンコード（例えばサイン波PE）を32×32に生成してマスクデコーダに与えることも考えられますが、設計を簡素に保つためここでは省略します。

### テキストプロンプト射影 (Text Prompt Projector)

LLMデコーダが出力した `<SEG>` トークン位置の隠れ状態ベクトル（サイズ \$D\_l\$, Qwen-3Bでは約2560次元）を、SAMマスクデコーダ用の**プロンプトベクトル**に変換するモジュールです。シンプルに**1層のLinear**で次元を圧縮します（必要に応じReLUを挟んだ2層MLPに拡張も可能ですが、まずは線形で十分です）。以下に実装します。

```python
class TextPromptProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
    def forward(self, text_hidden_state):
        # text_hidden_state: Tensor [B, D_l]
        return self.fc(text_hidden_state)

Dl = qwen_model.config.hidden_size  # Qwenデコーダ隠れ次元 (例: 2560)
text_prompt_proj = TextPromptProjector(in_dim=Dl, out_dim=256)
```

* **入力:** LLMデコーダ最終層から取得した `<SEG>` トークン位置の隠れ状態 `[B, D_l]`。各サンプルで1個（または複数）の `<SEG>` に対応するベクトルです。
* **処理:** 線形変換で次元を `$D_l \to 256$` にマッピングします。出力次元256はSAM Mask Decoderの**クエリ埋め込み**と同じサイズです（SAMではポイントやボックスEmbeddingも256次元）。
* **出力:** `[B, 256]` の**テキスト由来プロンプト埋め込み** \$p\_{\text{text}}\$。これは後でMask Decoderにクエリとして与えられ、該当オブジェクトのマスクを生成する鍵になります。

この設計により、LLM内部に保持された**対象物の意味情報**を直接セグメンテーションに利用できます。LISAではこれを **“embedding-as-mask”** と呼び、画像中の対象を指すテキストの埋め込みをそのままマスク生成に結び付けています。これは、従来の明示的な検出ボックス生成→セグメントという二段階よりも**効率的かつ強力**であることが報告されています。

### 統合モデルクラス LISA改

以上のQwenモデル、SAMマスクデコーダ、画像アダプタ、テキスト射影モジュールを統合し、一つのモデルクラスとして実装します。これにより**エンドツーエンドで画像を入力し、テキストとマスクを組み合わせて出力**できるようになります。解像度差の処理もこのクラス内で対処します。

```python
class LISA_Kai_Model(nn.Module):
    def __init__(self, qwen_model, sam_mask_decoder, image_adapter, text_prompt_proj):
        super().__init__()
        self.qwen = qwen_model                  # Qwen2.5-VL-3B (ViT + LLMデコーダ)
        self.sam_mask_decoder = sam_mask_decoder  # SAM2.1 Mask Decoder（原則凍結）
        self.image_adapter = image_adapter      # 画像特徴次元変換アダプタ
        self.text_prompt_proj = text_prompt_proj  # テキストembedding射影
        # SAMマスクデコーダのパラメータはデフォルトで凍結（後でLoRA調整のみ許可）
        for param in self.sam_mask_decoder.parameters():
            param.requires_grad = False

    def forward(self, input_ids, pixel_values, attention_mask=None, labels=None, mask_labels=None):
        B = input_ids.size(0)
        # 1. Qwenモデルで画像＋テキストを処理（必要な隠れ状態を取得）
        outputs = self.qwen(input_ids=input_ids, attention_mask=attention_mask,
                             pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
        logits = outputs.logits                # [B, seq_len, vocab_size] テキスト出力分布
        hidden_states = outputs.hidden_states[-1]  # 最終層隠れ状態 [B, seq_len, D_l]
        
        # 2. Qwen内部のViT出力特徴を取得
        if hasattr(self.qwen, "vision_tower"):
            vision_feats = self.qwen.vision_tower[0](pixel_values)  # [B, N_patch, D_v]
        else:
            # （注: Transformers実装によってはvision_tower属性が無い場合も）
            vision_feats = outputs.vision_hidden_states  # 仮：画像埋め込み出力を取得できる場合
        
        # 3. 画像特徴をSAM入力形式に変換
        image_features = self.image_adapter(vision_feats)  # [B, 256, H_feat, W_feat]
        
        # 4. <SEG>トークンの隠れ状態を取得しプロンプト埋め込みに変換
        if labels is not None:
            # 学習時：教師ラベル中の<SEG>位置を検出
            seg_indices = []
            for i in range(B):
                seg_pos = (labels[i] == seg_token_id).nonzero(as_tuple=True)[0]
                seg_indices.append(seg_pos[0].item() if len(seg_pos)>0 else None)
        else:
            # 推論時：入力（生成済み）シーケンス中の<SEG>位置を検出（最後の発話に含まれると仮定）
            seg_indices = []
            for i in range(B):
                if seg_token_id in input_ids[i]:
                    seg_idx = (input_ids[i] == seg_token_id).nonzero(as_tuple=True)[0][-1].item()
                    seg_indices.append(seg_idx)
                else:
                    seg_indices.append(None)
        
        # 5. 各サンプルについてマスク生成
        mask_logits = []
        for i, seg_idx in enumerate(seg_indices):
            if seg_idx is None:
                mask_logits.append(None)  # このサンプルではマスク生成なし
            else:
                # 対応する隠れ状態ベクトルを取得しプロンプト埋め込みへ
                seg_hidden = hidden_states[i, seg_idx, :]         # [D_l]
                prompt_embed = self.text_prompt_proj(seg_hidden)  # [256]
                # SAMのマスクデコーダでマスク推論
                mask_out = self.sam_mask_decoder(
                    image_embeddings=image_features[i].unsqueeze(0),  # [1, 256, H_feat, W_feat]
                    # （注: 実際のSAM2.1実装では image_pe や複数promptも指定可能）
                    prompt_embeddings=prompt_embed.view(1, 1, -1)    # [1, 1, 256]
                )
                # 出力マスク取得（dict形式対応）
                low_res_mask = mask_out["masks"] if isinstance(mask_out, dict) else mask_out  # [1, 1, H_mask, W_mask]
                # 6. マスクを元画像サイズにアップサンプリング
                orig_h, orig_w = pixel_values.shape[2], pixel_values.shape[3]
                mask_full = torch.nn.functional.interpolate(low_res_mask, size=(orig_h, orig_w), mode="bilinear")
                mask_logits.append(mask_full[0, 0])  # [orig_h, orig_w] 2次元マスク
        return logits, mask_logits
```

（コード説明）:

* コンストラクタ `__init__` では、事前読み込み済みの `qwen_model`、`sam_mask_decoder`、および上記 `image_adapter`・`text_prompt_proj` モジュールを受け取り、それぞれメンバ変数に格納します。SAMのマスクデコーダは**基本凍結**とし、`requires_grad=False` を設定しています（後述のLoRA部分以外は学習させません）。こうすることで、**SAM2.1が持つピクセル精度の高いセグメンテーション能力を維持**します。実際、LISA研究ではSAMデコーダを凍結した方が性能が良く、ファインチューニングすると却って精度が落ちたと報告されています。

* `forward` メソッドでは、画像テンソル `pixel_values`（形状 `[B,3,H_{\text{model}},W_{\text{model}}]`）とテキスト系列 `input_ids`（および必要に応じて`attention_mask`）、学習時は正解ラベル `labels`（テキスト用）と `mask_labels`（マスク画像用）を受け取ります。各ステップは以下のとおりです。

  1. **Qwenによるテキスト生成:** `self.qwen(...)` を呼び出し、画像とテキスト入力から次トークン予測を得ます。`output_hidden_states=True` としているので全層の隠れ状態が `outputs.hidden_states` に格納されます。ここから最終層隠れ状態列 `hidden_states` を取得します（形状 `[B, seq_len, D_l]`）。また、`outputs.logits` から言語モデルの次トークン確率分布が得られます。Qwen2.5-VLはVision Transformerによる画像特徴抽出と、マルチモーダルデコーダによるクロスアテンションを内部で行っています。例えば、Qwenは物体の位置を安定したJSONで出力する能力も持つため、その内部では視覚的な位置情報が適切に隠れ状態に埋め込まれていると考えられます。LISA改ではこの**位置情報埋め込みを直接マスクに活用**します。

  2. **ViT出力特徴の取得:** Qwenモデル内部から視覚エンコーダの出力特徴列を取り出します。HuggingFace実装では `model.vision_tower` プロパティにViTが格納されているため、`vision_tower[0](pixel_values)` で計算しています（この部分は実装依存です）。ここで `pixel_values` はモデル入力用にリサイズ・正規化済みの画像テンソルです（前述のように元画像を448×448等に調整）。`vision_feats` は `[B, N_patch, D_v]` のテンソルで、(例) \$N\_{patch}=1024, D\_v=768\$ といった形になります。

  3. **画像特徴のアダプタ変換:** (2)で得た `vision_feats` を `self.image_adapter` に通し、特徴マップ `image_features` を生成します。形状は `[B, 256, H_{feat}, W_{feat}]` で、\$H\_{feat}\times W\_{feat}=N\_{patch}\$ を満たします。例えば入力画像448×448なら \$H\_{feat}=W\_{feat}=32\$、256チャネルのマップになります。この特徴マップは**SAMマスクデコーダへの画像エンベディング**に相当します。

  4. **`<SEG>` トークン位置の特定:** **学習時**（`labels`あり）と**推論時**（`labels`なし）で処理が異なります。学習時には教師ラベル`labels`内に `<SEG>` トークンが含まれているので、それを検索します（各サンプルにつき通常1箇所）。推論時には、モデル生成出力（`input_ids`）内に `<SEG>` が現れたかをチェックします。ここでは簡単のため「最後に出力された `<SEG>`」のインデックスを取得しています。こうして各バッチ要素ごとに `<SEG>` の位置 `seg_idx` をリスト `seg_indices` に集めます（存在しない場合は None）。

  5. **マスクデコーダへの入力準備と推論:** ループで各サンプルの `seg_idx` を処理します。`seg_idx` が None の場合（そのサンプルでマスク出力が不要な場合）は `mask_logits.append(None)` とし、以降の処理をスキップします。そうでなければ:

     * デコーダ隠れ状態テンソル `hidden_states[i, seg_idx, :]` を取得します（サイズ \$D\_l\$）。これは `<SEG>` に対応する埋め込みベクトル \$\tilde{h}\_{seg}\$ です。
     * それを `self.text_prompt_proj` に通じて 256次元の `prompt_embed` ベクトルに変換します（サイズ `[256]`）。これが**マスク用プロンプト埋め込み** \$h\_{seg}\$ に相当します。
     * **SAMマスクデコーダ** (`self.sam_mask_decoder`) を呼び出し、画像特徴`image_features[i]`（サイズ `[1,256,H_{feat},W_{feat}]`）と、プロンプト埋め込み（ここでは dense embedding として `[1,1,256]` のテンソルにreshape）を与えてマスク推論させます。SAM2.1のMask Decoderは本来、画像埋め込みと位置エンコード、さらに点・ボックス等の**プロンプト埋め込み**（疎・密両方）を受け取り、対象のマスクを出力します。本ケースではテキストに由来する埋め込みを一つ与えるだけなので、SAMには「1つのプロンプトに対応するマスクを1枚生成する」動作をしてもらいます。SAM2.1は曖昧なプロンプトに対して複数マスク候補を返す機構もありますが、LISA改では1プロンプト=1マスクに限定します。

     ※ **解像度ギャップへの留意:** SAMマスクデコーダは通常**元画像の1/4解像度**程度のマスク（例えば256×256）を内部計算します。今回、画像埋め込みが32×32と粗いため、おそらくMask Decoder内部でさらに4倍の128×128程度のマスクが得られる想定です（厳密にはSAM2.1の設定依存）。このように**Qwen特徴はSAM標準より粗い解像度**ですが、学習時にLoRA調整などでMask Decoderが対応することを期待します。後述するように、LISA研究でも**バックボーンをMask2Former等に置換**した場合でも良好な性能が出ており、多少粗い特徴マップでも十分可能性があります。

     * Mask Decoderの出力 `mask_out` は、実装によって`{"masks": tensor(...), ...}` の辞書形式か、Tensor直接かが異なるため、それを判定して `low_res_mask` テンソルを取得します。サイズは `[1, 1, H_{\text{mask}}, W_{\text{mask}}]`（例: `[1,1,128,128]` など）です。

  6. **マスクのアップサンプリング:** 最後に、(5)で得た低解像度マスク `low_res_mask` を**元画像サイズ** `(orig_h, orig_w)` に `F.interpolate`（双線形補間）で拡大します。ここで `orig_h, orig_w = pixel_values.shape[2], pixel_values.shape[3]` ですが、`pixel_values` はモデル入力画像（リサイズ後）のサイズです。前述のようにモデル入力時にパディング等で**元画像と同じアスペクト比**を保っているため、アップサンプリングしたマスクは元画像上に正しく対応します（不要領域は後処理でクロップ可能）。得られた `mask_full` を `mask_logits` リストに格納します。最終的に `mask_logits` は長さBのリストで、各要素が `[orig_h, orig_w]` サイズ（2次元テンソル）の**マスク出力**です（マスク非対象サンプルは None）。

* 関数の戻り値は `(logits, mask_logits)` のタプルとします。`logits` は通常の次トークン予測用テンソル、`mask_logits` は各サンプルのマスク予測結果を格納したリストです。学習時はこれらから損失を計算し、推論時は `mask_logits` を閾値処理してバイナリマスク画像として出力します。

**解像度に関する補足:** 本実装では、**モデル内部でマスクは低解像度で計算**し、**出力時および損失計算時に高解像度へ拡大**しています。これは計算効率と精度のトレードオフ上、合理的な手法です。特に**学習時のマスク損失**については、元の高解像度GTマスクと比較するため、一旦アップサンプルしています。しかしこの際、粗い予測マスクは境界がぼやけ細部が再現できない可能性があります。その対策としてDice損失を組み合わせ（後述）、**形状の重なり具合**で評価することで小さなズレに寛容にしています。一方で、もし高解像度GTマスクの微細な部分をあまり重視しないのであれば、**GTマスクをモデル出力の低解像度にダウンサンプル**して損失計算する方法もあります。今回は「学習目的のみで解像度制約は強くない」との前提より【ユーザ質問3】、極力GTの細部も活かすため**アップサンプルして高解像度で比較**する設計にしています。

以上で、モデルの順伝播処理（forward）の流れが定義できました。次に、このモデルを効率よく学習させるためのパラメータ凍結戦略とLoRA適用について説明します。

## パラメータ凍結と LoRA による軽量微調整

**学習させるパラメータの選択:** LISA改では、事前学習済みモデルの知識を極力保持しつつ、新たな機能（セグメンテーション）に対応させるため、多くのパラメータを凍結し一部のみ更新します。特にQwen2.5-VL-3Bのような大規模モデルをフルチューニングするのは現実的ではないため、Low-Rank Adaptation (LoRA) を用いて効率的に適応させます。

* **Qwen（LLM部分）の凍結:** Qwenモデル（ViTエンコーダ＋トランスフォーマーデコーダ）の全パラメータを一旦 `requires_grad=False` に設定し凍結します。これにより、Qwenが持つ巨大な言語知識・視覚知識（約30億パラメータ）はそのまま維持されます。特に視覚エンコーダViTは事前学習済みの強力な特徴抽出器として固定し、出力特徴の**分布を変えない**ようにします。LISA論文のアブレーションでも、視覚バックボーン（SAMのimage encoder等）は**事前学習重みで初期化し凍結した方が良かった**と報告されています。また言語デコーダ部分も元の流暢な応答能力を保つため極力動かしません。

* **特殊トークン埋め込みの学習:** 凍結した中で、唯一**新規追加した `<SEG>` トークンの埋め込み**はランダム初期化のため学習させる必要があります。そこで、単語埋め込み行列から `<SEG>` に対応するベクトルのみ `requires_grad=True` と設定します。このベクトルは学習データを通じ、<SEG>の意味（「ここでマスク生成」）を表現するようになります。

* **Qwen内のLoRA挿入:** 次に、Qwenデコーダ内の**画像関連部分**にLoRAを適用します。具体的には、**クロスアテンション層**（画像のViT出力に注意を割く部分）の重みにLoRAモジュールを仕込みます。LoRAは例えば各注意層の\$W\_Q, W\_K, W\_V\$行列に対して低ランク行列を追加学習する手法です。これにより、Qwenの**画像融合挙動**のみを柔軟に調整できます。たとえば、モデルが `<SEG>` を出力するタイミングや、埋め込みに必要な情報を取り込む度合いを最適化できます。

* **SAM2.1側の凍結:** SAMのマスクデコーダ (`sam_mask_decoder`) は既に凍結済みです。SAM2.1は**大量の画像・動画で学習済み**であり、あらゆる物体のマスクを切り出す汎用能力を持つため、むやみに調整すると性能が劣化しかねません。実際、OneTokenSegAllやPixelLMなど多くの追随手法も、**LLMとセグメンテーションモデルを疎に接続する**デザインを採用しています。今回はSAMデコーダは完全凍結とし、必要であればごく一部の重みにLoRAを当てる程度に留めます（例えばMask Decoder内のCross-AttentionにLoRAを追加することも技術的には可能です）。

* **アダプタ部の学習:** 新規追加した **ImageFeatureAdapter**（線形射影）と **TextPromptProjector**（線形射影）はパラメータ数が小さいため、フルに学習させます。これらはランダム初期化であり、学習データを通じて**Qwen特徴とSAMデコーダの橋渡し**に最適化されます。

以上をコードで設定します。

```python
lisa_model = LISA_Kai_Model(qwen_model, sam_mask_decoder, image_adapter, text_prompt_proj)

# Qwenモデル全体を一旦凍結
for param in lisa_model.qwen.parameters():
    param.requires_grad = False

# 特殊トークン <SEG> の埋め込みのみ学習可に
word_embeddings = lisa_model.qwen.get_input_embeddings()  # Qwen単語埋め込み層
word_embeddings.weight[seg_token_id].requires_grad = True

# LoRAをQwenのクロスアテンション層に適用 (PEFTライブラリ使用)
from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(
    r=8, lora_alpha=32,
    target_modules=["cross_attn", "cross_attention"],  # クロスアテンション層を指定（実際のモジュール名に応じ調整）
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
lisa_model.qwen = get_peft_model(lisa_model.qwen, lora_config)

# 画像アダプタとテキスト射影モジュールは全パラメータ学習可
for param in lisa_model.image_adapter.parameters():
    param.requires_grad = True
for param in lisa_model.text_prompt_proj.parameters():
    param.requires_grad = True

print("Trainable parameters:",
      sum(p.numel() for p in lisa_model.parameters() if p.requires_grad))
```

（コード説明）:

* `lisa_model.qwen.parameters()` をループし全て `requires_grad=False` に。Qwenの既存パラメータはこれで固定されます。
* 次に `get_input_embeddings()` で単語埋め込み層を取得し、`weight[seg_token_id]`（`<SEG>` の埋め込みベクトル）だけを `True` にします。こうして `<SEG>` ベクトルのみ勾配更新されるようにします。
* HuggingFaceのPEFTライブラリを用いて、LoRAをQwenモデルに適用します。`LoraConfig` で `r=8`（ボトルネック次元）、`lora_alpha=32`（スケーリング）、`lora_dropout=0.1` を設定し、`target_modules` に "cross\_attn" などクロスアテンション層名の一部を指定します（実際のQwen実装に応じ変更）。`get_peft_model` により Qwen モデルがLoRAラップされ、内部にLoRA用の小さなパラメータが追加されます。これらLoRAの重み（各層数万パラメータ程度）だけが学習対象となります。
* `image_adapter` と `text_prompt_proj` の全パラメータも `requires_grad=True` のままです（初期化時デフォルトでTrue）。これらは非常に軽量（数十万パラメータ）なので問題なく学習できます。
* 最後に学習対象パラメータ数を表示しています。Qwen2.5-VL-3B全体では30億以上ありますが、LoRA + アダプタ + `<SEG>` embeddingのみなら**数百万以下**に激減しているはずです。例えばLoRA（rank8で複数層）の合計が数百万、アダプタが約20万、特殊トークン埋め込みが数千程度となります。

**この戦略の利点:** 事前学習済みLLMとセグメントモデルの知識を**最大限保持**しつつ、新タスクへの対応に必要な**最小限の自由度**だけ学習させる点です。の示す通り、SAMの能力は凍結して使った方が良い結果を生む上、LLMの言語能力も保たれます。また計算資源の節約にもなり、例えば7Bや72Bモデルに拡張する場合もQLoRAなどと組み合わせれば現実的に微調整可能です。

## マルチタスク学習と損失関数設計

LISA改は**テキスト応答タスク**（例: 画像キャプション、VQA）と**セグメンテーション出力タスク**を統一的にこなします。そのためトレーニング時の損失関数も両者を組み合わせて定義します。設計の要点は以下の通りです。

* **言語生成損失 (\$L\_{\text{LM}}\$):** 画像キャプションやQA回答などテキストが正解として与えられる部分には、通常の**因果言語モデルのクロスエントロピー損失**を適用します。すなわち、モデルの出力logitsと教師テキスト系列を比較し、正解トークンに対応する対数尤度を最大化します。実装上は `torch.nn.functional.cross_entropy` を使用し、教師系列中学習に無関係な部分（例: ユーザ発話）は -100 にマスキングして無視します。

* **セグメンテーション損失 (\$L\_{\text{seg}}\$):** 画像中の対象物マスクが教師として与えられる場合、モデルが生成したマスクとGTマスクを比較する損失を定義します。具体的には以下の二つを組み合わせます。

  * **ピクセル単位のBinary Cross Entropy (BCE) 損失:** 予測マスク各画素の確率 \$\hat{M}(x,y)\$ と真のマスクラベル \$M^\*(x,y) \in {0,1}\$ との間で交差エントロピーを計算します。実装では `F.binary_cross_entropy_with_logits` を用いて、シグモイド適用込みの安定した損失計算をします。
  * **Dice損失:** 予測マスクとGTマスクの重なり（IoUに類似）に基づく損失です。Dice係数の微分可能な形式として、\$L\_{\text{Dice}} = 1 - \frac{2|\hat{M} \cap M^*| + \epsilon}{|\hat{M}| + |M^*| + \epsilon}\$ を使用します（\$\epsilon\$はゼロ除算防止項、\$\hat{M}\$はシグモイド出力を連続値のまま使用）。Dice損失は領域の一致度を評価するため、境界付近の微小なズレに対してBCEより寛容であり、粗いマスク予測でも形状が合っていれば低く抑えられます。BCEと組み合わせることで、**細部のピクセル一致と全体形状の一致**のバランスを取ります。

* **統合損失:** 最終的な損失は \$L = L\_{\text{LM}} + \lambda , L\_{\text{seg}}\$ で与えます。ここで \$\lambda\$ はタスク間の重みです。初期段階では \$\lambda=1\$ とし、学習の様子を見て調整します。この加重和により、言語とマスクの勾配が同時にモデルを更新し、モデルは両タスクを両立するようになります。

以下に損失計算のコード例を示します。

```python
import torch.nn.functional as F

def compute_loss(logits, labels, mask_logits=None, mask_labels=None):
    # 言語損失: 次トークンのCrossEntropy（ラベルが-100の部分は無視）
    vocab_size = logits.size(-1)
    lm_loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100)
    
    # セグメンテーション損失（maskがある場合）
    if mask_logits is not None and mask_labels is not None:
        # BCE損失（with logitsで安定計算）
        bce = F.binary_cross_entropy_with_logits(mask_logits, mask_labels.float())
        # Dice損失
        pred_mask = torch.sigmoid(mask_logits)
        B = mask_labels.size(0)
        pred_flat = pred_mask.view(B, -1)
        true_flat = mask_labels.view(B, -1)
        intersection = (pred_flat * true_flat).sum(dim=1)
        dice = 1 - (2. * intersection + 1e-5) / (pred_flat.sum(dim=1) + true_flat.sum(dim=1) + 1e-5)
        dice = dice.mean()
        seg_loss = bce + dice
    else:
        seg_loss = 0.0
    
    total_loss = lm_loss + seg_loss
    return total_loss, lm_loss, seg_loss
```

（コード説明）:

* `F.cross_entropy` は自動的にsoftmax計算と対数取りを行った上で平均損失を返します。`ignore_index=-100` を指定することで、教師中 -100 にマスクされた部分は勾配に寄与しなくなります（学習データではユーザ発話やシステムプロンプト部分を-100にしておき、アシスタントの回答部分と `<SEG>` のみ損失計算対象にします）。
* セグメンテーション損失計算では、まず `binary_cross_entropy_with_logits` を用いてBCE損失を計算しています。これは内部でシグモイドを取ってから \$-\frac{1}{N}\sum \[M^*\log\hat{M} + (1-M^*)\log(1-\hat{M})]\$ を計算します。次にDice損失では、`pred_mask = torch.sigmoid(mask_logits)` として確率マップを得てから計算しています。二値化はせず連続値のまま計算することで微分可能にしています（soft Dice）。
* 戻り値として、`total_loss`（合計損失）のほか、`lm_loss` と `seg_loss` も返しています。学習中にそれぞれの大きさをモニタリングする用途です。

**解像度の扱い:** ここで `mask_logits` と `mask_labels` はともに `[B, H_{orig}, W_{orig}]` サイズを想定しています（前節のforwardでアップサンプル済みマスクを使用）。そのためGTマスク `mask_labels` は元画像解像度そのまま（もしくはデータセット統一解像度）で持っておきます。前述の通り、アップサンプルした予測マスクを直接GTと比べることで、高解像度での一致を促します。一方、もしGTマスクを学習時に予め下げた解像度で統一しておけば（例えばすべて256×256等）、そのサイズで比較することもできます。今回ユーザの要望では「GTマスクは特段1024×1024にこだわらず適切に」とあるため【ユーザ質問1】、**データセット内では可能な限り元解像度を保持しつつ**、計算量に応じて適宜リサイズする方針とします。例えば、非常に高解像度のマスクデータは学習前処理で512×512程度に縮小しておき、モデルもそれに合わせて入力画像サイズを調整する、といった方法です。この辺りはハードウェアリソースと相談になります。

**データセットの混合同時学習:** LISA改の訓練には複数種類のデータを組み合わせます。

* **画像キャプションデータ:** (例: COCO Caption) 画像+説明文。マスク出力は無いので \$L\_{\text{LM}}\$ のみ計算。
* **視覚質問応答データ:** (例: VQAv2やLLaVA-Instruct-150k) 画像+質問文+回答文。これもテキスト応答のみで \$L\_{\text{LM}}\$ のみ。
* **参照セグメンテーションデータ:** (RefCOCO/+/gやRefCLEFなど) 画像+「指示文（明示的）")+対象物マスク。モデルには`USER: <IMAGE> この指示に従いマスクして\nASSISTANT: <SEG>` のような形式で与え、マスクとテキスト両方の損失を計算します。
* **通常セグメンテーションデータ:** (COCO-Stuff, ADE20K, PACOなど) 画像+カテゴリ名（単語や短文）+対象マスク。これも上と同様、指示を組み立て `<SEG>` 出力を学習させます。これらは豊富なマスク例を提供するため重要です。
* **推論セグメンテーションデータ:** LISAで新たに収集された ReasonSeg (約1218組) 等。**「暗黙的な指示」によるマスク**のデータです。例えば画像と言葉遊び的な質問（「この画像で普通でない部分は？」）+マスク。これらは高度な推論が必要ですが、LISAでは**学習になくともゼロショットである程度対応**できることを示しています。とはいえ最終的な精度向上のためには一部を学習に含め fine-tuning すると良いでしょう。本モデルでもデータがあれば取り入れます。ユーザ提示では「学習目的のみで解像度制約なし」とあり、ReasonSegのGTマスクは1024×1024に整形しているようですが、こちらも統一せず扱えます。

これらデータを**1つのバッチ内で混ぜて学習**することで、モデルは多様なタスクに対応できるようになります。実装上、DataLoaderで各種データをサンプリングして混合し、`mask_labels` が存在しない場合は自動的にセグ損失を0にする、といった仕組みで同時学習を進めます。LISA論文では**推論的セグメンテーション（ReasonSeg）データは239件だけfine-tuning**に使えば充分とされていますが、本モデルでは最初から一部含めて学習させても問題ありません（むしろユーザ指示に応じて）。

## 実行時の挙動と応答フォーマット

学習を経たLISA改モデルの**推論時**の動きを確認します。モデルは画像と言語の入力に対し、テキスト＋セグメントマスクを組み合わせた応答を生成します。いくつかのシナリオを例に挙げます。

* **セグメンテーション指示への応答:** ユーザが「この画像のリンゴを全てマスクで示してください」と画像付きで依頼した場合、モデルは内部でまずテキスト指示を解析し、リンゴという対象を認識します。LLMは応答生成中に `<SEG>` を出力し、その隠れベクトルを用いてMask Decoderが画像中のリンゴ領域を推定します。出力はリンゴ領域を白抜き等でハイライトした**バイナリマスク画像**となります。モデルは加えて「リンゴをマスクしました。」等のテキストを返すことも可能です。最終的なユーザへの応答は**テキスト＋画像**となります（システム上はまずテキストが送られ、続いて画像ストリームが送信される形になります）。

* **質問応答への応答:** ユーザが「この料理には何が写っていますか？」と画像付きで質問した場合、モデルは**マスクモードに入らず**通常のテキスト応答のみを行います。「トマトとバジルが乗っています」のような回答を返し、`<SEG>` は一切出力しません。これは学習時にそのようなデータでは常にテキストで答えるよう教師しているためで、モデルが不必要に `<SEG>` を出すことはありません。

* **複合指示への応答:** 「この料理に何が何個写っていますか？できればそれぞれ強調してください。」のように、まず質問に答え、さらにハイライトも要求されるケースでは、モデルは**テキストとマスクを組み合わせた応答**を行います。例えば「リンゴは3個あります。【マスク画像】」のように、文章で数を答えつつ `<SEG>` で3つのリンゴをまとめてマスク出力します（もしくは複数回の `<SEG>` で個別に出力することも考えられますが、UI上1枚で対応可能なら1回にまとめます）。このような複合出力はデータ次第ですが、LISAでは**マルチターン対話や理由説明付きマスク**も可能であることが示されました。LISA改も適切に学習すれば類似の応答が期待できます。

**実装上の注意:** 推論時は通常、`model.generate()` 等でテキストを自動回帰生成しますが、`<SEG>` トークンが出力されたら**一時的に生成を中断しマスクを計算**し、その結果を画像として保存・提示する必要があります。この制御はシステム側で適宜組む必要があります。すなわち、生成ループ内で `<SEG>` を検出したら、現時点のデコーダ隠れ状態を取り出し、本モデルの `forward` を呼んで `mask_logits` を得てから、次のテキスト生成へ進む、というカスタムステップになります。今回のモデル定義では簡単のためこの処理も forward 内で行っていますが、実運用では**モデル部分と出力制御部分を明確に分離**すると良いでしょう。

## 解像度ギャップと性能に関する考察

最後に、**Qwenの特徴解像度とSAMのマスク解像度のギャップ**について補足し、本モデルの性能ポテンシャルを考察します。Qwen2.5-VLは視覚エンコーダの出力解像度こそ中程度ですが、**高度な物体認識・定位能力**を持っています。例えばテキストとしてバウンディングボックス座標を安定出力できることは、モデル内部で視覚要素を構造化して捉えている証拠です。一方SAM2.1は微細な輪郭まで切り出せる汎用セグメンターであり、視覚的に類似した物体の識別や被写体の一部隠れにも強い耐性があります。

LISA改では、この両者を**エンドツーエンド統合**することで、ユーザの指示を深く理解し（LLM部分）つつ、該当物体をピクセルレベルで正確にハイライトすることが可能となります。モデル内部でembeddingを介して視覚と言語を接続する構造は、従来のAPI連携型（二段階）手法に比べ**効果的である**ことが報告されています。実際、LISAではこの統合型アプローチにより、単にRefCOCOのような明示指示だけでなく、**高度な推論を要するセグメンテーション**（理由を考えなければ何を指示しているか分からないようなケース）にも対応できています。一方で、後発のSeg-ZeroなどではLLMに座標出力をさせてから固定のSAMに渡すという手法も提案されており、データが少ない場合には有効という知見もあります。本モデルは十分に事前学習されたLLMとSAMをベースとしているため、まずは統合学習で性能を引き出し、もし課題が残れば座標出力モードへの切り替えなども検討できる柔軟性があります。

LISA改で得られたモデルは、将来的に様々な応用が可能です。例えば**料理画像の解析**では、まず料理領域をマスクで正確に抽出し（量や範囲を把握）、その上でカロリー推定や比較を行うなど、より高精度なビジョン＆ラングエージタスクに繋げられます。ユーザが興味を持っていると思われるFoodLMMのような課題にも、LISA改はまず料理をセグメントする役割として統合できます。また、医用画像で特定の所見を指摘するといった高度専門領域にも、LLMの知識とセグメンテーション能力の組み合わせは有用でしょう。

以上、Qwen2.5-VL-3BとSAM2.1を統合した「**LISA改**」モデルの設計ガイドと実装例を示しました。各モジュールの入出力shape、接続方法、学習戦略、解像度の扱いについて詳細に述べていますので、これを基に実装・学習を進めれば、ユーザ要望に沿った**高精度なセグメンテーション対応マルチモーダルモデル**が構築できるはずです。最後に、本設計の根拠となった参考文献を挙げておきます。

**参考文献:** Qwen2.5-VL 技術報告、LISA: Reasoning Segmentation via LLM 論文および解説記事、Segment Anything Model 2.1 発表ブログ、Seg-Zero 論文など。これらは本回答中に適宜引用しています。各引用番号をクリックすると詳細が参照できます。
