# LISA改モデルへのSAM ViT特徴統合の詳細設計

## 現状のモデル構造とSAM ViT未使用部分

現在、LISA改モデルでは**Qwen2.5-VL**の視覚エンコーダ（ViT）から得られた特徴ベクトルのみを使用し、**SAM2.1**の画像エンコーダ（ViT）出力は学習・推論ともに直接使用していません。データ処理段階では、`HybridDataset`にて入力画像を**448px**程度（Qwen用）と**1024px**（SAM用）に別々に前処理し、後者は`sam_images`としてバッチに含まれています。実際にモデルの`forward`関数も`sam_images`引数を受け取るよう定義されていますが、現時点ではこれを使った処理が実装されておらず、`SAM ImageEncoder`のパラメータは凍結されたまま「未使用」です。

> **補足:** データセット前処理では、`dataset.py`内の`preprocess_sam_image`で入力画像を**1024×1024**にリサイズ・パディングし正規化する処理が既に実装されています。この結果が`sam_images`テンソルとしてモデルに渡されており、今後この高解像度画像をSAMのViTに投入して特徴を取得できる下地が整っています。

## SAM ViT出力を統合するステージ

**統合を行うタイミング**としては、**マスクデコーダに画像特徴を入力する直前の段階**が適切です。具体的には、Qwenの視覚特徴（現状は**256チャネル・空間解像度ℎ×𝑤の特徴マップ**に変換済み）と、SAM2.1の画像エンコーダから得られる**256チャネル・空間解像度64×64の特徴マップ**を融合し、**単一の画像埋め込み**としてSAMのMaskDecoderに渡す設計にします。これは、LISAモデルがLLM内部の視覚特徴を\*\*<SEG>トークン位置の出力埋め込み**として抽出し、それをSAMのMaskDecoderへのプロンプト埋め込みに利用していた流れに倣ったものです。当該部分はLISAで既に導入されている「マスク埋め込みパラダイム」であり、**LLM由来のembeddingをSAMに供給してマスクを生成**していました。今回、そこに**SAM画像エンコーダ由来の視覚情報も加える\*\*ことで、両者の利点を活かした高精度なマスク推定を目指します。

統合の**具体的な箇所**は、コード上では`LISA_Model.forward`内で言えばSEGトークン位置検出後、MaskDecoderにマスク予測をさせる直前です。現在はQwen由来の`image_features_sam`（256×H×W）だけをMaskDecoderに渡していますが、この`image_features_sam`を、SAMのViT出力特徴と統合した新しい特徴マップに差し替えます。

## 統合方法の検討（Concat vs Cross-Attention vs 加算融合）

**特徴融合の手法**としてはいくつか考えられますが、**LISAの設計思想**に合わせて**シンプルで効果的な加算型の融合**を採用することを提案します。以下の方法を比較検討しました。

* **チャンネル次元でのConcat＋小規模ネットワーク:** QwenとSAMの特徴マップ（各256チャネル）をチャネル方向に連結（計512チャネル）し、1×1畳み込みなどの**アダプターモジュール**で256チャネルに圧縮する方法です。学習可能な層を挟むことで重み付き統合を実現できますが、パラメータが増加し実装も複雑になります。

* **Cross-Attentionでの融合:** Qwen特徴をQuery、SAM特徴をKey/Valueとする**クロスアテンション層**を挿入し、空間的対応付けを行う方法です。2つの特徴マップ間で複雑な関係を学習できますが、新たにTransformerブロックを追加する大掛かりな改修となります。

* **要素ごとの加算融合（提案）:** **空間位置ごと**にSAMの特徴とQwenの特徴を足し合わせる手法です。単純な加算では寄与の調整ができないため、**学習可能なスカラー係数**でQwen側の寄与を調整します。このアプローチは、LISAモデルでテキストのSEG埋め込みとSAMの位置埋め込みを融合していた方法に類似しています。具体的には、LISAではSAMの座標位置埋め込みに対してLLMからの<SEG>トークン埋め込みを\*\*\$\sigma(\beta)\$でスケーリングして加算\*\*することで、位置情報を保ったまま言語的な意味特徴を注入していました。これにならい、**画像特徴についてもSAMの特徴マップに対しQwenの特徴マップを\$\sigma\$でスケーリングして加算**する形で融合します。こうすることで、SAM ViTが持つ高解像度な境界・位置情報を維持しつつ、Qwen ViTの高レベルな意味特徴を適切に補完的に与えることができます。

以上を踏まえ、本設計では**加算融合方式**を採用します。これなら追加パラメータもごく少なく（スカラー1つのみ）済み、既存コードへの変更も最小限です。また、LISAの「mask-as-embedding」戦略を踏襲しつつ、SAMの視覚的知識を取り込めるため、実装上も一貫性があります。

## 具体的なコード修正案

以下に、必要となる具体的なコード修正内容を示します（`lisa_model.py`を中心に説明します）。

**1. 学習可能パラメータの追加（β係数）**
まず、SAM特徴とQwen特徴の融合比重を制御するスカラー係数として、新たに`prompt_beta`にならぶ\*\*`image_fusion_beta`パラメータ\*\*をモデルに追加します。`LISA_Model.__init__`内で以下を追加定義します。これにより初期値0の学習可能パラメータ`image_fusion_beta`（PyTorchのParameter）が導入されます。

```python
class LISA_Model(nn.Module):
    def __init__(...):
        ...
        # （既存コード略）
        self.prompt_beta = nn.Parameter(torch.zeros(1))  # 既存のテキスト埋め込み融合係数
        self.image_fusion_beta = nn.Parameter(torch.zeros(1))  # ★追加: 画像埋め込み融合係数
        logger.info("Added image_fusion_beta parameter for SAM-Qwen feature fusion")
        ...
```

**2. SAM ViT出力特徴の計算**
`forward`関数内で、渡された高解像度画像テンソル`sam_images`をSAM2.1の画像エンコーダへ入力し、その出力特徴マップを取得します。コード上は以下を追加します。

```python
def forward(..., pixel_values=None, ..., sam_images=None, ...):
    ...
    vision_features = None
    image_features_sam = None
    sam_high_res_features = None

    # 1. Qwenからの視覚特徴抽出（既存処理）
    if pixel_values is not None:
        vision_features = self.extract_vision_features(pixel_values, image_grid_thw)
        image_features_sam = self.image_adapter(vision_features, image_grid_thw)  # [B,256,H_q, W_q]
        # Token-FPNまたはHighResFeatureGeneratorでマルチスケール特徴生成（既存処理）
        if self.use_token_fpn:
            image_features_sam, sam_high_res_features = self.token_fpn(
                image_features_sam, image_grid_thw, use_hooks=True)
        else:
            sam_high_res_features = self.high_res_generator(image_features_sam)
    # 2. SAM ViTからの特徴抽出（★新規追加）
    if sam_images is not None:
        # SAM2.1のImageEncoderに高解像度画像を入力し、特徴マップを取得
        # 出力は[B, 256, 64, 64]になる想定（1024入力時） 
        sam_image_embedding = self.sam_image_encoder(sam_images)  # [B,256,64,64]
        # Qwen特徴マップをSAM特徴と同じ解像度にリサイズ（必要なら）
        if image_features_sam is not None and image_features_sam.shape[-2:] != sam_image_embedding.shape[-2:]:
            image_features_sam = torch.nn.functional.interpolate(
                image_features_sam, size=sam_image_embedding.shape[-2:], mode='bilinear', align_corners=False)
        # Qwen特徴が未計算（pixel_values=None）の場合でもsam_image_embeddingをそのまま使う
    ...
```

上記では、まず`self.sam_image_encoder`（事前に`__init__`で構築済みのSAMモデルのViT部分）に`sam_images`を入力し、**64×64空間解像度・256次元チャネル**の特徴テンソル`sam_image_embedding`を取得しています。次に、既に計算済みの`image_features_sam`（Qwen側特徴）が存在し、解像度がSAM側と異なる場合には、**バイリニア補間**でSAM側に解像度を合わせています。この処理により、\*\*Qwen特徴マップ（H\_q×W\_q）**を**SAM特徴マップ（64×64）\*\*と対応付けられるサイズに変換します（例えばQwenが32×32なら2倍拡大して64×64に）。

**3. Qwen特徴とSAM特徴の融合**
次に、上で揃えた2つの特徴を融合します。学習可能係数`image_fusion_beta`をシグモイド関数で0〜1に変換し、これを用いて**SAM特徴 + β \* Qwen特徴**の加算を行います。コードは以下です。

```python
    ...
    # 3. 特徴マップの融合（★新規追加）
    if sam_images is not None and image_features_sam is not None:
        # β係数を0〜1に正規化
        beta_scaled = torch.sigmoid(self.image_fusion_beta)
        # SAM特徴にQwen特徴を加算融合
        fused_image_embedding = sam_image_embedding + beta_scaled * image_features_sam
    elif sam_image_embedding is not None:
        # Pixel-VL特徴が無い場合はSAM特徴のみ使用
        fused_image_embedding = sam_image_embedding
    else:
        fused_image_embedding = image_features_sam
    # 融合後の埋め込みは[B,256,64,64]（想定）
    ...
```

ここで、`beta_scaled = sigmoid(image_fusion_beta)`により、例えば`image_fusion_beta=0`なら`beta_scaled=0.5`程度、正の値なら0.5超、負なら0.5未満と調整可能です（初期値0なら0.5でスタート）。**SAMの特徴マップ**をベースに、**Qwenの特徴**をこの係数でスケーリングして加算することで融合結果`fused_image_embedding`を得ています。LISAでプロンプト埋め込みにLLM特徴を加算していたのと同様、**空間位置対応を保ちながら語義情報を付加する処理**になっています。なお、もし`pixel_values`が無い場合（画像入力がSAM側のみ提供されるケース）にはSAM特徴のみを使い、逆に`sam_images`が無い場合は従来通りQwen特徴のみを使う分岐も入れています。

**4. 高解像度特徴の再生成**
融合後の`fused_image_embedding`を、MaskDecoderに渡す**画像埋め込み**および**高解像度マルチスケール特徴**として準備します。Qwen特徴単独時と同様に、**stride-4**（4×アップサンプル）と**stride-8**（2×アップサンプル）の特徴を生成し、それぞれMaskDecoderでの32チャネル・64チャネル圧縮に対応させます。既存コードでは、Token-FPN使用時は内部で`upsample_s0`/`s1`を適用済みの`sam_high_res_features`リストが得られていました。しかし融合後は`fused_image_embedding`が新たなベース特徴となるため、これに対して改めてアップサンプルを行います。具体的には:

* **Token-FPNを使用する場合**: `self.token_fpn.upsample_s1`（2倍）と`self.token_fpn.upsample_s0`（4倍）のモジュールを再利用して、`fused_image_embedding`から`feat_s1`（128×128）および`feat_s0`（256×256）の**高解像度特徴**を生成します。これらをリストにまとめ`high_res_features = [feat_s0, feat_s1]`とします。

* **HighResFeatureGeneratorの場合**: Token-FPN未使用時は、既存の`self.high_res_generator`を用いて`fused_image_embedding`から同様に2倍・4倍アップサンプル特徴を取得します（関数がリストで返します）。

コード例:

```python
    # 4. 高解像度マルチスケール特徴の生成（★修正）
    if self.use_token_fpn:
        # Token-FPNのアップサンプラで高解像度特徴生成
        feat_s1 = self.token_fpn.upsample_s1(fused_image_embedding)  # stride-8相当 [B,256,128,128]
        feat_s0 = self.token_fpn.upsample_s0(fused_image_embedding)  # stride-4相当 [B,256,256,256]
        sam_high_res_features = [ self.sam_mask_decoder.conv_s0(feat_s0.to(sam_dtype)),
                                  self.sam_mask_decoder.conv_s1(feat_s1.to(sam_dtype)) ]
    else:
        sam_high_res_features = self.high_res_generator(fused_image_embedding)  # [feat_s0, feat_s1]
        # それぞれconv_s0/conv_s1でチャネル圧縮
        sam_high_res_features = [ self.sam_mask_decoder.conv_s0(sam_high_res_features[0].to(sam_dtype)),
                                  self.sam_mask_decoder.conv_s1(sam_high_res_features[1].to(sam_dtype)) ]
    # MaskDecoderの畳み込み層で256→32/64チャネルに圧縮:contentReference[oaicite:15]{index=15}
    image_features_sam = fused_image_embedding  # MaskDecoderに渡す画像埋め込み
```

上では、`sam_dtype = next(self.sam_mask_decoder.parameters()).dtype`でSAMの型に合わせつつ、MaskDecoder内部の`conv_s0`/`conv_s1`を使ってチャネル圧縮（256→32, 256→64）しています。最終的に、**画像埋め込み**`image_features_sam`（融合256チャネル, 64×64）と**高解像度特徴**`sam_high_res_features`（リストで32チャネル256×256と64チャネル128×128）が準備できました。これらを後段のMaskDecoder呼び出しに渡します（既存コードから`mask_decoder`呼び出し部分は変わりませんが、渡す引数が更新されます）。

**5. MaskDecoderへの入力**
MaskDecoder呼び出し部分では、上で生成した新しい`image_features_sam`と`sam_high_res_features`を使用します。該当箇所（SEGトークンごとのマスク生成ループ内）は次のように修正します。

```python
    low_res_masks, iou_predictions, sam_tokens_out, obj_score_logits = self.sam_mask_decoder(
        image_embeddings=image_features_sam[i:i+1].to(sam_dtype),        # ★融合後の特徴マップ（64x64, 256ch）
        image_pe=image_pe.to(sam_dtype),
        sparse_prompt_embeddings=sparse_embeddings.to(sam_dtype),
        dense_prompt_embeddings=dense_embeddings.to(sam_dtype),
        multimask_output=False,
        repeat_image=True,
        high_res_features=sam_high_res_features,  # ★融合後の高解像度特徴リスト（stride-4,8）
)
```

こうして、MaskDecoderには**QwenとSAMの両ViTの情報が統合された特徴**が入力されます。MaskDecoder内部ではこれらを用いてマスク推定が行われ、従来より高精細で精度の高いマスク出力が期待できます。

> **参考:** 上記修正により、MaskDecoderへ渡す`image_embeddings`は従来のQwen由来のものから**SAM+Qwen融合**に置き換わります。一方、SEGトークンから生成した**テキストプロンプト埋め込み**（`prompt_embeds`）の処理フローは変更ありません。LLM隠れ状態は既存通りProjection層で256次元に射影され、SAM PromptEncoderから得た**位置埋め込み**と**スパース埋め込み**に融合されます（この融合も前述の通り`prompt_beta`による加算です）。画像側の融合と相まって、**言語・画像の両モダリティで深く統合**された特徴に基づきマスク推定が行われることになります。

## 補足: 既存SAM ViT関連コードの所在

ご質問にあった\*\*「SAM ViT関連コードの実装がどのファイルに含まれているか」**について補足します。上述のとおり、データ処理部分では`src/data/dataset.py`内に`preprocess_sam_image`関数があり、**SAM用画像の前処理**（リサイズ＆パディング）を行っています。また`HybridDataset`経由で`sam_images`としてモデルに入力され、モデル側では`LISA_Model`初期化時に**SAM2.1モデルの読み込み\*\*と`sam_image_encoder`の取得を行っています。現状その出力は未使用でしたが、今回の修正で`sam_image_encoder`の出力を活用するようになります。さらに、SAMのMaskDecoderやPromptEncoderは`LISA_Model`内で`self.sam_mask_decoder`や`self.sam_prompt_encoder`として保持されており、前述のようにMaskDecoder内部の畳み込み層（conv\_s0/s1）も利用可能です。

以上の修正案により、最新のVLM（Qwen2.5-VL）の**意味理解力**と、最新SAM2.1の**高解像度セグメンテーション能力**を統合したLISA改モデルとなります。これにより、将来的なFoodLMM改などドメイン特化の微調整を行う際の強力な基盤モデルとなることが期待できます。各処理の入出力次元（例えばSAM ViT出力256チャネル、空間64×64など）もコード内コメントで明示した通りであり、曖昧さのない形で実装可能です。ぜひこの方針で実装を進めてみてください。

**参考ソース:** 特に重要な該当箇所のコード断片を以下に示します。

* SAM ImageEncoderを凍結して未使用だった箇所（現状）
* モデルforwardが`sam_images`引数を受け取る定義
* Collatorが`sam_images`をバッチに含めている箇所
* LISA実装での埋め込み加算融合（prompt\_beta使用）

以上を踏まえ、コード実装を行えば、QwenとSAMのViT両方の利点を活かしたLISA改モデルの完成に近づけるはずです。頑張ってください！

