LISA改モデルへのSAM ViT特徴統合（Cross-Attention版）の詳細設計
現状のモデル構造とSAM ViT未使用部分

現在のLISA改モデルでは、Qwen2.5-VL の視覚エンコーダ（ViT）から得られた画像特徴ベクトルのみをマスクデコーダに入力しています。一方、SAM2.1 の画像エンコーダ（ViT）は読み込まれているものの、その出力は学習・推論時に直接使用されておらずパラメータも凍結されています（self.sam_image_encoder.parameters()が全てrequires_grad=Falseになっています）。データ処理上は、HybridDataset内で入力画像をQwen用に約448px、SAM用に1024pxにそれぞれ前処理し、後者はsam_imagesテンソルとしてバッチに含まれモデルのforwardに渡されています
GitHub
GitHub
。実際にLISA_Model.forwardもsam_images引数を受け取る定義になっています
GitHub
が、現状この高解像度画像を用いた処理は未実装であり、SAMのImageEncoder出力は使われないままとなっています
GitHub
。

このように、SAM ViT由来の高精細な空間情報が現状活かされていないため、モデルのマスク精度向上の余地があります。今後、LISAモデル（改）ではSAMの画像特徴も取り込むことで、LLM由来の抽象的な視覚特徴と、SAM ViTの持つピクセルレベルの精密な特徴とを組み合わせ、高精度なセグメンテーションを実現したいと考えています。

補足: データセット前処理ではsrc/data/dataset.py内のpreprocess_sam_image関数にて、入力画像を1024×1024にリサイズ・パディングし正規化する処理が既に実装済みです
GitHub
GitHub
。この結果がtensor化されsam_imagesとしてモデルに渡されており、SAMのImageEncoderに高解像度画像を入力できる下地は整っています。

SAM ViT出力を統合するタイミングと方針

統合のタイミングは、マスクデコーダ（MaskDecoder）に画像特徴を渡す直前が適切です。具体的には、Qwenの視覚特徴（現在はimage_features_samとして256チャネル・空間解像度H×Wで得られている特徴マップ）と、SAM2.1の画像エンコーダ出力特徴（256チャネル・空間解像度64×64を想定）を融合し、単一の統合画像特徴としてMaskDecoderに入力する設計にします。この位置は、LISAモデルにおいてLLM内部から抽出した視覚埋め込みを**「マスク生成用の埋め込み」として利用していた箇所に相当し、そこを強化する形でSAM由来の情報を加える狙いです。既存LISAではLLM出力の<SEG>トークン埋め込みをSAM側のプロンプト埋め込みに転用してマスク生成していましたが、今回は画像特徴レベルで**SAMの視覚情報を統合することで、空間的精度と意味的理解の両面を強化します。

統合の実装方針として、今回は Cross-Attention（クロスアテンション） による特徴融合を採用します。前回の設計案では要素ごとの加算融合（Residual Addition）を用いましたが、クロスアテンションを用いることで位置ごとに動的にQwen特徴を取り入れることが可能となり、より柔軟で強力な融合が期待できます。クロスアテンションは、Transformerの注意機構を応用してある特徴（Query）が他の特徴群（Key/Value）のどの部分に注目すべきかを学習する手法です。これにより、各空間位置の特徴が、それに対応する重要な別モダリティの情報を選択的に取り込めます。今回の場合、SAM ViTの高解像度特徴マップとQwen ViTの高レベル特徴マップの対応付けを学習し、各ピクセル位置ごとに適切な意味情報を付加することが可能になります。

従来手法との比較:

単純加算融合では、全ての位置でほぼ一様な重み（学習可能スカラーβ）でQwen特徴を加えるため、位置対応や文脈に応じた融合調整が困難でした。それに対し、

クロスアテンション融合では、位置ごとに異なるAttention重みを計算し、必要なQwen特徴を必要なだけ取り入れることができます。これは、例えば**「ある領域ではSAM特徴だけで十分境界が明瞭だが、別の領域ではLLM由来の物体カテゴリ情報が補助的に必要」**といった場合に、Attention機構が自動的に寄与度を調整してくれることを意味します。

このアプローチは計算コストが若干増加しますが、Otter + SAMといった先行研究でもクロスアテンション導入によるマスク精度の向上が報告されており
GitHub
、本プロジェクトにおいても有効と判断しました。

クロスアテンション融合の具体的な設計

クロスアテンションによる特徴統合では、Queryを一方の特徴マップ、Key/Valueをもう一方の特徴マップとします。どちらをQueryとするかで設計が変わりますが、ここではSAM側の特徴をQuery、Qwen側の特徴をKey/Valueとする方針を採ります。こうすることで、「細粒度なSAM特徴の各位置」が「高レベルなQwen特徴のどの場所に対応するか」を学習し、その情報を取り込めます。最終的な出力もQueryと同じ空間解像度（64×64）となるため、MaskDecoderへの入力形状に合わせやすい利点もあります。

実装面では、まずQwen特徴マップをSAM特徴マップと同じ空間解像度に揃える前処理を行います。典型的にはQwenのViT出力は32×32程度（448px入力時）なので、バイリニア補間で64×64にアップサンプリングし、各位置が概ね対応するようにします（各Qwenピクセル特徴を2×2ブロックに拡大）。この対処により、クロスアテンションが学習初期からおおよその対応関係を掴みやすくなります。アップサンプリングなしでもAttention自体は機能しますが、Key/Valueの数がQueryより少ない場合（32×32 vs 64×64）には、一つのKeyが複数Query領域に対応することになり、きめ細かな対応付けが難しくなります。そこで、解像度を一致させ1対1対応に近い形から開始することで、より直観的な学習が可能となります。

続いて、Multi-Head Cross-Attention層を設けます。PyTorchのnn.MultiheadAttention（embed_dim=256, num_heads=8程度を想定）を用い、Query=SAM特徴、Key=Value=Qwen特徴として注意機構を計算します。各ヘッドが特徴空間を分担して注意重みを計算し、出力として各Query位置に対応する融合後特徴を得ます。出力のテンソル形状は[B, 4096, 256]（B:バッチサイズ, 4096=64×64, 256チャネル）となり、これを[B, 256, 64, 64]にリシェイプすれば、統合後の特徴マップが得られます。

Residual接続と正規化: クロスアテンション出力には残差接続を加えることを推奨します。すなわち、得られた融合特徴に元のSAM特徴（Query）を加算し、その上でLayerNormによる正規化を行います。こうすることで、元の高解像度な境界情報を損なわずに必要な意味情報だけを付加する形になります。これはTransformerブロックの標準的な構造で、学習の安定性も高めます。残差を入れない場合、クロスアテンションが不完全な初期段階で元の詳細情報が失われてしまう恐れがありますが、残差ありなら常に元特徴がベースとして残るため安心です。

ゲーティング機構（オプション）: さらに高度な融合調整のために、動的なゲート機構を導入することも検討できます
GitHub
。ゲーティングとは、クロスアテンション後の新しい特徴と元の特徴をチャネルごとに重み付けて混合する仕組みです。具体的には、新旧特徴を結合した上で全結合層＋Sigmoidからなるゲート値（0〜1）を計算し、output = gate * new_features + (1-gate) * original_featuresで最終特徴を得ます。これにより、各空間位置・各特徴次元ごとに「どれだけLLM由来の情報を取り入れるか」をモデル自身が調整できるようになります。初期状態ではゲートが0寄り（元のSAM特徴重視）になるようバイアスを設定しておき、学習に従い必要な次元ではゲートが開いてQwen情報を通す、といった振る舞いが期待できます。ただし、この機構は若干実装が複雑になるため、まずは残差接続のみでも十分効果が出ることが予想されます。必要に応じて拡張する形で良いでしょう。

以上のクロスアテンション融合アプローチにより、SAMのピクセルレベル特徴にQwenの文脈的特徴を柔軟に注入することが可能になります。次節では、具体的なコード修正内容を示します。

具体的なコード修正案（クロスアテンション版）

以下では、src/models/lisa_model.pyを中心に必要な修正・追記箇所をステップごとに説明します。新規のモジュール追加も含め、できるだけ詳細に記述します。

クロスアテンション融合モジュールの追加

まず、SAM特徴とQwen特徴を統合するためのクロスアテンション層をモデルに追加します。実装の見通しを良くするため、fusion_layers.pyなど新規ファイルにクラスを定義し、それをLISA_Modelでインスタンス化する形が望ましいです。以下はシンプルなクロスアテンション融合クラスの例です。

# 新規ファイル: src/models/fusion_layers.py
import torch
import torch.nn as nn

class ImageFeatureFusion(nn.Module):
    """
    高解像度SAM特徴とQwen特徴をクロスアテンションで融合するモジュール
    """
    def __init__(self, dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        # オプション: ゲート機構
        self.gate_fc = nn.Linear(dim * 2, dim)
        self.gate_sigmoid = nn.Sigmoid()

    def forward(self, sam_feats: torch.Tensor, qwen_feats: torch.Tensor) -> torch.Tensor:
        """
        sam_feats: [B, N, D] (Query: SAM特徴をフラットに展開したもの)
        qwen_feats: [B, M, D] (Key/Value: Qwen特徴をフラットに展開したもの)
        """
        # Cross-Attention計算
        attn_output, _ = self.cross_attn(query=sam_feats, key=qwen_feats, value=qwen_feats)
        # 残差接続 + LayerNorm
        fused = self.norm(attn_output + sam_feats)
        # ゲート機構による融合（動的重み付け）
        gate_input = torch.cat([sam_feats, fused], dim=-1)  # [B, N, 2*D]
        gate = self.gate_sigmoid(self.gate_fc(gate_input))  # [B, N, D]
        output = gate * fused + (1 - gate) * sam_feats
        return output


上記ではImageFeatureFusionクラスとして、nn.MultiheadAttentionを用いたクロスアテンション層（8ヘッド、dropout=0.1）を定義しています。forwardではSAM特徴をQuery、Qwen特徴をKey/Valueに指定し、attn_outputを得ています。その後、attn_outputと元のsam_featsを足し合わせてLayerNormを適用し（残差接続）、さらにゲート機構で出力fusedと元sam_featsをチャネル方向に結合してシグモイドゲートを計算、最終的な出力を得ています。ゲート機構部分はオプションですが、追加する場合は上記のようにできます（ゲートを使わない場合は単にreturn fusedで十分です）。

次に、LISA_Model.__init__内でこのモジュールをインスタンス化します。TextPromptProjector等の初期化が終わった箇所で例えば以下を追記します。

from src.models.fusion_layers import ImageFeatureFusion  # ファイル先頭でインポート

class LISA_Model(nn.Module):
    def __init__(...):
        ...
        self.text_prompt_proj = TextPromptProjector(...).to(dtype=model_dtype)
        # ★追加: クロスアテンション融合層の初期化
        self.image_fusion = ImageFeatureFusion(dim=256, num_heads=8, dropout=0.1)
        logger.info("Initialized ImageFeatureFusion (Cross-Attention module for SAM-Qwen features)")
        ...


これで、256次元でヘッド数8のクロスアテンション層self.image_fusionがモデルに組み込まれます。なお、image_fusionモジュール内部のパラメータ（MultiheadAttentionの重みやゲート用Linearの重み）は学習可能となり、エンドツーエンドで最適化されます。

SAM ViT出力特徴の計算と前処理

次に、モデルのforward関数内でSAMの画像エンコーダを実行し、その出力特徴マップを取得します。元のコードではsam_images引数を受け取るだけで未使用でしたが、これを活用する処理を追加します。Qwen特徴抽出後（既存のvision_features計算ブロックの後）あたりに、以下の処理を挿入します。

def forward(..., pixel_values=None, ..., sam_images=None, ...):
    ...
    vision_features = None
    image_features_sam = None
    sam_high_res_features = None
    sam_image_embedding = None  # ★SAM特徴用変数を追加

    # （1）Qwenからの視覚特徴抽出（既存処理）
    if pixel_values is not None:
        vision_features = self.extract_vision_features(pixel_values, image_grid_thw)
        if vision_features.dim() == 2:
            vision_features = vision_features.unsqueeze(0)
        image_features_sam = self.image_adapter(vision_features, image_grid_thw)  # [B, 256, H_q, W_q]
        # マルチスケール特徴生成（Token-FPNまたはHighResFeatureGenerator）
        if self.use_token_fpn:
            image_features_sam, sam_high_res_features = self.token_fpn(image_features_sam, image_grid_thw, use_hooks=True)
        else:
            sam_high_res_features = self.high_res_generator(image_features_sam)
    else:
        # 画像未入力の場合は特徴無し
        image_features_sam = None

    # （2）SAM ViTからの特徴抽出（★新規追加）
    if sam_images is not None:
        # SAM2.1のImageEncoderに高解像度画像を入力し、特徴マップを取得
        # （1024×1024入力に対し[B, 256, 64, 64]が得られる想定）
        sam_image_embedding = self.sam_image_encoder(sam_images)  # [B, 256, 64, 64]
        # 勾配計算有無の制御（必要に応じて凍結解除）
        # for param in self.sam_image_encoder.parameters():
        #     param.requires_grad = True  # 必要ならTrueにして微調整可能にする
        logger.debug(f"SAM image encoder output shape: {sam_image_embedding.shape}")
        # Qwen特徴マップをSAM特徴と同じ解像度（64×64）にリサイズ
        if image_features_sam is not None and image_features_sam.shape[-2:] != sam_image_embedding.shape[-2:]:
            image_features_sam = torch.nn.functional.interpolate(
                image_features_sam, size=sam_image_embedding.shape[-2:], mode='bilinear', align_corners=False
            )
            logger.debug(f"Resized Qwen feature map to {image_features_sam.shape[-2:]} to match SAM features")


上記により、sam_imagesからSAM ViT特徴sam_image_embeddingが得られます（想定形状は[B, 256, 64, 64]）。そして、Qwen側の特徴image_features_sam（こちらは例えば32×32だったもの）をF.interpolateで64×64にアップサンプリングしています。これで両者の空間解像度が一致し、対応する位置同士を統合しやすくなります。また、上記コードではコメントアウトしていますが、requires_gradをTrueに設定すればSAMのImageEncoderも微調整可能になります。初期段階ではSAMエンコーダを凍結したまま（プリトレの強力な特徴を保持）でも良いですが、学習が進んでクロスアテンション層を介してLLM側と特徴を擦り合わせる際に、必要であればSAMエンコーダもファインチューニングを許可するとさらに性能向上が見込めます。リソースに応じて検討してください。

Qwen特徴とSAM特徴のクロスアテンション融合

準備した2つの特徴マップを統合する処理を実装します。アップサンプリング後のQwen特徴image_features_sam（名前はそのままですが内容は256×64×64）とSAM特徴sam_image_embedding（256×64×64）をそれぞれFlatten（B, N, 256）し、クロスアテンションモジュールself.image_fusionに渡します。その出力を再度空間マップに戻して統合特徴fused_image_embeddingとします。

    # （3）特徴マップの融合（★新規追加）
    fused_image_embedding = None
    if sam_image_embedding is not None and image_features_sam is not None:
        B, C, H_feat, W_feat = sam_image_embedding.shape  # 通常H_feat=W_feat=64
        # [B, N, C] にフラット化（N = H_feat * W_feat = 4096）
        sam_tokens = sam_image_embedding.view(B, H_feat * W_feat, C)      # [B, 4096, 256]
        qwen_tokens = image_features_sam.view(B, H_feat * W_feat, C)      # [B, 4096, 256]
        # クロスアテンション融合を実行
        fused_tokens = self.image_fusion(sam_tokens, qwen_tokens)         # [B, 4096, 256]
        fused_image_embedding = fused_tokens.view(B, C, H_feat, W_feat)   # [B, 256, 64, 64] に再構成
    elif sam_image_embedding is not None:
        # Qwen特徴が無い場合（pixel_values未提供）はSAM特徴のみを使用
        fused_image_embedding = sam_image_embedding
    else:
        # SAM特徴が無い場合はQwen特徴のみ使用
        fused_image_embedding = image_features_sam
    # Debug用ログ
    if fused_image_embedding is not None:
        logger.debug(f"Fused image feature map shape: {fused_image_embedding.shape}")


ここでは、両方の特徴が存在する場合にクロスアテンション融合を行い、fused_image_embeddingを得ています。ImageFeatureFusionモジュール内部で既に残差接続やゲート融合を行っているため、この出力は元のSAM詳細情報を保持しつつQwenの意味情報が反映された特徴となっています。以降のMaskDecoderにはこのfused_image_embeddingを渡していきます（片方しか無い場合は従来どおり片方の特徴で代用）。

実装ノート: 上記ではバッチ全体をまとめてimage_fusionに通しています（shape: [B, 4096, 256]）。内部ではバッチファーストでMultiheadAttentionを計算しますが、高解像度ゆえ1画像あたり4096query×4096key（約1677万）のアテンション計算となります。ただしヘッド分割やテンソル並列がある程度最適化されており、8ヘッドなら実質8×4096×4096のスコア行列を処理します。メモリ・計算量は増えますが、3B+SAMのモデル規模やGPU性能を踏まえると十分トレードオフ可能と考えられます。必要であればヘッド数を減らす（例: 4ヘッド）や低精度計算（bf16/FP16）でメモリ削減も検討してください。

高解像度マルチスケール特徴の再生成

統合後のfused_image_embeddingをもとに、MaskDecoderに与える高解像度のマルチスケール特徴（stride-4とstride-8相当）を生成します。従来はQwen特徴からHighResFeatureGeneratorで128×128・256×256を作っていましたが、今回は既に64×64の詳細なベース特徴が得られているため、それを2倍・4倍アップサンプルします。具体的には以下のようにします。

    # （4）高解像度マルチスケール特徴の生成（★修正）
    sam_high_res_features = None
    if fused_image_embedding is None:
        sam_high_res_features = sam_high_res_features  # 既存計算を流用（基本ここには来ない想定）
    else:
        if self.use_token_fpn:
            # Token-FPNアップサンプルモジュールを使用して2倍・4倍特徴生成
            feat_s1 = self.token_fpn.upsample_s1(fused_image_embedding)  # [B, 256, 128, 128]
            feat_s0 = self.token_fpn.upsample_s0(fused_image_embedding)  # [B, 256, 256, 256]
            # MaskDecoderのconv層でチャネル圧縮（256→32/64）
            sam_high_res_features = [
                self.sam_mask_decoder.conv_s0(feat_s0.to(sam_dtype)),
                self.sam_mask_decoder.conv_s1(feat_s1.to(sam_dtype))
            ]
        else:
            # 従来のHighResFeatureGeneratorを使用
            highres_feats = self.high_res_generator(fused_image_embedding)  # [ [B,256,256,256], [B,256,128,128] ]
            sam_high_res_features = [
                self.sam_mask_decoder.conv_s0(highres_feats[0].to(sam_dtype)),  # 256ch->32ch
                self.sam_mask_decoder.conv_s1(highres_feats[1].to(sam_dtype))   # 256ch->64ch
            ]
    # MaskDecoderに渡す最終画像埋め込みを更新
    image_features_sam = fused_image_embedding


まず、fused_image_embedding（64×64, 256ch）から2×アップサンプル（stride-8相当の128×128特徴）と4×アップサンプル（stride-4相当の256×256特徴）を生成します。self.use_token_fpnがTrueの場合は既存のToken-FPNのアップサンプラ（upsample_s1, upsample_s0メソッド）を再利用します。Falseの場合はHighResFeatureGeneratorで同等の結果を得ます。いずれにせよ、出力は256チャネルの特徴図（サイズ256×256および128×128）です。それらをSAM MaskDecoder内蔵のconv_s0・conv_s1畳み込みでそれぞれ32チャネル・64チャネルに圧縮しています
GitHub
GitHub
。これにより、MaskDecoderが受け取る形式（image_embeddingsは256ch、high_res_featuresは32chと64ch）の準備が整いました。

注: オリジナルのSAM2.1ではMaskDecoderに高解像度特徴を与える設計になっており、conv_s0/conv_s1は本来SAM内部で適用されます。今回それを明示的に呼び出しているのは、実装上MaskDecoderに直接渡せるようにするためです
GitHub
。この処理は既存コードから踏襲しています。

MaskDecoderへの統合特徴入力

最後に、MaskDecoderを呼び出す部分で、画像埋め込みとしてfused_image_embeddingを使うように修正します。該当箇所では、各SEGトークンに対して以下のようにMaskDecoderを実行します。

    # （5）MaskDecoderでマスク生成（一部抜粋、変更箇所のみ表示）
    low_res_masks, iou_predictions, sam_tokens_out, obj_score_logits = self.sam_mask_decoder(
        image_embeddings=image_features_sam[i:i+1].to(sam_dtype),   # ★統合後特徴（64×64, 256ch）
        image_pe=image_pe.to(sam_dtype),
        sparse_prompt_embeddings=sparse_embeddings.to(sam_dtype),
        dense_prompt_embeddings=dense_embeddings.to(sam_dtype),
        multimask_output=False,
        repeat_image=True,
        high_res_features=sam_high_res_features  # ★統合後高解像度特徴リスト
    )


ここでimage_features_sam[i:i+1]がクロスアテンション融合後の特徴マップになっている点が肝要です（従来はQwen由来特徴のみでした）。またhigh_res_featuresも対応するものに置き換わっています。これらの変更により、MaskDecoder内部ではQwen+SAM両方の情報を含んだ特徴に基づいてマスク計算が行われます。SEGプロンプト埋め込み（テキスト由来のembedding）に関しては既存通りで変更はありません。LLM隠れ状態から射影した256次元ベクトルと、SAM PromptEncoderから得たPE・ポイント埋め込みをprompt_betaで融合する処理もそのままです。今回追加した画像側の融合と合わせ、言語・画像の両モダリティが深く統合された状態でマスク推定が行われることになります。

以上の改修により、最新VLMであるQwen2.5-VLの意味理解力と、最新SAM2.1の高解像度セグメンテーション能力を兼ね備えたLISA改モデルとなります。特にクロスアテンションを用いた融合によって、各画素について「何をどの程度参照するか」をモデル自ら学習できる点が大きな強みです。各処理の入出力次元もコード中にコメントした通り明確に規定しており、実装上の不明点は少ないはずです。

追加の考察: クロスアテンション導入に伴い、学習時の安定性にも配慮してください。初期段階ではLLM側とSAM側の特徴分布にギャップがあるため、前述の残差接続やゲーティングにより元の情報を残しつつ徐々に融合が進むようにするのがポイントです。また、場合によっては事前アライメント学習（例えばSEGトークンの埋め込みをSAM特徴空間に近づける事前学習）を行うことで、よりスムーズに融合層の学習が進む可能性もあります。これは以前の提案Cと同様の発想ですが、今回のクロスアテンションでも有益と思われます。

最後に、既存コード上でSAM ViT関連の箇所を参照しておくと、データローダ側では上記のpreprocess_sam_image以外に、HybridDataset内でsam_imagesをバッチに含める処理が実装済みです。またモデル初期化時にself.sam_image_encoderがセットアップされている箇所
GitHub
や、MaskDecoderに高解像度特徴を渡す実装
GitHub
も確認しておくと良いでしょう。今回の修正ではそれらを前提に、新たにクロスアテンション融合を差し込む形になります。