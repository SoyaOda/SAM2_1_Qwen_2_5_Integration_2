# LISA改モデルのセグメンテーション精度向上改修仕様

## 背景・問題点

統合モデル「LISA改」はQwen2.5-VLの視覚・言語能力とSAM2.1の高精度マスク生成を組み合わせています。しかし現状の実装では、**セグメンテーション損失（seg\_loss）が全く低下せず**、学習が進んでもマスク生成性能が向上しない問題が発生しました。

原因解析の結果、以下の点が主なボトルネックと判明しています：

* **LLM埋め込みとSAM埋め込みのミスマッチ**: 現行ではSAMのポイントプロンプト埋め込みを**LLM（Qwen）の出力ベクトルに完全に置換**しています。つまり、GTマスクの重心点座標で得たSAMの埋め込み（位置情報）は捨てられ、LLMの<SEG>トークンから得た埋め込みに置き換えてマスクデコーダに渡しています。この結果、**位置情報が失われ**、LLM埋め込みも当初はセグメンテーションに適した表現になっていないため、マスク生成がうまく学習できません。

* **SAM MaskDecoderの凍結**: 初期設定ではSAMのMaskDecoderおよびPromptEncoderを**全てfreeze**しており、SAM側にはLoRAで少量の調整しか加えていません。LLM埋め込みとのミスマッチを埋めるのに、SAM側が全く学習できないため、ギャップが解消されないままとなっています。

実際のテストでは、**SAMの学習を解放**すると（MaskDecoder/PromptEncoderのfreeze解除）、少数データであってもseg\_lossが着実に低下しマスク精度が向上することが確認されました。これは、現行のfreeze設定ではLLM→SAM間の不整合をSAM側で補正できないためと推測されます。

## テスト結果から得られた知見

上記問題に対し、以下の**3つのアプローチ**（A, B, C）が検討・試行されました。それぞれの効果と知見を整理します。

* **A: 段階的ゲーティング（Gated Embedding）**
  *方法*: 学習初期はSAMの元来の位置埋め込み（e\_pos）を主に使い、徐々にLLM埋め込み（e\_llm）の比率を上げて混合します。具体的には係数αで`e_mix = (1-α)*e_pos + α*e_llm`とし、αを徐々に0→最大値に増加させました（最大でもLLM成分50%程度）。
  *結果*: 少数ステップのテストでは、seg\_lossが緩やかに減少し一定の効果が見られました（学習後半でマスク精度向上を確認）。しかし**初期段階の学習は依然として位置情報頼み**であり、LLM埋め込みへの依存を増やすと精度が不安定になるリスクがあります。スケジューリングする手間もあるため、よりシンプルで効果的な方法が望まれました。

* **B: 加算アプローチ（Residual Addition of Embeddings）**
  *方法*: 位置埋め込みe\_posを**100%保持**したまま、LLM埋め込みe\_llmを**加算**によって融合します。いわば残差接続のように、`e_add = e_pos + β * e_llm`で結合し、βは学習可能なパラメータとしました（sigmoidで0〜1に制限）。初期βはごく小さく設定し、ほぼ位置情報のみの状態からスタートさせます。
  *結果*: seg\_lossが**安定して着実に低下**し、終始位置情報を損なわないため学習が破綻しにくいことが確認されました。特に、混合比をモデル自身が最適に調整できるため、手動のスケジュール調整よりも効率的でした。少数ステップの検証でも、DiceスコアやIoUが着実に向上し、Aより早期に高精度マスクを獲得しています。

* **C: 埋め込みベクトルの事前アライメント（Pre-Alignment）**
  *方法*: 本学習の前段階（**ステージ0**）として、LLM側の<SEG>埋め込みをSAMの位置埋め込みに**事前に近づける**訓練を行います。具体的には、GTマスクの重心点に対応するSAM PromptEncoder出力ベクトルを \$e\_{pos}\$、対応するLLMの隠れ状態（最終層）を射影したベクトルを \$e\_{llm}\$ として、**Loss = MSE(\$e\_{llm}\$, \$e\_{pos}\$)** を数千ステップ学習します（※\$e\_{pos}\$はdetachしてSAM側は更新しない）。これにより、初期段階でLLMの<SEG>トークンが「セグメンテーション対象の位置情報」を表現できるようになります。
  *結果*: 単一サンプルでの簡易検証ですが、このアライメントによりLLM埋め込みとSAM埋め込みのコサイン類似度が大幅に向上しました。**ステージ1開始時点で既にseg\_lossが低めに抑えられる**効果が期待できます。学習初期の不安定さ（seg\_lossが高止まりする現象）を緩和し、全体の収束を速めることが見込まれます。

以上より、**根本解決策**としては「LLM埋め込みとSAM埋め込みの融合方法を見直し、必要に応じて初期アライメントでギャップを埋めること」が有効と判断しました。特にB（加算融合）はA（ゲーティング）よりシンプルで効果的であり、C（事前アライメント）と組み合わせることで相乗効果が期待できます。

## 改修方針 (提案する解決策)

**結論**: 今回の改修では**アプローチB（埋め込み加算）を中心に、必要に応じてC（事前アライメント）を組み合わせて適用**します。具体的には、

* **モデル側（LISA\_Model）の改修**: LLMの出力埋め込みとSAMの位置埋め込みを直接加算融合するロジックに変更します（ゲーティングは採用せず**加算法を正式採用**）。これに伴い、融合比率を調整する学習可能パラメータβをモデルに追加します。

* **学習スクリプト側（minimal\_train.py）の改修**: 本学習の前に**ステージ0: アライメント学習**をオプションで実行できるようにします。数千ステップのMSE訓練で<SEG>埋め込みを位置埋め込みに近づけ、その後通常のタスク学習（ステージ1）に入る流れを組み込みます。

* **LoRA調整**: 基本的にQwen側のLoRAは現状どおりAttentionのq\_proj, k\_proj, v\_projに適用し、SAM側もMaskDecoderの各AttentionにLoRA適用（r=8）を維持します。今回の改修（C+B）に合わせ、**必要ならLoRAの適用範囲拡張**も検討可能です。例えば、最新のマルチモーダル微調整ではFFN（全結合層）へのLoRA挿入やLayerNormの重み調整も導入されつつあります。現状すぐには変更しませんが、seg\_lossの改善幅が不十分な場合には、**LoRAのランクを上げる**（特にSAM側のrを8→16へ）ことや、**QwenデコーダのFFNにもLoRA**を追加する（`target_modules`に `"mlp"` などを含める）ことも許容範囲です。設計上問題はないため、精度と計算コストを見ながら追って拡張を検討します。

以下、上記方針に基づく**具体的な実装修正項目**を示します。

## 実装詳細・修正内容

### 1. モデル側の修正 (`src/models/lisa_model.py`)

**① 埋め込み融合ロジックの変更（置換 → 加算）**
LISA\_Modelのマスク生成部分で、SAM PromptEncoderから得たポイント埋め込みとLLM由来埋め込みを融合する処理を変更します。現在は以下のように**LLM埋め込みで完全置換**しています。

```python
# （現行実装）ポイント埋め込みをLLM埋め込みに置換
if sparse_embeddings.shape[1] > 0:
    sparse_embeddings[:, 0, :] = prompt_embed.unsqueeze(0)
```



これを**加算アプローチ**に改め、SAMの元の埋め込みを保持しつつLLM埋め込みを加えるようにします。具体的な修正コード案は以下のとおりです。

```python
# （改修後）位置埋め込みとLLM埋め込みの加算融合
if sparse_embeddings.shape[1] > 0:
    e_pos = sparse_embeddings[:, 0, :]                 # [1,256] SAM元の位置埋め込み
    beta_scaled = torch.sigmoid(self.prompt_beta)      # 学習可能スケーリング係数を0〜1に制限
    e_add = e_pos + beta_scaled * prompt_embed.unsqueeze(0)  # 加算による融合
    sparse_embeddings[:, 0, :] = e_add                 # SAMの埋め込みを上書き
```

上記のように、**self.prompt\_beta**を介してLLM埋め込みの寄与を調整します。sigmoidで0〜1に正規化することで、極端な負値や1超過の係数にならないよう制御しています。初期段階ではbetaを小さく設定するため、ほぼe\_posそのままでマスク生成しつつ、学習に応じて適切な比率に調整されるようになります。

**② βパラメータの追加**
上記コードの`self.prompt_beta`は新規に導入する**学習可能パラメータ**です。モデル初期化 (`LISA_Model.__init__`) 内で以下を追加してください。

```python
# LISA_Model __init__ 内（モデル構築時）
self.prompt_beta = nn.Parameter(torch.tensor(0.01))
```

初期値は0.01程度（sigmoid後約0.5）ですが、**必要に応じて調整可能**です。例えば初期からLLM埋め込み寄与を極小にしたい場合、`torch.tensor(-4.6)`のようにマイナス値に設定すればsigmoid(β)≒0.01となります。ひとまず0.01（50%程度）で開始し、学習の中で適応させる方針とします。

なお、追加したParameterは`named_parameters()`にも現れるため、後述のoptimizer設定で**adapter系パラメータ**に含まれます（名前に"prompt\_beta"を含みますが、adapterとみなして問題ありません）。学習率は他のアダプタ類（ImageAdapter, TextPromptProjector）と同じく`adapter_lr`を適用します。

**③ 上記変更に伴う関連部の確認**

* `TextPromptProjector`: LLM隠れ状態を256次元に射影するモジュールです。現状この出力をそのままprompt\_embedとして使用しています。今回の融合では射影後の値をスケーリングして加算しますが、特に追加修正は不要です。射影層自体も学習で更新され、適切な特徴抽出を行うようになります。

* `compute_mask_centroid` & PromptEncoder入力: 現行実装で、訓練時はGTマスクの重心座標を計算しポイント入力に使用しています。この点はそのまま活かします。SAM PromptEncoderから得られるsparse\_embeddingsには**正のポイント1個分の埋め込み e\_pos**が含まれるため、これとLLM側埋め込み e\_llm を加算するのが上記実装です。推論時（mask\_labelsなしの場合）は画像中心をポイントとしますが、その場合はe\_posが画像中心の埋め込みとなります（LLMは自由応答なので実質β次第）。推論時も同じ加算式が適用されます。

* **複数<SEG>トークン**への対応: 現状1会話に1つ<SEG>を想定していますが、コード上は複数SEG位置に対応できるようループしています。複数ある場合、各SEGについて同様に加算処理を行います（実装上もforループ内で同じ処理を適用します）。今回の変更は1つの場合と同様に各要素へ適用されるため、特段の修正は不要です。

### 2. 学習スクリプトの修正 (`minimal_train.py`)

**① ステージ0: 埋め込みアライメント学習の導入（オプション）**
seg\_loss改善を加速するため、**本学習に入る前にアライメント専用の事前学習ステージを追加**します。これはオプション機能とし、コマンドライン引数や設定で有効/無効を切り替えられるようにします。

* 引数の追加: `--align_steps`（整数）を追加し、デフォルト0（実行しない）。例えば`--align_steps 2000`と指定すればステージ0を2000ステップ実行します。

* ステージ0の実行フロー: `MinimalTrainer.setup_model_and_data()`実行後、`trainer.train()`を呼ぶ前に以下の処理を挿入します（Pseudoコード形式）。

```python
if config.align_steps and config.align_steps > 0:
    logger.info(f"ステージ0: 埋め込みアライメントを{config.align_steps}ステップ実行します")
    model = trainer.model  # LISA_Model
    model.train()
    # ステージ0ではSAM側を固定し、LLM関連パラメータのみ学習
    # （Qwenは元々freeze_qwen=TrueなのでLoRAとSEGトークンEmbedding、TextPromptProjのみ対象）
    # optimizer構築（LoRA＋SEGトークン＋TextPromptProj用に簡易版）
    align_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # 学習対象でないパラメータは無視
        # QwenのLoRAパラメータ
        if "qwen" in name and "lora" in name:
            align_params.append({"params": param, "lr": config.lora_lr})
        # SEGトークンEmbedding（word_embeddings中の追加トークン部分）
        elif "word_embeddings" in name:
            align_params.append({"params": param, "lr": config.seg_token_lr})
        # テキスト関連アダプタ（TextPromptProjectorの重みなど）
        elif "prompt_proj" in name or "prompt_beta" in name:
            align_params.append({"params": param, "lr": config.adapter_lr})
        # ※ image_adapterやSAM側のLoRA等は除外
    align_optimizer = torch.optim.AdamW(align_params, lr=0)  # 各param dictで個別lr設定済み
    # スケジューラ省略（短期学習のため一定学習率で充分）
    for step in range(config.align_steps):
        batch = next(iter(trainer.train_loader))  # 1バッチ取得（逐次でもshuffleでも可）
        for k,v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(model.device)
        outputs = model(
            input_ids=batch['input_ids'],
            pixel_values=batch['pixel_values'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],       # labelsを渡すことでSEG位置特定
            mask_labels=batch['mask_labels']  # GTマスクを渡す（重心計算に使用）
        )
        # LLM埋め込みとSAM埋め込みを取得
        # （モデル内部ですでにsparse_embeddings更新前のe_pos, prompt_embedを計算しているが取得しにくいため、
        #  再度計算。ただしcompute_mask_centroid等処理を二重に呼ぶコストは小さいため許容）
        seg_positions = outputs.seg_token_positions  # 各サンプルのSEG位置インデックスリスト
        hidden_states = outputs.language_hidden_states  # Qwen最終層hidden state [B, Seq, 2048]
        align_loss = 0.0
        for i, seg_list in enumerate(seg_positions):
            if len(seg_list) == 0: 
                continue  # このサンプルにSEGトークンなし
            # 1つ目のSEGトークン位置のみ使用（1会話1マスクを想定）
            j = seg_list[0]
            e_llm = model.text_prompt_proj(hidden_states[i, j])       # 256次元のLLM埋め込み
            # GTマスク重心からSAM埋め込み取得（PromptEncoderを個別に呼び出し）
            gt_mask = batch['mask_labels'][i]
            if gt_mask is None:
                continue
            center = model.compute_mask_centroid(gt_mask, orig_h=batch['pixel_values'].shape[-2], orig_w=batch['pixel_values'].shape[-1])
            cx, cy = int(center[0].item()), int(center[1].item())
            point = torch.tensor([[cx, cy]], dtype=torch.float32, device=model.device)
            point_label = torch.tensor([1], dtype=torch.int32, device=model.device)
            sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
                points=(point.unsqueeze(0), point_label.unsqueeze(0)), boxes=None, masks=None
            )
            if sparse_embeddings.shape[1] == 0:
                continue
            e_pos = sparse_embeddings[0, 0, :].detach()  # [256], detachしてSAM側に勾配を流さない
            # MSEロス計算
            align_loss += torch.nn.functional.mse_loss(e_llm, e_pos)
        align_loss /= batch['input_ids'].size(0)  # バッチあたり平均
        align_optimizer.zero_grad()
        align_loss.backward()
        align_optimizer.step()
        if (step+1) % 100 == 0:
            logger.info(f"[Align Stage] Step {step+1}/{config.align_steps}, align_loss={align_loss.item():.6f}")
    logger.info("ステージ0完了。LLMの<SEG>埋め込みを事前調整しました。")
```

上記は擬似コードですが、重要な点は以下です。

* **学習対象パラメータの限定**: QwenのLoRA、<SEG>トークンEmbedding、TextPromptProjector（および追加したprompt\_beta）**のみを最適化**します。SAM側（PromptEncoder, MaskDecoder, およびSAM側LoRA）はこの段階では不使用またはdetachしており、更新しません。

* **データ利用**: デフォルトでは`trainer.train_loader`（HybridDatasetによる全タスク混合データ）を使います。SEGトークンとGTマスクがあるサンプル（sem\_segやrefer\_segタスク）では上記計算が行われますが、例えばVQAサンプル（mask\_labelsなし）はループ内でskipされるだけです。結果的に**セグメンテーション系データに対してのみMSE訓練**する形になります。必要であれば、sem\_segデータセットのみにフィルタしたデータローダを別途用意することも検討可能ですが、まずは現行データローダを再利用します。

* **重心計算**: `model.compute_mask_centroid`を利用してGTマスク中心座標を算出しています。SAM PromptEncoderへはその点を正のポイントとして与え、対応する埋め込みベクトル\$e\_{pos}\$を取得しています。この\$e\_{pos}\$をdetachすることで、SAM側パラメータには勾配が流れません。

* **ステップ数とログ**: align\_stepsは数千程度を想定しています。ログは100ステップ毎など適宜出力し、学習完了後にも情報を表示します。

* **ステージ1への影響**: ステージ0で行った勾配更新により、Qwen側LoRAやSEG埋め込みがわずかに調整されます。これにより`prompt_embed`の初期値がSAM埋め込みに近くなり、続く本学習でのseg\_loss初期値低減が期待できます。実際、テストではアライメント後にseg\_lossがいくぶん下がった状態でepoch1に入ることを確認しています。

ステージ0は任意機能であり、`align_steps=0`なら従来どおり直接ステージ1に入ります。学習時間との兼ね合いで適切に活用してください。

**② 既存トレーニングループへの影響**
上記ステージ0追加に伴い、\*\*メインの学習ループ（trainer.train()）\*\*には大きな変更はありません。唯一、ステージ0実行後にOptimizerを切り替える点に注意してください。

* *Optimizer再初期化*: ステージ0では独自の`align_optimizer`を使いました。本番の`trainer.optimizer`および`trainer.scheduler`は、引き続き`trainer.setup_optimizer_and_scheduler()`で準備したものを**ステージ1開始前に再度zero\_grad()**して利用します。alignステージで更新していないパラメータにも一時的に勾配が溜まっている可能性があるため、`align_optimizer.step()`後に**モデル全パラメータのgradをリセット**しておくと安全です（例えば`model.zero_grad(set_to_none=True)`等で）。もっとも、上記コードでは毎stepでzero\_gradしているため不要かもしれませんが、念のためリセットしてから本学習に入りましょう。

* *学習率・スケジューラ*: ステージ0用Optimizerは簡易に定率で実施しました。本学習では既存通り、パラメータグループ（adapter\_params, lora\_params, seg\_token\_params）に対し別々のlearning rateを適用し、ウォームアップ付きスケジューラで学習を行います。この設定はそのまま維持します。

**③ LoRA適用範囲の見直し（必要に応じて）**
現時点では変更を加えませんが、質問にもあったように「LLMのLoRA適用範囲拡張」も将来のチューニング候補です。例えば、Qwenのクロスアテンションや自己注意の**出力投影（o\_proj）**や**FFNの中間層**にもLoRAを挿入することで、視覚情報の融合能力を高められる可能性があります。また、学習を進めた結果によっては**LoRAのランクrを増やす**ことも考えられます（メモリと相談ですが、例えばQwen側16、SAM側16など）。これらは最新の研究動向や当プロジェクトのベストプラクティスに応じて、実装面では問題なく拡張可能です。もし実施する場合は、`LoraConfig(target_modules=[...])`に追加するモジュール名を適切に指定してください（例：`"o_proj"`やQwenのFFN名など）。現在のところ大きな不具合が出ていないため、まずは**既存のq\_proj/k\_proj/v\_projへのLoRA適用で十分か検証**し、それでも不足があれば段階的に広げる方針とします。

以上が今回の改修内容です。まとめると、**モデル内部での埋め込み融合を「位置＋言語」の加算方式に変え、学習前にLLM埋め込みを位置情報に近づけるステージを導入**することで、seg\_lossの改善とマスク精度向上を図ります。これら改修により、seg\_loss曲線が初期から下がり始め、エポックを通じて順調に減少していくことが期待されます。実装した変更を反映させた上で再度学習を走らせ、seg\_loss・マスク出力精度の改善を確認してください。各変更箇所のソースコードも併記しましたので、実装時の参考にしてください。

**引用ソース**:

* LISA\_Modelによる埋め込み置換実装（現行）
* 加算アプローチ実装例（Test A4より）
* 段階的ゲーティング実装例（参考）
* Test A2（SAM unfrozen）設定及び考察
