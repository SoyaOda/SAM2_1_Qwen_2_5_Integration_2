# 統合モデルの実装概要と現状

現在のLISA改モデルは、Qwen2.5-VL（3B版）とSAM2.1を統合し、テキストと画像の両モーダルからセマンティックセグメンテーションを行う設計になっています。具体的には、Qwenの視覚エンコーダ（高度なViT）で抽出した**画像特徴**を256次元に圧縮する`ImageFeatureAdapter`、およびQwenの言語モデル最終層から得られる\*\*<SEG>特殊トークン位置の隠れ状態**を256次元に射影する`TextPromptProjector`を介して、SAM2.1の**Prompt Encoder**および**Mask Decoder**に接続しています。モデルの学習可能部分は、基本的にこのアダプタ類と追加したLoRAに限られており、QwenとSAMの**ベースモデルのパラメータは固定（freeze）\*\*されています（後述）。

**訓練データ**としてはLISAオリジナルと同様のマルチタスクデータセット（sem\_seg, refer\_seg, reason\_seg, vqa）を統合したものを使用します。`HybridDataset`クラスが各タスクデータ（ADE20K, COCO-Stuff等）を読み込み、`dataset_types`や`sample_rate`の指定に応じて1エポックあたりのサンプルを混合生成します。各サンプルには、ユーザ入力（画像+テキスト指示）とアシスタント出力（<SEG>トークンを含む応答）、さらに**対応する画像、Ground Truthマスク**が含まれます。コライタでバッチ化する際に`ground_truth_mask`がテンソル化され`mask_labels`として提供され、モデルの`forward`にはこの`mask_labels`が渡されます。以前発生した**画像とマスクのランダムな組み合わせバグ**については、`HybridDataset`初期化時に各データセットから対応するサンプルペアを正しく取得するよう修正済みと思われます。実際、テストスクリプト（後述）で単一サンプルを固定して過学習させた際に、適切に対応する画像とマスクで学習できており、高精度なマスク生成が確認できています。このことから**データ前処理段階での不整合は現在ほぼ解消された**と考えてよいでしょう。

# minimal\_train.pyでの設定とtest\_a1との違い

問題の再現している`minimal_train.py`では、**セマンティックセグメンテーション単独**（--dataset\_types "sem\_seg"）で学習を行っています。主要なハイパーパラメータはバッチサイズ2・累積16・エポック数4、学習率はアダプタ5e-3・SEGトークン埋め込み1e-3・LoRA 1.5e-4で、損失重みは言語:セグ=1.0:1.0と設定されています。コード上で注目すべきは以下の点です：

* **QwenとSAMのfreeze**: 設定上、`freeze_qwen=True`および`freeze_sam=True`となっており、Qwenモデル全体とSAM MaskDecoder・PromptEncoderの**全パラメータを凍結**しています。これにより学習可能なのは、Qwenに挿入したLoRA（Attentionの\$q,k,v\$プロジェクション）、SAM MaskDecoderに挿入したLoRA（自己注意および画像・トークン間クロス注意の\$q,k,v\$）、それにImageFeatureAdapter・TextPromptProjectorと、<SEG>トークン用に追加した単一Embeddingのみです。全体の**学習可能パラメータ数はモデル全体の約5–6%程度**に留まります。

* **SEGトークン処理**: <SEG>トークンを追加し、Qwenの語彙と埋め込み行列に追加しています（embedding行列の当該IDの勾配も有効化済み）。学習データのラベルには、アシスタントの応答位置にこの<SEG>トークンが含まれており、モデルはその位置で**マスク出力を生成**するよう期待されます。

* **マスク生成フロー**: `LISA_Model.forward`内で、まずQwenに入力（画像+テキスト）を通し最終隠れ状態とLMロジットを取得します。その後、画像があればQwen視覚特徴を抽出し（PatchMerge後の特徴256トークンを使用）、ImageAdapterで256chの特徴マップに変換します。Token-FPNが有効な場合はさらにマルチスケール特徴（stride-4,8）も生成します。そして、**<SEG>トークンの位置**をラベルから検出し（学習時）、その位置の隠れベクトルをTextPromptProjectorで256次元埋め込み（**テキスト由来のマスクプロンプト埋め込み**）に射影します。

* **SAMへのプロンプト入力**: 学習時には**GTマスクの重心座標**を計算し（各バッチごと）、それを正の点プロンプトとしてSAM PromptEncoderに与えます。PromptEncoderはこのポイントから`sparse_embeddings`（ポイントの埋め込み）と`dense_embeddings`（画像位置エンコーディング）を生成します。コードでは、この生成された`sparse_embeddings`の**先頭要素（ポイントの埋め込み）を、先ほどのテキスト由来埋め込みで**置き換えています。つまり\*\*「点の場所はGTから与えるが、その点に対応する埋め込みベクトルはLLM（Qwen）の出力に基づくベクトルに差し替える」\*\*という統合戦略です。この操作により、SAMのMaskDecoderには(`image_embeddings`と`image_pe`に加え)「位置＝GT重心、埋め込み特徴＝LLMからの指示」を与えてマスクを計算させています。

* **MaskDecoderによるマスク出力**: 上記プロンプトを元に、HighResFeature（stride-4,8特徴）と画像埋め込みをMaskDecoderに入力し、1つのマスクを出力させます。出力の低解像度マスクは元の画像解像度までアップサンプルされ、最終的な`mask_logits`として各サンプルに格納されます。学習時にはこれをGTマスクと比較して**BCE + Dice損失**を計算し、LM Lossと和をとって最終損失としています。

一方、テスト用スクリプト`test_a1_oracle_sam_fixed.py`では、上記のパイプラインの**セグメンテーション部分のみを取り出し**、極端な過学習テストを行っています。重要な違いは以下の通りです：

* **SAMのアンフリーズ**: `LISAConfig`を準備する際、`freeze_sam=False`とし、SAMのMaskDecoderを**学習可能**に設定しています（Qwenは引き続きfreeze）。さらに`train_seg_token=False`とすることでLLM側の<SEG>トークンembeddingは触らず（テキスト出力を使わないため不要）、代わりに**画像1枚・マスク1枚に対して直接SAMを動かす**検証に特化しています。

* **オラクルプロンプトの使用**: Qwenからのテキスト指示を一切使わず、**GTマスクの重心点のみ**をPromptEncoderに入力し、そのままMaskDecoderでマスクを生成します。**ポイント埋め込みの差し替えを行わない**ため、SAM本来の埋め込みが利用されます。この状態で1サンプルのみを何百ステップも学習させ、MaskDecoderの出力をGTに一致させる過学習を行いました。

* **結果**: このテストでは、数百～千ステップでDice係数0.95以上・mIoU0.9以上に達することが期待されており、実際に**seg\_lossが大幅に低下し高精度のマスクを出力できる**ことが確認されています（ログ: `test_outputs/test_a1_...` にてloss低下を確認）。

要するに、`test_a1`は\*\*「LLMの助け無しでSAM単体（+アダプタ）だけでも正しく学習できるか」\*\*を検証する目的でした。その結果、**データやMaskDecoder自体に致命的な問題はなく、学習設定次第では正しくマスク精度が向上する**ことが示唆されました。

# seg lossが下がらない原因の考察

以上を踏まえ、`minimal_train.py`で**seg lossが全く下がらずマスク生成も向上しない**原因として最も考えられるのは、**モデル設計上のミスマッチと過度なfreezeによる学習制限**です。

具体的には、**「LLM由来のテキスト埋め込みをSAMのポイント埋め込みに置換する」**部分がボトルネックになっている可能性が高いです。SAMのMaskDecoderは本来、PromptEncoderが生成する特徴（ポイントならその**場所情報をエンコードしたベクトル**）を前提としてマスク計算を行います。しかし現在の実装では、このベクトルが**LLMの隠れ状態に射影しただけのもの**に差し替えられています。初期段階ではこのベクトルは**ランダム同然**（LLMはまだ指示に対し適切な隠れ表現を獲得していない）であり、**SAM側から見ると全く予期しない分布の埋め込み**になります。その結果、

* **初期マスク予測精度が極めて低下する**: GTのポイント自体は与えているとはいえ、そのポイント埋め込みが不適切なため、MaskDecoderは正しく対象をセグメントできません。場合によっては全画面か空マスクに近い出力になるなど、**seg\_lossが極端に高いまま**スタートする可能性があります。

* **学習による補正が困難**: 本来であれば、学習を通じてTextPromptProjectorやLoRA経由で**LLM側の隠れ表現を「SAMが理解できる形式」に変換**することが期待されています。しかし今回の学習設定では、QwenもSAM MaskDecoderもベースは凍結され、\*\*ごく限られたパラメータ（アダプタとLoRA数％）\*\*しか調整できません。MaskDecoder側はLoRAで一部注意層を調整可能なだけで、出力層や畳み込み層は固定のままです。またPromptEncoderもfreezeされているため、LLM埋め込みとの相互作用を学習で最適化する余地がありません。**言い換えると、両者のインターフェースのミスマッチを埋める学習自由度が足りない**状況です。

* **タスク難易度と勾配配分**: セグメンテーションタスクは画像中のピクセル単位の予測であり、本来大量の学習が必要ですが、LM部分と同等の重み(1.0)で損失を足し合わせているため、LM出力（<SEG>トークンを正しく出す）側の学習にリソースが割かれ、セグメント部分の勾配寄与が相対的に小さい可能性があります。特にQwenは大規模モデルで凍結されていますが、LoRAやSEGトークンembeddingを通じて\*\*「どの質問に<SEG>を出すか」\*\*は学習します。この言語損失が安定してしまうと（<SEG>トークン出力自体は比較的容易な分類タスク）、残るマスク損失に十分な学習率が適用されにくく、**総合損失の中でseg\_lossが停滞**しやすいと考えられます。

以上の理由から、本質的な原因は\*\*「LLMのテキスト情報をマスク生成に活かす設計」が機能していない\*\*ことにあると推測されます。特に、**テキスト埋め込みとSAM MaskDecoderの間のギャップ**が埋まらないままfreezeされているため、モデルがセグメンテーションの学習を進められない状態です。事実、先述の`test_a1`ではテキスト埋め込みを介在させなかったため速やかにlossが低下しました。対して`minimal_train`では、テキスト埋め込みを介在させた上にMaskDecoderを凍結したため、**SAMが本来持つ性能を発揮できず、損失曲線が停滞**したと考えられます。

なお、データ不整合の可能性についても念のため検証しました。HybridDataset内で**画像とmask\_labelsのペアリング**にズレが生じていると、当然学習は進みません。しかし、テストスクリプトで同一サンプルを固定した際に正常に学習できたこと、HybridDatasetの実装では各データセットから対応するマスクを読み出す処理が整備されていることから、現在は**入力画像とGTマスクは正しく対応している**とみられます。従って、seg loss停滞の原因はデータではなく**モデル内部の問題**と判断できます。

最後に、別の観点として**SEGトークン特殊処理**のバグも考えられましたが、こちらもコード上は正しく対処済みです。過去には「LoRA適用後にresize\_token\_embeddingsを呼んでいた」不具合がありSEGトークンembeddingが無効化されていましたが、現在の実装では適切な順序で呼び出しています。また、Processor側のトークナイザにもSEGトークンを追加する対処も行われています。以上より、**SEGトークン自体は正常に機能**しており、損失停滞の直接原因ではないでしょう。

以上の考察から、**根本的な原因は統合モデル設計上のミスマッチ（テキスト埋め込みの扱い）と学習可動箇所の不足**にあると結論づけられます。

# 原因究明のための追加テスト提案

もし上記の推論だけでは不十分であれば、**さらなる検証スクリプト**によって原因を絞り込むことが望ましいです。特に有効なのは、**現在freezeしているSAMのMaskDecoderを一時的に学習可能にし、統合パイプライン全体でセグメンテーション精度が向上するか**を試すテストです。これにより、「MaskDecoderのfreeze（+LoRA微調整のみ）が問題か」を直接検証できます。加えて、テキスト埋め込み差し替え有無の比較もできれば尚良いですが、まずは最も影響が大きいであろうMaskDecoderのアンフリーズ効果を見るのが近道です。

以下に\*\*テストスクリプト例「test\_a2\_unfreeze\_sam.py」\*\*を示します。このスクリプトでは：

* minimal\_trainと同じデータセット（sem\_seg）を使用し、数ステップの学習でseg\_lossの挙動を観察します。
* **SAMのMaskDecoderとPromptEncoderを学習可能**にする（freeze\_sam=False）。Qwenは引き続きfreezeし、LLM部分の影響を固定します。
* SEGトークンは有効化したまま（train\_seg\_token=True）で、テキスト埋め込み差し替えも通常通り行います。こうすることで**現在の統合設計は維持しつつ、MaskDecoderの学習可動性のみ上げた場合**にlossが改善するか確認できます。
* LoRAについては、SAM側は本来freeze解除するので不要ですが、実装上自動で適用されるなら影響は小さいためそのままでも構いません（大勢に影響なし）。Qwen側LoRAは既定どおり有効にします。

もしこのテストで**seg\_lossが明確に低下**し始めるようであれば、やはり原因はMaskDecoderを固定しすぎた点（＝テキスト埋め込みとのマッチング不足）にあったと裏付けられます。一方、依然として改善しない場合、他の要因（例えばテキスト埋め込み差し替え自体の戦略ミス）を疑うべきでしょう。その場合はさらに、**ポイント埋め込みの差し替えを行わずテキスト情報を別途与える**ようなアプローチとの比較実験が考えられますが、まずは下記A2テストで大枠の課題を絞り込みます。

```python
#!/usr/bin/env python3
"""
A-2テスト: MaskDecoderを学習可能にした統合モデルのセグメンテーション学習検証

目的:
  minimal_train.pyと同等の統合パイプラインで、SAM MaskDecoder/PromptEncoderをfreezeしない場合に
  seg_lossが適切に低下するかを検証する。

手順:
  - LISA_Modelをfreeze_sam=Falseで初期化（SAM部分を全て学習対象に含める）
  - sem_segデータセットから少数サンプルでミニバッチ学習を行い、損失推移を観察
"""
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent))

from src.config import LISAConfig
from src.models import LISA_Model
from src.utils import prepare_tokenizer_for_lisa
from src.data.dataset import HybridDataset
from src.data.collators import MultiModalDataCollator

# ハイパーパラメータ設定
num_steps = 500       # 検証するステップ数（必要に応じて調整）
batch_size = 2
learning_rate = 5e-4  # 全パラメータ共通のシンプルな学習率
dataset_type = "sem_seg"
samples_per_epoch = 100  # データセットから使用するサンプル数（小さめに設定）

# 1. モデルとトークナイザの準備
config = LISAConfig(
    freeze_qwen=True,
    freeze_sam=False,        # MaskDecoderも学習させる
    train_seg_token=True,
    # LoRA設定はデフォルト（Qwen側 r=8 等）。SAM側もsam_lora_r=8だがfreeze解除するので気にしない
)
print(f"Config: freeze_qwen={config.freeze_qwen}, freeze_sam={config.freeze_sam}, train_seg_token={config.train_seg_token}")

# トークナイザとプロセッサの用意（SEGトークン追加）
tokenizer = prepare_tokenizer_for_lisa(model_name=config.qwen_model_name, seg_token=config.seg_token)
processor = None
if config.use_dynamic_resolution:
    processor = AutoProcessor.from_pretrained(
        config.qwen_model_name,
        min_pixels=config.qwen_min_pixels,
        max_pixels=config.qwen_max_pixels
    )
else:
    processor = AutoProcessor.from_pretrained(config.qwen_model_name)
# ProcessorのトークナイザにもSEGトークン追加
if config.seg_token not in processor.tokenizer.get_vocab():
    processor.tokenizer.add_special_tokens({"additional_special_tokens": [config.seg_token]})

# モデル初期化
model = LISA_Model(config)
model.set_tokenizer(tokenizer)
# （QwenモデルにLoRA適用。SAM側はfreeze解除なので必要なら適用しなくてもよいが、自動適用される設定ならそのままで）
if hasattr(config, 'sam_lora_r') and config.sam_lora_r > 0:
    model.add_sam_lora(config.sam_lora_r, config.sam_lora_alpha, config.sam_lora_dropout)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# 学習モードに設定
model.train()

# 学習対象のパラメータを確認
trainable_params = [n for n,p in model.named_parameters() if p.requires_grad]
print(f"Total trainable parameters: {len(trainable_params)} tensors")
for name in trainable_params[:10]:
    print(f" - {name}")
# （大量にある場合は適宜省略）

# 2. データセット準備
base_dir = config.dataset_base_dir
dataset = HybridDataset(
    base_image_dir=base_dir,
    qwen_processor=processor,
    samples_per_epoch=samples_per_epoch,
    dataset=dataset_type,
    sample_rate=[1.0]  # 単一データセットなのでレート1
)
collator = MultiModalDataCollator(
    tokenizer=tokenizer,
    max_length=config.model_max_length,
    config=config
)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, num_workers=0)

print(f"Dataset loaded: {dataset_type}, total samples={len(dataset)}")

# 3. オプティマイザ準備
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 4. 学習ループ開始
step = 0
for batch in tqdm(dataloader, total=num_steps, desc="Training"):
    # バッチからデバイスへ転送
    for k,v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(model.device)
    # モデルforward実行
    outputs = model(
        input_ids=batch['input_ids'],
        pixel_values=batch['pixel_values'],
        attention_mask=batch['attention_mask'],
        labels=batch['labels'],
        mask_labels=[batch['mask_labels'][i] for i in range(batch['mask_labels'].size(0))]
    )
    # 損失計算
    # Languageモデリング損失
    vocab_size = outputs.logits.size(-1)
    lm_loss = F.cross_entropy(outputs.logits.view(-1, vocab_size), batch['labels'].view(-1), ignore_index=-100)
    # セグメンテーション損失（BCE + Dice）
    seg_loss = 0.0
    seg_count = 0
    if outputs.mask_logits is not None:
        for i, pred_masks in enumerate(outputs.mask_logits):
            if pred_masks is None: 
                continue
            # リストで来る場合もあるので統一
            if isinstance(pred_masks, list):
                pred_mask = pred_masks[0] if len(pred_masks)>0 else None
            else:
                pred_mask = pred_masks
            gt_mask = batch['mask_labels'][i]
            if pred_mask is None or gt_mask is None:
                continue
            # 形状をあわせる（必要ならリサイズ）
            if pred_mask.dim() > 2:
                pred_mask_tensor = pred_mask.squeeze(0)  # (H,W)
            else:
                pred_mask_tensor = pred_mask
            gt_mask_tensor = gt_mask[0] if gt_mask.dim()==3 and gt_mask.shape[0]>1 else gt_mask  # 複数マスクは最初のみ
            # 必要ならリサイズ
            if pred_mask_tensor.shape != gt_mask_tensor.shape:
                gt_mask_resized = F.interpolate(gt_mask_tensor.unsqueeze(0).unsqueeze(0).float(),
                                                size=pred_mask_tensor.shape, mode='nearest').squeeze(0).squeeze(0)
            else:
                gt_mask_resized = gt_mask_tensor.float()
            # BCE
            bce = F.binary_cross_entropy_with_logits(pred_mask_tensor, gt_mask_resized)
            # Dice
            pred_prob = torch.sigmoid(pred_mask_tensor)
            intersection = (pred_prob * gt_mask_resized).sum()
            dice = 2 * intersection / (pred_prob.sum() + gt_mask_resized.sum() + 1e-8)
            dice_loss = 1 - dice
            seg_loss += (bce + dice_loss)
            seg_count += 1
    if seg_count > 0:
        seg_loss = seg_loss / seg_count
    total_loss = lm_loss + seg_loss
    # 逆伝播とオプティマイザステップ
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    # ログ出力
    step += 1
    if step % 50 == 0:
        print(f"Step {step}: total_loss={total_loss.item():.4f}, lm_loss={lm_loss.item():.4f}, seg_loss={seg_loss.item():.4f}")
    if step >= num_steps:
        break

print("Training finished.")
```

上記スクリプトでは、50ステップごとに損失を簡易表示するようにしています。`freeze_sam=False`であるため、もし**seg\_lossが徐々に低下する**ようであれば、MaskDecoder部分の学習可動化が奏功したと言えます。その場合、統合モデル本番でも\*\*SAM MaskDecoderに多少勾配を流す（LoRAのrank増大や一部レイヤーのunfreeze）\*\*等の対策で改善が期待できます。

逆に、このA2テストでもなおseg\_lossが下がらない場合、問題はより根深く、**テキスト埋め込みの扱いそのもの**を再検討する必要があります。その次の一手としては、例えば**ポイント埋め込みの差し替えをやめ、LLMで参照対象を特定した上で通常のSAMプロンプト（ポイントやボックス）を生成させる**方法や、\*\*テキスト埋め込みとポイント埋め込みを結合する設計（例えばLLM出力をMaskDecoder内の追加条件付けベクトルとして与える）**等が考えられます。このような代替アプローチを試す前段として、まずは上記テストで**「単にMaskDecoderが固すぎたせいか？」\*\*を確認するのが適切でしょう。
