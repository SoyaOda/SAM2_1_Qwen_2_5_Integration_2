## ステップ1: データセットモジュールの設計と実装

まずは**複数のデータセットを統合して読み込む仕組み**を構築します。オリジナルのLISAにならい、**複数種類のデータ**を混合したマルチタスク学習を最初から行う方針です。LISAでは以下の4カテゴリのデータを使用していました:

* **(1) セマンティックセグメンテーション**: ADE20K, COCO-Stuff, Mapillary, PACO-LVIS, PASCAL-Part など。画像ごとにピクセルごとのカテゴリーラベルがあるデータです。
* **(2) 参照表現セグメンテーション**: RefCOCO, RefCOCO+, RefCOCOg, RefCLEF など。画像と言語の指示（例：「画像中の青い椅子を指し示して」）と対象物のマスクからなるデータです。
* **(3) 視覚質問応答 (VQA)**: 例えば LLaVA-Instruct-150k など、画像＋質問文＋テキスト解答からなるデータです。
* **(4) 理由を要するセグメンテーション**: LISA独自のReasoning Segmentation (ReasonSeg)データセット（1000件規模）。画像＋高度な推論を要する指示文＋対象物マスク（＋場合によっては解説テキスト）のデータです。

**目標**は、上記のような多様なデータを統合し、一つのモデルに**言語理解＋視覚理解＋セグメンテーション**の能力を同時に学習させることです。マルチタスク学習により、モデルは例えば**単に画像キャプションを生成する**だけでなく、**指示に応じて適切に<SEG>トークンを使う**ことや、追加情報を説明する能力まで獲得できます。

### 1-1. データセット構造と前処理

各データカテゴリごとに、適切な**データセットクラス/ローダ**を実装します。それぞれのデータで必要となる入出力形式を整理すると:

* **セマンティックセグメンテーション**（Semantic Segmentation）: 元々はカテゴリごとにピクセルラベルがあるデータですが、LISA改ではこれを\*\*「特定カテゴリを指示してセグメンテーションさせるタスク」\*\*に変換します。具体的には、

  * 画像中からランダムに**1つのカテゴリ**（物体クラス）を選び、そのカテゴリに属するピクセルのGTマスクを作成します（複数該当する物体があれば全て含むマスク）。
  * 言語指示として\*\*「画像の中の`<カテゴリ名>`をマスクしてください」\*\*等のプロンプト文を生成します。カテゴリ名は人間にわかる表現（例: "the chair", "all chairs in the image" 等）にします。ADE20K等ではカテゴリリストがあるので、それを使い英文を組み立てます。
  * モデルの期待出力は、そのプロンプトに対し\*\*<SEG>トークン\*\*を出力し、対応するマスクを返すことです。教師データとしては、テキスト出力部分は例えば「Sure, <SEG>.」のように固定フレーズ＋<SEG>のみ（説明なし）とし、マスクのGTを対応付けます。
  * **注意**: 一枚の画像に複数カテゴリがありますが、一度に一つのカテゴリを指示する形式とします（全物体を一括セグメンテーションさせるのではなく、指示された対象のみを出力させる）。
  * 画像は適切にリサイズしてテンソル化（例えば短辺384ピクセル程度に縮小しViTに入力）します。マスクも同様にリサイズし、出力マスクと比較できるようにしておきます。
* **参照表現セグメンテーション**（Referring Segmentation）: これは既に\*\*「言語指示 → 対象マスク」\*\*の形式でデータが存在します。例えばRefCOCOでは「the woman in a red shirt」のような文と、その示す人物のマスクが対になっています。

  * 言語指示文はデータからそのまま取得し、一部前処理（文頭の大文字化や末尾のピリオド追加など整形）を行います。
  * モデル出力はセグメントマスクです。テキスト部分の教師強制（teacher forcing）として、\*\*可能であれば短い確認応答＋<SEG>**にします。例えばオリジナルLISAでは応答冒頭に「Sure,」を付けています。したがって参照セグメンテーションの教師テキストは**「Sure, <SEG>.」\*\*のフォーマットを採用します。
  * なお説明文は必要ありません（ユーザも特に求めない前提）。単に指示に「承諾 + マスク出力」をする形式です。
  * 画像とマスクの読み込み・テンソル変換を実装します。画像はRGBテンソル、マスクは0/1のバイナリテンソルにします。
* **視覚質問応答**（VQA）: 画像＋質問文に対し**テキストで回答**するタスクです。セグメンテーションは絡みません。

  * 入力の質問文はそのままトークン化してモデルに与えます。画像もViTへの入力テンソルにします。
  * モデル出力は**純粋なテキスト**回答です。<SEG>トークンは**一切含まれない**形式となります。
  * 教師データとしては、期待する回答文を`labels`に持ち、`mask_labels`は存在しない（None）となります。
  * 例えば、「この画像の空は何色ですか？」に対し「空は青色です。」といった回答が正解になります。モデルには通常の言語生成損失のみを適用します。
* **推論系セグメンテーション**（Reasoning Segmentation）: 画像＋高度な質問（推論・知識が必要）＋マスク＋説明テキスト（オプション）という複合出力タスクです。

  * 例として、「この画像でアメリカ合衆国の大統領だった人物をマスクで示し、その理由を説明してください」という入力に対し、モデルは**人物をマスク**しつつ、「Sure, \[SEG]. The President ... is ...」のように理由を述べるテキストも出力します。
  * このデータでは**出力に<SEG>トークン＋追加の説明文**が含まれる形となります。教師信号としては、例えば「Sure, <SEG>. In the image, ...」のように予め用意された説明文付き回答全文を`labels`に持ち、対応する`mask_labels`を持ちます。
  * 学習時には、テキスト部分でクロスエントロピー損失、マスク部分でセグ損失を同時に適用します。
  * **データ準備**: ReasonSegデータセットは1k程度と少ないですが、内容が多彩です。JSONやTSVで画像ファイルパス、指示文、マスク画像パス、そして期待する説明テキストが記載されているはずなので、それに基づき入力を構成します。説明文には事実知識や理由が書かれており、モデルが言語的推論を学習するのに役立ちます。
  * **補足**: オリジナルLISAでは説明あり版(Explanatory)と無し版でモデルを分けていました。我々は**単一のモデルで両方対応**できるよう、データ内の指示に応じて説明文の有無を切り替えて学習させます。

以上の各データセットについて、PyTorchの`Dataset`クラスを実装します。共通して必要なのは:

* 画像ファイルの読み込みと前処理：OpenCVやPILで読み取り、**QwenのViTが期待する正規化**（例えば`0〜1`正規化とImageNet平均/分散での標準化）を行い、`pixel_values`テンソルを作成。
* テキストのトークナイズ：共通のQwenトークナイザ（<SEG>追加済み）を用いて、プロンプト文および期待応答文をID列に変換。長すぎる場合はカットし、必要ならパディング。
* マスクラベルの読み込みと処理：セグメンテーションが絡む場合、PNG等のマスク画像を読み込んで**binary mask**テンソル（対象領域=1、それ以外=0）にします。解像度は後で損失計算時にモデル出力（元画像解像度近くにアップサンプルしたマスク）と比較できるよう、**元画像と同サイズ**に揃えます。
* データ項目のフォーマット統一：各`__getitem__`は辞書で `{"pixel_values": image_tensor, "input_ids": prompt_ids, "labels": answer_ids, "mask_labels": mask_tensor_or_None}` のような出力に統一し、DataLoaderのcollate関数でバッチ化できるようにします。

**実装例**: 参照セグメンテーション用データセット（RefCOCO）の簡易実装イメージを示します。セマンティックセグメンテーションやReasonSegもこれを参考に実装します。

```python
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class RefSegDataset(Dataset):
    def __init__(self, data_list, image_dir, tokenizer, max_length=128):
        """
        data_list: 各要素が (image_filename, instruction_text, mask_filename) のタプル
        image_dir: 画像ファイルのベースディレクトリ
        tokenizer: Qwen用トークナイザ (special token <SEG>含む)
        max_length: テキストシーケンス最大長
        """
        self.data_list = data_list
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        img_file, instruction, mask_file = self.data_list[idx]
        # 画像読み込み
        image = Image.open(f"{self.image_dir}/{img_file}").convert("RGB")
        # 必要に応じリサイズ（例: 短辺384px）しTensor化
        image_tensor = self.preprocess_image(image)
        # テキストプロンプトをトークナイズ（今回はそのまま指示文）
        prompt = instruction
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=self.max_length)["input_ids"]
        # 期待するモデル応答: "Sure, <SEG>."
        answer_text = "Sure, <SEG>."
        answer_ids = self.tokenizer(answer_text, add_special_tokens=False)["input_ids"]
        # ラベルとなるシーケンス: プロンプトに続けて応答を含む形で構築
        # （学習時はユーザ指示→アシスタント回答全体をモデルに食わせ、後半を予測させるため、ラベルに<SEG>含むアシスタント応答部分を設定）
        input_ids = prompt_ids + answer_ids  # シンプルには結合（実際は区切りトークンやフォーマット次第）
        labels = [-100]*len(prompt_ids) + answer_ids  # プロンプト部分は-100で無視し、回答部分のみ損失対象
        # マスク画像読み込み
        mask = Image.open(f"{self.image_dir}/{mask_file}")
        mask_array = np.array(mask.resize(image_tensor.shape[-2:][::-1], resample=Image.NEAREST))
        # mask_arrayを0/1に（二値化）
        mask_bin = (mask_array > 127).astype(np.float32)  # 255白マスク想定
        mask_tensor = torch.from_numpy(mask_bin).unsqueeze(0)  # [1, H, W]
        return {
            "pixel_values": image_tensor, 
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "mask_labels": mask_tensor  # mask_labelsはfloat32 Tensor (0/1)
        }
    def preprocess_image(self, image: Image.Image):
        # 画像をモデルの期待するテンソルに変換する関数（簡略版）
        # 例: リサイズ+正規化
        target_size = 384
        image = image.resize((target_size, target_size))
        img_arr = np.array(image).astype('float32') / 255.0
        # normalize (ImageNet mean/std)
        img_arr = (img_arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img_tensor = torch.from_numpy(img_arr).permute(2,0,1)  # [3,H,W]
        return img_tensor
```

上記はRefCOCOデータ用の例ですが、**重要なのは**:

* `prompt_ids`と`answer_ids`を結合してモデル入力`input_ids`を作り、**labelsではプロンプト部分を-100で無視**、応答部分（<SEG>含む）のみ正解トークンとして指定する点です。これにより、モデルはユーザ指示文から先のトークンを予測し、学習時には応答部分のみ損失計算されます。
* セグメンテーションマスクのGTは`mask_labels`としてtensorに含め、<SEG>トークンに対応する出力としてこのマスクが使われます。

**他データセットへの適用**:

* Semantic Segmentationデータでは、`instruction`生成部分で`"Segment the <class_name> in the image."`のような文章を組み立て、あとはRefSegDatasetと同様の処理をします。GTマスクはクラスIDマスクを2値化します（該当クラス=1、他=0）。
* VQAデータでは、`mask_labels`は常にNoneを返し、また<SEG>を含まない純粋テキスト回答なので、`answer_text`には通常の回答文（例："It is a cat."）を入れます。labelsは回答全文のトークン列となり、質問部分は-100でマスクします。
* Reasoning Segmentationデータでは、データに含まれる説明文も`answer_text`に含めます。例えば「Sure, <SEG>. The man is ...」といったテンプレートに埋め込むか、データがそのフォーマットで用意されているなら直接使います。説明文が長くmax\_lengthを超える場合は切り詰めも検討します。

### 1-2. 複数データセットの統合・サンプリング戦略

各データセットクラスが実装できたら、次は**それらを同時に学習させる仕組み**です。方法としては:

* **方法A: 単一Datasetに統合** – 異なるタスクのサンプルを一つの巨大なリストにまとめ、`Dataset`として扱う。ただしタスク間のデータサイズ差が大きいため、そのままだと出現頻度に偏りが出ます。
* **方法B: 複数DataLoaderを併用** – 複数のDataLoader（各タスク毎）からラウンドロビンまたは確率的にサンプルを取得し、ミニバッチを構成する。

LISAの実装ではDeepSpeedのカスタムローダを使い、**`--dataset="sem_seg||refer_seg||vqa||reason_seg"`のように指定して`--sample_rates="9,3,3,1"`で比率を制御**していました。これに倣い、**タスクごとのサンプル比率を決めてミニバッチを組む**のが望ましいです。例えば:

* セマンティックSeg: 9
* RefCOCO類: 3
* VQA: 3
* ReasonSeg: 1

上記の比率はLISAで使われた例ですが、LISA改でもベースはこれを参考に開始し、学習の進行に応じて調整します。実現方法の一例として、PyTorchの`IterableDataset`を用いて**各タスクのデータ流を重み付きにサンプリング**し続ける、またはバッチごとにタスクを固定して取り出すなどが考えられます。

**実装案**: シンプルなアプローチとして、各Datasetからデータを引く**イテレータ**を用意し、毎バッチごとにループでタスクを選択します。例えばバッチサイズ`N`に対し、以下のように構成:

```python
import random
from torch.utils.data import DataLoader

# 各サブデータセットとデータローダ
sem_loader = iter(DataLoader(sem_dataset, batch_size=1, shuffle=True, drop_last=True))
ref_loader = iter(DataLoader(ref_dataset, batch_size=1, shuffle=True, drop_last=True))
vqa_loader = iter(DataLoader(vqa_dataset, batch_size=1, shuffle=True, drop_last=True))
reason_loader = iter(DataLoader(reason_dataset, batch_size=1, shuffle=True, drop_last=True))

sample_rates = [9, 3, 3, 1]
task_loaders = [sem_loader, ref_loader, vqa_loader, reason_loader]
task_names = ["sem_seg", "refer_seg", "vqa", "reason_seg"]

batch = {"pixel_values": [], "input_ids": [], "labels": [], "mask_labels": []}
for i in range(N):
    # 確率的にタスク選択
    task_idx = random.choices(range(len(task_loaders)), weights=sample_rates, k=1)[0]
    loader = task_loaders[task_idx]
    try:
        sample = next(loader)
    except StopIteration:
        # データセットを最後まで読んだ場合は新たなイテレータ生成
        task_loaders[task_idx] = iter(DataLoader(datasets[task_idx], batch_size=1, shuffle=True, drop_last=True))
        loader = task_loaders[task_idx]
        sample = next(loader)
    # sampleは各Datasetの__getitem__が返す辞書
    for key in batch:
        batch[key].append(sample[key])
# 各リストをtensorにスタックし、最終的なミニバッチtensor作成
for key in batch:
    batch[key] = torch.cat(batch[key], dim=0)
```

上記は概念実装ですが、最終的に`batch`がバッチサイズ`N`分のデータを含む辞書となります。**注意**: 異なるタスク間で`input_ids`長さや画像サイズが異なる場合があるので、collate時にパディングやリサイズを行う必要があります。例えば:

* `input_ids`は最大長に合わせてパディングし、`attention_mask`も生成します。
* 画像`pixel_values`もサイズ統一済みであればスタックするだけですが、異なる解像度ならDataLoader側でtransformを統一するか、Collate関数内でパディング（これは画像ではなくCrop/Resizeで合わせる方が現実的）します。
* `mask_labels`は0/1マスクですが、バッチ内で画像サイズが違うと困るため、**事前に画像解像度は統一**しておく方がシンプルです（例えば全データセット画像を短辺384ピクセルにリサイズなど）。

以上のようなカスタムバッチ形成を行うクラスを実装し、`DataLoader`の`collate_fn`に渡す方法もあります。もしくは、最新のPyTorchでは`Generators`や`ChainDataset`を使ったりしてシーケンシャルに複数データセットを一つに纏めることも可能です。

**ポイント**:

* **タスク識別**: 各データポイントがどのタスク由来かを後段で知る必要は基本的にありません。Loss計算時に`mask_labels`がNoneかどうかで「テキストのみ損失」か「テキスト＋マスク損失」かを分岐しているので、データにその区別が内包されています。
* **エポック**: マルチデータ混合の場合、明確な「エポック」の概念は持ちにくくなります。各データセットが異なるサイズなので、適当な反復回数で1エポック相当とみなします（例えば一番大きなデータセット基準で揃える、もしくは総サンプル数合計をバッチサイズで割った回数など）。

