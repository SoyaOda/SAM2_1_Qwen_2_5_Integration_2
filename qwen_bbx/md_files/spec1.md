以下は、**FoodSegなど手元の料理画像**を使って **Qwen2.5‑VL‑72B‑Instruct** による **Bounding Box＋ラベル生成をテスト**し、**可視化で確認**できる**最小ながら実運用向け**のプロジェクト計画と、**曖昧性のないコード一式**です。
クラウド推論前提（Alibaba Cloud Model Studio / DashScope の **OpenAI互換API**）で、**ローカル画像は base64 の Data URL** で投入します（公式が Data URL 形式をサポート）([AlibabaCloud][1])。Qwen2.5‑VL は**BBox/ポイントのグラウンディング**と**安定したJSON**をサポートし([Hugging Face][2], [Qwen][3], [GitHub][4])、モデルIDは **`qwen2.5-vl-72b-instruct`** を使用します（OpenAI互換APIの例でも明示）([AlibabaCloud][1])。データは **FoodSeg103 / UEC‑FoodPix Complete** を想定します（ダウンロード元・概要も併記）([GitHub][5], [Xiongwei's Homepage][6], [mm.cs.uec.ac.jp][7])。

---

## 1) プロジェクト構成

```
qwen-vl-bbox-demo/
├─ README.md
├─ requirements.txt
├─ config.yaml
├─ src/
│  ├─ prompts.py
│  ├─ qwen_client.py
│  ├─ dataset_utils.py
│  ├─ visualize.py
│  └─ run_infer.py
└─ outputs/
   ├─ json/        # 生出力（JSON）
   └─ viz/         # 可視化画像（BBox描画）
```

* **バックエンド**: DashScopeのOpenAI互換API（`base_url=https://dashscope-intl.aliyuncs.com/compatible-mode/v1`）。モデルは **`qwen2.5-vl-72b-instruct`**。([AlibabaCloud][1])
* **画像入力**: 公式推奨の **Data URL (base64)** を `image_url` に埋め込み（`data:image/jpeg;base64,...` など）([AlibabaCloud][8])。
* **出力仕様（モデルへの強制）**: **厳格JSONのみ**を返すプロンプトで、**正規化座標**（0–1の `xyxy_norm`）を義務化。**元画像サイズ**はプロンプトに明示し、可視化側で**絶対座標に復元**。QwenでのBBox JSON出力は公式でもサポートが明記されています。([Hugging Face][2], [Qwen][3])

---

## 2) セットアップ

**requirements.txt**

```txt
openai>=1.40.0
opencv-python>=4.9.0.80
pillow>=10.3.0
numpy>=1.26.4
pyyaml>=6.0.1
tqdm>=4.66.4
```

**環境変数**

```bash
export DASHSCOPE_API_KEY="あなたのAPIキー"   # Alibaba Cloud Model Studioで発行
```

> APIのOpenAI互換エンドポイント、モデル名、使い方は公式ドキュメントを参照ください。([AlibabaCloud][1])

**config.yaml（例）**

```yaml
provider:
  base_url: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
  model: "qwen2.5-vl-72b-instruct"
  api_key_env: "DASHSCOPE_API_KEY"
  request_timeout_s: 120
  max_retries: 2
  temperature: 0.0
  top_p: 0.1
inference:
  max_items: 50            # 処理する最大枚数
  conf_threshold: 0.20     # 可視化用: 信頼度閾値
  iou_merge_threshold: 0.0 # 0ならマージしない（任意機能）
dataset:
  # FoodSeg103の画像ディレクトリを指定（例）
  input_dir: "/path/to/FoodSeg103/images" 
  # 出力
  out_json_dir: "outputs/json"
  out_viz_dir: "outputs/viz"
```

---

## 3) データ（FoodSeg / UEC-FoodPix）

* **FoodSeg103**: 食材レベルのピクセルラベル付き。公式・ベンチマークまとめやGitHub情報参照（7118～9490枚、103/154食材クラス）([GitHub][5], [Xiongwei's Homepage][6])。
* **UEC‑FoodPix Complete**: 10,000枚、102カテゴリ、ピクセル精密マスク（RチャンネルにラベルID）。公式ページと論文PDF参照。([mm.cs.uec.ac.jp][7])

> 本デモは **検出（BBox）とラベリングの目視検証**が目的なので、まずは**画像のみ**を使います（GTマスクは評価で応用可）。

---

## 4) 実装（コード一式）

### 4.1 `src/prompts.py`（モデル出力を厳格JSONに固定）

```python
# -*- coding: utf-8 -*-
# src/prompts.py
from textwrap import dedent

def build_bbox_prompt(image_w: int, image_h: int, language: str = "ja") -> str:
    """
    Qwen2.5-VL に対し、料理・食材の検出＋BBox＋ラベルを厳格JSONで出させるプロンプト。
    - 出力は JSON のみ（前後の説明文・コードフェンス禁止）。
    - BBox は正規化 xyxy (x_min, y_min, x_max, y_max) in [0,1]。小数6桁以内。
    - confidence は [0,1]。
    """
    jp_instr = f"""
    あなたは食品画像の物体検出アシスタントです。
    入力画像の元サイズは width={image_w}px, height={image_h}px です。
    次の要件で出力してください：

    1) 出力は日本語ラベルを含む **厳密なJSON** のみ。前後の文章やコードフェンスは出力しない。
    2) 各検出物体は「料理または食材」に限定し、食器・カトラリー・影は除外する。
    3) BBox は **正規化xyxy** 座標 (x_min, y_min, x_max, y_max) を [0,1] 範囲で返す。
       0未満や1超過は四捨五入でクリップ。小数点は最大6桁。
    4) 各要素に {{"label_ja": str, "bbox_xyxy_norm": [x1,y1,x2,y2], "confidence": float}} を含める。
    5) 料理名・食材名は可能な限り正確な日本語（漢字かな交じり）で。
    6) 返す JSON のトップレベルは {{ "detections": [ ... ] }} とする。

    出力例：
    {{
      "detections": [
        {{"label_ja": "ご飯（白飯）", "bbox_xyxy_norm": [0.12, 0.40, 0.58, 0.78], "confidence": 0.87}},
        {{"label_ja": "カレー（ルウ）", "bbox_xyxy_norm": [0.20, 0.45, 0.70, 0.82], "confidence": 0.81}}
      ]
    }}
    """
    return dedent(jp_instr).strip()
```

> **根拠**: Qwen2.5‑VL は**BBox/ポイントの位置合わせ**と**標準化JSON**の出力をサポートと明記（ブログ/モデルカード）([Qwen][3], [Hugging Face][2])。

---

### 4.2 `src/qwen_client.py`（OpenAI互換APIで呼び出し）

````python
# -*- coding: utf-8 -*-
# src/qwen_client.py
import os
import base64
import mimetypes
import json
from typing import Dict, Any, List
from openai import OpenAI, APIError, APITimeoutError

def encode_image_to_data_url(path: str) -> str:
    """ローカル画像を base64 Data URL (image/*) に変換。"""
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        # 既定はjpeg
        mime = "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    # 公式ガイドに従い data URL を構築
    return f"data:{mime};base64,{b64}"

def extract_text_content(chat_completion) -> str:
    """
    DashScopeのOpenAI互換APIはmessage.contentが文字列または配列の可能性がある。
    両対応で文字列テキストを抽出。
    """
    # v1: choices[0].message.content (str)
    content = None
    try:
        content = chat_completion.choices[0].message.content
        if isinstance(content, list):
            # [{"type":"text","text":"..."}] 形式を想定
            texts = [c.get("text") for c in content if isinstance(c, dict) and c.get("type") == "text"]
            content = "\n".join([t for t in texts if t])
    except Exception:
        pass
    if not content:
        # モデルダンプ経由の保険
        as_dict = json.loads(chat_completion.model_dump_json())
        msg = as_dict["choices"][0]["message"]["content"]
        if isinstance(msg, list):
            texts = [x.get("text") for x in msg if isinstance(x, dict)]
            content = "\n".join([t for t in texts if t])
        else:
            content = msg
    return content

def call_qwen_bbox(
    api_key: str,
    base_url: str,
    model: str,
    image_data_urls: List[str],
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    top_p: float = 0.1,
    timeout_s: int = 120,
    max_retries: int = 2
) -> Dict[str, Any]:
    """Qwen2.5‑VLに画像＋指示を送り、JSON文字列をdictにパースして返す。"""
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            *[
                {"type": "image_url", "image_url": {"url": data_url}}
                for data_url in image_data_urls
            ],
            {"type": "text", "text": user_prompt}
        ]}
    ]
    last_err = None
    for _ in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
            )
            text = extract_text_content(resp).strip()
            # モデルが ```json フェンスを出す場合を除去
            if text.startswith("```"):
                text = text.strip("`")
                # 先頭に "json" が付いている可能性
                if text.lower().startswith("json"):
                    text = text[4:].strip()
            # JSONへ
            return json.loads(text)
        except (APIError, APITimeoutError, json.JSONDecodeError) as e:
            last_err = e
            continue
    raise RuntimeError(f"Qwen呼び出しに失敗: {last_err}")
````

> **根拠**: モデル名とOpenAI互換APIの利用、複数画像・image\_url型の使い方は公式に記載。**base64のData URL**はサポートされる方式です。([AlibabaCloud][1])

---

### 4.3 `src/dataset_utils.py`（画像の列挙）

```python
# -*- coding: utf-8 -*-
# src/dataset_utils.py
import os
from typing import List, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_images(root_dir: str, max_items: int = 0) -> List[str]:
    """ディレクトリ配下の画像パス一覧を取得。max_items>0なら先頭N件に制限。"""
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in sorted(filenames):
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXTS:
                paths.append(os.path.join(dirpath, fn))
    if max_items and len(paths) > max_items:
        paths = paths[:max_items]
    return paths
```

> **補足**: FoodSeg103やUEC‑FoodPix Completeのディレクトリ配置は配布により若干異なります（公式ページ・論文参照）。まずは画像のみあればOKです。([GitHub][5], [mm.cs.uec.ac.jp][7])

---

### 4.4 `src/visualize.py`（BBoxの描画と保存）

```python
# -*- coding: utf-8 -*-
# src/visualize.py
import os
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple

def norm_xyxy_to_abs(b: List[float], w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = int(round(max(0.0, min(1.0, b[0])) * w))
    y1 = int(round(max(0.0, min(1.0, b[1])) * h))
    x2 = int(round(max(0.0, min(1.0, b[2])) * w))
    y2 = int(round(max(0.0, min(1.0, b[3])) * h))
    # 座標整合
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return x1, y1, x2, y2

def draw_detections(
    img_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
    conf_thres: float = 0.2
) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]
    for det in detections:
        try:
            label = det.get("label_ja", "item")
            conf = float(det.get("confidence", 0.0))
            if conf < conf_thres:
                continue
            box = det["bbox_xyxy_norm"]
            x1, y1, x2, y2 = norm_xyxy_to_abs(box, w, h)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label} {conf:.2f}"
            # テキスト背景
            (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
            cv2.putText(out, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        except Exception:
            continue
    return out

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
```

---

### 4.5 `src/run_infer.py`（CLI：推論→JSON保存→可視化保存）

```python
# -*- coding: utf-8 -*-
# src/run_infer.py
import os
import cv2
import json
import yaml
from tqdm import tqdm
from PIL import Image
from src.prompts import build_bbox_prompt
from src.qwen_client import call_qwen_bbox, encode_image_to_data_url
from src.dataset_utils import list_images
from src.visualize import draw_detections, ensure_dir

def main():
    # 設定読込
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_url    = cfg["provider"]["base_url"]
    model       = cfg["provider"]["model"]
    api_key_env = cfg["provider"]["api_key_env"]
    timeout_s   = int(cfg["provider"]["request_timeout_s"])
    max_retries = int(cfg["provider"]["max_retries"])
    temperature = float(cfg["provider"]["temperature"])
    top_p       = float(cfg["provider"]["top_p"])

    input_dir   = cfg["dataset"]["input_dir"]
    out_json    = cfg["dataset"]["out_json_dir"]
    out_viz     = cfg["dataset"]["out_viz_dir"]
    max_items   = int(cfg["inference"]["max_items"])
    conf_thres  = float(cfg["inference"]["conf_threshold"])

    ensure_dir(out_json); ensure_dir(out_viz)

    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"環境変数 {api_key_env} が未設定です。")

    img_paths = list_images(input_dir, max_items=max_items)
    if not img_paths:
        raise RuntimeError(f"画像が見つかりません: {input_dir}")

    for path in tqdm(img_paths, desc="Qwen2.5-VL 推論"):
        # 画像読み込み
        pil = Image.open(path).convert("RGB")
        w, h = pil.size

        # プロンプト
        system_prompt = "出力は厳格なJSONのみ。説明文・コードフェンスを一切含めないこと。"
        user_prompt   = build_bbox_prompt(w, h)

        # 画像→Data URL
        data_url = encode_image_to_data_url(path)

        # Qwen呼び出し
        result = call_qwen_bbox(
            api_key=api_key,
            base_url=base_url,
            model=model,
            image_data_urls=[data_url],  # 1画像（複数枚でも可）
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            top_p=top_p,
            timeout_s=timeout_s,
            max_retries=max_retries
        )

        # JSON保存
        stem = os.path.splitext(os.path.basename(path))[0]
        json_path = os.path.join(out_json, f"{stem}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # 可視化
        import numpy as np
        img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        detections = result.get("detections", [])
        viz = draw_detections(img_bgr, detections, conf_thres=conf_thres)
        viz_path = os.path.join(out_viz, f"{stem}.jpg")
        cv2.imwrite(viz_path, viz)

    print(f"完了: JSON→{out_json} / 可視化→{out_viz}")

if __name__ == "__main__":
    main()
```

**実行例**

```bash
# 仮想環境は任意
pip install -r requirements.txt

# config.yaml を編集（input_dirなど）
python -m src.run_infer
# => outputs/json/*.json, outputs/viz/*.jpg が生成されます
```

---

## 5) 期待される出力（例）

* `outputs/json/xxx.json`

```json
{
  "detections": [
    { "label_ja": "ご飯（白飯）", "bbox_xyxy_norm": [0.121, 0.402, 0.582, 0.781], "confidence": 0.87 },
    { "label_ja": "カレー（ルウ）", "bbox_xyxy_norm": [0.201, 0.449, 0.699, 0.823], "confidence": 0.81 }
  ]
}
```

* `outputs/viz/xxx.jpg`
  → 緑枠でBBox＋「ラベル 信頼度」を重畳表示。

> **注意（実務Tip）**: 一部環境では**画像の内部スケーリング**により、モデルが絶対ピクセル座標を直接返すとズレる報告があります。**正規化座標（0–1）で返させ、可視化で元解像度へ復元**するのが安全です（コミュニティでも議論あり）([GitHub][9])。

---

## 6) 複数画像・動画フレームの投入

Qwen2.5‑VLは**複数画像の同時入力**に対応（OpenAI互換APIで `messages[].content` に `image_url` を複数列挙）します。必要なら `image_data_urls=[img1, img2, ...]` として渡してください（公式でもマルチ画像・動画の扱いが紹介）([AlibabaCloud][10])。

---

## 7) FoodSeg / UEC-FoodPix の取得メモ

* **FoodSeg103**（研究ページ・GitHub・Kaggleまとめ等）([GitHub][5], [Xiongwei's Homepage][6], [Dataset Ninja][11], [Kaggle][12])
* **UEC‑FoodPix Complete**（公式ページと解説PDF・論文）([mm.cs.uec.ac.jp][7], [ACM Digital Library][13])

> それぞれ配布条件に従って取得・利用してください。

---

## 8) 発展：評価・改善のためのオプション

* **評価（擬似的）**: FoodSegはピクセルマスクのみだが、**GTマスク→最小外接BBox**を求め、Qwen検出とIoUで突合せ可能。
* **品質ゲート**: 低信頼度（`confidence<thres`）の除外、重複BBoxのNMS/WBFなど。
* **検出漏れ補強**: 汎用のオープン語彙検出器（Grounding DINO等）とWBFで併用→SAM2.1に誘導する構成も強力（今回はQwen単独デモ）。
* **日本語正規化**: ラベルの漢字/かな表記ゆれは正規化辞書で統一（成分DB連携の前段整備）。

---

## 9) 参考（主要根拠）

* **Qwen2.5‑VL‑72B‑Instruct** モデルカード（**BBox/ポイントの可視化／安定JSON出力**に言及）および 7B 版カード。([Hugging Face][2])
* **公式ブログ**：**標準化JSON**でのグラウンディング（BBox/ポイント）を強調。([Qwen][3])
* **GitHub: Qwen2.5‑VL**：**Precise Object Grounding Across Formats** を明示。([GitHub][4])
* **DashScope OpenAI互換API**：**`qwen2.5-vl-72b-instruct`** のモデル名と呼出し例、**複数画像/動画入力**対応。([AlibabaCloud][1])
* **Base64 Data URL入力**：公式ガイド。([AlibabaCloud][8])
* **FoodSeg103 / UEC‑FoodPix Complete**：データセット公式情報。([GitHub][5], [Xiongwei's Homepage][6], [mm.cs.uec.ac.jp][7])

---

### 付録：README.md（サンプル）

```md
# Qwen2.5‑VL‑72B の料理検出（BBox＋ラベル）デモ

## 概要
- Qwen2.5‑VL‑72B‑Instruct を DashScope の OpenAI 互換 API で呼び出し、
- FoodSeg などローカル画像を **base64 Data URL** で投入して、
- **厳格JSON**で返る BBox＋ラベルを保存＆**可視化**します。

## 使い方
1. `pip install -r requirements.txt`
2. `export DASHSCOPE_API_KEY="..."` を設定
3. `config.yaml` の `dataset.input_dir` を画像ディレクトリに変更
4. `python -m src.run_infer`
   - JSON → `outputs/json/`
   - 可視化 → `outputs/viz/`

## 注意
- BBoxは**正規化xyxy**で返させ、描画時に元解像度へ復元します。
- 返りJSONはモデル応答の素体なので、必要に応じてNMS等で整形してください。
```

---

必要であれば、このベースに **Grounding DINO→SAM2.1 併用**の二系統誘導や、**FoodSegマスク→擬似GT BBox**による簡易定量評価スクリプトも拡張でお付けできます。

[1]: https://www.alibabacloud.com/help/en/model-studio/use-qwen-by-calling-api?utm_source=chatgpt.com "Alibaba Cloud Model Studio:Qwen API reference"
[2]: https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct?utm_source=chatgpt.com "Qwen/Qwen2.5-VL-72B-Instruct"
[3]: https://qwenlm.github.io/blog/qwen2.5-vl/?utm_source=chatgpt.com "Qwen2.5 VL! Qwen2.5 VL! Qwen2.5 VL! | Qwen"
[4]: https://github.com/QwenLM/Qwen2.5-VL?utm_source=chatgpt.com "QwenLM/Qwen2.5-VL"
[5]: https://github.com/LARC-CMU-SMU/FoodSeg103-Benchmark-v1?utm_source=chatgpt.com "LARC-CMU-SMU/FoodSeg103-Benchmark-v1: MM'21 Main-Track paper"
[6]: https://xiongweiwu.github.io/foodseg103.html?utm_source=chatgpt.com "A Large-Scale Benchmark for Food Image Segmentation - Xiongwei Wu"
[7]: https://mm.cs.uec.ac.jp/uecfoodpix/?utm_source=chatgpt.com "UECFoodPix,UECFoodPixComplete"
[8]: https://www.alibabacloud.com/help/en/model-studio/vision/?utm_source=chatgpt.com "How to use the Qwen-VL model"
[9]: https://github.com/QwenLM/Qwen2.5-VL/issues/1030?utm_source=chatgpt.com "OCR Cookbook BBox Issues With Api/Local qwen2.5-vl- ..."
[10]: https://www.alibabacloud.com/help/en/model-studio/use-qwen-by-calling-api "
 Qwen API reference - Alibaba Cloud Model Studio - Alibaba Cloud Documentation Center

"
[11]: https://datasetninja.com/food-seg-103?utm_source=chatgpt.com "FoodSeg103 Dataset"
[12]: https://www.kaggle.com/datasets/ggrill/foodseg103?utm_source=chatgpt.com "foodseg103"
[13]: https://dl.acm.org/doi/10.1007/978-3-030-68821-9_51?utm_source=chatgpt.com "UEC-FoodPix Complete: A Large-Scale Food Image ..."
