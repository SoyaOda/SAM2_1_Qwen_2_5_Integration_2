ありがとうございます。添付の学習ログを確認しました（val\_iou / val\_dice が大きく振れており、train/seg\_loss は0.4〜0.8付近で横ばい、まれにスパイク、train/lm\_loss は初期に急落→ほぼ0、全体 loss は2付近で停滞、learning\_rate は 0→1e-3 まで約1エポックでウォームアップ後、最終まで線形減衰、ステップは \~60k＝10k samples/epoch × 6epochs に一致）。

以下、**何が起きているか → 原因の当たり → GitHub（`minimal_train_backup.py` と `./src`）のどこを直すか → 具体パッチ**の順でまとめます。
（※ご要望通り、参照対象は `minimal_train_backup.py` と `./src` のみ）

---

## 1) いま起きていること（ログから読み取れる事実）

* **val 指標（IOU / DICE）が大振れ**：0.05〜0.9の間で激しく上下。
  → 評価サンプルが `--inference_eval_samples 3` と極端に少ないため、**分散が大きい**。
* **train/seg\_loss が横ばい**（概ね 0.4〜0.8。たまに 2〜6 のスパイク）
  → 改善が緩慢。スパイクは **空マスクや解像度不一致、Dice の数値不安定**が典型。
* **train/lm\_loss は数千 step で 0 近傍**
  → 言語側は極めて簡単に収束。**総損失における寄与が小さくなる**。
* **train/loss が \~2 付近で停滞**
  → seg\_loss（\~0.5前後）だけでは説明できない値。**損失の重み付け／スケーリング不整合**の可能性。
* **learning\_rate の推移が「10k サンプルでウォームアップ完了」**
  → 10k は **サンプル数ベース**。`gradient_accumulation_steps=32` を使っているのに、**スケジューラが「オプティマイザ更新回数」ではなく「サンプル単位」で進んでいる**疑い。
  つまり **ウォームアップ／減衰の設計と実際の step の意味がズレている**可能性が濃厚。

---

## 2) 主要因（優先度順）

### A. **LR スケジューラが「累積（accumulation）を無視」して進んでいる**

* `total_training_steps` を `samples_per_epoch * num_epochs` のように計算し、さらに **`scheduler.step()` を各イテレーション（=サンプル／ミニバッチ）で呼んでいる**と、
  実際の「最適化更新回数」（= `total_samples / (batch * accum)`）より **32倍長い（あるいは短い）ウォームアップ・減衰**になります。
* このズレは **セグメント側の学習を著しく不安定にし、loss が伸び悩む**典型的要因です。

### B. **セグ損失の実装が不安定（Dice の数値不安定、空マスク、補間方法）**

* **Dice の ε・空マスク処理不足**、**BCE の `with_logits`／閾値**の不整合、**mask と logits の解像度／補間のミスマッチ（bilinear で補間→エッジが壊れる）**は、
  **スパイク**と**学習停滞**の定番です。

### C. **損失の重み・スケーリングがアンバランス**

* `lm_loss` は即ゼロ化、しかし `train/loss` が 2 前後で停滞していることから、
  \*\*セグ損失の重み付け（λ）\*\*や、\*\*他タスク損失（もしあれば）\*\*のスケーリング／正規化が正しくなく、**勾配が効いていない**可能性。

### D. **評価分散が大きすぎて正しい進捗が見えない**

* `--inference_eval_samples 3` は実質ほぼノイズ。**真の改善が見えづらい**。

---

## 3) どこを直すか（該当ファイルと想定箇所）

> **検索の目印**（まず該当箇所を探してください）

* `minimal_train_backup.py`

  * `"get_linear_schedule_with_warmup"` などスケジューラ生成箇所
  * `total_training_steps` / `num_training_steps` の計算式
  * 学習ループで `scheduler.step()` を **毎イテレーション**で呼んでいないか
  * 勾配累積の条件分岐（`(step+1) % gradient_accumulation_steps == 0`）の直後に
    `optimizer.step()` / `scheduler.step()` / `zero_grad()` が並んでいるか
  * ロスの合成（`total_loss = λ_seg * seg_loss + λ_lm * lm_loss + ...`）

* `./src/`（典型的な配置名の例を記します。実際の構成にあわせて置き換えてください）

  * `src/losses/*.py` あるいは `src/trainers/*.py`：**セグ損失計算**

    * `BCEWithLogitsLoss` の使用有無、Dice 実装、`ignore_index`、`pos_weight` 等
  * `src/data/*.py`：**マスクの前処理・補間**

    * `F.interpolate(..., mode='bilinear')` を **mask に対して使っていないか**
      → **`mode='nearest'` に修正**
  * `src/eval/*.py`：**IoU/Dice の集計**

    * サンプル数が固定 3 になっていないか、集計がステップ平均でなく**総和でのマクロ平均**になっているか

---

## 4) 具体的な修正パッチ（コピペ可）

### 4-1. **スケジューラと勾配累積の整合（最重要）**

**`minimal_train_backup.py` に適用**（該当箇所を置換／追加）

```python
# ===== (A) 総更新回数の正しい計算 =====
import math

total_samples = args.samples_per_epoch * args.num_epochs
updates_per_epoch = math.ceil(args.samples_per_epoch / (args.batch_size * args.gradient_accumulation_steps))
total_updates = updates_per_epoch * args.num_epochs

warmup_updates = max(1, int(total_updates * args.warmup_ratio))

from transformers import get_linear_schedule_with_warmup
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_updates,
    num_training_steps=total_updates
)
```

```python
# ===== (B) ループ内：scheduler.step() と optimizer.step() の位置 =====
scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

optimizer.zero_grad(set_to_none=True)
global_update = 0

for step, batch in enumerate(dataloader):
    with torch.cuda.amp.autocast(enabled=args.fp16):
        losses = model(**batch)  # 例: 戻り値に 'seg_loss', 'lm_loss' など
        seg_loss = losses['seg_loss']
        lm_loss  = losses.get('lm_loss', 0.0)

        # 勾配累積に合わせてスケール
        total_loss = args.lambda_seg * seg_loss + args.lambda_lm * lm_loss
        total_loss = total_loss / args.gradient_accumulation_steps

    scaler.scale(total_loss).backward()

    # ---- ここが重要：accum 到達時のみ step / scheduler ----
    if (step + 1) % args.gradient_accumulation_steps == 0:
        if args.max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        scheduler.step()            # ← optimizer.step() の後
        global_update += 1
```

> これで **LR のウォームアップ／減衰が「最適化更新回数」に同期**し、勾配累積と破綻しなくなります。

---

### 4-2. **セグ損失を安定化（Dice+BCE、空マスク対応、pos\_weight）**

**新規ファイル** `src/losses/segmentation.py`：

```python
import torch
import torch.nn.functional as F

def dice_loss_with_logits(logits, targets, eps=1e-6):
    # logits: (N,1,H,W) or (N,C,H,W) for binary C=1
    # targets: same spatial size, float {0,1}
    probs = torch.sigmoid(logits)
    dims = tuple(range(2, probs.ndim))

    num = 2.0 * (probs * targets).sum(dims)
    den = (probs.pow(2) + targets.pow(2)).sum(dims) + eps
    dice = 1.0 - (num + eps) / den
    return dice.mean()

def bce_with_logits(logits, targets, pos_weight=None):
    return F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction='mean'
    )

def safe_seg_loss(logits, targets, pos_weight=None, dice_weight=1.0, bce_weight=1.0):
    """
    - 近傍補間や dtype を呼び出し側で合わせてから渡す前提
    - 空マスク時は Dice を無効化し BCE のみで学習
    """
    # targets in {0,1}
    with torch.no_grad():
        has_fg = (targets.sum(dim=(2,3)) > 0)  # (N,1)

    bce = bce_with_logits(logits, targets, pos_weight=pos_weight)

    if has_fg.any():
        dice = dice_loss_with_logits(logits[has_fg.squeeze(1)], targets[has_fg.squeeze(1)])
        loss = bce_weight * bce + dice_weight * dice
    else:
        # 全バッチが空前景なら Dice は定義しない
        loss = bce_weight * bce
    return loss
```

**呼び出し側（例：`minimal_train_backup.py` または `src/trainers/*.py`）**：

```python
from src.losses.segmentation import safe_seg_loss

# 予測とターゲットの空間サイズを一致させ、マスクは NEAREST 補間
logits = seg_outputs["logits"]         # (N,1,H',W')
masks  = batch["masks"].float()        # 0/1 tensor, (N,1,H,W)

if masks.shape[-2:] != logits.shape[-2:]:
    masks = torch.nn.functional.interpolate(
        masks, size=logits.shape[-2:], mode='nearest'
    )

pos_frac = (masks.mean().clamp(min=1e-6)).item()
# 正例が希少なら補正（例: 正例1%なら pos_weight~99）
pos_weight = torch.tensor([(1. - pos_frac) / pos_frac], device=logits.device)

seg_loss = safe_seg_loss(
    logits, masks,
    pos_weight=pos_weight,
    dice_weight=args.dice_weight,    # 例: 1.0
    bce_weight=args.bce_weight       # 例: 1.0
)
```

---

### 4-3. **マスク補間と dtype の厳格化**

**`./src/data/...`（マスクのリサイズ／前処理を行っている箇所）**

```python
# 画像は bilinear, マスクは必ず nearest
img = F.interpolate(img, size=target_hw, mode='bilinear', align_corners=False)

mask = mask.to(torch.float32)
if mask.max() > 1.0:  # 0/255 → 0/1
    mask = (mask > 127.5).float()

mask = F.interpolate(mask, size=target_hw, mode='nearest')
```

> ここが bilinear のままだと **境界が崩れて Dice が不安定**になります。

---

### 4-4. **損失の重みと最適化パラメタグループ**

**`minimal_train_backup.py`**

```python
# λ のデフォルト（例）
args.lambda_seg = getattr(args, "lambda_seg", 1.0)
args.lambda_lm  = getattr(args, "lambda_lm", 0.1)

# パラメタグループ：LM と Seg/SAM を分けて学習率を変える
lr_lm   = getattr(args, "lr_lm", 5e-6)
lr_seg  = getattr(args, "lr_seg", 2e-4)
wd_lm   = getattr(args, "wd_lm", 0.01)
wd_seg  = getattr(args, "wd_seg", 0.01)

seg_params, lm_params = [], []
for n, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if any(k in n for k in [
        'sam', 'mask_decoder', 'prompt_encoder', 'seg_head', 'fusion', 'adapter.seg'
    ]):
        seg_params.append(p)
    else:
        lm_params.append(p)

optimizer = torch.optim.AdamW([
    {'params': lm_params,  'lr': lr_lm,  'weight_decay': wd_lm},
    {'params': seg_params, 'lr': lr_seg, 'weight_decay': wd_seg},
], betas=(0.9, 0.999), eps=1e-8)
```

> **LM をかなり低 LR**、**Seg/SAM は高めの LR** にするのが安定しやすいです。
> さらに、最初の 20% 更新は LM を凍結 → 解除、の段階学習も有効です。

```python
# 例：段階学習（最初の 20% 更新は LM を凍結）
freeze_updates = int(total_updates * 0.2)

def set_lm_requires_grad(enabled: bool):
    for n, p in model.named_parameters():
        if any(k in n for k in ['qwen', 'lm_head', 'language_model']):
            p.requires_grad = enabled

set_lm_requires_grad(False)

# ループ内の更新箇所で：
if global_update == freeze_updates:
    set_lm_requires_grad(True)
```

---

### 4-5. **メトリクスの集計（振れを抑える）**

**`minimal_train_backup.py`（評価時）**

```python
# 3枚評価はノイズが大きいので、最低でも 64〜256 に
args.inference_eval_samples = max(args.inference_eval_samples, 128)

# IoU / Dice は「サンプル毎→総和→最後に割る」マクロ平均にする
def accumulate_iou_dice(iou_sum, dice_sum, count, logits, masks):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    inter = (preds * masks).sum(dim=(2,3))
    union = (preds + masks - preds*masks).sum(dim=(2,3))
    iou  = (inter / (union + 1e-6)).mean()
    dice = (2*inter / (preds.sum(dim=(2,3)) + masks.sum(dim=(2,3)) + 1e-6)).mean()

    iou_sum  += iou.item()
    dice_sum += dice.item()
    count    += 1
    return iou_sum, dice_sum, count
```

---

### 4-6. **スパイク対策（空マスク／極端に小さい前景の扱い）**

**`src/losses/segmentation.py` の `safe_seg_loss` で既に対応**していますが、
**データ側でも**極端に前景が小さいケースには **タスク混合時のサンプリング重み付け**を追加すると安定します。

```python
# 例：Sampler で「前景画素率」が一定以上のサンプルを優先
# （既存 Sampler があれば、その weight を算出する前処理を追加）
```

---

## 5) すぐに試せる再学習コマンド（推奨値）

```bash
python minimal_train_backup.py \
  --samples_per_epoch 10000 \
  --batch_size 1 \
  --gradient_accumulation_steps 32 \
  --num_epochs 6 \
  --save_steps 1000 \
  --visualize --visualize_steps 1000 \
  --run_inference_eval --inference_eval_samples 256 \
  --warmup_ratio 0.1 \
  --use_wandb --wandb_project "full_sigma_add20250818_fix" \
  --fp16 \
  --lambda_seg 1.0 --lambda_lm 0.1 \
  --lr_lm 5e-6 --lr_seg 2e-4 \
  --wd_lm 0.01 --wd_seg 0.01 \
  --dice_weight 1.0 --bce_weight 1.0 \
  --max_grad_norm 1.0
```

> 変更点の要旨
>
> * **評価サンプル 256** に増加（振れを抑える）
> * **LM と Seg で別 LR**（LM は極小、Seg はやや大きめ）
> * **学習ループ／スケジューラ修正**を前提
> * **AMP + 勾配クリップ**でスパイク低減

---

## 6) 追加の健全性チェック（落とし穴つぶし）

* 乱数固定：`torch.manual_seed`, `np.random.seed`, `random.seed`、`cudnn.deterministic=True`
* ログ：`train/learning_rate` のカーブが **「総更新回数」に対して 10% で最大**になることを確認
  （= 1875 updates 程度なら **約 187 updates** でピーク）
* 予測とマスクのサイズ一致・dtype 一致を **学習と評価の両方**で assert
* BCE は **`with_logits`** を使い、推論時だけ `sigmoid→threshold`
* クラス不均衡の強いデータでは `pos_weight` の自動計算（上記コード）を必ず有効化
* まれに `seg_loss` が 5〜6 へスパイク → バッチ内に「サイズ不一致」「全ゼロ／全1」「NaN 発生」がないか即時ログ

---

## 7) まとめ（因果と対策の対応関係）

| 症状                   | 主要因                      | 対策                            |
| -------------------- | ------------------------ | ----------------------------- |
| seg\_loss が下がらない／横ばい | スケジューラが累積無視／LR スケジュールの破綻 | **4-1 の修正**（最優先）              |
| seg\_loss スパイク       | 空マスク・Dice 不安定・補間ミス       | **4-2/4-3/4-6 の修正**           |
| val 指標が大振れ           | 評価サンプル 3                 | **4-5 の修正（>=128）**            |
| total loss が2付近で停滞   | 損失重み／スケーリング不整合           | **4-4 で λ・LR を分離**、LM 初期凍結も検討 |

---

上記パッチを `minimal_train_backup.py` と `./src` にそのまま反映していただければ、
**学習率の挙動が安定 → セグ損失が素直に減少 → val 指標の分散縮小**が期待できます。
まずは **スケジューラと勾配累積の整合（4-1）** を最優先で修正し、次に **セグ損失（4-2）とマスク補間（4-3）** を適用する流れをおすすめします。

必要であれば、実際の `minimal_train_backup.py` / `src/*` の該当行に合わせて\*\*差分形式（unified diff）\*\*で作り直しますので、現在の該当ファイル断片を貼ってください。
