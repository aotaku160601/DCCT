# DCCT（Color Matters）再現実装

Zhong, Xu, Zou. *Color Matters: Demosaicing-Guided Color Correlation Training for Generalizable
AI-Generated Image Detection* (arXiv:2601.22778, 2026) のスタンドアロン再現実装。

設計資料（プロジェクト内資料）:

- `01_要件定義書_DCCT再現実装.md`
- `02_基本設計書_DCCT再現実装.md`
- `03_詳細設計書_DCCT再現実装.md`
- `04_DB設計書_DCCT再現実装.md`
- `05_図表集_DCCT再現実装.md`

## セットアップ

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`python` が「Python」とだけ表示して何も起きない場合は、Microsoft Store のスタブが
呼ばれている。`py -3` を使うか、python.org 版をインストールして PATH に通すこと。

CPUのみで動かす場合は、CPU版wheelを先に入れてから残りを入れる:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements.txt
```

### 実行デバイス

`configs/base.yaml` の `device.type` は既定 `auto` で、**cuda → mps → cpu** の順に自動選択する
（`src/experiment/device.py`）。`torch.cuda.*` を直接呼ぶコードは書かない（02_基本設計書 9節）。

**本実装の想定実行環境は macOS Tahoe + Apple Silicon（`mps`）**（01_要件定義書 5節）。
`device.enable_mps_fallback: true` により `PYTORCH_ENABLE_MPS_FALLBACK=1` を設定し、
MPS未対応演算をCPUへ自動フォールバックさせる。

- macOS + Apple Silicon → `mps`（本番）
- NVIDIA GPU 環境 → `cuda`
- いずれも無い場合 → `cpu`（CIやスモークテスト用）

`paths.dataset_root` は macOS のマウントポイント `/Volumes/Extreme Pro/...` を指す。
別OSで動かす場合はこの値だけを差し替える（コード側は変更不要）。

## ディレクトリ構成

02_基本設計書 5節の `dcct-reproduction/` に相当する内容を、本リポジトリのルート直下に配置している
（リポジトリ名が `DCCT` であり、階層を1段増やす意味がないため）。

```
.
├── configs/                  # 実験設定（YAML）
│   ├── base.yaml             # 共通設定（パス・seed・デバイス・前処理パラメータ）
│   ├── stage1_photo.yaml     # pθ学習
│   ├── stage1_ai.yaml        # qφ学習
│   ├── stage2_classifier.yaml# 分類器gψ学習
│   └── ablation/             # Table 4a〜4d相当
├── src/
│   ├── data/                 # ingest / DBアクセス層
│   ├── preprocess/           # CFAマスク / ハイパスフィルタ / パッチサンプリング
│   ├── model/                # ConditionalUNet / BinaryClassifier / losses
│   ├── train/                # Stage I-A/I-B, Stage II 学習制御
│   ├── eval/                 # cross-generator / ablation / robustness
│   ├── experiment/           # config読み込み・実験ID発行・ログ管理
│   ├── report/               # 結果集計・レポート出力
│   └── cli.py                # CLIエントリポイント
├── migrations/               # DBマイグレーションSQL
├── db/                       # メタデータDB（SQLite、gitignore対象）
├── checkpoints/              # 学習チェックポイント
├── logs/  reports/  tests/
```

## CLIコマンド

| コマンド | 内容 |
|---|---|
| `python -m src.cli init-db --config configs/base.yaml` | メタデータDBを作成 |
| `python -m src.cli ingest --config configs/base.yaml` | 外部SSD上のGenImageを走査しDBへ登録 |
| `python -m src.cli train-stage1 --target photo --config configs/stage1_photo.yaml` | pθを学習 |
| `python -m src.cli train-stage1 --target ai --config configs/stage1_ai.yaml` | qφを学習 |
| `python -m src.cli train-stage2 --config configs/stage2_classifier.yaml` | 分類器gψを学習 |
| `python -m src.cli evaluate --mode cross_generator --experiment-id <ID>` | Table1相当の評価 |
| `python -m src.cli report --experiment-id <ID> --out reports/` | レポート出力 |

## メタデータDB

04_DB設計書 4節のDDLと5節のインデックスを `migrations/0001_init.sql` にそのまま格納している。

```bash
python -m src.cli init-db --config configs/base.yaml          # 未適用のマイグレーションを適用
python -m src.cli init-db --config configs/base.yaml --force  # バックアップを取って作り直す
```

- 適用済みマイグレーションは `schema_migrations` テーブルで管理する（設計書のDDLには含まれない
  ランナー側の管理テーブル）。`init-db` は冪等で、再実行しても既存データを壊さない。
- `--force` は削除前に `db/backups/dcct_metadata_<timestamp>.sqlite3` へコピーを退避する
  （04_DB設計書 7節「バックアップ」）。
- 接続時に `PRAGMA foreign_keys = ON` と `journal_mode = WAL` を設定する。
- スキーマ変更はDBファイルを直接編集せず、`migrations/0002_*.sql` を追加して行う。

## テスト

```bash
python -m pytest
```

## データ取り込み（ingest）

```bash
python -m src.cli ingest --config configs/base.yaml
python -m src.cli ingest --config configs/base.yaml --generator SDv1.4   # 生成器を限定
python -m src.cli ingest --config configs/base.yaml --limit 100 --dry-run  # スモークテスト
```

走査対象は `configs/base.yaml` の `dataset.sources`（`dataset_root` からの相対パス）で定義する。
ディレクトリでもZIPでも同じインタフェースで扱う。

### ZIPは展開しない

SDv1.4 のZIPだけで 96.46 GB あるため、展開せず `zipfile` で個々の画像を遅延読み出しする。
`images.filepath` には次の形式で記録する（DBスキーマの変更は不要）。

```
ディレクトリ: /Volumes/Extreme Pro/BigGAN/train/ai/xxx.png
ZIP内:        /Volumes/Extreme Pro/genimage-...zip!train/ai/xxx.png
```

`src.data.sources.open_image_bytes()` がどちらの形式も透過的に読む。学習時は
DataLoaderのワーカーごとにZIPを開き直すこと（`ZipFile` はプロセス・スレッド間で共有しない）。

### ディレクトリ構成の判定

相対パスの末尾から「親ディレクトリ＝クラス（`ai` / `nature`）」「その親＝split（`train` / `val`）」
として判定するため、`imagenet_ai_0419_biggan/train/ai/...` のように上位フォルダが挟まっても
同じルールで取り込める。表記ゆれは `dataset.directory_aliases` で吸収する。

### split の割当

GenImage は `train/` と `val/` しか持たないため、以下のように割り当てる
（`dataset.split` で変更可能）。

| GenImage側 | DBの split | 用途 |
|---|---|---|
| `val/` | `test` | Table 1 の cross-generator 評価 |
| `train/` の95% | `train` | Stage I / Stage II の学習 |
| `train/` の5% | `val` | 閾値τのチューニング、チェックポイント選択 |

train/val の振り分けは `blake2b(seed + filepath)` で決まるため、再実行しても同じ画像が
同じ split に入る（NFR-1 再現性）。評価用データに触れずにτを決められる。

### 実写画像（nature/）の生成器紐づけ

`dataset.real_generator_naming: per_source`（既定）では、生成器ごとに
`ImageNet(real)@SDv1.4` のような実写用 generator を作る。cross-generator 評価で
「その生成器に対応する実写セット」を generator_id だけで一意に引けるようにするためで、
`shared` にすると全生成器の `nature/` を単一の `ImageNet(real)` にまとめる。

### 取得するメタデータ

パス・ラベル・split・ファイルサイズに加え、画像ヘッダ（先頭64KB）のみを復号して
width/height を取得する。これにより ingest 時点で

- 読み込めない画像 → `status='invalid'`
- 64×64 未満の画像 → `is_croppable=0`

を判定でき、`MetadataRepository.list_images()` が学習・評価対象から自動的に除外する
（03_詳細設計書 6節）。`checksum` は全データを読み直すことになるため ingest では算出しない。

生成器（ソース）単位でトランザクションを分けているので、途中で失敗しても成功済みの
生成器はDBに残り、`--generator` で失敗分だけ再実行できる。`filepath` が UNIQUE のため
再実行しても重複登録は起きない。

## 前処理とモデル

Algorithm 1 / 2 の各行に対応するクラスは次のとおり。

| クラス | ファイル | 役割 |
|---|---|---|
| `CFAMask` / `RandomChannelMask` | `src/preprocess/cfa_mask.py` | 観測1ch（x）と隠された2ch（y）への分離 |
| `HighPassFilterBank` | `src/preprocess/highpass.py` | SRM 30種フィルタと truncation |
| `PatchSampler` | `src/preprocess/patch_sampler.py` | 64×64クロップ、JPEG拡張、推論時のPパッチ |
| `DCCTPreprocessor` | `src/preprocess/pipeline.py` | 上記を束ねて (x', y') を作る |
| `ConditionalUNet` / `MixtureParams` | `src/model/conditional_unet.py` | pθ / qφ |
| `NLLLoss` / `BCELoss` | `src/model/losses.py` | Stage I / Stage II の損失 |
| `BinaryClassifier` | `src/model/classifier.py` | gψ（ResNet + Transformer） |
| ファクトリ | `src/experiment/builders.py` | configからの組み立て |

画素値は **0〜255スケールのfloatのまま**扱う。truncation閾値 t=7 はこのスケールに対する値で、
[0,1] へ正規化してしまうと意味を持たなくなるため。

### CFAマスク

Bayer配列（既定RGGB）に従って画素ごとに1chを観測とし、残り2chを y とする。
y のチャンネル順は常にRGBインデックスの昇順（観測がRなら (G,B)、Gなら (R,B)、Bなら (R,G)）。
x と y を合わせると元のRGBが過不足なく復元できることをテストで担保している。
Ablation-B の「CFAマスクなし」は `RandomChannelMask`（画素ごとにランダムな1ch）に差し替える。

### 30種ハイパスフィルタ

論文Fig.7のプロトタイプカーネルとその回転で30種を構成する方針（03_詳細設計書 7.2）に従い、
SRM (Fridrich & Kodovsky, 2012) の残差フィルタから次の内訳で構成した。

| 種類 | 枚数 | 正規化係数 q |
|---|---|---|
| 1次微分 8方向 | 8 | 1 |
| 2次微分 4方向 | 4 | 2 |
| 3次微分 8方向 | 8 | 3 |
| EDGE3x3 4回転 | 4 | 4 |
| SQUARE3x3 | 1 | 4 |
| EDGE5x5 4回転 | 4 | 12 |
| SQUARE5x5 | 1 | 12 |
| **合計** | **30** | |

全カーネルの係数和が0（＝平坦画像への応答が0）であることをテストで確認している。
SRMの残差量子化 `trunc(round(K*I/q), T)` に倣い、q で割った後に丸めてから [-t, t] に
クリップするため、残差は {-7, ..., 7} の15値をとる離散量になる
（`preprocess.quantize_residual: false` で丸めを外せる）。

### 条件付きモデルの予測対象 y'（設計書の記述の食い違い）

03_詳細設計書の中で y' のチャンネル数の扱いが分かれている。

- §1.2: `y' ← Truncate(Stack([h_m * y for m in 1..M]))` → 30種×2ch = **60ch**
- §3.3 / §3.4: μ, s は各 `[K,2,H,W]`、`NLLLoss` の入力 y' は `[2,H,W]` → **2ch**

§1.2 のとおり60chにすると混合分布パラメータは 3×K×60 = 1800ch/モデルとなり、
すでに確定した案A（分類器入力120ch）と両立しない。OI-2 で §3.3 を採ったのと同じ理由で、
ここでも §3.3 / §3.4 を優先し **既定は2ch** とする。

| `model.conditional_unet.target_mode` | y' | 備考 |
|---|---|---|
| `single_filter`（既定） | 2ch | `target_filter`（既定 `square5x5`）1種類のみ y に適用 |
| `filter_bank` | 60ch | §1.2の記述どおり。`classifier.feature_source: bottleneck` が必須（config検証で強制） |

x'（条件付けの入力）は **どちらのモードでも30種すべてを適用した30ch**である。

### 混合分布の尤度

残差が離散値であることに合わせ、PixelCNN++ と同じ**離散化ロジスティック混合**で尤度を計算する。
幅1のビンに対する確率 `σ((v+0.5-μ)/s) - σ((v-0.5-μ)/s)` を用い、両端 ±t のビンは裾全体を
確率質量とする。{-7,...,7} にわたる確率の総和が1になることをテストで確認している。

`NLLLoss` の `reduction` は既定 `mean`（要素平均）。03_詳細設計書 3.4 は「全画素で総和」と
記載しているが、損失スケールが約8000倍になり学習率1e-4と釣り合わないため既定を平均とした
（`sum` / `sum_per_sample` も選べる。勾配の向きは同じ）。

### 分類器のトークン化（03_詳細設計書 7.4 の確定）

ResNetブロック4段で 64×64 → 4×4 まで空間解像度を落とし、残った 4×4 = 16 個の空間位置を
それぞれ1トークン（次元256）として扱う。学習可能な位置埋め込みを加えて2層の
Transformer Encoder に通し、出力を平均プーリングしてから全結合層でロジットを出す。
シグモイドは `BCEWithLogits` 側に含める。

パラメータ数は pθ / qφ が各約1.9M、gψ が約4.1M。

## 学習

```bash
python -m src.cli train-stage1 --target photo --config configs/stage1_photo.yaml   # pθ
python -m src.cli train-stage1 --target ai    --config configs/stage1_ai.yaml      # qφ
python -m src.cli train-stage2 --config configs/stage2_classifier.yaml             # gψ

# スモークテスト（1エポック・2ステップだけ流す）
python -m src.cli train-stage1 --target photo --config configs/stage1_photo.yaml --epochs 1 --max-steps 2
# 中断からの再開
python -m src.cli train-stage1 --target photo --config configs/stage1_photo.yaml --resume checkpoints/stage1_photo/best.pt
```

Stage I-A（pθ）と Stage I-B（qφ）は独立しているので並列に流せる。両方が終わってから
そのチェックポイントを `configs/stage2_classifier.yaml` の `stage2.photo_model_checkpoint` /
`ai_model_checkpoint` に設定して Stage II を実行する。

### データの流れ

`MetadataRepository` → `select_image_rows` → `PatchDataset` → `DataLoader` → 学習ループ。

- 学習に使う画像は `dataset.train_generators`（論文プロトコルではSDv1.4のみ）と、
  それに対応する実写generator（`ImageNet(real)@SDv1.4`）から選ぶ。
  `dataset.excluded_generators` のものは自動的に外れる
- DataLoaderのワーカーは**デコードと64×64クロップまで**を担当し、CFAマスクと
  ハイパスフィルタはバッチ単位でデバイス上（MPS）で実行する。畳み込みをGPUに任せるため
- ZIP内画像はワーカーごとにハンドルをキャッシュして読む（`open_image_bytes_cached`）。
  `ZipFile` はスレッド安全ではないため、ワーカー内は単一スレッドで読む
- 読めない画像に当たった場合は警告を出して次の画像にフォールバックする（最大5枚まで）

### 記録されるもの（NFR-7）

| テーブル | 内容 |
|---|---|
| `experiments` | 実験名・stage・config全文（JSON）・gitコミット（dirty判定つき）・seed |
| `training_runs` | 実行の開始/終了時刻と status（running / completed / failed） |
| `training_logs` | エポックごとの損失・精度・学習率・所要秒数 |
| `model_checkpoints` | 保存したチェックポイントのパスと選択指標 |

例外で落ちた場合も `training_runs.status` は `failed` に更新される。

チェックポイントは `run<run_id>_epoch<NNN>.pt` として毎エポック保存し、検証指標が最良の
ものを `best.pt` にも保存する（Stage I は `val_nll` 最小、Stage II は `val_accuracy` 最大）。
モデル重み・オプティマイザ状態・エポック・config・run_idを含むので、`--resume` で
そのまま学習を再開できる（03_詳細設計書 6節「学習中断」）。

### Ablation の切り替え

| アブレーション | 設定 |
|---|---|
| A: 条件付きモデル構成 | `stage2.use_photo_model` / `use_ai_model` |
| B: ハイパスフィルタ・CFAマスク | `preprocess.use_high_pass` / `use_cfa_mask` |
| C: truncation閾値 | `preprocess.truncation_t`（Stage Iから再学習が必要） |
| D: fine-tuning戦略 | `stage2.freeze_conditional_models` |

D を `false` にすると pθ / qφ にも勾配が流れ、チェックポイントにも両モデルが保存される。

## 評価とレポート

```bash
# Table 1相当（--run-id 省略時は最新の完了済みStage II実行を使う）
python -m src.cli evaluate --mode cross_generator --config configs/stage2_classifier.yaml --run-id 3

# Figure 6相当（JPEG圧縮・ダウンサンプリング）
python -m src.cli evaluate --mode robustness --config configs/stage2_classifier.yaml --run-id 3

# Table 4相当（variantごとに学習 → 評価）
python -m src.cli evaluate --mode ablation --config configs/ablation/ablation_a_dual_model.yaml

# レポート出力
python -m src.cli report --run-id 3 --out reports/
```

### 推論（Algorithm 2）

1画像から `inference.num_patches`（既定16）枚のパッチを抽出し、各パッチのスコアを平均して
閾値τで判定する。パッチ位置はファイルパスをキーにした決定的な乱数で決まるので、
同じ画像は何度評価しても同じ位置が使われる（NFR-1）。

### 閾値τのチューニング

論文にτの具体値がないため、**val split上でYouden指数（TPR - FPR）が最大になる点**を選ぶ
（03_詳細設計書 7.3）。test split には触れないので、評価用データで閾値を決めてしまうことはない。
`evaluate.tune_threshold: false` にすると `inference.threshold` の値をそのまま使う。

判定は Algorithm 2 に従い `score > τ` で行う。scikit-learn の `roc_curve` が返す閾値は
`score >= τ` を陽性とする規約なので、境界のサンプルが逆に判定されないよう、返す閾値を
ひとつ下のスコアとの中点まで下げている。

### 評価対象の組み立て

生成器 G の評価には、G のAI画像と `ImageNet(real)@G` の実写画像を使う。
`evaluate.max_images_per_generator` は**ラベルごとに**適用するので、間引いても
AI画像と実写画像が同数ずつ残る（まとめてLIMITすると片方のクラスだけになり、
AccuracyもAUCも意味を持たなくなる）。`dataset.excluded_generators` のものは自動的に外れる。

### 出力

`reports/run<run_id>/` に以下を出力する。

| ファイル | 内容 |
|---|---|
| `report.md` | 実験メタ情報＋全表＋グラフを埋め込んだMarkdown |
| `cross_generator.csv` | 生成器別のAccuracy/AUC/AP（平均行つき） |
| `ablation.csv` | アブレーション条件ごとの平均Accuracy |
| `robustness.csv` | 劣化種別・強度ごとのAccuracy |
| `robustness_jpeg.png` / `robustness_downsample.png` | 劣化強度に対する精度推移の折れ線グラフ |

同じ条件を再評価した場合、`evaluation_results` には履歴として残るが、レポートには
最新の1件だけを載せる。グラフは日本語フォントに依存しないよう、図中のラベルは英語で描画する。

## 設計上の決定事項（Open Issues への回答）

### OI-1: Midjourneyデータ未取得

Midjourney生成画像は現時点で未取得のため、**当面すべての学習・評価対象から除外する**。
ただしコード・DBスキーマは8生成器を前提のまま構築し、除外は
`configs/base.yaml` の `dataset.excluded_generators` で制御する。
`generators` テーブルには `notes` 付きでレコードのみ登録しておき、データ取得後は
`excluded_generators` を空にするだけで8生成器のTable 1再現に復帰できる。
当面の cross-generator 評価は **7生成器平均**として報告する。

### OI-2: 分類器 gψ への入力特徴（03_詳細設計書 1.3節）

**案A（U-Net最終出力の混合分布パラメータをそのまま特徴とする）** を採用する。

チャンネル数については設計書内に不整合があるため、以下のとおり解釈を確定した。

- 03_詳細設計書 §1.3 案A: 「3K = 30ch、2モデル連結で60ch」
- 03_詳細設計書 §3.3 `forward` 仕様: 「w, μ, s 各 `Tensor[K,2,64,64]`」

y′ は2チャンネル（CFAで隠された残り2色）であり、各チャンネルごとに混合ロジスティック分布の
パラメータを持つため、§3.3 の定義が論文の定式化と整合する。したがって

> **fθ / fφ の出力特徴は 3 × K × 2 = 60ch（K=10）、pθ・qφ 連結で 120ch**

とする。§1.3 の「30ch/60ch」は μ, s の2ch分を数え落とした記述として扱う。

案B（デコーダ最終層手前のボトルネック特徴）は `configs/base.yaml` の
`model.classifier.feature_source: mixture_params | bottleneck` で切替可能にしておき、
必要になった時点でアブレーション比較を行う。

### 残りの要確認事項

| ID | 状態 |
|---|---|
| OI-3 外部SSDマウントパス・ZIP展開先 | `paths.dataset_root` に外出し済み（既定 `/Volumes/Extreme Pro`）。ZIPは展開せず直読みするため展開先は不要（`paths.extract_root` は未使用） |
| OI-4 / OI-5 GPU・チップ種別・SSDフォーマット | デバイス抽象化で吸収済み。実機がWindowsのため、GPUの有無と `dataset_root` のドライブレターは要確認 |
| 03-7.2 SRM 30種カーネル係数 | 実装済み（上表の内訳）。論文Fig.7との照合は論文入手時に実施 |
| 03-7.3 判定閾値 τ | 既定0.5。val上でYouden指数最大化によりチューニング |
| 03-7.4 Transformerのトークン化 | 確定済み（4×4=16トークン、次元256、平均プーリング）|
| 03-1.2 vs 3.3 y'のチャンネル数 | 2ch（`target_mode: single_filter`）を既定として確定。60chも設定で選べる |

## 実装進捗

- [x] Step 1: プロジェクト雛形・仮想環境・requirements.txt
- [x] Step 2: メタデータDB初期化（04_DB設計書 4節のDDL）
- [x] Step 3: データ取り込みバッチ（`ingest`）
- [x] Step 4: CFAマスク → ハイパスフィルタ → 条件付きモデル(pθ/qφ) → 分類器gψ
- [x] Step 5: 学習ループ（Stage I-A/I-B, Stage II）とDataLoader
- [x] Step 6: 評価（cross-generator / ablation / robustness）とレポート出力
