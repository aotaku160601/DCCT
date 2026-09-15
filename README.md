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

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Linux/コンテナ上でCPUのみ動かす場合は、CPU版wheelを先に入れてから残りを入れる:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements.txt
```

### 実行デバイス

`configs/base.yaml` の `device.type` は既定 `auto` で、**cuda → mps → cpu** の順に自動選択する。
`torch.cuda.*` を直接呼ぶコードは書かない（02_基本設計書 9節）。macOS実行時は
`device.enable_mps_fallback: true` により `PYTORCH_ENABLE_MPS_FALLBACK=1` を設定し、
MPS未対応演算をCPUへ自動フォールバックさせる。

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
| OI-3 外部SSDマウントパス・ZIP展開先 | `configs/base.yaml` の `paths.dataset_root` / `paths.extract_root` に外出し済み。実パスは ingest 実行時に確定 |
| OI-4 / OI-5 GPU・チップ種別・SSDフォーマット | デバイス抽象化で吸収。学習時間はスモークテストで実測する |
| 03-7.2 SRM 30種カーネル係数 | SRM (Fridrich & Kodovsky, 2012) の公開係数を一次参照として実装予定 |
| 03-7.3 判定閾値 τ | 既定0.5。val上でYouden指数最大化によりチューニング |
| 03-7.4 Transformerのトークン化 | ResNet出力の空間特徴マップをパッチ分割してトークン化。実装確定時に本READMEへ追記 |

## 実装進捗

- [x] Step 1: プロジェクト雛形・仮想環境・requirements.txt
- [ ] Step 2: メタデータDB初期化（04_DB設計書 4節のDDL）
- [ ] Step 3: データ取り込みバッチ（`ingest`）
- [ ] Step 4: CFAマスク → ハイパスフィルタ → 条件付きモデル(pθ/qφ) → 分類器gψ
