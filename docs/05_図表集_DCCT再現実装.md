# 図表集：DCCT（Color Matters）再現実装

**担当ロール:** diagram-writer
**表記法:** Mermaid記法
**参照元:** 01_要件定義書, 02_基本設計書, 03_詳細設計書, 04_DB設計書
**作成日:** 2026年9月

本書は、他4文書に登場する図をシーケンス図・フローチャート中心に集約し、実装時に参照しやすい形でまとめたものである。

---

## 1. システム構成図

```mermaid
flowchart TB
    subgraph EXT["外部ストレージ：Extreme Pro (外部SSD)"]
        GI["GenImage 各生成器フォルダ/ZIP"]
    end

    subgraph LOCAL["ローカルワークステーション"]
        INGEST["データ取り込みバッチ"]
        DB[("メタデータDB SQLite")]
        PREP["前処理<br/>CFAマスク→ハイパスフィルタ→パッチ抽出"]
        UNET_P["条件付きモデル pθ"]
        UNET_Q["条件付きモデル qφ"]
        CLS["二値分類器 gψ"]
        EVAL["評価（cross-generator/ablation/robustness）"]
        REPORT["レポート出力"]
    end

    GI --> INGEST --> DB --> PREP
    PREP --> UNET_P
    PREP --> UNET_Q
    UNET_P -->|Frozen特徴| CLS
    UNET_Q -->|Frozen特徴| CLS
    CLS --> EVAL --> DB
    DB --> REPORT
```

---

## 2. シーケンス図：Stage I-A / I-B（条件付き分布モデル学習）

```mermaid
sequenceDiagram
    participant CLI as CLI(train-stage1)
    participant Cfg as ExperimentConfig
    participant Repo as MetadataRepository/DB
    participant Loader as DataLoader
    participant CFA as CFAMask
    participant HPF as HighPassFilterBank
    participant Net as ConditionalUNet(fθ or fφ)
    participant Loss as NLLLoss
    participant Opt as Optimizer(Adam)

    CLI->>Cfg: load(config.yaml)
    CLI->>Repo: register_experiment(config_snapshot)
    Repo-->>CLI: experiment_id
    CLI->>Repo: create training_run
    Repo-->>CLI: run_id

    loop 各epoch
        loop 各mini-batch
            Loader->>Repo: 対象画像パス取得(split=train, label=photo|ai)
            Repo-->>Loader: image paths
            Loader->>Loader: RandomCrop(64x64)
            Loader->>CFA: apply(image)
            CFA-->>Loader: x, y
            Loader->>HPF: apply(x) / apply(y)
            HPF->>HPF: truncate(-t, t)
            HPF-->>Net: x'
            HPF-->>Loss: y'
            Net->>Net: forward(x') -> {w,mu,s}
            Net-->>Loss: MixtureParams
            Loss->>Loss: NLL計算
            Loss-->>Opt: loss
            Opt->>Net: パラメータ更新
        end
        CLI->>Repo: log_training_metric(run_id, epoch, nll_loss)
        CLI->>Repo: save_checkpoint_meta(run_id, epoch, path)
    end

    CLI->>Repo: update training_run status=completed
    Note over Net: 学習収束後、パラメータをFrozenにして<br/>Stage IIへ引き渡す
```

---

## 3. シーケンス図：Stage II（二値分類器学習）

```mermaid
sequenceDiagram
    participant CLI as CLI(train-stage2)
    participant Repo as MetadataRepository/DB
    participant Loader as DataLoader
    participant CFA as CFAMask
    participant HPF as HighPassFilterBank
    participant Pth as pθ (Frozen)
    participant Phi as qφ (Frozen)
    participant Cls as BinaryClassifier gψ
    participant Loss as BCELoss
    participant Opt as Optimizer(Adam)

    CLI->>Repo: register_experiment / create training_run
    Repo-->>CLI: run_id

    loop 各epoch
        loop 各mini-batch (image, label c)
            Loader->>CFA: RandomCrop + apply(image)
            CFA-->>HPF: x
            HPF-->>Pth: x'
            HPF-->>Phi: x'
            Pth-->>Cls: fθ(x')
            Phi-->>Cls: fφ(x')
            Cls->>Cls: forward(Concat(fθ(x'), fφ(x')))
            Cls-->>Loss: c_hat, c
            Loss-->>Opt: BCE loss
            Opt->>Cls: パラメータ更新（pθ,qφは更新しない）
        end
        CLI->>Repo: log_training_metric(run_id, epoch, bce_loss)
        CLI->>Repo: save_checkpoint_meta(run_id, epoch, path)
    end
    CLI->>Repo: update training_run status=completed
```

---

## 4. シーケンス図：推論・評価フェーズ

```mermaid
sequenceDiagram
    participant CLI as CLI(evaluate)
    participant Repo as MetadataRepository/DB
    participant Loader as DataLoader(test images)
    participant CFA as CFAMask
    participant HPF as HighPassFilterBank
    participant Pth as pθ (Frozen)
    participant Phi as qφ (Frozen)
    participant Cls as BinaryClassifier gψ
    participant Agg as ScoreAggregator

    CLI->>Repo: 評価対象画像・生成器一覧取得
    loop 各テスト画像
        loop p = 1..16パッチ
            Loader->>CFA: RandomCrop + apply
            CFA-->>HPF: x_p
            HPF-->>Pth: x'_p
            HPF-->>Phi: x'_p
            Pth-->>Cls: fθ(x'_p)
            Phi-->>Cls: fφ(x'_p)
            Cls-->>Agg: s_p
        end
        Agg->>Agg: s_bar = mean(s_1..s_16)
        Agg->>Agg: c_hat = 1[s_bar > τ]
    end
    CLI->>Repo: save_evaluation_result(run_id, generator_id, accuracy, auc, ap)
    CLI->>Repo: レポート集計用データ取得
```

---

## 5. 全体処理フローチャート

```mermaid
flowchart TD
    S([開始]) --> A[外部SSDからGenImageデータ取り込み<br/>ingest]
    A --> B{データ検証OK?}
    B -->|NG: 破損/欠落| B2[status=invalidとして除外<br/>OI-1等をログ記録]
    B2 --> C
    B -->|OK| C[Stage I-A: pθ学習]
    A --> D[Stage I-B: qφ学習]
    C --> E{両モデル学習完了?}
    D --> E
    E -->|Yes| F[両モデルをFrozen化]
    F --> G[Stage II: 分類器gψ学習]
    G --> H[Cross-generator評価]
    G --> I[アブレーション実験 A/B/C/D]
    G --> J[ロバスト性評価<br/>JPEG/ダウンサンプリング]
    H --> K[結果をDBに保存]
    I --> K
    J --> K
    K --> L[レポート生成<br/>CSV/Markdown/グラフ]
    L --> M([終了])
```

---

## 6. クラス図

```mermaid
classDiagram
    class CFAMask
    class HighPassFilterBank
    class PatchSampler
    class ConditionalUNet
    class MixtureParams
    class NLLLoss
    class BinaryClassifier
    class Trainer
    class TrainerStage1
    class TrainerStage2
    class Evaluator
    class ExperimentConfig
    class MetadataRepository

    Trainer <|-- TrainerStage1
    Trainer <|-- TrainerStage2
    TrainerStage1 --> ConditionalUNet
    TrainerStage1 --> NLLLoss
    TrainerStage2 --> ConditionalUNet
    TrainerStage2 --> BinaryClassifier
    ConditionalUNet --> MixtureParams
    Evaluator --> ConditionalUNet
    Evaluator --> BinaryClassifier
    Evaluator --> MetadataRepository
    TrainerStage1 --> MetadataRepository
    TrainerStage2 --> MetadataRepository
    CFAMask --> PatchSampler
    HighPassFilterBank --> ConditionalUNet
```

*(詳細なメソッドシグネチャは 03_詳細設計書 の2節を参照)*

---

## 7. ER図（DB設計書 再掲・参照用）

```mermaid
erDiagram
    DATASETS ||--o{ GENERATORS : contains
    DATASETS ||--o{ IMAGES : contains
    GENERATORS ||--o{ IMAGES : produces
    EXPERIMENTS ||--o{ TRAINING_RUNS : has
    TRAINING_RUNS ||--o{ TRAINING_LOGS : records
    TRAINING_RUNS ||--o{ MODEL_CHECKPOINTS : saves
    TRAINING_RUNS ||--o{ EVALUATION_RESULTS : produces
    GENERATORS ||--o{ EVALUATION_RESULTS : evaluated_on
    EXPERIMENTS ||--o{ ABLATION_CONFIGS : defines

    DATASETS {
        int dataset_id PK
        string name
        string root_path
    }
    GENERATORS {
        int generator_id PK
        int dataset_id FK
        string name
        string category
        boolean is_real
    }
    IMAGES {
        int image_id PK
        int dataset_id FK
        int generator_id FK
        string filepath
        string label
        string split
        string status
    }
    EXPERIMENTS {
        int experiment_id PK
        string name
        string stage
        text config_snapshot
    }
    TRAINING_RUNS {
        int run_id PK
        int experiment_id FK
        string status
    }
    EVALUATION_RESULTS {
        int result_id PK
        int run_id FK
        int generator_id FK
        string eval_type
        real accuracy
    }
```

*(完全なカラム定義・DDLは 04_DB設計書 を参照)*

---

## 8. アブレーション実験の分岐フローチャート（Table 4対応）

```mermaid
flowchart TD
    Start([Stage II 学習済モデル準備完了]) --> Choose{アブレーション種別}

    Choose -->|A: 条件付きモデル構成| A1[pθのみ使用]
    Choose -->|A| A2[qφのみ使用]
    Choose -->|A| A3[pθ+qφ 両方使用（既定）]

    Choose -->|B: モデル構造| B1[ハイパスフィルタなし]
    Choose -->|B| B2[CFAマスクなし（ランダムRGB選択）]
    Choose -->|B| B3[両方あり（既定）]

    Choose -->|C: truncation閾値| C1[t=3]
    Choose -->|C| C2[t=7（既定）]
    Choose -->|C| C3[t=15]

    Choose -->|D: fine-tuning戦略| D1[pθ,qφをFrozenのまま分類器のみ学習（既定）]
    Choose -->|D| D2[pθ,qφも含めFull fine-tuning]

    A1 --> Eval[評価: GenImage mAcc算出]
    A2 --> Eval
    A3 --> Eval
    B1 --> Eval
    B2 --> Eval
    B3 --> Eval
    C1 --> Eval
    C2 --> Eval
    C3 --> Eval
    D1 --> Eval
    D2 --> Eval

    Eval --> Save[(evaluation_results<br/>eval_type=ablation)]
    Save --> Report[レポート: Table4a〜4d相当を再構成]
```
