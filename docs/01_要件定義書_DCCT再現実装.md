# 要件定義書：DCCT（Color Matters）再現実装

**担当ロール:** requirements-writer
**対象論文:** Zhong, Xu, Zou. *Color Matters: Demosaicing-Guided Color Correlation Training for Generalizable AI-Generated Image Detection*. arXiv:2601.22778, 2026.
**位置づけ:** 卒業研究「3枝アーキテクチャによるAI生成画像検出モデルの構築と評価」における第3枝候補（DCCT）の元となる、DCCT単体のスタンドアロン・フル再現実装
**作成日:** 2026年9月
**ステータス:** 草稿

---

## 1. 文書の目的

本書は、DCCT論文の手法をスタンドアロンでフル再現し、論文相当の精度（GenImageベンチマーク平均Acc ≈ 97%前後）を目指す実装プロジェクトの要件を定義する。後続の基本設計書・詳細設計書・DB設計書・図表集はすべて本書の要件をもとに作成する。

## 2. 背景

- 既存のAIGI（AI-Generated Image）検出手法の多くは、特定の生成器が残す人工的なアーティファクトを学習するため、未知の生成器への汎化性能が低い。
- DCCTは、カメラの撮像パイプライン（カラーフィルタアレイ CFA によるBayerサンプリングとデモザイキング）が生む色チャンネル間の物理的相関に着目し、生成器に依存しない普遍的な特徴で検出を行う。
- 本卒業研究では、DCCTを3枝アーキテクチャの第3枝候補として組み込む前段階として、まずDCCT単体を論文に忠実に再現し、論文の報告値との整合性を検証する必要がある。

## 3. 研究目的・ゴール

1. DCCTのフル学習パイプライン（条件付き分布モデル pθ / qφ の学習 → 二値分類器 gψ の学習）を実装する。
2. GenImageデータセットを用い、論文のプロトコル（SDv1.4で学習し、Midjourney / SDv1.4 / SDv1.5 / ADM / GLIDE / Wukong / VQDM / BigGANでcross-generator評価）を再現する。
3. 論文の主要アブレーション（Table 4a〜4d相当）を再現し、設計上の各要素（CFAマスク、ハイパスフィルタ、truncation閾値t、fine-tuning有無）の寄与を定量的に確認する。
4. ロバスト性評価（JPEG圧縮・ダウンサンプリング、Figure 6相当）を再現する。
5. 実験・データ・結果を再現可能な形でメタデータDBに記録し、後続の3枝統合研究に引き継げる状態にする。

## 4. 用語定義

| 用語 | 説明 |
|---|---|
| CFA | Color Filter Array。カメラセンサー上のBayer配列フィルタ |
| デモザイキング | CFAで欠落した色チャンネルを周囲画素から補間する処理 |
| pθ | 写真画像で学習する条件付き分布モデル（U-Net） |
| qφ | AI生成画像で学習する条件付き分布モデル（U-Net） |
| gψ | pθ・qφの特徴を入力とする二値分類器（ResNet＋Transformer） |
| NLL | Negative Log-Likelihood。条件付き分布モデルの学習損失 |
| 高域通過フィルタ (High-pass filter) | SRM由来の30種5×5カーネル。高周波成分のみを抽出 |
| truncation threshold *t* | 高域通過フィルタ適用後の残差値を[-t, t]にクリップする閾値（既定 t=7） |
| GenImage | 8種の生成器（Midjourney, SDv1.4, SDv1.5, ADM, GLIDE, Wukong, VQDM, BigGAN）を含む評価用データセット |
| DRCT-2M | SD系列＋DRバリアントを含む拡張評価用データセット（本フェーズではスコープ外） |

## 5. 前提条件・制約条件

- **データセット格納場所:** 外部SSD「Extreme Pro」（マウントポイントは環境依存、設定ファイルでパスを外出しする）に、GenImage各生成器のフォルダ／ZIPが格納済み。
  - 確認済みフォルダ例: `BigGAN/`, `genimage-adm.zip`, `genimage-stable-diffusion-v1-4.zip`, `genimage-wukong.zip`, `imagenet_glide/`, `stable_diffusion_v_1_5/`, `VQDM/`
  - **未確認事項:** Midjourney生成画像フォルダの所在が画像上で確認できていない。実装着手前に配置場所を確認する（5.1節参照）。
  - ZIPファイルは展開が必要（前処理パイプラインに展開ステップを含む）。
- **実行環境:** macOS Tahoe 26.1（Apple Silicon搭載Mac）のローカルワークステーション＋外部SSD「Extreme Pro」。GPUはNVIDIA CUDAではなく、Apple SiliconのGPU（Metal / PyTorchの`mps`バックエンド）を利用する前提とする。VRAM相当のUnified Memory容量は要確認、バッチサイズ16は論文既定値だが実機のメモリ状況に応じて調整する。
- **フレームワーク:** PyTorch（卒業研究仕様書の方針を踏襲）。macOS環境ではCUDAが利用できないため、デバイス選択は`torch.backends.mps.is_available()`を用いて`mps`／利用不可時は`cpu`にフォールバックする。一部の演算がMPSバックエンドで未対応の場合、環境変数`PYTORCH_ENABLE_MPS_FALLBACK=1`でCPU実行に自動フォールバックさせる。
- **メタデータ管理:** SQLiteによる軽量DB。学習データ本体（画像ファイル）はDB化せず、パス参照のみ管理する。
- **再現性:** 論文には公開実装が存在しないため、Algorithm 1 / Algorithm 2（論文Appendix A）および実装詳細節の記述から独自に実装する。厳密な再現ができない箇所は「7.仮定・未確定事項」に明記し、設計書側にも引き継ぐ。

### 5.1 着手前に確認・解消すべき事項（Open Issues）

| ID | 内容 | 影響範囲 |
|---|---|---|
| OI-1 | Midjourney生成画像データの所在確認・追加取得 | GenImage 8種全体でのTable1再現に必須 |
| OI-2 | 分類器 gψ への入力特徴の正確な定義（後述7.2） | 詳細設計・実装全体 |
| OI-3 | 外部SSDのマウントパス・ZIP展開先の確定 | データ取り込みバッチ、DB設計 |
| OI-4 | 学習に使えるGPU/VRAM容量 | バッチサイズ・学習時間見積り |
| OI-5 | 使用MacのApple Siliconチップ種別（M1〜M4等）とUnified Memory容量、外部SSDのフォーマット（APFS/exFAT）の確認 | MPSでの学習時間見積り、フルディスクアクセス権限設定 |

---

## 6. スコープ

### 6.1 対象（In Scope）

- GenImageデータセットを用いたDCCTのフル学習・評価パイプライン再現
- Stage I-A（pθ学習）、Stage I-B（qφ学習）、Stage II（分類器gψ学習）の実装
- Cross-generator評価（Table 1相当）
- アブレーション実験（Table 4a〜4d相当：条件付きモデル構成／モデル構造／truncation閾値／fine-tuning戦略）
- ロバスト性評価（JPEG圧縮・ダウンサンプリング、Figure 6相当）
- 実験・データ・結果のメタデータDB管理
- 学習/評価/アブレーションの結果をまとめるレポート出力

### 6.2 対象外（Out of Scope）

- DRCT-2Mデータセットでの評価（Table 2相当）※将来拡張として設計上は考慮するが本フェーズでは実装しない
- DCCT†（One-classバリアント、Appendix B）
- 3枝アーキテクチャへの統合（本書はDCCT単体の再現のみを対象。統合は卒業研究仕様書のPhase 2で別途実施）
- 学習済みモデルの配布・サービング（Web API化等）

---

## 7. 機能要件

| ID | 機能 | 内容 |
|---|---|---|
| FR-1 | データ取り込み・管理 | 外部SSD上のGenImage各生成器フォルダ／ZIPを走査し、画像パス・ラベル（photo/AI）・生成器名・split（train/val/test）をメタデータDBに登録する |
| FR-2 | 前処理パイプライン | Bayer CFAマスク適用、64×64ランダムクロップ、30種ハイパスフィルタ適用、truncation（既定t=7）を行う |
| FR-3 | 条件付き分布モデル学習（Stage I-A/I-B） | U-Net構造でpθ（写真）・qφ（AI画像）を、K=10のロジスティック混合分布に対するNLL損失で個別に学習する |
| FR-4 | 二値分類器学習（Stage II） | pθ・qφを固定（Frozen）し、両モデルの特徴を結合した入力でResNet＋Transformerの分類器gψをBCE損失で学習する |
| FR-5 | 推論・評価 | 1画像あたり16パッチを抽出してスコアを平均し、閾値τで判定。GenImage各生成器に対するAccuracy/AUC/APを算出する |
| FR-6 | アブレーション実験 | (a) pθ/qφ単体 vs 両方、(b) ハイパスフィルタ有無×CFAマスク有無、(c) truncation閾値 t∈{3,7,15}、(d) 分類器学習時のfull fine-tuning有無、を切り替え可能な設定で実行できる |
| FR-7 | ロバスト性評価 | JPEG圧縮（QF可変）、空間ダウンサンプリング（倍率可変）を適用した画像に対する精度を測定する |
| FR-8 | 実験管理・再現性確保 | 実験設定（config）、乱数シード、使用データ範囲、モデルチェックポイント、学習ログ、評価結果をDBおよびファイルに記録する |
| FR-9 | 結果レポート出力 | Table 1／Table 4相当の集計結果を表形式（CSV/Markdown）およびグラフで出力する |
| FR-10 | データ拡張（学習時） | 学習時にJPEG圧縮（QF〜U(70,100)、適用確率5%）をベナイン摂動として付与する |

## 8. 非機能要件

| ID | 分類 | 内容 |
|---|---|---|
| NFR-1 | 再現性 | 同一config・同一seedで学習した場合、評価指標のばらつきが実務上許容できる範囲（例：Acc ±1pt程度）に収まること |
| NFR-2 | 性能 | 1エポックの学習時間・推論時間を計測可能にし、GPU 1枚での現実的な学習時間内（要見積り）で完走できること |
| NFR-3 | 拡張性 | 将来的にDRCT-2M等の追加データセット、DCCT†（one-class）バリアントを追加しやすいモジュール構成とすること |
| NFR-4 | 保守性 | 前処理・モデル・学習ループ・評価・DBアクセスを明確にモジュール分割し、単体テスト可能な単位で実装すること |
| NFR-5 | 可搬性 | 外部SSDのマウントパス等、環境依存情報は設定ファイル（YAML等）に外出しし、コード変更なしに環境切替できること |
| NFR-6 | データ管理 | 学習データ本体は複製せず外部SSD上のパス参照とし、メタデータのみSQLiteで一元管理すること |
| NFR-7 | 監査性・トレーサビリティ | どのconfig・どのデータ範囲・どのコードバージョンで得た結果かを後から追跡できること |

## 9. 受け入れ基準

- GenImage（SDv1.4学習）でのcross-generator評価において、8生成器平均Accuracyが論文報告値（約97%）に対し実務上妥当な範囲で近似していること（厳密一致は求めないが、傾向：Midjourney/ADM/BigGANでも90%超を目安とする）。
- Table 4相当のアブレーションで、論文と同方向の傾向（例：CFAマスク＋ハイパスフィルタ併用が最良、t=7が最良、fine-tuningなしが最良）が再現されること。
- ロバスト性評価で、JPEG圧縮・ダウンサンプリングに対する精度劣化が緩やかであることが確認できること。
- 全実験がDB上の実験IDと紐づけて再実行・追跡可能であること。

## 10. リスク・未確定事項（要検討）

| ID | 内容 | 対応方針（案） |
|---|---|---|
| R-1 | 論文に公式実装が公開されていないため、細部（分類器への入力特徴の定義、Transformer部分のトークン化方法等）は独自解釈が必要 | 詳細設計書で仮定を明記し、必要に応じて複数案を実装・比較する |
| R-2 | GenImageのMidjourneyデータが手元にない可能性 | 追加取得 or 当該生成器を評価対象から一時除外し、後日追加する設計とする |
| R-3 | 30種ハイパスフィルタの正確な係数はFig.7（論文Appendix C）に基づき手動実装が必要 | SRM（Fridrich & Kodovsky, 2012）の公開係数を一次参照として実装し、論文Fig.7との整合を確認する |
| R-4 | 計算資源（Apple Silicon GPU/Unified Memory/学習時間）が卒業研究のスケジュールに収まるか未確定。MPSバックエンドはCUDA GPUと性能特性が異なり、一部演算はCPUフォールバックで速度低下する可能性がある | 少量データでのスモークテストをMPS実機で実施し学習時間を実測、必要なら生成器・枚数を間引く。CPUフォールバックが多発する演算は代替実装を検討する |

## 11. 参考文献

1. Zhong, N., Xu, Y., Zou, M. *Color Matters: Demosaicing-Guided Color Correlation Training for Generalizable AI-Generated Image Detection*. arXiv:2601.22778, 2026.
2. Zhu, M. et al. *GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image*. NeurIPS 2023.
3. Fridrich, J., Kodovsky, J. *Rich Models for Steganalysis of Digital Images*. IEEE TIFS, 2012.
4. Ronneberger, O. et al. *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI 2015.
5. Salimans, T. et al. *PixelCNN++*. ICLR 2017.
6. 卒業研究仕様書「3枝アーキテクチャによるAI生成画像検出モデルの構築と評価」（プロジェクト内資料）
7. DCCT論文学習ノート（プロジェクト内資料）
