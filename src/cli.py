"""DCCT再現実装のコマンドラインエントリポイント。

02_基本設計書 6節「実行フロー（CLIコマンド遷移）」に対応する。

    python -m src.cli init-db       --config configs/base.yaml
    python -m src.cli ingest        --config configs/base.yaml
    python -m src.cli train-stage1  --target photo --config configs/stage1_photo.yaml
    python -m src.cli train-stage1  --target ai    --config configs/stage1_ai.yaml
    python -m src.cli train-stage2  --config configs/stage2_classifier.yaml
    python -m src.cli evaluate      --mode cross_generator --experiment-id <ID>
    python -m src.cli report        --experiment-id <ID> --out reports/
"""

from __future__ import annotations

import argparse
import logging
import sys

from .data import db as db_module
from .experiment.config import Config
from .experiment.paths import resolve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dcct", description="DCCT (Color Matters) 再現実装")
    sub = parser.add_subparsers(dest="command", required=True)

    p_initdb = sub.add_parser("init-db", help="メタデータDBを作成する（04_DB設計書 4節のDDLを適用）")
    p_initdb.add_argument("--config", default="configs/base.yaml")
    p_initdb.add_argument("--force", action="store_true", help="既存DBを削除して作り直す")

    p_ingest = sub.add_parser("ingest", help="外部SSD上のGenImageを走査しDBへ登録する")
    p_ingest.add_argument("--config", default="configs/base.yaml")

    p_s1 = sub.add_parser("train-stage1", help="条件付き分布モデル pθ / qφ を学習する")
    p_s1.add_argument("--target", choices=["photo", "ai"], required=True)
    p_s1.add_argument("--config", required=True)

    p_s2 = sub.add_parser("train-stage2", help="二値分類器 gψ を学習する")
    p_s2.add_argument("--config", default="configs/stage2_classifier.yaml")

    p_eval = sub.add_parser("evaluate", help="評価を実行する")
    p_eval.add_argument("--mode", choices=["cross_generator", "ablation", "robustness"], required=True)
    p_eval.add_argument("--config")
    p_eval.add_argument("--experiment-id", type=int)

    p_report = sub.add_parser("report", help="結果レポートを出力する")
    p_report.add_argument("--experiment-id", type=int, required=True)
    p_report.add_argument("--out", default="reports/")

    return parser


def cmd_init_db(args: argparse.Namespace) -> int:
    config = Config.load(args.config)
    db_path = config.get("paths.db_path")

    applied = db_module.init_db(db_path, force=args.force)
    tables = db_module.table_names(db_path)

    print(f"メタデータDB: {resolve(db_path)}")
    if applied:
        print(f"適用したマイグレーション: {', '.join(applied)}")
    else:
        print("適用したマイグレーション: なし（すべて適用済み）")
    print(f"テーブル({len(tables)}): {', '.join(tables)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)

    handlers = {
        "init-db": cmd_init_db,
    }
    handler = handlers.get(args.command)
    if handler is None:
        # Step 3以降で順次実装していく。未実装コマンドは明示的に失敗させる。
        raise NotImplementedError(f"コマンド '{args.command}' は未実装です。")
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
