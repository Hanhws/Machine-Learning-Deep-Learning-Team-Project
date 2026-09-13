"""experiments.csv → experiments.md (그룹별 표, 그룹 최고 성능 ★).

    python make_report.py --csv runs/experiments.csv
    python make_report.py --csv runs/experiments.csv --split test --sort vehicle_IoU
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from carseg.experiment_log import render_markdown


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="runs/experiments.csv")
    p.add_argument("--split", default="val")
    p.add_argument("--sort", default="mask_mAP")
    p.add_argument("--out", default=None, help="기본: CSV 옆 experiments_<split>.md")
    args = p.parse_args(argv)

    md = render_markdown(args.csv, split=args.split, sort_by=args.sort)
    out = Path(args.out) if args.out else Path(args.csv).with_name(f"experiments_{args.split}.md")
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"[report] 저장: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
