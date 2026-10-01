"""Import daily futures bars from a local archive folder into data/daily/.

The archive is a folder of ``<SYMBOL>.parquet`` files with columns ts/open/high/low/close/volume
(the format of the BSQF project archive). Only symbols in the instrument universe are imported.
Every imported series goes through the data-quality checks and a summary is printed.

    python scripts/import_archive.py /path/to/archive/daily
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from quant_risk.bars import data_dir, normalize_bars, write_bars
from quant_risk.instruments import load_instruments
from quant_risk.quality import ERROR, check_bars, summarize


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archive", type=Path, help="folder containing <SYMBOL>.parquet daily files")
    parser.add_argument("--out", type=Path, default=None, help=f"destination (default {data_dir()})")
    args = parser.parse_args(argv)

    universe = load_instruments()
    files = sorted(args.archive.glob("*.parquet"))
    if not files:
        print(f"no .parquet files in {args.archive}", file=sys.stderr)
        return 1

    issues = []
    rows = []
    for path in files:
        symbol = path.stem
        if symbol not in universe:
            print(f"skip {symbol}: not in instruments.csv")
            continue
        bars = normalize_bars(pd.read_parquet(path))
        write_bars(symbol, bars, args.out)
        issues += check_bars(symbol, bars)
        rows.append((symbol, len(bars), bars.index[0].date(), bars.index[-1].date()))

    print(f"\nimported {len(rows)} instruments into {args.out or data_dir()}")
    print(pd.DataFrame(rows, columns=["symbol", "bars", "first", "last"]).to_string(index=False))
    print("\ndata-quality summary")
    print(summarize(issues).to_string(index=False))

    errors = sum(i.severity == ERROR for i in issues)
    if errors:
        print(f"\n{errors} error(s): those bars must be fixed or excluded before use", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
