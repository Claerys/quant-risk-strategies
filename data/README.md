# data/

Everything in this folder except this file is ignored by git.

Market data is licensed to the account that downloads it, so it is not redistributed here. Each
user builds their own copy locally.

## Layout

```
data/daily/<SYMBOL>.parquet    one file per instrument: date, open, high, low, close, volume
```

Symbols and contract specs are in [`src/quant_risk/instruments.csv`](../src/quant_risk/instruments.csv).

## Getting the data

**From a local archive** of daily back-adjusted continuous futures (one `<SYMBOL>.parquet` per
instrument with columns `ts, open, high, low, close, volume`):

```bash
python scripts/import_archive.py /path/to/archive/daily
```

The script imports every symbol in the universe and prints a data-quality summary.

**From Interactive Brokers**: downloader coming next.

## Notes on the prices

The futures series are **back-adjusted continuous contracts**. Older prices are shifted so that
contract rolls do not appear as jumps, which can push them below zero (crude oil, heating oil,
gasoline, soybeans, soybean meal). Price differences are preserved by the adjustment; ratios are
not. Risk in this project is therefore measured on daily P&L per contract
(price change x multiplier), never on percentage returns of these prices.
