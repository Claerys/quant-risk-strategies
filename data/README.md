# data/

Everything in this folder except this file is ignored by git.

Market data is licensed to the account that downloads it, so it is not redistributed here. Each
user builds their own copy locally.

## Layout

```
data/daily/<SYMBOL>.parquet    one file per instrument: date, open, high, low, close, volume
data/fx/EURUSD.parquet         daily spot EUR/USD, used to convert euro P&L into USD
```

Symbols and contract specs are in [`src/quant_risk/instruments.csv`](../src/quant_risk/instruments.csv).

## Getting the data

**From a local archive** of daily back-adjusted continuous futures (one `<SYMBOL>.parquet` per
instrument with columns `ts, open, high, low, close, volume`):

```bash
python scripts/import_archive.py /path/to/archive/daily
```

The script imports every symbol in the universe and prints a data-quality summary. Add
`--fx-hourly /path/to/EURUSD.parquet` to import spot EUR/USD as well (hourly bars are aggregated to
daily). Outside the spot file's dates, EUR/USD is extended with Euro FX futures price changes.



## Notes on the prices

The futures series are **back-adjusted continuous contracts**. Older prices are shifted so that
contract rolls do not appear as jumps, which can push them below zero (crude oil, heating oil,
gasoline, soybeans, soybean meal). Price differences are preserved by the adjustment; ratios are
not. Risk in this project is therefore measured on daily P&L per contract
(price change x multiplier), never on percentage returns of these prices.

**From Interactive Brokers** (IB Gateway or TWS, paper account, API socket enabled):

```bash
pip install -e ".[ibkr]"
python scripts/download_ibkr.py --dry-run          # qualify contracts, write nothing
python scripts/download_ibkr.py --symbols ES CL --years 10 --fx
```

Every listed and expired contract month is fetched, chained into one back-adjusted series and written to
`data/daily/<SYMBOL>.parquet`. A later run fetches only the newest months, reconciles them against the
file already on disk and refuses to write if the two disagree. Nothing is written if the result fails
the data-quality checks.
