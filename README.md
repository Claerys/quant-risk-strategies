# quant-risk-strategies

Market risk analytics for a systematic futures portfolio: position sizing, exposure limits,
drawdown monitoring, VaR / Expected Shortfall, stress testing (including climate scenarios),
and a daily risk report, on market data pulled from Interactive Brokers.

> Personal rebuild of the risk layer I worked on in the BSQF IBKR algo-trading project
> (Bocconi Students Quantitative Finance), where I was a Quantitative Risk Management Analyst.

## Why this project

A trading strategy answers *"what should we hold?"*. A risk function answers the questions that
come after it:

- **How much can we lose?** Value-at-Risk and Expected Shortfall, three ways.
- **Is that estimate any good?** VaR backtesting against realised P&L.
- **What happens in a crisis?** Historical and hypothetical stress scenarios.
- **Are we within mandate?** Exposure, concentration and loss limits, checked before every trade.
- **What if the climate transition is disorderly?** Climate stress scenarios on energy and
  agricultural futures.

The strategies here are deliberately simple. The point of the project is the risk layer around them.

## Roadmap

- [x] **Market data** – 35 back-adjusted futures (1990–2025) and EUR/USD, with data-quality checks
- [x] **Strategies** – time-series momentum, multi-horizon trend, MA crossover, mean reversion, buy & hold
- [x] **Position sizing** – volatility targeting, sector risk budgets, diversification multiplier, ex-ante risk overlay
- [ ] **Risk limits** – gross/net exposure, max order and position size, max loss, kill switch (fail-closed)
- [ ] **Risk metrics** – realised volatility, drawdown, VaR and ES (historical, parametric, Monte Carlo)
- [ ] **VaR backtesting** – exception counts, Kupiec POF test, Basel traffic-light zones
- [ ] **Stress testing** – 2008 GFC, March 2020, 2022 rate shock, plus custom shocks
- [ ] **Climate scenarios** – NGFS-style transition and physical-risk shocks on energy and grain futures
- [ ] **Risk dashboard** – a one-page daily risk report
- [ ] **Paper trading** – IBKR paper account with pre-trade limit checks and a trade journal

## Project layout

```
src/quant_risk/   library code
tests/            unit tests
data/             market data (downloaded locally, never committed)
```

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Running

```bash
python scripts/import_archive.py /path/to/archive/daily --fx-hourly /path/to/EURUSD.parquet
python scripts/run_backtest.py
```

`run_backtest.py` sizes every strategy to the volatility target, backtests it from 2003 and prints
return, volatility (predicted and realised), Sharpe, Sortino, drawdown, skew and trading costs.

## Data

Market data is **not** stored in this repository. It is imported locally into `data/` (ignored by
git) from an archive of back-adjusted continuous futures; an Interactive Brokers downloader is in
progress. See [`data/README.md`](data/README.md).

## Disclaimer

Educational project. Nothing here is investment advice, and live trading is not supported.
