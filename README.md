# Bitcoin Quant ML Research Pipeline

A quantitative machine learning research project for BTC/USDT short-horizon return prediction and trading signal evaluation.

The project uses 5-minute Bitcoin market data and builds a complete workflow from data validation and feature engineering to model training, walk-forward backtesting, monitoring, and final out-of-sample testing.

---

## Project Flow

```text
BTC/USDT 5-minute OHLCV
        ↓
Data Validation
        ↓
Feature Engineering
        ↓
Baseline & Model Selection
        ↓
Research Decisions
        ↓
Walk-Forward Backtest
        ↓
Monitoring
        ↓
Final OOS Test
