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
```

---

## Pipeline

### 1. Data Pipeline
Loads BTC/USDT 5-minute OHLCV data and checks:
- missing values
- duplicate timestamps
- invalid OHLC data
- timestamp gaps

### 2. Feature Engineering
Builds 21 market features covering:
- momentum
- volatility
- volume
- price vs moving average
- candle structure
- time features
The prediction target is based on the future 15-minute return.

### 3. Baseline & Model Selection
Simple baselines are used to verify whether the ML model learns additional information:
- Zero Prediction
- Momentum
- Ridge Regression
LightGBM is then evaluated with robust objectives.

### 4. Research Decisions
Several design choices are tested during development, including:
- prediction horizon
- target normalization
- feature complexity
- robust evaluation metrics
The final model uses a compact feature set and a volatility-normalized target.

### 5. Walk-Forward Backtest
The model is evaluated using time-series walk-forward validation instead of random splitting.
The trading candidate focuses on extreme prediction signals using a past-only rolling threshold.

### 6. Monitoring
The monitoring layer tracks:
- feature / prediction drift
- signal return
- hit rate
- drawdown
- signal frequency
This helps distinguish data drift from actual performance deterioration.

### 7. Final OOS Test
The final research configuration is frozen before evaluating the 2026 out-of-sample period.

---

## Results

Key summary files are stored in:
```text
reports/
```

including:
- model comparison
- target comparison
- walk-forward summary
- final OOS summary

---

## Environment
```text
pip install -r requirements.txt
```
