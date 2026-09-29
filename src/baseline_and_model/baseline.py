from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


DATA_PATH = Path(
    "data/processed/btc_usdt_5m_features.csv"
)


TARGET = "future_return_15m"


FEATURE_COLUMNS = [
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "return_24",

    "price_ma_ratio_6",
    "price_ma_ratio_12",
    "price_ma_ratio_48",

    "volatility_6",
    "volatility_12",
    "volatility_48",

    "volume_change_1",
    "volume_ratio_12",
    "volume_ratio_48",
    "volume_zscore_48",

    "candle_return",
    "high_low_range",
    "upper_shadow",
    "lower_shadow",

    "hour",
    "day_of_week"
]


def load_data():
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["timestamp"]
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return df


def split_data(df):

    train = df[
        df["timestamp"]
        < "2025-01-01"
    ].copy()

    validation = df[
        (
            df["timestamp"]
            >= "2025-01-01"
        )
        &
        (
            df["timestamp"]
            < "2026-01-01"
        )
    ].copy()

    test = df[
        df["timestamp"]
        >= "2026-01-01"
    ].copy()

    return (
        train,
        validation,
        test
    )


def calculate_metrics(
    y_true,
    y_pred
):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    if np.std(y_pred) == 0:
        correlation = np.nan
    else:
        correlation = np.corrcoef(
            y_true,
            y_pred
        )[0, 1]

    return {
        "MAE": mae,
        "RMSE": rmse,
        "IC": correlation
    }


def print_metrics(
    model_name,
    metrics
):

    print(
        f"\n=== {model_name} ==="
    )

    for key, value in metrics.items():

        print(
            f"{key}: {value:.8f}"
        )


def zero_baseline(validation):

    y_true = validation[TARGET]

    y_pred = np.zeros(
        len(validation)
    )

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    return metrics


def momentum_baseline(validation):

    y_true = validation[TARGET]

    y_pred = validation[
        "return_3"
    ].values

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    return metrics

def ridge_baseline(
    train,
    validation
):

    X_train = train[
        FEATURE_COLUMNS
    ]

    y_train = train[
        TARGET
    ]

    X_val = validation[
        FEATURE_COLUMNS
    ]

    y_val = validation[
        TARGET
    ]

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "ridge",
            Ridge(alpha=1.0)
        )
    ])

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_val
    )

    metrics = calculate_metrics(
        y_val,
        y_pred
    )

    return (
        model,
        metrics
    )

def main():

    df = load_data()

    train, validation, test = split_data(
        df
    )

    print(
        "Train rows:",
        len(train)
    )

    print(
        "Validation rows:",
        len(validation)
    )

    print(
        "Test rows:",
        len(test)
    )

    print(
        "\nTrain period:",
        train["timestamp"].min(),
        "->",
        train["timestamp"].max()
    )

    print(
        "\nValidation period:",
        validation["timestamp"].min(),
        "->",
        validation["timestamp"].max()
    )

    print(
        "\nTest period:",
        test["timestamp"].min(),
        "->",
        test["timestamp"].max()
    )

    zero_metrics = zero_baseline(
        validation
    )

    print_metrics(
        "Zero Baseline",
        zero_metrics
    )

    momentum_metrics = momentum_baseline(
        validation
    )

    print_metrics(
        "Momentum Baseline",
        momentum_metrics
    )

    ridge_model, ridge_metrics = ridge_baseline(
        train,
        validation
    )

    print_metrics(
        "Ridge Regression",
        ridge_metrics
    )


if __name__ == "__main__":
    main()