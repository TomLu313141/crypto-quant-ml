from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("data/raw/btc_usdt_5m.csv")
OUTPUT_PATH = Path("data/processed/btc_usdt_5m_features.csv")


def load_data(path):
    df = pd.read_csv(
        path,
        parse_dates=["timestamp"]
    )

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def add_segment_id(df):
    """
    Split the dataset whenever timestamp gap > 5 minutes.
    This prevents rolling features from crossing missing-data gaps.
    """

    expected_interval = pd.Timedelta(minutes=5)

    time_diff = df["timestamp"].diff()

    new_segment = (
        time_diff.isna()
        | (time_diff > expected_interval)
    )

    df["segment_id"] = new_segment.cumsum()

    return df


def add_return_features(df):
    for period in [1, 3, 6, 12, 24]:
        df[f"return_{period}"] = (
            df.groupby("segment_id")["close"]
            .pct_change(period)
        )

    return df


def add_momentum_features(df):
    for window in [6, 12, 48]:

        rolling_ma = (
            df.groupby("segment_id")["close"]
            .rolling(window=window)
            .mean()
            .reset_index(
                level=0,
                drop=True
            )
        )

        df[f"ma_{window}"] = rolling_ma

        df[f"price_ma_ratio_{window}"] = (
            df["close"]
            / df[f"ma_{window}"]
            - 1
        )

    return df


def add_volatility_features(df):
    for window in [6, 12, 48]:

        rolling_vol = (
            df.groupby("segment_id")["return_1"]
            .rolling(window=window)
            .std()
            .reset_index(
                level=0,
                drop=True
            )
        )

        df[f"volatility_{window}"] = rolling_vol

    return df


def add_volume_features(df):

    df["volume_change_1"] = (
        df.groupby("segment_id")["volume"]
        .pct_change(1)
    )

    for window in [12, 48]:

        volume_ma = (
            df.groupby("segment_id")["volume"]
            .rolling(window=window)
            .mean()
            .reset_index(
                level=0,
                drop=True
            )
        )

        df[f"volume_ma_{window}"] = volume_ma

        df[f"volume_ratio_{window}"] = (
            df["volume"]
            / df[f"volume_ma_{window}"]
        )

    volume_mean_48 = (
        df.groupby("segment_id")["volume"]
        .rolling(window=48)
        .mean()
        .reset_index(
            level=0,
            drop=True
        )
    )

    volume_std_48 = (
        df.groupby("segment_id")["volume"]
        .rolling(window=48)
        .std()
        .reset_index(
            level=0,
            drop=True
        )
    )

    df["volume_zscore_48"] = (
        (df["volume"] - volume_mean_48)
        / volume_std_48
    )

    return df


def add_candle_features(df):

    df["candle_return"] = (
        (df["close"] - df["open"])
        / df["open"]
    )

    df["high_low_range"] = (
        (df["high"] - df["low"])
        / df["open"]
    )

    df["upper_shadow"] = (
        df["high"]
        - df[["open", "close"]].max(axis=1)
    ) / df["open"]

    df["lower_shadow"] = (
        df[["open", "close"]].min(axis=1)
        - df["low"]
    ) / df["open"]

    return df


def add_time_features(df):

    df["hour"] = df["timestamp"].dt.hour

    df["day_of_week"] = (
        df["timestamp"].dt.dayofweek
    )

    return df


def add_label(df):
    """
    Future 15-minute return.
    Must not cross a missing-data gap.
    """

    future_close = (
        df.groupby("segment_id")["close"]
        .shift(-3)
    )

    df["future_return_15m"] = (
        future_close
        / df["close"]
        - 1
    )

    return df


def clean_features(df):

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna().reset_index(drop=True)

    return df


def save_data(df, path):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        path,
        index=False
    )

    print(
        f"Saved {len(df):,} rows to {path}"
    )


def main():

    df = load_data(INPUT_PATH)

    print("Original rows:", len(df))

    df = add_segment_id(df)

    print(
        "Segments:",
        df["segment_id"].nunique()
    )

    df = add_return_features(df)

    df = add_momentum_features(df)

    df = add_volatility_features(df)

    df = add_volume_features(df)

    df = add_candle_features(df)

    df = add_time_features(df)

    df = add_label(df)

    df = clean_features(df)

    print("Final rows:", len(df))

    print("\nColumns:")
    print(df.columns.tolist())

    save_data(
        df,
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()