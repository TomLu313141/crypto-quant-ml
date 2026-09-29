from pathlib import Path

import pandas as pd


DATA_PATH = Path(
    "data/raw/btc_usdt_5m.csv"
)


def load_data(path):

    df = pd.read_csv(
        path,
        parse_dates=["timestamp"]
    )

    return df


def check_missing_values(df):

    print("\n=== Missing Values ===")

    print(
        df.isna().sum()
    )


def check_duplicates(df):

    duplicate_count = (
        df["timestamp"]
        .duplicated()
        .sum()
    )

    print(
        "\nDuplicate timestamps:",
        duplicate_count
    )


def check_price_validity(df):

    invalid_open = (
        df["open"] <= 0
    ).sum()

    invalid_high = (
        df["high"] <= 0
    ).sum()

    invalid_low = (
        df["low"] <= 0
    ).sum()

    invalid_close = (
        df["close"] <= 0
    ).sum()

    print(
        "\n=== Invalid Prices ==="
    )

    print(
        "Open:",
        invalid_open
    )

    print(
        "High:",
        invalid_high
    )

    print(
        "Low:",
        invalid_low
    )

    print(
        "Close:",
        invalid_close
    )


def check_ohlc_relationship(df):

    invalid_high = (
        df["high"]
        <
        df[["open", "close", "low"]]
        .max(axis=1)
    )

    invalid_low = (
        df["low"]
        >
        df[["open", "close", "high"]]
        .min(axis=1)
    )

    print(
        "\n=== OHLC Relationship ==="
    )

    print(
        "Invalid high rows:",
        invalid_high.sum()
    )

    print(
        "Invalid low rows:",
        invalid_low.sum()
    )


def check_volume(df):

    invalid_volume = (
        df["volume"] < 0
    ).sum()

    print(
        "\nNegative volume rows:",
        invalid_volume
    )

def check_time_gaps(df):

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    diff = df["timestamp"].diff()

    expected_interval = pd.Timedelta(
        minutes=5
    )

    gap_indices = diff[
        diff > expected_interval
    ].index

    print(
        "\n=== Timestamp Gaps ==="
    )

    print(
        "Number of gaps:",
        len(gap_indices)
    )

    for idx in gap_indices:

        previous_time = (
            df.loc[idx - 1, "timestamp"]
        )

        current_time = (
            df.loc[idx, "timestamp"]
        )

        gap_duration = (
            current_time - previous_time
        )

        missing_candles = int(
            gap_duration / expected_interval
        ) - 1

        print("\nGap found:")
        print(
            "Previous timestamp:",
            previous_time
        )

        print(
            "Current timestamp:",
            current_time
        )

        print(
            "Gap duration:",
            gap_duration
        )

        print(
            "Missing candles:",
            missing_candles
        )

def main():

    df = load_data(
        DATA_PATH
    )

    print(
        "Rows:",
        len(df)
    )

    print(
        "Start:",
        df["timestamp"].min()
    )

    print(
        "End:",
        df["timestamp"].max()
    )

    check_missing_values(df)

    check_duplicates(df)

    check_price_validity(df)

    check_ohlc_relationship(df)

    check_volume(df)

    check_time_gaps(df)


if __name__ == "__main__":
    main()