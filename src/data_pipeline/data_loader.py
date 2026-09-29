import time
from pathlib import Path

import ccxt
import pandas as pd


SYMBOL = "BTC/USDT"
TIMEFRAME = "5m"

START_DATE = "2023-01-01T00:00:00Z"

OUTPUT_PATH = Path("data/raw/btc_usdt_5m.csv")


def create_exchange():
    exchange = ccxt.binance({
        "enableRateLimit": True
    })

    return exchange


def fetch_historical_ohlcv(
    exchange,
    symbol,
    timeframe,
    start_date,
    limit=1000
):
    since = exchange.parse8601(start_date)

    all_data = []

    while True:

        print(
            f"Fetching from "
            f"{pd.to_datetime(since, unit='ms', utc=True)}"
        )

        ohlcv = exchange.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=since,
            limit=limit
        )

        if len(ohlcv) == 0:
            break

        all_data.extend(ohlcv)

        last_timestamp = ohlcv[-1][0]

        since = last_timestamp + 1

        if len(ohlcv) < limit:
            break

        time.sleep(exchange.rateLimit / 1000)

    return all_data


def convert_to_dataframe(data):

    columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    df = pd.DataFrame(
        data,
        columns=columns
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True
    )

    return df


def clean_data(df):

    df = df.sort_values("timestamp")

    df = df.drop_duplicates(
        subset=["timestamp"]
    )

    df = df.reset_index(drop=True)

    return df


def save_data(df, output_path):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nSaved {len(df):,} rows to {output_path}"
    )


def main():

    exchange = create_exchange()

    data = fetch_historical_ohlcv(
        exchange=exchange,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        start_date=START_DATE
    )

    df = convert_to_dataframe(data)

    df = clean_data(df)

    print(df.head())

    print(df.tail())

    print(
        "\nDataset shape:",
        df.shape
    )

    save_data(
        df,
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()