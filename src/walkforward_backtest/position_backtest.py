from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path(
    "data/processed/walkforward_vol48_predictions.csv"
)

OUTPUT_PATH = Path(
    "data/processed/position_based_backtest.csv"
)


RAW_TARGET = "future_return_15m"

TAIL = 0.01

# Cost per full round trip.
ROUND_TRIP_COST_BPS = 1.0

ROLLING_WINDOW = (
    30 * 24 * 4
)

MIN_HISTORY = (
    7 * 24 * 4
)

PERIODS_PER_YEAR = (
    365 * 24 * 4
)


def load_data():

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["timestamp"]
    )

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )

    return df


def add_threshold(df):

    df = df.copy()

    past_prediction = (
        df["prediction"]
        .shift(1)
    )

    rolling = (
        past_prediction
        .rolling(
            window=ROLLING_WINDOW,
            min_periods=MIN_HISTORY
        )
    )

    df["long_threshold"] = (
        rolling.quantile(
            1 - TAIL
        )
    )

    return df


def generate_desired_position(df):

    df = df.copy()

    df["desired_position"] = 0

    valid = (
        df["long_threshold"]
        .notna()
    )

    long_mask = (
        valid
        &
        (
            df["prediction"]
            >=
            df["long_threshold"]
        )
    )

    df.loc[
        long_mask,
        "desired_position"
    ] = 1

    return df


def create_position(df):

    df = df.copy()

    # Current strategy:
    # each signal decides exposure for
    # the following 15-minute return.
    df["position"] = (
        df["desired_position"]
    )

    return df


def calculate_turnover(df):

    df = df.copy()

    previous_position = (
        df["position"]
        .shift(1)
        .fillna(0)
    )

    df["position_change"] = (
        df["position"]
        -
        previous_position
    )

    df["turnover"] = (
        df["position_change"]
        .abs()
    )

    return df


def calculate_returns(df):

    df = df.copy()

    df["gross_return"] = (
        df["position"]
        *
        df[RAW_TARGET]
    )

    # If round trip = 1 bps,
    # one side (entry OR exit) = 0.5 bps.
    one_way_cost = (
        ROUND_TRIP_COST_BPS
        / 2
        / 10000
    )

    df["transaction_cost"] = (
        df["turnover"]
        *
        one_way_cost
    )

    df["net_return"] = (
        df["gross_return"]
        -
        df["transaction_cost"]
    )

    return df


def calculate_metrics(df):

    valid = df[
        df["long_threshold"]
        .notna()
    ].copy()

    active = valid[
        valid["position"] != 0
    ].copy()

    if len(active) == 0:

        return {}

    returns = (
        valid["net_return"]
    )

    gross_active_bps = (
        active[
            "gross_return"
        ].mean()
        * 10000
    )

    total_cost_bps = (
        valid[
            "transaction_cost"
        ].sum()
        * 10000
    )

    total_turnover = (
        valid["turnover"]
        .sum()
    )

    entries = (
        valid[
            "position_change"
        ]
        > 0
    ).sum()

    exits = (
        valid[
            "position_change"
        ]
        < 0
    ).sum()

    if (
        returns.std()
        > 0
    ):

        sharpe = (
            returns.mean()
            /
            returns.std()
            *
            np.sqrt(
                PERIODS_PER_YEAR
            )
        )

    else:

        sharpe = np.nan

    equity = (
        1
        +
        returns
    ).cumprod()

    running_max = (
        equity.cummax()
    )

    drawdown = (
        equity
        /
        running_max
        - 1
    )

    max_drawdown = (
        drawdown.min()
    )

    total_return = (
        equity.iloc[-1]
        - 1
    )

    return {
        "active_periods":
            len(active),

        "entries":
            entries,

        "exits":
            exits,

        "total_turnover":
            total_turnover,

        "avg_gross_active_bps":
            gross_active_bps,

        "total_cost_bps":
            total_cost_bps,

        "sharpe_ratio":
            sharpe,

        "max_drawdown":
            max_drawdown,

        "net_total_return":
            total_return
    }


def calculate_monthly(df):

    df = df.copy()

    df["month"] = (
        df["timestamp"]
        .dt.strftime("%Y-%m")
    )

    rows = []

    for month, group in df.groupby(
        "month"
    ):

        active = group[
            group["position"] != 0
        ]

        monthly_return = (
            (
                1
                +
                group["net_return"]
            )
            .prod()
            - 1
        )

        if len(active) > 0:

            gross_bps = (
                active[
                    "gross_return"
                ].mean()
                * 10000
            )

        else:

            gross_bps = np.nan

        turnover = (
            group["turnover"]
            .sum()
        )

        rows.append({
            "month":
                month,

            "active_periods":
                len(active),

            "turnover":
                turnover,

            "avg_gross_bps":
                gross_bps,

            "monthly_return":
                monthly_return
        })

    return pd.DataFrame(
        rows
    )


def analyze_holding_runs(df):

    df = df.copy()

    # Identify blocks where position changes.
    df["position_group"] = (
        df["position"]
        .ne(
            df["position"]
            .shift()
        )
        .cumsum()
    )

    active = df[
        df["position"] == 1
    ]

    holding_lengths = (
        active.groupby(
            "position_group"
        )
        .size()
    )

    if len(holding_lengths) == 0:

        return

    print(
        "\n=== Holding Duration ==="
    )

    print(
        "Number of long runs:",
        len(holding_lengths)
    )

    print(
        "Average periods:",
        holding_lengths.mean()
    )

    print(
        "Median periods:",
        holding_lengths.median()
    )

    print(
        "Max periods:",
        holding_lengths.max()
    )

    print(
        "Average minutes:",
        holding_lengths.mean()
        * 15
    )


def main():

    df = load_data()

    print(
        "Rows:",
        len(df)
    )

    df = add_threshold(
        df
    )

    df = generate_desired_position(
        df
    )

    df = create_position(
        df
    )

    df = calculate_turnover(
        df
    )

    df = calculate_returns(
        df
    )

    metrics = (
        calculate_metrics(
            df
        )
    )

    print(
        "\n=========================="
    )

    print(
        "=== Position-Based Backtest ==="
    )

    for key, value in (
        metrics.items()
    ):

        print(
            f"{key}: {value}"
        )

    monthly = (
        calculate_monthly(
            df
        )
    )

    print(
        "\n=========================="
    )

    print(
        "=== Monthly Performance ==="
    )

    print(
        monthly.to_string(
            index=False
        )
    )

    positive_month_ratio = (
        monthly[
            "monthly_return"
        ]
        > 0
    ).mean()

    print(
        "\nPositive month ratio:",
        positive_month_ratio
    )

    analyze_holding_runs(
        df
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        "\nSaved to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()