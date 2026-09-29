from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb


DATA_PATH = Path(
    "data/processed/btc_usdt_5m_features.csv"
)

PREDICTION_OUTPUT = Path(
    "data/processed/walkforward_vol48_predictions.csv"
)

RESULT_OUTPUT = Path(
    "data/processed/walkforward_vol48_backtest.csv"
)

MONTHLY_OUTPUT = Path(
    "data/processed/walkforward_vol48_monthly.csv"
)


RAW_TARGET = "future_return_15m"

VOL_COLUMN = "volatility_48"


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


# =========================================================
# Walk-forward settings
# =========================================================

PREDICTION_MONTHS = pd.date_range(
    start="2025-01-01",
    end="2025-12-01",
    freq="MS",
    tz="UTC"
)


# Target horizon = 15 minutes.
#
# Rows too close to a split boundary are removed so that
# their future label cannot cross into the next period.
PURGE_MINUTES = 15


# Internal validation period used ONLY to determine
# LightGBM best_iteration.
INTERNAL_VALIDATION_DAYS = 30


# =========================================================
# Trading settings
# =========================================================

TAIL = 0.01

TRANSACTION_COST_BPS = 1.0


# 30 days of 15-minute observations:
# 30 * 24 * 4
ROLLING_WINDOW = (
    30 * 24 * 4
)

# Require at least 7 days of prediction history before
# calculating a percentile threshold.
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


# =========================================================
# Target
# =========================================================

def create_normalized_target(
    df,
    lower=None,
    upper=None
):

    denominator = (
        df[VOL_COLUMN]
        .clip(lower=1e-6)
    )

    target = (
        df[RAW_TARGET]
        /
        denominator
    )

    if lower is not None:

        target = target.clip(
            lower=lower,
            upper=upper
        )

    return target


def calculate_target_bounds(
    df
):

    target = (
        create_normalized_target(
            df
        )
    )

    lower = target.quantile(
        0.005
    )

    upper = target.quantile(
        0.995
    )

    return (
        lower,
        upper
    )


# =========================================================
# Model training
# =========================================================

def find_best_iteration(
    history,
    prediction_start
):

    purge = pd.Timedelta(
        minutes=PURGE_MINUTES
    )

    validation_start = (
        prediction_start
        -
        pd.Timedelta(
            days=INTERNAL_VALIDATION_DAYS
        )
    )

    # Fit set is purged before the internal validation.
    fit_cutoff = (
        validation_start
        -
        purge
    )

    fit_df = history[
        history["timestamp"]
        <
        fit_cutoff
    ].copy()

    validation_df = history[
        (
            history["timestamp"]
            >= validation_start
        )
        &
        (
            history["timestamp"]
            <
            prediction_start
            -
            purge
        )
    ].copy()

    if (
        len(fit_df) == 0
        or
        len(validation_df) == 0
    ):

        raise ValueError(
            "Insufficient data for internal validation."
        )

    # Target bounds calculated from fit only.
    lower, upper = (
        calculate_target_bounds(
            fit_df
        )
    )

    fit_target = (
        create_normalized_target(
            fit_df,
            lower,
            upper
        )
    )

    validation_target = (
        create_normalized_target(
            validation_df,
            lower,
            upper
        )
    )

    model = lgb.LGBMRegressor(
        objective="regression_l1",

        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,

        subsample=0.8,
        colsample_bytree=0.8,

        reg_alpha=0.1,
        reg_lambda=0.1,

        random_state=42,
        n_jobs=-1,

        force_row_wise=True
    )

    model.fit(
        fit_df[
            FEATURE_COLUMNS
        ],
        fit_target,

        eval_X=validation_df[
            FEATURE_COLUMNS
        ],

        eval_y=validation_target,

        callbacks=[
            lgb.early_stopping(
                stopping_rounds=50,
                verbose=False
            )
        ]
    )

    best_iteration = (
        model.best_iteration_
    )

    return best_iteration


def train_final_model(
    history,
    prediction_start,
    best_iteration
):

    purge = pd.Timedelta(
        minutes=PURGE_MINUTES
    )

    train_cutoff = (
        prediction_start
        -
        purge
    )

    train_df = history[
        history["timestamp"]
        <
        train_cutoff
    ].copy()

    # At prediction time all of this historical
    # information is available.
    lower, upper = (
        calculate_target_bounds(
            train_df
        )
    )

    train_target = (
        create_normalized_target(
            train_df,
            lower,
            upper
        )
    )

    model = lgb.LGBMRegressor(
        objective="regression_l1",

        n_estimators=max(
            1,
            int(best_iteration)
        ),

        learning_rate=0.03,
        num_leaves=31,

        subsample=0.8,
        colsample_bytree=0.8,

        reg_alpha=0.1,
        reg_lambda=0.1,

        random_state=42,
        n_jobs=-1,

        force_row_wise=True
    )

    model.fit(
        train_df[
            FEATURE_COLUMNS
        ],
        train_target
    )

    return (
        model,
        len(train_df),
        lower,
        upper
    )


# =========================================================
# Monthly walk-forward prediction
# =========================================================

def predict_month(
    df,
    prediction_start
):

    prediction_end = (
        prediction_start
        +
        pd.offsets.MonthBegin(1)
    )

    history = df[
        df["timestamp"]
        <
        prediction_start
    ].copy()

    month_df = df[
        (
            df["timestamp"]
            >= prediction_start
        )
        &
        (
            df["timestamp"]
            <
            prediction_end
        )
    ].copy()

    if len(month_df) == 0:

        return None

    best_iteration = (
        find_best_iteration(
            history,
            prediction_start
        )
    )

    (
        model,
        train_rows,
        lower,
        upper
    ) = train_final_model(
        history,
        prediction_start,
        best_iteration
    )

    month_df[
        "prediction"
    ] = model.predict(
        month_df[
            FEATURE_COLUMNS
        ]
    )

    month_df[
        "prediction_month"
    ] = (
        prediction_start.strftime(
            "%Y-%m"
        )
    )

    month_df[
        "best_iteration"
    ] = best_iteration

    month_df[
        "train_rows"
    ] = train_rows

    print(
        "\n========================="
    )

    print(
        "Prediction month:",
        prediction_start.strftime(
            "%Y-%m"
        )
    )

    print(
        "Train rows:",
        train_rows
    )

    print(
        "Best iteration:",
        best_iteration
    )

    print(
        "Target bounds:",
        lower,
        upper
    )

    print(
        "Prediction rows:",
        len(month_df)
    )

    return month_df


def run_walkforward_predictions(
    df
):

    predictions = []

    for prediction_start in (
        PREDICTION_MONTHS
    ):

        month_df = (
            predict_month(
                df,
                prediction_start
            )
        )

        if month_df is not None:

            predictions.append(
                month_df
            )

    prediction_df = (
        pd.concat(
            predictions,
            ignore_index=True
        )
    )

    prediction_df = (
        prediction_df.sort_values(
            "timestamp"
        )
        .reset_index(drop=True)
    )

    return prediction_df


# =========================================================
# Non-overlapping 15-minute observations
# =========================================================

def select_non_overlap(
    df
):

    mask = (
        df["timestamp"]
        .dt.minute
        % 15
        == 0
    )

    return (
        df[mask]
        .copy()
        .reset_index(drop=True)
    )


# =========================================================
# Rolling thresholds
# =========================================================

def add_rolling_thresholds(
    df
):

    df = df.copy()

    # Critical:
    # current prediction cannot participate in
    # its own percentile calculation.
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

    df[
        "short_threshold"
    ] = rolling.quantile(
        TAIL
    )

    df[
        "long_threshold"
    ] = rolling.quantile(
        1 - TAIL
    )

    return df


# =========================================================
# Signal
# =========================================================

def generate_signal(
    df,
    side
):

    df = df.copy()

    df["signal"] = 0

    valid = (
        df["long_threshold"]
        .notna()
        &
        df["short_threshold"]
        .notna()
    )

    if side in [
        "short",
        "long_short"
    ]:

        short_mask = (
            valid
            &
            (
                df["prediction"]
                <=
                df["short_threshold"]
            )
        )

        df.loc[
            short_mask,
            "signal"
        ] = -1

    if side in [
        "long",
        "long_short"
    ]:

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
            "signal"
        ] = 1

    return df


# =========================================================
# Returns and transaction cost
# =========================================================

def calculate_returns(
    df
):

    df = df.copy()

    df[
        "gross_return"
    ] = (
        df["signal"]
        *
        df[RAW_TARGET]
    )

    cost = (
        TRANSACTION_COST_BPS
        /
        10000
    )

    df[
        "transaction_cost"
    ] = np.where(
        df["signal"] != 0,
        cost,
        0.0
    )

    df[
        "net_return"
    ] = (
        df["gross_return"]
        -
        df["transaction_cost"]
    )

    return df


# =========================================================
# Performance
# =========================================================

def calculate_performance(
    df
):

    valid = df[
        df["long_threshold"]
        .notna()
    ].copy()

    active = valid[
        valid["signal"] != 0
    ].copy()

    trade_count = len(
        active
    )

    if trade_count == 0:

        return {
            "trade_count": 0
        }

    returns = (
        valid["net_return"]
    )

    avg_gross_bps = (
        active[
            "gross_return"
        ].mean()
        *
        10000
    )

    avg_net_bps = (
        active[
            "net_return"
        ].mean()
        *
        10000
    )

    gross_hit_rate = (
        active[
            "gross_return"
        ]
        > 0
    ).mean()

    net_hit_rate = (
        active[
            "net_return"
        ]
        > 0
    ).mean()

    if (
        len(returns) > 1
        and
        returns.std() > 0
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

    long_trades = active[
        active["signal"] == 1
    ]

    short_trades = active[
        active["signal"] == -1
    ]

    long_avg_bps = (
        long_trades[
            "gross_return"
        ].mean()
        * 10000
        if len(long_trades) > 0
        else np.nan
    )

    short_avg_bps = (
        short_trades[
            "gross_return"
        ].mean()
        * 10000
        if len(short_trades) > 0
        else np.nan
    )

    return {
        "trade_count":
            trade_count,

        "long_count":
            len(long_trades),

        "short_count":
            len(short_trades),

        "avg_gross_bps":
            avg_gross_bps,

        "avg_net_bps":
            avg_net_bps,

        "gross_hit_rate":
            gross_hit_rate,

        "net_hit_rate":
            net_hit_rate,

        "long_avg_bps":
            long_avg_bps,

        "short_avg_bps":
            short_avg_bps,

        "sharpe_ratio":
            sharpe,

        "max_drawdown":
            max_drawdown,

        "net_total_return":
            total_return
    }


def calculate_monthly_performance(
    df,
    side
):

    temp = df.copy()

    temp["month"] = (
        temp["timestamp"]
        .dt.strftime(
            "%Y-%m"
        )
    )

    rows = []

    for month, group in temp.groupby(
        "month"
    ):

        active = group[
            group["signal"] != 0
        ]

        if len(active) == 0:

            rows.append({
                "side":
                    side,

                "month":
                    month,

                "trade_count":
                    0,

                "avg_gross_bps":
                    np.nan,

                "avg_net_bps":
                    np.nan,

                "monthly_return":
                    0.0
            })

            continue

        monthly_return = (
            (
                1
                +
                group[
                    "net_return"
                ]
            )
            .prod()
            - 1
        )

        rows.append({
            "side":
                side,

            "month":
                month,

            "trade_count":
                len(active),

            "avg_gross_bps":
                active[
                    "gross_return"
                ].mean()
                * 10000,

            "avg_net_bps":
                active[
                    "net_return"
                ].mean()
                * 10000,

            "hit_rate":
                (
                    active[
                        "net_return"
                    ]
                    > 0
                ).mean(),

            "monthly_return":
                monthly_return
        })

    return pd.DataFrame(
        rows
    )


# =========================================================
# Backtest
# =========================================================

def run_side(
    prediction_df,
    side
):

    temp = (
        generate_signal(
            prediction_df,
            side
        )
    )

    temp = (
        calculate_returns(
            temp
        )
    )

    performance = (
        calculate_performance(
            temp
        )
    )

    performance[
        "side"
    ] = side

    performance[
        "tail"
    ] = TAIL

    performance[
        "cost_bps"
    ] = (
        TRANSACTION_COST_BPS
    )

    monthly = (
        calculate_monthly_performance(
            temp,
            side
        )
    )

    positive_month_ratio = (
        monthly[
            "monthly_return"
        ]
        > 0
    ).mean()

    performance[
        "positive_month_ratio"
    ] = (
        positive_month_ratio
    )

    return (
        performance,
        monthly
    )


# =========================================================
# Main
# =========================================================

def main():

    df = load_data()

    print(
        "Total dataset rows:",
        len(df)
    )

    # ---------------------------------------------
    # Monthly walk-forward model retraining
    # ---------------------------------------------

    prediction_df = (
        run_walkforward_predictions(
            df
        )
    )

    # ---------------------------------------------
    # Non-overlapping observations
    # ---------------------------------------------

    prediction_df = (
        select_non_overlap(
            prediction_df
        )
    )

    print(
        "\nWalk-forward non-overlap rows:",
        len(prediction_df)
    )

    # ---------------------------------------------
    # Rolling trading threshold
    # ---------------------------------------------

    prediction_df = (
        add_rolling_thresholds(
            prediction_df
        )
    )

    # ---------------------------------------------
    # Save predictions
    # ---------------------------------------------

    PREDICTION_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    prediction_df.to_csv(
        PREDICTION_OUTPUT,
        index=False
    )

    # ---------------------------------------------
    # Strategies
    # ---------------------------------------------

    results = []

    monthly_results = []

    for side in [
        "long",
        "short",
        "long_short"
    ]:

        (
            result,
            monthly
        ) = run_side(
            prediction_df,
            side
        )

        results.append(
            result
        )

        monthly_results.append(
            monthly
        )

    result_df = pd.DataFrame(
        results
    )

    monthly_df = pd.concat(
        monthly_results,
        ignore_index=True
    )

    # ---------------------------------------------
    # Print overall
    # ---------------------------------------------

    print(
        "\n=========================="
    )

    print(
        "=== Walk-Forward Backtest ==="
    )

    display_columns = [
        "side",

        "trade_count",
        "long_count",
        "short_count",

        "avg_gross_bps",
        "avg_net_bps",

        "long_avg_bps",
        "short_avg_bps",

        "net_hit_rate",

        "sharpe_ratio",

        "max_drawdown",

        "net_total_return",

        "positive_month_ratio"
    ]

    print(
        result_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    # ---------------------------------------------
    # Print monthly
    # ---------------------------------------------

    print(
        "\n=========================="
    )

    print(
        "=== Monthly Performance ==="
    )

    print(
        monthly_df.to_string(
            index=False
        )
    )

    # ---------------------------------------------
    # Save
    # ---------------------------------------------

    result_df.to_csv(
        RESULT_OUTPUT,
        index=False
    )

    monthly_df.to_csv(
        MONTHLY_OUTPUT,
        index=False
    )

    print(
        "\nSaved predictions to:"
    )

    print(
        PREDICTION_OUTPUT
    )

    print(
        "\nSaved backtest to:"
    )

    print(
        RESULT_OUTPUT
    )

    print(
        "\nSaved monthly results to:"
    )

    print(
        MONTHLY_OUTPUT
    )


if __name__ == "__main__":
    main()