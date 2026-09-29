from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb


# =========================================================
# Paths
# =========================================================

FEATURE_DATA_PATH = Path(
    "data/processed/btc_usdt_5m_features.csv"
)

# 2025 walk-forward predictions are used ONLY as
# historical prediction context for the rolling threshold.
#
# They are not included in 2026 performance.
HISTORY_PREDICTION_PATH = Path(
    "data/processed/walkforward_vol48_predictions.csv"
)

PREDICTION_OUTPUT = Path(
    "data/processed/final_test_2026_predictions.csv"
)

FOLD_OUTPUT = Path(
    "data/processed/final_test_2026_folds.csv"
)

MONTHLY_OUTPUT = Path(
    "data/processed/final_test_2026_monthly.csv"
)

SUMMARY_OUTPUT = Path(
    "data/processed/final_test_2026_summary.csv"
)


# =========================================================
# Frozen research assumptions
# =========================================================

TEST_START = pd.Timestamp(
    "2026-01-01",
    tz="UTC"
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

PURGE_MINUTES = 15

INTERNAL_VALIDATION_DAYS = 30


# =========================================================
# Signal settings
# =========================================================

TAIL = 0.01

ROLLING_DAYS = 30

ROLLING_WINDOW = (
    ROLLING_DAYS
    * 24
    * 4
)

MIN_HISTORY = (
    7
    * 24
    * 4
)


# =========================================================
# Transaction cost
# =========================================================

ROUND_TRIP_COST_BPS = 1.0

ONE_WAY_COST = (
    ROUND_TRIP_COST_BPS
    / 2
    / 10000
)


PERIODS_PER_YEAR = (
    365
    * 24
    * 4
)


# =========================================================
# Data
# =========================================================

def load_feature_data():

    df = pd.read_csv(
        FEATURE_DATA_PATH,
        parse_dates=["timestamp"]
    )

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )

    return df


def load_prediction_history():

    history = pd.read_csv(
        HISTORY_PREDICTION_PATH,
        parse_dates=["timestamp"]
    )

    history = (
        history[
            history["timestamp"]
            <
            TEST_START
        ][
            [
                "timestamp",
                "prediction"
            ]
        ]
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"]
        )
        .reset_index(drop=True)
    )

    return history


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

    if (
        lower is not None
        and
        upper is not None
    ):

        target = target.clip(
            lower=lower,
            upper=upper
        )

    return target


def calculate_target_bounds(df):

    target = (
        create_normalized_target(
            df
        )
    )

    lower = (
        target.quantile(
            0.005
        )
    )

    upper = (
        target.quantile(
            0.995
        )
    )

    return (
        lower,
        upper
    )


# =========================================================
# Model
# =========================================================

def build_model(
    n_estimators
):

    return lgb.LGBMRegressor(

        objective="regression_l1",

        n_estimators=n_estimators,

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


# =========================================================
# Inner validation
# =========================================================

def find_best_iteration(
    df,
    prediction_start
):

    purge = pd.Timedelta(
        minutes=PURGE_MINUTES
    )

    available_cutoff = (
        prediction_start
        -
        purge
    )

    validation_start = (
        prediction_start
        -
        pd.Timedelta(
            days=INTERNAL_VALIDATION_DAYS
        )
    )

    fit_cutoff = (
        validation_start
        -
        purge
    )

    fit_df = df[
        df["timestamp"]
        <
        fit_cutoff
    ].copy()

    validation_df = df[
        (
            df["timestamp"]
            >= validation_start
        )
        &
        (
            df["timestamp"]
            <
            available_cutoff
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

    model = build_model(
        n_estimators=500
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

    if (
        best_iteration is None
        or
        best_iteration <= 0
    ):

        best_iteration = 1

    return {
        "best_iteration":
            int(best_iteration),

        "inner_train_rows":
            len(fit_df),

        "inner_validation_rows":
            len(validation_df),

        "inner_lower_bound":
            lower,

        "inner_upper_bound":
            upper
    }


# =========================================================
# Final fold model
# =========================================================

def train_fold_model(
    df,
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

    train_df = df[
        df["timestamp"]
        <
        train_cutoff
    ].copy()

    lower, upper = (
        calculate_target_bounds(
            train_df
        )
    )

    target = (
        create_normalized_target(
            train_df,
            lower,
            upper
        )
    )

    model = build_model(
        n_estimators=best_iteration
    )

    model.fit(
        train_df[
            FEATURE_COLUMNS
        ],
        target
    )

    return (
        model,
        train_df,
        lower,
        upper
    )


# =========================================================
# 2026 walk-forward
# =========================================================

def get_prediction_months(df):

    test_df = df[
        df["timestamp"]
        >=
        TEST_START
    ]

    if len(test_df) == 0:

        raise ValueError(
            "No 2026 test data found."
        )

    latest_timestamp = (
        test_df["timestamp"]
        .max()
    )

    latest_month = (
        latest_timestamp
        .to_period("M")
        .start_time
        .tz_localize("UTC")
    )

    months = pd.date_range(
        start=TEST_START,
        end=latest_month,
        freq="MS",
        tz="UTC"
    )

    return (
        months,
        latest_timestamp
    )


def predict_fold(
    df,
    prediction_start,
    latest_timestamp
):

    prediction_end = (
        prediction_start
        +
        pd.offsets.MonthBegin(1)
    )

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
        &
        (
            df["timestamp"]
            <= latest_timestamp
        )
    ].copy()

    if len(month_df) == 0:

        return (
            None,
            None
        )

    inner_result = (
        find_best_iteration(
            df,
            prediction_start
        )
    )

    best_iteration = (
        inner_result[
            "best_iteration"
        ]
    )

    (
        model,
        train_df,
        lower,
        upper
    ) = train_fold_model(
        df,
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

    fold_info = {

        "prediction_month":
            prediction_start.strftime(
                "%Y-%m"
            ),

        "train_rows":
            len(train_df),

        "best_iteration":
            best_iteration,

        "target_lower_bound":
            lower,

        "target_upper_bound":
            upper,

        "inner_train_rows":
            inner_result[
                "inner_train_rows"
            ],

        "inner_validation_rows":
            inner_result[
                "inner_validation_rows"
            ]
    }

    print(
        "\n=========================="
    )

    print(
        "Prediction month:",
        prediction_start.strftime(
            "%Y-%m"
        )
    )

    print(
        "Train rows:",
        len(train_df)
    )

    print(
        "Best iteration:",
        best_iteration
    )

    print(
        "Prediction rows:",
        len(month_df)
    )

    return (
        month_df,
        fold_info
    )


def run_walkforward(df):

    (
        months,
        latest_timestamp
    ) = get_prediction_months(
        df
    )

    print(
        "Latest test timestamp:",
        latest_timestamp
    )

    predictions = []
    fold_rows = []

    for month_start in months:

        (
            prediction_df,
            fold_info
        ) = predict_fold(
            df,
            month_start,
            latest_timestamp
        )

        if prediction_df is None:
            continue

        predictions.append(
            prediction_df
        )

        fold_rows.append(
            fold_info
        )

    result = pd.concat(
        predictions,
        ignore_index=True
    )

    result = (
        result.sort_values(
            "timestamp"
        )
        .reset_index(drop=True)
    )

    folds = pd.DataFrame(
        fold_rows
    )

    return (
        result,
        folds,
        latest_timestamp
    )


# =========================================================
# Non-overlapping 15-minute samples
# =========================================================

def select_non_overlap(df):

    mask = (
        df["timestamp"]
        .dt.minute
        .mod(15)
        ==
        0
    )

    return (
        df[mask]
        .copy()
        .reset_index(drop=True)
    )


# =========================================================
# Prediction threshold
# =========================================================

def add_past_only_threshold(
    history_predictions,
    test_predictions
):

    history = (
        history_predictions[
            [
                "timestamp",
                "prediction"
            ]
        ]
        .copy()
    )

    history[
        "is_test"
    ] = False

    test = (
        test_predictions.copy()
    )

    test[
        "is_test"
    ] = True

    combined = pd.concat(
        [
            history,
            test
        ],
        ignore_index=True,
        sort=False
    )

    combined = (
        combined.sort_values(
            "timestamp"
        )
        .drop_duplicates(
            subset=["timestamp"],
            keep="last"
        )
        .reset_index(drop=True)
    )

    past_prediction = (
        combined["prediction"]
        .shift(1)
    )

    combined[
        "long_threshold"
    ] = (
        past_prediction
        .rolling(
            window=ROLLING_WINDOW,
            min_periods=MIN_HISTORY
        )
        .quantile(
            1 - TAIL
        )
    )

    return combined


# =========================================================
# Signal + position
# =========================================================

def add_position(df):

    df = df.copy()

    valid = (
        df[
            "long_threshold"
        ]
        .notna()
    )

    df[
        "position"
    ] = np.where(
        valid
        &
        (
            df["prediction"]
            >=
            df["long_threshold"]
        ),
        1,
        0
    )

    previous_position = (
        df["position"]
        .shift(1)
        .fillna(0)
    )

    df[
        "position_change"
    ] = (
        df["position"]
        -
        previous_position
    )

    df[
        "turnover"
    ] = (
        df[
            "position_change"
        ]
        .abs()
    )

    return df


# =========================================================
# Returns
# =========================================================

def add_test_returns(df):

    df = df.copy()

    # Only evaluate rows belonging to 2026 test.
    test_mask = (
        df["is_test"]
        ==
        True
    )

    df[
        "gross_return"
    ] = 0.0

    df.loc[
        test_mask,
        "gross_return"
    ] = (
        df.loc[
            test_mask,
            "position"
        ]
        *
        df.loc[
            test_mask,
            RAW_TARGET
        ]
    )

    df[
        "transaction_cost"
    ] = 0.0

    df.loc[
        test_mask,
        "transaction_cost"
    ] = (
        df.loc[
            test_mask,
            "turnover"
        ]
        *
        ONE_WAY_COST
    )

    df[
        "net_return"
    ] = (
        df[
            "gross_return"
        ]
        -
        df[
            "transaction_cost"
        ]
    )

    return df


# =========================================================
# Metrics
# =========================================================

def max_drawdown(returns):

    if len(returns) == 0:
        return np.nan

    equity = (
        1
        +
        returns
    ).cumprod()

    peak = (
        equity.cummax()
    )

    dd = (
        equity
        /
        peak
        - 1
    )

    return dd.min()


def calculate_summary(test_df):

    active = test_df[
        test_df["position"] == 1
    ].copy()

    returns = (
        test_df["net_return"]
    )

    entries = (
        test_df[
            "position_change"
        ]
        .clip(lower=0)
        .sum()
    )

    exits = (
        -test_df[
            "position_change"
        ]
        .clip(upper=0)
        .sum()
    )

    if len(active) > 0:

        avg_gross_bps = (
            active[
                "gross_return"
            ].mean()
            * 10000
        )

        # Allocate all test costs across active exposure
        # for an intuitive per-active-period number.
        net_edge_bps = (
            (
                test_df[
                    "gross_return"
                ].sum()
                -
                test_df[
                    "transaction_cost"
                ].sum()
            )
            /
            len(active)
            *
            10000
        )

        gross_hit_rate = (
            active[
                "gross_return"
            ]
            > 0
        ).mean()

    else:

        avg_gross_bps = np.nan
        net_edge_bps = np.nan
        gross_hit_rate = np.nan

    if (
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

    total_return = (
        (
            1
            +
            returns
        )
        .prod()
        - 1
    )

    return {

        "test_rows":
            len(test_df),

        "active_periods":
            len(active),

        "entries":
            entries,

        "exits":
            exits,

        "signal_frequency":
            (
                len(active)
                /
                len(test_df)
            ),

        "avg_gross_active_bps":
            avg_gross_bps,

        "effective_net_edge_bps":
            net_edge_bps,

        "gross_hit_rate":
            gross_hit_rate,

        "total_cost_bps":
            (
                test_df[
                    "transaction_cost"
                ].sum()
                * 10000
            ),

        "sharpe_ratio":
            sharpe,

        "max_drawdown":
            max_drawdown(
                returns
            ),

        "net_total_return":
            total_return
    }


# =========================================================
# IC
# =========================================================

def calculate_ic(test_df):

    valid = test_df[
        [
            "prediction",
            RAW_TARGET
        ]
    ].dropna()

    if len(valid) < 2:

        return (
            np.nan,
            np.nan
        )

    pearson = (
        valid[
            "prediction"
        ]
        .corr(
            valid[
                RAW_TARGET
            ],
            method="pearson"
        )
    )

    spearman = (
        valid[
            "prediction"
        ]
        .corr(
            valid[
                RAW_TARGET
            ],
            method="spearman"
        )
    )

    return (
        pearson,
        spearman
    )


# =========================================================
# Monthly results
# =========================================================

def calculate_monthly(test_df):

    temp = (
        test_df.copy()
    )

    temp[
        "month"
    ] = (
        temp[
            "timestamp"
        ]
        .dt.strftime(
            "%Y-%m"
        )
    )

    rows = []

    for month, group in (
        temp.groupby("month")
    ):

        active = group[
            group["position"] == 1
        ]

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

        (
            pearson_ic,
            spearman_ic
        ) = calculate_ic(
            group
        )

        if len(active) > 0:

            avg_gross_bps = (
                active[
                    "gross_return"
                ].mean()
                * 10000
            )

            hit_rate = (
                active[
                    "gross_return"
                ]
                > 0
            ).mean()

        else:

            avg_gross_bps = np.nan
            hit_rate = np.nan

        rows.append({

            "month":
                month,

            "rows":
                len(group),

            "active_periods":
                len(active),

            "signal_frequency":
                (
                    len(active)
                    /
                    len(group)
                ),

            "avg_gross_bps":
                avg_gross_bps,

            "gross_hit_rate":
                hit_rate,

            "pearson_ic":
                pearson_ic,

            "spearman_ic":
                spearman_ic,

            "transaction_cost_bps":
                (
                    group[
                        "transaction_cost"
                    ].sum()
                    * 10000
                ),

            "monthly_return":
                monthly_return
        })

    return pd.DataFrame(
        rows
    )


# =========================================================
# Main
# =========================================================

def main():

    print(
        "=================================="
    )

    print(
        "FINAL UNTOUCHED 2026 TEST"
    )

    print(
        "=================================="
    )

    df = load_feature_data()

    print(
        "Feature rows:",
        len(df)
    )

    print(
        "Feature data end:",
        df["timestamp"].max()
    )

    history_predictions = (
        load_prediction_history()
    )

    print(
        "Historical threshold rows:",
        len(history_predictions)
    )

    # ---------------------------------------------
    # 2026 monthly walk-forward
    # ---------------------------------------------

    (
        predictions,
        folds,
        latest_timestamp
    ) = run_walkforward(
        df
    )

    predictions = (
        select_non_overlap(
            predictions
        )
    )

    print(
        "\n2026 non-overlap rows:",
        len(predictions)
    )

    # ---------------------------------------------
    # Combine with historical predictions
    # for threshold warm-up.
    # ---------------------------------------------

    combined = (
        add_past_only_threshold(
            history_predictions,
            predictions
        )
    )

    # Add raw target/features back to historical/test rows
    # where available.
    lookup_columns = (
        ["timestamp", RAW_TARGET]
        +
        FEATURE_COLUMNS
    )

    lookup = (
        df[
            lookup_columns
        ]
        .drop_duplicates(
            subset=["timestamp"]
        )
    )

    combined = (
        combined.drop(
            columns=[
                column
                for column
                in (
                    [RAW_TARGET]
                    +
                    FEATURE_COLUMNS
                )
                if column
                in combined.columns
            ],
            errors="ignore"
        )
        .merge(
            lookup,
            on="timestamp",
            how="left"
        )
    )

    # ---------------------------------------------
    # Position including continuity across
    # 2025 -> 2026 boundary.
    # ---------------------------------------------

    combined = (
        add_position(
            combined
        )
    )

    combined = (
        add_test_returns(
            combined
        )
    )

    test_df = combined[
        combined["is_test"] == True
    ].copy()

    test_df = (
        test_df.sort_values(
            "timestamp"
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------
    # Overall IC
    # ---------------------------------------------

    (
        pearson_ic,
        spearman_ic
    ) = calculate_ic(
        test_df
    )

    # ---------------------------------------------
    # Overall strategy performance
    # ---------------------------------------------

    summary = (
        calculate_summary(
            test_df
        )
    )

    summary[
        "pearson_ic"
    ] = pearson_ic

    summary[
        "spearman_ic"
    ] = spearman_ic

    summary[
        "test_start"
    ] = (
        test_df[
            "timestamp"
        ].min()
    )

    summary[
        "test_end"
    ] = (
        test_df[
            "timestamp"
        ].max()
    )

    summary[
        "test_end_is_partial_month"
    ] = (
        latest_timestamp
        <
        (
            latest_timestamp
            .to_period("M")
            .end_time
            .tz_localize("UTC")
        )
    )

    summary_df = pd.DataFrame(
        [summary]
    )

    # ---------------------------------------------
    # Monthly
    # ---------------------------------------------

    monthly_df = (
        calculate_monthly(
            test_df
        )
    )

    positive_month_ratio = (
        monthly_df[
            "monthly_return"
        ]
        > 0
    ).mean()

    summary_df[
        "positive_month_ratio"
    ] = (
        positive_month_ratio
    )

    # ---------------------------------------------
    # Output
    # ---------------------------------------------

    print(
        "\n=================================="
    )

    print(
        "=== FINAL 2026 SUMMARY ==="
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        "\n=================================="
    )

    print(
        "=== 2026 MONTHLY RESULTS ==="
    )

    print(
        monthly_df.to_string(
            index=False
        )
    )

    print(
        "\n=================================="
    )

    print(
        "=== WALK-FORWARD FOLDS ==="
    )

    print(
        folds.to_string(
            index=False
        )
    )

    # ---------------------------------------------
    # Save
    # ---------------------------------------------

    PREDICTION_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    test_df.to_csv(
        PREDICTION_OUTPUT,
        index=False
    )

    folds.to_csv(
        FOLD_OUTPUT,
        index=False
    )

    monthly_df.to_csv(
        MONTHLY_OUTPUT,
        index=False
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT,
        index=False
    )

    print(
        "\nSaved:"
    )

    print(
        PREDICTION_OUTPUT
    )

    print(
        FOLD_OUTPUT
    )

    print(
        MONTHLY_OUTPUT
    )

    print(
        SUMMARY_OUTPUT
    )


if __name__ == "__main__":
    main()