from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from scipy.stats import spearmanr


DATA_PATH = Path(
    "data/processed/btc_usdt_5m_features.csv"
)

OUTPUT_PATH = Path(
    "data/processed/normalized_target_comparison.csv"
)


RAW_TARGET = "future_return_15m"


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


TARGET_CONFIGS = [
    {
        "name": "raw",
        "vol_column": None
    },

    {
        "name": "normalized_vol12",
        "vol_column": "volatility_12"
    },

    {
        "name": "normalized_vol48",
        "vol_column": "volatility_48"
    }
]


def load_data():

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["timestamp"]
    )

    return (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )


def split_data(df):

    train = df[
        df["timestamp"] < "2025-01-01"
    ].copy()

    validation = df[
        (
            df["timestamp"] >= "2025-01-01"
        )
        &
        (
            df["timestamp"] < "2026-01-01"
        )
    ].copy()

    return train, validation


def build_training_target(
    train,
    vol_column
):

    if vol_column is None:

        return train[
            RAW_TARGET
        ].copy()

    denominator = (
        train[
            vol_column
        ]
        .clip(
            lower=1e-6
        )
    )

    target = (
        train[
            RAW_TARGET
        ]
        /
        denominator
    )

    # Prevent extremely small volatility
    # from generating absurd normalized labels.
    lower = target.quantile(
        0.005
    )

    upper = target.quantile(
        0.995
    )

    target = target.clip(
        lower=lower,
        upper=upper
    )

    print(
        "Normalized target bounds:",
        lower,
        upper
    )

    return target


def train_model(
    train,
    validation,
    y_train
):

    model = lgb.LGBMRegressor(
        objective="regression_l1",

        n_estimators=1000,
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
        train[
            FEATURE_COLUMNS
        ],
        y_train,

        eval_X=validation[
            FEATURE_COLUMNS
        ],

        # Early stopping must evaluate
        # against the SAME transformed target.
        eval_y=None,

        callbacks=[
            lgb.early_stopping(
                stopping_rounds=50,
                verbose=False
            )
        ]
        if False
        else None
    )

    return model


def train_model_with_eval(
    train,
    validation,
    train_target,
    validation_target
):

    model = lgb.LGBMRegressor(
        objective="regression_l1",

        n_estimators=300,

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
        train[
            FEATURE_COLUMNS
        ],
        train_target,

        eval_set=[
            (
                validation[
                    FEATURE_COLUMNS
                ],
                validation_target
            )
        ],

        callbacks=[
            lgb.early_stopping(
                stopping_rounds=50,
                verbose=False
            )
        ]
    )

    return model


def build_target_for_period(
    df,
    vol_column,
    bounds=None
):

    if vol_column is None:

        return (
            df[
                RAW_TARGET
            ].copy(),
            bounds
        )

    denominator = (
        df[
            vol_column
        ]
        .clip(
            lower=1e-6
        )
    )

    target = (
        df[
            RAW_TARGET
        ]
        /
        denominator
    )

    if bounds is not None:

        lower, upper = bounds

        target = target.clip(
            lower=lower,
            upper=upper
        )

    return target, bounds


def create_targets(
    train,
    validation,
    vol_column
):

    if vol_column is None:

        return (
            train[
                RAW_TARGET
            ].copy(),

            validation[
                RAW_TARGET
            ].copy()
        )

    train_denominator = (
        train[
            vol_column
        ]
        .clip(
            lower=1e-6
        )
    )

    train_target = (
        train[
            RAW_TARGET
        ]
        /
        train_denominator
    )

    lower = (
        train_target
        .quantile(0.005)
    )

    upper = (
        train_target
        .quantile(0.995)
    )

    train_target = (
        train_target.clip(
            lower=lower,
            upper=upper
        )
    )

    validation_denominator = (
        validation[
            vol_column
        ]
        .clip(
            lower=1e-6
        )
    )

    validation_target = (
        validation[
            RAW_TARGET
        ]
        /
        validation_denominator
    )

    # IMPORTANT:
    # use TRAIN bounds only
    validation_target = (
        validation_target.clip(
            lower=lower,
            upper=upper
        )
    )

    print(
        "Train target bounds:",
        lower,
        upper
    )

    return (
        train_target,
        validation_target
    )


def select_non_overlap(df):

    mask = (
        df["timestamp"].dt.minute
        % 15
        == 0
    )

    return (
        df[mask]
        .copy()
        .reset_index(drop=True)
    )


def calculate_decile_metrics(df):

    temp = df.copy()

    ranks = (
        temp["prediction"]
        .rank(method="first")
    )

    temp["decile"] = pd.qcut(
        ranks,
        q=10,
        labels=False
    )

    grouped = (
        temp.groupby(
            "decile"
        )[RAW_TARGET]
        .mean()
    )

    bottom = (
        grouped.loc[0]
    )

    top = (
        grouped.loc[9]
    )

    spread = (
        top - bottom
    )

    return (
        bottom,
        top,
        spread
    )


def calculate_tail_metrics(df):

    low_threshold = (
        df["prediction"]
        .quantile(0.01)
    )

    high_threshold = (
        df["prediction"]
        .quantile(0.99)
    )

    bottom = df[
        df["prediction"]
        <= low_threshold
    ]

    top = df[
        df["prediction"]
        >= high_threshold
    ]

    short_returns = (
        -bottom[
            RAW_TARGET
        ]
    )

    long_returns = (
        top[
            RAW_TARGET
        ]
    )

    return {
        "bottom1_short_bps":
            short_returns.mean()
            * 10000,

        "bottom1_short_hit":
            (
                short_returns > 0
            ).mean(),

        "top1_long_bps":
            long_returns.mean()
            * 10000,

        "top1_long_hit":
            (
                long_returns > 0
            ).mean()
    }


def calculate_monthly_ic(df):

    temp = df.copy()

    temp["month"] = (
        temp["timestamp"]
        .dt.strftime("%Y-%m")
    )

    values = []

    for _, group in temp.groupby(
        "month"
    ):

        ic, _ = spearmanr(
            group[
                "prediction"
            ],

            group[
                RAW_TARGET
            ]
        )

        values.append(
            ic
        )

    values = np.array(
        values
    )

    return {
        "monthly_spearman_mean":
            values.mean(),

        "monthly_spearman_std":
            values.std(),

        "positive_month_ratio":
            (
                values > 0
            ).mean()
    }


def winner_removal_test(df):

    threshold = (
        df["prediction"]
        .quantile(0.99)
    )

    top = (
        df[
            df["prediction"]
            >= threshold
        ][
            RAW_TARGET
        ]
        .sort_values(
            ascending=False
        )
        .reset_index(drop=True)
    )

    results = {}

    for n in [
        0,
        1,
        3,
        5
    ]:

        remaining = (
            top.iloc[n:]
        )

        results[
            f"top1_remove_{n}_bps"
        ] = (
            remaining.mean()
            * 10000
        )

    return results


def evaluate_config(
    config,
    train,
    validation
):

    print(
        "\n========================"
    )

    print(
        "Target:",
        config["name"]
    )

    (
        train_target,
        validation_target
    ) = create_targets(
        train,
        validation,
        config[
            "vol_column"
        ]
    )

    model = (
        train_model_with_eval(
            train,
            validation,
            train_target,
            validation_target
        )
    )

    temp = (
        validation.copy()
    )

    temp[
        "prediction"
    ] = model.predict(
        temp[
            FEATURE_COLUMNS
        ]
    )

    temp = (
        select_non_overlap(
            temp
        )
    )

    pearson = np.corrcoef(
        temp[
            "prediction"
        ],

        temp[
            RAW_TARGET
        ]
    )[0, 1]

    spearman, _ = spearmanr(
        temp[
            "prediction"
        ],

        temp[
            RAW_TARGET
        ]
    )

    (
        bottom,
        top,
        spread
    ) = calculate_decile_metrics(
        temp
    )

    tail_metrics = (
        calculate_tail_metrics(
            temp
        )
    )

    monthly_metrics = (
        calculate_monthly_ic(
            temp
        )
    )

    robustness = (
        winner_removal_test(
            temp
        )
    )

    print(
        f"Pearson IC: "
        f"{pearson:.8f}"
    )

    print(
        f"Spearman IC: "
        f"{spearman:.8f}"
    )

    print(
        f"Decile spread: "
        f"{spread * 10000:.4f} bps"
    )

    print(
        "Bottom 1% short:",
        f"{tail_metrics['bottom1_short_bps']:.4f} bps"
    )

    print(
        "Top 1% long:",
        f"{tail_metrics['top1_long_bps']:.4f} bps"
    )

    print(
        "Positive months:",
        f"{monthly_metrics['positive_month_ratio']:.2%}"
    )

    for key, value in (
        robustness.items()
    ):

        print(
            f"{key}: "
            f"{value:.4f}"
        )

    return {
        "target":
            config["name"],

        "best_iteration":
            model.best_iteration_,

        "pearson_ic":
            pearson,

        "spearman_ic":
            spearman,

        "bottom_decile_bps":
            bottom * 10000,

        "top_decile_bps":
            top * 10000,

        "decile_spread_bps":
            spread * 10000,

        **tail_metrics,
        **monthly_metrics,
        **robustness
    }


def main():

    df = load_data()

    train, validation = (
        split_data(
            df
        )
    )

    results = []

    for config in (
        TARGET_CONFIGS
    ):

        result = (
            evaluate_config(
                config,
                train,
                validation
            )
        )

        results.append(
            result
        )

    result_df = pd.DataFrame(
        results
    )

    print(
        "\n========================"
    )

    print(
        "=== Normalized Target Comparison ==="
    )

    print(
        result_df.to_string(
            index=False
        )
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result_df.to_csv(
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