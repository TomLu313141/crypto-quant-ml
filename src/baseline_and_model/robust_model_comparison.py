from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from scipy.stats import spearmanr

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


DATA_PATH = Path(
    "data/processed/btc_usdt_5m_features.csv"
)

OUTPUT_PATH = Path(
    "data/processed/robust_model_comparison.csv"
)

BEST_PREDICTION_PATH = Path(
    "data/processed/validation_predictions_robust.csv"
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


MODEL_CONFIGS = {
    "l2": {
        "objective": "regression"
    },

    "l1": {
        "objective": "regression_l1"
    },

    "huber": {
        "objective": "huber",
        "alpha": 0.9
    }
}


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


def select_non_overlapping(df):

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


def train_model(
    train,
    validation,
    config
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

    parameters = {
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,

        "subsample": 0.8,
        "colsample_bytree": 0.8,

        "reg_alpha": 0.1,
        "reg_lambda": 0.1,

        "random_state": 42,
        "n_jobs": -1,

        "force_row_wise": True
    }

    parameters.update(
        config
    )

    model = lgb.LGBMRegressor(
        **parameters
    )

    model.fit(
        X_train,
        y_train,

        eval_X=X_val,
        eval_y=y_val,

        callbacks=[
            lgb.early_stopping(
                stopping_rounds=50,
                verbose=False
            )
        ]
    )

    return model


def calculate_decile_spread(
    y_true,
    y_pred
):

    temp = pd.DataFrame({
        "prediction": y_pred,
        "target": y_true
    })

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
        temp.groupby("decile")[
            "target"
        ]
        .mean()
    )

    bottom_return = grouped.loc[0]

    top_return = grouped.loc[9]

    spread = (
        top_return
        -
        bottom_return
    )

    return (
        bottom_return,
        top_return,
        spread
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

    pearson_ic = np.corrcoef(
        y_true,
        y_pred
    )[0, 1]

    spearman_ic, _ = spearmanr(
        y_true,
        y_pred
    )

    (
        bottom_return,
        top_return,
        decile_spread
    ) = calculate_decile_spread(
        y_true,
        y_pred
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "pearson_ic": pearson_ic,
        "spearman_ic": spearman_ic,
        "bottom_decile_return":
            bottom_return,
        "top_decile_return":
            top_return,
        "decile_spread":
            decile_spread
    }


def calculate_trimmed_ic(
    y_true,
    y_pred,
    quantile=0.99
):

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    threshold = np.quantile(
        np.abs(y_true),
        quantile
    )

    mask = (
        np.abs(y_true)
        <= threshold
    )

    trimmed_y = y_true[
        mask
    ]

    trimmed_pred = y_pred[
        mask
    ]

    pearson = np.corrcoef(
        trimmed_y,
        trimmed_pred
    )[0, 1]

    spearman, _ = spearmanr(
        trimmed_y,
        trimmed_pred
    )

    return (
        pearson,
        spearman
    )


def evaluate_model(
    name,
    model,
    validation
):

    X_val = validation[
        FEATURE_COLUMNS
    ]

    y_val = validation[
        TARGET
    ].to_numpy()

    prediction = model.predict(
        X_val
    )

    metrics = calculate_metrics(
        y_val,
        prediction
    )

    (
        trimmed_pearson,
        trimmed_spearman
    ) = calculate_trimmed_ic(
        y_val,
        prediction,
        quantile=0.99
    )

    metrics[
        "trimmed_99_pearson_ic"
    ] = trimmed_pearson

    metrics[
        "trimmed_99_spearman_ic"
    ] = trimmed_spearman

    metrics[
        "best_iteration"
    ] = model.best_iteration_

    print(
        f"\n=== {name.upper()} ==="
    )

    for key, value in metrics.items():

        if isinstance(
            value,
            (float, np.floating)
        ):
            print(
                f"{key}: {value:.8f}"
            )

        else:
            print(
                f"{key}: {value}"
            )

    return (
        metrics,
        prediction
    )


def main():

    df = load_data()

    train, validation = split_data(
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
        "\nTrain period:"
    )

    print(
        train["timestamp"].min(),
        "->",
        train["timestamp"].max()
    )

    print(
        "\nValidation period:"
    )

    print(
        validation["timestamp"].min(),
        "->",
        validation["timestamp"].max()
    )

    results = []

    predictions = {}

    for name, config in (
        MODEL_CONFIGS.items()
    ):

        print(
            "\n=========================="
        )

        print(
            "Training:",
            name
        )

        model = train_model(
            train,
            validation,
            config
        )

        metrics, prediction = (
            evaluate_model(
                name,
                model,
                validation
            )
        )

        row = {
            "model": name,
            **metrics
        }

        results.append(
            row
        )

        predictions[
            name
        ] = prediction

    result_df = pd.DataFrame(
        results
    )

    print(
        "\n=== Model Comparison ==="
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

    # Choose model based primarily on
    # Spearman / robust ranking signal.
    best_model_name = (
        result_df
        .sort_values(
            [
                "trimmed_99_spearman_ic",
                "spearman_ic"
            ],
            ascending=False
        )
        .iloc[0][
            "model"
        ]
    )

    print(
        "\nBest robust model:",
        best_model_name
    )

    prediction_df = validation[
        [
            "timestamp",
            "close",
            TARGET
        ]
    ].copy()

    prediction_df[
        "prediction"
    ] = predictions[
        best_model_name
    ]

    prediction_df.to_csv(
        BEST_PREDICTION_PATH,
        index=False
    )

    print(
        "\nSaved comparison to:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "\nSaved best model predictions to:"
    )

    print(
        BEST_PREDICTION_PATH
    )


if __name__ == "__main__":
    main()