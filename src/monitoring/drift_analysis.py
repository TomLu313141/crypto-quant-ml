from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path(
    "data/processed/position_based_backtest.csv"
)

OUTPUT_PATH = Path(
    "data/processed/model_drift_analysis.csv"
)


DRIFT_COLUMNS = [
    "prediction",
    "volatility_48",
    "volume_ratio_48",
    "return_12",
    "price_ma_ratio_48"
]


PSI_BINS = 10

EPSILON = 1e-6


def load_data():

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["timestamp"]
    )

    return (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )


def calculate_psi(
    reference,
    current,
    bins=10
):

    reference = (
        pd.Series(reference)
        .dropna()
        .to_numpy()
    )

    current = (
        pd.Series(current)
        .dropna()
        .to_numpy()
    )

    if (
        len(reference) == 0
        or
        len(current) == 0
    ):

        return np.nan

    # Bin boundaries are created from
    # REFERENCE distribution only.
    quantiles = np.linspace(
        0,
        1,
        bins + 1
    )

    edges = np.quantile(
        reference,
        quantiles
    )

    edges = np.unique(
        edges
    )

    if len(edges) < 3:

        return np.nan

    # Ensure future values outside the
    # reference range still fall into bins.
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(
        reference,
        bins=edges
    )

    current_counts, _ = np.histogram(
        current,
        bins=edges
    )

    ref_pct = (
        ref_counts
        /
        ref_counts.sum()
    )

    current_pct = (
        current_counts
        /
        current_counts.sum()
    )

    ref_pct = np.clip(
        ref_pct,
        EPSILON,
        None
    )

    current_pct = np.clip(
        current_pct,
        EPSILON,
        None
    )

    psi = np.sum(
        (
            current_pct
            -
            ref_pct
        )
        *
        np.log(
            current_pct
            /
            ref_pct
        )
    )

    return psi


def calculate_monthly_summary(df):

    temp = df.copy()

    temp["month"] = (
        temp["timestamp"]
        .dt.strftime("%Y-%m")
    )

    rows = []

    for month, group in temp.groupby(
        "month"
    ):

        active = group[
            group["position"] == 1
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

            avg_signal_gross_bps = (
                active[
                    "gross_return"
                ].mean()
                * 10000
            )

            signal_hit_rate = (
                active[
                    "gross_return"
                ]
                > 0
            ).mean()

        else:

            avg_signal_gross_bps = np.nan
            signal_hit_rate = np.nan

        rows.append({
            "month":
                month,

            "rows":
                len(group),

            "active_periods":
                len(active),

            "signal_frequency":
                len(active)
                /
                len(group),

            "prediction_mean":
                group[
                    "prediction"
                ].mean(),

            "prediction_std":
                group[
                    "prediction"
                ].std(),

            "prediction_p01":
                group[
                    "prediction"
                ].quantile(
                    0.01
                ),

            "prediction_p99":
                group[
                    "prediction"
                ].quantile(
                    0.99
                ),

            "avg_signal_gross_bps":
                avg_signal_gross_bps,

            "signal_hit_rate":
                signal_hit_rate,

            "monthly_return":
                monthly_return
        })

    return pd.DataFrame(
        rows
    )


def calculate_monthly_psi(df):

    temp = df.copy()

    temp["month"] = (
        temp["timestamp"]
        .dt.strftime("%Y-%m")
    )

    months = sorted(
        temp["month"]
        .unique()
    )

    rows = []

    # Expanding reference:
    # every month's drift is measured
    # against ALL PREVIOUS months.
    for i, month in enumerate(
        months
    ):

        if i == 0:

            continue

        current = temp[
            temp["month"]
            == month
        ]

        reference = temp[
            temp["month"]
            <
            month
        ]

        row = {
            "month":
                month
        }

        for column in DRIFT_COLUMNS:

            row[
                f"{column}_psi"
            ] = calculate_psi(
                reference[
                    column
                ],
                current[
                    column
                ],
                bins=PSI_BINS
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def calculate_feature_monthly_stats(
    df
):

    temp = df.copy()

    temp["month"] = (
        temp["timestamp"]
        .dt.strftime("%Y-%m")
    )

    feature_columns = [
        "volatility_48",
        "volume_ratio_48",
        "return_12",
        "price_ma_ratio_48"
    ]

    rows = []

    for month, group in temp.groupby(
        "month"
    ):

        row = {
            "month": month
        }

        for column in feature_columns:

            row[
                f"{column}_mean"
            ] = (
                group[column]
                .mean()
            )

            row[
                f"{column}_std"
            ] = (
                group[column]
                .std()
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )

def interpret_psi(
    value
):

    if pd.isna(
        value
    ):

        return "NA"

    if value < 0.10:

        return "stable"

    if value < 0.25:

        return "moderate"

    return "large"


def main():

    df = load_data()

    print(
        "Rows:",
        len(df)
    )

    monthly_summary = (
        calculate_monthly_summary(
            df
        )
    )

    psi_df = (
        calculate_monthly_psi(
            df
        )
    )

    feature_stats = (
        calculate_feature_monthly_stats(
            df
        )
    )

    result = (
        monthly_summary
        .merge(
            psi_df,
            on="month",
            how="left"
        )
        .merge(
            feature_stats,
            on="month",
            how="left"
        )
    )

    # -----------------------------------
    # Aggregate PSI
    # -----------------------------------

    psi_columns = [
        f"{column}_psi"
        for column
        in DRIFT_COLUMNS
    ]

    result[
        "mean_psi"
    ] = (
        result[
            psi_columns
        ]
        .mean(
            axis=1
        )
    )

    result[
        "max_psi"
    ] = (
        result[
            psi_columns
        ]
        .max(
            axis=1
        )
    )

    result[
        "max_psi_level"
    ] = (
        result[
            "max_psi"
        ]
        .apply(
            interpret_psi
        )
    )

    # -----------------------------------
    # Print summary
    # -----------------------------------

    print(
        "\n=========================="
    )

    print(
        "=== Monthly Drift Summary ==="
    )

    display_columns = [
        "month",

        "active_periods",
        "signal_frequency",

        "avg_signal_gross_bps",
        "signal_hit_rate",
        "monthly_return",

        "prediction_mean",
        "prediction_std",

        "prediction_psi",
        "volatility_48_psi",
        "volume_ratio_48_psi",
        "return_12_psi",
        "price_ma_ratio_48_psi",

        "mean_psi",
        "max_psi",
        "max_psi_level"
    ]

    print(
        result[
            display_columns
        ].to_string(
            index=False
        )
    )

    # -----------------------------------
    # Worst signal months
    # -----------------------------------

    print(
        "\n=========================="
    )

    print(
        "=== Worst Signal Months ==="
    )

    worst = (
        result.sort_values(
            "avg_signal_gross_bps"
        )
        .head(5)
    )

    print(
        worst[
            display_columns
        ].to_string(
            index=False
        )
    )

    # -----------------------------------
    # Highest drift months
    # -----------------------------------

    print(
        "\n=========================="
    )

    print(
        "=== Highest Drift Months ==="
    )

    highest_drift = (
        result.dropna(
            subset=[
                "max_psi"
            ]
        )
        .sort_values(
            "max_psi",
            ascending=False
        )
        .head(5)
    )

    print(
        highest_drift[
            display_columns
        ].to_string(
            index=False
        )
    )

    # -----------------------------------
    # Drift vs performance correlation
    # -----------------------------------

    print(
        "\n=========================="
    )

    print(
        "=== Drift vs Signal Performance ==="
    )

    valid = result.dropna(
        subset=[
            "mean_psi",
            "avg_signal_gross_bps"
        ]
    )

    pearson = (
        valid[
            [
                "mean_psi",
                "avg_signal_gross_bps"
            ]
        ]
        .corr()
        .iloc[
            0,
            1
        ]
    )

    spearman = (
        valid[
            [
                "mean_psi",
                "avg_signal_gross_bps"
            ]
        ]
        .corr(
            method="spearman"
        )
        .iloc[
            0,
            1
        ]
    )

    print(
        "Pearson:",
        pearson
    )

    print(
        "Spearman:",
        spearman
    )

    # -----------------------------------
    # Save
    # -----------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
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