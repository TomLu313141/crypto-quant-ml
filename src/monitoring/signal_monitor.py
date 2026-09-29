from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path(
    "data/processed/position_based_backtest.csv"
)

OUTPUT_PATH = Path(
    "data/processed/signal_performance_monitor_v3.csv"
)


# =========================================================
# Monitoring settings
# =========================================================

ROLLING_DAYS = 30
PERIODS_PER_DAY = 24 * 4

ROLLING_WINDOW = (
    ROLLING_DAYS
    * PERIODS_PER_DAY
)


# =========================================================
# Sample sufficiency
# =========================================================

VERY_LOW_SAMPLE_THRESHOLD = 10
LOW_SAMPLE_THRESHOLD = 20


# =========================================================
# Warning thresholds
# =========================================================

WARNING_AVG_GROSS_BPS = 0.0
WARNING_HIT_RATE = 0.48

WARNING_CURRENT_DD = -0.03
WARNING_MAX_DD_30D = -0.05

SIGNAL_FREQ_WARNING_LOW = 0.50
SIGNAL_FREQ_WARNING_HIGH = 2.00


# =========================================================
# Critical thresholds
# =========================================================

CRITICAL_AVG_GROSS_BPS = -1.0
CRITICAL_HIT_RATE = 0.45

CRITICAL_CURRENT_DD = -0.05
CRITICAL_MAX_DD_30D = -0.08

SIGNAL_FREQ_CRITICAL_LOW = 0.25
SIGNAL_FREQ_CRITICAL_HIGH = 3.00


# =========================================================
# Recovery thresholds
# =========================================================

RECOVERY_AVG_GROSS_BPS = 1.0
RECOVERY_HIT_RATE = 0.52
RECOVERY_CURRENT_DD = -0.015


# =========================================================
# Load data
# =========================================================

def load_data():

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["timestamp"]
    )

    return (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )


# =========================================================
# Basic columns
# =========================================================

def add_basic_columns(df):

    df = df.copy()

    df["is_active"] = (
        df["position"] != 0
    ).astype(int)

    df["is_win"] = np.where(
        df["position"] != 0,
        (
            df["gross_return"] > 0
        ).astype(float),
        np.nan
    )

    return df


# =========================================================
# Rolling metrics
# =========================================================

def add_rolling_metrics(df):

    df = df.copy()

    df["rolling_signal_count"] = (
        df["is_active"]
        .rolling(
            ROLLING_WINDOW,
            min_periods=1
        )
        .sum()
    )

    df["rolling_signal_frequency"] = (
        df["is_active"]
        .rolling(
            ROLLING_WINDOW,
            min_periods=1
        )
        .mean()
    )

    active_gross = (
        df["gross_return"]
        .where(
            df["position"] != 0
        )
    )

    df["rolling_avg_gross_return"] = (
        active_gross
        .rolling(
            ROLLING_WINDOW,
            min_periods=1
        )
        .mean()
    )

    df["rolling_avg_gross_bps"] = (
        df["rolling_avg_gross_return"]
        * 10000
    )

    df["rolling_hit_rate"] = (
        df["is_win"]
        .rolling(
            ROLLING_WINDOW,
            min_periods=1
        )
        .mean()
    )

    return df


# =========================================================
# Rolling return
# =========================================================

def add_rolling_return(df):

    df = df.copy()

    log_return = (
        np.log1p(
            df["net_return"]
        )
    )

    rolling_log_return = (
        log_return
        .rolling(
            ROLLING_WINDOW,
            min_periods=1
        )
        .sum()
    )

    df["rolling_30d_return"] = (
        np.exp(
            rolling_log_return
        )
        - 1
    )

    return df


# =========================================================
# Drawdown
# =========================================================

def calculate_max_drawdown(returns):

    if len(returns) == 0:
        return np.nan

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

    return drawdown.min()


def add_drawdown_metrics(df):

    df = df.copy()

    df["equity"] = (
        1
        +
        df["net_return"]
    ).cumprod()

    rolling_peak = (
        df["equity"]
        .rolling(
            ROLLING_WINDOW,
            min_periods=1
        )
        .max()
    )

    df["current_drawdown_30d"] = (
        df["equity"]
        /
        rolling_peak
        - 1
    )

    df["rolling_max_drawdown_30d"] = (
        df["net_return"]
        .rolling(
            ROLLING_WINDOW,
            min_periods=2
        )
        .apply(
            calculate_max_drawdown,
            raw=False
        )
    )

    return df


# =========================================================
# Frequency baseline
# =========================================================

def add_frequency_baseline(df):

    df = df.copy()

    df["historical_signal_frequency"] = (
        df["is_active"]
        .shift(1)
        .expanding(
            min_periods=1
        )
        .mean()
    )

    denominator = (
        df["historical_signal_frequency"]
        .replace(
            0,
            np.nan
        )
    )

    df["signal_frequency_ratio"] = (
        df["rolling_signal_frequency"]
        /
        denominator
    )

    return df


# =========================================================
# Sample status
# =========================================================

def determine_sample_status(signal_count):

    if pd.isna(signal_count):
        return "NO_DATA"

    if signal_count < VERY_LOW_SAMPLE_THRESHOLD:
        return "VERY_LOW_SAMPLE"

    if signal_count < LOW_SAMPLE_THRESHOLD:
        return "LOW_SAMPLE"

    return "SUFFICIENT"


# =========================================================
# Alert reasons
# =========================================================

def get_warning_reasons(row):

    reasons = []

    avg_bps = row["rolling_avg_gross_bps"]
    hit_rate = row["rolling_hit_rate"]

    current_dd = row[
        "current_drawdown_30d"
    ]

    max_dd = row[
        "rolling_max_drawdown_30d"
    ]

    freq_ratio = row[
        "signal_frequency_ratio"
    ]

    if (
        pd.notna(avg_bps)
        and
        avg_bps < WARNING_AVG_GROSS_BPS
    ):
        reasons.append(
            "NEGATIVE_EDGE"
        )

    if (
        pd.notna(hit_rate)
        and
        hit_rate < WARNING_HIT_RATE
    ):
        reasons.append(
            "LOW_HIT_RATE"
        )

    if (
        pd.notna(current_dd)
        and
        current_dd < WARNING_CURRENT_DD
    ):
        reasons.append(
            "CURRENT_DRAWDOWN"
        )

    if (
        pd.notna(max_dd)
        and
        max_dd < WARNING_MAX_DD_30D
    ):
        reasons.append(
            "MAX_DRAWDOWN"
        )

    if (
        pd.notna(freq_ratio)
        and
        (
            freq_ratio
            < SIGNAL_FREQ_WARNING_LOW
            or
            freq_ratio
            > SIGNAL_FREQ_WARNING_HIGH
        )
    ):
        reasons.append(
            "SIGNAL_FREQUENCY"
        )

    return reasons


def get_critical_reasons(row):

    reasons = []

    avg_bps = row["rolling_avg_gross_bps"]
    hit_rate = row["rolling_hit_rate"]

    current_dd = row[
        "current_drawdown_30d"
    ]

    max_dd = row[
        "rolling_max_drawdown_30d"
    ]

    freq_ratio = row[
        "signal_frequency_ratio"
    ]

    if (
        pd.notna(avg_bps)
        and
        avg_bps < CRITICAL_AVG_GROSS_BPS
    ):
        reasons.append(
            "NEGATIVE_EDGE"
        )

    if (
        pd.notna(hit_rate)
        and
        hit_rate < CRITICAL_HIT_RATE
    ):
        reasons.append(
            "LOW_HIT_RATE"
        )

    if (
        pd.notna(current_dd)
        and
        current_dd < CRITICAL_CURRENT_DD
    ):
        reasons.append(
            "CURRENT_DRAWDOWN"
        )

    if (
        pd.notna(max_dd)
        and
        max_dd < CRITICAL_MAX_DD_30D
    ):
        reasons.append(
            "MAX_DRAWDOWN"
        )

    if (
        pd.notna(freq_ratio)
        and
        (
            freq_ratio
            < SIGNAL_FREQ_CRITICAL_LOW
            or
            freq_ratio
            > SIGNAL_FREQ_CRITICAL_HIGH
        )
    ):
        reasons.append(
            "SIGNAL_FREQUENCY"
        )

    return reasons


# =========================================================
# Raw status
# =========================================================

def determine_raw_status(row):

    critical_reasons = (
        get_critical_reasons(
            row
        )
    )

    warning_reasons = (
        get_warning_reasons(
            row
        )
    )

    # Keep same philosophy as v2:
    # require at least two critical dimensions.
    if len(critical_reasons) >= 2:
        return "CRITICAL"

    if len(warning_reasons) >= 1:
        return "WARNING"

    return "HEALTHY"


# =========================================================
# Recovery logic
# =========================================================

def recovery_conditions_met(row):

    avg_bps = (
        row[
            "rolling_avg_gross_bps"
        ]
    )

    hit_rate = (
        row[
            "rolling_hit_rate"
        ]
    )

    current_dd = (
        row[
            "current_drawdown_30d"
        ]
    )

    if (
        pd.isna(avg_bps)
        or
        pd.isna(hit_rate)
        or
        pd.isna(current_dd)
    ):
        return False

    return (
        avg_bps
        > RECOVERY_AVG_GROSS_BPS
        and
        hit_rate
        > RECOVERY_HIT_RATE
        and
        current_dd
        > RECOVERY_CURRENT_DD
    )


def apply_staged_state_machine(df):

    df = df.copy()

    final_status = []

    previous_status = "HEALTHY"

    for _, row in df.iterrows():

        raw_status = (
            row[
                "raw_performance_status"
            ]
        )

        # ----------------------------------
        # If conditions are currently critical
        # ----------------------------------

        if raw_status == "CRITICAL":

            current_status = "CRITICAL"

        # ----------------------------------
        # If conditions are currently warning
        # ----------------------------------

        elif raw_status == "WARNING":

            # CRITICAL is allowed to step down
            # immediately to WARNING.
            current_status = "WARNING"

        # ----------------------------------
        # Raw status is HEALTHY
        # ----------------------------------

        else:

            if previous_status == "CRITICAL":

                # Staged recovery:
                # never CRITICAL -> HEALTHY directly.
                current_status = "WARNING"

            elif previous_status == "WARNING":

                if recovery_conditions_met(
                    row
                ):
                    current_status = "HEALTHY"

                else:
                    current_status = "WARNING"

            else:

                current_status = "HEALTHY"

        final_status.append(
            current_status
        )

        previous_status = (
            current_status
        )

    df[
        "performance_status"
    ] = final_status

    return df


# =========================================================
# Alert reason classification
# =========================================================

def determine_alert_reason(row):

    status = (
        row[
            "performance_status"
        ]
    )

    if status == "HEALTHY":
        return "NONE"

    if status == "CRITICAL":

        reasons = (
            get_critical_reasons(
                row
            )
        )

        if not reasons:

            reasons = (
                get_warning_reasons(
                    row
                )
            )

    else:

        reasons = (
            get_warning_reasons(
                row
            )
        )

    if not reasons:

        return "RECOVERY_HOLD"

    return "|".join(
        reasons
    )


# =========================================================
# Overall status
# =========================================================

def combine_status(row):

    sample_status = (
        row["sample_status"]
    )

    performance_status = (
        row[
            "performance_status"
        ]
    )

    if sample_status == "SUFFICIENT":
        return performance_status

    if sample_status == "LOW_SAMPLE":

        if performance_status == "HEALTHY":

            return (
                "HEALTHY_LOW_CONFIDENCE"
            )

        return (
            f"{performance_status}"
            "_LOW_CONFIDENCE"
        )

    if sample_status == "VERY_LOW_SAMPLE":

        if performance_status == "HEALTHY":

            return (
                "VERY_LOW_SAMPLE"
            )

        return (
            f"{performance_status}"
            "_VERY_LOW_SAMPLE"
        )

    return "NO_DATA"


# =========================================================
# Daily monitor
# =========================================================

def create_daily_monitor(df):

    temp = df.copy()

    temp["date"] = (
        temp["timestamp"]
        .dt.date
    )

    return (
        temp.groupby("date")
        .tail(1)
        .copy()
    )


# =========================================================
# Monthly summary
# =========================================================

def print_monthly_summary(df):

    temp = df.copy()

    temp["month"] = (
        temp["timestamp"]
        .dt.strftime("%Y-%m")
    )

    rows = []

    for month, group in (
        temp.groupby("month")
    ):

        daily = (
            create_daily_monitor(
                group
            )
        )

        last = daily.iloc[-1]

        rows.append({

            "month":
                month,

            "sample_status":
                last[
                    "sample_status"
                ],

            "performance_status":
                last[
                    "performance_status"
                ],

            "overall_status":
                last[
                    "overall_status"
                ],

            "alert_reason":
                last[
                    "alert_reason"
                ],

            "rolling_signal_count":
                last[
                    "rolling_signal_count"
                ],

            "rolling_avg_gross_bps":
                last[
                    "rolling_avg_gross_bps"
                ],

            "rolling_hit_rate":
                last[
                    "rolling_hit_rate"
                ],

            "rolling_30d_return":
                last[
                    "rolling_30d_return"
                ],

            "current_drawdown_30d":
                last[
                    "current_drawdown_30d"
                ],

            "rolling_max_drawdown_30d":
                last[
                    "rolling_max_drawdown_30d"
                ],

            "signal_frequency_ratio":
                last[
                    "signal_frequency_ratio"
                ],

            "warning_day_ratio":
                (
                    daily[
                        "performance_status"
                    ]
                    .isin(
                        [
                            "WARNING",
                            "CRITICAL"
                        ]
                    )
                    .mean()
                ),

            "critical_day_ratio":
                (
                    daily[
                        "performance_status"
                    ]
                    .eq(
                        "CRITICAL"
                    )
                    .mean()
                )
        })

    result = pd.DataFrame(
        rows
    )

    print(
        "\n=========================="
    )

    print(
        "=== Monthly Monitor Summary V3 ==="
    )

    print(
        result.to_string(
            index=False
        )
    )

    return result


# =========================================================
# Alert reason summary
# =========================================================

def print_alert_reason_summary(df):

    alerted = df[
        df["alert_reason"]
        != "NONE"
    ].copy()

    print(
        "\n=========================="
    )

    print(
        "=== Alert Reason Counts ==="
    )

    reason_counts = {}

    for value in (
        alerted[
            "alert_reason"
        ]
        .dropna()
    ):

        for reason in (
            value.split("|")
        ):

            reason_counts[
                reason
            ] = (
                reason_counts
                .get(
                    reason,
                    0
                )
                + 1
            )

    reason_df = (
        pd.Series(
            reason_counts
        )
        .sort_values(
            ascending=False
        )
    )

    print(
        reason_df.to_string()
    )


# =========================================================
# Transitions
# =========================================================

def print_transitions(df):

    temp = df.copy()

    changed = (
        temp["overall_status"]
        !=
        temp["overall_status"]
        .shift(1)
    )

    transitions = (
        temp[
            changed
        ][
            [
                "timestamp",
                "sample_status",
                "raw_performance_status",
                "performance_status",
                "overall_status",
                "alert_reason",

                "rolling_signal_count",
                "rolling_avg_gross_bps",
                "rolling_hit_rate",

                "rolling_30d_return",
                "current_drawdown_30d",
                "rolling_max_drawdown_30d",

                "signal_frequency_ratio"
            ]
        ]
    )

    print(
        "\n=========================="
    )

    print(
        "=== Alert Transitions V3 ==="
    )

    print(
        transitions.tail(
            40
        ).to_string(
            index=False
        )
    )


# =========================================================
# Main
# =========================================================

def main():

    df = load_data()

    print(
        "Rows:",
        len(df)
    )

    df = add_basic_columns(
        df
    )

    df = add_rolling_metrics(
        df
    )

    df = add_rolling_return(
        df
    )

    df = add_drawdown_metrics(
        df
    )

    df = add_frequency_baseline(
        df
    )

    df["sample_status"] = (
        df["rolling_signal_count"]
        .apply(
            determine_sample_status
        )
    )

    df[
        "raw_performance_status"
    ] = (
        df.apply(
            determine_raw_status,
            axis=1
        )
    )

    df = (
        apply_staged_state_machine(
            df
        )
    )

    df["alert_reason"] = (
        df.apply(
            determine_alert_reason,
            axis=1
        )
    )

    df["overall_status"] = (
        df.apply(
            combine_status,
            axis=1
        )
    )

    print(
        "\n=========================="
    )

    print(
        "=== Overall Status Counts V3 ==="
    )

    print(
        df["overall_status"]
        .value_counts()
        .to_string()
    )

    print_monthly_summary(
        df
    )

    print_alert_reason_summary(
        df
    )

    print_transitions(
        df
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
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