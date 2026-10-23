"""Training-only Sen slope, median trend, and positive-demand IQR handling."""
import numpy as np
import pandas as pd


def sens_slope(y):
    y = np.asarray(y, dtype=float)
    if len(y) < 2:
        return 0.0
    i, j = np.triu_indices(len(y), k=1)
    return float(np.median((y[j] - y[i]) / (j - i)))


def fit_outlier_rule(values, config):
    y = np.asarray(values, dtype=float)
    if not len(y) or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Outlier estimation requires nonnegative finite training data.")
    split = len(y) // 2
    left = float(np.median(y[:split])) if split else float(y[0])
    right = float(np.median(y[split:]))
    slope = sens_slope(y)
    rising = slope > config.trend_slope_threshold and right - left > config.median_increase_threshold
    positive = y[y > 0]
    enough = len(positive) >= config.min_positive_for_outliers
    q25, q75 = np.quantile(positive, [0.25, 0.75]) if len(positive) else (np.nan, np.nan)
    upper = float(q75 + config.iqr_multiplier * (q75 - q25)) if enough else np.inf
    median = float(np.median(positive)) if len(positive) else 0.0
    return {"sen_slope": slope, "first_half_median": left, "second_half_median": right,
            "rising_trend": bool(rising), "q25_positive": q25, "q75_positive": q75,
            "upper": upper, "median_positive": median, "positive_count": len(positive),
            "enough_positive_samples": enough}


def outlier_versions(values, config):
    y = np.asarray(values, dtype=float)
    rule = fit_outlier_rule(y, config)
    flagged = y > rule["upper"]
    replace = flagged & (not rule["rising_trend"])
    # Confirmation is statistical (IQR with both trend tests), not business confirmation.
    # No future/test actual is used to fit these replacement values.
    winsorized, median = y.copy(), y.copy()
    winsorized[replace] = rule["upper"]
    median[replace] = rule["median_positive"]
    return {"Raw": y.copy(), "Winsorized": winsorized, "Median": median}, rule, flagged, replace


def compare_outliers(panel, config):
    summaries, points = [], []
    for sku, group in panel.groupby("sku_id", sort=True):
        group = group.sort_values("date")
        versions, rule, flagged, replaced = outlier_versions(group["sales"].to_numpy(), config)
        summaries.append({"sku_id": sku, "product_status": group["product_status"].iloc[0], **rule,
                          "flagged_periods": int(flagged.sum()), "replaced_periods": int(replaced.sum()),
                          "raw_total": float(versions["Raw"].sum()), "winsorized_total": float(versions["Winsorized"].sum()),
                          "median_total": float(versions["Median"].sum()), "scope": "full-window descriptive / final model fit only"})
        for i in np.flatnonzero(flagged):
            points.append({"sku_id": sku, "date": group["date"].iloc[i], "Raw": versions["Raw"][i],
                           "Winsorized": versions["Winsorized"][i], "Median": versions["Median"][i],
                           "replaced": bool(replaced[i]), "rising_trend": rule["rising_trend"]})
    return pd.DataFrame(summaries), pd.DataFrame(points, columns=["sku_id", "date", "Raw", "Winsorized", "Median", "replaced", "rising_trend"])
