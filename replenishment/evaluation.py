"""Rolling-origin validation, an untouched 16-week test, and final refitting."""
import json
import numpy as np
import pandas as pd
from .classification import classify_series
from .forecasting import METHODS, forecast, method_applicable
from .outliers import outlier_versions


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    error = predicted - actual
    denominator = float(np.abs(actual).sum())
    return {"mae": float(np.mean(np.abs(error))),
            "wape": float(np.abs(error).sum() / denominator) if denominator else np.nan,
            "bias": float(error.mean()), "bias_pct": float(error.sum() / denominator) if denominator else np.nan,
            "absolute_error_sum": float(np.abs(error).sum()), "error_sum": float(error.sum()),
            "actual_abs_sum": denominator, "observations": len(actual)}


def window_plan(n, config):
    """Comparable windows share validation origins entirely before the held-out test."""
    test_origin = n - config.forecast_horizon
    available = {}
    for window in sorted(set(config.windows)):
        origins = list(range(window, test_origin - config.validation_horizon + 1, config.validation_step))
        available[window] = origins
    eligible = [w for w, origins in available.items() if len(origins) >= config.min_validation_origins]
    common = available[max(eligible)][-config.max_validation_origins:] if eligible else []
    rows = []
    for window, origins in available.items():
        full16 = max(0, n - window - config.forecast_horizon + 1)
        rows.append({"window_weeks": window, "available_weeks": n,
                     "enough_final_training": n >= window,
                     "possible_16_week_origins": full16,
                     "validation_origins_before_holdout": len(origins),
                     "common_validation_origins_used": len(common) if window in eligible else 0,
                     "validation_horizon_weeks": config.validation_horizon,
                     "heldout_test_weeks": config.forecast_horizon,
                     "status": "comparable" if window in eligible else "insufficient samples / not comparable"})
    return eligible, common, test_origin, rows


def aggregate_scores(frame, group_cols):
    if frame.empty:
        return pd.DataFrame()
    grouped = frame.groupby(group_cols, dropna=False).agg(
        absolute_error_sum=("absolute_error_sum", "sum"), error_sum=("error_sum", "sum"),
        actual_abs_sum=("actual_abs_sum", "sum"), observations=("observations", "sum"),
        skus=("sku_id", "nunique")).reset_index()
    grouped["mae"] = grouped["absolute_error_sum"] / grouped["observations"]
    grouped["wape"] = grouped["absolute_error_sum"] / grouped["actual_abs_sum"].replace(0, np.nan)
    grouped["bias"] = grouped["error_sum"] / grouped["observations"]
    grouped["bias_pct"] = grouped["error_sum"] / grouped["actual_abs_sum"].replace(0, np.nan)
    return grouped[group_cols + ["skus", "mae", "wape", "bias", "bias_pct", "observations",
                                 "actual_abs_sum", "absolute_error_sum", "error_sum"]]


def run_forecasts(panel, config, logger, reference_date=None):
    validation_rows, test_rows, selected_rows, forecasts, availability, applicability = [], [], [], [], [], []
    for number, (sku, group) in enumerate(panel.groupby("sku_id", sort=True), 1):
        group = group.sort_values("date")
        y, dates = group["sales"].to_numpy(float), group["date"].reset_index(drop=True)
        status = group["product_status"].iloc[0]
        windows, origins, test_origin, plan = window_plan(len(y), config)
        for row in plan:
            availability.append({"sku_id": sku, "product_status": status, **row})
        if not windows:
            selected_rows.append({"sku_id": sku, "product_status": status, "status": "insufficient validation and held-out test samples"})
            continue
        scores = []
        test_cache, final_cache = {}, {}
        for window in windows:
            test_cache[window] = outlier_versions(y[test_origin - window:test_origin], config)[0]
            final_cache[window] = outlier_versions(y[-window:], config)[0]

        def record_unavailable(phase, window, version, method, train, origin):
            applicability.append({"sku_id": sku, "product_status": status, "phase": phase,
                                  "window_weeks": window, "version": version, "method": method,
                                  "origin": origin, "negative_net_periods": int((np.asarray(train) < 0).sum()),
                                  "applicable": False, "reason": "negative net sales; no return clipping or reallocation"})
        # Labels for each fold are computed from that fold's historical window only.
        for window in windows:
            fold_cache = {}
            for origin in origins:
                train = y[origin - window:origin]
                fold_cache[origin] = (outlier_versions(train, config)[0], classify_series(train, config)["demand_type"])
            for version in ("Raw", "Winsorized", "Median"):
                for method in METHODS:
                    invalid_origins = [o for o in origins if not method_applicable(fold_cache[o][0][version], method)]
                    if invalid_origins:
                        for origin in invalid_origins:
                            record_unavailable("rolling_validation", window, version, method,
                                               fold_cache[origin][0][version], dates.iloc[origin])
                        continue  # Require every common origin; never compare partial candidate scores.
                    predicted, actual = [], []
                    for origin in origins:
                        variations, kind = fold_cache[origin]
                        p = forecast(variations[version], method, config.validation_horizon, config.smoothing_alpha, config.tsb_beta)
                        a = y[origin:origin + config.validation_horizon]
                        score = metrics(a, p)
                        validation_rows.append({"sku_id": sku, "product_status": status, "demand_type": kind,
                                                "window_weeks": window, "version": version, "method": method,
                                                "origin": dates.iloc[origin], "train_start": dates.iloc[origin - window],
                                                "train_end": dates.iloc[origin - 1], "phase": "rolling_validation", **score})
                        predicted.extend(p)
                        actual.extend(a)
                    scores.append({"window_weeks": window, "version": version, "method": method,
                                   **metrics(actual, predicted)})
        # MAE is defined even for Zero SKUs. Prefer 26 weeks only when scores tie.
        # All smoothing parameters are fixed in config, never tuned on test actuals.
        version_order = {"Raw": 0, "Winsorized": 1, "Median": 2}
        def rank_score(s):
            return (round(s["mae"], 12), s["window_weeks"] != config.preferred_window,
                    abs(s["bias"]), version_order[s["version"]], METHODS.index(s["method"]))
        # Applicability for the historical decision only sees the test-origin past.
        historical_candidates = [s for s in scores if method_applicable(test_cache[s["window_weeks"]][s["version"]], s["method"])]
        chosen = min(historical_candidates, key=rank_score)
        selected_identity = (chosen["window_weeks"], chosen["version"], chosen["method"])
        for window in windows:
            train = y[test_origin - window:test_origin]
            variations = test_cache[window]
            test_kind = classify_series(train, config)["demand_type"]
            for version, values in variations.items():
                for method in METHODS:
                    if not method_applicable(values, method):
                        record_unavailable("heldout_16_week_test", window, version, method, values, dates.iloc[test_origin])
                        continue
                    p = forecast(values, method, config.forecast_horizon, config.smoothing_alpha, config.tsb_beta)
                    score = metrics(y[test_origin:], p)
                    test_rows.append({"sku_id": sku, "product_status": status, "demand_type": test_kind,
                                      "window_weeks": window, "version": version, "method": method,
                                      "selected_by_validation": (window, version, method) == selected_identity,
                                      "test_origin": dates.iloc[test_origin], "train_end": dates.iloc[test_origin - 1],
                                      "phase": "heldout_16_week_test", **score})
        # At the run date, newly observed returns can change which model can be fitted.
        # Reuse the validation ranking, never test errors. Keep historical test selection separate.
        for s in scores:
            values = final_cache[s["window_weeks"]][s["version"]]
            if not method_applicable(values, s["method"]):
                record_unavailable("final_refit", s["window_weeks"], s["version"], s["method"], values, dates.iloc[-1])
        final_candidates = [s for s in scores if method_applicable(final_cache[s["window_weeks"]][s["version"]], s["method"])]
        final_chosen = min(final_candidates, key=rank_score)
        window, version, method = (final_chosen["window_weeks"], final_chosen["version"], final_chosen["method"])
        final_versions = final_cache[window]
        ref = pd.Timestamp(reference_date if reference_date is not None else pd.Timestamp.today()).normalize()
        # Forecast the next 16 full weeks after the run date. Bridge the reporting lag
        # with predictions only, never with missing/current-month actuals.
        first_future_week = ref - pd.Timedelta(days=ref.weekday()) + pd.Timedelta(weeks=1)
        first_future_week = max(first_future_week, dates.iloc[-1] + pd.Timedelta(weeks=1))
        lead = int((first_future_week - dates.iloc[-1]).days // 7)
        p_all = forecast(final_versions[version], method, lead - 1 + config.forecast_horizon, config.smoothing_alpha, config.tsb_beta)
        p = p_all[lead - 1:]
        recent_mean = float(np.mean(y[-min(13, len(y)):]))
        trailing = 0
        for value in y[::-1]:
            if value != 0:
                break
            trailing += 1
        growth = bool(p.mean() > recent_mean * config.phasing_growth_ratio) if recent_mean > 0 else bool(p.mean() > 0)
        review = status == "Phasing Out" and (growth or (trailing >= config.trailing_zero_review_periods and p.sum() > 0))
        selected_rows.append({"sku_id": sku, "material": group["material"].iloc[0], "product_status": status,
                              "window_weeks": window, "version": version, "method": method,
                              "validation_mae": final_chosen["mae"], "validation_origins": len(origins),
                              "historical_selected_window": selected_identity[0], "historical_selected_version": selected_identity[1],
                              "historical_selected_method": selected_identity[2],
                              "final_selection_changed_for_applicability": (window, version, method) != selected_identity,
                              "negative_net_periods": int((y < 0).sum()), "sales_measure": "net_sales_quantity",
                              "validation_origin_dates": json.dumps([str(dates.iloc[o].date()) for o in origins]),
                              "test_origin": dates.iloc[test_origin], "test_end": group["period_end"].iloc[-1],
                              "training_end": group["period_end"].iloc[-1], "forecast_16_week_total": float(p.sum()),
                              "first_future_week": first_future_week, "unobserved_bridge_weeks": lead - 1,
                              "recent_13_week_mean": recent_mean, "trailing_zero_weeks": trailing,
                              "growth_vs_recent": growth, "phasing_out_review": review,
                              "replenishment_action": "Review stop/reduce rules; no automatic order" if status == "Phasing Out" else "Demand forecast only; inventory policy needs additional inputs",
                              "status": "forecasted"})
        for step, value in enumerate(p, 1):
            first = first_future_week + pd.Timedelta(weeks=step - 1)
            forecasts.append({"sku_id": sku, "material": group["material"].iloc[0], "product_status": status,
                              "forecast_origin": ref, "training_last_week_start": dates.iloc[-1], "week_start": first,
                              "week_end": first + pd.Timedelta(days=6), "horizon_week": step, "lead_from_training_weeks": lead + step - 1,
                              "forecast": float(value), "method": method, "version": version,
                              "window_weeks": window, "phasing_out_review": review})
        if number % 500 == 0:
            logger.info("Forecasted %s SKUs", number)
    selected, test = pd.DataFrame(selected_rows), pd.DataFrame(test_rows)
    validation = pd.DataFrame(validation_rows)
    scores = aggregate_scores(test, ["product_status", "demand_type", "window_weeks", "version", "method"])
    selected_test = test.loc[test["selected_by_validation"]].copy() if not test.empty else pd.DataFrame()
    selected_summary = aggregate_scores(selected_test, ["product_status", "demand_type"])
    return {"validation_detail": validation, "test_detail": test, "selected_models": selected,
            "forecast_16_weeks": pd.DataFrame(forecasts), "test_summary": scores,
            "selected_test": selected_test, "selected_test_summary": selected_summary,
            "window_availability": pd.DataFrame(availability),
            "method_applicability": pd.DataFrame(applicability, columns=["sku_id", "product_status", "phase", "window_weeks", "version", "method", "origin", "negative_net_periods", "applicable", "reason"]),
            "validation_summary": aggregate_scores(validation, ["product_status", "demand_type", "window_weeks", "version", "method"])}
