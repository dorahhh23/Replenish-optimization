"""Deterministic level forecasts. No randomized split or reinforcement learning."""
import numpy as np

METHODS = ("MovingAverage", "SES", "Croston", "SBA", "TSB")


def method_applicable(values, method):
    y = np.asarray(values, dtype=float)
    return bool(len(y) and np.isfinite(y).all() and
                (method in {"MovingAverage", "SES"} or not (y < 0).any()))


def forecast(values, method, horizon, alpha=0.1, beta=0.1):
    y = np.asarray(values, dtype=float)
    if not method_applicable(y, method):
        raise ValueError(f"{method} is not applicable to this training window; Croston/SBA/TSB require nonnegative input.")
    if method == "MovingAverage":
        level = float(y.mean())
    elif method == "SES":
        level = float(y[0])
        for value in y[1:]:
            level = alpha * value + (1 - alpha) * level
    elif method in {"Croston", "SBA", "TSB"}:
        positive = np.flatnonzero(y > 0)
        if not len(positive):
            return np.zeros(horizon)
        first = int(positive[0])
        size, interval, probability = float(y[first]), float(first + 1), 1.0 / (first + 1)
        since = 1
        for value in y[first + 1:]:
            if method == "TSB":
                probability = beta * float(value > 0) + (1 - beta) * probability
                if value > 0:
                    size = alpha * value + (1 - alpha) * size
            else:
                if value > 0:
                    size = alpha * value + (1 - alpha) * size
                    interval = alpha * since + (1 - alpha) * interval
                    since = 1
                else:
                    since += 1
        level = size * probability if method == "TSB" else size / interval
        if method == "SBA":
            level *= 1 - alpha / 2
    else:
        raise ValueError(f"Unsupported forecast method: {method}")
    # Signed actuals are preserved. Only the operational demand forecast has floor 0.
    return np.full(horizon, max(0.0, level), dtype=float)
