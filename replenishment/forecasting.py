"""Deterministic level forecasts. No randomized split or reinforcement learning."""
import numpy as np

METHODS = ("MovingAverage", "SES", "Croston", "SBA", "TSB")


def forecast(values, method, horizon, alpha=0.1, beta=0.1):
    y = np.asarray(values, dtype=float)
    if not len(y) or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Training series must be finite and nonnegative.")
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
    return np.full(horizon, max(0.0, level), dtype=float)
