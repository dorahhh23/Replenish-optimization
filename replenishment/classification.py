"""ADI/CV-squared classification using only the series passed by the caller."""
import numpy as np
import pandas as pd


def classify_series(values, config):
    y = np.asarray(values, dtype=float)
    if not len(y) or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Classification requires finite, nonnegative observed demand.")
    positive = y[y > 0]
    if not len(positive):
        return {"demand_type": "Zero", "adi": np.inf, "cv2": np.nan,
                "positive_periods": 0, "periods": len(y), "cv2_reliable": False}
    adi = len(y) / len(positive)
    reliable = len(positive) > max(1, config.variance_ddof)
    # A singleton has no dispersion evidence; record that limitation explicitly.
    cv2 = float(np.var(positive, ddof=config.variance_ddof) / np.mean(positive) ** 2) if reliable else 0.0
    sparse = adi >= config.adi_threshold
    variable = cv2 >= config.cv2_threshold
    kind = ("Lumpy" if variable else "Intermittent") if sparse else ("Erratic" if variable else "Smooth")
    return {"demand_type": kind, "adi": adi, "cv2": cv2, "positive_periods": len(positive),
            "periods": len(y), "cv2_reliable": reliable}


def classify_skus(panel, config):
    rows = []
    for sku, group in panel.groupby("sku_id", sort=True):
        group = group.sort_values("date")
        rows.append({"sku_id": sku, "material": group["material"].iloc[0],
                     "product_status": group["product_status"].iloc[0],
                     "grain": group["grain"].iloc[0], "first_date": group["date"].min(),
                     "last_period_end": group["period_end"].max(),
                     "total_sales": group["sales"].sum(),
                     **classify_series(group["sales"].to_numpy(), config)})
    detail = pd.DataFrame(rows)
    index = pd.MultiIndex.from_product([sorted(detail["product_status"].unique()),
                                       ["Smooth", "Erratic", "Intermittent", "Lumpy", "Zero"]],
                                      names=["product_status", "demand_type"])
    summary = detail.groupby(["product_status", "demand_type"]).size().reindex(index, fill_value=0).rename("skus").reset_index()
    summary["status_share"] = summary["skus"] / summary.groupby("product_status")["skus"].transform("sum")
    summary["all_sku_share"] = summary["skus"] / len(detail)
    return detail, summary
