"""ADI/CV-squared classification using only the series passed by the caller."""
import numpy as np
import pandas as pd


def classify_series(values, config):
    y = np.asarray(values, dtype=float)
    if not len(y) or not np.isfinite(y).all():
        raise ValueError("Classification requires finite observed net sales.")
    positive = y[y > 0]
    negative_periods = int((y < 0).sum())
    shared = {"negative_net_periods": negative_periods, "classification_basis": "positive_net_sales_profile",
              "classification_provisional_due_to_returns": negative_periods > 0}
    if not len(positive):
        return {"demand_type": pd.NA if negative_periods else "Zero", "adi": np.inf, "cv2": np.nan,
                "positive_periods": 0, "periods": len(y), "cv2_reliable": False,
                "classification_status": "not_classifiable_return_only" if negative_periods else "classified", **shared}
    adi = len(y) / len(positive)
    reliable = len(positive) > max(1, config.variance_ddof)
    # A singleton has no dispersion evidence; record that limitation explicitly.
    cv2 = float(np.var(positive, ddof=config.variance_ddof) / np.mean(positive) ** 2) if reliable else 0.0
    sparse = adi >= config.adi_threshold
    variable = cv2 >= config.cv2_threshold
    kind = ("Lumpy" if variable else "Intermittent") if sparse else ("Erratic" if variable else "Smooth")
    return {"demand_type": kind, "adi": adi, "cv2": cv2, "positive_periods": len(positive),
            "periods": len(y), "cv2_reliable": reliable, "classification_status": "classified", **shared}


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
    totals = detail.groupby("product_status").size()
    summary["status_share"] = summary["skus"] / summary["product_status"].map(totals)
    summary["all_sku_share"] = summary["skus"] / len(detail)
    return detail, summary
