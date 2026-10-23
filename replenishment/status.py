"""Exact status normalization, without fuzzy matching or null inference."""
import re
import pandas as pd


def normalize_product_status(value):
    if value is None or pd.isna(value):
        return pd.NA
    text = re.sub(r"\s+", " ", str(value).strip())
    if not text:
        return pd.NA
    return {"normal": "Normal", "phasing out": "Phasing Out"}.get(text.casefold(), text)


def status_counts(data, field="product_status"):
    return (data.assign(**{field: data[field].fillna("<MISSING>")})
            .groupby(field, dropna=False).agg(records=("sku_id", "size"), skus=("sku_id", "nunique"))
            .reset_index())


def filter_product_status(data, allowed=("Normal", "Phasing Out")):
    result = data.copy()
    result["product_status_raw"] = result["product_status"]
    result["product_status"] = result["product_status"].map(normalize_product_status)
    mapping = (result.assign(raw=result["product_status_raw"].fillna("<MISSING>"),
                             normalized=result["product_status"].fillna("<MISSING>"))
               .groupby(["raw", "normalized"], dropna=False)
               .agg(records=("sku_id", "size"), skus=("sku_id", "nunique")).reset_index())
    mapping["included"] = mapping["normalized"].isin(allowed)
    before = status_counts(result)
    result = result.loc[result["product_status"].isin(allowed)].copy()
    assert set(result["product_status"].unique()) <= set(allowed) <= {"Normal", "Phasing Out"}
    return result, mapping, before
