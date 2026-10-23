"""Key construction, normalization, auditable quarantine, and period completion."""
import json
import numpy as np
import pandas as pd
from .excel_io import SchemaError, identifier
from .periods import assert_period


def count_stage(name, data, unit="SKU-period rows"):
    return {"stage": name, "record_unit": unit, "records": len(data),
            "skus": int(data["sku_id"].nunique()) if "sku_id" in data else None}


def numeric(series):
    # Unambiguous comma thousands separators only; unknown decimal conventions stay missing.
    text = series.astype("string").str.strip()
    thousands = text.str.fullmatch(r"[-+]?\d{1,3}(,\d{3})+(\.\d+)?", na=False)
    text = text.where(~thousands, text.str.replace(",", "", regex=False))
    return pd.to_numeric(text, errors="coerce").replace([np.inf, -np.inf], np.nan)


def keyed_source(data, mapping):
    out = data.copy()
    for role in ("material", "sales_org", "storage_location"):
        out[role] = out[mapping[role]].map(identifier) if role in mapping else ""
    out["missing_key"] = out["material"].eq("") | out["material"].eq("<INVALID_NUMERIC_ID>")
    # A blank location is a disclosed unknown identifier, not an inferred physical warehouse.
    out["storage_location"] = out["storage_location"].replace("", "<UNSPECIFIED>")
    out["sales_org"] = out["sales_org"].replace("", "<UNSPECIFIED>")
    out["sku_id"] = [json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))
                     for values in out[["sales_org", "storage_location", "material"]].itertuples(index=False, name=None)]
    out["product_status"] = out[mapping["product_status"]]
    if "unit" in mapping:
        out["unit"] = out[mapping["unit"]].astype("string").str.strip().str.upper()
    else:
        out["unit"] = "<NOT PROVIDED>"
    return out


def to_long(source, periods, mapping, config):
    metadata = ["sku_id", "material", "sales_org", "storage_location", "product_status", "missing_key", "unit", "_source_row"]
    if "product_category" in mapping:
        source = source.copy()
        source["product_category"] = source[mapping["product_category"]]
        metadata.append("product_category")
    if periods:
        grains = {p["grain"] for p in periods}
        grain = config.grain if config.grain != "auto" else ("weekly" if "weekly" in grains else "monthly" if "monthly" in grains else "daily")
        chosen = [p for p in periods if p["grain"] == grain]
        if not chosen:
            raise SchemaError(f"No actual sales columns for configured grain {grain}.")
        lookup = {p["field"]: p for p in chosen}
        result = source.melt(id_vars=metadata, value_vars=list(lookup), var_name="source_field", value_name="sales_raw")
        result["date"] = pd.to_datetime(result["source_field"].map(lambda name: lookup[name]["start"]))
        result["period_end"] = pd.to_datetime(result["source_field"].map(lambda name: lookup[name]["end"]))
        return result, grain
    if "date" not in mapping or "sales" not in mapping:
        raise SchemaError("No year-anchored actual sales periods, or unambiguous date/sales fields.")
    if config.grain == "auto":
        raise SchemaError("Long-form dates require an explicit grain: transaction/daily/weekly/monthly. Do not infer grain from sparse row spacing.")
    result = source[metadata].copy()
    result["sales_raw"] = source[mapping["sales"]]
    result["source_field"] = mapping["sales"]
    result["date"] = pd.to_datetime(source[mapping["date"]], errors="coerce")
    grain = config.grain
    if grain == "monthly":
        result["date"] = result["date"].dt.to_period("M").dt.to_timestamp()
        result["period_end"] = result["date"] + pd.offsets.MonthEnd(0)
    elif grain == "weekly":
        if result["date"].dropna().dt.weekday.ne(0).any():
            raise SchemaError("Weekly date fields must be verified ISO week-start dates (Mondays).")
        result["period_end"] = result["date"] + pd.Timedelta(days=6)
    else:
        result["period_end"] = result["date"].dt.normalize()
    return result, grain


def quality_check(data, grain, wide_snapshot=True, status_conflicts=()):
    """Never merge ambiguous Parent/Son rows or turn returns/missing sales into zero."""
    result = data.copy()
    result["sales"] = numeric(result["sales_raw"])
    bad = {}

    def flag(keys, reason):
        for key in keys:
            bad.setdefault(key, set()).add(reason)

    flag(result.loc[result["missing_key"], "sku_id"].unique(), "missing_or_invalid_material")
    flag(result.loc[result["sales"].isna(), "sku_id"].unique(), "missing_or_nonnumeric_sales")
    flag(result.loc[result["sales"].lt(0), "sku_id"].unique(), "negative_sales_possible_return")
    flag(status_conflicts, "conflicting_product_status")
    multiunit = result.groupby("sku_id")["unit"].nunique(dropna=False)
    flag(multiunit[multiunit > 1].index, "inconsistent_unit")
    if wide_snapshot or grain in {"weekly", "monthly"}:
        dup = result.duplicated(["sku_id", "date"], keep=False)
        flag(result.loc[dup, "sku_id"].unique(), "duplicate_sku_period_or_parent_son_overlap")
    else:
        # Without transaction IDs, identical daily rows could be separate legitimate sales.
        # Quarantine rather than silently deduplicating or aggregating suspicious repeats.
        dup = result.duplicated(["sku_id", "date", "sales", "product_status"], keep=False)
        flag(result.loc[dup, "sku_id"].unique(), "duplicate_event_without_transaction_id")
    quality = []
    for issue, mask in [
        ("negative_sales_possible_return", result["sales"].lt(0)),
        ("missing_or_nonnumeric_sales", result["sales"].isna()),
        ("missing_or_invalid_material", result["missing_key"]),
        ("duplicate_sku_period", dup),
        ("blank_storage_location", result["storage_location"].eq("<UNSPECIFIED>")),
        ("missing_sales_unit", result["unit"].eq("<NOT PROVIDED>") | result["unit"].isna()),
    ]:
        quality.append({"issue": issue, "records": int(mask.sum()), "skus": int(result.loc[mask, "sku_id"].nunique())})
    quarantined = result.loc[result["sku_id"].isin(bad)].copy()
    quarantined["reason"] = quarantined["sku_id"].map(lambda key: "; ".join(sorted(bad[key])))
    reason_rows = [{"sku_id": key, "reason": reason} for key, reasons in sorted(bad.items()) for reason in sorted(reasons)]
    clean = result.loc[~result["sku_id"].isin(bad)].copy()
    assert clean["sales"].notna().all() and clean["sales"].ge(0).all()
    return clean, pd.DataFrame(quality), quarantined, pd.DataFrame(reason_rows, columns=["sku_id", "reason"])


def complete_panel(data, grain, start, end, fill_internal_missing_weeks=True):
    """Only complete observed coverage. Never pad the unavailable start of the 12-month window."""
    assert_period(data, start, end)
    if data.empty:
        raise ValueError("No eligible demand remains after quality quarantine.")
    working = data.copy()
    if grain in {"daily", "transaction"}:
        working["date"] -= pd.to_timedelta(working["date"].dt.weekday, unit="D")
        working["period_end"] = working["date"] + pd.Timedelta(days=6)
        working = working.loc[working["date"].ge(pd.Timestamp(start)) & working["period_end"].le(pd.Timestamp(end))]
        grain = "weekly"
    freq = "W-MON" if grain == "weekly" else "MS"
    dates = pd.DatetimeIndex(sorted(working["date"].unique()))
    grid = pd.date_range(dates.min(), dates.max(), freq=freq)
    if grain == "weekly" and len(grid.difference(dates)):
        raise SchemaError("A complete source week is absent for every SKU. Unknown coverage must not be zero-filled.")
    # Missing monthly source columns are unknown, not zero observations.
    if grain == "monthly":
        grid = dates
    grouped = working.groupby(["sku_id", "date"], as_index=False)["sales"].sum(min_count=1)
    idx = pd.MultiIndex.from_product([sorted(grouped["sku_id"].unique()), grid], names=["sku_id", "date"])
    panel = grouped.set_index(["sku_id", "date"]).reindex(idx)
    panel["imputed_no_sale_period"] = panel["sales"].isna()
    if panel["imputed_no_sale_period"].any() and not fill_internal_missing_weeks:
        raise SchemaError("Per-SKU periods are absent and zero completion is disabled.")
    panel["sales"] = panel["sales"].fillna(0.0)
    panel = panel.reset_index()
    metadata = working.drop_duplicates("sku_id")[[c for c in ["sku_id", "material", "sales_org", "storage_location", "product_status", "unit", "product_category"] if c in working]]
    panel = panel.merge(metadata, on="sku_id", how="left", validate="many_to_one")
    panel["period_end"] = panel["date"] + (pd.Timedelta(days=6) if grain == "weekly" else pd.offsets.MonthEnd(0))
    panel["grain"] = grain
    assert_period(panel, start, end)
    assert not panel.duplicated(["sku_id", "date"]).any()
    assert set(panel["product_status"].unique()) <= {"Normal", "Phasing Out"}
    return panel, grain
