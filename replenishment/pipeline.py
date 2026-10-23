"""Orchestrate inspection, strict filters, quality checks, estimation, and exports."""
from datetime import date, datetime, timezone
import importlib.metadata
import logging
from pathlib import Path
import random
import re
import sys
import numpy as np
import pandas as pd
from .classification import classify_skus
from .cleaning import count_stage, keyed_source, to_long, quality_check, complete_panel
from .evaluation import run_forecasts, window_plan
from .excel_io import (SchemaError, read_workbook, identify_fields, identify_period_columns,
                       field_candidates, sha256)
from .export import write_csv, write_excel, write_json, verify_outputs
from .outliers import compare_outliers
from .periods import get_analysis_period, filter_dates, assert_period
from .reporting import summary_text
from .status import filter_product_status, normalize_product_status, status_counts

REQUIRED = ["data_dictionary.xlsx", "data_quality_report.xlsx", "filter_audit.xlsx",
            "cleaned_last_12_complete_months.csv", "sku_classification.xlsx",
            "outlier_comparison.xlsx", "forecast_evaluation.xlsx", "analysis_summary.md"]


def get_logger(output):
    logger = logging.getLogger("replenishment")
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(output / "pipeline.log", encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def annotate_dictionary(dictionary, mapping, periods, start, end, source_grain):
    dictionary = dictionary.copy()
    by_name = {p["field"]: p for p in periods}
    names = {name: role for role, name in mapping.items()}
    for i, row in dictionary.iterrows():
        name = row["cleaned_name"]
        roles = []
        if name in names:
            role = names[name]
            dictionary.loc[i, "meaning"] = {"material": "物料/SKU编码，按字符串保留前导零", "sales_org": "销售组织",
                "storage_location": "库位；空值不推断仓库", "product_status": "产品生命周期状态（Normal/Phasing Out筛选）",
                "product_category": "源产品分类，与ADI需求类型独立", "hierarchy_level": "Parent/Son层级，重复键需澄清",
                "unit": "销售数量计量单位"}.get(role, role)
            roles.append(role)
            dictionary.loc[i, "used_for_analysis"] = role in {"material", "sales_org", "storage_location", "product_status", "unit", "product_category", "date", "sales"}
            dictionary.loc[i, "purpose"] = "筛选 / 唯一键 / 单位校验 / 产品分组元数据"
        if name in by_name:
            period = by_name[name]
            included = period["start"] >= start and period["end"] <= end and period["grain"] == source_grain
            dictionary.loc[i, "used_for_analysis"] = included
            dictionary.loc[i, "meaning"] = f"真实{period['grain']}销量，{period['start']} 至 {period['end']}"
            dictionary.loc[i, "purpose"] = "窗口内真实需求：分类 / 异常值 / 训练 / 验证 / 测试" if included else "排除：日期范围外或不同粒度，避免重复计量"
            roles.append("actual_sales")
        if row["group"] == "预测销量":
            roles.append("company_forecast")
            dictionary.loc[i, "purpose"] = "未使用：缺少预测发布时间/历史版本，不能保证原点前可见"
        if row["group"] == "库存信息":
            roles.append("inventory")
            dictionary.loc[i, "purpose"] = "未使用：库存快照/未来投影无分析期内历史生效日期"
        if row["group"] == "在途在产信息":
            roles.append("in_transit_or_production")
            dictionary.loc[i, "purpose"] = "未使用：当前/未来排程，缺少历史到货及生效日期"
        if "欠单" in row["leaf"] or "backorder" in row["leaf"].casefold():
            roles.append("backorder")
        if re.search(r"金额|单价|价值|成本|amount|price|value|cost", row["leaf"], re.I):
            roles.append("value_related_candidate")
        dictionary.loc[i, "detected_roles"] = ", ".join(roles) or "other source metadata"
    return dictionary


def run_pipeline(input_path, output_path, config, reference_date=None, inspect_only=False):
    config.validate()
    output = Path(output_path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    input_path = Path(input_path).resolve()
    if input_path == output or output in input_path.parents:
        raise ValueError("The input workbook must be outside the output directory.")
    logger = get_logger(output)
    start, end = get_analysis_period(reference_date, config.months)
    ref = pd.Timestamp(reference_date if reference_date is not None else date.today()).date()
    manifest = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "reference_date": str(ref), "analysis_start": str(start), "analysis_end": str(end),
                "excluded_current_month": str(ref.replace(day=1))[:7], "config": config.to_dict(),
                "python": sys.version.split()[0],
                "dependencies": {p: importlib.metadata.version(p) for p in ("pandas", "numpy", "openpyxl")}}
    write_json(output / "run_manifest.json", manifest)
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    try:
        logger.info("Read-only source inspection: %s", input_path.name)
        tables, dictionary, structure, digest = read_workbook(input_path, config)
        manifest["source_sha256"] = digest
        write_json(output / "workbook_inspection.json", {"structure": structure, "source_sha256": digest})
        possible_sheets = [name for name, (_, fields) in tables.items()
                           if any(f["leaf"].casefold() in {"产品状态", "product status", "product_status"} for f in fields)]
        selected = config.sheet
        if selected is None:
            if len(possible_sheets) != 1:
                write_excel(output / "data_dictionary.xlsx", {"Fields": dictionary, "Structure": pd.DataFrame(structure)})
                raise SchemaError(f"No unique data sheet. Candidates: {possible_sheets}. Configure sheet after inspection.")
            selected = possible_sheets[0]
        if selected not in tables:
            raise SchemaError(f"Configured sheet not found: {selected}")
        data, fields = tables[selected]
        write_json(output / "field_candidates.json", field_candidates(data, fields))
        # A dictionary remains available even if strict field identification pauses analysis.
        write_excel(output / "data_dictionary.xlsx", {"Fields": dictionary, "Structure": pd.DataFrame(structure)})
        mapping, candidates = identify_fields(data, fields, config)
        write_json(output / "identified_fields.json", mapping)
        periods, unknown = identify_period_columns(fields, config.actual_sales_group)
        source = keyed_source(data, mapping)
        long, grain = to_long(source, periods, mapping, config)
        dictionary = annotate_dictionary(dictionary, mapping, periods, start, end, grain)
        dictionary.loc[dictionary["sheet"] != selected, ["used_for_analysis", "purpose"]] = [False, "其他工作表：未参与分析"]
        context = {"reference_date": str(ref), "analysis_start": str(start), "analysis_end": str(end),
                   "excluded_current_month": str(ref)[:7], "source_grain": grain, "source_sha256": digest,
                   "week_calendar": config.week_calendar, "adi_threshold": config.adi_threshold,
                   "cv2_threshold": config.cv2_threshold, "variance_ddof": config.variance_ddof,
                   "random_seed": config.random_seed, "snapshot_status_without_historical_effective_dates": True}
        write_excel(output / "data_dictionary.xlsx", {"Fields": dictionary, "Structure": pd.DataFrame(structure),
                    "Field candidates": candidates, "Undated sales fields": pd.DataFrame(unknown)}, context)
        write_json(output / "period_columns.json", {"dated_actual_fields": periods, "excluded_undated_fields": unknown})
        if inspect_only:
            manifest.update(status="inspection_only", identified_fields=mapping, source_grain=grain)
            write_json(output / "run_manifest.json", manifest)
            return manifest
        logger.info("Analysis period %s through %s; source grain %s", start, end, grain)
        audit = [count_stage("source_before_any_filter", source, "snapshot rows"),
                 count_stage("dated_actuals_before_date_filter", long)]
        dated = filter_dates(long, start, end)
        audit.append(count_stage("after_strict_complete_period_date_filter", dated))
        if dated.empty:
            raise ValueError("No source periods are fully inside the dynamic analysis window.")
        statuses_before_snapshot = source[["sku_id", "product_status"]].copy()
        statuses_before_snapshot["product_status"] = statuses_before_snapshot["product_status"].map(normalize_product_status)
        source_status_counts = status_counts(statuses_before_snapshot)
        source_status_counts["record_unit"] = "snapshot rows"
        normalized_dated = dated.assign(product_status=dated["product_status"].map(normalize_product_status))
        conflict_counts = normalized_dated.groupby("sku_id")["product_status"].nunique(dropna=False)
        conflicts = conflict_counts[conflict_counts > 1].index
        filtered, status_mapping, counts = filter_product_status(dated, config.allowed_product_status)
        audit.append(count_stage("after_allowed_product_status_filter", filtered))
        filtered_source = source.loc[source["product_status"].map(normalize_product_status).isin(config.allowed_product_status)]
        audit.append(count_stage("allowed_status_snapshot_rows", filtered_source, "snapshot rows"))
        clean, quality, quarantined, reasons = quality_check(filtered, grain, bool(periods), conflicts)
        audit.append(count_stage("after_quality_quarantine", clean))
        panel, analysis_grain = complete_panel(clean, grain, start, end, config.fill_internal_missing_weeks)
        context["analysis_grain"] = analysis_grain
        audit.append(count_stage("final_analysis_panel", panel))
        assert_period(panel, start, end)
        assert set(panel["product_status"].unique()) <= {"Normal", "Phasing Out"}
        audit_frame = pd.DataFrame(audit)
        exclusions = pd.DataFrame([
            {"reason": "date outside window or partial boundary period", "records": len(long) - len(dated)},
            {"reason": "product status not allowed", "records": len(dated) - len(filtered)},
            {"reason": "data quality quarantine", "records": len(filtered) - len(clean)}])
        period_audit = pd.DataFrame([{**p, "included": p["start"] >= start and p["end"] <= end and p["grain"] == grain}
                                    for p in periods])
        write_excel(output / "filter_audit.xlsx", {"Filter stages": audit_frame, "Status counts": counts,
                    "Source status counts": source_status_counts, "Status mapping": status_mapping,
                    "Exclusions": exclusions, "Source periods": period_audit}, context)
        inventory_limits = pd.DataFrame([
            {"field_family": "inventory / transit / backorder", "usable": False, "reason": "No historical as-of dates; snapshots excluded from window-limited analysis"},
            {"field_family": "company forecast", "usable": False, "reason": "No issued_at / forecast_version; historical leakage cannot be excluded"},
            {"field_family": "sales units", "usable": "unit" in mapping, "reason": "Units/conversions must be confirmed before combining SKU quantities or value analysis"}])
        write_excel(output / "data_quality_report.xlsx", {"Quality findings": quality, "Quarantined SKUs": reasons,
                    "Coverage": pd.DataFrame([{"first_date": panel["date"].min(), "last_date": panel["date"].max(),
                    "last_period_end": panel["period_end"].max(), "periods": panel["date"].nunique(),
                    "months_with_observations": panel["date"].dt.to_period("M").nunique(),
                    "zero_completed_periods": int(panel["imputed_no_sale_period"].sum())}]),
                    "Unavailable inputs": inventory_limits}, context)
        write_csv(output / "cleaned_last_12_complete_months.csv", panel)
        write_csv(output / "quarantined_records.csv", quarantined)
        classification, class_summary = classify_skus(panel, config)
        logger.info("Classified %s eligible SKUs over %s periods", len(classification), panel["date"].nunique())
        formulas = pd.DataFrame([
            {"item": "ADI", "formula_or_rule": "T / count(d_t > 0)"},
            {"item": "CV2", "formula_or_rule": "Var(D+, ddof=config.variance_ddof) / Mean(D+)^2"},
            {"item": "Boundary", "formula_or_rule": "Equal to threshold belongs to high ADI / high CV2"},
            {"item": "Singleton positive", "formula_or_rule": "CV2=0 provisional, cv2_reliable=False"},
            {"item": "Zero", "formula_or_rule": "all true observed demand equals zero"}])
        write_excel(output / "sku_classification.xlsx", {"Summary": class_summary, "SKU classification": classification, "Definitions": formulas}, context)
        outlier_summary, outlier_points = compare_outliers(panel, config)
        write_excel(output / "outlier_comparison.xlsx", {"SKU comparison": outlier_summary, "Flagged periods": outlier_points}, context)
        write_csv(output / "outlier_flagged_periods.csv", outlier_points)
        evaluation = {}
        n = int(panel["date"].nunique())
        plan = pd.DataFrame(window_plan(n, config)[3]) if analysis_grain == "weekly" else pd.DataFrame([
            {"window_weeks": w, "status": "not applicable: monthly source cannot support weekly experiments"} for w in config.windows])
        if analysis_grain == "weekly":
            logger.info("Starting rolling validation and independent 16-week test")
            evaluation = run_forecasts(panel, config, logger, ref)
            for name in ("validation_detail", "test_detail", "forecast_16_weeks", "selected_models", "window_availability"):
                write_csv(output / f"{name}.csv", evaluation[name])
        company = pd.DataFrame([{"method": "CompanyForecast", "status": "unavailable / not comparable",
                                 "reason": "Missing issued_at and archived forecast versions; current snapshot excluded"}])
        write_excel(output / "forecast_evaluation.xlsx", {
            "Selected model test": evaluation.get("selected_test_summary", pd.DataFrame()), "Window availability": plan,
            "Rolling validation": evaluation.get("validation_summary", pd.DataFrame()),
            "Candidate test comparison": evaluation.get("test_summary", pd.DataFrame()),
            "Selected SKU test": evaluation.get("selected_test", pd.DataFrame()),
            "Selected models": evaluation.get("selected_models", pd.DataFrame()),
            "Unavailable baseline": company}, context)
        text = summary_text(context, audit_frame, class_summary, panel, plan, evaluation, quality)
        (output / "analysis_summary.md").write_text(text, encoding="utf-8")
        if sha256(input_path) != digest:
            raise RuntimeError("Input workbook was modified during the run.")
        checks = verify_outputs(output, REQUIRED)
        manifest.update(status="success", finished_at_utc=datetime.now(timezone.utc).isoformat(),
                        context=context, identified_fields=mapping, filters=audit,
                        final_skus=int(panel["sku_id"].nunique()), final_records=len(panel),
                        available_periods=n, source_unchanged=True, outputs=checks)
        write_json(output / "run_manifest.json", manifest)
        logger.info("Completed and verified all %s required outputs", len(checks))
        return manifest
    except Exception as error:
        logger.exception("Analysis stopped: %s", error)
        manifest.update(status="failed", error_type=type(error).__name__, error=str(error))
        write_json(output / "run_manifest.json", manifest)
        # Previous outputs are never evidence for a failed rerun; manifest gates acceptance.
        (output / "analysis_summary.md").write_text(
            f"# 分析已暂停\n\nreference_date={ref}；analysis_start={start}；analysis_end={end}。\n\n"
            f"原因：{error}\n\n请检查日志、字段候选和配置。本次运行未通过验收；该目录中已有结果不得视为本次有效输出。\n", encoding="utf-8")
        raise
    finally:
        # Release Windows file handles so repeated runs and test cleanup are safe.
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
