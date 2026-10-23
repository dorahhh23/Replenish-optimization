"""All artificial demand here is a unit-test fixture, never business output."""
from datetime import date, timedelta
import logging
import tempfile
import unittest
from pathlib import Path
import numpy as np
import openpyxl
import pandas as pd
from replenishment.classification import classify_series
from replenishment.cleaning import complete_panel, numeric
from replenishment.config import Config
from replenishment.evaluation import metrics, run_forecasts, window_plan
from replenishment.excel_io import SchemaError, identifier, identify_fields, parse_headers, sha256
from replenishment.forecasting import forecast
from replenishment.outliers import outlier_versions, sens_slope
from replenishment.periods import filter_dates, get_analysis_period, parse_period_label
from replenishment.pipeline import REQUIRED, run_pipeline
from replenishment.status import filter_product_status, normalize_product_status


class PeriodTests(unittest.TestCase):
    def test_requested_example(self):
        self.assertEqual(get_analysis_period("2026-10-23"), (date(2025, 10, 1), date(2026, 9, 30)))

    def test_january_rollover(self):
        self.assertEqual(get_analysis_period("2026-01-01"), (date(2025, 1, 1), date(2025, 12, 31)))

    def test_leap_day(self):
        self.assertEqual(get_analysis_period("2024-03-31", 1), (date(2024, 2, 1), date(2024, 2, 29)))

    def test_month_invariance(self):
        for year in range(2023, 2028):
            for month in range(1, 13):
                with self.subTest(year=year, month=month):
                    self.assertEqual(get_analysis_period(date(year, month, 1)), get_analysis_period(date(year, month, 28)))

    def test_default_today(self):
        self.assertEqual(get_analysis_period(), get_analysis_period(date.today()))

    def test_invalid_months(self):
        for value in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                get_analysis_period("2026-10-23", value)

    def test_strict_interval_filter(self):
        frame = pd.DataFrame({"date": ["2025-09-29", "2025-10-01", "2026-09-21", "2026-09-28", "2026-10-01", "invalid"],
                              "period_end": ["2025-10-05", "2025-10-01", "2026-09-27", "2026-10-04", "2026-10-01", "invalid"]})
        actual = filter_dates(frame, *get_analysis_period("2026-10-23"))
        self.assertEqual(actual.index.tolist(), [1, 2])

    def test_timestamp_end_day(self):
        frame = pd.DataFrame({"date": ["2026-09-30 23:59:59", "2026-10-01 00:00:00"]})
        self.assertEqual(len(filter_dates(frame, *get_analysis_period("2026-10-23"))), 1)

    def test_real_week_and_yearless_month(self):
        self.assertEqual(parse_period_label("2026'39")["end"], date(2026, 9, 27))
        self.assertIsNone(parse_period_label("Oct"))
        self.assertEqual(parse_period_label("2025-10")["end"], date(2025, 10, 31))
        with self.assertRaises(ValueError):
            parse_period_label("2025'53")


class StatusTests(unittest.TestCase):
    def test_exact_variants(self):
        for value in ("normal", "NORMAL", " Normal  ", "\tNormal\n"):
            self.assertEqual(normalize_product_status(value), "Normal")
        for value in ("phasing out", "PHASING OUT", "  Phasing   Out  ", "phasing\tout"):
            self.assertEqual(normalize_product_status(value), "Phasing Out")

    def test_no_fuzzy_or_null_mapping(self):
        for value in (None, "", " ", np.nan, pd.NA):
            self.assertTrue(pd.isna(normalize_product_status(value)))
        for value in ("Abnormal", "Normally", "Phasing-out", "Phase out", "EOL"):
            self.assertNotIn(normalize_product_status(value), {"Normal", "Phasing Out"})

    def test_filter_audit(self):
        data = pd.DataFrame({"sku_id": list("abcdef"), "product_status": ["normal", "PHASING  OUT", "EOL", "New release", "", None]})
        result, mapping, counts = filter_product_status(data)
        self.assertEqual(result["sku_id"].tolist(), ["a", "b"])
        self.assertEqual(int(mapping["records"].sum()), 6)
        self.assertEqual(int(counts["records"].sum()), 6)

    def test_ambiguous_status_stops(self):
        fields = parse_headers([["产品状态", "Product status", "物料"]])
        data = pd.DataFrame([["Normal", "Normal", "001"]], columns=[f["cleaned_name"] for f in fields])
        with self.assertRaises(SchemaError):
            identify_fields(data, fields, Config())

    def test_missing_status_stops(self):
        fields = parse_headers([["物料", "总部跨工厂物料状态"]])
        data = pd.DataFrame([["001", "S1"]], columns=[f["cleaned_name"] for f in fields])
        with self.assertRaises(SchemaError):
            identify_fields(data, fields, Config())


class EstimationTests(unittest.TestCase):
    def setUp(self):
        self.config = Config()

    def test_headers_and_leading_zero(self):
        fields = parse_headers([["Sales", None, "物料"], ["Oct", "Oct", "物料"]], ["A1:B1"])
        self.assertEqual(fields[1]["cleaned_name"], "Sales | Oct [2]")
        self.assertEqual(identifier("00123"), "00123")
        self.assertEqual(identifier(123.0, "00000"), "00123")

    def test_numeric_no_locale_guess(self):
        values = numeric(pd.Series(["1,234.50", "123", "1.234,50", "bad", -4]))
        self.assertEqual(values.iloc[0], 1234.5)
        self.assertTrue(pd.isna(values.iloc[2]))
        self.assertEqual(values.iloc[4], -4)

    def test_five_classes(self):
        cases = {"Smooth": [10] * 20, "Erratic": [1, 100] * 10,
                 "Intermittent": [0, 10] * 10, "Lumpy": [0, 1, 0, 100] * 5, "Zero": [0] * 20}
        for kind, values in cases.items():
            self.assertEqual(classify_series(values, self.config)["demand_type"], kind)

    def test_single_positive_provisional(self):
        self.assertFalse(classify_series([0, 0, 1], self.config)["cv2_reliable"])

    def test_outlier_replacement(self):
        y = np.array([10, 11, 9, 1000, 10, 11, 9, 10], float)
        versions, rule, flagged, replaced = outlier_versions(y, self.config)
        self.assertTrue(replaced[3])
        self.assertEqual(versions["Raw"][3], 1000)
        self.assertLess(versions["Winsorized"][3], 1000)
        self.assertEqual(y[3], 1000)  # caller's actuals must never be mutated

    def test_rising_trend_not_replaced(self):
        y = np.array([1, 2, 3, 4, 5, 6, 7, 1000], float)
        versions, rule, flagged, replaced = outlier_versions(y, self.config)
        self.assertTrue(rule["rising_trend"])
        self.assertTrue(flagged[-1])
        self.assertFalse(replaced.any())
        np.testing.assert_array_equal(versions["Median"], y)
        self.assertEqual(sens_slope([2, 4, 6]), 2)

    def test_model_equations(self):
        for method in ("MovingAverage", "SES", "Croston", "TSB"):
            np.testing.assert_allclose(forecast([10] * 26, method, 16), 10)
        np.testing.assert_allclose(forecast([10] * 26, "SBA", 16), 9.5)
        for method in ("MovingAverage", "SES", "Croston", "SBA", "TSB"):
            np.testing.assert_array_equal(forecast([0] * 26, method, 16), np.zeros(16))
        self.assertAlmostEqual(forecast([10, 0, 0], "TSB", 1)[0], 8.1)
        # Interval between positive events at indices 0 and 2 is two periods.
        self.assertAlmostEqual(forecast([10, 0, 10], "Croston", 1)[0], 10 / 1.1)

    def test_metrics_zero_denominator(self):
        score = metrics([0, 0], [1, 1])
        self.assertEqual(score["mae"], 1)
        self.assertTrue(np.isnan(score["wape"]))
        self.assertEqual(score["bias"], 1)

    def test_insufficient_52_weeks(self):
        windows, origins, test_origin, plan = window_plan(45, self.config)
        self.assertEqual(windows, [13, 26])
        self.assertEqual(origins, [26, 27, 28])
        self.assertEqual(test_origin, 29)
        self.assertFalse(next(p for p in plan if p["window_weeks"] == 52)["enough_final_training"])

    def test_weekly_unknown_coverage_is_not_zero(self):
        data = pd.DataFrame({"sku_id": ["001", "001"], "material": ["001", "001"],
                             "sales_org": ["org"] * 2, "storage_location": ["loc"] * 2,
                             "product_status": ["Normal"] * 2, "unit": ["EA"] * 2,
                             "date": pd.to_datetime(["2026-01-05", "2026-01-19"]),
                             "period_end": pd.to_datetime(["2026-01-11", "2026-01-25"]), "sales": [1, 2]})
        with self.assertRaises(SchemaError):
            complete_panel(data, "weekly", date(2025, 10, 1), date(2026, 9, 30))

    def test_test_actuals_do_not_change_selection(self):
        def panel(y):
            dates = pd.date_range("2025-11-17", periods=len(y), freq="W-MON")
            return pd.DataFrame({"sku_id": "test", "material": "001", "product_status": "Normal", "date": dates,
                                 "period_end": dates + pd.Timedelta(days=6), "sales": y})
        a = np.arange(45, dtype=float) % 5
        b = a.copy()
        b[29:] = 10000
        one = run_forecasts(panel(a), self.config, logging.getLogger("test"), "2026-10-23")
        two = run_forecasts(panel(b), self.config, logging.getLogger("test"), "2026-10-23")
        cols = ["window_weeks", "version", "method", "validation_mae"]
        pd.testing.assert_frame_equal(one["selected_models"][cols], two["selected_models"][cols])
        pd.testing.assert_frame_equal(one["validation_detail"], two["validation_detail"])
        self.assertEqual(one["forecast_16_weeks"]["week_start"].min(), pd.Timestamp("2026-10-26"))
        self.assertEqual(one["forecast_16_weeks"]["week_start"].nunique(), 16)


def make_fixture(path, monthly=False):
    """Small, clearly isolated synthetic source used only to exercise code paths."""
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "Test"
    dates = pd.date_range("2025-10-01", periods=13, freq="MS") if monthly else pd.date_range("2025-11-17", periods=49, freq="W-MON")
    labels = [str(d.to_period("M")) if monthly else f"{d.isocalendar().year}'{d.isocalendar().week}" for d in dates]
    sheet.append(["Metadata"] * 4 + ["Actual"] * len(labels))
    sheet.append(["Material", "Sales organization", "Storage location", "Product status"] + labels)
    for sku, status, amount in [("001", "NORMAL", 10), ("002", "phasing  OUT", 5), ("003", "Normal", 0),
                                ("004", "EOL", 99), ("005", "Normal", -1), ("006", "Normal", 3), ("006", "Normal", 4)]:
        sheet.append([sku, "Org", "", status] + [amount] * len(labels))
    book.save(path)


class PipelineTests(unittest.TestCase):
    def test_real_run_and_repeat_source_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source = folder / "synthetic_unit_test_input.xlsx"
            make_fixture(source)
            before = sha256(source)
            cfg = Config(sheet="Test", actual_sales_group="Actual")
            first = run_pipeline(source, folder / "output", cfg, "2026-10-23")
            self.assertEqual(first["status"], "success")
            self.assertEqual(first["final_skus"], 3)
            self.assertEqual(first["available_periods"], 45)
            self.assertEqual(before, sha256(source))
            for filename in REQUIRED:
                self.assertTrue((folder / "output" / filename).is_file())
            cleaned = pd.read_csv(folder / "output" / "cleaned_last_12_complete_months.csv", dtype={"material": "string"})
            self.assertEqual(set(cleaned["material"]), {"001", "002", "003"})
            self.assertEqual(set(cleaned["product_status"]), {"Normal", "Phasing Out"})
            repeat = run_pipeline(source, folder / "output", cfg, "2026-10-23")
            self.assertEqual(repeat["filters"], first["filters"])

    def test_monthly_branch_never_generates_weekly_sales(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source = folder / "synthetic_unit_test_input.xlsx"
            make_fixture(source, monthly=True)
            result = run_pipeline(source, folder / "output", Config(sheet="Test", actual_sales_group="Actual"), "2026-10-23")
            self.assertEqual(result["context"]["analysis_grain"], "monthly")
            self.assertEqual(result["available_periods"], 12)
            self.assertFalse((folder / "output" / "forecast_16_weeks.csv").exists())


if __name__ == "__main__":
    unittest.main()
