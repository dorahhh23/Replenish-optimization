"""Validated, portable configuration. No date ranges are hardcoded."""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass
class Config:
    months: int = 12
    allowed_product_status: list[str] = field(default_factory=lambda: ["Normal", "Phasing Out"])
    header_rows: int = 2
    sheet: str | None = None
    fields: dict[str, str] = field(default_factory=dict)
    excluded_hierarchy_levels: list[str] = field(default_factory=lambda: ["Son"])
    sales_measure: str = "net_sales_quantity"
    actual_sales_group: str = "实际销量"
    company_forecast_group: str = "预测销量"
    week_calendar: str = "ISO"
    grain: str = "auto"
    adi_threshold: float = 1.32
    cv2_threshold: float = 0.49
    variance_ddof: int = 0
    iqr_multiplier: float = 1.5
    min_positive_for_outliers: int = 4
    trend_slope_threshold: float = 0.0
    median_increase_threshold: float = 0.0
    windows: list[int] = field(default_factory=lambda: [13, 26, 52])
    preferred_window: int = 26
    forecast_horizon: int = 16
    validation_horizon: int = 1
    min_validation_origins: int = 3
    max_validation_origins: int = 8
    validation_step: int = 1
    smoothing_alpha: float = 0.1
    tsb_beta: float = 0.1
    phasing_growth_ratio: float = 1.25
    trailing_zero_review_periods: int = 4
    random_seed: int = 42
    # An absent source column is unknown coverage; never invent demand for it.
    fill_internal_missing_weeks: bool = True
    negative_demand_policy: str = "retain_net_sales"
    duplicate_policy: str = "quarantine_sku"

    def validate(self):
        if self.months != 12:
            raise ValueError("This study requires exactly 12 complete calendar months.")
        allowed = {"Normal", "Phasing Out"}
        if not self.allowed_product_status or not set(self.allowed_product_status) <= allowed:
            raise ValueError("Allowed statuses must be a nonempty subset of Normal / Phasing Out.")
        if self.week_calendar != "ISO":
            raise ValueError("Only ISO weeks are implemented. Supply a verified calendar before changing this.")
        if self.grain not in {"auto", "weekly", "monthly", "daily", "transaction"}:
            raise ValueError("Unsupported time grain.")
        if self.negative_demand_policy != "retain_net_sales" or self.sales_measure != "net_sales_quantity":
            raise ValueError("Source is confirmed net sales; retain signed actuals without deducting returns again.")
        if self.duplicate_policy != "quarantine_sku":
            raise ValueError("Unresolved duplicate snapshots must be quarantined after Son exclusion.")
        if {str(x).strip().casefold() for x in self.excluded_hierarchy_levels} != {"son"}:
            raise ValueError("The confirmed hierarchy rule is to exclude Son only.")
        if self.header_rows < 1 or self.variance_ddof not in (0, 1):
            raise ValueError("Invalid header or variance setting.")
        if not 0 < self.smoothing_alpha <= 1 or not 0 < self.tsb_beta <= 1:
            raise ValueError("Smoothing parameters must be in (0, 1].")
        if any(w < 1 for w in self.windows) or not self.windows:
            raise ValueError("History windows must be positive.")
        if self.forecast_horizon != 16:
            raise ValueError("The requested forecast horizon is 16 weeks.")
        if self.validation_horizon < 1 or self.min_validation_origins < 1 or self.validation_step < 1:
            raise ValueError("Invalid rolling-origin settings.")
        if self.max_validation_origins < self.min_validation_origins:
            raise ValueError("max_validation_origins must be >= min_validation_origins.")
        if self.adi_threshold <= 0 or self.cv2_threshold <= 0 or self.iqr_multiplier < 0:
            raise ValueError("Invalid classification/outlier thresholds.")
        return self

    def to_dict(self):
        return asdict(self)


def load_config(path=None):
    if path is None:
        return Config().validate()
    content = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return Config(**content).validate()
