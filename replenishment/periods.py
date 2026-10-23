"""Calendar boundaries and strict interval filtering."""
from datetime import date, datetime, timedelta
import calendar
import re
import pandas as pd


def get_analysis_period(reference_date=None, months=12):
    """Return the first/last date of the preceding complete calendar months."""
    if not isinstance(months, int) or isinstance(months, bool) or months < 1:
        raise ValueError("months must be a positive integer")
    ref = pd.Timestamp(reference_date if reference_date is not None else date.today())
    if pd.isna(ref):
        raise ValueError("reference_date must be a valid date")
    if ref.tzinfo is not None:
        ref = ref.tz_localize(None)
    first = ref.normalize().replace(day=1)
    start = (first - pd.DateOffset(months=months)).date()
    end = (first - pd.Timedelta(days=1)).date()
    assert start.day == 1
    assert end.day == calendar.monthrange(end.year, end.month)[1]
    assert (end.year - start.year) * 12 + end.month - start.month + 1 == months
    assert end < first.date()
    return start, end


def parse_period_label(value):
    """Parse real year/week or year/month labels; yearless month names stay unknown."""
    if isinstance(value, (date, datetime, pd.Timestamp)):
        d = pd.Timestamp(value).date()
        return {"grain": "daily", "start": d, "end": d}
    text = str(value).strip()
    week = re.fullmatch(r"(\d{4})\s*(?:['’]|-?W|年)\s*(\d{1,2})(?:周)?", text, re.I)
    if week:
        year, number = map(int, week.groups())
        start = date.fromisocalendar(year, number, 1)  # rejects invalid ISO weeks
        return {"grain": "weekly", "start": start, "end": start + timedelta(days=6)}
    month = re.fullmatch(r"(\d{4})[-/.年](\d{1,2})(?:月)?", text)
    if month:
        year, number = map(int, month.groups())
        start = date(year, number, 1)
        return {"grain": "monthly", "start": start,
                "end": date(year, number, calendar.monthrange(year, number)[1])}
    if re.fullmatch(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", text):
        d = pd.Timestamp(text).date()
        return {"grain": "daily", "start": d, "end": d}
    return None


def filter_dates(data, analysis_start, analysis_end):
    """Keep entire source periods inside the window, including neither partial week."""
    result = data.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    if "period_end" not in result:
        result["period_end"] = result["date"]
    result["period_end"] = pd.to_datetime(result["period_end"], errors="coerce").dt.normalize()
    mask = (result["date"].ge(pd.Timestamp(analysis_start)) &
            result["period_end"].le(pd.Timestamp(analysis_end)) &
            result["period_end"].ge(result["date"]))
    result = result.loc[mask].copy()
    assert_period(result, analysis_start, analysis_end)
    return result


def assert_period(data, analysis_start, analysis_end):
    if not data.empty:
        assert data["date"].min() >= pd.Timestamp(analysis_start)
        assert data["date"].max() <= pd.Timestamp(analysis_end)
        assert data["period_end"].max() <= pd.Timestamp(analysis_end)
        assert (data["date"] <= data["period_end"]).all()
