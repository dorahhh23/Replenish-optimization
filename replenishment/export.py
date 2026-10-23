"""Portable Python report exports with atomic file replacement and safe literal text."""
from datetime import date, datetime
import json
import math
import os
from pathlib import Path
import numpy as np
import openpyxl
import pandas as pd
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


def json_default(value):
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is pd.NA:
        return None
    raise TypeError(f"Cannot serialize {type(value)}")


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    os.replace(temp, path)


def cell_value(value):
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return "infinity" if math.isinf(value) else None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=json_default)
    return value


def write_excel(path, sheets, context=None):
    """Static reports; all calculations live in the reproducible Python pipeline."""
    path = Path(path)
    book = openpyxl.Workbook(write_only=True)
    if context:
        sheets = {**sheets, "Run": pd.DataFrame([{"setting": k, "value": v} for k, v in context.items()])}
    for name, frame in sheets.items():
        sheet = book.create_sheet(name[:31])
        sheet.sheet_view.showGridLines = False
        sheet.freeze_panes = "B2"
        if frame is None or (frame.empty and len(frame.columns) == 0):
            frame = pd.DataFrame({"status": ["No applicable records"]})
        frame = frame.copy()
        # Width uses header and a sample of actual values. Wrap long audit text.
        for i, col in enumerate(frame.columns, 1):
            sample = [str(col)] + [str(cell_value(x) or "") for x in frame[col].head(50)]
            width = min(65, max(14, max(min(len(x), 60) for x in sample) + 2))
            sheet.column_dimensions[get_column_letter(i)].width = width
        header = []
        for column in frame.columns:
            cell = WriteOnlyCell(sheet, str(column))
            cell.font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="23435B")
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            header.append(cell)
        sheet.row_dimensions[1].height = 32
        sheet.append(header)
        for values in frame.itertuples(index=False, name=None):
            row = []
            for column, value in zip(frame.columns, values):
                value = cell_value(value)
                cell = WriteOnlyCell(sheet, value)
                if isinstance(value, str):
                    # Prevent source text beginning '=' from becoming executable Excel formulas.
                    cell.data_type = "s"
                if isinstance(value, (date, datetime)):
                    cell.number_format = "yyyy-mm-dd"
                elif isinstance(value, float):
                    cell.number_format = "0.0%" if any(k in str(column) for k in ("share", "rate", "wape", "bias_pct")) else "#,##0.000"
                row.append(cell)
            sheet.append(row)
        if len(frame):
            sheet.auto_filter.ref = f"A1:{get_column_letter(len(frame.columns))}{len(frame) + 1}"
    temp = path.with_name(path.stem + ".tmp.xlsx")
    book.save(temp)
    os.replace(temp, path)


def write_csv(path, frame):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    frame.to_csv(temp, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    os.replace(temp, path)


def verify_outputs(output, required):
    checks = []
    for filename in required:
        path = Path(output) / filename
        if not path.is_file() or not path.stat().st_size:
            raise RuntimeError(f"Missing or empty output: {filename}")
        if path.suffix == ".xlsx":
            book = openpyxl.load_workbook(path, read_only=True, data_only=False)
            try:
                for sheet in book:
                    if not list(sheet.iter_rows(min_row=1, max_row=1, values_only=True)):
                        raise RuntimeError(f"Empty report header: {filename}/{sheet.title}")
                    for row in sheet.iter_rows():
                        for cell in row:
                            if cell.data_type in {"e", "f"}:
                                raise RuntimeError(f"Unexpected error/formula in static report: {filename}/{sheet.title}/{cell.coordinate}")
            finally:
                book.close()
        checks.append({"file": filename, "bytes": path.stat().st_size, "verified": True})
    return checks
