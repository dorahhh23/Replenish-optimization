#!/usr/bin/env python3
"""Regenerate all current five-group images from existing results."""
from pathlib import Path
import json
import subprocess
import sys
import pandas as pd
import plot_all_figures as plots


def main():
    home=Path(__file__).resolve().parent
    subprocess.run([sys.executable,str(home/'make_combined_charts.py')],check=True)
    audits=[]
    for case in plots.CASE_LABELS:
        root=home/'results'/case
        if not root.is_dir():continue
        points=pd.read_csv(root/'all_outside_points.csv.gz',dtype={'sku_id':str})
        weeks=sorted(pd.read_csv(root/'weekly_group_metrics.csv',usecols=['iso_week']).iso_week.unique())
        _,audit=plots.plot_outside_points(points,weeks,home/'figures'/case/'outside_points',case,
                                          list(plots.NORMS),plots.LEVELS,160)
        audits+=audit
    (home/'figures/plot_audit.json').write_text(json.dumps(audits,indent=2),encoding='utf-8')
    subprocess.run([sys.executable,str(home/'plot_signed_scatter.py')],check=True)
    subprocess.run([sys.executable,str(home/'plot_outlier_histogram.py'),
                    '--levels','70','80','90','--normalizations',*plots.NORMS],check=True)


if __name__=='__main__':main()
