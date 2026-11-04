"""Plot saved Zero-threshold comparisons without importing or fitting models."""
from pathlib import Path
import argparse
import json
import math
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HOME = Path(__file__).resolve().parent
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8')
sys.path.insert(0, str(HOME / 'engine'))
import make_combined_charts as style
GROUPS = ['Smooth', 'Erratic', 'Intermittent', 'Lumpy']

TITLES = {
    'Current_nonzero': 'Current non-Zero SKUs (sample varies by threshold)',
    'Count_eligible_nonzero': 'Count-eligible non-Zero SKUs (sample varies by threshold)',
    **{g: f'{g} group (membership can vary by threshold)' for g in GROUPS},
}

WIDTH_VIEWS = [
    ('mean_normalized_width_nonzero_median', 'width_coverage',
     'Primary: width normalized by training nonzero absolute median',
     'Mean (width / max(training nonzero absolute median, 1)) — bars / bottom axis'),
    ('mean_normalized_width_training_sd', 'width_training_sd_coverage',
     'Secondary: width normalized by training standard deviation',
     'Mean (width / max(training SD, 1)) — bars / bottom axis'),
]


def width_coverage_image(part, output, case, panel, dpi, column, suffix, title, axis_label):
    thresholds = list(part.threshold.unique())
    nrows = math.ceil(len(thresholds)/2)
    fig, axes = plt.subplots(nrows, 2, figsize=(24, 6.5*nrows), squeeze=False)
    for ax, threshold in zip(axes.flat, thresholds):
        d = part[part.threshold.eq(threshold)]
        style.TITLES[panel] = f'Zero threshold: {threshold:.0%}'
        if d.observations.max() == 0:
            ax.text(.5, .5, 'No observed points', ha='center'); ax.axis('off')
        else:
            style.plot_panel(ax, d, panel, width_column=column, width_label=axis_label)
            ax.set_xlim(0, max(1., part[column].max()*1.12))
    for ax in list(axes.flat)[len(thresholds):]:
        ax.axis('off')
    period = '2026-W25 to W42' if case == 'main_W42' else '2026-W25 to W43 (W43 may be incomplete)'
    fig.suptitle(f'{title} and coverage\n{TITLES[panel]}\nTraining: 2025-W41 to 2026-W24 | Testing: {period}', fontsize=15, y=.99)
    handles = [Patch(color=c, label=f'{level}% interval') for level,c in zip(style.LEVELS, style.INTERVAL_COLORS)]
    handles += [Line2D([],[],marker='o',linestyle='None',color=style.DOT,label='Observed coverage (top axis)')]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False)
    fig.subplots_adjust(left=.14, right=.96, bottom=.09, top=.72 if nrows==1 else .81, wspace=.60, hspace=1.25)
    style.save_png(fig, output / case / 'figures' / f'{panel}_{suffix}.png', dpi)


def plots(metrics, output, dpi):
    if set(metrics.table_id) - set(TITLES):
        raise ValueError('These metrics use the old Zero-prediction policy. Run the revised experiment first.')
    missing = {view[0] for view in WIDTH_VIEWS} - set(metrics.columns)
    if missing:
        raise ValueError('Saved metrics lack the new normalized widths. Regenerate metrics with the revised run_experiment.py before plotting.')
    style.TITLES.update(TITLES)
    for (case, panel), part in metrics.groupby(['case', 'table_id'], sort=False):
        thresholds = list(part.threshold.unique())
        for column, suffix, title, axis_label in WIDTH_VIEWS:
            width_coverage_image(part, output, case, panel, dpi, column, suffix, title, axis_label)
        period = '2026-W25 to W42' if case == 'main_W42' else '2026-W25 to W43 (W43 may be incomplete)'
        methods = [m for m in style.ALL_METHODS if m in part.method.unique()]
        fig, axes = plt.subplots(1,3,figsize=(18,5.8),squeeze=False)
        for ax, level in zip(axes.flat, style.LEVELS):
            for i, method in enumerate(methods):
                d = part[part.method.eq(method) & part.interval_level.eq(level)].sort_values('threshold')
                ax.plot(d.threshold*100, d.mean_normalized_outside_distance, marker='o', label=style.SHORT[method])
            ax.set_title(f'{level}% prediction interval', color='#17475D')
            ax.set_xticks(np.array(thresholds)*100, [f'{t:.0%}' for t in thresholds])
            ax.set_xlabel('Zero threshold'); ax.set_ylabel('Mean normalized outside distance')
            ax.grid(alpha=.18); ax.spines[['top','right']].set_visible(False)
        fig.suptitle(f'Mean normalized outside distance | {TITLES[panel]}\nTraining: 2025-W41 to 2026-W24 | Testing: {period}', fontsize=13)
        handles, labels = axes[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center',ncol=3,frameon=False)
        fig.subplots_adjust(top=.78,bottom=.23,wspace=.32)
        style.save_png(fig, output / case / 'figures' / f'{panel}_deviation.png', dpi)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HOME/'config.json')
    parser.add_argument('--results', type=Path, help='Folder containing metrics_long.csv')
    parser.add_argument('--case', choices=['all','main_W42','sensitivity_W43'], default='all')
    parser.add_argument('--panels', nargs='+', choices=list(TITLES), help='For example: Current_nonzero or Smooth Lumpy')
    parser.add_argument('--dpi', type=int)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    output = args.results.resolve() if args.results else (config_path.parent/config['output']).resolve()
    if not output.is_relative_to(HOME) or output == HOME:
        parser.error('Results must be a subfolder of test_zero_threshold.')
    dpi = args.dpi if args.dpi is not None else config.get('dpi',160)
    if dpi <= 0:
        parser.error('DPI must be positive.')
    source = output/'metrics_long.csv'
    if not source.is_file():
        parser.error(f'Saved metrics not found: {source}. Run the experiment first.')
    metrics = pd.read_csv(source)
    if set(metrics.table_id) - set(TITLES):
        parser.error('Old Zero-prediction metrics are not supported. Run the revised experiment first.')
    if args.case != 'all':
        metrics = metrics.loc[metrics.case.eq(args.case)]
    if args.panels:
        metrics = metrics.loc[metrics.table_id.isin(args.panels)]
    if metrics.empty:
        parser.error('No saved metrics match the selected case and panels.')
    plots(metrics, output, dpi)
    print('Plotting completed. Saved predictions and models were not changed.',flush=True)


if __name__ == '__main__':
    main()
