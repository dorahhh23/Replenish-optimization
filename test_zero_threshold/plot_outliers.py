"""Draw signed outside-distance histograms for non-Zero SKUs from saved results."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from plot_results import HOME, GROUPS, style

COLORS = ['#26758F', '#C2762D', '#449B85', '#8864AD']


def transform(values):
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.log1p(np.abs(values) / .1)


def inverse_transform(values):
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.expm1(np.abs(values)) * .1


def bin_edges(values, bins):
    if bins < 10:
        raise ValueError('Use at least 10 bins.')
    if not len(values):
        return np.linspace(-1., 1., bins + 1)
    z = transform(values)
    low, high = min(0., float(z.min()))*1.015, max(0., float(z.max()))*1.015
    if low < 0 < high:
        left = max(1, min(bins-1, round(bins * -low / (high-low))))
        return np.r_[np.linspace(low, 0., left+1), np.linspace(0., high, bins-left+1)[1:]]
    return np.linspace(low, high, bins+1)


def make_histogram(data, output, case, threshold, level, panel, methods, bins=72, dpi=160):
    if not set(data.demand_group) <= set(GROUPS):
        raise ValueError('Zero or unknown groups found in outside points. Regenerate results.')
    data = data.loc[data.method.isin(methods)]
    values = data.signed_normalized_nonzero_median.to_numpy(float)
    if not np.isfinite(values).all() or (values == 0).any():
        raise ValueError('Outside distances must be finite and nonzero.')
    edges = bin_edges(values, bins)
    raw_edges = inverse_transform(edges)
    z = transform(values)
    candidates = np.array([-100000.,-10000.,-1000.,-100.,-10.,-1.,0.,1.,10.,100.,1000.,10000.,100000.])
    positions = transform(candidates)
    keep = (positions >= edges[0]) & (positions <= edges[-1])
    columns = 2 if len(methods) == 4 else 3
    fig, axes = plt.subplots(2, columns, figsize=(6*columns, 10), sharex=True, sharey=True)
    records, summaries = [], []
    total_counted = 0
    for ax, method in zip(axes.flat, methods):
        method_mask = data.method.eq(method).to_numpy()
        local = values[method_mask]
        bottom = np.zeros(len(edges)-1, dtype=int)
        for group, color in zip(GROUPS, COLORS):
            mask = method_mask & data.demand_group.eq(group).to_numpy()
            counts, _ = np.histogram(z[mask], bins=edges)
            assert counts.sum() == mask.sum(), 'Histogram dropped an outside point.'
            total_counted += int(counts.sum())
            ax.bar(edges[:-1], counts, width=np.diff(edges), align='edge',
                   bottom=bottom, color=color, edgecolor='none', linewidth=0)
            bottom += counts
            for i, count in enumerate(counts):
                records.append(dict(threshold=threshold,case=case,table_id=panel,interval_level=level,
                    method=method,demand_group=group,bin_index=i,transformed_left=edges[i],
                    transformed_right=edges[i+1],signed_distance_left=raw_edges[i],
                    signed_distance_right=raw_edges[i+1],count=int(count)))
        below, above = int((local<0).sum()), int((local>0).sum())
        ax.axvline(0,color='#D97732',linewidth=.8,linestyle=(0,(3,3)),zorder=5)
        ax.set_title(f'{style.LABELS[method]}\nOutside points: {len(local):,} | Below: {below:,} | Above: {above:,}',
                     loc='left',fontsize=10.5,pad=10)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.16); ax.set_axisbelow(True)
        ax.set_xlim(edges[0],edges[-1])
        ax.set_xticks(positions[keep],[f'{v:g}' for v in candidates[keep]],fontsize=9)
        ax.tick_params(axis='y',labelsize=9)
        summaries.append(dict(threshold=threshold,case=case,table_id=panel,interval_level=level,
            method=method,outside_points=len(local),below_lower_bound=below,above_upper_bound=above,
            negative_actual_outside_points=int(data.loc[method_mask,'actual_sales'].lt(0).sum())))
    assert total_counted == len(data)
    period = '2026-W25 to W42' if case == 'main_W42' else '2026-W25 to W43 (W43 may be incomplete)'
    fig.suptitle(f'Outside-point distribution | {level}% interval | Zero threshold: {threshold:.0%}\n'
                 f'Training: 2025-W41 to 2026-W24 | Testing: {period}',fontsize=13,y=.995)
    fig.supylabel('Outside points per transformed bin',x=.008)
    fig.supxlabel('Signed normalized outside distance (symmetric log1p scale)',y=.097)
    fig.text(.5,.064,'Left of 0: below interval | Right of 0: above interval | Zero SKUs excluded',
             ha='center',fontsize=9,color='#4D5962')
    sample = 'Current non-Zero sample' if panel == 'Current_nonzero' else 'Current count-eligible non-Zero sample'
    fig.text(.5,.041,f'{sample} | Same evaluated sample across methods within this panel | All outside points retained',
             ha='center',fontsize=8.5,color='#4D5962')
    fig.legend(handles=[Patch(color=c,label=g) for c,g in zip(COLORS,GROUPS)],
               loc='lower center',ncol=4,frameon=False,fontsize=10)
    fig.tight_layout(rect=(.025,.13,1,.93),h_pad=2.5)
    output.mkdir(parents=True,exist_ok=True)
    stem = output/f'{panel}_histogram_{level}'
    style.save_png(fig,stem.with_suffix('.png'),dpi)
    pd.DataFrame(records).to_csv(stem.with_name(stem.name+'_counts.csv'),index=False,encoding='utf-8-sig')
    pd.DataFrame(summaries).to_csv(stem.with_name(stem.name+'_summary.csv'),index=False,encoding='utf-8-sig')
    audit = dict(threshold=threshold,case=case,table_id=panel,interval_level=level,
        source_rows=len(data),histogram_count=total_counted,groups=GROUPS,
        zero_excluded=True,no_tail_clipping=True,no_downsampling=True,
        normalization='distance / max(median(abs(nonzero training sales)), 1)',
        minimum_signed_distance=float(values.min()) if len(values) else None,
        maximum_signed_distance=float(values.max()) if len(values) else None)
    stem.with_suffix('.audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')


def plot_saved_outliers(output, case='all', levels=(70,80,90), thresholds=None, bins=72, dpi=160):
    output = Path(output).resolve()
    if not output.is_relative_to(HOME) or output == HOME or output.is_relative_to(HOME/'results'):
        raise ValueError('Use a revised results folder under test_zero_threshold, not the historical results folder.')
    found = 0
    for folder in sorted(output.glob('zero_*')):
        if not (folder/'settings.json').is_file():
            continue
        settings = json.loads((folder/'settings.json').read_text(encoding='utf-8'))
        if settings.get('zero_policy') != 'Excluded from fitting, prediction, and evaluation':
            raise ValueError('Old Zero-prediction results are not supported.')
        threshold = settings['threshold']
        if thresholds is not None and not any(np.isclose(threshold,t) for t in thresholds):
            continue
        for case_name in ('main_W42','sensitivity_W43'):
            if case != 'all' and case != case_name:
                continue
            source = folder/case_name/'all_outside_points.csv.gz'
            if not source.is_file():
                continue
            points = pd.read_csv(source,dtype={'sku_id':str})
            if not set(points.demand_group) <= set(GROUPS):
                raise ValueError('Outside points contain Zero or unknown groups.')
            zero_ids = set(pd.read_csv(folder/'zerolist.csv',dtype={'sku_id':str}).sku_id)
            if set(points.sku_id) & zero_ids:
                raise ValueError('Zero-list SKU appears in outside points.')
            for level in levels:
                data = points.loc[points.interval_level.eq(level)]
                for panel, eligible, methods in [
                    ('Current_nonzero',data,style.SIGNED_METHODS),
                    ('Count_eligible_nonzero',data.loc[data.six_method_eligible],style.ALL_METHODS)]:
                    make_histogram(eligible,folder/case_name/'histograms',case_name,threshold,
                                   level,panel,methods,bins,dpi)
                    found += 1
    if not found:
        raise ValueError('No revised outside-point results match the selection. Run the experiment first.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=HOME/'config.json')
    parser.add_argument('--results',type=Path)
    parser.add_argument('--case',choices=['all','main_W42','sensitivity_W43'],default='all')
    parser.add_argument('--levels',type=int,nargs='+',choices=[70,80,90],default=[70,80,90])
    parser.add_argument('--thresholds',type=float,nargs='+',help='Fractions or percentages, e.g. 60 or 0.6')
    parser.add_argument('--bins',type=int,default=72)
    parser.add_argument('--dpi',type=int)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    output = args.results.resolve() if args.results else (config_path.parent/config['output']).resolve()
    thresholds = [t/100 if t>1 else t for t in args.thresholds] if args.thresholds else None
    dpi = args.dpi if args.dpi is not None else config['dpi']
    if args.bins < 10 or dpi < 72:
        parser.error('Use at least 10 bins and DPI of at least 72.')
    plot_saved_outliers(output,args.case,args.levels,thresholds,args.bins,dpi)


if __name__ == '__main__':
    main()
