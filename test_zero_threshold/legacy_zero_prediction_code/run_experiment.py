"""Refit Experiment B independently at each Zero threshold; audit fixed cohorts."""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import math
import platform
import sys
import time

HOME = Path(__file__).resolve().parent
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8')
sys.path.insert(0, str(HOME / '_vendor'))
sys.path.insert(0, str(HOME / 'engine'))
import numpy as np
import pandas as pd
import lightgbm as lgb
import scipy
import matplotlib
matplotlib.use('Agg')
import run_b_experiment as engine
from plot_results import plots

GROUPS = ['Smooth', 'Erratic', 'Intermittent', 'Lumpy', 'Zero']
PRIMARY = engine.BASE + ['LightGBM']



def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=True), encoding='utf-8')


def profiles_for(wide, dates, threshold):
    return pd.DataFrame([dict(sku_id=sku, **engine.profile(row, threshold))
                         for sku, row in zip(wide.index, wide[dates].to_numpy(float))])


def cohorts(meta, fixed60, count_ids):
    result = [
        ('All_SKUs', set(meta.sku_id), PRIMARY, True),
        ('Current_nonzero', set(meta.loc[meta.demand_group.ne('Zero'), 'sku_id']), PRIMARY, False),
        ('Fixed_60_nonzero', fixed60, PRIMARY, True),
        ('Common_count', count_ids, engine.MODELS, True),
        ('Common_count_fixed_60_nonzero', count_ids & fixed60, engine.MODELS, True),
    ]
    result += [(g, set(meta.loc[meta.demand_group.eq(g), 'sku_id']), PRIMARY, False) for g in GROUPS]
    return result


def summarize(pred, meta, fixed60, count_ids, threshold, case):
    rows = []
    scored = pred.loc[pred.actual_sales.notna()]
    for panel, ids, methods, fixed in cohorts(meta, fixed60, count_ids):
        for method in methods:
            d = scored.loc[scored.method.eq(method) & scored.sku_id.isin(ids)]
            y = d.actual_sales.to_numpy(float)
            for level, (lo, hi) in engine.INTERVALS.items():
                lower, upper = d[lo].to_numpy(float), d[hi].to_numpy(float)
                outside = np.maximum.reduce([lower-y, y-upper, np.zeros(len(y))])
                norm = outside / d.nonzero_median_divisor.to_numpy(float)
                rows.append(dict(threshold=threshold, case=case, table_id=panel,
                    sample_policy='Fixed across thresholds' if fixed else 'Varies with threshold',
                    method=method, interval_level=level, cohort_skus=len(ids),
                    skus=d.sku_id.nunique(), observations=len(d), negative_actuals=int((y<0).sum()),
                    coverage=float(np.mean(outside == 0)) if len(d) else np.nan,
                    mean_width=float(np.mean(upper-lower)) if len(d) else np.nan,
                    mean_normalized_outside_distance=float(np.mean(norm)) if len(d) else np.nan,
                    outside_points=int((outside>0).sum())))
    return pd.DataFrame(rows)


def verify_predictions(pred, raw, train_dates, dates, meta, count_ids):
    """Check raw actuals, exact cohorts, independent scales and Zero overrides."""
    assert not pred.duplicated(['sku_id', 'iso_week', 'method']).any()
    assert np.isfinite(pred[engine.QNAMES].to_numpy()).all()
    assert (np.diff(pred[engine.QNAMES].to_numpy(), axis=1) >= 0).all()
    assert pred.loc[pred.zero_rule, engine.QNAMES].eq(0).all().all()
    assert pred.demand_group.equals(pred.sku_id.map(meta.set_index('sku_id').demand_group))
    # Build independently from the original input, retaining NaNs and all returns.
    reference = raw.loc[meta.sku_id, dates].copy()
    reference.columns = [engine.week_label(d) for d in dates]
    reference.index.name = 'sku_id'
    expected = reference.rename_axis(columns='iso_week').stack(future_stack=True)
    expected = expected.sort_index()
    for method in PRIMARY:
        d = pred.loc[pred.method.eq(method)]
        actual = d.set_index(['sku_id', 'iso_week']).actual_sales.sort_index()
        pd.testing.assert_series_equal(actual, expected, check_names=False)
    for method in engine.COUNT:
        d = pred.loc[pred.method.eq(method) & pred.sku_id.isin(count_ids)]
        actual = d.set_index(['sku_id', 'iso_week']).actual_sales.sort_index()
        pd.testing.assert_series_equal(actual, expected.loc[expected.index.get_level_values(0).isin(count_ids)], check_names=False)
    scales = {}
    for sku in meta.sku_id:
        z = raw.loc[sku, train_dates].to_numpy(float)
        scales[sku] = max(float(np.median(np.abs(z[z != 0]))) if np.any(z != 0) else 0., 1.)
    np.testing.assert_allclose(pred.nonzero_median_divisor, pred.sku_id.map(scales), rtol=0, atol=0)
    valid = expected.dropna()
    fingerprint = hashlib.sha256(pd.util.hash_pandas_object(valid).values.tobytes()).hexdigest()
    return dict(case_observations=len(valid), sku_week_actual_sha256=fingerprint,
                negative_actuals_per_primary_method=int((valid < 0).sum()),
                missing_actuals_not_scored=int(expected.isna().sum()),
                maximum_actual=float(valid.max()), minimum_actual=float(valid.min()),
                zero_intervals_verified=True, signed_actuals_match_input=True,
                count_methods_share_fixed_applicable_sample=True, normalization_verified=True)


def write_tables(metrics, output):
    engine.write_csv(metrics, output / 'metrics_long.csv')
    rows = []
    keys = ['threshold', 'case', 'table_id', 'sample_policy', 'method']
    for key, part in metrics.groupby(keys, sort=False):
        first = part.iloc[0]
        row = dict(zip(['Zero Threshold', 'Case', 'Table', 'Sample Policy', 'Model'], key))
        row.update({'Cohort SKUs': int(first.cohort_skus), 'Scored SKUs': int(first.skus),
                    'Observed Points': int(first.observations), 'Negative Actuals': int(first.negative_actuals)})
        for r in part.itertuples():
            row[f'{r.interval_level}% Coverage'] = r.coverage
            row[f'{r.interval_level}% Mean Width'] = r.mean_width
            row[f'{r.interval_level}% Mean Normalized Outside Distance'] = r.mean_normalized_outside_distance
        rows.append(row)
    combined = pd.DataFrame(rows)
    engine.write_csv(combined, output / 'comparison_all_tables.csv')
    engine.write_csv(combined[combined.Table.eq('All_SKUs')], output / 'overall_comparison.csv')
    for (case, panel), part in combined.groupby(['Case', 'Table']):
        engine.write_csv(part, output / case / 'tables' / f'{panel}.csv')
        keep = ['Zero Threshold', 'Case', 'Table', 'Sample Policy', 'Model', 'Cohort SKUs', 'Scored SKUs', 'Observed Points', 'Negative Actuals']
        engine.write_csv(part[keep + [c for c in part if 'Coverage' in c or 'Mean Width' in c]],
                         output / case / 'coverage_width_tables' / f'{panel}.csv')
        engine.write_csv(part[keep + [c for c in part if 'Normalized Outside' in c]],
                         output / case / 'deviation_tables' / f'{panel}.csv')



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HOME/'config.json')
    parser.add_argument('--thresholds', nargs='+', type=float, help='Fractions or percentages, e.g. 50 60 70 80')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--plots-only', action='store_true')
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    thresholds = args.thresholds or config['thresholds']
    thresholds = sorted(set(t/100 if t>1 else t for t in thresholds))
    if not thresholds or any(not 0<t<=1 for t in thresholds):
        parser.error('Thresholds must be in (0, 1] or (1, 100].')
    source = (config_path.parent/config['input']).resolve()
    output = (args.output.resolve() if args.output else (config_path.parent/config['output']).resolve())
    if not output.is_relative_to(HOME) or output == HOME:
        parser.error('Output must be a subfolder of test_zero_threshold to protect original results.')
    output.mkdir(parents=True,exist_ok=True)
    if args.plots_only:
        plots(pd.read_csv(output/'metrics_long.csv'),output,config['dpi']); return
    started = time.time()
    print(f'Python: {sys.executable}\nInput: {source}\nOutput: {output}', flush=True)
    # Hash original project results and code before/after to verify preservation.
    protected = [p for folder in ['results','revised_charts','figures','figures_histogram','figures_signed_scatter']
                 for p in (HOME.parent/folder).rglob('*') if p.is_file()]
    protected += list(HOME.parent.glob('*.py')) + [source]
    original_hashes = {str(p.relative_to(HOME.parent)): digest(p) for p in protected}
    save_json(original_hashes, output/'original_files_sha256.json')
    raw = pd.read_csv(source,dtype={'sku_id':str}).set_index('sku_id')
    assert raw.index.is_unique and raw.index.notna().all(), 'SKU IDs must be unique and present.'
    raw.columns = pd.DatetimeIndex([engine.week_date(c) for c in raw.columns])
    assert raw.columns.is_unique
    assert not np.isinf(raw.to_numpy(float)).any(), 'Infinite sales are invalid.'
    train_dates = pd.date_range(engine.week_date('2025-W41'),engine.week_date('2026-W24'),freq='7D')
    main_dates = pd.date_range(engine.week_date('2026-W25'),engine.week_date('2026-W42'),freq='7D')
    assert len(train_dates)==36 and len(main_dates)==18
    assert set(train_dates)|set(main_dates) <= set(raw.columns), 'Required week columns missing.'
    complete = raw[train_dates].notna().all(axis=1)
    wide = raw.loc[complete]
    engine.write_csv(pd.DataFrame({'sku_id':raw.index[~complete], 'reason':'Incomplete training; missing is not zero'}),output/'excluded_skus.csv')
    reference = profiles_for(wide, train_dates, .6)
    fixed60 = set(reference.loc[reference.demand_group.ne('Zero'),'sku_id'])
    count_ids = set(reference.loc[reference.count_training_eligible,'sku_id'])
    membership = pd.DataFrame({'sku_id':wide.index})
    membership['fixed_60_nonzero'] = membership.sku_id.isin(fixed60)
    membership['count_training_eligible'] = membership.sku_id.isin(count_ids)
    cases = [('main_W42', main_dates)]
    if config['include_week43_sensitivity']:
        assert engine.week_date('2026-W43') in raw.columns
        cases.append(('sensitivity_W43', pd.date_range(main_dates[0], engine.week_date('2026-W43'), freq='7D')))
    target_dates = cases[-1][1]
    horizons = np.arange(1,len(target_dates)+1)
    all_metrics, counts, audits, training_runs = [], [], [], []
    for threshold in thresholds:
        tag = f'zero_{threshold*100:g}'.replace('.', 'p')
        folder = output/tag; folder.mkdir(exist_ok=True)
        print(f'\n=== Zero threshold {threshold:.0%}: fresh model fitting ===',flush=True)
        meta = profiles_for(wide, train_dates, threshold)
        minimum = math.ceil(36*threshold-1e-12)
        assert meta.demand_group.eq('Zero').equals(meta.zero_weeks.ge(minimum))
        assert set(meta.demand_group)<=set(GROUPS)
        assert meta.sku_id.tolist()==reference.sku_id.tolist()
        engine.write_csv(meta,folder/'training_profiles.csv')
        membership[tag] = membership.sku_id.map(meta.set_index('sku_id').demand_group)
        for group in GROUPS:
            sub = meta[meta.demand_group.eq(group)]
            counts.append(dict(threshold=threshold,minimum_zero_weeks=minimum,demand_group=group,
                               skus=len(sub),skus_with_negative_training=int(sub.has_negative_training.sum())))
        light, logs = engine.fit_lightgbm(wide, meta, train_dates, horizons, folder/'models',
                                        config['max_rounds'],config['threads'],config['seed'])
        for log in logs:
            if log['group']!='Zero':
                assert log['final_fit_end']=='2026-W24'
                assert engine.week_date(log['last_internal_fit_target']) < engine.week_date(log['first_validation_target'])
        save_json(logs, folder/'training_log.json')
        training_runs.append(dict(threshold=threshold,training_log=logs))
        pred = engine.generate_predictions(wide,meta,train_dates,target_dates,light,horizons)
        # Count distributions are compared on a strict, training-only fixed cohort.
        # Zero overrides outside that cohort remain represented by the four primary methods.
        pred = pred.loc[~pred.method.isin(engine.COUNT) | pred.sku_id.isin(count_ids)].copy()
        for case, dates in cases:
            p = pred.loc[pred.week_start.isin(dates)]
            audit = verify_predictions(p,raw,train_dates,dates,meta,count_ids)
            audit.update(threshold=threshold,case=case)
            audits.append(audit)
            metrics = summarize(p,meta,fixed60,count_ids,threshold,case)
            all_metrics.append(metrics)
            case_folder = folder/case; case_folder.mkdir(exist_ok=True)
            engine.write_csv(metrics,case_folder/'metrics_long.csv')
        pred.to_csv(folder/'predictions.csv.gz',index=False,encoding='utf-8-sig',compression='gzip')
        save_json(dict(threshold=threshold,minimum_zero_weeks=minimum,training_weeks=36,
                       training_start='2025-W41',training_end='2026-W24',max_horizon=int(horizons[-1]),
                       fresh_lightgbm_fit=True,seed=config['seed'],max_rounds=config['max_rounds'],
                       threads=config['threads'],group_counts=meta.demand_group.value_counts().to_dict()),folder/'settings.json')
        del pred,p,light; gc.collect()
    metrics = pd.concat(all_metrics,ignore_index=True)
    for _, d in metrics.groupby(['threshold','case','table_id','interval_level']):
        assert d.observations.nunique()==1 and d.skus.nunique()==1
    for _, d in metrics[metrics.sample_policy.eq('Fixed across thresholds')].groupby(['case','table_id','method','interval_level']):
        assert d.observations.nunique()==1 and d.skus.nunique()==1 and d.negative_actuals.nunique()==1
    for _, d in pd.DataFrame(audits).groupby('case'):
        assert d.sku_week_actual_sha256.nunique()==1
    for _,d in metrics.groupby(['threshold','case','table_id','method']):
        d=d.sort_values('interval_level').dropna(subset=['coverage'])
        assert (np.diff(d.coverage)>=-1e-10).all()
        assert (np.diff(d.mean_width)>=-1e-10).all()
        assert (np.diff(d.mean_normalized_outside_distance)<=1e-10).all()
    write_tables(metrics,output)
    engine.write_csv(pd.DataFrame(counts),output/'group_counts.csv')
    engine.write_csv(membership,output/'cohort_membership.csv')
    engine.write_csv(pd.DataFrame(audits),output/'sample_audit.csv')
    plots(metrics,output,config['dpi'])
    assert all(digest(HOME.parent/p)==h for p,h in original_hashes.items()), 'Original files changed.'
    save_json(dict(status='passed',thresholds=thresholds,training_weeks=36,
        complete_training_skus=len(wide),excluded_incomplete_training_skus=int((~complete).sum()),
        fixed_60_nonzero_skus=len(fixed60),fixed_count_eligible_skus=len(count_ids),
        negative_training_values=int((wide[train_dates]<0).sum().sum()),
        negative_training_skus=int((wide[train_dates]<0).any(axis=1).sum()),
        input_sha256=digest(source),original_files_unchanged=True,
        original_files_checked=len(original_hashes),fresh_model_fit_per_threshold=True,
        same_sku_week_actuals_across_thresholds=True,audits=audits,
        elapsed_seconds=round(time.time()-started,2)),output/'validation.json')
    save_json(dict(python=sys.executable,python_version=platform.python_version(),
        numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,lightgbm=lgb.__version__,
        matplotlib=matplotlib.__version__,config=config,thresholds=thresholds,training_runs=training_runs,
        source_hashes={str(p.relative_to(HOME)):digest(p) for p in list((HOME/'engine').glob('*.py'))+[Path(__file__), HOME/'plot_results.py']}),output/'run_manifest.json')
    print(f'\nCOMPLETED: all validation checks passed. Results: {output}',flush=True)


if __name__=='__main__':
    main()
