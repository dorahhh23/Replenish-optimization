"""Refit Experiment B on each threshold's non-Zero SKUs; export Zero lists only."""
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
from plot_outliers import plot_saved_outliers

GROUPS = ['Smooth', 'Erratic', 'Intermittent', 'Lumpy']
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


def cohorts(meta, count_ids):
    result = [
        ('Current_nonzero', set(meta.sku_id), PRIMARY),
        ('Count_eligible_nonzero', count_ids, engine.MODELS),
    ]
    result += [(g, set(meta.loc[meta.demand_group.eq(g), 'sku_id']), PRIMARY) for g in GROUPS]
    return result


def summarize(pred, meta, count_ids, threshold, case):
    rows = []
    scored = pred.loc[pred.actual_sales.notna()]
    for panel, ids, methods in cohorts(meta, count_ids):
        for method in methods:
            d = scored.loc[scored.method.eq(method) & scored.sku_id.isin(ids)]
            y = d.actual_sales.to_numpy(float)
            for level, (lo, hi) in engine.INTERVALS.items():
                lower, upper = d[lo].to_numpy(float), d[hi].to_numpy(float)
                width = upper - lower
                # Normalize each SKU-week first, then average the same scored points.
                width_median = width / d.nonzero_median_divisor.to_numpy(float)
                width_sd = width / d.sd_divisor.to_numpy(float)
                outside = np.maximum.reduce([lower-y, y-upper, np.zeros(len(y))])
                norm = outside / d.nonzero_median_divisor.to_numpy(float)
                rows.append(dict(threshold=threshold, case=case, table_id=panel,
                    sample_policy='Current threshold non-Zero sample; varies with threshold',
                    method=method, interval_level=level, cohort_skus=len(ids),
                    skus=d.sku_id.nunique(), observations=len(d), negative_actuals=int((y<0).sum()),
                    coverage=float(np.mean(outside == 0)) if len(d) else np.nan,
                    mean_width=float(np.mean(width)) if len(d) else np.nan,
                    mean_normalized_width_nonzero_median=float(np.mean(width_median)) if len(d) else np.nan,
                    mean_normalized_width_training_sd=float(np.mean(width_sd)) if len(d) else np.nan,
                    median_scale_floor_skus=d.loc[d.nonzero_median_floor_applied,'sku_id'].nunique(),
                    sd_scale_floor_skus=d.loc[d.sd_floor_applied,'sku_id'].nunique(),
                    mean_normalized_outside_distance=float(np.mean(norm)) if len(d) else np.nan,
                    outside_points=int((outside>0).sum())))
    return pd.DataFrame(rows)


def verify_predictions(pred, raw, train_dates, dates, meta, count_ids):
    """Check original signed actuals and ensure Zero never enters predictions."""
    assert not pred.duplicated(['sku_id', 'iso_week', 'method']).any()
    assert np.isfinite(pred[engine.QNAMES].to_numpy()).all()
    assert (np.diff(pred[engine.QNAMES].to_numpy(), axis=1) >= 0).all()
    assert not pred.zero_rule.any() and not pred.demand_group.eq('Zero').any()
    assert set(pred.sku_id) == set(meta.sku_id)
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
    scales, sd_scales = {}, {}
    for sku in meta.sku_id:
        z = raw.loc[sku, train_dates].to_numpy(float)
        scales[sku] = max(float(np.median(np.abs(z[z != 0]))) if np.any(z != 0) else 0., 1.)
        sd_scales[sku] = max(float(np.std(z,ddof=1)),1.)
    np.testing.assert_allclose(pred.nonzero_median_divisor, pred.sku_id.map(scales), rtol=0, atol=0)
    np.testing.assert_allclose(pred.sd_divisor, pred.sku_id.map(sd_scales), rtol=1e-12, atol=1e-12)
    valid = expected.dropna()
    fingerprint = hashlib.sha256(pd.util.hash_pandas_object(valid).values.tobytes()).hexdigest()
    return dict(case_observations=len(valid), sku_week_actual_sha256=fingerprint,
                negative_actuals_per_primary_method=int((valid < 0).sum()),
                missing_actuals_not_scored=int(expected.isna().sum()),
                maximum_actual=float(valid.max()), minimum_actual=float(valid.min()),
                zero_excluded_from_predictions=True, signed_actuals_match_input=True,
                count_methods_share_applicable_sample_within_threshold=True, normalization_verified=True)


def write_tables(metrics, output):
    engine.write_csv(metrics, output / 'metrics_long.csv')
    rows = []
    keys = ['threshold', 'case', 'table_id', 'sample_policy', 'method']
    for key, part in metrics.groupby(keys, sort=False):
        first = part.iloc[0]
        row = dict(zip(['Zero Threshold', 'Case', 'Table', 'Sample Policy', 'Model'], key))
        row.update({'Cohort SKUs': int(first.cohort_skus), 'Scored SKUs': int(first.skus),
                    'Observed Points': int(first.observations), 'Negative Actuals': int(first.negative_actuals)})
        row['Scored SKUs with Median Scale Below 1'] = int(first.median_scale_floor_skus)
        row['Scored SKUs with SD Scale Below 1'] = int(first.sd_scale_floor_skus)
        for r in part.itertuples():
            row[f'{r.interval_level}% Coverage'] = r.coverage
            row[f'{r.interval_level}% Mean Median-Normalized Width'] = r.mean_normalized_width_nonzero_median
            row[f'{r.interval_level}% Mean SD-Normalized Width'] = r.mean_normalized_width_training_sd
            row[f'{r.interval_level}% Mean Normalized Outside Distance'] = r.mean_normalized_outside_distance
        rows.append(row)
    combined = pd.DataFrame(rows)
    engine.write_csv(combined, output / 'comparison_all_tables.csv')
    engine.write_csv(combined[combined.Table.eq('Current_nonzero')], output / 'overall_comparison.csv')
    for (case, panel), part in combined.groupby(['Case', 'Table']):
        engine.write_csv(part, output / case / 'tables' / f'{panel}.csv')
        keep = ['Zero Threshold', 'Case', 'Table', 'Sample Policy', 'Model', 'Cohort SKUs', 'Scored SKUs', 'Observed Points', 'Negative Actuals',
                'Scored SKUs with Median Scale Below 1', 'Scored SKUs with SD Scale Below 1']
        engine.write_csv(part[keep + [c for c in part if 'Coverage' in c or 'Normalized Width' in c]],
                         output / case / 'coverage_width_tables' / f'{panel}.csv')
        engine.write_csv(part[keep + [c for c in part if 'Normalized Outside' in c]],
                         output / case / 'deviation_tables' / f'{panel}.csv')
    # Raw width is reference information only, not the primary comparison metric.
    absolute = metrics[['threshold','case','table_id','sample_policy','method','interval_level','skus','observations','mean_width']].copy()
    absolute = absolute.rename(columns={'mean_width':'mean_absolute_width_sales_units'})
    engine.write_csv(absolute,output/'absolute_width_reference.csv')



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
    if output == (HOME/'results').resolve() or output.is_relative_to(HOME/'results'):
        parser.error('results/ contains the previous Zero-prediction experiment. Use results_zero_excluded or a new folder.')
    output.mkdir(parents=True,exist_ok=True)
    if (output/'metrics_long.csv').is_file():
        existing = pd.read_csv(output/'metrics_long.csv',usecols=['table_id'])
        if set(existing.table_id) - {'Current_nonzero','Count_eligible_nonzero',*GROUPS}:
            parser.error('Output contains old Zero-prediction metrics. Choose a new output folder.')
    if args.plots_only:
        plots(pd.read_csv(output/'metrics_long.csv'),output,config['dpi'])
        plot_saved_outliers(output, dpi=config['dpi'])
        return
    started = time.time()
    print(f'Python: {sys.executable}\nInput: {source}\nOutput: {output}', flush=True)
    # Hash original project results and code before/after to verify preservation.
    protected = [p for folder in ['results','revised_charts','figures','figures_histogram','figures_signed_scatter']
                 for p in (HOME.parent/folder).rglob('*') if p.is_file()]
    protected += list(HOME.parent.glob('*.py')) + [source]
    protected += [p for p in (HOME/'results').rglob('*') if p.is_file()]
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
    membership = pd.DataFrame({'sku_id':wide.index})
    cases = [('main_W42', main_dates)]
    if config['include_week43_sensitivity']:
        assert engine.week_date('2026-W43') in raw.columns
        cases.append(('sensitivity_W43', pd.date_range(main_dates[0], engine.week_date('2026-W43'), freq='7D')))
    target_dates = cases[-1][1]
    horizons = np.arange(1,len(target_dates)+1)
    all_metrics, counts, audits, training_runs = [], [], [], []
    zero_lists, threshold_counts = [], []
    for threshold in thresholds:
        tag = f'zero_{threshold*100:g}'.replace('.', 'p')
        folder = output/tag; folder.mkdir(exist_ok=True)
        print(f'\n=== Zero threshold {threshold:.0%}: fresh model fitting ===',flush=True)
        meta = profiles_for(wide, train_dates, threshold)
        minimum = math.ceil(36*threshold-1e-12)
        assert meta.demand_group.eq('Zero').equals(meta.zero_weeks.ge(minimum))
        assert set(meta.demand_group)<=set(GROUPS + ['Zero'])
        assert meta.sku_id.tolist()==wide.index.tolist()
        engine.write_csv(meta,folder/'training_profiles.csv')
        membership[tag] = membership.sku_id.map(meta.set_index('sku_id').demand_group)
        zero_list = meta.loc[meta.demand_group.eq('Zero'),
            ['sku_id','zero_weeks','train_weeks','zero_share','has_negative_training']].copy()
        zero_list.insert(0,'threshold',threshold)
        zero_list['minimum_zero_weeks'] = minimum
        zero_list['reason'] = 'Training zero share at or above threshold; excluded from all modeling and evaluation'
        engine.write_csv(zero_list, folder/'zerolist.csv')
        zero_lists.append(zero_list)
        meta = meta.loc[meta.demand_group.ne('Zero')].copy()
        if meta.empty:
            raise ValueError(f'Threshold {threshold:g} leaves no non-Zero SKUs. Zero list was saved; no models or metrics were produced.')
        count_ids = set(meta.loc[meta.count_training_eligible,'sku_id'])
        active_wide = wide.loc[meta.sku_id]
        engine.write_csv(meta, folder/'model_training_profiles.csv')
        threshold_counts.append(dict(threshold=threshold,minimum_zero_weeks=minimum,
            complete_training_skus=len(wide),zero_list_skus=len(zero_list),modeled_skus=len(meta),
            count_eligible_nonzero_skus=len(count_ids),
            negative_training_values_retained=int((active_wide[train_dates]<0).sum().sum()),
            negative_training_skus_retained=int(meta.has_negative_training.sum())))
        for group in GROUPS:
            sub = meta[meta.demand_group.eq(group)]
            counts.append(dict(threshold=threshold,minimum_zero_weeks=minimum,demand_group=group,
                               skus=len(sub),skus_with_negative_training=int(sub.has_negative_training.sum())))
        print(f'Excluded Zero SKUs: {len(zero_list)}; modeling non-Zero SKUs: {len(meta)}',flush=True)
        light, logs = engine.fit_lightgbm(active_wide, meta, train_dates, horizons, folder/'models',
                                        config['max_rounds'],config['threads'],config['seed'])
        for log in logs:
            assert log['group'] in GROUPS
            assert log['final_fit_end']=='2026-W24'
            assert engine.week_date(log['last_internal_fit_target']) < engine.week_date(log['first_validation_target'])
        save_json(logs, folder/'training_log.json')
        training_runs.append(dict(threshold=threshold,training_log=logs))
        assert set(light) == set(meta.sku_id)
        print('Generating non-Zero predictions and metrics',flush=True)
        pred = engine.generate_predictions(active_wide,meta,train_dates,target_dates,light,horizons)
        assert set(pred.sku_id).isdisjoint(set(zero_list.sku_id))
        pred = pred.loc[~pred.method.isin(engine.COUNT) | pred.sku_id.isin(count_ids)].copy()
        for case, dates in cases:
            p = pred.loc[pred.week_start.isin(dates)]
            audit = verify_predictions(p,raw,train_dates,dates,meta,count_ids)
            audit.update(threshold=threshold,case=case,modeled_skus=len(meta),excluded_zero_skus=len(zero_list))
            audits.append(audit)
            metrics = summarize(p,meta,count_ids,threshold,case)
            all_metrics.append(metrics)
            case_folder = folder/case; case_folder.mkdir(exist_ok=True)
            engine.write_csv(metrics,case_folder/'metrics_long.csv')
            points = engine.interval_points(p)
            outside = points.loc[~points.covered].copy()
            assert set(outside.demand_group) <= set(GROUPS)
            assert set(outside.sku_id).isdisjoint(set(zero_list.sku_id))
            outside.to_csv(case_folder/'all_outside_points.csv.gz',index=False,encoding='utf-8-sig',compression='gzip')
            del points, outside
        pred.to_csv(folder/'predictions.csv.gz',index=False,encoding='utf-8-sig',compression='gzip')
        save_json(dict(threshold=threshold,minimum_zero_weeks=minimum,training_weeks=36,
                       training_start='2025-W41',training_end='2026-W24',max_horizon=int(horizons[-1]),
                       fresh_lightgbm_fit=True,seed=config['seed'],max_rounds=config['max_rounds'],
                       threads=config['threads'],zero_policy='Excluded from fitting, prediction, and evaluation',
                       zero_list_skus=len(zero_list),modeled_skus=len(meta),
                       group_counts=meta.demand_group.value_counts().to_dict()),folder/'settings.json')
        del pred,p,light; gc.collect()
    metrics = pd.concat(all_metrics,ignore_index=True)
    for _, d in metrics.groupby(['threshold','case','table_id','interval_level']):
        assert d.observations.nunique()==1 and d.skus.nunique()==1
    assert set(metrics.table_id) <= {'Current_nonzero','Count_eligible_nonzero',*GROUPS}
    for _,d in metrics.groupby(['threshold','case','table_id','method']):
        d=d.sort_values('interval_level').dropna(subset=['coverage'])
        assert (np.diff(d.coverage)>=-1e-10).all()
        assert (np.diff(d.mean_width)>=-1e-10).all()
        assert (np.diff(d.mean_normalized_width_nonzero_median)>=-1e-10).all()
        assert (np.diff(d.mean_normalized_width_training_sd)>=-1e-10).all()
        assert (np.diff(d.mean_normalized_outside_distance)<=1e-10).all()
    write_tables(metrics,output)
    engine.write_csv(pd.DataFrame(counts),output/'group_counts.csv')
    engine.write_csv(pd.DataFrame(threshold_counts),output/'threshold_summary.csv')
    engine.write_csv(pd.concat(zero_lists,ignore_index=True),output/'zerolist_all_thresholds.csv')
    engine.write_csv(membership,output/'cohort_membership.csv')
    engine.write_csv(pd.DataFrame(audits),output/'sample_audit.csv')
    plots(metrics,output,config['dpi'])
    plot_saved_outliers(output,thresholds=thresholds,dpi=config['dpi'])
    assert all(digest(HOME.parent/p)==h for p,h in original_hashes.items()), 'Original files changed.'
    save_json(dict(status='passed',thresholds=thresholds,training_weeks=36,
        complete_training_skus=len(wide),excluded_incomplete_training_skus=int((~complete).sum()),
        threshold_samples=threshold_counts,
        negative_training_values_in_complete_input=int((wide[train_dates]<0).sum().sum()),
        input_sha256=digest(source),original_files_unchanged=True,
        original_files_checked=len(original_hashes),fresh_model_fit_per_threshold=True,
        zero_excluded_from_all_modeling_and_evaluation=True,
        cross_threshold_sample_policy='Each threshold uses its own non-Zero SKUs',audits=audits,
        elapsed_seconds=round(time.time()-started,2)),output/'validation.json')
    save_json(dict(python=sys.executable,python_version=platform.python_version(),
        numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,lightgbm=lgb.__version__,
        matplotlib=matplotlib.__version__,config=config,input_path=str(source),thresholds=thresholds,training_runs=training_runs,
        zero_policy='Exclude Zero SKUs from fitting, prediction, and evaluation',
        width_metrics={
            'primary':'mean((upper - lower) / max(median(abs(nonzero training sales)), 1))',
            'secondary':'mean((upper - lower) / max(sample SD of all 36 signed training sales, 1))',
            'aggregation':'Normalize each scored SKU-week before averaging; same observations as coverage and outside distance',
            'absolute_width':'Reference table only'},
        source_hashes={str(p.relative_to(HOME)):digest(p) for p in list((HOME/'engine').glob('*.py'))+[Path(__file__), HOME/'plot_results.py', HOME/'plot_outliers.py']}),output/'run_manifest.json')
    print(f'\nCOMPLETED: all validation checks passed. Results: {output}',flush=True)


if __name__=='__main__':
    main()
