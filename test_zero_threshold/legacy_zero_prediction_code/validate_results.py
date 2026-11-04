"""Independent saved-result verification, including each persisted LightGBM model."""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import pandas as pd
import run_experiment as runner
from lightgbm_features import origin_features
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=runner.HOME/'results')
    args = parser.parse_args()
    output = args.output.resolve()
    manifest = json.loads((output/'run_manifest.json').read_text(encoding='utf-8'))
    original = json.loads((output/'original_files_sha256.json').read_text(encoding='utf-8'))
    assert all(runner.digest(runner.HOME.parent/p)==h for p,h in original.items())
    raw = pd.read_csv(runner.HOME.parent/'input/weekly_sales.csv',dtype={'sku_id':str}).set_index('sku_id')
    weeks = list(raw.columns)
    train_cols = weeks[weeks.index('2025-W41'):weeks.index('2026-W24')+1]
    assert len(train_cols)==36
    complete = raw[train_cols].notna().all(axis=1)
    train = raw.loc[complete,train_cols]
    zero_weeks = train.eq(0).sum(axis=1)
    fixed60 = set(zero_weeks[zero_weeks<22].index)
    count_mask = (train.ge(0).all(axis=1) & pd.Series(np.isclose(train,np.round(train)).all(axis=1),index=train.index))
    count_ids = set(train.index[count_mask])
    membership = pd.read_csv(output/'cohort_membership.csv',dtype={'sku_id':str}).set_index('sku_id')
    assert set(membership.index)==set(train.index)
    assert set(membership.index[membership.fixed_60_nonzero])==fixed60
    assert set(membership.index[membership.count_training_eligible])==count_ids
    scales = train.apply(lambda z:max(float(np.median(np.abs(z[z!=0]))) if (z!=0).any() else 0.,1.),axis=1)
    metrics = pd.read_csv(output/'metrics_long.csv')
    assert not any('wis' in col.lower() for col in metrics.columns)
    expected_boundaries={.5:18,.6:22,.7:26,.8:29}
    for threshold, boundary in expected_boundaries.items():
        for n in (boundary-1,boundary):
            z=np.r_[np.zeros(n),np.ones(35-n),-2.]
            assert (runner.engine.profile(z,threshold)['demand_group']=='Zero')==(n>=boundary)
    audits=[]; model_checks=0; metric_checks=0
    sample_hashes={}; model_hashes={}
    for threshold in manifest['thresholds']:
        tag=f'zero_{threshold*100:g}'.replace('.','p'); folder=output/tag
        meta=pd.read_csv(folder/'training_profiles.csv',dtype={'sku_id':str}).set_index('sku_id')
        pd.testing.assert_series_equal(meta.zero_weeks,zero_weeks,check_names=False)
        pd.testing.assert_series_equal(meta.has_negative_training,train.lt(0).any(axis=1),check_names=False)
        assert meta.demand_group.eq('Zero').equals(zero_weeks.div(36).ge(threshold))
        # Independently reproduce the four non-Zero classification rules.
        for sku,z in train.iterrows():
            positives=z[z>0]
            adi=36/len(positives) if len(positives) else np.nan
            cv2=float(positives.var(ddof=1)/positives.mean()**2) if len(positives)>1 else 0.
            group=('Smooth' if cv2<.49 else 'Erratic') if adi<1.32 else ('Intermittent' if cv2<.49 else 'Lumpy')
            if zero_weeks[sku]/36>=threshold:group='Zero'
            assert meta.loc[sku,'demand_group']==group
        pred=pd.read_csv(folder/'predictions.csv.gz',dtype={'sku_id':str})
        assert not pred.duplicated(['sku_id','method','iso_week']).any()
        np.testing.assert_allclose(pred.nonzero_median_divisor,pred.sku_id.map(scales))
        assert pred.loc[pred.demand_group.eq('Zero'),runner.engine.QNAMES].eq(0).all().all()
        assert np.isfinite(pred[runner.engine.QNAMES]).all().all()
        assert (np.diff(pred[runner.engine.QNAMES].to_numpy(),axis=1)>=0).all()
        assert pred.demand_group.equals(pred.sku_id.map(meta.demand_group))
        for case in metrics.case.unique():
            end='2026-W42' if case=='main_W42' else '2026-W43'
            cols=weeks[weeks.index('2026-W25'):weeks.index(end)+1]
            expected=raw.loc[train.index,cols].rename_axis(columns='iso_week').stack(future_stack=True).dropna().sort_index()
            p=pred[pred.iso_week.isin(cols) & pred.actual_sales.notna()]
            for method in runner.engine.MODELS:
                d=p[p.method.eq(method)]
                exp=expected if method in runner.PRIMARY else expected[expected.index.get_level_values(0).isin(count_ids)]
                actual=d.set_index(['sku_id','iso_week']).actual_sales.sort_index()
                pd.testing.assert_series_equal(actual,exp,check_names=False)
                sample_hash=hashlib.sha256(pd.util.hash_pandas_object(actual).values.tobytes()).hexdigest()
                key=(case,method)
                if key in sample_hashes:assert sample_hashes[key]==sample_hash
                sample_hashes[key]=sample_hash
            panels={'All_SKUs':set(train.index),'Current_nonzero':set(meta.index[meta.demand_group.ne('Zero')]),
                    'Fixed_60_nonzero':fixed60,'Common_count':count_ids,
                    'Common_count_fixed_60_nonzero':count_ids & fixed60,
                    **{g:set(meta.index[meta.demand_group.eq(g)]) for g in runner.GROUPS}}
            for (panel,method),rows in metrics[metrics.threshold.eq(threshold)&metrics.case.eq(case)].groupby(['table_id','method']):
                d=p[p.sku_id.isin(panels[panel]) & p.method.eq(method)]
                for r in rows.itertuples():
                    lo,hi=runner.engine.INTERVALS[r.interval_level]
                    below=(d[lo]-d.actual_sales).clip(lower=0)
                    above=(d.actual_sales-d[hi]).clip(lower=0)
                    distance=pd.concat([below,above],axis=1).max(axis=1)
                    normalized=distance/d.sku_id.map(scales)
                    np.testing.assert_allclose([r.coverage,r.mean_width,r.mean_normalized_outside_distance],
                        [(distance==0).mean() if len(d) else np.nan,(d[hi]-d[lo]).mean(),normalized.mean()],rtol=1e-11,atol=1e-11,equal_nan=True)
                    assert r.observations==len(d) and r.skus==d.sku_id.nunique()
                    assert r.negative_actuals==int(d.actual_sales.lt(0).sum())
                    assert r.outside_points==int(distance.gt(0).sum())
                    metric_checks+=1
            audits.append(dict(threshold=threshold,case=case,observations=len(expected),negative_actuals=int(expected.lt(0).sum())))
        # Recreate predictions from each saved model to verify the persisted fits.
        maxh=int(pred.horizon_weeks.max())
        horizons=np.arange(1,maxh+1)
        for group in runner.GROUPS[:-1]:
            ids=meta.index[meta.demand_group.eq(group)].tolist()
            if not ids:continue
            values=train.loc[ids].to_numpy(float)
            fx,scale,_=origin_features(values)
            future=np.concatenate([np.column_stack([fx,np.full(len(ids),h,dtype=np.float32)]) for h in horizons]).astype(np.float32)
            forecasts=[]
            for alpha in runner.engine.QUANTILES:
                path=folder/'models'/f'{group}_q{alpha:g}.txt'
                booster=runner.lgb.Booster(model_str=path.read_text(encoding='utf-8'))
                forecasts.append(booster.predict(future,num_threads=2).reshape(maxh,len(ids))*scale[None,:])
                model_checks+=1
            q=np.stack(forecasts,axis=2)
            nonnegative=~meta.loc[ids,'has_negative_training'].to_numpy(bool)
            q[:,nonnegative,:]=np.maximum(q[:,nonnegative,:],0)
            q=np.sort(q,axis=2)
            saved=pred[pred.method.eq('LightGBM') & pred.sku_id.isin(ids)].set_index(['horizon_weeks','sku_id'])
            order=pd.MultiIndex.from_product([horizons,ids],names=['horizon_weeks','sku_id'])
            np.testing.assert_allclose(saved.loc[order,runner.engine.QNAMES].to_numpy(),q.reshape(-1,7),rtol=1e-10,atol=1e-8)
        model_hashes[tag]={p.name:runner.digest(p) for p in (folder/'models').glob('*.txt')}
        print(f'Validated saved predictions and models: {threshold:.0%}',flush=True)
    pngs=list(output.rglob('*.png'))
    for path in pngs:
        with Image.open(path) as im:im.verify()
    changed={}
    if 'zero_60' in model_hashes:
        for tag, hashes in model_hashes.items():
            changed[tag]=sum(h!=model_hashes['zero_60'].get(name) for name,h in hashes.items())
    result=dict(status='passed',metric_rows_independently_recomputed=metric_checks,
        saved_models_loaded_and_predictions_verified=model_checks,png_files_verified=len(pngs),
        model_files_differing_from_60_percent=changed,original_files_unchanged=True,
        threshold_boundary_checks=True,missing_and_signed_actuals_verified=True,audits=audits)
    runner.save_json(result,output/'independent_validation.json')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
