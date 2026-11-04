#!/usr/bin/env python3
"""Independent checks of five-group classification, raw returns and generated results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image
from run_b_experiment import profile,week_date,QNAMES


def main():
    home=Path(__file__).resolve().parent
    root=home/'results'
    groups={'Smooth','Erratic','Intermittent','Lumpy','Zero'}
    # Negative returns do not create a sixth class in any demand pattern.
    assert profile(np.r_[np.ones(35),-1.])['demand_group']=='Smooth'
    assert profile(np.r_[np.tile([1.,20.],17),1.,-1.])['demand_group']=='Erratic'
    assert profile(np.r_[np.ones(18),np.zeros(17),-1.])['demand_group']=='Intermittent'
    assert profile(np.r_[np.tile([1.,20.],9),np.zeros(17),-1.])['demand_group']=='Lumpy'
    assert profile(np.r_[np.zeros(22),np.ones(13),-1.])['demand_group']=='Zero'
    assert profile(np.r_[np.zeros(21),np.ones(14),-1.])['demand_group']!='Zero'
    meta=pd.read_csv(root/'training_profiles.csv',dtype={'sku_id':str}).set_index('sku_id')
    assert set(meta.demand_group)==groups and len(meta)==6464
    expected={'Zero':5258,'Lumpy':604,'Erratic':484,'Smooth':95,'Intermittent':23}
    assert meta.demand_group.value_counts().to_dict()==expected
    raw=pd.read_csv(home/'input/weekly_sales.csv',dtype={'sku_id':str}).set_index('sku_id')
    train=raw.loc[meta.index].iloc[:,:36]
    assert train.notna().all().all()
    pd.testing.assert_series_equal(train.lt(0).any(axis=1),meta.has_negative_training,check_names=False)
    assert int(meta.has_negative_training.sum())==182
    med=train.apply(lambda z:float(np.median(np.abs(z[z.ne(0)]))) if z.ne(0).any() else 0.,axis=1)
    np.testing.assert_allclose(med,meta.nonzero_abs_median)
    config=json.loads((root/'settings.json').read_text())
    assert set(config['demand_groups'])==groups and config['zero_threshold']==.6
    assert len(list((root/'models').glob('*.txt')))==28
    for entry in config['training_log']:
        assert entry['group'] in groups
        if entry['group']=='Zero':continue
        assert week_date(entry['last_internal_fit_target'])<week_date(entry['first_validation_target'])
        assert entry['final_fit_end']=='2026-W24'
    reassigned=pd.read_csv(root/'group_reassignment.csv')
    assert len(reassigned)==119 and set(reassigned.demand_group)<=groups
    assert reassigned.demand_group.value_counts().to_dict()=={'Lumpy':55,'Erratic':50,'Smooth':13,'Intermittent':1}
    metrics=pd.read_csv(root/'all_metrics.csv')
    for _,d in metrics.groupby(['case','table_id','interval_level']):
        assert d.skus.nunique()==d.observations.nunique()==1
    for _,d in metrics.groupby(['case','table_id','method']):
        d=d.sort_values('interval_level')
        assert (np.diff(d.coverage)>=-1e-10).all()
        assert (np.diff(d.mean_width)>=-1e-10).all()
        assert (np.diff(d.mean_normalized_nonzero_median)<=1e-10).all()
    cases=[]
    for case,end in [('main_W42',54),('sensitivity_W43',55)]:
        pred=pd.read_csv(root/case/'predictions.csv.gz',dtype={'sku_id':str})
        assert set(pred.demand_group)==groups
        assert not pred.duplicated(['sku_id','method','iso_week']).any()
        assert np.isfinite(pred[QNAMES].to_numpy()).all()
        assert (np.diff(pred[QNAMES].to_numpy(),axis=1)>=-1e-9).all()
        assert pred.loc[pred.zero_rule,QNAMES].eq(0).all().all()
        assert pred.demand_group.equals(pred.sku_id.map(meta.demand_group))
        actual=raw.loc[meta.index].iloc[:,36:end].stack().rename('actual_sales')
        actual.index.names=['sku_id','iso_week']
        for method in ['Empirical','Recent13','DampedTrendResidual','LightGBM']:
            d=pred[pred.method.eq(method)].dropna(subset=['actual_sales'])
            series=d.set_index(['sku_id','iso_week']).actual_sales.sort_index()
            pd.testing.assert_series_equal(series,actual.sort_index(),check_names=False)
            assert d.sku_id.nunique()==6464
            for level,lo,hi in [(70,'q15','q85'),(80,'q10','q90'),(90,'q05','q95')]:
                coverage=d.actual_sales.between(d[lo],d[hi]).mean()
                row=metrics[metrics['case'].eq(case)&metrics.table_id.eq('All_SKUs')&metrics.method.eq(method)&metrics.interval_level.eq(level)].iloc[0]
                assert np.isclose(coverage,row.coverage)
                assert row.observations==len(actual)
        out=pd.read_csv(root/case/'all_outside_points.csv.gz',dtype={'sku_id':str})
        assert set(out.demand_group)<=groups and out.outside_distance.gt(0).all()
        np.testing.assert_allclose(out.normalized_nonzero_median,out.outside_distance/out.nonzero_median_divisor)
        np.testing.assert_allclose(out.outside_distance,np.abs(out.signed_outside_distance))
        cases.append(dict(case=case,skus=6464,observations_per_full_method=len(actual),
                          retained_negative_actuals_per_full_method=int(actual.lt(0).sum()),outside_rows_all_intervals=len(out)))
    heat=json.loads((home/'figures/plot_audit.json').read_text())
    scatter=json.loads((home/'figures_signed_scatter/scatter_audit.json').read_text())
    assert len(heat)==len(scatter)==18
    expected_counts={(a['case'],a['interval_level'],a['normalization']):a['source_outside_rows'] for a in heat}
    assert all(a['source_outside_rows']==a['scatter_points']==a['heatmap_count'] for a in heat)
    for a in scatter:
        assert a['source_rows']==a['plotted_rows']==expected_counts[(a['case'],a['interval_level'],a['normalization'])]
        assert a['y_axis_lower']<a['actual_minimum_signed_distance'] and a['y_axis_upper']>a['actual_maximum_signed_distance']
    # Histograms must count exactly the saved outside points, including both
    # signed tails and every extreme. Check each exported bin independently.
    from plot_outlier_histogram import bin_edges,transform,inverse_transform
    edge_case=np.array([-40.,-.001,.001,1.,3400.])
    edges=bin_edges(edge_case,72)
    assert 0. in edges
    assert np.histogram(transform(edge_case),edges)[0].sum()==len(edge_case)
    np.testing.assert_allclose(inverse_transform(transform(edge_case)),edge_case)
    histogram_audits=list((home/'figures_histogram').rglob('*.audit.json'))
    assert len(histogram_audits)==18,len(histogram_audits)
    for case in ['main_W42','sensitivity_W43']:
        source=pd.read_csv(root/case/'all_outside_points.csv.gz',dtype={'sku_id':str})
        for audit_path in (home/'figures_histogram'/case).glob('*.audit.json'):
            audit=json.loads(audit_path.read_text())
            level=audit['interval_level'];norm=audit['normalization']
            d=source[source.interval_level.eq(level)]
            stem=audit_path.name.removesuffix('.audit.json')
            counts=pd.read_csv(audit_path.parent/(stem+'_counts.csv'))
            summary=pd.read_csv(audit_path.parent/(stem+'_summary.csv'))
            assert audit['source_rows']==audit['histogram_count']==len(d)
            assert audit['no_downsampling'] and audit['no_tail_clipping'] and audit['signed_values_unchanged']
            assert counts['count'].sum()==len(d) and set(counts.demand_group)==groups
            for method,m in d.groupby('method'):
                c=counts[counts.method.eq(method)]
                s=summary[summary.method.eq(method)].iloc[0]
                assert c['count'].sum()==len(m)==s.outside_points
                key='signed_normalized_'+norm
                assert c.loc[c.transformed_right.le(0),'count'].sum()==m[key].lt(0).sum()==s.below_lower_bound
                assert c.loc[c.transformed_left.ge(0),'count'].sum()==m[key].gt(0).sum()==s.above_upper_bound
                unique=c.drop_duplicates('bin_index').sort_values('bin_index')
                edge_array=np.r_[unique.transformed_left.to_numpy(),unique.transformed_right.iloc[-1]]
                for group,g in m.groupby('demand_group'):
                    expected_hist=np.histogram(transform(g[key].to_numpy()),edge_array)[0]
                    observed=c[c.demand_group.eq(group)].sort_values('bin_index')['count'].to_numpy()
                    np.testing.assert_array_equal(expected_hist,observed)
    for folder in ['figures','figures_signed_scatter','revised_charts','figures_histogram']:
        for path in (home/folder).rglob('*.tmp.png'):path.unlink()
    images=[p for folder in ['figures','figures_signed_scatter','revised_charts','figures_histogram'] for p in (home/folder).rglob('*.png')]
    assert len(images)==98,len(images)
    for path in images:
        with Image.open(path) as im:im.verify()
    forbidden='Signed'+'Adjustments'
    for path in home.rglob('*.csv'):
        assert forbidden not in path.read_text(encoding='utf-8-sig')
    report=dict(status='passed',group_counts=expected,reassigned_skus=119,negative_training_skus=182,
                raw_returns_unchanged=True,png_files_verified=len(images),histograms_verified=len(histogram_audits),cases=cases)
    (home/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
