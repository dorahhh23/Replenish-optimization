#!/usr/bin/env python3
"""Experiment B model helpers; callers must exclude Zero SKUs before modeling.

All quantity data, output columns, labels, model names and generated tables use
English. Original sales and extreme orders are retained. Missing is not zero.
"""
from pathlib import Path
from datetime import date
import argparse,gc,hashlib,json,re,sys
import numpy as np
import pandas as pd
from scipy.stats import poisson,nbinom
import lightgbm as lgb
from lightgbm_features import origin_features,examples,validation_block

QUANTILES=np.array([.05,.10,.15,.50,.85,.90,.95])
QNAMES=['q05','q10','q15','q50','q85','q90','q95']
INTERVALS={70:('q15','q85'),80:('q10','q90'),90:('q05','q95')}
BASE=['Empirical','Recent13','DampedTrendResidual']
COUNT=['Poisson','NegativeBinomial']
MODELS=BASE+['LightGBM']+COUNT
NORMALIZATIONS=['training_sd','nonzero_median','interval_width']


def week_date(text):
    m=re.fullmatch(r'(20\d\d)-W(\d{1,2})',str(text))
    if m is None:raise ValueError(f'Invalid ISO week: {text}')
    return pd.Timestamp(date.fromisocalendar(int(m[1]),int(m[2]),1))


def week_label(dt):
    y,w,_=pd.Timestamp(dt).isocalendar();return f'{y}-W{w:02d}'


def write_csv(frame,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(path,index=False,encoding='utf-8-sig')


def profile(train,threshold=.6):
    positive=train[train>0];nonzero=np.abs(train[train!=0]);share=float(np.mean(train==0))
    negative=bool(np.any(train<0));is_count=bool(not negative and np.allclose(train,np.round(train)))
    adi=len(train)/len(positive) if len(positive) else np.nan
    cv2=float(np.var(positive,ddof=1)/np.mean(positive)**2) if len(positive)>1 else np.nan
    if share>=threshold:group='Zero'
    else:
        c=0. if not np.isfinite(cv2) else cv2
        group=('Smooth' if c<.49 else 'Erratic') if adi<1.32 else ('Intermittent' if c<.49 else 'Lumpy')
    sd=float(np.std(train,ddof=1));nzmedian=float(np.median(nonzero)) if len(nonzero) else 0.
    return dict(demand_group=group,zero_share=share,zero_weeks=int((train==0).sum()),
       train_weeks=len(train),has_negative_training=negative,count_training_eligible=is_count,
       six_method_eligible=bool(group!='Zero' and is_count),training_sd=sd,
       sd_divisor=max(sd,1.),sd_floor_applied=bool(sd<1.),
       nonzero_abs_median=nzmedian,nonzero_median_divisor=max(nzmedian,1.),
       nonzero_median_floor_applied=bool(nzmedian<1.),adi=adi,positive_cv2=cv2)


def baseline_quantiles(train,horizons,method):
    h=len(horizons)
    if method in BASE:
        if method=='Empirical':values=train;shift=np.zeros(h)
        elif method=='Recent13':values=train[-13:];shift=np.zeros(h)
        else:
            z=train[-26:];x=np.arange(len(z),dtype=float);xc=x-x.mean()
            slope=float(xc@(z-z.mean())/(xc@xc)) if len(z)>1 else 0.
            fitted=z.mean()+slope*xc;values=z-fitted
            shift=fitted[-1]+slope*(.9*(1-.9**horizons)/(1-.9))
        return np.quantile(values,QUANTILES,method='inverted_cdf')[None,:]+shift[:,None]
    mu=float(train.mean());var=float(train.var(ddof=1))
    if mu==0:return np.zeros((h,len(QUANTILES)))
    if method=='NegativeBinomial' and var>mu+1e-8:
        dist=nbinom(mu*mu/(var-mu),mu/var)
    else:dist=poisson(mu)
    return np.tile(dist.ppf(QUANTILES),(h,1))


def fit_lightgbm(wide,features,train_dates,horizons,output,max_rounds=200,threads=2,seed=20261030):
    if features.demand_group.eq('Zero').any():
        raise ValueError('Zero SKUs must be excluded before model fitting.')
    forecasts={};logs=[];importance=[];t=len(train_dates)
    validation_weeks=min(8,max(4,t//6));origin=t-validation_weeks
    output.mkdir(parents=True,exist_ok=True)
    for group,meta in features.groupby('demand_group',sort=True):
        ids=meta.sku_id.tolist();values=wide.loc[ids,train_dates].to_numpy(float)
        print(f'LightGBM {group}: {len(ids)} SKUs; seven quantiles',flush=True)
        X,y,target,origins,names=examples(values,int(max(horizons)))
        mask=target<origin
        assert target.max()<t and np.all(origins[mask]<=target[mask])
        xv,yv=validation_block(values,origin,int(max(horizons)))
        dtrain=lgb.Dataset(X[mask],label=y[mask],feature_name=names,params={'max_bin':63})
        dvalid=lgb.Dataset(xv,label=yv,reference=dtrain,feature_name=names)
        dall=lgb.Dataset(X,label=y,feature_name=names,params={'max_bin':63})
        fx,scale,_=origin_features(values)
        future=np.concatenate([np.column_stack([fx,np.full(len(ids),h,dtype=np.float32)]) for h in horizons]).astype(np.float32)
        pred=np.empty((len(horizons),len(ids),len(QUANTILES)));iterations=[]
        for j,alpha in enumerate(QUANTILES):
            params=dict(objective='quantile',alpha=float(alpha),metric='quantile',learning_rate=.05,
              num_leaves=15,min_data_in_leaf=100,lambda_l2=1.,max_bin=63,verbosity=-1,
              num_threads=threads,seed=seed,deterministic=True,force_col_wise=True)
            initial=lgb.train(params,dtrain,num_boost_round=max_rounds,valid_sets=[dvalid],
                             callbacks=[lgb.early_stopping(25,verbose=False)])
            rounds=max(initial.best_iteration,1)
            final=lgb.train(params,dall,num_boost_round=rounds)
            # Python file I/O supports Chinese and spaces in Windows paths.
            (output/f'{group}_q{alpha:g}.txt').write_text(final.model_to_string(),encoding='utf-8')
            pred[:,:,j]=final.predict(future).reshape(len(horizons),len(ids))*scale[None,:]
            iterations.append(rounds)
            for name,gain in zip(names,final.feature_importance(importance_type='gain')):
                importance.append(dict(group=group,quantile=float(alpha),feature=name,gain=float(gain)))
            del initial,final
        crossing=int((np.diff(pred,axis=2)<0).any(axis=2).sum())
        nonnegative=~meta.has_negative_training.to_numpy(bool)
        pred[:,nonnegative,:]=np.maximum(pred[:,nonnegative,:],0.)
        pred=np.sort(pred,axis=2)
        for i,sku in enumerate(ids):forecasts[sku]=pred[:,i,:]
        logs.append(dict(group=group,skus=len(ids),implementation='Grouped LightGBM quantile regression',
             supervised_rows=len(y),internal_fit_rows=int(mask.sum()),validation_rows=len(yv),
             last_internal_fit_target=week_label(train_dates[target[mask].max()]),
             first_validation_target=week_label(train_dates[origin]),
             final_fit_end=week_label(train_dates.max()),selected_rounds=iterations,
             quantile_crossings_repaired=crossing))
        del X,y,target,origins,xv,yv,dtrain,dvalid,dall,future,pred
        gc.collect()
    write_csv(pd.DataFrame(importance),output/'feature_importance.csv')
    return forecasts,logs


def generate_predictions(wide,features,train_dates,target_dates,light,horizons):
    if features.demand_group.eq('Zero').any():
        raise ValueError('Zero SKUs must be excluded before prediction.')
    blocks=[];lookup={int(h):i for i,h in enumerate(horizons)}
    h=((target_dates-train_dates.max()).days//7).to_numpy()
    for meta in features.to_dict('records'):
        sku=meta['sku_id'];z=wide.loc[sku,train_dates].to_numpy(float)
        y=wide.reindex(index=[sku],columns=target_dates).iloc[0].to_numpy(float)
        methods=MODELS if meta['six_method_eligible'] else BASE+['LightGBM']
        for method in methods:
            if method=='LightGBM':q=light[sku][[lookup[int(v)] for v in h]]
            else:q=baseline_quantiles(z,h,method)
            data={key:q[:,i] for i,key in enumerate(QNAMES)}
            data.update(sku_id=sku,method=method,demand_group=meta['demand_group'],
                 week_start=target_dates,iso_week=[week_label(d) for d in target_dates],
                 horizon_weeks=h,actual_sales=y,zero_rule=False,
                 six_method_eligible=meta['six_method_eligible'],
                 training_sd=meta['training_sd'],sd_divisor=meta['sd_divisor'],
                 sd_floor_applied=meta['sd_floor_applied'],
                 nonzero_abs_median=meta['nonzero_abs_median'],
                 nonzero_median_divisor=meta['nonzero_median_divisor'],
                 nonzero_median_floor_applied=meta['nonzero_median_floor_applied'])
            blocks.append(pd.DataFrame(data))
    p=pd.concat(blocks,ignore_index=True)
    assert np.isfinite(p[QNAMES].to_numpy()).all()
    assert (np.diff(p[QNAMES].to_numpy(),axis=1)>=-1e-9).all()
    return p


def interval_points(predictions):
    if predictions.demand_group.eq('Zero').any():
        raise ValueError('Zero SKUs must be excluded from interval evaluation.')
    p=predictions.loc[predictions.actual_sales.notna()].copy();parts=[]
    keep=['sku_id','method','demand_group','week_start','iso_week','horizon_weeks','actual_sales',
          'zero_rule','six_method_eligible','training_sd','sd_divisor','sd_floor_applied',
          'nonzero_abs_median','nonzero_median_divisor','nonzero_median_floor_applied']
    for level,(lo,hi) in INTERVALS.items():
        d=p[keep].copy();y=p.actual_sales.to_numpy();lower=p[lo].to_numpy();upper=p[hi].to_numpy()
        signed=np.where(y>upper,y-upper,np.where(y<lower,y-lower,0.))
        d['interval_level']=level;d['lower_bound']=lower;d['upper_bound']=upper
        d['interval_width']=upper-lower;d['covered']=signed==0
        d['outside_distance']=np.abs(signed);d['signed_outside_distance']=signed
        d['tail']=np.where(signed>0,'Above',np.where(signed<0,'Below','Inside'))
        d['width_divisor']=np.maximum(d.interval_width,1.)
        d['width_floor_applied']=d.interval_width<1.
        for label,divisor in [('training_sd',d.sd_divisor),('nonzero_median',d.nonzero_median_divisor),('interval_width',d.width_divisor)]:
            d[f'normalized_{label}']=d.outside_distance/divisor
            d[f'signed_normalized_{label}']=d.signed_outside_distance/divisor
        parts.append(d)
    return pd.concat(parts,ignore_index=True)


def summaries(points):
    parts=[]
    common=points[points.six_method_eligible]
    full=points[points.method.isin(BASE+['LightGBM'])]
    panels=[('All_SKUs','All SKUs; four methods supporting signed sales',full),
            ('Nonzero_groups','All non-Zero SKUs; four methods supporting signed sales',full[full.demand_group.ne('Zero')]),
            ('CountEligible_all','Supplemental common six-method cohort',common),
            ('CountEligible_nonzero','Supplemental common six-method cohort, excluding Zero',common[common.demand_group.ne('Zero')])]
    panels += [(group,'All SKUs in demand group: '+group,d) for group,d in full.groupby('demand_group')]
    panels += [(group+'_CountEligible','Supplemental common six-method demand group: '+group,d) for group,d in common.groupby('demand_group')]
    for table_id,cohort,d in panels:
        for (method,level),g in d.groupby(['method','interval_level']):
            row=dict(table_id=table_id,cohort=cohort,method=method,interval_level=int(level),
                     skus=g.sku_id.nunique(),observations=len(g),coverage=g.covered.mean(),
                     mean_width=g.interval_width.mean(),outside_points=int((~g.covered).sum()),
                     mean_outside_distance=g.outside_distance.mean(),
                     mean_outside_distance_conditional=g.loc[~g.covered,'outside_distance'].mean())
            for norm in NORMALIZATIONS:
                row[f'mean_normalized_{norm}']=g[f'normalized_{norm}'].mean()
                row[f'conditional_mean_normalized_{norm}']=g.loc[~g.covered,f'normalized_{norm}'].mean()
            parts.append(row)
    return pd.DataFrame(parts)


def export_tables(metrics,output):
    output.mkdir(parents=True,exist_ok=True)
    for norm in NORMALIZATIONS:
        for table_id,d in metrics.groupby('table_id'):
            rows=[]
            for method,m in d.groupby('method'):
                row={'Model':method,'SKUs':int(m.skus.iloc[0]),'Observations':int(m.observations.iloc[0])}
                for level in INTERVALS:
                    r=m[m.interval_level.eq(level)].iloc[0]
                    row[f'{level}% Coverage']=r.coverage
                    row[f'{level}% Mean Width']=r.mean_width
                    row[f'{level}% Mean Normalized Outside Distance']=r[f'mean_normalized_{norm}']
                rows.append(row)
            write_csv(pd.DataFrame(rows),output/f'{table_id}__{norm}.csv')


def combine_tables(output):
    folder=output/'combined_tables';folder.mkdir(exist_ok=True)
    for norm in NORMALIZATIONS:
        frames=[]
        for path in sorted(output.glob(f'*/tables/*__{norm}.csv')):
            f=pd.read_csv(path)
            f.insert(0,'Table',path.stem.split('__')[0])
            f.insert(0,'Case',path.parent.parent.name)
            frames.append(f)
        if frames:write_csv(pd.concat(frames,ignore_index=True),folder/f'B_tables_{norm}.csv')


def main():
    raise RuntimeError('Legacy standalone workflow disabled. Use test_zero_threshold/run_experiment.py.')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path(__file__).parent/'input'/'weekly_sales.csv')
    p.add_argument('--output',type=Path,default=Path(__file__).parent/'results')
    p.add_argument('--zero-threshold',type=float,default=.6)
    p.add_argument('--max-rounds',type=int,default=200);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--include-week43-sensitivity',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    wide=pd.read_csv(args.input,dtype={'sku_id':str}).set_index('sku_id')
    wide.columns=pd.DatetimeIndex([week_date(c) for c in wide.columns]);wide=wide.sort_index(axis=1)
    complete_dates=wide.columns[wide.columns<=week_date('2026-W42')]
    split=len(complete_dates)*2//3;train_dates=complete_dates[:split]
    assert len(train_dates)==36,'This B revision expects the supplied 2025-W41 to 2026-W43 dataset.'
    eligible=wide.loc[:,train_dates].notna().all(axis=1);ids=wide.index[eligible]
    features=pd.DataFrame([dict(sku_id=sku,**profile(wide.loc[sku,train_dates].to_numpy(float),args.zero_threshold)) for sku in ids])
    write_csv(features,args.output/'training_profiles.csv')
    # Recreate the historical reassignment audit on a clean run as well.
    # This report does not define an additional group or alter any sales.
    reassigned=features.loc[features.has_negative_training & features.demand_group.ne('Zero'),
                           ['sku_id','demand_group','has_negative_training','adi','positive_cv2']]
    write_csv(reassigned,args.output/'group_reassignment.csv')
    write_csv(pd.DataFrame({'sku_id':wide.index[~eligible],'reason':'Missing training history; not filled with zero'}),args.output/'excluded_skus.csv')
    cases=[('main_W42',complete_dates[split:])]
    if args.include_week43_sensitivity:cases.append(('sensitivity_W43',wide.columns[(wide.columns>train_dates.max())&(wide.columns<=week_date('2026-W43'))]))
    maxh=max(int((dates.max()-train_dates.max()).days//7) for _,dates in cases)
    horizons=np.arange(1,maxh+1)
    light,training_log=fit_lightgbm(wide,features,train_dates,horizons,args.output/'models',args.max_rounds,args.threads)
    all_metrics=[];case_log=[]
    for name,dates in cases:
        print('Evaluating',name,flush=True);folder=args.output/name;folder.mkdir(exist_ok=True)
        pred=generate_predictions(wide,features,train_dates,dates,light,horizons)
        pred.to_csv(folder/'predictions.csv.gz',index=False,encoding='utf-8-sig',compression='gzip')
        points=interval_points(pred)
        metrics=summaries(points);metrics['case']=name;all_metrics.append(metrics)
        write_csv(metrics,folder/'metrics_long.csv');export_tables(metrics,folder/'tables')
        outside=points.loc[~points.covered]
        outside.to_csv(folder/'all_outside_points.csv.gz',index=False,encoding='utf-8-sig',compression='gzip')
        count=points.groupby(['method','demand_group','interval_level','iso_week']).agg(observations=('covered','size'),coverage=('covered','mean'),mean_sd_normalized_distance=('normalized_training_sd','mean')).reset_index()
        write_csv(count,folder/'weekly_group_metrics.csv')
        case_log.append(dict(case=name,training_start=week_label(train_dates.min()),training_end=week_label(train_dates.max()),
             training_weeks=len(train_dates),test_start=week_label(dates.min()),test_end=week_label(dates.max()),
             test_weeks=len(dates),training_complete_skus=len(features),outside_rows_all_intervals=len(outside)))
        del pred,points,outside;gc.collect()
    write_csv(pd.concat(all_metrics,ignore_index=True),args.output/'all_metrics.csv')
    combine_tables(args.output)
    config=dict(experiment='B',zero_threshold=args.zero_threshold,
      zero_policy='Training zero share >= threshold => point forecast and every prediction interval equal zero for all methods; test demand remains scored',
      demand_groups=['Smooth','Erratic','Intermittent','Lumpy','Zero'],
      negative_sales_policy='Returns retained at original signed values; no separate returns group',
      classification='Zero share first; remaining groups use positive-demand ADI and CV squared regardless of returns',
      quantiles=QUANTILES.tolist(),intervals=list(INTERVALS),
      distance='max(lower - actual, actual - upper, 0)',
      table_mean='All scored observations; covered observations contribute zero',
      plot_points='All outside observations; no downsampling',
      normalizations={'training_sd':'distance / max(sample standard deviation of training sales, 1 unit)',
        'nonzero_median':'distance / max(median(abs(nonzero training sales)), 1 unit)',
        'interval_width':'distance / max(predicted upper - lower, 1 unit)'},
      group_counts=features.demand_group.value_counts().to_dict(),
      common_six_method_skus=int(features.six_method_eligible.sum()),
      lightgbm_version=lgb.__version__,max_rounds=args.max_rounds,threads=args.threads,
      training_log=training_log,cases=case_log,
      input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest())
    (args.output/'settings.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(json.dumps({'group_counts':config['group_counts'],'cases':case_log}),flush=True)


if __name__=='__main__':
    raise SystemExit('Use test_zero_threshold/run_experiment.py; this file supplies model helpers only.')
