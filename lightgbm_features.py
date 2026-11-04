"""Leakage-safe lag features and multi-horizon training examples."""
import numpy as np

def origin_features(history):
    """Return only features available at this origin, and its quantity scale."""
    n,t=history.shape
    scale=np.maximum(np.mean(np.abs(history),axis=1),1.)
    features=[];names=[]
    def add(name,value):
        names.append(name);features.append(np.asarray(value,dtype=np.float32))
    for k in [1,2,4,8,13]:add(f'lag_{k}',history[:,-k]/scale)
    for w in [4,8,13,26]:
        z=history[:,-w:]
        add(f'mean_{w}',z.mean(axis=1)/scale)
        add(f'std_{w}',z.std(axis=1)/scale)
        add(f'zero_share_{w}',(z==0).mean(axis=1))
    pos=history>0;np_=pos.sum(axis=1)
    positive_sum=np.where(pos,history,0).sum(axis=1)
    posmean=np.divide(positive_sum,np_,out=np.zeros(n),where=np_>0)
    add('positive_mean',posmean/scale)
    add('positive_frequency',pos.mean(axis=1))
    add('negative_frequency',(history<0).mean(axis=1))
    since=np.where(pos.any(axis=1),np.argmax(pos[:,::-1],axis=1),t)
    add('weeks_since_positive',since)
    z=history[:,-13:];x=np.arange(z.shape[1]);xc=x-x.mean()
    add('slope_13',(z@xc)/np.dot(xc,xc)/scale)
    add('recent_level_change',(history[:,-4:].mean(axis=1)-z.mean(axis=1))/scale)
    add('history_weeks',np.full(n,t))
    return np.column_stack(features),scale,names

def examples(values,max_horizon,min_history=13):
    n,t=values.shape;parts=[];labels=[];targets=[];origins=[]
    for origin in range(min_history,t):
        x,scale,names=origin_features(values[:,:origin])
        for h in range(1,min(max_horizon,t-origin)+1):
            parts.append(np.column_stack([x,np.full(n,h,dtype=np.float32)]))
            labels.append((values[:,origin+h-1]/scale).astype(np.float32))
            targets.append(np.full(n,origin+h-1,dtype=np.int16))
            origins.append(np.full(n,origin,dtype=np.int16))
    return (np.concatenate(parts).astype(np.float32),np.concatenate(labels),
            np.concatenate(targets),np.concatenate(origins),names+['horizon_weeks'])

def validation_block(values,origin,max_horizon):
    x,scale,_=origin_features(values[:,:origin]);parts=[];ys=[]
    for h in range(1,min(max_horizon,values.shape[1]-origin)+1):
        parts.append(np.column_stack([x,np.full(len(x),h,dtype=np.float32)]))
        ys.append((values[:,origin+h-1]/scale).astype(np.float32))
    return np.concatenate(parts).astype(np.float32),np.concatenate(ys)

