import numpy as np
from scipy import stats

def paired_summary(a,b,alpha=0.05):
    a=np.asarray(a,float); b=np.asarray(b,float)
    d=b-a
    n=len(d); mean=d.mean(); sd=d.std(ddof=1); se=sd/np.sqrt(n)
    q=stats.t.ppf(1-alpha/2,n-1)
    lo,hi=mean-q*se,mean+q*se
    dz=mean/sd
    t,p=stats.ttest_rel(b,a)
    return dict(mean_difference=mean,ci_low=lo,ci_high=hi,dz=dz,t=t,p=p)

def holm(pvalues):
    p=np.asarray(pvalues,float); m=len(p)
    order=np.argsort(p); out=np.empty(m)
    running=0.0
    for rank,idx in enumerate(order):
        val=min(1.0,(m-rank)*p[idx])
        running=max(running,val)
        out[idx]=running
    return out

def class_stratified_bootstrap(y_true, pred_a_by_seed, pred_b_by_seed,
                               n_boot=10000, seed=202601):
    """
    Same resampled class-stratified image indices are applied to both methods
    and all matched seed pairs. Returns averaged paired accuracy differences.
    Extend analogously for macro F1.
    """
    rng=np.random.default_rng(seed)
    y=np.asarray(y_true)
    classes=np.unique(y)
    by_class={c:np.flatnonzero(y==c) for c in classes}
    diffs=[]
    for _ in range(n_boot):
        idx=np.concatenate([rng.choice(v,size=len(v),replace=True)
                            for v in by_class.values()])
        ds=[]
        for pa,pb in zip(pred_a_by_seed,pred_b_by_seed):
            ds.append(100*((np.asarray(pb)[idx]==y[idx]).mean()
                           -(np.asarray(pa)[idx]==y[idx]).mean()))
        diffs.append(np.mean(ds))
    return np.percentile(diffs,[2.5,97.5])
