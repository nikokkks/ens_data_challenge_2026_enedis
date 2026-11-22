"""EDA reproductible centrée sur les erreurs et les trous du challenge."""
import json
import numpy as np, pandas as pd
from evaluate import load_data, fill_nan_with_interpolation

X,y,Xt=load_data(); cols=list(y.columns); A=X[cols].to_numpy(float); Y=y.to_numpy(float)
missing=np.isnan(A); scored=missing&np.isfinite(Y); idx=pd.DatetimeIndex(X.index)
runs=[]
for j in range(len(cols)):
    d=np.diff(np.r_[False,missing[:,j],False].astype(np.int8)); runs.extend((np.where(d==-1)[0]-np.where(d==1)[0]).tolist())
E=np.full_like(Y,np.nan)
for j in range(len(cols)):
    p=fill_nan_with_interpolation(pd.Series(A[:,j])).to_numpy(); m=scored[:,j]&np.isfinite(p); E[m,j]=np.abs(p[m]-Y[m,j])
curve_mae=np.nanmean(E,axis=0); level=np.nanmedian(Y,axis=0); vol=np.nanmedian(np.abs(np.diff(Y,axis=0)),axis=0)
span=np.nanpercentile(Y,95,axis=0)-np.nanpercentile(Y,5,axis=0)
def corr(z):
    ok=np.isfinite(curve_mae)&np.isfinite(z); return float(np.corrcoef(curve_mae[ok],z[ok])[0,1])
def grouped(key):
    return {str(k):{"n":int(scored[np.asarray(key)==k].sum()),"mae_interp":float(np.nanmean(E[np.asarray(key)==k]))} for k in np.unique(key)}
report={
 "shapes":{"X_train":X.shape,"y_train":y.shape,"X_test":Xt.shape},
 "timeline":{"start":str(idx[0]),"end":str(idx[-1]),"cadence_minutes":float((idx[1]-idx[0]).total_seconds()/60)},
 "missing":{"total":int(missing.sum()),"scored":int(scored.sum()),"rate":float(missing.mean()),
            "per_curve_quantiles":np.quantile(missing.sum(0),[0,.1,.5,.9,1]).tolist(),
            "run_length_quantiles":np.quantile(runs,[0,.5,.75,.9,.95,.99,1]).tolist()},
 "interpolation":{"mae_watts":float(np.nanmean(E)),"curve_mae_quantiles":np.nanquantile(curve_mae,[0,.1,.5,.9,1]).tolist()},
 "difficulty_correlations":{"level":corr(level),"volatility":corr(vol),"amplitude":corr(span),"missing_count":corr(scored.sum(0))},
 "by_dayofweek":grouped(idx.dayofweek),"by_weekend":grouped((idx.dayofweek>=5).astype(int)),
 "by_four_hour_band":grouped((idx.hour*2+idx.minute//30)//8),
}
with open("eda_v15_report.json","w") as f: json.dump(report,f,indent=2)
print(json.dumps(report,indent=2))
