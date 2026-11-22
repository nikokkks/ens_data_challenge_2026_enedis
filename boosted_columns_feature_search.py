"""Search 25 leak-free feature families with curves/columns as individuals."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import boosted_columns_eda_bench as base
import tabfm_lora_finetune as ft
import tabpfn_column_individual_bench as column_bench
from evaluate import load_data

SEED = 42
SCREEN_SEED = 42
CONFIRM_SEED = 1042
BASE_GROUPS = ("dynamics", "profiles")
RESULTS = Path("boosted_columns_feature_search_results.json")
REPORT = Path("boosted_columns_feature_search_results.md")

FAMILIES = (
    "short_lags", "periodic_lags", "rolling_location", "rolling_quantiles",
    "rolling_dispersion", "local_ramps", "local_acceleration", "boundary_trends",
    "polynomial_bridges", "log_bridge", "ewma_context", "weekday_slot_profile",
    "daytype_slot_profile", "recent_weighted_profile", "homologous_day_trend",
    "fourier_calendar", "curve_load_shape", "curve_ramp_signature",
    "curve_autocorrelation", "curve_spectral_signature", "load_period_ratios",
    "donor_trimmed_consensus", "donor_slot_reliability", "donor_local_calibration",
    "donor_low_rank",
)


def _safe(x):
    return np.nan_to_num(np.asarray(x, dtype=float), nan=0., posinf=0., neginf=0.).astype(np.float32)


def _shift(v, lag):
    out=np.full(len(v),np.nan)
    if lag>0: out[lag:]=v[:-lag]
    elif lag<0: out[:lag]=v[-lag:]
    else: out[:]=v
    return out


def _boundaries(v):
    n=len(v); pos=np.arange(n); known=np.isfinite(v)
    prev=np.maximum.accumulate(np.where(known,pos,-1)); nxt=np.minimum.accumulate(np.where(known,pos,n)[::-1])[::-1]
    li=np.r_[-1,prev[:-1]]; ri=np.r_[nxt[1:],n]
    left=np.where(li>=0,v[np.maximum(li,0)],np.nan); right=np.where(ri<n,v[np.minimum(ri,n-1)],np.nan)
    return left,right,li,ri


def _two_sided_rolling(v, window, stat):
    s=pd.Series(v); past=s.shift(1).rolling(window,min_periods=2)
    reversed_future=s.shift(-1)[::-1].rolling(window,min_periods=2)
    if stat=="median": a,b=past.median(),reversed_future.median()
    elif stat=="mean": a,b=past.mean(),reversed_future.mean()
    elif stat=="std": a,b=past.std(),reversed_future.std()
    else: a,b=past.quantile(float(stat)),reversed_future.quantile(float(stat))
    return a.to_numpy(),b[::-1].to_numpy()


def _loo_slot_profile(v, keys, center):
    out=np.full(len(v),center); frame=pd.DataFrame({"v":v,"k":keys})
    for _,idx in frame.groupby("k").groups.items():
        idx=np.asarray(list(idx),int); z=v[idx]; finite=np.isfinite(z)
        if finite.any():
            med=float(np.median(z[finite])); out[idx]=med
            # Explicit leave-one-out for any observed training timestamp.
            for i in idx[finite]:
                zz=z[finite & (idx!=i)]
                if len(zz): out[i]=float(np.median(zz))
    return out


def build_25_features(s, index, neighbors, center, scale):
    v=s.to_numpy(float).copy(); missing=~np.isfinite(v); n=len(v); pos=np.arange(n)
    dt=pd.DatetimeIndex(pd.to_datetime(index)); slot=(dt.hour*2+dt.minute//30).to_numpy()
    dow=dt.dayofweek.to_numpy(); left,right,li,ri=_boundaries(v)
    local=(left+right)/2; local=np.where(np.isfinite(local),local,center)
    feats={}

    feats["short_lags"]=_safe(np.column_stack([_shift(v,k) for k in (1,-1,2,-2,3,-3,6,-6,12,-12,24,-24)])/scale)
    feats["periodic_lags"]=_safe(np.column_stack([_shift(v,k) for k in (48,-48,96,-96,144,-144,240,-240,336,-336)])/scale)

    loc=[]; disp=[]
    for w in (6,12,24,48,96):
        pm,fm=_two_sided_rolling(v,w,"median"); pa,fa=_two_sided_rolling(v,w,"mean")
        ps,fs=_two_sided_rolling(v,w,"std"); loc += [(pm-center)/scale,(fm-center)/scale,(pa-center)/scale,(fa-center)/scale]
        disp += [ps/scale,fs/scale,(fm-pm)/scale]
    feats["rolling_location"]=_safe(np.column_stack(loc))
    quant=[]
    for w in (24,48,96):
        for q in (.1,.25,.75,.9):
            a,b=_two_sided_rolling(v,w,q); quant += [(a-center)/scale,(b-center)/scale]
    feats["rolling_quantiles"]=_safe(np.column_stack(quant))
    feats["rolling_dispersion"]=_safe(np.column_stack(disp))

    ramps=[]; acc=[]
    for k in (1,2,3,6,12,24):
        past=_shift(v,k); future=_shift(v,-k)
        ramps += [(v-past)/(k*scale),(future-v)/(k*scale),(future-past)/(2*k*scale)]
    d=np.diff(v,prepend=np.nan); dd=np.diff(d,prepend=np.nan)
    for k in (1,2,3,6,12): acc += [_shift(dd,k)/scale,_shift(dd,-k)/scale]
    feats["local_ramps"]=_safe(np.column_stack(ramps)); feats["local_acceleration"]=_safe(np.column_stack(acc))

    trend=[]
    for w in (2,4,8,16,32):
        lp=np.clip(li-w,0,n-1); rr=np.clip(ri+w,0,n-1)
        ls=np.where((li>=w)&np.isfinite(v[lp]),(left-v[lp])/w,np.nan)
        rs=np.where((ri+w<n)&np.isfinite(v[rr]),(v[rr]-right)/w,np.nan)
        trend += [ls/scale,rs/scale,(rs-ls)/scale]
    feats["boundary_trends"]=_safe(np.column_stack(trend))
    duration=np.maximum(ri-li,1); u=np.clip((pos-li)/duration,0,1)
    linear=(1-u)*left+u*right; midpoint=(left+right)/2
    quad=linear+4*u*(1-u)*(midpoint-linear)
    smooth=(1-(3*u*u-2*u*u*u))*left+(3*u*u-2*u*u*u)*right
    feats["polynomial_bridges"]=_safe(np.column_stack([(linear-center)/scale,(quad-center)/scale,(smooth-center)/scale,(quad-linear)/scale]))
    logleft=np.log1p(np.maximum(left,0)); logright=np.log1p(np.maximum(right,0))
    logpred=np.expm1((1-u)*logleft+u*logright)
    feats["log_bridge"]=_safe(np.column_stack([(logpred-center)/scale,(logpred-linear)/scale,(left<0),(right<0)]))

    ew=[]; ss=pd.Series(v)
    for span in (6,12,24,48,96):
        p=ss.shift(1).ewm(span=span,min_periods=2).mean().to_numpy()
        f=ss.shift(-1)[::-1].ewm(span=span,min_periods=2).mean()[::-1].to_numpy()
        ew += [(p-center)/scale,(f-center)/scale,(f-p)/scale]
    feats["ewma_context"]=_safe(np.column_stack(ew))

    weekday=_loo_slot_profile(v,dow*48+slot,center); daytype=np.where(dow<5,0,np.where(dow==5,1,2))
    dtype=_loo_slot_profile(v,daytype*48+slot,center)
    feats["weekday_slot_profile"]=_safe(np.column_stack([(weekday-center)/scale,(local-weekday)/scale]))
    feats["daytype_slot_profile"]=_safe(np.column_stack([(dtype-center)/scale,(local-dtype)/scale]))
    candidates=[]; weights=[]
    for lag,w in ((48,1.),(-48,1.),(96,.7),(-96,.7),(144,.49),(-144,.49),(336,.35),(-336,.35)):
        z=_shift(v,lag); candidates.append(z); weights.append(np.isfinite(z)*w)
    C=np.column_stack(candidates); W=np.column_stack(weights); recent=np.divide(np.nansum(C*W,1),W.sum(1),out=np.full(n,np.nan),where=W.sum(1)>0)
    feats["recent_weighted_profile"]=_safe(np.column_stack([(recent-center)/scale,(recent-local)/scale,W.sum(1)]))
    d1p=_shift(v,48)-_shift(v,96); d1f=_shift(v,-48)-_shift(v,-96)
    feats["homologous_day_trend"]=_safe(np.column_stack([d1p/scale,d1f/scale,(d1p+d1f)/(2*scale),(_shift(v,48)+d1p-center)/scale,(_shift(v,-48)+d1f-center)/scale]))

    phase_day=2*np.pi*slot/48.; phase_week=2*np.pi*(dow*48+slot)/336.
    four=[]
    for h in (1,2,3,4,6,8): four += [np.sin(h*phase_day),np.cos(h*phase_day)]
    for h in (1,2,3): four += [np.sin(h*phase_week),np.cos(h*phase_week)]
    feats["fourier_calendar"]=_safe(np.column_stack(four))

    obs=v[np.isfinite(v)]; q=np.nanpercentile(obs,[1,5,10,25,50,75,90,95,99]) if len(obs) else np.zeros(9)
    feats["curve_load_shape"]=_safe(np.repeat(np.r_[q/scale,np.mean(obs)/scale,np.std(obs)/scale][None],n,0))
    dv=np.diff(obs) if len(obs)>1 else np.zeros(1); rq=np.percentile(dv,[1,5,25,50,75,95,99])
    feats["curve_ramp_signature"]=_safe(np.repeat(np.r_[rq/scale,np.mean(np.abs(dv))/scale,np.std(dv)/scale][None],n,0))
    ac=[]
    for lag in (1,2,3,6,12,24,48,96,144,336):
        a=v[:-lag]; b=v[lag:]; ok=np.isfinite(a)&np.isfinite(b)
        ac.append(np.corrcoef(a[ok],b[ok])[0,1] if ok.sum()>5 and np.std(a[ok])>0 and np.std(b[ok])>0 else 0.)
    feats["curve_autocorrelation"]=_safe(np.repeat(np.asarray(ac)[None],n,0))
    filled=pd.Series(v).interpolate(limit_direction="both").fillna(center).to_numpy(); spec=np.abs(np.fft.rfft(filled-center))
    bands=np.array([spec[1:4].sum(),spec[4:8].sum(),spec[8:16].sum(),spec[16:32].sum(),spec[32:65].sum()])/(spec[1:].sum()+1e-9)
    feats["curve_spectral_signature"]=_safe(np.repeat(bands[None],n,0))
    periods=[(slot<12),(slot>=12)&(slot<34),(slot>=34),(dow>=5)]
    ratios=[]
    for mask in periods:
        z=v[mask&np.isfinite(v)]; ratios.append((np.median(z)-center)/scale if len(z) else 0.)
    ratios += [np.max(obs)/(np.mean(obs)+1e-9) if len(obs) else 0.,np.mean(obs<.1*np.median(obs)) if len(obs) else 0.]
    feats["load_period_ratios"]=_safe(np.repeat(np.asarray(ratios)[None],n,0))

    N=np.asarray(neighbors,float)
    nq=np.nanpercentile(N,[10,20,25,50,75,80,90],axis=1).T; trimmed=np.nanmean(np.sort(N,axis=1)[:,4:-4],axis=1)
    feats["donor_trimmed_consensus"]=_safe(np.column_stack([(trimmed-center)/scale,(nq-center)/scale,(nq[:,5]-nq[:,1])/scale]))
    residual=np.abs(N-v[:,None]); residual[missing]=np.nan
    rel=[]
    keys=daytype*48+slot
    for j in range(N.shape[1]):
        frame=pd.DataFrame({"r":residual[:,j],"k":keys}); mp=frame.groupby("k").r.transform("median").to_numpy(); rel.append(mp)
    R=np.column_stack(rel); w=1/(R+.05*scale); slotpred=np.nansum(N*w,1)/np.maximum(np.nansum(np.isfinite(N)*w,1),1e-9)
    feats["donor_slot_reliability"]=_safe(np.column_stack([(slotpred-center)/scale,np.nanmedian(R,1)/scale,np.nanmin(R,1)/scale]))
    calibrated=[]
    for j in range(min(12,N.shape[1])):
        ok=np.isfinite(v)&np.isfinite(N[:,j]); x=N[ok,j]; yy=v[ok]
        if ok.sum()>10 and np.std(x)>1e-8:
            slope=np.cov(x,yy,bias=True)[0,1]/np.var(x); intercept=np.mean(yy)-slope*np.mean(x)
            calibrated.append(intercept+slope*N[:,j])
        else: calibrated.append(N[:,j])
    Cal=np.column_stack(calibrated)
    feats["donor_local_calibration"]=_safe(np.column_stack([(np.nanmedian(Cal,1)-center)/scale,np.nanstd(Cal,1)/scale,np.nanpercentile(Cal,[25,75],axis=1).T/scale]))
    Nc=np.nan_to_num(N-np.nanmean(N,axis=1,keepdims=True)); U,S,_=np.linalg.svd(Nc,full_matrices=False)
    low=U[:,:3]*S[:3]; feats["donor_low_rank"]=_safe(np.column_stack([low/scale,(np.nanmean(N,1)-center)/scale,np.nanstd(N,1)/scale]))
    assert tuple(feats)==FAMILIES
    return feats


def build_row(col,X,y,context):
    row=base.build_rows(col,X,y,context)
    if row is None: return None
    s=X[col]; built=ft.feature_matrix(col,s,*context,s.index)
    matrix=built[0]; neighbors=matrix[:,:40]
    all_features=build_25_features(s,s.index,neighbors,row["center"],row["scale"])
    row["features"].update({name:block[row["q"]] for name,block in all_features.items()})
    return row


def splits(columns,repeats,nval,start_seed):
    from sklearn.model_selection import train_test_split
    for repeat in range(repeats):
        seed=start_seed+repeat; dev,hold=train_test_split(list(columns),test_size=.1,random_state=seed)
        val=list(np.random.RandomState(seed).choice(hold,min(nval,len(hold)),replace=False))
        yield repeat,dev,val


def evaluate(name,groups,cache,X,y,repeats,nval,start_seed):
    scores=[]; dims=None; started=time.time()
    for repeat,dev,val in splits(y.columns,repeats,nval,start_seed):
        train=[cache[c] for c in dev if cache[c] is not None]; valid=[cache[c] for c in val if cache[c] is not None]
        score,nrows,nfeatures=base.score_model(base.make_model("xgboost",start_seed+repeat),train,valid,groups,X,y)
        scores.append(score); dims=(nrows,nfeatures)
        print(f"{name} repeat {repeat+1}/{repeats}: {score:.3f} W",flush=True)
    return {"groups":list(groups),"scores":scores,"mean":float(np.mean(scores)),"std":float(np.std(scores)),
            "n_train_rows":dims[0],"n_features":dims[1],"seconds":time.time()-started}


def save(payload):
    tmp=Path(str(RESULTS)+".tmp"); tmp.write_text(json.dumps(payload,indent=2)); os.replace(tmp,RESULTS)


def write_report(r):
    accepted=r["confirmation"]["accepted"]
    lines=["# Recherche de 25 familles de features","",f"Baseline: {r['confirmation']['baseline']['mean']:.3f} W",
           f"Modèle enrichi: {r['confirmation']['enriched']['mean']:.3f} W",
           f"Delta enrichi - baseline: {r['confirmation']['delta']:+.3f} W",
           f"Repeats gagnés: {r['confirmation']['wins']}/5",
           f"Décision finale: **{'accepter les nouvelles features' if accepted else 'conserver la baseline'}**.",
           "", "## Tests individuels",""]
    for name,row in sorted(r["individual"].items(),key=lambda kv:kv[1]["mean"]):
        lines.append(f"- {name}: {row['mean']:.3f} W ({row['delta']:+.3f} W, {row['wins']}/3)")
    lines += ["", "Retenues au screening: **"+(", ".join(r["selected"]) or "aucune")+"**."]
    REPORT.write_text("\n".join(lines)+"\n")


def run_search(X,y,cache):
    r={"baseline_groups":list(BASE_GROUPS),"individual":{},"forward":{},"selected":[],"confirmation":{}}
    baseline=evaluate("screen_baseline",BASE_GROUPS,cache,X,y,3,20,SCREEN_SEED); r["screen_baseline"]=baseline; save(r)
    b=np.asarray(baseline["scores"]); candidates=[]
    for family in FAMILIES:
        row=evaluate("feature_"+family,BASE_GROUPS+(family,),cache,X,y,3,20,SCREEN_SEED)
        s=np.asarray(row["scores"]); row["delta"]=float(s.mean()-b.mean()); row["wins"]=int((s<b).sum())
        r["individual"][family]=row
        if row["delta"]<0 and row["wins"]>=2: candidates.append(family)
        save(r)
    candidates.sort(key=lambda x:r["individual"][x]["mean"]); selected=[]; current=b
    for family in candidates:
        trial=selected+[family]; row=evaluate("forward_"+"+".join(trial),BASE_GROUPS+tuple(trial),cache,X,y,3,20,SCREEN_SEED)
        s=np.asarray(row["scores"]); row["delta_previous"]=float(s.mean()-current.mean()); row["wins_previous"]=int((s<current).sum())
        r["forward"]["+".join(trial)]=row
        if row["delta_previous"]<0 and row["wins_previous"]>=2: selected=trial; current=s
        r["selected"]=selected; save(r)
    cb=evaluate("confirm_baseline",BASE_GROUPS,cache,X,y,5,100,CONFIRM_SEED)
    ce=evaluate("confirm_enriched",BASE_GROUPS+tuple(selected),cache,X,y,5,100,CONFIRM_SEED)
    bs=np.asarray(cb["scores"]); es=np.asarray(ce["scores"])
    delta=float(es.mean()-bs.mean()); wins=int((es<bs).sum()); accepted=bool(delta<0 and wins>=4)
    r["confirmation"]={"baseline":cb,"enriched":ce,"delta":delta,"wins":wins,"accepted":accepted}
    r["recommended_groups"]=list(BASE_GROUPS+tuple(selected)) if accepted else list(BASE_GROUPS)
    save(r); write_report(r); return r


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("stage",choices=("build","search","all"),nargs="?",default="all")
    args=parser.parse_args(); X,y,_=load_data(); donors=[c for c in X if not str(c).startswith("holed_")]
    context=ft.build_donor_context(X,donors); cache={}
    for i,col in enumerate(y.columns,1):
        cache[col]=build_row(col,X,y,context)
        if i%25==0: print(f"features {i}/{len(y.columns)}",flush=True)
    if args.stage=="build": return
    result=run_search(X,y,cache); print(json.dumps(result["confirmation"],indent=2),flush=True)


if __name__=="__main__": main()
