"""EDA and boosted-tree ablations for column-centred Enedis imputation."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from evaluate import load_data
import feature_engineering_bench as feb
import tabfm_lora_finetune as ft
import tabpfn_column_individual_bench as column_bench
from tabpfn_v15_features import build_v15_features


SEED=42
EDA_JSON=Path("boosted_columns_eda.json")
EDA_MD=Path("boosted_columns_eda.md")
RESULTS=Path("boosted_columns_results.json")
RESULTS_MD=Path("boosted_columns_results.md")
BASELINE="baseline96"
FEATURE_GROUPS=("boundaries","dynamics","profiles","donor_quality","curve_mask_stats")


def atomic_json(path,payload):
    tmp=Path(str(path)+".tmp"); tmp.write_text(json.dumps(payload,indent=2)); os.replace(tmp,path)


def write_results_report(results):
    summary=results["confirmation_summary"]
    lines=["# Benchmark arbres boostés colonne-centré","",
           "## Comparaison des modèles (screening 3 × 20 courbes)",""]
    for name,row in results["screen"].items():
        lines.append(f"- {name}: {row['mean']:.3f} W")
    lines.extend(["",f"Modèle retenu : **{results['winner_model']}**.","",
                  "## Ablations de features",""])
    for name,row in results["feature_tests"].items():
        lines.append(f"- {name}: {row['mean']:.3f} W "
                     f"(delta {row['delta_vs_baseline']:+.3f} W, "
                     f"{row['wins_vs_baseline']}/3 repeats gagnés)")
    selected=", ".join(results["selected_groups"]) or "aucune"
    lines.extend(["",f"Features retenues : **{selected}**.","",
                  "## Confirmation (5 × 100 courbes)","",
                  f"- Baseline 96 features : {summary['baseline_mae_w']:.3f} W",
                  f"- Modèle enrichi : {summary['enriched_mae_w']:.3f} W",
                  f"- Gain : {-summary['delta_w']:.3f} W",
                  f"- Repeats gagnés : {summary['wins']}/5"])
    RESULTS_MD.write_text("\n".join(lines)+"\n")


def run_eda(X,y):
    rows=[]; gap_lengths=[]; interpolation_errors=[]
    for col in y.columns:
        s=X[col]; truth=y[col]; observed=s.dropna().to_numpy(float)
        missing=s.isna().to_numpy(); runs=feb.run_lengths_for(missing)
        starts=np.flatnonzero(missing & ~np.r_[False,missing[:-1]])
        lengths=[int(runs[i]) for i in starts]; gap_lengths.extend(lengths)
        valid=missing & truth.notna().to_numpy()
        interp=s.interpolate(limit_direction="both").to_numpy(float)
        interpolation_errors.extend(np.abs(interp[valid]-truth.to_numpy(float)[valid]))
        center=float(np.median(observed)); mad=float(np.median(np.abs(observed-center))*1.4826)
        rows.append({"column":col,"observed":int(len(observed)),"missing":int(missing.sum()),
                     "missing_fraction":float(missing.mean()),"median_w":center,"mad_w":mad,
                     "n_gaps":len(lengths),"max_gap":max(lengths,default=0)})
    frame=pd.DataFrame(rows); gaps=np.asarray(gap_lengths); errors=np.asarray(interpolation_errors)
    payload={"n_curves":len(frame),"n_timestamps":len(X),
             "curve_summary":frame.drop(columns="column").describe().to_dict(),
             "gap_length_quantiles":{str(q):float(np.quantile(gaps,q)) for q in (0,.25,.5,.75,.9,.95,.99,1)},
             "interpolation_mae_w":float(errors.mean()),"n_scored_holes":int(len(errors)),
             "feature_hypotheses":list(FEATURE_GROUPS)}
    atomic_json(EDA_JSON,payload)
    lines=["# EDA colonne-centrée",f"- Courbes supervisées : {len(frame)}",
           f"- Timestamps : {len(X)}",f"- Points manquants notables : {len(errors)}",
           f"- MAE interpolation : {errors.mean():.3f} W",
           f"- Médiane longueur de trou : {np.median(gaps):.1f}",
           f"- P95 longueur de trou : {np.quantile(gaps,.95):.1f}","",
           "## Familles à tester",*[f"- {name}" for name in FEATURE_GROUPS]]
    EDA_MD.write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True); return payload


def slot48_profile(s,center,scale):
    values=s.to_numpy(float); dt=pd.DatetimeIndex(s.index)
    slot=dt.hour.to_numpy()*2+dt.minute.to_numpy()//30
    out=[]
    for k in range(48):
        z=values[(slot==k)&np.isfinite(values)]
        out.append((np.median(z)-center)/scale if len(z) else 0.)
    return np.asarray(out,np.float32)


def curve_mask_stats(s,center,scale):
    missing=s.isna().to_numpy(); runs=feb.run_lengths_for(missing)
    lengths=runs[np.flatnonzero(missing & ~np.r_[False,missing[:-1]])]
    values=s.dropna().to_numpy(float)
    return np.asarray([missing.mean(),len(lengths)/len(s),np.mean(lengths) if len(lengths) else 0.,
                       np.max(lengths) if len(lengths) else 0.,np.log1p(scale),
                       center/scale,np.std(values)/scale if len(values) else 0.],np.float32)


def build_rows(col,X,y,context):
    s=X[col]; q=ft.real_gap_indices(s,y[col])
    if not len(q): return None
    built=ft.feature_matrix(col,s,*context,s.index)
    if built is None: return None
    matrix,_,_,interp,missing,_=built
    center,scale=column_bench.robust_location_scale(s)
    base=matrix[q,:72]
    if base.shape[1]<72: base=np.pad(base,((0,0),(0,72-base.shape[1])))
    profile=np.repeat(column_bench.curve_profile(s,s.index,center,scale)[None],len(q),axis=0)
    geometry=column_bench.point_geometry(s,q,interp.to_numpy(float),center,scale)
    features={BASELINE:np.hstack([base,profile,geometry]).astype(np.float32)}
    scaled=matrix[:,:40]
    v=s.to_numpy(float)
    definitions={
        "boundaries":("boundary_slopes","gap_geometry"),
        "dynamics":("local_volatility","neighbor_dynamics","periodic_residuals"),
        "profiles":("daytype_profile","recent_profile"),
        "donor_quality":("neighbor_reliability","neighbor_distribution"),
    }
    for name,groups in definitions.items():
        extra=build_v15_features(v,s.index,scaled,missing,groups)[q].astype(np.float32)
        if name=="profiles":
            p=np.repeat(slot48_profile(s,center,scale)[None],len(q),axis=0)
            extra=np.hstack([extra,p])
        features[name]=extra
    stats=np.repeat(curve_mask_stats(s,center,scale)[None],len(q),axis=0)
    features["curve_mask_stats"]=stats
    target=((y[col].iloc[q].to_numpy(float)-center)/scale).astype(np.float32)
    return {"col":col,"features":features,"target":target,"q":q,"center":center,
            "scale":scale,"interp":interp.to_numpy(float),"missing":missing}


def matrix(rows,groups):
    names=(BASELINE,)+tuple(groups)
    X=np.vstack([np.hstack([row["features"][name] for name in names]) for row in rows]).astype(np.float32)
    target=np.concatenate([row["target"] for row in rows]).astype(np.float32)
    return X,target


def make_model(name,seed):
    if name=="lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(objective="mae",n_estimators=350,learning_rate=.05,
            num_leaves=31,min_child_samples=80,subsample=.8,colsample_bytree=.9,
            reg_lambda=1.,random_state=seed,n_jobs=-1,verbosity=-1)
    if name=="catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(loss_function="MAE",iterations=350,depth=6,
            learning_rate=.05,l2_leaf_reg=6.,random_seed=seed,verbose=False,
            allow_writing_files=False,thread_count=-1)
    if name=="xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(objective="reg:absoluteerror",n_estimators=500,max_depth=6,
            learning_rate=.05,min_child_weight=10,subsample=.8,colsample_bytree=.9,
            reg_lambda=1.,random_state=seed,n_jobs=-1,tree_method="hist")
    raise ValueError(name)


def score_model(model,train_rows,val_rows,groups,X,y):
    Xfit,yfit=matrix(train_rows,groups); model.fit(Xfit,yfit)
    errors=[]
    for row in val_rows:
        Xq,_=matrix([row],groups); normalized=np.asarray(model.predict(Xq),float)
        raw=normalized*row["scale"]+row["center"]
        ep={"missing":row["missing"],"interp":row["interp"]}
        pred=ft.blend_episode_prediction(raw,ep,X[row["col"]],row["q"])
        errors.append(np.abs(pred-y[row["col"]].iloc[row["q"]].to_numpy(float)))
    return float(np.concatenate(errors).mean()),len(Xfit),Xfit.shape[1]


def splits(columns,repeats,nval):
    for repeat in range(repeats):
        seed=SEED+repeat; dev,hold=train_test_split(list(columns),test_size=.1,random_state=seed)
        val=list(np.random.RandomState(seed).choice(hold,min(nval,len(hold)),replace=False))
        yield repeat,dev,val


def evaluate(name,groups,model_name,cache,y,X,repeats,nval):
    scores=[]; started=time.time(); nrows=nfeatures=0
    for repeat,dev,val in splits(y.columns,repeats,nval):
        train_rows=[cache[c] for c in dev if cache[c] is not None]
        val_rows=[cache[c] for c in val if cache[c] is not None]
        score,nrows,nfeatures=score_model(make_model(model_name,SEED+repeat),train_rows,val_rows,groups,X,y)
        scores.append(score); print(f"{name} repeat {repeat+1}/{repeats}: {score:.3f} W",flush=True)
    return {"model":model_name,"groups":list(groups),"scores":scores,
            "mean":float(np.mean(scores)),"std":float(np.std(scores)),
            "n_train_rows":nrows,"n_features":nfeatures,"seconds":time.time()-started}


def run_benchmark(X,y,cache):
    results={"screen":{},"feature_tests":{},"forward":{},"confirmation":{}}
    for model in ("lightgbm","catboost","xgboost"):
        results["screen"][model]=evaluate(f"model_{model}",(),model,cache,y,X,3,20)
        atomic_json(RESULTS,results)
    winner=min(results["screen"],key=lambda k:results["screen"][k]["mean"])
    baseline=np.asarray(results["screen"][winner]["scores"])
    candidates=[]
    for group in FEATURE_GROUPS:
        row=evaluate(f"feature_{group}",(group,),winner,cache,y,X,3,20)
        scores=np.asarray(row["scores"]); row["delta_vs_baseline"]=float(scores.mean()-baseline.mean())
        row["wins_vs_baseline"]=int((scores<baseline).sum()); results["feature_tests"][group]=row
        if row["delta_vs_baseline"]<0 and row["wins_vs_baseline"]>=2: candidates.append(group)
        atomic_json(RESULTS,results)
    candidates.sort(key=lambda g:results["feature_tests"][g]["mean"])
    selected=[]; current=baseline
    for group in candidates:
        trial=selected+[group]; row=evaluate("forward_"+"+".join(trial),trial,winner,cache,y,X,3,20)
        scores=np.asarray(row["scores"]); row["wins_vs_previous"]=int((scores<current).sum())
        row["delta_vs_previous"]=float(scores.mean()-current.mean())
        results["forward"]["+".join(trial)]=row
        if row["delta_vs_previous"]<0 and row["wins_vs_previous"]>=2:
            selected=trial; current=scores
        atomic_json(RESULTS,results)
    results["winner_model"]=winner; results["selected_groups"]=selected
    results["confirmation"]["baseline"]=evaluate("confirm_baseline",(),winner,cache,y,X,5,100)
    results["confirmation"]["enriched"]=evaluate("confirm_enriched",selected,winner,cache,y,X,5,100)
    b=np.asarray(results["confirmation"]["baseline"]["scores"])
    e=np.asarray(results["confirmation"]["enriched"]["scores"])
    results["confirmation_summary"]={"baseline_mae_w":float(b.mean()),
        "enriched_mae_w":float(e.mean()),"delta_w":float(e.mean()-b.mean()),
        "wins":int((e<b).sum())}
    atomic_json(RESULTS,results); write_results_report(results); return results


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("stage",choices=("eda","screen","all"),default="all",nargs="?")
    args=parser.parse_args(); X,y,_=load_data()
    if args.stage in ("eda","all"):
        eda_columns,_=train_test_split(list(y.columns),test_size=.1,random_state=SEED)
        run_eda(X[eda_columns],y[eda_columns])
    if args.stage=="eda": return
    donors=[c for c in X if not str(c).startswith("holed_")]
    context=ft.build_donor_context(X,donors); cache={}
    for number,col in enumerate(y.columns,1):
        cache[col]=build_rows(col,X,y,context)
        if number%50==0: print(f"features {number}/{len(y.columns)}",flush=True)
    result=run_benchmark(X,y,cache)
    print(json.dumps(result["confirmation_summary"],indent=2),flush=True)


if __name__=="__main__": main()
