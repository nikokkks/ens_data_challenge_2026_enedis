"""Short TabPFN screen of 40 leak-free feature families, columns as individuals."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from evaluate import load_data
import tabfm_lora_finetune as ft
import tabpfn_column_individual_bench as old
import tabpfn_v14_ablation as v14
from tabpfn_v15_features import FEATURE_GROUPS as V15_GROUPS, build_v15_features
from boosted_columns_feature_search import FAMILIES as NEW_GROUPS, build_25_features

SCREEN_SEED=44
CONTROL_SEEDS=(43,45)
K_NEIGHBORS=40
MIN_GREEDY_GAIN_W=.1
PROGRESS_PATH=Path("tabpfn_column_feature_screen_progress.json")
RESULTS_PATH=Path("tabpfn_column_feature_screen_results.json")
REPORT_PATH=Path("tabpfn_column_feature_screen_results.md")


def save_json(path,payload):
    tmp=Path(str(path)+".tmp"); tmp.write_text(json.dumps(payload,indent=2)); os.replace(tmp,path)


def measure_extra_width(X,y,context):
    for col in y.columns:
        built=v14.build_v14_features(col,X[col],*context,X.index,ft.V14_FEATURE_OPTS)
        if built is not None: return int(built[6].shape[1])
    raise RuntimeError("Unable to measure V14 extra width")


def aligned_base(matrix,extra_width):
    n_neighbors=matrix.shape[1]-extra_width
    if n_neighbors<0 or n_neighbors>K_NEIGHBORS:
        raise RuntimeError(f"Invalid neighbor width {n_neighbors}")
    neighbors=matrix[:,:n_neighbors]; extra=matrix[:,n_neighbors:]
    padded=np.pad(neighbors,((0,0),(0,K_NEIGHBORS-n_neighbors)))
    return np.hstack([extra,padded]).astype(np.float32),padded.astype(np.float32)


def build_row(col,X,y,context,extra_width):
    source=X[col]; q=ft.real_gap_indices(source,y[col])
    if not len(q): return None
    built=ft.feature_matrix(col,source,*context,source.index)
    if built is None: return None
    matrix,_,_,interp,missing,_=built; aligned,neighbors=aligned_base(matrix,extra_width)
    center,scale=old.robust_location_scale(source)
    profile=np.repeat(old.curve_profile(source,source.index,center,scale)[None],len(q),axis=0)
    geometry=old.point_geometry(source,q,interp.to_numpy(float),center,scale)
    blocks={"baseline":np.hstack([aligned[q],profile,geometry]).astype(np.float32)}
    values=source.to_numpy(float)
    for name in V15_GROUPS:
        blocks["v15::"+name]=build_v15_features(values,source.index,neighbors,missing,(name,))[q].astype(np.float32)
    new=build_25_features(source,source.index,neighbors,center,scale)
    blocks.update({"new::"+name:value[q].astype(np.float32) for name,value in new.items()})
    if len(blocks)!=1+len(V15_GROUPS)+len(NEW_GROUPS): raise RuntimeError("Feature family count mismatch")
    if any(x.shape[0]!=len(q) or x.shape[1]<1 or not np.isfinite(x).all() for x in blocks.values()):
        raise RuntimeError(f"Invalid feature block for {col}")
    target=((y[col].iloc[q].to_numpy(float)-center)/scale).astype(np.float32)
    return {"col":col,"blocks":blocks,"y":target,"q":q,"center":center,"scale":scale,
            "interp":interp.to_numpy(float),"missing":missing}


def materialize(row,groups):
    return {**row,"X":np.hstack([row["blocks"]["baseline"]]+[row["blocks"][g] for g in groups]).astype(np.float32)}


def split_for_seed(columns,seed):
    return old.split_columns(columns,seed-old.SEED,old.N_TRAIN_CURVES,old.N_VAL_COLS)


def evaluate(seed,groups,cache,X,y,regressor_cls,max_train_rows):
    train_cols,val_cols=split_for_seed(y.columns,seed)
    train=[materialize(cache[c],groups) for c in train_cols if cache[c] is not None]
    valid=[materialize(cache[c],groups) for c in val_cols if cache[c] is not None]
    Xfit,yfit=old.balanced_context(train,max_train_rows,seed)
    query=np.vstack([r["X"] for r in valid]); print(f"seed {seed}: groups={groups}, fit={Xfit.shape}, query={query.shape}",flush=True)
    started=time.time(); last_error=None
    for attempt in range(old.MAX_RETRIES):
        try:
            reg=regressor_cls(random_state=seed); reg.fit(Xfit,yfit)
            mae,per_curve,n_points=old.predict_and_score(reg,valid,X,y); break
        except Exception as exc:
            last_error=f"{type(exc).__name__}: {exc}"
            if attempt+1==old.MAX_RETRIES: raise
            time.sleep(2*(attempt+1))
    return {"seed":seed,"groups":list(groups),"mae_w":mae,"n_features":Xfit.shape[1],
            "n_train_rows":len(Xfit),"n_points":n_points,"per_curve_mae_w":per_curve,
            "seconds":time.time()-started,"last_error":last_error}


def signature(extra_width,args):
    return {"schema_version":1,"screen_seed":SCREEN_SEED,"control_seeds":list(CONTROL_SEEDS),
            "extra_width":extra_width,"k_neighbors":K_NEIGHBORS,"baseline_features":extra_width+64,
            "v15_groups":list(V15_GROUPS),"new_groups":list(NEW_GROUPS),
            "n_train_curves":old.N_TRAIN_CURVES,"n_val_cols":old.N_VAL_COLS,
            "max_train_rows":args.max_train_rows,"min_greedy_gain_w":MIN_GREEDY_GAIN_W}


def load_progress(expected,resume):
    if resume and PROGRESS_PATH.exists():
        p=json.loads(PROGRESS_PATH.read_text())
        if p.get("signature")!=expected: raise RuntimeError("Progress signature mismatch")
        return p
    return {"signature":expected,"screen":{},"forward":{},"controls":{}}


def result_key(seed,groups): return f"{seed}::"+("+".join(groups) if groups else "baseline")


def write_report(result):
    c=result["confirmation"]
    lines=["# Screening TabPFN colonne-centré","",f"Familles testées: {len(result['individual'])}",
           f"Baseline corrigée seed 44: {result['baseline']['mae_w']:.3f} W",
           "Sélection: **"+(", ".join(result["selected"]) or "aucune")+"**","",
           f"Baseline 3 seeds: {c['baseline_mean_w']:.3f} W",
           f"Enrichi 3 seeds: {c['enriched_mean_w']:.3f} W",
           f"Delta: {c['delta_w']:+.3f} W; victoires: {c['wins']}/3",
           f"Décision: **{'enrichi' if c['accepted'] else 'baseline corrigée'}**","","## Classement seed 44",""]
    for name,row in sorted(result["individual"].items(),key=lambda z:z[1]["mae_w"]):
        lines.append(f"- {name}: {row['mae_w']:.3f} W ({row['delta_w']:+.3f} W)")
    REPORT_PATH.write_text("\n".join(lines)+"\n")


def main():
    p=argparse.ArgumentParser(); p.add_argument("--resume",action="store_true")
    p.add_argument("--dry-run",action="store_true"); p.add_argument("--max-train-rows",type=int,default=old.MAX_TRAIN_ROWS)
    args=p.parse_args()
    if not args.dry_run and not os.environ.get("TABPFN_API_KEY"): raise SystemExit("TABPFN_API_KEY manquant")
    X,y,_=load_data(); donors=[c for c in X if not str(c).startswith("holed_")]
    context=ft.build_donor_context(X,donors); extra_width=measure_extra_width(X,y,context)
    needed=set()
    for seed in (SCREEN_SEED,)+CONTROL_SEEDS:
        a,b=split_for_seed(y.columns,seed); needed.update(a); needed.update(b)
    if args.dry_run:
        a,b=split_for_seed(y.columns,SCREEN_SEED); needed=set(a[:2]+b[:2])
    cache={}
    for number,col in enumerate(sorted(needed),1):
        cache[col]=build_row(col,X,y,context,extra_width)
        if number%25==0: print(f"features {number}/{len(needed)}",flush=True)
    all_groups=tuple("v15::"+x for x in V15_GROUPS)+tuple("new::"+x for x in NEW_GROUPS)
    if args.dry_run:
        row=next(r for r in cache.values() if r is not None); baseline=materialize(row,())
        every=materialize(row,all_groups)
        print(f"dry-run OK: families={len(all_groups)}, baseline={baseline['X'].shape}, all={every['X'].shape}, aucun appel TabPFN",flush=True); return
    import tabpfn_client
    from tabpfn_client import TabPFNRegressor
    tabpfn_client.set_access_token(os.environ["TABPFN_API_KEY"])
    expected=signature(extra_width,args); progress=load_progress(expected,args.resume)
    if "baseline" not in progress["screen"]:
        progress["screen"]["baseline"]=evaluate(SCREEN_SEED,(),cache,X,y,TabPFNRegressor,args.max_train_rows); save_json(PROGRESS_PATH,progress)
    baseline=progress["screen"]["baseline"]; b=baseline["mae_w"]
    for group in all_groups:
        if group in progress["screen"]: continue
        row=evaluate(SCREEN_SEED,(group,),cache,X,y,TabPFNRegressor,args.max_train_rows)
        row["delta_w"]=row["mae_w"]-b; progress["screen"][group]=row; save_json(PROGRESS_PATH,progress)
        print(f"{group}: {row['mae_w']:.3f} W ({row['delta_w']:+.3f})",flush=True)
    candidates=[g for g in all_groups if progress["screen"][g]["delta_w"]<=-MIN_GREEDY_GAIN_W]
    candidates.sort(key=lambda g:progress["screen"][g]["mae_w"]); selected=[]; current=b
    for group in candidates:
        trial=selected+[group]; key="+".join(trial)
        if key not in progress["forward"]:
            progress["forward"][key]=evaluate(SCREEN_SEED,trial,cache,X,y,TabPFNRegressor,args.max_train_rows); save_json(PROGRESS_PATH,progress)
        row=progress["forward"][key]; row["delta_previous_w"]=row["mae_w"]-current
        if row["mae_w"]<=current-MIN_GREEDY_GAIN_W: selected=trial; current=row["mae_w"]
        progress["selected"]=selected; save_json(PROGRESS_PATH,progress)
    baseline_rows={SCREEN_SEED:baseline}; enriched_rows={SCREEN_SEED:(progress["forward"].get("+".join(selected),baseline) if selected else baseline)}
    for seed in CONTROL_SEEDS:
        for label,groups,target in (("baseline",(),baseline_rows),("enriched",selected,enriched_rows)):
            key=result_key(seed,groups)
            if key not in progress["controls"]:
                progress["controls"][key]=evaluate(seed,groups,cache,X,y,TabPFNRegressor,args.max_train_rows); save_json(PROGRESS_PATH,progress)
            target[seed]=progress["controls"][key]
    bs=np.array([baseline_rows[s]["mae_w"] for s in (SCREEN_SEED,)+CONTROL_SEEDS])
    es=np.array([enriched_rows[s]["mae_w"] for s in (SCREEN_SEED,)+CONTROL_SEEDS])
    confirmation={"seeds":list((SCREEN_SEED,)+CONTROL_SEEDS),"baseline_scores":bs.tolist(),"enriched_scores":es.tolist(),
                  "baseline_mean_w":float(bs.mean()),"enriched_mean_w":float(es.mean()),"delta_w":float(es.mean()-bs.mean()),
                  "wins":int((es<bs).sum()),"accepted":bool(es.mean()<bs.mean() and (es<bs).sum()>=2)}
    result={"signature":expected,"baseline":baseline,"individual":{g:progress["screen"][g] for g in all_groups},
            "selected":selected,"forward":progress["forward"],"confirmation":confirmation,
            "recommended_groups":selected if confirmation["accepted"] else []}
    save_json(RESULTS_PATH,result); write_report(result); print(json.dumps(confirmation,indent=2),flush=True)


if __name__=="__main__": main()
