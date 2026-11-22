"""Paired TabPFN comparison: historical 96 vs validated 160 column features."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import boosted_columns_eda_bench as boosted
import tabpfn_column_individual_bench as old
import tabfm_lora_finetune as ft
from evaluate import load_data
from tabpfn_v15_features import build_v15_features

REFERENCE_FEATURES_DIR = "/home/nicolas/PycharmProjects/hackathon_enedis2"
import sys
if REFERENCE_FEATURES_DIR not in sys.path:
    sys.path.insert(0, REFERENCE_FEATURES_DIR)
import features as reference_features

FEATURE_GROUPS=("dynamics","profiles","H28_H07")
FEATURE_COUNT=162
PROGRESS_PATH=Path("tabpfn_best_features_column_progress.json")
RESULTS_PATH=Path("tabpfn_best_features_column_results.json")
REPORT_PATH=Path("tabpfn_best_features_column_results.md")


def _stack_best_features(source,q,matrix,interp,missing):
    center,scale=old.robust_location_scale(source)
    base_features=matrix[q,:old.BASE_FEATURES]
    if base_features.shape[1]<old.BASE_FEATURES:
        base_features=np.pad(base_features,((0,0),(0,old.BASE_FEATURES-base_features.shape[1])))
    profile=np.repeat(old.curve_profile(source,source.index,center,scale)[None],len(q),axis=0)
    geometry=old.point_geometry(source,q,interp.to_numpy(float),center,scale)
    baseline=np.hstack([base_features,profile,geometry]).astype(np.float32)
    definitions={
        "dynamics":("local_volatility","neighbor_dynamics","periodic_residuals"),
        "profiles":("daytype_profile","recent_profile"),
    }
    blocks=[]; values=source.to_numpy(float); neighbors=matrix[:,:40]
    for name in FEATURE_GROUPS:
        if name == "H28_H07":
            continue
        extra=build_v15_features(values,source.index,neighbors,missing,definitions[name])[q]
        if name=="profiles":
            slot=np.repeat(boosted.slot48_profile(source,center,scale)[None],len(q),axis=0)
            extra=np.hstack([extra,slot])
        blocks.append(extra.astype(np.float32))
    ref_values=np.asarray(neighbors, float)
    corr=np.nan_to_num(np.corrcoef(np.nan_to_num(values, nan=np.nanmean(values)), ref_values.T)[0, 1:])
    h_raw=reference_features.candidate_feature_blocks_v2(
        values, ref_values, np.arange(ref_values.shape[1]), corr,
        pd.DatetimeIndex(source.index))
    h_blocks={key.split("_", 1)[0]: value for key, value in h_raw.items()}
    h_extra=np.hstack([h_blocks["H28"][q], h_blocks["H07"][q]])
    blocks.append(np.nan_to_num(h_extra, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
    features=np.hstack([baseline]+blocks).astype(np.float32)
    if features.shape!=(len(q),FEATURE_COUNT):
        raise RuntimeError(f"Expected {(len(q),FEATURE_COUNT)}, got {features.shape}")
    if not np.isfinite(features).all(): raise RuntimeError("Non-finite best features")
    return features,center,scale


def build_curve_rows(col,X,y,context):
    source=X[col]; q=ft.real_gap_indices(source,y[col])
    if not len(q): return None
    built=ft.feature_matrix(col,source,*context,source.index)
    if built is None: return None
    matrix,_,_,interp,missing,_=built
    features,center,scale=_stack_best_features(source,q,matrix,interp,missing)
    target=((y[col].iloc[q].to_numpy(float)-center)/scale).astype(np.float32)
    return {"col":col,"X":features,"y":target,"q":q,"center":center,"scale":scale,
            "interp":interp.to_numpy(float),"missing":missing}


def build_query_rows(col,Xtest,context):
    """Build the exact same 160 features without requiring unavailable truth."""
    source=Xtest[col]; q=np.flatnonzero(source.isna().to_numpy())
    if not len(q): return None
    built=ft.feature_matrix(col,source,*context,source.index)
    if built is None: return None
    matrix,_,_,interp,missing,_=built
    features,center,scale=_stack_best_features(source,q,matrix,interp,missing)
    return {"col":col,"X":features,"q":q,"center":center,"scale":scale,
            "interp":interp.to_numpy(float),"missing":missing}


def signature(args):
    return {"schema_version":1,"repeats":args.repeats,"n_val_cols":args.n_val_cols,
            "n_train_curves":args.n_train_curves,"max_train_rows":args.max_train_rows,
            "feature_groups":list(FEATURE_GROUPS),"feature_count":FEATURE_COUNT,
            "comparison":"paired_to_tabpfn_column_individual_96"}


def save_json(path,payload):
    tmp=Path(str(path)+".tmp"); tmp.write_text(json.dumps(payload,indent=2)); os.replace(tmp,path)


def load_progress(args):
    expected=signature(args)
    if args.resume and PROGRESS_PATH.exists():
        progress=json.loads(PROGRESS_PATH.read_text())
        if progress.get("signature")!=expected:
            raise RuntimeError("Progress checkpoint uses different parameters")
        return progress
    return {"signature":expected,"repeats":[]}


def old_results():
    if not old.RESULTS_PATH.exists(): raise RuntimeError(f"Missing {old.RESULTS_PATH}")
    payload=json.loads(old.RESULTS_PATH.read_text())
    by_repeat={int(row["repeat"]):row for row in payload["repeats"]}
    return payload,by_repeat


def validate_pairing(repeat,train_cols,val_cols,reference,args):
    if reference["seed"]!=old.SEED+repeat: raise RuntimeError("Historical seed mismatch")
    if reference["n_train_curves"]!=args.n_train_curves: raise RuntimeError("Training curve count mismatch")
    # A historical repeat can contain one curve without finite scoring holes.
    if abs(reference["n_val_curves"]-len(val_cols))>1: raise RuntimeError("Validation curve count mismatch")
    if len(train_cols)!=args.n_train_curves or len(val_cols)!=args.n_val_cols:
        raise RuntimeError("Split cardinality mismatch")


def summarize(progress,history):
    rows=sorted(progress["repeats"],key=lambda x:x["repeat"])
    new=np.asarray([r["mae_w"] for r in rows],float)
    prior=np.asarray([history[r["repeat"]]["mae_w"] for r in rows],float)
    return {"old_96_scores_mae_w":prior.tolist(),"new_160_scores_mae_w":new.tolist(),
            "deltas_new_minus_old_w":(new-prior).tolist(),
            "old_mean_mae_w":float(prior.mean()),"new_mean_mae_w":float(new.mean()),
            "delta_mean_w":float(new.mean()-prior.mean()),"wins":int((new<prior).sum()),
            "new_std_mae_w":float(new.std()),
            "accepted":bool(len(new)==5 and new.mean()<prior.mean() and (new<prior).sum()>=3)}


def write_report(result):
    s=result["summary"]
    lines=["# TabPFN colonne-centré : 96 vs 160 features","",
           f"- Ancienne version : {s['old_mean_mae_w']:.3f} W",
           f"- Nouvelle version : {s['new_mean_mae_w']:.3f} W",
           f"- Delta nouvelle - ancienne : {s['delta_mean_w']:+.3f} W",
           f"- Repeats gagnés : {s['wins']}/5",
           f"- Décision : **{'160 features' if s['accepted'] else 'conserver 96 features'}**","",
           "## Résultats appariés",""]
    for i,(a,b,d) in enumerate(zip(s["old_96_scores_mae_w"],s["new_160_scores_mae_w"],s["deltas_new_minus_old_w"]),1):
        lines.append(f"- Repeat {i}: {a:.3f} -> {b:.3f} W ({d:+.3f} W)")
    REPORT_PATH.write_text("\n".join(lines)+"\n")


def main():
    p=argparse.ArgumentParser(); p.add_argument("--resume",action="store_true")
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--repeats",type=int,default=old.N_REPEATS)
    p.add_argument("--n-val-cols",type=int,default=old.N_VAL_COLS)
    p.add_argument("--n-train-curves",type=int,default=old.N_TRAIN_CURVES)
    p.add_argument("--max-train-rows",type=int,default=old.MAX_TRAIN_ROWS)
    args=p.parse_args()
    if not args.dry_run and not os.environ.get("TABPFN_API_KEY"):
        raise SystemExit("TABPFN_API_KEY manquant")
    historical,history=old_results()
    if historical["signature"]!={"schema_version":1,"repeats":5,"n_val_cols":10,
            "n_train_curves":100,"max_train_rows":8000,"base_features":72,"max_features":96}:
        raise RuntimeError("Unexpected historical benchmark signature")
    X,y,_=load_data(); donors=[c for c in X if not str(c).startswith("holed_")]
    context=ft.build_donor_context(X,donors); cache={}; progress=load_progress(args)
    completed={r["repeat"] for r in progress["repeats"]}; stop=1 if args.dry_run else args.repeats
    if not args.dry_run:
        import tabpfn_client
        from tabpfn_client import TabPFNRegressor
        tabpfn_client.set_access_token(os.environ["TABPFN_API_KEY"])
    for repeat in range(stop):
        if repeat in completed: continue
        started=time.time(); train_cols,val_cols=old.split_columns(
            y.columns,repeat,args.n_train_curves,args.n_val_cols)
        validate_pairing(repeat,train_cols,val_cols,history[repeat],args)
        for number,col in enumerate(train_cols+val_cols,1):
            if col not in cache: cache[col]=build_curve_rows(col,X,y,context)
            if number%25==0: print(f"repeat {repeat+1}: features {number}/{len(train_cols)+len(val_cols)}",flush=True)
        train_rows=[cache[c] for c in train_cols if cache[c] is not None]
        val_rows=[cache[c] for c in val_cols if cache[c] is not None]
        if set(history[repeat]["per_curve_mae_w"])!={r["col"] for r in val_rows}:
            raise RuntimeError("Validation curves differ from historical benchmark")
        Xfit,yfit=old.balanced_context(train_rows,args.max_train_rows,old.SEED+repeat)
        Xquery=np.vstack([r["X"] for r in val_rows]).astype(np.float32)
        print(f"repeat {repeat+1}/{args.repeats}: fit={Xfit.shape}, query={Xquery.shape}",flush=True)
        if args.dry_run:
            assert Xfit.shape==(args.max_train_rows,FEATURE_COUNT) and Xquery.shape[1]==FEATURE_COUNT
            print("dry-run OK: mêmes splits, 160 features, aucun appel TabPFN",flush=True); return
        error=None
        for attempt in range(old.MAX_RETRIES):
            try:
                reg=TabPFNRegressor(random_state=old.SEED+repeat); reg.fit(Xfit,yfit)
                mae,per_curve,n_points=old.predict_and_score(reg,val_rows,X,y); break
            except Exception as exc:
                error=f"{type(exc).__name__}: {exc}"
                if attempt+1==old.MAX_RETRIES: raise
                time.sleep(2*(attempt+1))
        row={"repeat":repeat,"seed":old.SEED+repeat,"mae_w":mae,
             "old_96_mae_w":history[repeat]["mae_w"],"delta_w":mae-history[repeat]["mae_w"],
             "n_points":n_points,"n_train_rows":len(Xfit),"n_train_curves":len(train_rows),
             "n_val_curves":len(val_rows),"per_curve_mae_w":per_curve,
             "seconds":time.time()-started,"last_error":error}
        progress["repeats"].append(row); save_json(PROGRESS_PATH,progress)
        print(f"repeat {repeat+1}: old={row['old_96_mae_w']:.3f}, new={mae:.3f}, delta={row['delta_w']:+.3f} W",flush=True)
    summary=summarize(progress,history); result={**progress,"summary":summary}
    save_json(RESULTS_PATH,result); write_report(result); print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__": main()
