"""Column-centred TabPFN benchmark for the Enedis imputation challenge.

One supervised row represents a (curve, missing timestamp) pair. Curve-level
profiles are repeated on its rows, so the original dataframe columns remain
the statistical individuals while TabPFN keeps its required scalar target.
V14/V15 source files are imported read-only and never modified.
"""
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


SEED = 42
N_REPEATS = 5
N_VAL_COLS = 10
N_TRAIN_CURVES = 100
MAX_TRAIN_ROWS = 8_000
BASE_FEATURES = 72
MAX_FEATURES = 96
MAX_RETRIES = 3
PROGRESS_PATH = Path("tabpfn_column_individual_progress.json")
RESULTS_PATH = Path("tabpfn_column_individual_results.json")
V14_RESULTS_PATH = Path("tabpfn_v14_confirm_results.json")
SUBMISSION_PROGRESS_PATH = Path("submission_tabpfn_column_individual_progress.json")
DEFAULT_SUBMISSION_PATH = Path("submission.csv")
SAMPLE_PATH = Path("data/new_output_sample.csv")


def robust_location_scale(s: pd.Series) -> tuple[float, float]:
    values=s.dropna().to_numpy(float)
    if not len(values): return 0.,1.
    center=float(np.median(values))
    scale=max(float(np.median(np.abs(values-center))*1.4826),
              .02*(abs(center)+1.),1.)
    return center,scale


def curve_profile(s: pd.Series, index: pd.Index, center: float, scale: float) -> np.ndarray:
    """Twenty leak-free descriptors computed from observed/interpolated X."""
    filled=s.interpolate(method="linear",limit_direction="both")
    values=filled.fillna(center).to_numpy(float)
    normalized=np.clip((values-center)/scale,-20.,20.)
    quantiles=np.quantile(normalized,[.05,.25,.5,.75,.95])
    dt=pd.DatetimeIndex(index)
    hour_bin=(dt.hour//3).to_numpy()
    day_profile=np.array([np.median(normalized[hour_bin==h]) for h in range(8)])
    dow=dt.dayofweek.to_numpy()
    week_profile=np.array([np.median(normalized[dow==d]) for d in range(7)])
    out=np.r_[quantiles,day_profile,week_profile]
    assert out.shape==(20,)
    return np.nan_to_num(out,nan=0.,posinf=0.,neginf=0.).astype(np.float32)


def point_geometry(s: pd.Series, q: np.ndarray, interp: np.ndarray,
                   center: float, scale: float) -> np.ndarray:
    missing=s.isna().to_numpy()
    runs=feb.run_lengths_for(missing)
    edges=feb.edge_distance_for(missing)
    rel=np.zeros(len(q),dtype=float)
    for j,i in enumerate(q):
        left=i
        while left>0 and missing[left-1]: left-=1
        rel[j]=(i-left+.5)/max(runs[i],1)
    return np.column_stack([
        np.clip((interp[q]-center)/scale,-20.,20.),
        np.log1p(runs[q]),np.log1p(edges[q]),rel,
    ]).astype(np.float32)


def build_curve_rows(col: str, X: pd.DataFrame, y: pd.DataFrame, context) -> dict | None:
    source=X[col]; q=ft.real_gap_indices(source,y[col])
    if not len(q): return None
    built=ft.feature_matrix(col,source,*context,source.index)
    if built is None: return None
    matrix,_,_,interp,missing,_=built
    center,scale=robust_location_scale(source)
    base=matrix[q,:BASE_FEATURES]
    if base.shape[1]<BASE_FEATURES:
        base=np.pad(base,((0,0),(0,BASE_FEATURES-base.shape[1])))
    profile=np.repeat(curve_profile(source,source.index,center,scale)[None],len(q),axis=0)
    geometry=point_geometry(source,q,interp.to_numpy(float),center,scale)
    features=np.hstack([base,profile,geometry]).astype(np.float32)
    assert features.shape==(len(q),MAX_FEATURES)
    target=((y[col].iloc[q].to_numpy(float)-center)/scale).astype(np.float32)
    return {"col":col,"X":features,"y":target,"q":q,"center":center,
            "scale":scale,"interp":interp.to_numpy(float),"missing":missing}


def build_query_rows(col: str, X: pd.DataFrame, context) -> dict | None:
    """Build submission rows without requiring unavailable test targets."""
    source=X[col]; q=np.flatnonzero(source.isna().to_numpy())
    if not len(q): return None
    built=ft.feature_matrix(col,source,*context,source.index)
    if built is None: return None
    matrix,_,_,interp,missing,_=built
    center,scale=robust_location_scale(source)
    base=matrix[q,:BASE_FEATURES]
    if base.shape[1]<BASE_FEATURES:
        base=np.pad(base,((0,0),(0,BASE_FEATURES-base.shape[1])))
    profile=np.repeat(curve_profile(source,source.index,center,scale)[None],len(q),axis=0)
    geometry=point_geometry(source,q,interp.to_numpy(float),center,scale)
    features=np.hstack([base,profile,geometry]).astype(np.float32)
    assert features.shape==(len(q),MAX_FEATURES)
    return {"col":col,"X":features,"q":q,"center":center,"scale":scale,
            "interp":interp.to_numpy(float),"missing":missing}


def balanced_context(rows: list[dict], max_rows: int, seed: int) -> tuple[np.ndarray,np.ndarray]:
    """Cap context while giving every selected curve an equal first allocation."""
    if not rows: raise RuntimeError("No finite training holes")
    rng=np.random.RandomState(seed); per=max(1,int(np.ceil(max_rows/len(rows))))
    chosen=[]; remaining=[]
    for row in rows:
        idx=rng.permutation(len(row["y"])); take=min(len(idx),per)
        chosen.extend((row,int(i)) for i in idx[:take])
        remaining.extend((row,int(i)) for i in idx[take:])
    if len(chosen)>max_rows:
        keep=np.sort(rng.choice(len(chosen),max_rows,replace=False))
        chosen=[chosen[i] for i in keep]
    elif len(chosen)<max_rows and remaining:
        take=min(max_rows-len(chosen),len(remaining))
        keep=rng.choice(len(remaining),take,replace=False)
        chosen.extend(remaining[i] for i in keep)
    Xfit=np.vstack([row["X"][i] for row,i in chosen]).astype(np.float32)
    yfit=np.asarray([row["y"][i] for row,i in chosen],dtype=np.float32)
    return Xfit,yfit


def split_columns(columns, repeat: int, n_train_curves: int, n_val_cols: int):
    seed=SEED+repeat
    dev,valfull=train_test_split(list(columns),test_size=.1,random_state=seed)
    rng=np.random.RandomState(seed)
    val=list(rng.choice(valfull,min(n_val_cols,len(valfull)),replace=False))
    train=list(rng.choice(dev,min(n_train_curves,len(dev)),replace=False))
    return train,val


def predict_and_score(reg, val_rows: list[dict], X: pd.DataFrame, y: pd.DataFrame):
    Xquery=np.vstack([row["X"] for row in val_rows]).astype(np.float32)
    normalized=np.asarray(reg.predict(Xquery,output_type="median"),float)
    errors=[]; per_curve={}; offset=0
    for row in val_rows:
        n=len(row["q"]); q=row["q"]
        raw=normalized[offset:offset+n]*row["scale"]+row["center"]
        ep={"missing":row["missing"],"interp":row["interp"]}
        pred=ft.blend_episode_prediction(raw,ep,X[row["col"]],q)
        err=np.abs(pred-y[row["col"]].iloc[q].to_numpy(float))
        errors.append(err); per_curve[row["col"]]=float(err.mean()); offset+=n
    if offset!=len(normalized): raise RuntimeError("Prediction/row alignment mismatch")
    return float(np.concatenate(errors).mean()),per_curve,int(offset)


def load_progress(args) -> dict:
    signature={"schema_version":1,"repeats":args.repeats,"n_val_cols":args.n_val_cols,
               "n_train_curves":args.n_train_curves,"max_train_rows":args.max_train_rows,
               "base_features":BASE_FEATURES,"max_features":MAX_FEATURES}
    if PROGRESS_PATH.exists() and args.resume:
        progress=json.loads(PROGRESS_PATH.read_text())
        if progress.get("signature")!=signature:
            raise RuntimeError("Existing progress uses different benchmark parameters")
        return progress
    return {"signature":signature,"repeats":[]}


def save_json(path: Path, payload):
    tmp=Path(str(path)+".tmp"); tmp.write_text(json.dumps(payload,indent=2)); os.replace(tmp,path)


def v14_reference(repeat: int):
    if not V14_RESULTS_PATH.exists(): return None
    scores=json.loads(V14_RESULTS_PATH.read_text()).get("scores",[])
    return float(scores[repeat]) if repeat<len(scores) else None


def submission_signature(args) -> dict:
    return {"schema_version":1,"seeds":list(range(SEED,SEED+N_REPEATS)),
            "n_train_curves":args.n_train_curves,"max_train_rows":args.max_train_rows,
            "predict_batch_rows":args.predict_batch_rows,
            "base_features":BASE_FEATURES,"max_features":MAX_FEATURES,
            "strategy":"pure_column_individual_median_ensemble"}


def load_submission_progress(args) -> dict:
    signature=submission_signature(args)
    if args.resume and SUBMISSION_PROGRESS_PATH.exists():
        progress=json.loads(SUBMISSION_PROGRESS_PATH.read_text())
        if progress.get("signature")!=signature:
            raise RuntimeError("Existing submission progress uses different parameters")
        return progress
    return {"signature":signature,"completed":{}}


def member_path(seed: int) -> Path:
    return Path(f"submission_tabpfn_column_individual_seed{seed}.csv")


def validate_member(frame: pd.DataFrame, Xtest: pd.DataFrame, holes: list[str]):
    if frame.shape!=(len(Xtest),len(holes)) or list(frame.columns)!=holes:
        raise RuntimeError("Invalid member checkpoint shape or columns")
    if not frame.index.equals(Xtest.index):
        raise RuntimeError("Invalid member checkpoint index")
    if not np.isfinite(frame.to_numpy(float)).all():
        raise RuntimeError("Member checkpoint contains NaN or infinity")
    known=Xtest[holes].notna().to_numpy()
    if not np.array_equal(frame.to_numpy()[known],Xtest[holes].to_numpy()[known]):
        raise RuntimeError("Member checkpoint changed observed X_test values")


def predict_submission_member(reg, query_rows: list[dict], Xtest: pd.DataFrame,
                              holes: list[str], batch_rows: int) -> pd.DataFrame:
    Xquery=np.vstack([row["X"] for row in query_rows]).astype(np.float32)
    pieces=[]
    for start in range(0,len(Xquery),batch_rows):
        stop=min(start+batch_rows,len(Xquery))
        pieces.append(np.asarray(reg.predict(
            Xquery[start:stop],output_type="median"),float))
        print(f"prediction {stop}/{len(Xquery)}",flush=True)
    normalized=np.concatenate(pieces)
    member=Xtest[holes].copy(); offset=0
    for row in query_rows:
        n=len(row["q"]); q=row["q"]
        raw=normalized[offset:offset+n]*row["scale"]+row["center"]
        ep={"missing":row["missing"],"interp":row["interp"]}
        pred=ft.blend_episode_prediction(raw,ep,Xtest[row["col"]],q)
        member.loc[member.index[q],row["col"]]=np.clip(pred,0,None)
        offset+=n
    if offset!=len(normalized): raise RuntimeError("Submission prediction alignment mismatch")
    validate_member(member,Xtest,holes)
    return member


def save_frame_atomic(frame: pd.DataFrame, path: Path):
    tmp=Path(str(path)+".tmp"); frame.to_csv(tmp); os.replace(tmp,path)


def run_submission(args):
    if not args.dry_run and not os.environ.get("TABPFN_API_KEY"):
        raise SystemExit("TABPFN_API_KEY manquant")
    if args.submit_limit is not None and not args.dry_run:
        raise SystemExit("--submit-limit is reserved for --dry-run")
    print("AVERTISSEMENT: colonne-centré=68.17 W, V14=58.21 W sur les 5 repeats.",flush=True)
    print("Chargement des données...",flush=True)
    X,y,Xtest=load_data()
    if not X.index.equals(Xtest.index):
        raise RuntimeError("X_train and X_test timestamps differ")
    holes=list(pd.read_csv(SAMPLE_PATH,nrows=0,index_col=0).columns)
    if any(col not in Xtest for col in holes): raise RuntimeError("Sample columns absent from X_test")
    donors=[c for c in X if not str(c).startswith("holed_")]
    context=ft.build_donor_context(X,donors)
    limit=(args.submit_limit if args.submit_limit is not None
           else (2 if args.dry_run else len(holes)))
    selected_holes=holes[:limit]
    query_rows=[]
    for number,col in enumerate(selected_holes,1):
        row=build_query_rows(col,Xtest,context)
        if row is not None: query_rows.append(row)
        if number%25==0: print(f"features test {number}/{len(selected_holes)}",flush=True)
    if not query_rows: raise RuntimeError("No test holes found")

    train_cache={}; first_train,_=split_columns(y.columns,0,args.n_train_curves,N_VAL_COLS)
    dry_train=first_train[:min(2,len(first_train))] if args.dry_run else first_train
    for col in dry_train: train_cache[col]=build_curve_rows(col,X,y,context)
    dry_rows=[train_cache[c] for c in dry_train if train_cache[c] is not None]
    Xfit,yfit=balanced_context(dry_rows,min(args.max_train_rows,100) if args.dry_run
                              else args.max_train_rows,SEED)
    if args.dry_run:
        Xquery=np.vstack([row["X"] for row in query_rows])
        print(f"submit dry-run OK: fit={Xfit.shape}, query={Xquery.shape}, "
              f"courbes={len(query_rows)}, aucun appel TabPFN",flush=True)
        return

    import tabpfn_client
    from tabpfn_client import TabPFNRegressor
    tabpfn_client.set_access_token(os.environ["TABPFN_API_KEY"])
    progress=load_submission_progress(args)
    for repeat in range(N_REPEATS):
        seed=SEED+repeat; path=member_path(seed)
        if str(seed) in progress["completed"] and path.exists():
            frame=pd.read_csv(path,index_col=0)
            validate_member(frame,Xtest,holes); print(f"seed {seed}: reprise",flush=True); continue
        train_cols,_=split_columns(y.columns,repeat,args.n_train_curves,N_VAL_COLS)
        for number,col in enumerate(train_cols,1):
            if col not in train_cache: train_cache[col]=build_curve_rows(col,X,y,context)
            if number%25==0: print(f"seed {seed}: features train {number}/{len(train_cols)}",flush=True)
        rows=[train_cache[c] for c in train_cols if train_cache[c] is not None]
        Xfit,yfit=balanced_context(rows,args.max_train_rows,seed)
        print(f"seed {seed}: fit {Xfit.shape}",flush=True)
        for attempt in range(MAX_RETRIES):
            try:
                reg=TabPFNRegressor(random_state=seed); reg.fit(Xfit,yfit)
                member=predict_submission_member(
                    reg,query_rows,Xtest,holes,args.predict_batch_rows)
                break
            except Exception as exc:
                if attempt+1==MAX_RETRIES: raise
                print(f"seed {seed}: tentative {attempt+1} échouée: {exc}",flush=True)
                time.sleep(2*(attempt+1))
        save_frame_atomic(member,path)
        progress["completed"][str(seed)]={"path":str(path),"n_train_rows":len(Xfit)}
        save_json(SUBMISSION_PROGRESS_PATH,progress)

    members=[]
    for seed in range(SEED,SEED+N_REPEATS):
        frame=pd.read_csv(member_path(seed),index_col=0)
        validate_member(frame,Xtest,holes); members.append(frame.to_numpy(float))
    median=np.median(np.stack(members),axis=0)
    final=Xtest[holes].copy(); missing=final.isna().to_numpy()
    values=final.to_numpy(float); values[missing]=median[missing]
    final=pd.DataFrame(values,index=Xtest.index,columns=holes)
    final.index.name="Horodate"
    sample=pd.read_csv(SAMPLE_PATH,index_col=0)
    if final.shape!=sample.shape or list(final.columns)!=list(sample.columns):
        raise RuntimeError("Final submission does not match sample schema")
    if not final.index.equals(sample.index): raise RuntimeError("Final submission index mismatch")
    validate_member(final,Xtest,holes)
    save_frame_atomic(final,args.output)
    print(f"Soumission écrite: {args.output} ({final.shape[0]} x {final.shape[1]})",flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--submit",action="store_true")
    p.add_argument("--output",type=Path,default=DEFAULT_SUBMISSION_PATH)
    p.add_argument("--predict-batch-rows",type=int,default=5_000)
    p.add_argument("--submit-limit",type=int,default=None,
                   help="limit test curves for submission dry-runs")
    p.add_argument("--repeats",type=int,default=N_REPEATS)
    p.add_argument("--n-val-cols",type=int,default=N_VAL_COLS)
    p.add_argument("--n-train-curves",type=int,default=N_TRAIN_CURVES)
    p.add_argument("--max-train-rows",type=int,default=MAX_TRAIN_ROWS)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--dry-run",action="store_true",
                   help="build and validate one repeat without calling TabPFN")
    args=p.parse_args()
    if args.submit: return run_submission(args)
    if not args.dry_run and not os.environ.get("TABPFN_API_KEY"):
        raise SystemExit("TABPFN_API_KEY manquant")

    print("Chargement des données...",flush=True)
    X,y,_=load_data(); donors=[c for c in X if not str(c).startswith("holed_")]
    context=ft.build_donor_context(X,donors); cache={}; progress=load_progress(args)
    completed={row["repeat"] for row in progress["repeats"]}
    stop=1 if args.dry_run else args.repeats
    if not args.dry_run:
        import tabpfn_client
        from tabpfn_client import TabPFNRegressor
        tabpfn_client.set_access_token(os.environ["TABPFN_API_KEY"])

    for repeat in range(stop):
        if repeat in completed: continue
        started=time.time(); train_cols,val_cols=split_columns(
            y.columns,repeat,args.n_train_curves,args.n_val_cols)
        needed=train_cols+val_cols
        for number,col in enumerate(needed,1):
            if col not in cache: cache[col]=build_curve_rows(col,X,y,context)
            if number%25==0: print(f"repeat {repeat+1}: features {number}/{len(needed)}",flush=True)
        train_rows=[cache[c] for c in train_cols if cache[c] is not None]
        val_rows=[cache[c] for c in val_cols if cache[c] is not None]
        Xfit,yfit=balanced_context(train_rows,args.max_train_rows,SEED+repeat)
        Xquery=np.vstack([row["X"] for row in val_rows])
        print(f"repeat {repeat+1}/{args.repeats}: fit={Xfit.shape}, query={Xquery.shape}",flush=True)
        if args.dry_run:
            print("dry-run OK: aucun appel TabPFN",flush=True); return
        error=None
        for attempt in range(MAX_RETRIES):
            try:
                reg=TabPFNRegressor(random_state=SEED+repeat)
                reg.fit(Xfit,yfit)
                mae,per_curve,n_points=predict_and_score(reg,val_rows,X,y)
                break
            except Exception as exc:
                error=f"{type(exc).__name__}: {exc}"
                if attempt+1==MAX_RETRIES: raise
                time.sleep(2*(attempt+1))
        row={"repeat":repeat,"seed":SEED+repeat,"mae_w":mae,"n_points":n_points,
             "n_train_rows":len(Xfit),"n_train_curves":len(train_rows),
             "n_val_curves":len(val_rows),"v14_reference_mae_w":v14_reference(repeat),
             "per_curve_mae_w":per_curve,"seconds":time.time()-started,"last_error":error}
        progress["repeats"].append(row); save_json(PROGRESS_PATH,progress)
        print(f"repeat {repeat+1}/{args.repeats}: MAE={mae:.3f} W",flush=True)

    scores=np.array([row["mae_w"] for row in progress["repeats"]],float)
    result={**progress,"scores_mae_w":scores.tolist(),"mean_mae_w":float(scores.mean()),
            "std_mae_w":float(scores.std())}
    save_json(RESULTS_PATH,result); print(json.dumps(result,indent=2))


if __name__=="__main__": main()
