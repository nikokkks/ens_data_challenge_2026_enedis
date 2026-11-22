"""Generate an Enedis submission with the 162-feature H28+H07 pipeline."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import load_data
import tabfm_lora_finetune as ft
import tabpfn_column_individual_bench as old
import tabpfn_best_features_column_bench as best

DEFAULT_OUTPUT=Path("submission_tabpfn_h28_h07.csv")
PROGRESS_PATH=Path("submission_tabpfn_h28_h07_progress.json")
SAMPLE_PATH=Path("data/new_output_sample.csv")
DEFAULT_SEEDS=(42,43,44,45,46)


def member_path(seed): return Path(f"submission_tabpfn_h28_h07_seed{seed}.csv")


def signature(args,seeds):
    return {"schema_version":1,"seeds":list(seeds),"n_train_curves":args.n_train_curves,
            "max_train_rows":args.max_train_rows,"predict_batch_rows":args.predict_batch_rows,
            "feature_count":best.FEATURE_COUNT,"feature_groups":list(best.FEATURE_GROUPS),
            "strategy":"column_individual_median_ensemble_162_h28_h07"}


def save_json(path,payload):
    tmp=Path(str(path)+".tmp"); tmp.write_text(json.dumps(payload,indent=2)); os.replace(tmp,path)


def save_frame(frame,path):
    tmp=Path(str(path)+".tmp"); frame.to_csv(tmp); os.replace(tmp,path)


def validate_frame(frame,Xtest,holes):
    if frame.shape!=(len(Xtest),len(holes)) or list(frame.columns)!=holes:
        raise RuntimeError("Submission member schema mismatch")
    if not frame.index.equals(Xtest.index): raise RuntimeError("Submission member index mismatch")
    values=frame.to_numpy(float)
    if not np.isfinite(values).all(): raise RuntimeError("NaN or infinity in submission")
    known=Xtest[holes].notna().to_numpy()
    if not np.array_equal(values[known],Xtest[holes].to_numpy(float)[known]):
        raise RuntimeError("Observed X_test values were modified")


def load_progress(args,seeds):
    expected=signature(args,seeds)
    if args.resume and PROGRESS_PATH.exists():
        progress=json.loads(PROGRESS_PATH.read_text())
        if progress.get("signature")!=expected: raise RuntimeError("Progress signature mismatch")
        return progress
    return {"signature":expected,"completed":{}}


def predict_member(reg,rows,Xtest,holes,batch_rows):
    query=np.vstack([r["X"] for r in rows]).astype(np.float32); pieces=[]
    for start in range(0,len(query),batch_rows):
        stop=min(start+batch_rows,len(query))
        pieces.append(np.asarray(reg.predict(query[start:stop],output_type="median"),float))
        print(f"prediction {stop}/{len(query)}",flush=True)
    normalized=np.concatenate(pieces); member=Xtest[holes].copy(); offset=0
    for row in rows:
        n=len(row["q"]); q=row["q"]
        raw=normalized[offset:offset+n]*row["scale"]+row["center"]
        episode={"missing":row["missing"],"interp":row["interp"]}
        pred=ft.blend_episode_prediction(raw,episode,Xtest[row["col"]],q)
        member.loc[member.index[q],row["col"]]=np.clip(pred,0,None); offset+=n
    if offset!=len(normalized): raise RuntimeError("Prediction alignment mismatch")
    validate_frame(member,Xtest,holes); return member


def main():
    p=argparse.ArgumentParser(); p.add_argument("--resume",action="store_true")
    p.add_argument("--dry-run",action="store_true"); p.add_argument("--limit",type=int)
    p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    p.add_argument("--predict-batch-rows",type=int,default=5000)
    p.add_argument("--n-train-curves",type=int,default=old.N_TRAIN_CURVES)
    p.add_argument("--max-train-rows",type=int,default=old.MAX_TRAIN_ROWS)
    p.add_argument("--seeds",type=int,nargs="+",default=list(DEFAULT_SEEDS))
    args=p.parse_args(); seeds=tuple(args.seeds)
    if len(set(seeds))!=len(seeds) or not seeds: raise SystemExit("--seeds must be unique and non-empty")
    if args.limit is not None and not args.dry_run: raise SystemExit("--limit is reserved for --dry-run")
    token=os.environ.get("TABPFN_API_KEY")
    if not args.dry_run and not token: raise SystemExit("TABPFN_API_KEY manquant")
    print("Pipeline TabPFN 162 features: base 160 + H28 + H07.",flush=True)
    X,y,Xtest=load_data(); sample=pd.read_csv(SAMPLE_PATH,index_col=0); holes=list(sample.columns)
    if not X.index.equals(Xtest.index) or not sample.index.equals(Xtest.index):
        raise RuntimeError("Train/test/sample timestamps differ")
    if any(c not in Xtest for c in holes): raise RuntimeError("Sample columns absent from X_test")
    donors=[c for c in X if not str(c).startswith("holed_")]
    context=ft.build_donor_context(X,donors)
    selected=holes[:(args.limit if args.limit is not None else (2 if args.dry_run else len(holes)))]
    query_rows=[]
    for number,col in enumerate(selected,1):
        row=best.build_query_rows(col,Xtest,context)
        if row is not None: query_rows.append(row)
        if number%25==0: print(f"features test {number}/{len(selected)}",flush=True)
    if len(query_rows)!=len(selected): raise RuntimeError("A test curve has no query rows")
    train_cache={}; first_train,_=old.split_columns(y.columns,seeds[0]-old.SEED,args.n_train_curves,old.N_VAL_COLS)
    dry_train=first_train[:2] if args.dry_run else first_train
    for col in dry_train: train_cache[col]=best.build_curve_rows(col,X,y,context)
    rows=[train_cache[c] for c in dry_train if train_cache[c] is not None]
    Xfit,yfit=old.balanced_context(rows,min(args.max_train_rows,100) if args.dry_run else args.max_train_rows,seeds[0])
    Xquery=np.vstack([r["X"] for r in query_rows])
    if args.dry_run:
        if Xfit.shape[1]!=best.FEATURE_COUNT or Xquery.shape[1]!=best.FEATURE_COUNT:
            raise RuntimeError("Dry-run feature count mismatch")
        print(f"dry-run OK: fit={Xfit.shape}, query={Xquery.shape}, courbes={len(query_rows)}, aucun appel TabPFN",flush=True); return
    import tabpfn_client
    from tabpfn_client import TabPFNRegressor
    tabpfn_client.set_access_token(token); progress=load_progress(args,seeds)
    for seed in seeds:
        path=member_path(seed)
        if str(seed) in progress["completed"] and path.exists():
            frame=pd.read_csv(path,index_col=0); validate_frame(frame,Xtest,holes)
            print(f"seed {seed}: reprise",flush=True); continue
        repeat=seed-old.SEED
        if repeat<0: raise RuntimeError("Seeds must be >= 42 for historical split mapping")
        train_cols,_=old.split_columns(y.columns,repeat,args.n_train_curves,old.N_VAL_COLS)
        for number,col in enumerate(train_cols,1):
            if col not in train_cache: train_cache[col]=best.build_curve_rows(col,X,y,context)
            if number%25==0: print(f"seed {seed}: features train {number}/{len(train_cols)}",flush=True)
        train_rows=[train_cache[c] for c in train_cols if train_cache[c] is not None]
        Xfit,yfit=old.balanced_context(train_rows,args.max_train_rows,seed)
        print(f"seed {seed}: fit={Xfit.shape}",flush=True); last_error=None
        for attempt in range(old.MAX_RETRIES):
            try:
                reg=TabPFNRegressor(random_state=seed); reg.fit(Xfit,yfit)
                member=predict_member(reg,query_rows,Xtest,holes,args.predict_batch_rows); break
            except Exception as exc:
                last_error=f"{type(exc).__name__}: {exc}"
                if attempt+1==old.MAX_RETRIES: raise
                print(f"seed {seed}: tentative {attempt+1} échouée: {last_error}",flush=True)
                time.sleep(2*(attempt+1))
        save_frame(member,path); progress["completed"][str(seed)]={"path":str(path),"n_train_rows":len(Xfit),"last_error":last_error}
        save_json(PROGRESS_PATH,progress)
    members=[]
    for seed in seeds:
        frame=pd.read_csv(member_path(seed),index_col=0); validate_frame(frame,Xtest,holes)
        members.append(frame.to_numpy(float))
    median=np.median(np.stack(members),axis=0); final=Xtest[holes].copy()
    missing=final.isna().to_numpy(); values=final.to_numpy(float); values[missing]=median[missing]
    final=pd.DataFrame(values,index=Xtest.index,columns=holes); final.index.name="Horodate"
    validate_frame(final,Xtest,holes)
    if final.shape!=sample.shape: raise RuntimeError("Final submission shape mismatch")
    save_frame(final,args.output); print(f"Soumission écrite: {args.output} {final.shape}",flush=True)


if __name__=="__main__": main()
