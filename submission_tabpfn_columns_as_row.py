"""Generate an Enedis submission with exact TabFM-shaped TabPFN episodes."""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import load_data
import tabfm_lora_finetune as ft


SEED=42
EXPECTED_FEATURES=96
MAX_RETRIES=3
DEFAULT_OUTPUT=Path("submission_tabpfn_columns_as_row.csv")
DEFAULT_PARTIAL=Path("submission_tabpfn_columns_as_row_partial.csv")
SAMPLE_PATH=Path("data/new_output_sample.csv")
DEFAULT_WORKERS=20


def maximal_pool(X,y,Xtest):
    test_donors=[c for c in Xtest if not str(c).startswith("holed_")]
    train_donors=[c for c in X if not str(c).startswith("holed_")]
    # Prefixes guarantee unique identifiers; feature construction depends on
    # values and clusters, not on the original random column identifiers.
    test=Xtest[test_donors].copy(); test.columns=[f"test::{c}" for c in test.columns]
    train=X[train_donors].copy(); train.columns=[f"train::{c}" for c in train.columns]
    truth=y.copy(); truth.columns=[f"truth::{c}" for c in truth.columns]
    pool=pd.concat([test,train,truth],axis=1)
    if not pool.columns.is_unique: raise RuntimeError("Maximal pool columns are not unique")
    return pool


def make_submission_episode(col,s,context,config):
    q=np.flatnonzero(s.isna().to_numpy())
    episode=ft.make_episode(col,s,s,context,config,query_idx=q,
                            rng=np.random.RandomState(SEED),require_targets=False)
    if episode is None: return None
    if episode["x"].shape[1]!=EXPECTED_FEATURES:
        raise RuntimeError(f"Expected {EXPECTED_FEATURES} features")
    if episode["train_size"]>config.max_context:
        raise RuntimeError("Episode exceeds max_context")
    return episode


def predict_episode(regressor_cls,episode,s):
    n=episode["train_size"]
    reg=regressor_cls(random_state=SEED)
    reg.fit(episode["x"][:n],episode["y"][:n])
    normalized=np.asarray(reg.predict(
        episode["x"][n:],output_type="median"),float)
    raw=normalized*episode["y_scale"]+episode["y_center"]
    q=episode["query_idx"]
    pred=ft.blend_episode_prediction(raw,episode,s,q)
    return np.clip(pred,0,None),q


def complete_column(col,s,context,config,regressor_cls):
    """Worker operation: no shared DataFrame or checkpoint mutation."""
    episode=make_submission_episode(col,s,context,config)
    if episode is None:
        return col,None,None,"No submission episode"
    last_error=None
    for attempt in range(MAX_RETRIES):
        try:
            pred,q=predict_episode(regressor_cls,episode,s)
            filled=s.copy(); filled.iloc[q]=pred
            stats={"context":int(episode["train_size"]),"queries":int(len(q))}
            return col,filled,stats,last_error
        except Exception as exc:
            last_error=f"{type(exc).__name__}: {exc}"
            if attempt+1<MAX_RETRIES: time.sleep(2*(attempt+1))
    return col,None,None,last_error


def batches(values,size):
    if size<1: raise ValueError("batch size must be positive")
    for start in range(0,len(values),size): yield values[start:start+size]


def validate_frame(frame,Xtest,holes,complete=False):
    if not frame.index.equals(Xtest.index): raise RuntimeError("Checkpoint index mismatch")
    if any(col not in holes for col in frame.columns):
        raise RuntimeError("Checkpoint contains an unexpected column")
    if complete and list(frame.columns)!=holes: raise RuntimeError("Submission columns mismatch")
    if frame.size and not np.isfinite(frame.to_numpy(float)).all():
        raise RuntimeError("Checkpoint contains NaN or infinity")
    for col in frame.columns:
        known=Xtest[col].notna().to_numpy()
        if not np.array_equal(frame[col].to_numpy()[known],Xtest[col].to_numpy()[known]):
            raise RuntimeError(f"Observed values changed for {col}")


def save_frame_atomic(frame,path):
    tmp=Path(str(path)+".tmp"); frame.to_csv(tmp); os.replace(tmp,path)


def load_partial(path,Xtest,holes,resume):
    if resume and path.exists():
        done=pd.read_csv(path,index_col=0)
        if not done.index.equals(Xtest.index):
            # CSV parsing and the source loader both normally yield strings;
            # fail rather than silently replacing a genuinely different index.
            raise RuntimeError("Partial checkpoint index differs from X_test")
        validate_frame(done,Xtest,holes)
        return done
    return pd.DataFrame(index=Xtest.index)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--limit",type=int,default=None)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--partial",type=Path,default=DEFAULT_PARTIAL)
    parser.add_argument("--workers",type=int,default=DEFAULT_WORKERS)
    args=parser.parse_args()
    if args.workers<1: raise SystemExit("--workers must be at least 1")
    if args.limit is not None and not args.dry_run:
        raise SystemExit("--limit is reserved for --dry-run")
    token=os.environ.get("TABPFN_API_KEY")
    if not args.dry_run and not token: raise SystemExit("TABPFN_API_KEY manquant")

    print("Chargement des données...",flush=True)
    X,y,Xtest=load_data(); sample=pd.read_csv(SAMPLE_PATH,index_col=0)
    if not X.index.equals(Xtest.index) or not sample.index.equals(Xtest.index):
        raise RuntimeError("Train/test/sample timestamps differ")
    holes=list(sample.columns)
    if any(col not in Xtest for col in holes): raise RuntimeError("Sample columns absent from X_test")
    pool=maximal_pool(X,y,Xtest)
    print(f"Pool maximal: {pool.shape[1]} donneuses",flush=True)
    context=ft.build_donor_context(pool,list(pool.columns)); config=ft.TrainConfig()

    selected=holes[:(args.limit if args.limit is not None else (2 if args.dry_run else len(holes)))]
    if args.dry_run:
        for col in selected:
            episode=make_submission_episode(col,Xtest[col],context,config)
            if episode is None: raise RuntimeError(f"No holes for {col}")
            print(f"{col}: context={episode['train_size']}, queries={len(episode['query_idx'])}, "
                  f"features={episode['x'].shape[1]}",flush=True)
        print("dry-run OK: aucun appel TabPFN",flush=True); return

    import tabpfn_client
    from tabpfn_client import TabPFNRegressor
    tabpfn_client.set_access_token(token)
    done=load_partial(args.partial,Xtest,holes,args.resume)
    remaining=[col for col in holes if col not in done]
    print(f"Reprise: {len(done.columns)} faites, {len(remaining)} restantes; "
          f"workers={args.workers}",flush=True)
    started=time.time(); processed=0; new_successes=0; failures={}
    for batch_number,batch in enumerate(batches(remaining,args.workers),1):
        successes=[]
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures={executor.submit(complete_column,col,Xtest[col],context,config,
                                     TabPFNRegressor):col for col in batch}
            for future in as_completed(futures):
                col,filled,stats,error=future.result()
                if filled is None:
                    failures[col]=error
                    print(f"ÉCHEC {col}: {error}",flush=True)
                    continue
                # Only the main thread mutates pandas state.
                done[col]=filled; validate_frame(done[[col]],Xtest,holes)
                successes.append((col,stats,error))
                print(f"OK {col}: context={stats['context']} queries={stats['queries']} "
                      f"retry_error={error}",flush=True)
        if successes: save_frame_atomic(done,args.partial)
        processed+=len(batch); new_successes+=len(successes); elapsed=time.time()-started
        eta=(elapsed/processed*(len(remaining)-processed)/60 if processed else 0.)
        rate=(new_successes/elapsed*60 if elapsed else 0.)
        print(f"lot {batch_number}: succès={len(successes)}/{len(batch)} "
              f"total={len(done.columns)}/{len(holes)} vitesse={rate:.1f} col/min "
              f"ETA={eta:.1f} min échecs={len(failures)}",flush=True)

    missing=[col for col in holes if col not in done]
    if missing:
        preview=", ".join(f"{col} ({failures.get(col,'non traité')})" for col in missing[:10])
        raise RuntimeError(f"{len(missing)} courbes incomplètes; relancer --resume: {preview}")

    final=done[holes]
    final.index.name="Horodate"; validate_frame(final,Xtest,holes,complete=True)
    if final.shape!=sample.shape: raise RuntimeError("Submission shape mismatch")
    save_frame_atomic(final,args.output)
    print(f"Soumission écrite: {args.output} ({final.shape[0]} x {final.shape[1]})",flush=True)


if __name__=="__main__": main()
