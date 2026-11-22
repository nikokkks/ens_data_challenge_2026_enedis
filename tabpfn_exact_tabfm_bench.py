"""TabPFN on the exact seed-42 TabFM-LoRA validation and test episodes."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from evaluate import load_data
import tabfm_lora_finetune as ft


PROGRESS_PATH=Path("tabpfn_exact_tabfm_progress.json")
RESULTS_PATH=Path("tabpfn_exact_tabfm_results.json")
MAX_RETRIES=3
EXPECTED_FEATURES=96


def validate_manifest(manifest,config):
    if manifest.get("schema_version")!=ft.MANIFEST_SCHEMA_VERSION:
        raise RuntimeError("Expected TabFM-LoRA manifest v2")
    if manifest.get("training_protocol")!=ft.TRAINING_PROTOCOL:
        raise RuntimeError("Unexpected TabFM-LoRA training protocol")
    if manifest.get("outer_seed")!=ft.SEED:
        raise RuntimeError("Expected outer_seed=42")
    groups=[set(manifest[name]) for name in ("train","val","test")]
    if [len(g) for g in groups]!=[config.train_columns,config.val_columns,100]:
        raise RuntimeError("Expected a 880/20/100 split")
    if groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2]:
        raise RuntimeError("Manifest groups overlap")


def progress_signature(manifest):
    return {"schema_version":1,"outer_seed":manifest["outer_seed"],
            "training_protocol":manifest["training_protocol"],
            "feature_count":EXPECTED_FEATURES,
            "regressor":"TabPFNRegressor(default regression model, median)",
            "random_state":ft.SEED}


def load_progress(manifest,resume):
    signature=progress_signature(manifest)
    if resume and PROGRESS_PATH.exists():
        progress=json.loads(PROGRESS_PATH.read_text())
        if progress.get("signature")!=signature:
            raise RuntimeError("Existing progress uses another protocol")
        return progress
    return {"signature":signature,"val":{},"test":{}}


def make_exact_episode(split,j,col,X,y,context,config):
    q=ft.real_gap_indices(X[col],y[col])
    # This mirrors train-time validation_mae (42+j) and evaluate_manifest
    # test inference (predict_curve uses 42 for every curve), respectively.
    episode_seed=ft.SEED+j if split=="val" else ft.SEED
    episode=ft.make_episode(col,X[col],y[col],context,config,query_idx=q,
                            rng=np.random.RandomState(episode_seed),require_targets=True)
    if episode is None: return None
    if episode["x"].shape[1]!=EXPECTED_FEATURES:
        raise RuntimeError(f"Expected {EXPECTED_FEATURES} features")
    return episode


def predict_episode(regressor_cls,episode,s):
    n=episode["train_size"]
    reg=regressor_cls(random_state=ft.SEED)
    reg.fit(episode["x"][:n],episode["y"][:n])
    normalized=np.asarray(reg.predict(
        episode["x"][n:],output_type="median"),float)
    raw=normalized*episode["y_scale"]+episode["y_center"]
    q=episode["query_idx"]
    return ft.blend_episode_prediction(raw,episode,s,q)


def run_split(split,columns,X,y,context,config,progress,regressor_cls):
    rows=progress[split]
    for j,col in enumerate(columns):
        if col in rows: continue
        episode=make_exact_episode(split,j,col,X,y,context,config)
        if episode is None: continue
        last_error=None; started=time.time()
        for attempt in range(MAX_RETRIES):
            try:
                pred=predict_episode(regressor_cls,episode,X[col]); break
            except Exception as exc:
                last_error=f"{type(exc).__name__}: {exc}"
                if attempt+1==MAX_RETRIES: raise
                time.sleep(2*(attempt+1))
        q=episode["query_idx"]
        truth=y[col].iloc[q].to_numpy(float)
        error=np.abs(pred-truth)
        rows[col]={"sum_abs_error":float(error.sum()),"n_points":int(len(error)),
                   "mae_w":float(error.mean()),"train_size":int(episode["train_size"]),
                   "seconds":time.time()-started,"last_error":last_error}
        ft._write_json_atomic(PROGRESS_PATH,progress)
        print(f"TabPFN {split} {j+1}/{len(columns)} {col}: "
              f"MAE={error.mean():.4f} W",flush=True)


def summarize(rows):
    if not rows: raise RuntimeError("No completed predictions")
    total=sum(row["n_points"] for row in rows.values())
    return {"mae_w":sum(row["sum_abs_error"] for row in rows.values())/total,
            "n_points":total,"n_curves":len(rows),
            "column_mean_mae_w":float(np.mean([row["mae_w"] for row in rows.values()])),
            "column_std_mae_w":float(np.std([row["mae_w"] for row in rows.values()])),
            "per_curve_mae_w":{col:row["mae_w"] for col,row in rows.items()}}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("val","test","all"))
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    token=os.environ.get("TABPFN_API_KEY")
    if not token: raise SystemExit("TABPFN_API_KEY manquant")
    import tabpfn_client
    from tabpfn_client import TabPFNRegressor
    tabpfn_client.set_access_token(token)

    print("Chargement des données...",flush=True)
    X,y,_=load_data(); config=ft.TrainConfig()
    manifest=ft.load_or_create_split_manifest(X,y,config,ft.SEED)
    validate_manifest(manifest,config)
    donors=[c for c in X if not str(c).startswith("holed_")]
    context=ft.build_donor_context(X,donors)
    progress=load_progress(manifest,args.resume)
    requested=("val","test") if args.stage=="all" else (args.stage,)
    for split in requested:
        run_split(split,manifest[split],X,y,context,config,progress,TabPFNRegressor)
    result={"signature":progress["signature"],"split_manifest":str(ft.SPLIT_PATH),
            "scores":{split:summarize(progress[split]) for split in requested}}
    ft._write_json_atomic(RESULTS_PATH,result)
    for split,score in result["scores"].items():
        print(f"MAE TabPFN {split}: {score['mae_w']:.6f} W sur "
              f"{score['n_points']} points / {score['n_curves']} courbes",flush=True)


if __name__=="__main__": main()
