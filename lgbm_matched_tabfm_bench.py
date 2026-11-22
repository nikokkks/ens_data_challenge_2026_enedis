"""LightGBM on the exact seed-42 TabFM-LoRA v2 episodes and holdout."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from lightgbm import LGBMRegressor

from evaluate import load_data
import tabfm_lora_finetune as ft


RESULTS_PATH=Path("lgbm_matched_tabfm_results.json")
EXPECTED_FEATURES=96


def make_regressor():
    return LGBMRegressor(
        objective="mae",n_estimators=200,max_depth=3,learning_rate=.1,
        subsample=.8,colsample_bytree=.8,random_state=ft.SEED,
        n_jobs=1,verbosity=-1,
    )


def training_matrix(X,y,donor_cols,context,config,manifest):
    """Reproduce the first deterministic TabFM epoch and retain query rows."""
    train_cols=list(manifest["train"]); xs=[]; ys=[]
    counts={"real":0,"augmented":0}; curve_count=0
    loader=ft.make_curve_loader(train_cols,config.batch_size,ft.SEED)
    for batch_number,batch in enumerate(loader,1):
        episodes,batch_counts=ft.make_training_batch(
            batch,train_cols,X,y,context,config)
        for kind,count in batch_counts.items(): counts[kind]+=count
        for episode in episodes:
            start=episode["train_size"]
            query_x=episode["x"][start:]
            if query_x.shape[1]!=EXPECTED_FEATURES:
                raise RuntimeError(f"Expected {EXPECTED_FEATURES} features, got {query_x.shape[1]}")
            if len(query_x)!=len(episode["target"]):
                raise RuntimeError("Episode feature/target alignment mismatch")
            xs.append(query_x); ys.append(episode["target"]); curve_count+=1
        print(f"train batch {batch_number}/{len(loader)}: episodes={curve_count}",flush=True)
    if not xs: raise RuntimeError("No training episodes")
    Xfit=np.vstack(xs).astype(np.float32); yfit=np.concatenate(ys).astype(np.float32)
    if not np.isfinite(Xfit).all() or not np.isfinite(yfit).all():
        raise RuntimeError("Non-finite LightGBM training data")
    return Xfit,yfit,counts,curve_count


def evaluate_holdout(reg,X,y,context,config,columns):
    errors=[]; per_curve={}; n_features=None
    for number,col in enumerate(columns,1):
        q=ft.real_gap_indices(X[col],y[col])
        episode=ft.make_episode(col,X[col],y[col],context,config,query_idx=q,
                                rng=np.random.RandomState(ft.SEED),require_targets=True)
        if episode is None: continue
        query_x=episode["x"][episode["train_size"]:]
        n_features=query_x.shape[1]
        if n_features!=EXPECTED_FEATURES: raise RuntimeError("Holdout feature count mismatch")
        normalized=np.asarray(reg.predict(query_x),float)
        raw=normalized*episode["y_scale"]+episode["y_center"]
        q=episode["query_idx"]
        pred=ft.blend_episode_prediction(raw,episode,X[col],q)
        truth=y[col].iloc[q].to_numpy(float)
        err=np.abs(pred-truth)
        errors.append(err); per_curve[col]=float(err.mean())
        if number%10==0:
            print(f"test {number}/{len(columns)}: cumulative points={sum(map(len,errors))}",flush=True)
    if not errors: raise RuntimeError("No finite holdout targets")
    all_errors=np.concatenate(errors)
    return {"mae_w":float(all_errors.mean()),"n_points":int(len(all_errors)),
            "n_curves":len(per_curve),"n_features":n_features,
            "per_curve_mae_w":per_curve}


def validate_manifest(manifest,config):
    if manifest.get("schema_version")!=ft.MANIFEST_SCHEMA_VERSION:
        raise RuntimeError("Expected TabFM-LoRA manifest v2")
    if manifest.get("training_protocol")!=ft.TRAINING_PROTOCOL:
        raise RuntimeError("Unexpected TabFM training protocol")
    if manifest.get("outer_seed")!=ft.SEED:
        raise RuntimeError("Expected outer_seed=42")
    groups=[set(manifest[name]) for name in ("train","val","test")]
    if [len(group) for group in groups]!=[config.train_columns,config.val_columns,100]:
        raise RuntimeError("Expected a 880/20/100 split")
    if groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2]:
        raise RuntimeError("Manifest groups overlap")


def main():
    started=time.time(); print("Chargement des données...",flush=True)
    X,y,_=load_data(); config=ft.TrainConfig()
    manifest=ft.load_or_create_split_manifest(X,y,config,ft.SEED)
    validate_manifest(manifest,config)
    donor_cols=[c for c in X if not str(c).startswith("holed_")]
    context=ft.build_donor_context(X,donor_cols)
    Xfit,yfit,counts,n_episodes=training_matrix(
        X,y,donor_cols,context,config,manifest)
    print(f"LightGBM fit: X={Xfit.shape}, y={yfit.shape}, episodes={counts}",flush=True)
    reg=make_regressor(); fit_started=time.time(); reg.fit(Xfit,yfit)
    score=evaluate_holdout(reg,X,y,context,config,manifest["test"])
    result={"protocol":"tabfm_lora_v2_first_epoch_matched",
            "split_manifest":str(ft.SPLIT_PATH),"outer_seed":ft.SEED,
            "train_columns":len(manifest["train"]),"val_columns_excluded":len(manifest["val"]),
            "test_columns":len(manifest["test"]),"training_rows":len(Xfit),
            "training_episodes":n_episodes,"episode_counts":counts,
            "model_params":reg.get_params(),"fit_seconds":time.time()-fit_started,
            "total_seconds":time.time()-started,"test":score}
    ft._write_json_atomic(RESULTS_PATH,result)
    print(json.dumps({k:v for k,v in result.items() if k!="test"},indent=2),flush=True)
    print(f"MAE LightGBM appariée: {score['mae_w']:.6f} W "
          f"sur {score['n_points']} points / {score['n_curves']} courbes",flush=True)


if __name__=="__main__": main()
