"""Best TabPFN V14 on the exact TabFM-LoRA 880/20/100 split and masks."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import tabpfn_client
from tabpfn_client import TabPFNRegressor

from evaluate import load_data
import tabfm_lora_finetune as ft

CHECKPOINT=Path("tabpfn_matched_lora_v2_progress.json")
RESULTS=Path("tabpfn_matched_lora_results.json")
MAX_RETRIES=3


def predict_tabpfn_episode(ep, s, q):
    n=ep["train_size"]
    reg=TabPFNRegressor(random_state=ft.SEED)
    reg.fit(ep["x"][:n],ep["y"][:n])
    normalized=np.asarray(reg.predict(ep["x"][n:],output_type="median"),float)
    raw=normalized*ep["y_scale"]+ep["y_center"]
    return ft.blend_episode_prediction(raw,ep,s,q)


def load_progress():
    if CHECKPOINT.exists(): return json.loads(CHECKPOINT.read_text())
    return {"schema_version":1,"val":{},"test":{}}


def save_progress(progress):
    tmp=Path(str(CHECKPOINT)+".tmp"); tmp.write_text(json.dumps(progress,indent=2)); os.replace(tmp,CHECKPOINT)


def run_split(name,columns,manifest,X,y,context,config,progress):
    for number,col in enumerate(columns,1):
        if col in progress[name]: continue
        q=ft.real_gap_indices(X[col],y[col])
        ep=ft.make_episode(col,X[col],y[col],context,config,query_idx=q,
                           rng=np.random.RandomState(ft.SEED),require_targets=True)
        if ep is None: continue
        error=None
        for attempt in range(MAX_RETRIES):
            try:
                pred=predict_tabpfn_episode(ep,X[col],q); break
            except Exception as exc:
                error=f"{type(exc).__name__}: {exc}"
                if attempt+1==MAX_RETRIES: raise
                time.sleep(2*(attempt+1))
        true=y[col].iloc[q].to_numpy(); valid=np.isfinite(true)
        abs_err=np.abs(pred[valid]-true[valid])
        progress[name][col]={"sum_abs_error":float(abs_err.sum()),"n":int(len(abs_err)),
                             "mae":float(abs_err.mean()),"last_error":error}
        save_progress(progress)
        print(f"TabPFN {name} {number}/{len(columns)} {col}: {abs_err.mean():.2f} W",flush=True)


def summarize(rows):
    values=list(rows.values()); total_n=sum(v["n"] for v in values)
    per=np.array([v["mae"] for v in values],float)
    return {"mae":sum(v["sum_abs_error"] for v in values)/total_n,
            "column_mean":float(per.mean()),"column_std":float(per.std()),
            "n_columns":len(values),"n_points":total_n,
            "per_column":{k:v["mae"] for k,v in rows.items()}}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("stage",choices=["val","test","all"])
    parser.add_argument("--outer-seed",type=int,default=ft.SEED); args=parser.parse_args()
    token=os.environ.get("TABPFN_API_KEY")
    if not token: raise SystemExit("TABPFN_API_KEY manquant")
    tabpfn_client.set_access_token(token)
    X,y,_=load_data(); config=ft.TrainConfig()
    manifest=ft.load_or_create_split_manifest(X,y,config,args.outer_seed)
    donors=[c for c in X if not str(c).startswith("holed_")]
    context=ft.build_donor_context(X,donors); progress=load_progress()
    progress_manifest={k:manifest[k] for k in
                       ("schema_version","outer_seed","training_protocol",
                        "augmentation_fraction")}
    if "manifest" in progress and progress["manifest"]!=progress_manifest:
        raise RuntimeError("Checkpoint created with a different split manifest")
    progress["manifest"]=progress_manifest
    if args.stage in ("val","all"): run_split("val",manifest["val"],manifest,X,y,context,config,progress)
    if args.stage in ("test","all"): run_split("test",manifest["test"],manifest,X,y,context,config,progress)
    requested=("val","test") if args.stage=="all" else (args.stage,)
    result={"split_manifest":str(ft.SPLIT_PATH),
            "training_protocol":ft.TRAINING_PROTOCOL,
            "tabpfn":{s:summarize(progress[s]) for s in requested}}
    if ft.RESULTS_PATH.exists():
        lora=json.loads(ft.RESULTS_PATH.read_text())
        if (lora.get("protocol")=="matched" and
                lora.get("training_protocol")==ft.TRAINING_PROTOCOL):
            result["lora"]=lora["lora"]
    RESULTS.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))


if __name__=="__main__": main()
