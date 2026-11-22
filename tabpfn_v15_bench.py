"""Criblage apparié V15 : LightGBM local puis TabPFN médian 3x10/10x10."""
from __future__ import annotations
import argparse, csv, json, os, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor

from evaluate import load_data
import feature_engineering_bench as feb
import tabpfn_v14_ablation as v14
from tabpfn_v15_features import FEATURE_GROUPS, build_v15_features

REFERENCE_FEATURES_DIR = "/home/nicolas/PycharmProjects/hackathon_enedis2"
if REFERENCE_FEATURES_DIR not in sys.path:
    sys.path.insert(0, REFERENCE_FEATURES_DIR)
import features as reference_features

V14_OPTS=dict(volatility_alpha=True,vol_weight=.8,periodic_alpha_strength=1.,extra_lags=(24,),neighbor_rolling=True)
RESULTS="tabpfn_v15_feature_results.json"; RANDOM_STATE=42
PROVIDED_RESULTS_CSV = Path("tabpfn_v15_provided_features_comparison.csv")
PROVIDED_G_FEATURES = tuple(f"G{i:02d}" for i in range(1, 61))
PROVIDED_H_FEATURES = tuple(f"H{i:02d}" for i in range(1, 51))
PROVIDED_R_FEATURES = tuple(f"R{i:02d}" for i in range(1, 101))
PROVIDED_S_FEATURES = tuple(f"S{i:02d}" for i in range(1, 11))
PROVIDED_FEATURES = PROVIDED_G_FEATURES + PROVIDED_H_FEATURES + PROVIDED_R_FEATURES + PROVIDED_S_FEATURES


def _provided_anchors(values):
    """Ancres strictes : une ligne observée ne se réutilise jamais elle-même."""
    n=len(values); pos=np.arange(n); obs=np.isfinite(values)
    prev=np.maximum.accumulate(np.where(obs,pos,-1)); nxt=np.minimum.accumulate(np.where(obs,pos,n)[::-1])[::-1]
    left=np.r_[-1,prev[:-1]]; right=np.r_[n,nxt[1:]]
    lv=np.where(left>=0,values[np.clip(left,0,n-1)],np.nan); rv=np.where(right<n,values[np.clip(right,0,n-1)],np.nan)
    return pos-left, lv, right-pos, rv


def provided_feature_blocks(values, index, scaled_neighbors):
    """Réimplémentation locale des candidates G01--G60 fournies par l'utilisateur.

    Les voisins V14 déjà sélectionnés sont le panier de références. Aucune
    valeur interpolée de la cible n'est ajoutée comme feature.
    """
    n=len(values); obs=np.isfinite(values); ld,lv,rd,rv=_provided_anchors(values)
    ref=np.asarray(scaled_neighbors,float); ref_mean=ref.mean(1); ref_std=ref.std(1)
    q25,med,q75=np.quantile(ref,[.25,.5,.75],axis=1); phase=index.hour.to_numpy()*2+index.minute.to_numpy()//30; dow=index.dayofweek.to_numpy()
    own=values[obs]; mean=float(np.nanmean(own)); std=float(np.nanstd(own)); eps=1e-6
    def c(v):
        a=np.asarray(v,float); return np.full((n,1),a,float) if a.ndim==0 else a.reshape(n,1)
    slope=np.divide(rv-lv,ld+rd,out=np.full(n,np.nan),where=(ld+rd)>0)
    corr=np.nan_to_num(np.corrcoef(np.nan_to_num(values,nan=mean), ref.T)[0,1:]) if ref.shape[1] else np.zeros(1)
    weighted=ref@(corr/(np.abs(corr).sum()+eps))
    sin=np.sin(2*np.pi*phase/48); cos=np.cos(2*np.pi*phase/48); ds=np.sin(2*np.pi*dow/7); dc=np.cos(2*np.pi*dow/7)
    blocks={
        "G01":np.c_[lv,rv], "G02":np.c_[ld,rd], "G03":c(np.divide(ld,ld+rd,out=np.full(n,np.nan),where=(ld+rd)>0)), "G04":c(slope),
        "G05":c(np.abs(rv-lv)), "G06":c((lv+rv)/2), "G07":c(np.fmin(lv,rv)), "G08":c(np.fmax(lv,rv)), "G09":c(lv-ref_mean), "G10":c(rv-ref_mean),
        "G11":c(mean), "G12":c(std), "G13":c(np.nanmin(own)), "G14":c(np.nanmax(own)), "G15":c(np.nanquantile(own,.1)), "G16":c(np.nanquantile(own,.25)), "G17":c(np.nanmedian(own)), "G18":c(np.nanquantile(own,.75)), "G19":c(np.nanquantile(own,.9)), "G20":c(std/(abs(mean)+eps)),
        "G21":c(ref_mean), "G22":c(ref_std), "G23":c(med), "G24":c(q75-q25), "G25":c(ref.min(1)), "G26":c(ref.max(1)), "G27":c(ref.max(1)-ref.min(1)), "G28":c(q25), "G29":c(q75), "G30":c(med-ref_mean),
        "G31":c(np.mean(abs(corr))), "G32":c(np.max(abs(corr))), "G33":c(np.std(abs(corr))), "G34":c(np.mean(corr)), "G35":c(np.mean(corr>0)), "G36":c(np.max(abs(corr))-np.partition(abs(corr),-2)[-2] if len(corr)>1 else 0), "G37":c(weighted),
        "G38":c(sin), "G39":c(cos), "G40":c(ds), "G41":c(dc), "G42":c((dow>=5).astype(float)), "G43":c(((index.hour>=8)&(index.hour<20)).astype(float)), "G44":c((((index.hour>=7)&(index.hour<10))|((index.hour>=17)&(index.hour<21))).astype(float)), "G45":c(sin**2),
        "G46":c(obs), "G47":c(np.cumsum(obs)/np.arange(1,n+1)), "G48":c(np.arange(n)), "G49":c(n-1-np.arange(n)), "G50":c(np.log1p(ld)), "G51":c(np.log1p(rd)), "G52":c(ld+rd-1),
        "G53":np.c_[ld,lv,rd,rv], "G54":np.c_[q25,med,q75], "G55":np.c_[ref_mean,ref_std], "G56":np.c_[sin,cos,ds,dc], "G57":np.c_[lv/(mean+eps),rv/(mean+eps)], "G58":c((ref_mean-ref_mean.mean())/(ref_mean.std()+eps)), "G59":c(ref_std/(abs(ref_mean)+eps)), "G60":c(np.abs(rv-lv)/(ref_std+eps)),
    }
    return blocks


def write_provided_csv(results, model):
    rows=[]
    for key,value in results.items():
        prefix=f"{model}_10x10_"
        if key.startswith(prefix) and "mean" in value:
            rows.append((key.removeprefix(prefix),value["mean"],value["std"],value.get("delta_vs_baseline",0.0)))
    rows.sort(key=lambda row:row[1])
    output = (PROVIDED_RESULTS_CSV if model == "lightgbm"
              else Path("tabpfnv15_compariason_all_features.csv"))
    with output.open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(("feature","mae_mean_w","mae_std_w","delta_vs_baseline_w")); w.writerows(rows)

class LightGBMMae:
    def __init__(self,random_state=42):
        self.reg=LGBMRegressor(objective="mae",n_estimators=200,max_depth=3,num_leaves=15,
            learning_rate=.04,subsample=.8,colsample_bytree=.8,random_state=random_state,verbosity=-1,n_jobs=1)
    def fit(self,X,y): self.reg.fit(X,y); return self
    def predict(self,X): return self.reg.predict(X)

def predict(col,s,pool_vals,km,cluster_idx,tfeat,index,groups,regressor_cls,research_context=None):
    observed=s.notna().to_numpy(); missing=~observed; obs=np.flatnonzero(observed); miss=np.flatnonzero(missing)
    interp=v14.fill_nan_with_interpolation(s); cluster=feb.cluster_of(interp,km)
    same=cluster_idx.get(cluster); sub=pool_vals[:,same] if same is not None and len(same)>=80 else pool_vals
    knn,_,scaled=v14.weighted_knn_pred_tunable(sub,obs,miss,s.to_numpy()[obs],40,5,.15,.3,(.1,10.))
    periodic=feb.periodic_copy_features(s.to_numpy(),lags=(48,96,144,336,24),include_summary=True,normalize=True,include_availability=True)
    seasonal=feb.same_curve_seasonality_feature(s.to_numpy(),index)
    provided=[g for g in groups if g in PROVIDED_FEATURES]
    v15_groups=[g for g in groups if g not in PROVIDED_FEATURES]
    extra=np.hstack([tfeat,seasonal,periodic,feb.neighbor_rolling_features(scaled),
                     build_v15_features(s.to_numpy(),index,scaled,missing,v15_groups,pool=sub)])
    if provided:
        candidates=provided_feature_blocks(s.to_numpy(float),pd.DatetimeIndex(index),scaled)
        ref_indices=np.arange(scaled.shape[1])
        ref_values=np.asarray(scaled,float)
        corr=np.nan_to_num(np.corrcoef(np.nan_to_num(s.to_numpy(float), nan=np.nanmean(s.to_numpy(float))), ref_values.T)[0,1:])
        if any(g in PROVIDED_H_FEATURES for g in provided):
            candidates.update({key[:3]: value for key, value in reference_features.candidate_feature_blocks_v2(
                s.to_numpy(float), ref_values, ref_indices, corr, pd.DatetimeIndex(index)).items()})
        if research_context is not None:
            r_blocks = {}
            for builder in (reference_features.candidate_feature_blocks_r1, reference_features.candidate_feature_blocks_r2,
                            reference_features.candidate_feature_blocks_r3, reference_features.candidate_feature_blocks_r4,
                            reference_features.candidate_feature_blocks_r5, reference_features.candidate_feature_blocks_s1):
                r_blocks.update(builder(s.to_numpy(float), ref_values, ref_indices, corr, pd.DatetimeIndex(index), research_context, col))
            # R100 has four characters before the descriptive suffix; slicing
            # to three characters turns it into ``R10`` and makes the feature
            # impossible to retrieve.  Split at the first underscore so all
            # R/S identifiers retain their complete numeric suffix.
            candidates.update({key.split("_", 1)[0]: value for key, value in r_blocks.items()})
        extra=np.hstack([extra]+[candidates[g] for g in provided])
    try:
        pred=feb.local_regression_pred(lambda:regressor_cls(),scaled,obs,miss,s.to_numpy()[obs],extra)
        error=None
    except Exception as exc: pred=knn; error=f"{type(exc).__name__}: {exc}"
    runs=feb.run_lengths_for(missing); edges=feb.edge_distance_for(missing)
    alpha=np.array([feb.continuous_alpha(runs[i],edges[i],True) for i in miss])
    agree=feb.periodic_agreement(s.to_numpy())[miss]
    alpha*=1-agree
    alpha*=feb.volatility_alpha_mult(feb.curve_volatility(s.to_numpy()),vol_weight=.8,cap=.6)
    filled=s.copy(); filled.iloc[miss]=np.clip(alpha,0,.95)*interp.to_numpy()[miss]+(1-np.clip(alpha,0,.95))*pred
    return filled,error

def eval_split(X,y,donors,dev,val,groups,regressor_cls):
    donor=X[donors]; pool=pd.concat([donor,y[dev]],axis=1); pool_vals=pool.ffill().bfill().to_numpy(float)
    km,dl=feb.fit_shape_clusters(donor,15,RANDOM_STATE); z=(y[dev].ffill().bfill()-y[dev].ffill().bfill().mean())/(y[dev].ffill().bfill().std()+1e-9)
    labels=pd.concat([dl,pd.Series(km.predict(z.T.values),index=dev)]); pos=pd.Series(np.arange(len(labels)),index=labels.index)
    ci={cl:pos[ix].values for cl,ix in labels.groupby(labels).groups.items()}; tf=feb.time_features_full_seasonality(X.index,("hour","dow","dom","year"))
    errors=[]; fallbacks=0
    needs_research=any(g in PROVIDED_R_FEATURES or g in PROVIDED_S_FEATURES for g in groups)
    research_context=None
    if needs_research:
        # Le contexte R/S est construit une fois par split depuis le même pool
        # de donneurs que le modèle, puis partagé par les colonnes évaluées.
        ref_frame=pd.DataFrame(pool_vals[:,:min(40,pool_vals.shape[1])],index=X.index)
        research_context=reference_features.build_r_context(X,y,list(y.columns),ref_frame,ref_frame.to_numpy(float))
    with ThreadPoolExecutor(max_workers=20) as ex:
        fs={ex.submit(predict,c,X[c],pool_vals,km,ci,tf,X.index,groups,regressor_cls,research_context):c for c in val}
        for f in as_completed(fs):
            filled,err=f.result(); c=fs[f]; m=X[c].isna().to_numpy()&y[c].notna().to_numpy()
            errors.append(np.abs(filled.to_numpy()[m]-y[c].to_numpy()[m])); fallbacks+=bool(err)
    return float(np.concatenate(errors).mean()),fallbacks

def run(name,groups,X,y,donors,repeats,nval,regressor_cls):
    scores=[]
    for i in range(repeats):
        seed=RANDOM_STATE+i; dev,valfull=train_test_split(list(y.columns),test_size=.1,random_state=seed)
        val=list(np.random.RandomState(seed).choice(valfull,min(nval,len(valfull)),replace=False)); t=time.time()
        score,fb=eval_split(X,y,donors,dev,val,groups,regressor_cls); scores.append(score)
        print(f"[{name}] repeat {i+1}/{repeats}: MAE={score:.3f} W time={time.time()-t:.1f}s fallbacks={fb}",flush=True)
    return np.asarray(scores)

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--model",choices=("lightgbm","tabpfn"),default="lightgbm")
    p.add_argument("--confirm",action="store_true")
    p.add_argument("--repeats",type=int,default=None,help="override le nombre de repeats (ex: 1 pour un sanity check rapide)")
    p.add_argument("--nval",type=int,default=10,help="colonnes de validation par repeat")
    p.add_argument("--provided-all",action="store_true",help="teste G01..G60, une par une, contre la baseline")
    p.add_argument("--all-one-by-one",action="store_true",help="teste toutes les features cataloguées une par une")
    p.add_argument("--results",default=RESULTS,help="fichier JSON de sortie/reprise")
    p.add_argument("--baseline-groups",default="",help="groupes toujours actifs dans la baseline, séparés par +")
    p.add_argument("groups",nargs="*"); a=p.parse_args()
    RESULTS = a.results
    if a.model != "lightgbm":
        RESULTS = f"tabpfn_v15_{a.model}_feature_results.json"
    cls=LightGBMMae
    if a.model=="tabpfn":
        if not os.environ.get("TABPFN_API_KEY"): raise SystemExit("TABPFN_API_KEY manquant")
        import tabpfn_client; tabpfn_client.set_access_token(os.environ["TABPFN_API_KEY"]); cls=v14.MedianTabPFN
    X,y,_=load_data(); donors=[c for c in X if not str(c).startswith("holed_")]
    baseline_groups=tuple(g for g in a.baseline_groups.split("+") if g)
    all_names=list(dict.fromkeys(list(PROVIDED_FEATURES)+list(FEATURE_GROUPS)))
    names=a.groups or (["baseline"]+all_names if a.all_one_by_one else (["baseline"]+list(PROVIDED_FEATURES) if a.provided_all else (["baseline"]+list(FEATURE_GROUPS)))); repeats=a.repeats if a.repeats is not None else (10 if a.confirm else 3)
    try: results=json.load(open(RESULTS))
    except FileNotFoundError: results={}
    for name in names:
        groups=baseline_groups if name=="baseline" else (tuple(FEATURE_GROUPS) if name=="all" else baseline_groups+tuple(name.split("+")))
        if any(g not in FEATURE_GROUPS and g not in PROVIDED_FEATURES for g in groups): raise SystemExit(f"groupe inconnu: {groups}")
        key=f"{a.model}_{repeats}x{a.nval}_{name}" if not baseline_groups else f"{a.model}_{repeats}x{a.nval}_H28base_{name}"
        if key in results:
            print(f"[{key}] déjà évaluée, skip", flush=True)
            continue
        s=run(key,groups,X,y,donors,repeats,a.nval,cls)
        baseline_key=f"{a.model}_10x10_baseline" if not baseline_groups else f"{a.model}_10x10_H28base_baseline"
        baseline=results.get(baseline_key,{}).get("mean",float(s.mean()))
        results[key]={"groups":groups,"scores":s.tolist(),"mean":float(s.mean()),"std":float(s.std()),"delta_vs_baseline":float(s.mean()-baseline)}
        with open(RESULTS,"w") as f: json.dump(results,f,indent=2)
        write_provided_csv(results, a.model)
