"""Validation TabPFN appariée des cinq améliorations post-C2."""
import json, os, sys, warnings
import numpy as np
import tabpfn_client
from tabpfn_client import TabPFNRegressor
from lightgbm import LGBMRegressor
from evaluate import load_data
import feature_engineering_bench as feb

warnings.filterwarnings("ignore")
RANDOM_STATE=42

class MedianTabPFN:
    def __init__(self): self.reg=TabPFNRegressor(random_state=RANDOM_STATE)
    def fit(self,X,y): self.reg.fit(X,y); return self
    def predict(self,X): return self.reg.predict(X,output_type="median")

class TabPFNLightGBM:
    def __init__(self, tab_weight=.8):
        self.tab=MedianTabPFN(); self.tab_weight=tab_weight
        self.lgb=LGBMRegressor(objective="mae",n_estimators=200,max_depth=3,
            learning_rate=.1,subsample=.8,colsample_bytree=.8,
            random_state=RANDOM_STATE,n_jobs=1,verbosity=-1)
    def fit(self,X,y): self.tab.fit(X,y); self.lgb.fit(X,y); return self
    def predict(self,X):
        return self.tab_weight*self.tab.predict(X)+(1-self.tab_weight)*self.lgb.predict(X)

BASE=dict(use_seasonality_feat=True,use_full_calendar_feat=True,
          use_periodic_copy_feat=True,use_periodic_consensus_feat=True)
TRIM=("hour","dow","dom","year")
CONFIGS={
 "I1_calendar_ablation": (dict(BASE,calendar_groups=TRIM), MedianTabPFN),
 "I2_normalized_periodic": (dict(BASE,calendar_groups=TRIM,normalize_periodic_feat=True), MedianTabPFN),
 "I3_availability_flags": (dict(BASE,calendar_groups=TRIM,normalize_periodic_feat=True,
                                  include_periodic_availability=True), MedianTabPFN),
 "I4_confidence_alpha": (dict(BASE,calendar_groups=TRIM,normalize_periodic_feat=True,
                                include_periodic_availability=True,periodic_alpha_strength=.5), MedianTabPFN),
 "I5_tabpfn_lgbm_80_20": (dict(BASE,calendar_groups=TRIM,normalize_periodic_feat=True,
                                 include_periodic_availability=True,periodic_alpha_strength=.5), TabPFNLightGBM),
}

if __name__=="__main__":
    token=os.environ.get("TABPFN_API_KEY")
    if not token:
        from tabpfn_bench_gemini import API_KEY as token
    tabpfn_client.set_access_token(token)
    only=sys.argv[1] if len(sys.argv)>1 else None
    X,y,_=load_data(); donors=[c for c in X if not str(c).startswith("holed_")]
    path="tabpfn_five_improvements_results.json"
    try: results=json.load(open(path))
    except FileNotFoundError: results={}
    for name in ([only] if only else CONFIGS):
        kwargs,factory=CONFIGS[name]; print("===",name,"===",flush=True)
        model=feb.make_v10_model(k=40,n_clusters=15,adaptive_k=True,use_euclid=True,
            make_regressor=factory,random_state=RANDOM_STATE,**kwargs)
        progress_path=f"tabpfn_{name}_progress.json"
        previous=[]
        if len(sys.argv)>2 and sys.argv[2]=="resume":
            try: previous=json.load(open(progress_path))["scores_so_far"]
            except FileNotFoundError: pass
        fresh=feb.evaluate_repeated_splits(X,y,donors,model,n_repeats=10,test_size=.1,
            random_state=RANDOM_STATE,n_val_sub=10,save_path=None,start_repeat=len(previous))
        scores=np.asarray(previous+fresh.tolist())
        with open(progress_path,"w") as f:
            json.dump({"scores_so_far":scores.tolist(),"n_repeats_done":len(scores),
                       "n_repeats_target":10},f)
        results[name]=scores.tolist()
        with open(path,"w") as f: json.dump(results,f,indent=2)
