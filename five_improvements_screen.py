"""Criblage LightGBM rapide des cinq améliorations proposées après C2."""
import json, sys
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from evaluate import load_data
import feature_engineering_bench as feb

RANDOM_STATE = 42
BASE = dict(use_seasonality_feat=True, use_full_calendar_feat=True,
            use_periodic_copy_feat=True, use_periodic_consensus_feat=True)
CONFIGS = {
    "A1_calendar_hour_dow_dom_year": dict(BASE, calendar_groups=("hour","dow","dom","year")),
    "A1b_calendar_hour_dow_dom": dict(BASE, calendar_groups=("hour","dow","dom")),
    "A2_normalized_periodic": dict(BASE, normalize_periodic_feat=True),
    "A3_availability_flags": dict(BASE, include_periodic_availability=True),
    "A23_norm_and_flags": dict(BASE, normalize_periodic_feat=True, include_periodic_availability=True),
    "A4_alpha_conf_025": dict(BASE, normalize_periodic_feat=True,
                               include_periodic_availability=True, periodic_alpha_strength=.25),
    "A4_alpha_conf_050": dict(BASE, normalize_periodic_feat=True,
                               include_periodic_availability=True, periodic_alpha_strength=.50),
}

class LGBMRidgeEnsemble:
    def __init__(self, lgb_weight=.8):
        self.lgb=factory(); self.ridge=Ridge(alpha=10.0); self.w=lgb_weight
    def fit(self,X,y):
        self.lgb.fit(X,y); self.ridge.fit(np.nan_to_num(X,nan=0.0),y); return self
    def predict(self,X):
        return self.w*self.lgb.predict(X)+(1-self.w)*self.ridge.predict(np.nan_to_num(X,nan=0.0))

def factory():
    return LGBMRegressor(objective="mae", n_estimators=200, max_depth=3,
                         learning_rate=.1, subsample=.8, colsample_bytree=.8,
                         random_state=RANDOM_STATE, n_jobs=1, verbosity=-1)

if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv)>1 else None
    X,y,_=load_data(); donors=[c for c in X if not str(c).startswith("holed_")]
    path="five_improvements_screen_results.json"
    try: results=json.load(open(path))
    except FileNotFoundError: results={}
    for name in ([only] if only else CONFIGS):
        print("===",name,"===",flush=True)
        reg_factory = LGBMRidgeEnsemble if name == "A5_lgbm_ridge_80_20" else factory
        kwargs = (dict(BASE,calendar_groups=("hour","dow","dom","year"),
                       normalize_periodic_feat=True,include_periodic_availability=True,
                       periodic_alpha_strength=.5) if name == "A5_lgbm_ridge_80_20"
                  else CONFIGS[name])
        model=feb.make_v10_model(k=40,n_clusters=15,adaptive_k=True,use_euclid=True,
                                 make_regressor=reg_factory,random_state=RANDOM_STATE,
                                 **kwargs)
        scores=feb.evaluate_repeated_splits(X,y,donors,model,n_repeats=10,test_size=.1,
                  random_state=RANDOM_STATE,n_val_sub=10,save_path=f"screen_{name}_progress.json")
        results[name]=scores.tolist()
        with open(path,"w") as f: json.dump(results,f,indent=2)
