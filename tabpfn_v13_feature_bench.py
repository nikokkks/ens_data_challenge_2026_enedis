"""Criblage apparié des 15 axes V13, puis confirmation TabPFN.

Usage:
  .venv/bin/python tabpfn_v13_feature_bench.py screen
  .venv/bin/python tabpfn_v13_feature_bench.py tabpfn [axe1,axe2,...]
"""
import json, os, sys, warnings
import numpy as np
from lightgbm import LGBMRegressor
import tabpfn_client
from tabpfn_client import TabPFNRegressor

from evaluate import load_data
import feature_engineering_bench as feb
from tabpfn_v13_features import ALL_AXES, make_v13_model

warnings.filterwarnings("ignore")
RANDOM_STATE = 42


class MedianTabPFN:
    def __init__(self): self.reg = TabPFNRegressor(random_state=RANDOM_STATE)
    def fit(self, X, y): self.reg.fit(X, y); return self
    def predict(self, X): return self.reg.predict(X, output_type="median")


def lightgbm():
    return LGBMRegressor(objective="mae", n_estimators=200, max_depth=3,
        learning_rate=.08, num_leaves=15, min_child_samples=20,
        subsample=.85, colsample_bytree=.85, random_state=RANDOM_STATE,
        n_jobs=1, verbosity=-1)


def evaluate(axes, factory, repeats, ncols, data=None, start_repeat=0):
    X, y = data if data is not None else load_data()[:2]
    donors = [c for c in X if not str(c).startswith("holed_")]
    model = make_v13_model(factory, axes=axes)
    return feb.evaluate_repeated_splits(X, y, donors, model, n_repeats=repeats,
        test_size=.1, random_state=RANDOM_STATE, n_val_sub=ncols,
        start_repeat=start_repeat)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "screen"
    path = "tabpfn_v13_feature_results.json"
    try: results = json.load(open(path))
    except FileNotFoundError: results = {}
    X, y, _ = load_data()
    data = (X, y)
    if mode == "screen":
        configs = {"V12_baseline": ()} | {f"A{i:02d}_{a}": (a,)
            for i, a in enumerate(ALL_AXES, 1)}
        for name, axes in configs.items():
            print(f"=== {name}: {axes} ===", flush=True)
            scores = evaluate(axes, lightgbm, repeats=3, ncols=20, data=data)
            results[name] = {"model": "LightGBM screen 3x20", "axes": list(axes),
                             "scores": scores.tolist(), "mean": float(scores.mean())}
            with open(path, "w") as f: json.dump(results, f, indent=2)
    elif mode == "tabpfn":
        token = os.environ.get("TABPFN_API_KEY")
        if not token:
            from tabpfn_bench_gemini import API_KEY as token
        tabpfn_client.set_access_token(token)
        axes = tuple(filter(None, sys.argv[2].split(","))) if len(sys.argv) > 2 else ()
        name = "TabPFN_" + ("+".join(axes) if axes else "V12_baseline")
        scores = evaluate(axes, MedianTabPFN, repeats=3, ncols=10, data=data)
        results[name] = {"model": "TabPFN median 3x10", "axes": list(axes),
                         "scores": scores.tolist(), "mean": float(scores.mean())}
        with open(path, "w") as f: json.dump(results, f, indent=2)
    elif mode == "confirm_v13":
        token = os.environ.get("TABPFN_API_KEY")
        if not token:
            from tabpfn_bench_gemini import API_KEY as token
        tabpfn_client.set_access_token(token)
        name = "TabPFN_neighbor_slopes_10x10"
        source = results.get(name, results.get("TabPFN_neighbor_slopes", {}))
        previous = list(source.get("scores", []))
        if len(previous) > 10:
            previous = previous[:10]
        print(f"Reprise V13 : {len(previous)}/10 repeats déjà présents", flush=True)
        for start in range(len(previous), 10):
            fresh = evaluate(("neighbor_slopes",), MedianTabPFN, repeats=start+1,
                             ncols=10, data=data, start_repeat=start)
            previous.extend(fresh.tolist())
            scores = np.asarray(previous)
            results[name] = {
                "model": "TabPFN median 10x10",
                "axes": ["neighbor_slopes"],
                "scores": previous,
                "mean": float(scores.mean()),
                "std": float(scores.std()),
                "n_repeats_done": len(previous),
            }
            with open(path, "w") as f: json.dump(results, f, indent=2)
            print(f"Checkpoint V13 {len(previous)}/10 : "
                  f"MAE={scores.mean():.3f} ± {scores.std():.3f}", flush=True)
    else:
        raise SystemExit("mode attendu: screen, tabpfn ou confirm_v13")
