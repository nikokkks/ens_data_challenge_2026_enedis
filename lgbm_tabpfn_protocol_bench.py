"""Best fast pre-existing model evaluated on the exact TabPFN protocol."""
import json
from lightgbm import LGBMRegressor
from evaluate import load_data
import feature_engineering_bench as feb

RANDOM_STATE = 42
PARAMS = dict(n_estimators=200, max_depth=3, learning_rate=0.1,
              subsample=0.8, colsample_bytree=0.8, objective="mae",
              n_jobs=1, verbosity=-1)

if __name__ == "__main__":
    import sys
    variant = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    full_scale = len(sys.argv) > 2 and sys.argv[2] == "full"
    use_periodic = variant in ("periodic", "consensus", "full_calendar")
    use_consensus = variant in ("consensus", "full_calendar")
    use_full_calendar = variant == "full_calendar"
    X, y, _ = load_data()
    donors = [c for c in X if not str(c).startswith("holed_")]
    model = feb.make_v10_model(
        k=40, n_clusters=15, adaptive_k=True, use_euclid=True,
        use_seasonality_feat=True,
        use_periodic_copy_feat=use_periodic,
        use_periodic_consensus_feat=use_consensus,
        use_full_calendar_feat=use_full_calendar,
        make_regressor=lambda: LGBMRegressor(random_state=RANDOM_STATE, **PARAMS),
        random_state=RANDOM_STATE)
    scores = feb.evaluate_repeated_splits(
        X, y, donors, model, n_repeats=(5 if full_scale else 10), test_size=0.10,
        random_state=RANDOM_STATE, n_val_sub=(None if full_scale else 10),
        save_path=(f"lgbm_{variant}_progress.json" if use_periodic else
                   "lgbm_tabpfn_protocol_progress.json"))
    path = "lgbm_tabpfn_protocol_results.json"
    try:
        results = json.load(open(path))
    except FileNotFoundError:
        results = {}
    key = ({"periodic": "E6_plus_periodic_copies",
            "consensus": "E6_plus_periodic_copies_and_consensus"}.get(
                variant, "E6_LightGBM_10x10"))
    if variant == "full_calendar":
        key = "E6_plus_periodic_consensus_full_calendar"
    if full_scale:
        key += "_5x100"
    results[key] = scores.tolist()
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
