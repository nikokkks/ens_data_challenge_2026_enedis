"""Fine-tuning du meilleur modèle "après TabPFN" (CatBoost, 69.43 W en
référence rapide, cf. benchmark.md §5) sur GPU, avec :

  1. TOUTES les features de voisinage pertinentes (pas de cap arbitraire à
     40/100) : k_max est ouvert (KMAX_NEIGHBORS très grand), la vraie
     sélection se fait par seuil de corrélation (corr_threshold), donc chaque
     colonne récupère autant de voisins que réellement corrélés (mesuré :
     ~400 voisins pour une colonne typique avec le seuil par défaut 0.15).
  2. Réglage d'hyperparamètres CatBoost (depth, learning_rate, iterations,
     l2_leaf_reg, bagging_temperature) par recherche aléatoire, écran rapide
     puis confirmation sur un budget plus large (cf. conseil #5 : ne pas
     décider sur un protocole trop bruité).
  3. GPU explicitement activé pour CatBoost (mesuré ~2x plus rapide que CPU
     à ~300 features / ~950 lignes sur ce poste, cf. essai préalable) --
     fit strictement séquentiel (un seul GPU, pas de parallélisme thread).

TabPFN reste hors d'atteinte ici (pas d'accès réseau dans ce sandbox) --
CatBoost est donc utilisé comme le meilleur modèle réellement testable
localement, avec un pipeline de features identique à V12/BEST_FEATURES.
"""
import json
import sys
import time
import warnings

import numpy as np

from evaluate import load_data
import feature_engineering_bench as feb

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
RESULTS_PATH = "catboost_gpu_finetune_results.json"

# Cap "large" plutôt qu'illimité : au-delà de quelques centaines de voisins
# on ne fait qu'ajouter du bruit/coût sans signal (les scores de corrélation
# chutent vite après le pic, cf. mesure préalable : 421 voisins > seuil 0.15
# pour une colonne type, 5 seulement > seuil 0.30). 500 couvre la queue
# entière observée en pratique sans dégénérer en un GEMM à 20000 colonnes.
KMAX_NEIGHBORS = 500

BEST_FEATURES = dict(
    use_seasonality_feat=True, use_full_calendar_feat=True,
    calendar_groups=("hour", "dow", "dom", "year"),
    use_periodic_copy_feat=True, use_periodic_consensus_feat=True,
    normalize_periodic_feat=True, include_periodic_availability=True,
    periodic_alpha_strength=.5,
)


class SafeCatBoostGPU:
    """cf. xgboost_bench_round2.SafeCatBoost : CatBoost plante sur les cibles
    constantes (paliers nocturnes à 0). Repli déterministe sur la constante."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y):
        if np.all(y == y[0]):
            self._constant = float(y[0])
            self.reg = None
        else:
            self._constant = None
            from catboost import CatBoostRegressor
            self.reg = CatBoostRegressor(**self.kwargs)
            self.reg.fit(X, y)
        return self

    def predict(self, X):
        if self.reg is None:
            return np.full(X.shape[0], self._constant)
        return self.reg.predict(X)


def make_catboost_factory(depth, learning_rate, iterations, l2_leaf_reg,
                           bagging_temperature, use_gpu):
    def factory():
        kw = dict(
            iterations=iterations, depth=depth, learning_rate=learning_rate,
            loss_function="MAE", l2_leaf_reg=l2_leaf_reg,
            bagging_temperature=bagging_temperature,
            random_state=RANDOM_STATE, verbose=False,
            allow_writing_files=False,
        )
        if use_gpu:
            kw["task_type"] = "GPU"
            kw["devices"] = "0"
        else:
            kw["thread_count"] = 1
        return SafeCatBoostGPU(**kw)
    return factory


def build_model(regressor_factory, k=KMAX_NEIGHBORS, corr_threshold=0.15):
    return feb.make_v10_model(
        k=k, n_clusters=15, adaptive_k=True, use_euclid=True,
        make_regressor=regressor_factory, random_state=RANDOM_STATE,
        **BEST_FEATURES,
    )


CONFIGS = {
    # baseline de référence : config existante (69.43 W, k=40) pour vérifier
    # que k élargi + mêmes hyperparamètres n'est pas déjà une régression.
    "baseline_k40_ref": dict(depth=4, learning_rate=.06, iterations=250,
                              l2_leaf_reg=3.0, bagging_temperature=1.0, k=40),
    "allfeat_k500_ref_hp": dict(depth=4, learning_rate=.06, iterations=250,
                                 l2_leaf_reg=3.0, bagging_temperature=1.0, k=KMAX_NEIGHBORS),
    "allfeat_deep6": dict(depth=6, learning_rate=.05, iterations=300,
                           l2_leaf_reg=3.0, bagging_temperature=1.0, k=KMAX_NEIGHBORS),
    "allfeat_deep8_reg": dict(depth=8, learning_rate=.04, iterations=400,
                               l2_leaf_reg=6.0, bagging_temperature=0.5, k=KMAX_NEIGHBORS),
    "allfeat_shallow_fastlr": dict(depth=4, learning_rate=.1, iterations=200,
                                    l2_leaf_reg=1.0, bagging_temperature=1.0, k=KMAX_NEIGHBORS),
    "allfeat_depth5_l2_10": dict(depth=5, learning_rate=.05, iterations=350,
                                  l2_leaf_reg=10.0, bagging_temperature=0.3, k=KMAX_NEIGHBORS),
    "allfeat_depth6_bt2": dict(depth=6, learning_rate=.03, iterations=500,
                                l2_leaf_reg=5.0, bagging_temperature=2.0, k=KMAX_NEIGHBORS),
}


def run_config(name, cfg, X, y, donors, use_gpu, n_repeats, n_val_sub):
    factory = make_catboost_factory(
        depth=cfg["depth"], learning_rate=cfg["learning_rate"],
        iterations=cfg["iterations"], l2_leaf_reg=cfg["l2_leaf_reg"],
        bagging_temperature=cfg["bagging_temperature"], use_gpu=use_gpu,
    )
    model = build_model(factory, k=cfg["k"])
    t0 = time.time()
    scores = feb.evaluate_repeated_splits(
        X, y, donors, model, n_repeats=n_repeats, test_size=.1,
        random_state=RANDOM_STATE, n_val_sub=n_val_sub,
        save_path=f"catboost_gpu_finetune_{name}_progress.json",
    )
    dt = time.time() - t0
    return {"scores": scores.tolist(), "mean": float(scores.mean()),
            "std": float(scores.std()), "seconds": dt, "config": cfg,
            "gpu": use_gpu, "n_repeats": n_repeats, "n_val_sub": n_val_sub}


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "screen"
    use_gpu = "--cpu" not in sys.argv

    print("Chargement des données...", flush=True)
    X, y, _ = load_data()
    donors = [c for c in X if not str(c).startswith("holed_")]

    try:
        results = json.load(open(RESULTS_PATH))
    except FileNotFoundError:
        results = {}

    if stage == "screen":
        # Écran rapide : budget réduit (3 repeats x 15 colonnes) pour trier
        # les configs candidates avant de dépenser du temps GPU sur toutes.
        for name, cfg in CONFIGS.items():
            print(f"=== SCREEN {name} (gpu={use_gpu}) ===", flush=True)
            res = run_config(name, cfg, X, y, donors, use_gpu,
                              n_repeats=3, n_val_sub=15)
            print(f"  -> {res['mean']:.2f} +/- {res['std']:.2f} W "
                  f"({res['seconds']:.0f}s)", flush=True)
            results[f"screen_{name}"] = res
            json.dump(results, open(RESULTS_PATH, "w"), indent=2)

    elif stage == "confirm":
        # Confirmation sur budget plus large (conseil #5) pour la/les
        # meilleure(s) config(s) issues de l'écran, passée(s) en argv.
        names = sys.argv[2:] if len(sys.argv) > 2 else list(CONFIGS)
        names = [n for n in names if not n.startswith("--")]
        for name in names:
            cfg = CONFIGS[name]
            print(f"=== CONFIRM {name} (gpu={use_gpu}) ===", flush=True)
            res = run_config(f"confirm_{name}", cfg, X, y, donors, use_gpu,
                              n_repeats=8, n_val_sub=30)
            print(f"  -> {res['mean']:.2f} +/- {res['std']:.2f} W "
                  f"({res['seconds']:.0f}s)", flush=True)
            results[f"confirm_{name}"] = res
            json.dump(results, open(RESULTS_PATH, "w"), indent=2)

    print("Terminé. Résultats dans", RESULTS_PATH)
