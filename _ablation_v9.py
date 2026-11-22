"""Ablation des 2 pistes candidates pour V9 (distance euclidienne, alpha
modulé par volatilité), sur 3 splits (Ridge, gratuit/rapide) avant de les
valider ou non. Résultats -> benchmark.md § V9."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.linear_model import Ridge

from evaluate import load_data
import feature_engineering_bench as feb

X_train, y_train, X_test = load_data()
donor_cols = [c for c in X_train.columns if not str(c).startswith("holed_")]

configs = {
    "V8 (référence)": dict(use_euclid=False, use_volatility=False),
    "+euclid only": dict(use_euclid=True, use_volatility=False),
    "+volatility only": dict(use_euclid=False, use_volatility=True),
    "V9 (both)": dict(use_euclid=True, use_volatility=True),
}

for name, cfg in configs.items():
    model = feb.make_v9_model(k=20, n_clusters=25, adaptive_k=True,
                               make_regressor=lambda: Ridge(alpha=1.0), **cfg)
    print(f"--- {name} ---")
    scores = feb.evaluate_repeated_splits(X_train, y_train, donor_cols, model,
                                           n_repeats=3, test_size=0.10, random_state=42, n_val_sub=10)
