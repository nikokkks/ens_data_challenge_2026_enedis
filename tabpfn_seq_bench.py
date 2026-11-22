"""TabPFNRegressor évalué EXACTEMENT comme le réseau séquentiel (§ 7ter).

Objectif : mesurer ce que TabPFN obtient à information ÉGALE avec le
LSTM/Transformer, c'est-à-dire en information purement INTRA-COURBE (aucune
courbe voisine), puis en rajoutant les blocs d'information un par un. On
isole ainsi ce que rapporte chaque source d'information plutôt que de
comparer deux modèles qui ne voient pas la même chose.

Attention à un point de conception : on ne peut PAS donner la valeur masquée
brute en feature à un régresseur tabulaire ligne-à-ligne. Le LSTM le peut
(il est séquentiel : à une position masquée la valeur d'entrée est nulle et
l'information vient des voisins temporels), mais pour TabPFN la ligne
d'entraînement contiendrait sa propre cible -> fuite triviale. Les canaux
équivalents côté tabulaire sont donc l'interpolation + le calendrier.

MEMES DEUX NIVEAUX DE SPLIT que `lstm_bench.py` :

1. Validation TEMPORELLE interne (sélection de variante) : masques
   artificiels tirés de la distribution réelle des trous sur des courbes
   DONNEUSES, notation restreinte aux points masqués de `[T_SPLIT, 1057)`
   (25 -> 31 janvier), MAE normalisée par l'échelle de la courbe -> le
   chiffre est directement comparable au `val_mae_norm` du sweep LSTM.

   Nuance à garder en tête : le split temporel bride réellement le LSTM
   (ses poids ne sont ajustés que sur `[0, 792)`), alors que TabPFN n'a
   aucun poids à entraîner -- il apprend en contexte sur les points observés
   de la courbe. TabPFN n'est donc pas pénalisé par ce split ; la
   comparaison est à l'avantage de TabPFN et c'est assumé.

2. Split par COLONNES `10x10` (mesure officielle), `random_state=42`,
   variantes `standalone` et `blend_alpha_interp`, exactement comme
   `lstm_official_eval.py`.
"""

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import tabpfn_client
from tabpfn_client import TabPFNRegressor

from evaluate import load_data, official_mae, fill_nan_with_interpolation
import feature_engineering_bench as feb
import lstm_bench as lb

RANDOM_STATE = 42
T_SPLIT = lb.T_SPLIT
N_REPEATS = 10
TEST_SIZE = 0.10
N_VAL_SUB = 10
N_VAL_CURVES = 20          # courbes donneuses pour la validation temporelle

VAL_PATH = "tabpfn_seq_val_results.json"
OFFICIAL_PATH = "tabpfn_seq_official_results.json"


def median_tabpfn():
    class _M:
        def __init__(self):
            self.reg = TabPFNRegressor(random_state=RANDOM_STATE)

        def fit(self, X, y):
            self.reg.fit(X, y)
            return self

        def predict(self, X):
            return self.reg.predict(X, output_type="median")
    return _M()


# ---------------------------------------------------------------------------
# Blocs d'information, cumulatifs (Q1 = strict équivalent des canaux du LSTM)
# ---------------------------------------------------------------------------

VARIANTS = {
    "Q1_lstm_equivalent":  dict(calendar=False, periodic=False, seasonality=False),
    "Q2_full_calendar":    dict(calendar=True,  periodic=False, seasonality=False),
    "Q3_periodic_copies":  dict(calendar=True,  periodic=True,  seasonality=False),
    "Q4_own_seasonality":  dict(calendar=True,  periodic=True,  seasonality=True),
}


def build_features(values, index, opts):
    """`values` : courbe AVEC NaN aux positions masquées. Toutes les features
    ne dépendent que de points réellement observés (ou du calendrier)."""
    s = pd.Series(values, index=index)
    interp = fill_nan_with_interpolation(s).values[:, None]
    if opts["calendar"]:
        tf = feb.time_features_full_seasonality(index, ("hour", "dow", "dom", "year"))
    else:
        # strictement les canaux calendaires du LSTM : sin/cos heure + sin/cos jour
        tf = lb.time_channels(index)
    cols = [interp, tf]
    if opts["periodic"]:
        cols.append(feb.periodic_copy_features(values, include_summary=True,
                                               normalize=True, include_availability=True))
    if opts["seasonality"]:
        cols.append(feb.same_curve_seasonality_feature(values, index))
    return np.hstack(cols)


def predict_curve(values, index, opts):
    """Complète une courbe : TabPFN entraîné sur les lignes OBSERVEES de la
    courbe elle-même, prédit sur les lignes manquantes. Aucune courbe
    voisine n'intervient (c'est tout l'objet de la comparaison)."""
    missing = np.isnan(values)
    obs_idx, miss_idx = np.where(~missing)[0], np.where(missing)[0]
    X = build_features(values, index, opts)
    if len(obs_idx) < 15:
        return fill_nan_with_interpolation(pd.Series(values, index=index)).values[miss_idx]
    reg = median_tabpfn()
    reg.fit(X[obs_idx], values[obs_idx])
    return reg.predict(X[miss_idx])


# ---------------------------------------------------------------------------
# 1. Validation TEMPORELLE interne
# ---------------------------------------------------------------------------

def temporal_validation(name, opts, curves_raw, index):
    rng = np.random.RandomState(RANDOM_STATE + 1000)
    errs = []
    for c in range(curves_raw.shape[0]):
        values = curves_raw[c].astype(np.float64).copy()
        mask = lb.sample_mask((1, len(values)), rng)[0]
        scored = mask & (np.arange(len(values)) >= T_SPLIT)
        if not scored.any():
            continue
        truth = values.copy()
        values[mask] = np.nan
        pred_missing = predict_curve(values, index, opts)
        miss_idx = np.where(mask)[0]
        pred_full = np.full(len(values), np.nan)
        pred_full[miss_idx] = pred_missing
        _, scale = lb.robust_scale(truth[~mask][None, :].astype(np.float32))
        errs.append(np.abs(pred_full[scored] - truth[scored]).mean() / scale.item())
    return float(np.mean(errs))


# ---------------------------------------------------------------------------
# 2. Protocole officiel 10x10 (identique à lstm_official_eval.py)
# ---------------------------------------------------------------------------

def official_eval(name, opts, X_train, y_train):
    """Les deux variantes partagent EXACTEMENT les mêmes prédictions TabPFN
    (seul le mélange final diffère) : on ne paie donc les appels API qu'une
    seule fois et les deux colonnes du tableau restent strictement
    appariées."""
    from sklearn.model_selection import train_test_split

    baseline = json.load(open("advanced_regressors_results.json"))["R0_lightgbm_reference"]
    tabpfn_i4 = json.load(open("tabpfn_five_improvements_results.json"))["I4_confidence_alpha"]
    lstm = json.load(open("lstm_official_results.json"))
    lstm_key = [k for k in lstm if k.endswith("blend_alpha_interp")][0]

    scores = {"standalone": [], "blend_alpha_interp": []}
    holed_cols = list(y_train.columns)
    for i in range(N_REPEATS):
        seed_i = RANDOM_STATE + i
        dev_cols, val_full = train_test_split(holed_cols, test_size=TEST_SIZE,
                                              random_state=seed_i)
        val_cols = list(np.random.RandomState(seed_i).choice(val_full, size=N_VAL_SUB,
                                                             replace=False))
        t0 = time.time()
        preds = {"standalone": {}, "blend_alpha_interp": {}}
        for col in val_cols:
            s = X_train[col]
            missing = s.isna().values
            if not missing.any():
                preds["standalone"][col] = preds["blend_alpha_interp"][col] = s
                continue
            miss_idx = np.where(missing)[0]
            raw = predict_curve(s.values.astype(np.float64), X_train.index, opts)

            interp = fill_nan_with_interpolation(s).values[miss_idx]
            run_len = feb.run_lengths_for(missing)
            edge = feb.edge_distance_for(missing)
            alphas = np.array([feb.continuous_alpha(run_len[m], edge[m], adaptive=True)
                               for m in miss_idx])
            for variant, values in (("standalone", raw),
                                    ("blend_alpha_interp", alphas * interp + (1 - alphas) * raw)):
                filled = s.copy()
                filled.iloc[miss_idx] = values
                preds[variant][col] = filled

        line = []
        for variant in scores:
            y_pred = pd.DataFrame(preds[variant], index=X_train.index)
            mae = official_mae(y_train[val_cols], y_pred, X_train[val_cols].isna())
            scores[variant].append(mae)
            line.append(f"{variant[:5]}={mae:7.2f}")
        print(f"  repeat {i+1}/{N_REPEATS}: {' '.join(line)} W | LSTM blend = "
              f"{lstm[lstm_key][i]:7.2f} | R0_lgbm = {baseline[i]:7.2f}"
              f" | TabPFN I4 = {tabpfn_i4[i]:7.2f}  ({time.time()-t0:.0f}s)", flush=True)

    for variant, vals in scores.items():
        v = np.array(vals)
        print(f"\n  TabPFN seq {name} / {variant:19s} : {v.mean():.2f} +/- {v.std():.2f} W")
    print(f"  LSTM/Transformer blend (même info)     : {np.mean(lstm[lstm_key]):.2f} W")
    print(f"  R0 LightGBM (avec voisins)             : {np.mean(baseline):.2f} W")
    print(f"  TabPFN I4 (pipeline complet)           : {np.mean(tabpfn_i4):.2f} W\n", flush=True)
    return {k: np.array(v) for k, v in scores.items()}


if __name__ == "__main__":
    token = os.environ.get("TABPFN_API_KEY")
    if not token:
        from tabpfn_bench_gemini import API_KEY as token
    tabpfn_client.set_access_token(token)

    stage = sys.argv[1] if len(sys.argv) > 1 else "val"
    print("Chargement des données...", flush=True)
    X_train, y_train, _ = load_data()

    if stage == "val":
        donor_cols = [c for c in X_train.columns if not str(c).startswith("holed_")]
        rng = np.random.RandomState(RANDOM_STATE)
        picked = list(rng.choice(donor_cols, size=N_VAL_CURVES, replace=False))
        curves_raw = X_train[picked].ffill().bfill().values.T
        print(f"Validation temporelle sur {N_VAL_CURVES} courbes donneuses, "
              f"notation sur les points masqués de [{T_SPLIT}, 1057)\n", flush=True)
        try:
            results = json.load(open(VAL_PATH))
        except FileNotFoundError:
            results = {}
        for name, opts in VARIANTS.items():
            if name in results:
                print(f"=== {name} déjà fait ===", flush=True)
                continue
            print(f"=== {name} : {opts} ===", flush=True)
            t0 = time.time()
            v = temporal_validation(name, opts, curves_raw, X_train.index)
            results[name] = {"val_mae_norm": v, "opts": opts, "seconds": time.time() - t0}
            with open(VAL_PATH, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  -> val MAE (normalisée) = {v:.4f}  ({time.time()-t0:.0f}s)\n", flush=True)
    else:
        name = sys.argv[2]
        opts = VARIANTS[name]
        try:
            results = json.load(open(OFFICIAL_PATH))
        except FileNotFoundError:
            results = {}
        print(f"=== TabPFN seq {name} (10x10 officiel) ===", flush=True)
        out = official_eval(name, opts, X_train, y_train)
        for variant, s in out.items():
            results[f"{name}_{variant}"] = s.tolist()
        with open(OFFICIAL_PATH, "w") as f:
            json.dump(results, f, indent=2)
