"""Évaluation appariée, protocole officiel 10x10, TabPFN v14 vs SAITS (PyPOTS).

Contexte
--------
La discussion jointe recommande la famille des "transformers masqués
bidirectionnels spécialisés imputation" (SAITS/M-SAITS via PyPOTS) comme
meilleur candidat "modèle profond" pour ce problème, en s'appuyant sur un
benchmark quasi identique (UCI Electricity Load Diagrams, 370 clients, MAE) où
SAITS bat BRITS/GP-VAE/M-RNN/Transformer classique.

ATTENTION (lire benchmark.md §7ter avant de lancer) : un modèle
séquentiel global (LSTM/GRU/Transformer, une courbe = un canal, PAS
d'attention inter-courbes) a déjà été testé sur ce protocole EXACT et perd
nettement contre TabPFN v14 :

    | Modèle                                    | MAE (W) | std   |
    |--------------------------------------------|--------:|------:|
    | LSTM/Transformer + mélange alpha-interp.    |   74,66 | 36,30 |
    | R0 LightGBM (features V12, voisins)         |   69,75 | 31,80 |
    | TabPFN v14 (BEST confirmé)                  |   66,24 | 32,45 |

Conclusion du repo à ce sujet : un modèle "global" ne voit pas l'information
INTER-courbes qui fait la performance de TabPFN/KNN ici (in-context learning
propre à chaque colonne cible). SAITS diffère architecturalement de ce
Transformer déjà rejeté : son attention porte aussi sur l'axe des FEATURES
(chaque courbe = une feature, comme les 370 clients UCI), donc il peut en
principe capter cette information inter-courbes -- c'est pour ça qu'il reste
le candidat le plus prometteur de la discussion (devant Chronos, du pur
forecasting, et devant CSDI, qui optimise une vraisemblance et non la MAE).
Mais il reste proche en esprit de la complétion bas-rang déjà testée et
perdante ici (SVD/NMF, §7 A2 : 70-84 W, "un pool global dilue la spécificité
de chaque courbe") -- SAITS n'en est qu'une variante non linéaire. D'où ce
script : vérifier empiriquement plutôt que de se fier à l'a priori.

Protocole (strictement comparable à tabpfn_v14_confirm.py, 66,24 +/- 32,45 W)
------------------------------------------------------------------------
Pour i in 0..9 : seed_i = 42 + i,
    dev_cols, val_cols_full = train_test_split(holed_cols, test_size=.1,
                                                random_state=seed_i)
    val_cols = RandomState(seed_i).choice(val_cols_full, size=10, replace=False)
-- BIT-IDENTIQUE à tabpfn_v14_ablation.run_config(..., n_repeats=10,
n_val_sub=10), donc aux mêmes seeds/colonnes que la table validée. TabPFN et
SAITS sont évalués sur les MÊMES val_cols à chaque repeat : comparaison
appariée par repeat ET par colonne (pas seulement sur la moyenne), comme
l'exige CLAUDE.md pour les comparaisons TabPFN.

Zéro fuite : SAITS ne voit jamais y_train pour les val_cols, seulement leur
vrai masque de trous (X_train). Features SAITS = les val_cols (avec leurs
vrais trous) + jusqu'à MAX_FEATURES donneurs les plus corrélés (union par
colonne, feb.adaptive_top_k_corr_euclid -- même sélection que la KNN de
TabPFN v14), bornage nécessaire pour que l'attention sur l'axe features reste
traitable (même ordre de grandeur que le benchmark UCI Electricity cité : 370
clients).

Usage
-----
    .venv/bin/pip install pypots     # ajoute pypots (torch déjà présent)

    # sanity check rapide (~1-3 min, petite config, 2 repeats) :
    SAITS_PROFILE=smoke .venv/bin/python claude2/saits_v14_paired_eval.py

    # protocole officiel complet (10x10, config proche du papier SAITS) :
    SAITS_PROFILE=full .venv/bin/python claude2/saits_v14_paired_eval.py

Resumable comme les autres scripts du repo : progress JSON écrit après
chaque repeat, relancer le script reprend où il s'est arrêté.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
import tabpfn_client

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

from evaluate import load_data
import feature_engineering_bench as feb
import tabpfn_v14_ablation as v14

PROGRESS_PATH = os.path.join(HERE, "saits_v14_paired_progress.json")
RESULTS_PATH = os.path.join(HERE, "saits_v14_paired_results.json")

# Config TabPFN v14 = BEST confirmée (tabpfn_v14_confirm.py, 66,24 +/- 32,45 W)
V14_BEST = dict(volatility_alpha=True, vol_weight=0.8, periodic_alpha_strength=1.0,
                 extra_lags=(24,), neighbor_rolling=True)

PROFILE = os.environ.get("SAITS_PROFILE", "smoke")
N_REPEATS = int(os.environ.get("SAITS_N_REPEATS", 2 if PROFILE == "smoke" else 10))
N_VAL_SUB = 10  # colonnes de validation par repeat, comme run_config(n_val_sub=10)

_SMOKE = dict(
    k_per_col=10, max_features=60, window_len=96, stride=48,
    n_layers=1, d_model=64, n_heads=2, d_k=32, d_v=32, d_ffn=64, dropout=0.1,
    epochs=5, patience=None, batch_size=8, device=None,
)
_FULL = dict(
    # ordre de grandeur de la config "papier" SAITS (2 couches, d_model=256,
    # 4 têtes) trouvée dans la doc PyPOTS ; k_per_col=40/max_features=400
    # bornent l'axe features à l'échelle du benchmark UCI Electricity (370
    # clients) cité dans la discussion -- au-delà, l'attention sur l'axe
    # features devient hors de portée en temps raisonnable.
    k_per_col=40, max_features=400, window_len=336, stride=48,
    n_layers=2, d_model=256, n_heads=4, d_k=64, d_v=64, d_ffn=128, dropout=0.1,
    epochs=200, patience=None, batch_size=16, device=None,
)
SAITS_OPTS = _SMOKE if PROFILE == "smoke" else _FULL


# ---------------------------------------------------------------------------
# Côté TabPFN v14 : copie de tabpfn_v14_ablation.eval_split, mais renvoie les
# erreurs absolues PAR COLONNE (pas seulement la moyenne agrégée du repeat),
# pour permettre l'appariement colonne-par-colonne avec SAITS ci-dessous.
# ---------------------------------------------------------------------------

def eval_tabpfn_repeat(X_train, y_train, donor_cols, dev_cols, val_cols, opts):
    donor_df = X_train[donor_cols]
    dev_true_df = y_train[dev_cols]
    val_holed_df = X_train[val_cols]

    pool = pd.concat([donor_df, dev_true_df], axis=1)
    pool_vals = pool.ffill().bfill().to_numpy(dtype=np.float64)
    km, donor_labels = feb.fit_shape_clusters(donor_df, n_clusters=15, random_state=v14.RANDOM_STATE)
    dev_filled = dev_true_df.ffill().bfill()
    Zdev = (dev_filled - dev_filled.mean()) / (dev_filled.std() + 1e-9)
    dev_labels = pd.Series(km.predict(Zdev.T.values), index=dev_true_df.columns)
    pool_labels = pd.concat([donor_labels, dev_labels])
    pos = pd.Series(np.arange(len(pool_labels)), index=pool_labels.index)
    cluster_col_idx = {cl: pos[idx].values for cl, idx in pool_labels.groupby(pool_labels).groups.items()}
    tfeat = feb.time_features_full_seasonality(val_holed_df.index, ("hour", "dow", "dom", "year"))

    abs_err_by_col = {}
    with ThreadPoolExecutor(max_workers=v14.MAX_WORKERS) as executor:
        futures = [executor.submit(v14.complete_column_v14, c, val_holed_df[c], y_train[c],
                                    pool_vals, km, cluster_col_idx, tfeat,
                                    val_holed_df.index, opts) for c in val_cols]
        for fut in as_completed(futures):
            col, abs_err, error = fut.result()
            if abs_err is not None:
                abs_err_by_col[col] = abs_err
            if error:
                print(f"  [tabpfn] repli sur {col}: {error}", flush=True)
    return abs_err_by_col


# ---------------------------------------------------------------------------
# Côté SAITS : matrice (temps x courbes) sur un pool de donneurs corrélés +
# les val_cols avec leurs VRAIS trous, fenêtrée pour PyPOTS, imputée, puis
# reconvertie à l'échelle d'origine (Watts) pour comparer à y_train.
# ---------------------------------------------------------------------------

def select_saits_features(X_train, val_cols, donor_cols, k_per_col, max_features,
                            corr_threshold=0.15, euclid_weight=0.3):
    """Union, sur les val_cols du repeat, des k_per_col donneurs les plus
    corrélés (même sélection que le KNN de TabPFN v14 :
    feb.adaptive_top_k_corr_euclid), plafonnée à max_features pour que
    l'attention SAITS sur l'axe features reste traitable."""
    donor_vals = X_train[donor_cols].ffill().bfill().to_numpy(dtype=np.float64)
    chosen_idx, seen = [], set()
    for col in val_cols:
        s = X_train[col]
        obs_idx = np.flatnonzero(s.notna().to_numpy())
        if len(obs_idx) < 5:
            continue
        target_obs = s.to_numpy()[obs_idx]
        idx, _ = feb.adaptive_top_k_corr_euclid(
            target_obs, donor_vals[obs_idx, :], k_min=5, k_max=k_per_col,
            corr_threshold=corr_threshold, euclid_weight=euclid_weight)
        for j in idx:
            if j not in seen:
                seen.add(j)
                chosen_idx.append(j)
        if len(chosen_idx) >= max_features:
            break
    return [donor_cols[j] for j in chosen_idx[:max_features]]


def robust_scale_matrix(mat):
    """Z-score robuste PAR COLONNE (par courbe), ajusté uniquement sur les
    valeurs observées (nanmedian/nanmedian ignore les NaN) -- nécessaire
    avant SAITS car les courbes vivent à des échelles de puissance très
    différentes ; sans ça l'attention sur l'axe features serait dominée par
    les courbes de plus forte amplitude. Aucune fuite : ne regarde jamais
    y_train, seulement les points déjà observés de X_train."""
    center = np.nanmedian(mat, axis=0)
    mad = np.nanmedian(np.abs(mat - center), axis=0)
    scale = np.maximum(1.4826 * mad, 0.02 * (np.abs(center) + 1.0))
    scale = np.where(scale < 1e-6, 1.0, scale)
    return (mat - center) / scale, center, scale


def build_windows(mat, window_len, stride):
    """Découpe l'axe temps en fenêtres chevauchantes (les "samples" attendus
    par PyPOTS) ; garde une dernière fenêtre alignée sur la fin de série même
    si le stride ne tombe pas juste, pour ne perdre aucun point."""
    T = mat.shape[0]
    window_len = min(window_len, T)
    starts = list(range(0, T - window_len + 1, stride))
    if not starts or starts[-1] + window_len < T:
        starts.append(T - window_len)
    starts = sorted(set(starts))
    windows = np.stack([mat[s:s + window_len] for s in starts], axis=0)
    return windows, starts


def reassemble(imputed_windows, starts, window_len, T, F):
    """Moyenne les prédictions des fenêtres qui se chevauchent pour
    reconstituer une seule matrice (T, F)."""
    out_sum = np.zeros((T, F))
    out_cnt = np.zeros((T, F))
    for w, s in zip(imputed_windows, starts):
        out_sum[s:s + window_len] += w
        out_cnt[s:s + window_len] += 1.0
    return out_sum / np.maximum(out_cnt, 1e-9)


def eval_saits_repeat(X_train, y_train, val_cols, donor_cols, opts):
    from pypots.imputation import SAITS

    feat_cols = select_saits_features(X_train, val_cols, donor_cols,
                                        opts["k_per_col"], opts["max_features"])
    cols = list(val_cols) + feat_cols
    n_target = len(val_cols)
    mat = X_train[cols].to_numpy(dtype=np.float64)  # vrais NaN gardés pour val_cols
    mat[:, n_target:] = pd.DataFrame(mat[:, n_target:]).ffill().bfill().to_numpy()  # donneurs "quasi-complets"

    scaled, center, scale = robust_scale_matrix(mat)
    T, F = scaled.shape
    window_len = opts["window_len"]
    windows, starts = build_windows(scaled, window_len, opts["stride"])

    saits = SAITS(
        n_steps=window_len, n_features=F, n_layers=opts["n_layers"],
        d_model=opts["d_model"], n_heads=opts["n_heads"], d_k=opts["d_k"], d_v=opts["d_v"],
        d_ffn=opts["d_ffn"], dropout=opts["dropout"], epochs=opts["epochs"],
        patience=opts["patience"], batch_size=min(opts["batch_size"], len(windows)),
        device=opts["device"], verbose=False,
    )
    saits.fit({"X": windows})
    imputed_windows = saits.predict({"X": windows})["imputation"]
    full_scaled = reassemble(imputed_windows, starts, window_len, T, F)
    full = full_scaled * scale + center  # retour à l'échelle d'origine (Watts)

    abs_err_by_col = {}
    for j, col in enumerate(val_cols):
        missing = X_train[col].isna().to_numpy()
        miss_idx = np.flatnonzero(missing)
        if len(miss_idx) == 0:
            continue
        pred = full[miss_idx, j]
        y_true = y_train[col].to_numpy()[miss_idx]
        valid = ~np.isnan(y_true)
        abs_err_by_col[col] = np.abs(pred[valid] - y_true[valid])
    return abs_err_by_col


# ---------------------------------------------------------------------------
# Boucle 10x10 appariée + rapport
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    token = os.environ.get("TABPFN_API_KEY")
    if not token:
        raise SystemExit("TABPFN_API_KEY manquant (export TABPFN_API_KEY=...)")
    tabpfn_client.set_access_token(token)

    try:
        import pypots  # noqa: F401
    except ImportError:
        raise SystemExit("pypots manquant : .venv/bin/pip install pypots")

    print(f"=== profil={PROFILE}  n_repeats={N_REPEATS}  saits_opts={SAITS_OPTS} ===", flush=True)
    print("Chargement des données...", flush=True)
    X_train, y_train, _ = load_data()
    donor_cols = [c for c in X_train.columns if not str(c).startswith("holed_")]
    holed_cols = list(y_train.columns)

    progress = json.load(open(PROGRESS_PATH)) if os.path.exists(PROGRESS_PATH) else {}

    for i in range(N_REPEATS):
        if str(i) in progress:
            print(f"repeat {i + 1}/{N_REPEATS} déjà fait, skip", flush=True)
            continue
        seed_i = v14.RANDOM_STATE + i
        dev_cols, val_cols_full = train_test_split(holed_cols, test_size=.1, random_state=seed_i)
        rng = np.random.RandomState(seed_i)
        val_cols = list(rng.choice(val_cols_full, size=min(N_VAL_SUB, len(val_cols_full)), replace=False))

        t0 = time.time()
        tabpfn_err = eval_tabpfn_repeat(X_train, y_train, donor_cols, dev_cols, val_cols, V14_BEST)
        print(f"  [repeat {i + 1}] tabpfn v14 : {time.time() - t0:.1f}s", flush=True)

        t0 = time.time()
        saits_err = eval_saits_repeat(X_train, y_train, val_cols, donor_cols, SAITS_OPTS)
        print(f"  [repeat {i + 1}] saits      : {time.time() - t0:.1f}s", flush=True)

        progress[str(i)] = {
            "val_cols": val_cols,
            "tabpfn": {c: e.tolist() for c, e in tabpfn_err.items()},
            "saits": {c: e.tolist() for c, e in saits_err.items()},
        }
        json.dump(progress, open(PROGRESS_PATH, "w"))
        print(f"repeat {i + 1}/{N_REPEATS} sauvegardé.\n", flush=True)

    # --- Rapport apparié (repeat par repeat ET colonne par colonne) ---
    rows = []
    all_tabpfn, all_saits = [], []
    wins_saits = wins_tabpfn = ties = 0
    for i in range(N_REPEATS):
        rec = progress[str(i)]
        tb = {c: np.array(v) for c, v in rec["tabpfn"].items()}
        sa = {c: np.array(v) for c, v in rec["saits"].items()}
        common = [c for c in rec["val_cols"] if c in tb and c in sa]
        if not common:
            continue
        tb_cat = np.concatenate([tb[c] for c in common])
        sa_cat = np.concatenate([sa[c] for c in common])
        all_tabpfn.append(tb_cat)
        all_saits.append(sa_cat)
        for c in common:
            m_t, m_s = tb[c].mean(), sa[c].mean()
            if m_s < m_t - 1e-9:
                wins_saits += 1
            elif m_t < m_s - 1e-9:
                wins_tabpfn += 1
            else:
                ties += 1
        rows.append((i + 1, float(tb_cat.mean()), float(sa_cat.mean())))

    print("\n| Repeat | TabPFN v14 | SAITS | Δ |")
    print("|---:|---:|---:|---:|")
    for r, t, s in rows:
        print(f"| {r} | {t:.2f} | {s:.2f} | {s - t:+.2f} |")

    mae_tabpfn = float(np.concatenate(all_tabpfn).mean()) if all_tabpfn else float("nan")
    mae_saits = float(np.concatenate(all_saits).mean()) if all_saits else float("nan")
    print(f"\nMAE TabPFN v14 (pool): {mae_tabpfn:.2f} W")
    print(f"MAE SAITS      (pool): {mae_saits:.2f} W")
    if mae_tabpfn == mae_tabpfn:  # not NaN
        print(f"gain relatif SAITS vs TabPFN v14: {100 * (1 - mae_saits / mae_tabpfn):.1f}%")
    print(f"colonnes gagnées par SAITS: {wins_saits} / par TabPFN v14: {wins_tabpfn} / égalité: {ties} "
          f"/ total: {wins_saits + wins_tabpfn + ties}")

    out = {
        "profile": PROFILE, "saits_opts": SAITS_OPTS,
        "rows": [{"repeat": r, "tabpfn": t, "saits": s, "delta": s - t} for r, t, s in rows],
        "mae_tabpfn": mae_tabpfn, "mae_saits": mae_saits,
        "wins_saits": wins_saits, "wins_tabpfn": wins_tabpfn, "ties": ties,
    }
    json.dump(out, open(RESULTS_PATH, "w"), indent=2)
    print(f"\nrésultats écrits dans {RESULTS_PATH}", flush=True)
