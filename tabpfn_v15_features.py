"""Features V15 sans fuite pour l'imputation des courbes Enedis.

Chaque groupe est activable séparément afin de mesurer son apport sur les
splits appariés de V14. Les valeurs manquantes de la cible ne sont jamais
consultées; les statistiques cible sont calculées uniquement sur ``observed``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import feature_engineering_bench as feb

FEATURE_GROUPS = (
    "gap_geometry", "boundary_values", "boundary_slopes", "curvature",
    "hermite_bridge", "local_volatility", "regime_shift", "daytype_profile",
    "recent_profile", "adjusted_periodic", "periodic_residuals",
    "neighbor_reliability", "neighbor_local_corr", "neighbor_distribution",
    "neighbor_dynamics",
)

# Groupes "round 2" : nécessitent le pool de donneurs brut (pas seulement les
# ~40 voisins déjà sélectionnés globalement) -> calculés à part dans
# tabpfn_v15_bench.predict() et non dans build_v15_features(). Ajoutés au
# tuple pour que le CLI/l'écran d'ablation les reconnaisse comme groupes
# valides au même titre que les autres.
POOL_FEATURE_GROUPS = (
    "local_reselect", "diverse_donor_consensus",
    "donor_residual_synchrony", "micro_shift_consensus",
)
ROW_FEATURE_GROUPS = ("shape_match_block",)
FEATURE_GROUPS = FEATURE_GROUPS + POOL_FEATURE_GROUPS + ROW_FEATURE_GROUPS

# Campagne T01--T50 : familles volontairement disjointes des blocs V15,
# G/H/R/S et du catalogue ``new::*``. Elles sont conçues comme des signaux
# candidats légers (pas comme des remplaçants des modèles nommés dans leur
# libellé) afin de pouvoir être criblées une par une avec LightGBM.
T_FEATURE_GROUPS = tuple(f"T{i:02d}" for i in range(1, 51))
FEATURE_GROUPS = FEATURE_GROUPS + T_FEATURE_GROUPS


def _finite_fill(x: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    return np.nan_to_num(np.asarray(x, float), nan=fallback, posinf=fallback, neginf=fallback)


def _shift_known(values: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    out = np.full(len(values), np.nan); available = np.zeros(len(values))
    if lag > 0:
        out[lag:] = values[:-lag]; available[lag:] = np.isfinite(values[:-lag])
    elif lag < 0:
        out[:lag] = values[-lag:]; available[:lag] = np.isfinite(values[-lag:])
    else:
        out = values.copy(); available = np.isfinite(values)
    return out, available


def _boundaries(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(values); pos = np.arange(n); known = np.isfinite(values)
    # Frontières strictes : à une ligne observée, ne jamais réutiliser la
    # cible courante comme feature. Pour un vrai trou, elles restent les deux
    # observations les plus proches qui l'encadrent.
    previous = np.maximum.accumulate(np.where(known, pos, -1))
    following = np.minimum.accumulate(np.where(known, pos, n)[::-1])[::-1]
    li = np.r_[-1, previous[:-1]]
    ri = np.r_[following[1:], n]
    left = np.where(li >= 0, values[np.maximum(li, 0)], np.nan)
    right = np.where(ri < n, values[np.minimum(ri, n - 1)], np.nan)
    return left, right, li, ri


def _rolling_stat(values: np.ndarray, window: int, stat: str) -> np.ndarray:
    s = pd.Series(values)
    if stat == "median": return s.rolling(window, min_periods=2, center=True).median().to_numpy()
    return s.rolling(window, min_periods=2, center=True).apply(
        lambda z: np.nanmedian(np.abs(z - np.nanmedian(z))), raw=True).to_numpy()


def _daytype_profiles(values: np.ndarray, index) -> np.ndarray:
    dt = pd.DatetimeIndex(pd.to_datetime(index)); slot = dt.hour * 2 + dt.minute // 30
    daytype = np.where(dt.dayofweek < 5, 0, np.where(dt.dayofweek == 5, 1, 2))
    frame = pd.DataFrame({"v": values, "slot": slot, "type": daytype})
    obs = frame.v.notna(); global_med = float(np.nanmedian(values))
    result=np.empty((len(values),2),float)
    for i,(typ,sl) in enumerate(zip(daytype,slot)):
        use=(daytype==typ)&(slot==sl)&np.isfinite(values)
        use[i]=False  # leave-one-out explicite aux lignes d'entraînement
        z=values[use]
        if len(z):
            m=np.median(z); result[i]=(m,np.median(np.abs(z-m)))
        else: result[i]=(global_med,0.)
    return result


def _recent_slot_profile(values: np.ndarray, decay: float = 0.7) -> np.ndarray:
    candidates=[]; weights=[]
    for lag, weight in ((48, 1.), (96, decay), (144, decay**2), (336, decay**3)):
        v, a = _shift_known(values, lag); candidates.append(v); weights.append(weight*a)
    V=np.column_stack(candidates); W=np.column_stack(weights)
    return np.divide(np.nansum(V*W, axis=1), W.sum(1), out=np.full(len(values), np.nan), where=W.sum(1)>0)


def build_v15_features(values, index, scaled_neighbors, missing, groups=FEATURE_GROUPS, pool=None):
    """Retourne une matrice (temps, features) pour les groupes demandés.

    ``pool`` (optionnel) : matrice brute (temps x donneurs) du pool de
    candidats utilisé par l'appelant (ex. ``sub_vals`` dans
    ``build_v14_features``) -- nécessaire uniquement pour les groupes "round
    2" (``local_reselect``, ``diverse_donor_consensus``, ``shape_match_block``)
    qui doivent resélectionner des voisins localement plutôt que d'annoter
    les ~40 déjà choisis globalement (``scaled_neighbors``)."""
    values=np.asarray(values,float).copy(); missing=np.asarray(missing,bool)
    values[missing]=np.nan
    n=len(values); groups=set(groups); feats=[]
    center=float(np.nanmedian(values)); scale=max(float(np.nanpercentile(values,95)-np.nanpercentile(values,5)),1.)
    left,right,li,ri=_boundaries(values); pos=np.arange(n)
    dl=np.where(li>=0,pos-li,n); dr=np.where(ri<n,ri-pos,n)
    run=feb.run_lengths_for(missing).astype(float); edge=feb.edge_distance_for(missing).astype(float)

    if "gap_geometry" in groups:
        # Les observations apprennent le cas majoritaire d'un trou ponctuel.
        train_run=np.where(missing,run,1.); train_edge=np.where(missing,edge,0.)
        rel=np.divide(train_edge,np.maximum(train_run-1,1),out=np.full(n,.5),where=train_run>1)
        feats.append(np.column_stack([np.log1p(train_run),np.log1p(dl),np.log1p(dr),rel,
                                      (li<0).astype(float),(ri>=n).astype(float)]))
    if "boundary_values" in groups:
        feats.append(np.column_stack([(left-center)/scale,(right-center)/scale,
                                      ((left+right)/2-center)/scale,(right-left)/scale,
                                      np.divide(right,left,out=np.ones(n),where=np.abs(left)>1.)]))

    slopes=[]
    for w in (1,2,4,8):
        lp=np.clip(li-w,0,n-1); rn=np.clip(ri+w,0,n-1)
        ls=np.where((li>=w)&np.isfinite(values[lp]),(left-values[lp])/w,np.nan)
        rs=np.where((ri+w<n)&np.isfinite(values[rn]),(values[rn]-right)/w,np.nan)
        slopes.extend([ls/scale,rs/scale])
    if "boundary_slopes" in groups: feats.append(np.column_stack(slopes))
    if "curvature" in groups:
        d=np.diff(values,prepend=np.nan); dd=np.diff(d,prepend=np.nan)
        lc=np.where(li>=0,dd[np.maximum(li,0)],np.nan); rc=np.where(ri<n,dd[np.minimum(ri,n-1)],np.nan)
        feats.append(np.column_stack([lc/scale,rc/scale,(rc-lc)/scale]))
    if "hermite_bridge" in groups:
        duration=np.maximum(ri-li,1); u=np.clip((pos-li)/duration,0,1)
        m0=np.asarray(slopes[0])*scale; m1=np.asarray(slopes[1])*scale
        h=(2*u**3-3*u**2+1)*left+(u**3-2*u**2+u)*duration*m0+(-2*u**3+3*u**2)*right+(u**3-u**2)*duration*m1
        linear=(1-u)*left+u*right
        feats.append(np.column_stack([(h-center)/scale,(h-linear)/scale]))
    if "local_volatility" in groups:
        cols=[]
        for w in (6,12,24,48,96):
            stat=[]
            for i in range(n):
                z=np.r_[values[max(0,i-w):i],values[i+1:min(n,i+w+1)]]
                dz=np.diff(z[np.isfinite(z)]); stat.append(np.median(np.abs(dz-np.median(dz))) if len(dz)>1 else 0.)
            cols.append(np.asarray(stat)/scale)
        feats.append(np.column_stack(cols))
    if "regime_shift" in groups:
        cols=[]
        for w in (6,12,24,48):
            lm=np.array([np.nanmedian(values[max(0,int(x)-w+1):int(x)+1]) if x>=0 else np.nan for x in li])
            rm=np.array([np.nanmedian(values[int(x):min(n,int(x)+w)]) if x<n else np.nan for x in ri])
            cols.extend([(rm-lm)/scale,(rm+lm-2*center)/(2*scale)])
        feats.append(np.column_stack(cols))
    dayprof=_daytype_profiles(values,index)
    if "daytype_profile" in groups: feats.append(np.column_stack([(dayprof[:,0]-center)/scale,dayprof[:,1]/scale]))
    recent=_recent_slot_profile(values)
    if "recent_profile" in groups: feats.append(np.column_stack([(recent-center)/scale,np.isfinite(recent)]))

    periodic=[]
    for lag in (48,96,144,336):
        past,ap=_shift_known(values,lag); future,af=_shift_known(values,-lag)
        periodic.extend([past,future]);
        if "adjusted_periodic" in groups:
            local=np.where(np.isfinite((left+right)/2),(left+right)/2,center)
            base=_rolling_stat(values,48,"median")
            feats.append(np.column_stack([(past+(local-base)-center)/scale,
                                          (future+(local-base)-center)/scale,ap,af]))
    if "periodic_residuals" in groups:
        P=np.column_stack(periodic); consensus=np.nanmedian(P,axis=1)
        feats.append(np.column_stack([(consensus-dayprof[:,0])/scale,
                                      np.nanmedian(np.abs(P-consensus[:,None]),axis=1)/scale,
                                      (recent-consensus)/scale]))

    N=np.asarray(scaled_neighbors,float)
    if N.ndim==2 and N.shape[0]==n and N.shape[1]>0:
        residual=np.abs(N-values[:,None]); residual[missing]=np.nan
        valid=np.isfinite(residual); total=np.nansum(residual,axis=0); count=valid.sum(axis=0)
        # Fiabilité leave-one-out pour chaque timestamp observé; aux trous,
        # toute l'information visible est utilisée.
        reliability=(total[None,:]-np.nan_to_num(residual))/(np.maximum(count[None,:]-valid,1))
        weights=1/(reliability+0.05*scale)
        wmean=np.nansum(N*weights,axis=1)/np.maximum(np.nansum(np.isfinite(N)*weights,axis=1),1e-9)
        if "neighbor_reliability" in groups:
            feats.append(np.column_stack([(wmean-center)/scale,
                np.nanmedian(reliability,axis=1)/scale,np.nanmin(reliability,axis=1)/scale]))
        if "neighbor_local_corr" in groups:
            local=[]; target=pd.Series(values)
            for j in range(min(10,N.shape[1])):
                donor=pd.Series(N[:,j])
                past=target.shift(1).rolling(48,min_periods=8).corr(donor.shift(1))
                future=target.shift(-1)[::-1].rolling(48,min_periods=8).corr(donor.shift(-1)[::-1])[::-1]
                local.append(pd.concat([past,future],axis=1).mean(axis=1).to_numpy())
            feats.append(np.column_stack([np.nanmedian(local,axis=0),np.nanmax(local,axis=0)]))
        if "neighbor_distribution" in groups:
            q=np.nanpercentile(N,[10,25,50,75,90],axis=1).T
            feats.append(np.column_stack([(q-center)/scale,(q[:,3]-q[:,1])/scale,
                                          np.nanstd(N,axis=1)/scale]))
        if "neighbor_dynamics" in groups:
            med=np.nanmedian(N,axis=1); d1=np.diff(med,prepend=np.nan); d2=np.diff(d1,prepend=np.nan)
            donor_d=np.diff(N,axis=0,prepend=np.nan)
            feats.append(np.column_stack([d1/scale,d2/scale,np.nanmedian(donor_d,axis=1)/scale,
                                          np.nanstd(donor_d,axis=1)/scale]))

    if pool is not None:
        obs_idx=np.flatnonzero(~missing); miss_idx=np.flatnonzero(missing)
        if "local_reselect" in groups:
            feats.append(local_reselect_feature(values,pool,obs_idx,miss_idx))
        if "diverse_donor_consensus" in groups:
            feats.append(diverse_donor_consensus_feature(pool,obs_idx,values))
        if "shape_match_block" in groups:
            block=shape_match_block_feature(values)
            feats.append(np.column_stack([np.nan_to_num(block,nan=0.),np.isfinite(block).astype(float)]))
        if "donor_residual_synchrony" in groups:
            feats.append(donor_residual_synchrony_feature(pool))
        if "micro_shift_consensus" in groups:
            feats.append(micro_shift_consensus_feature(values,pool,obs_idx))

    # Pont vers le catalogue `boosted_columns_feature_search.py` (25 familles
    # "new::*" issues du criblage tabpfn_column_feature_screen.py, cf.
    # tabpfn_column_feature_screen_results.md) : réutilise build_25_features
    # telle quelle plutôt que de redupliquer sa logique, import différé pour
    # éviter le coût (torch) quand aucun groupe "new::*" n'est demandé.
    boosted_names=[g[len("new::"):] for g in groups if g.startswith("new::")]
    if boosted_names:
        import boosted_columns_feature_search as bcf
        from tabpfn_column_individual_bench import robust_location_scale
        sr=pd.Series(values,index=index)
        center,scale=robust_location_scale(sr)
        neighbors=scaled_neighbors if scaled_neighbors is not None else np.empty((n,0))
        all25=bcf.build_25_features(sr,index,neighbors,center,scale)
        for name in boosted_names:
            feats.append(all25[name])
    t_names=tuple(g for g in groups if g in T_FEATURE_GROUPS)
    if t_names:
        feats.append(build_t_features(values, index, scaled_neighbors, missing, t_names))
    return _finite_fill(np.hstack(feats) if feats else np.empty((n,0)))


def _missing_runs(missing: np.ndarray) -> list[np.ndarray]:
    idx = np.flatnonzero(missing)
    if len(idx) == 0:
        return []
    breaks = np.where(np.diff(idx) > 1)[0]
    return np.split(idx, breaks + 1)


def local_reselect_feature(values, pool, obs, miss, window=96, k=20, min_local=15):
    """Feature colonnes x lignes combinée (jamais testée jusqu'ici, cf.
    benchmark.md §10 pt.2 / §15quater) : au lieu de sélectionner les
    voisins une fois pour toute la courbe par corrélation globale, on
    resélectionne les k donneurs les plus corrélés **localement** (fenêtre
    de `window` points autour de chaque trou), donneur par donneur et trou
    par trou -- un donneur peut être pertinent sur un segment de la courbe et
    pas sur un autre. Repli sur la corrélation globale si le contexte local
    est trop pauvre (<min_local points observés dans la fenêtre).
    Renvoie 2 colonnes : la prédiction locale, et son écart à une prédiction
    globale (même k, mêmes donneurs candidats) -- l'écart signale les cas où
    la resélection locale change vraiment d'avis par rapport au global."""
    n = len(values)
    local_pred = np.full(n, np.nan)
    global_pred = np.full(n, np.nan)
    if len(miss) == 0 or pool.shape[1] == 0:
        return np.zeros((n, 2))
    obs = np.sort(np.asarray(obs))
    y_obs = values[obs]
    X_obs = pool[obs]

    def _weighted_pred(y_ctx, X_ctx, X_target, k):
        if len(y_ctx) < 3:
            return None
        yc = y_ctx - y_ctx.mean()
        Xc = X_ctx - X_ctx.mean(axis=0, keepdims=True)
        denom = np.sqrt((yc ** 2).sum()) * np.sqrt((Xc ** 2).sum(axis=0)) + 1e-9
        corr = (Xc.T @ yc) / denom
        top = np.argsort(-np.abs(corr))[:min(k, X_ctx.shape[1])]
        w = np.abs(corr[top]); w = w / (w.sum() + 1e-9)
        return X_target[:, top] @ w

    global_gp = _weighted_pred(y_obs, X_obs, pool, k)
    if global_gp is not None:
        global_pred[:] = global_gp

    missing_mask = np.zeros(n, dtype=bool); missing_mask[miss] = True
    for run in _missing_runs(missing_mask):
        lo, hi = run[0] - window, run[-1] + window
        local_mask = (obs >= lo) & (obs <= hi)
        if local_mask.sum() < min_local:
            continue  # repli implicite : local_pred reste NaN -> rempli par global_pred via nan_to_num plus bas
        pred = _weighted_pred(y_obs[local_mask], X_obs[local_mask], pool[run], k)
        if pred is not None:
            local_pred[run] = pred

    fallback = np.where(np.isfinite(local_pred), local_pred, global_pred)
    disagreement = local_pred - global_pred
    return np.column_stack([fallback, disagreement])


def diverse_donor_consensus_feature(pool, obs, values, k_pool=60, k_diverse=12):
    """Feature colonnes seules (variante non testée de §10 pt.5) : les
    groupes `neighbor_distribution`/`neighbor_reliability` déjà criblés
    résument les ~40 voisins bruts sélectionnés par corrélation, qui sont
    souvent redondants (quasi-doublons). Ici on présélectionne un pool plus
    large (k_pool par corrélation), puis on choisit greedy k_diverse donneurs
    mutuellement éloignés en forme (max-min) avant de résumer -- un
    consensus sur un sous-ensemble diversifié plutôt que sur les meilleurs
    voisins bruts."""
    n, m = pool.shape
    if m == 0:
        return np.zeros((n, 2))
    obs = np.sort(np.asarray(obs))
    y = values[obs]; X = pool[obs]
    yc = y - y.mean(); Xc = X - X.mean(axis=0, keepdims=True)
    denom = np.sqrt((yc ** 2).sum()) * np.sqrt((Xc ** 2).sum(axis=0)) + 1e-9
    corr = (Xc.T @ yc) / denom
    top = np.argsort(-np.abs(corr))[:min(k_pool, m)]
    if len(top) <= 1:
        sel = top
    else:
        Zfull = pool[:, top]
        Z = (Zfull - Zfull.mean(axis=0)) / (Zfull.std(axis=0) + 1e-9)
        chosen = [int(np.argmax(np.abs(corr[top])))]
        dist = np.full(len(top), np.inf)
        for _ in range(1, min(k_diverse, len(top))):
            last = Z[:, chosen[-1]]
            d = np.mean((Z - last[:, None]) ** 2, axis=0)
            dist = np.minimum(dist, d)
            dist[chosen] = -1
            chosen.append(int(np.argmax(dist)))
        sel = top[chosen]
    consensus = np.median(pool[:, sel], axis=1)
    spread = np.std(pool[:, sel], axis=1)
    return np.column_stack([consensus, spread])


def donor_residual_synchrony_feature(pool, lag=48):
    """Feature colonnes x lignes (round 2, jamais testée) : au lieu du résidu
    périodique de la SEULE courbe cible (`periodic_residuals`, déjà testé,
    neutre) ou du niveau collectif brut (indice de consommation collective,
    §15quater, rejeté), calcule le résidu périodique (valeur - valeur à
    `lag`) de CHAQUE donneur du pool, puis en prend la médiane à chaque
    instant -- un "choc collectif" (météo, jour spécial) mesuré sur le
    résidu inter-donneurs plutôt que sur leur niveau brut. Varie à toutes
    les positions (observées et manquantes), donc entraînable."""
    n, m = pool.shape
    if m == 0 or lag >= n:
        return np.zeros((n, 2))
    shifted = np.full_like(pool, np.nan)
    shifted[lag:] = pool[:-lag]
    residual = pool - shifted
    collective = np.nanmedian(residual, axis=1)
    spread = np.nanmedian(np.abs(residual - collective[:, None]), axis=1)
    return np.column_stack([np.nan_to_num(collective, nan=0.), np.nan_to_num(spread, nan=0.)])


def micro_shift_consensus_feature(values, pool, obs, shifts=(-2, -1, 0, 1, 2)):
    """Feature colonnes x lignes (round 2, jamais testée) : les copies
    périodiques et le KNN supposent tous un alignement exact au pas de 30
    minutes entre la courbe cible et son pool de voisins. Cherche ici, sur
    toute la courbe, le petit décalage (+/-2 pas = +/-1h) qui minimise
    l'écart entre la cible observée et le consensus des donneurs, puis
    renvoie le consensus décalé de ce montant -- capture un éventuel
    décalage de routine (ex. routine du foyer avancée/retardée de 30-60 min
    par rapport à son pool de voisins)."""
    n, m = pool.shape
    if m == 0:
        return np.zeros((n, 2))
    consensus = np.nanmedian(pool, axis=1)
    obs = np.asarray(obs)
    y = values[obs]
    best_shift, best_score = 0, np.inf
    for d in shifts:
        src = obs - d
        valid = (src >= 0) & (src < n)
        if valid.sum() < 20:
            continue
        score = np.nanmean((y[valid] - consensus[src[valid]]) ** 2)
        if score < best_score:
            best_score, best_shift = score, d
    shifted = np.full(n, np.nan)
    src_idx = np.arange(n) - best_shift
    ok = (src_idx >= 0) & (src_idx < n)
    shifted[ok] = consensus[src_idx[ok]]
    return np.column_stack([np.nan_to_num(shifted, nan=0.), np.full(n, float(best_shift))])


def shape_match_block_feature(values, ctx=3, search_max=400, max_run=200):
    """Feature lignes seules (variante non testée de §10 pt.3) : au lieu
    d'une copie périodique à décalage fixe (±48/96/144/336, déjà en place),
    on cherche dans l'historique **propre à la courbe** le bloc de même
    longueur dont le contexte des deux bords (les `ctx` points avant/après)
    ressemble le plus (moindres carrés) à celui du trou courant, puis on
    recopie ce bloc -- un alignement libre sur les deux bords plutôt qu'un
    lag calendaire fixe, utile quand le rythme de la courbe est décalé
    (ex. jour férié, changement d'horaire de la maisonnée)."""
    n = len(values)
    known = np.isfinite(values)
    out = np.full(n, np.nan)
    for run in _missing_runs(~known):
        L = len(run)
        if L > max_run:
            continue
        s0, s1 = run[0], run[-1]
        left_ctx = np.arange(s0 - ctx, s0)
        right_ctx = np.arange(s1 + 1, s1 + 1 + ctx)
        if left_ctx[0] < 0 or right_ctx[-1] >= n:
            continue
        if not (known[left_ctx].all() and known[right_ctx].all()):
            continue
        target_ctx = np.r_[values[left_ctx], values[right_ctx]]
        lo, hi = max(0, s0 - search_max), min(n, s1 + search_max)
        best_score, best_pos = np.inf, None
        for cand0 in range(lo, hi - L - 2 * ctx + 1):
            cl = np.arange(cand0, cand0 + ctx)
            cr = np.arange(cand0 + ctx + L, cand0 + 2 * ctx + L)
            if cr[-1] >= n:
                break
            if cl[0] <= s1 and cr[-1] >= s0:
                continue  # ne pas piocher dans le trou lui-même
            if not (known[cl].all() and known[cr].all()):
                continue
            score = np.sum((np.r_[values[cl], values[cr]] - target_ctx) ** 2)
            if score < best_score:
                best_score, best_pos = score, cand0 + ctx
        if best_pos is not None:
            out[run] = values[best_pos:best_pos + L]
    return out


# ===========================================================================
# Campagne T01--T50 : familles disjointes des blocs V15 ci-dessus et des
# candidates G/H/R/S fournies (voir tabpfn_v15_bench.py) -- modèles d'état,
# décompositions multi-échelles, régimes/motifs/complexité, dépendances
# entre courbes/graphes, complétion globale + méta-prédictions OOF.
#
# Règles communes :
# - Toute statistique "propre à la cible" n'utilise QUE les positions
#   observées (jamais la valeur manquante elle-même).
# - Beaucoup de méthodes ici (Kalman/GP/SSA/ondelettes/entropie...) exigent
#   un signal continu : `ctx["interp"]` est une interpolation linéaire
#   locale (mêmes bords que evaluate.fill_nan_with_interpolation), qui ne
#   consulte aucune vérité terrain -- comme les ponts Hermite/périodiques
#   déjà présents dans build_v15_features.
# - Piège identifié et évité : une feature qui ne varie QU'AUX positions
#   manquantes (ex. limiter le calcul aux seuls trous, comme
#   `shape_match_block_feature`) est quasi constante aux lignes
#   d'ENTRAÎNEMENT (les positions observées) -> LightGBM n'a aucun exemple
#   pour apprendre à s'en servir et la feature devient un no-op. `_t_blocks`
#   ci-dessous découpe donc la série entière en blocs cœur non chevauchants
#   (avec contexte élargi pour l'ajustement local) afin que les méthodes
#   coûteuses (GP, SSA, matrix profile, RQA, visibility graph, DTW,
#   Laplacien de graphe...) couvrent aussi les lignes observées à coût
#   borné, plutôt que de se limiter aux trous ou de tourner sur chaque ligne.
# - T41--T50 (méta-prédictions par complétion de matrice/tenseur ou proxy
#   allégé de modèle neuronal) suivent la règle stricte demandée : le
#   générateur n'est ajusté QUE sur les lignes observées, et sa valeur à ces
#   lignes est produite en LOOCV analytique (matrice chapeau d'une
#   régression ridge sur une base fixe issue des donneurs) -- jamais en
#   réutilisant la ligne courante. `_ridge_oof_basis` implémente ce motif
#   une fois, réutilisé par les 10 groupes.
# - `build_v15_features` (l'appelant) est invoqué HORS du try/except de
#   `predict()` dans tabpfn_v15_bench.py : une exception ici ferait donc
#   échouer toute la colonne plutôt que de retomber sur le KNN. Chaque
#   groupe est donc enveloppé individuellement (cf. `build_t_features`).
# ===========================================================================

T_FEATURE_WIDTHS = {
    "T01": 4, "T02": 4, "T03": 2, "T04": 3, "T05": 3, "T06": 3, "T07": 3,
    "T08": 2, "T09": 3, "T10": 3, "T11": 5, "T12": 6, "T13": 4, "T14": 2,
    "T15": 4, "T16": 2, "T17": 3, "T18": 3, "T19": 3, "T20": 2,
    "T21": 3, "T22": 4, "T23": 6, "T24": 2, "T25": 2, "T26": 3, "T27": 2,
    "T28": 2, "T29": 2, "T30": 3, "T31": 3, "T32": 2, "T33": 2, "T34": 2,
    "T35": 3, "T36": 2, "T37": 2, "T38": 2, "T39": 2, "T40": 2,
    "T41": 2, "T42": 2, "T43": 2, "T44": 2, "T45": 2, "T46": 2, "T47": 2,
    "T48": 4, "T49": 4, "T50": 5,
}


def _t_local_interp(values: np.ndarray) -> np.ndarray:
    """Interpolation linéaire locale (bords en constant) -- signal continu
    utilisé UNIQUEMENT comme entrée de méthodes spectrales/de forme
    (ondelettes, SSA, EMD, cepstre, visibility graph, matrix profile,
    RQA, entropie...) qui exigent l'absence de NaN. Ne consulte aucune
    vérité terrain : même construction que
    evaluate.fill_nan_with_interpolation."""
    s = pd.Series(values)
    return s.interpolate(method="linear", limit_direction="both").ffill().bfill().to_numpy(float)


def _t_context(values, index, scaled_neighbors, missing) -> dict:
    n = len(values)
    values = np.asarray(values, float).copy()
    missing = np.asarray(missing, bool)
    values[missing] = np.nan
    observed = ~missing
    left, right, li, ri = _boundaries(values)
    pos = np.arange(n)
    own = values[observed]
    center = float(np.nanmedian(values)) if observed.any() else 0.0
    scale = max(float(np.nanpercentile(own, 95) - np.nanpercentile(own, 5)), 1.0) if observed.any() else 1.0
    interp = _t_local_interp(values)
    dt = pd.DatetimeIndex(pd.to_datetime(index))
    phase = (dt.hour * 2 + dt.minute // 30).to_numpy()
    dow = dt.dayofweek.to_numpy()
    N = np.asarray(scaled_neighbors, float) if scaled_neighbors is not None else np.empty((n, 0))
    if N.ndim != 2 or N.shape[0] != n:
        N = np.empty((n, 0))
    N = np.nan_to_num(N, nan=center)
    return dict(n=n, values=values, missing=missing, observed=observed, pos=pos,
                left=left, right=right, li=li, ri=ri, center=center, scale=scale,
                interp=interp, index=dt, phase=phase, dow=dow, N=N)


def _t_blocks(n: int, size: int = 144, pad: int = 48):
    """Découpe [0,n) en blocs cœur non chevauchants de taille `size`, chacun
    avec un contexte élargi de `pad` de chaque côté pour l'ajustement local.
    Voir la note sur la couverture pleine série en tête de section."""
    size = max(8, min(size, n))
    out = []
    for lo in range(0, n, size):
        hi = min(n, lo + size)
        clo, chi = max(0, lo - pad), min(n, hi + pad)
        out.append((lo, hi, clo, chi))
    return out


# --- T01--T10 : modèles temporels et modèles d'état ------------------------

def _kalman_uc(values, missing, **kwargs):
    import statsmodels.api as sm
    endog = np.where(missing, np.nan, values)
    mod = sm.tsa.UnobservedComponents(endog, **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mod.fit(disp=False, maxiter=50)


def _t01(ctx):
    """Modèle local-level : moyenne filtrée/lissée, variance postérieure, innovation."""
    res = _kalman_uc(ctx["values"], ctx["missing"], level="local level")
    c, s = ctx["center"], ctx["scale"]
    filt, smooth, var = res.level.filtered, res.level.smoothed, res.level.smoothed_cov
    innov = np.nan_to_num(res.forecasts_error[0])
    return np.column_stack([(filt - c) / s, (smooth - c) / s, var / s ** 2, innov / s])


def _t02(ctx):
    """Local-linear-trend : état niveau/pente, incertitude de pente, désaccord filtre-lisseur."""
    res = _kalman_uc(ctx["values"], ctx["missing"], level="local linear trend")
    c, s = ctx["center"], ctx["scale"]
    level_f, level_s = res.level.filtered, res.level.smoothed
    slope_s, slope_var = res.trend.smoothed, res.trend.smoothed_cov
    return np.column_stack([(level_s - c) / s, slope_s / s, slope_var / s ** 2, (level_f - level_s) / s])


def _t03(ctx):
    """Composante saisonnière stochastique à 48 pas (harmoniques trig.) + variance."""
    res = _kalman_uc(ctx["values"], ctx["missing"], level="local level",
                      freq_seasonal=[{"period": 48, "harmonics": 6}])
    s = ctx["scale"]; fs = res.freq_seasonal[0]
    return np.column_stack([np.asarray(fs.smoothed) / s, np.nan_to_num(np.asarray(fs.smoothed_cov)) / s ** 2])


def _t04(ctx):
    """État espace combinant saisonnalités journalière (48) et hebdomadaire (336)."""
    res = _kalman_uc(ctx["values"], ctx["missing"], level="local level",
                      freq_seasonal=[{"period": 48, "harmonics": 4}, {"period": 336, "harmonics": 2}])
    s = ctx["scale"]
    f48, f336 = np.asarray(res.freq_seasonal[0].smoothed), np.asarray(res.freq_seasonal[1].smoothed)
    return np.column_stack([(f48 + f336) / s, f48 / s, f336 / s])


def _fit_ridge_lags(y_cont, target_obs_mask, order, reverse=False):
    """Ridge sur `order` décalages du signal interpolé, ajustée seulement aux
    positions où la CIBLE est réellement observée -- les décalages utilisés
    comme features viennent du signal interpolé (jamais de la valeur
    manquante elle-même), et à une position observée le décalage k>=1 ne
    revoit jamais cette position (décalage strict)."""
    from sklearn.linear_model import Ridge
    z = y_cont[::-1].copy() if reverse else y_cont
    n = len(z)
    cols = [np.r_[np.full(k, z[0]), z[:-k]] for k in range(1, order + 1)]
    X = np.column_stack(cols)
    mask = target_obs_mask[::-1] if reverse else target_obs_mask
    if mask.sum() < order + 3:
        pred = np.full(n, float(np.nanmean(z)))
    else:
        reg = Ridge(alpha=1.0).fit(X[mask], z[mask])
        pred = reg.predict(X)
    return pred[::-1] if reverse else pred


def _t05(ctx):
    """Prédictions AR régularisées depuis la gauche et la droite, + désaccord."""
    y, obs, c, s = ctx["interp"], ctx["observed"], ctx["center"], ctx["scale"]
    fwd = _fit_ridge_lags(y, obs, 8, reverse=False)
    bwd = _fit_ridge_lags(y, obs, 8, reverse=True)
    return np.column_stack([(fwd - c) / s, (bwd - c) / s, (fwd - bwd) / s])


def _burg(x, order):
    """Algorithme de Burg standard -- coefficients AR par minimisation des
    erreurs de prédiction avant/arrière (pas de perte de points contrairement
    à Yule-Walker sur autocovariance)."""
    x = np.asarray(x, float)
    n = len(x)
    order = min(order, max(n // 3, 1))
    f = x.copy(); b = x.copy()
    a = np.zeros(order + 1); a[0] = 1.0
    dk = float(np.sum(2 * x[1:] ** 2)) - x[0] ** 2 - x[-1] ** 2
    for k in range(order):
        fk, bk = f[k + 1:], b[:-k - 1] if k > 0 else b[:-1]
        num = -2.0 * np.sum(fk * bk)
        dk = max(dk, 1e-9)
        mu = num / dk
        f_new = fk + mu * bk
        b_new = bk + mu * fk
        f[k + 1:], b[k:-1] = f_new, b_new
        a_prev = a[:k + 2].copy()
        a[:k + 2] = a_prev + mu * a_prev[::-1]
        dk = (1 - mu ** 2) * dk - f_new[-1] ** 2 - b_new[0] ** 2 if len(f_new) else dk
    return -a[1:order + 1]


def _t06(ctx):
    """Coefficients AR de Burg : stabilité des pôles + puissance spectrale autour de 48/336 pas."""
    y = ctx["interp"]; n = ctx["n"]
    z = (y - y.mean())
    try:
        coeffs = _burg(z, min(20, max(n // 4, 2)))
        roots = np.roots(np.r_[1.0, -coeffs]) if len(coeffs) else np.array([0.0])
        pole_stability = float(np.max(np.abs(roots))) if len(roots) else 0.0
        freqs = np.array([1.0 / 48, 1.0 / 336])
        w = 2 * np.pi * freqs
        ar_poly = np.r_[1.0, -coeffs]
        power = 1.0 / (np.abs(np.polyval(ar_poly[::-1], np.exp(-1j * w))) ** 2 + 1e-9)
        p48, p336 = np.log1p(power)
    except Exception:
        pole_stability, p48, p336 = 0.0, 0.0, 0.0
    return np.column_stack([np.full(n, pole_stability), np.full(n, p48), np.full(n, p336)])


def _t07(ctx):
    """Forecast/hindcast ARIMA ajusté uniquement sur les observations disponibles (filtre de Kalman gère les NaN)."""
    from statsmodels.tsa.arima.model import ARIMA
    y, missing, c, s = ctx["values"], ctx["missing"], ctx["center"], ctx["scale"]
    n = ctx["n"]
    endog = np.where(missing, np.nan, y)

    def _fit(z):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ARIMA(z, order=(2, 0, 1), trend="c").fit(method_kwargs={"maxiter": 50})
            return np.nan_to_num(res.fittedvalues, nan=np.nanmean(z))
        except Exception:
            return np.full(len(z), float(np.nanmean(z)) if np.isfinite(np.nanmean(z)) else 0.0)

    fwd = _fit(endog)
    bwd = _fit(endog[::-1])[::-1]
    return np.column_stack([(fwd - c) / s, (bwd - c) / s, (fwd - bwd) / s])


def _gp_window_predict(x_obs, y_obs, x_query, kernel, max_points=150, n_restarts=0):
    from sklearn.gaussian_process import GaussianProcessRegressor
    if len(y_obs) > max_points:
        idx = np.linspace(0, len(y_obs) - 1, max_points).astype(int)
        x_obs, y_obs = x_obs[idx], y_obs[idx]
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-6,
                                   n_restarts_optimizer=n_restarts)
    gp.fit(x_obs.reshape(-1, 1), y_obs)
    mean, std = gp.predict(x_query.reshape(-1, 1), return_std=True)
    return mean, std, float(gp.log_marginal_likelihood_value_)


def _t08(ctx):
    """Moyenne/écart-type d'un GP Matérn conditionné par les observations, par blocs (couverture pleine série)."""
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    values, obs, pos, n = ctx["values"], ctx["observed"], ctx["pos"], ctx["n"]
    c, s = ctx["center"], ctx["scale"]
    out = np.zeros((n, 2))
    for lo, hi, clo, chi in _t_blocks(n, 144, 60):
        m = obs[clo:chi]
        if m.sum() < 10:
            continue
        x_obs, y_obs = pos[clo:chi][m].astype(float), values[clo:chi][m]
        try:
            kernel = ConstantKernel(s ** 2) * Matern(length_scale=20.0, nu=1.5) + WhiteKernel(0.05 * s ** 2)
            mean, std, _ = _gp_window_predict(x_obs, y_obs, pos[lo:hi].astype(float), kernel)
        except Exception:
            continue
        out[lo:hi, 0] = (mean - c) / s
        out[lo:hi, 1] = std / s
    return out


def _t09(ctx):
    """GP périodique x Matérn (quasi-périodique) : valeur, variance, vraisemblance locale."""
    from sklearn.gaussian_process.kernels import ConstantKernel, ExpSineSquared, Matern, WhiteKernel
    values, obs, pos, n = ctx["values"], ctx["observed"], ctx["pos"], ctx["n"]
    c, s = ctx["center"], ctx["scale"]
    out = np.zeros((n, 3))
    for lo, hi, clo, chi in _t_blocks(n, 144, 60):
        m = obs[clo:chi]
        if m.sum() < 10:
            continue
        x_obs, y_obs = pos[clo:chi][m].astype(float), values[clo:chi][m]
        try:
            kernel = (ConstantKernel(s ** 2) * ExpSineSquared(length_scale=5.0, periodicity=48.0)
                      * Matern(length_scale=100.0, nu=1.5) + WhiteKernel(0.05 * s ** 2))
            mean, std, lml = _gp_window_predict(x_obs, y_obs, pos[lo:hi].astype(float), kernel)
        except Exception:
            continue
        out[lo:hi, 0] = (mean - c) / s
        out[lo:hi, 1] = std / s
        out[lo:hi, 2] = lml / max(m.sum(), 1)
    return out


def _t10(ctx):
    """Poids de modèles par comparaison de vraisemblance entre kernels lisse/périodique/quasi-périodique."""
    from sklearn.gaussian_process.kernels import ConstantKernel, ExpSineSquared, Matern, WhiteKernel
    values, obs, pos, n = ctx["values"], ctx["observed"], ctx["pos"], ctx["n"]
    s = ctx["scale"]
    out = np.zeros((n, 3))
    for lo, hi, clo, chi in _t_blocks(n, 144, 60):
        m = obs[clo:chi]
        if m.sum() < 10:
            continue
        x_obs, y_obs = pos[clo:chi][m].astype(float), values[clo:chi][m]
        lmls = []
        kernels = [ConstantKernel(s ** 2) * Matern(length_scale=30.0, nu=1.5) + WhiteKernel(0.05 * s ** 2),
                   ConstantKernel(s ** 2) * ExpSineSquared(length_scale=5.0, periodicity=48.0) + WhiteKernel(0.05 * s ** 2),
                   ConstantKernel(s ** 2) * ExpSineSquared(length_scale=5.0, periodicity=48.0)
                   * Matern(length_scale=100.0, nu=1.5) + WhiteKernel(0.05 * s ** 2)]
        for kernel in kernels:
            try:
                _, _, lml = _gp_window_predict(x_obs, y_obs, pos[lo:lo + 1].astype(float), kernel, n_restarts=0)
                lmls.append(lml / max(m.sum(), 1))
            except Exception:
                lmls.append(-1e3)
        w = np.exp(np.asarray(lmls) - np.max(lmls))
        w = w / w.sum()
        out[lo:hi] = w
    return out


# --- T11--T20 : décompositions multi-échelles et signal --------------------

def _wavelet_decompose(y, level=4, wl="db4"):
    import pywt
    n = len(y)
    maxlev = pywt.dwt_max_level(n, pywt.Wavelet(wl).dec_len)
    level = max(1, min(level, maxlev))
    return pywt.wavedec(y, wl, level=level), wl, n


def _wavelet_band(coeffs, wl, n, i):
    import pywt
    z = [c if j == i else np.zeros_like(c) for j, c in enumerate(coeffs)]
    return pywt.waverec(z, wl)[:n]


def _t11(ctx):
    """Reconstructions ondelettes approximation/détails à plusieurs échelles."""
    y, n, c, s = ctx["interp"], ctx["n"], ctx["center"], ctx["scale"]
    coeffs, wl, _ = _wavelet_decompose(y)
    bands = [(_wavelet_band(coeffs, wl, n, i) - c) / s for i in range(len(coeffs))]
    while len(bands) < 5:
        bands.append(np.zeros(n))
    return np.column_stack(bands[:5])


def _t12(ctx):
    """Énergie de chaque bande (rapide/moyenne/lente) avant et après le trou."""
    y, n, s = ctx["interp"], ctx["n"], ctx["scale"]
    li, ri = ctx["li"], ctx["ri"]
    coeffs, wl, _ = _wavelet_decompose(y)
    idx_groups = [list(range(max(1, len(coeffs) - 2), len(coeffs))),
                  [len(coeffs) - 3] if len(coeffs) >= 3 else [],
                  [0]]
    out = np.zeros((n, 6))
    col = 0
    for idxs in idx_groups:
        recon = np.zeros(n)
        for i in idxs:
            recon += _wavelet_band(coeffs, wl, n, i)
        e = (recon / s) ** 2
        before = pd.Series(e).rolling(24, min_periods=1).mean().to_numpy()
        after = pd.Series(e[::-1]).rolling(24, min_periods=1).mean().to_numpy()[::-1]
        out[:, col] = np.where(li >= 0, before[np.clip(li, 0, n - 1)], 0.0); col += 1
        out[:, col] = np.where(ri < n, after[np.clip(ri, 0, n - 1)], 0.0); col += 1
    return out


def _t13(ctx):
    """Cohérence ondelette locale (corrélation glissante par bande) entre cible et consensus donneurs."""
    N, n = ctx["N"], ctx["n"]
    if N.shape[1] == 0:
        return np.zeros((n, 4))
    y = ctx["interp"]
    consensus = np.median(N, axis=1)
    cy, wl, _ = _wavelet_decompose(y, level=3)
    cc, _, _ = _wavelet_decompose(consensus, level=3)
    out = []
    for i in range(min(4, len(cy), len(cc))):
        ry = _wavelet_band(cy, wl, n, i)
        rc = _wavelet_band(cc, wl, n, i)
        coh = pd.Series(ry).rolling(96, min_periods=16).corr(pd.Series(rc)).to_numpy()
        out.append(np.nan_to_num(coh))
    while len(out) < 4:
        out.append(np.zeros(n))
    return np.column_stack(out)


def _t14(ctx):
    """Déphasage cible-donneur (corrélation croisée à petits décalages), bandes lente et rapide."""
    N, n = ctx["N"], ctx["n"]
    if N.shape[1] == 0:
        return np.zeros((n, 2))
    y = ctx["interp"]; consensus = np.median(N, axis=1)
    slow_y = pd.Series(y).rolling(48, min_periods=1, center=True).mean().to_numpy()
    slow_c = pd.Series(consensus).rolling(48, min_periods=1, center=True).mean().to_numpy()
    fast_y, fast_c = y - slow_y, consensus - slow_c
    out = np.zeros((n, 2))
    for lo, hi, clo, chi in _t_blocks(n, 144, 48):
        for band_idx, (a, b) in enumerate(((slow_y, slow_c), (fast_y, fast_c))):
            seg_a, seg_b = a[clo:chi], b[clo:chi]
            best_lag, best_corr = 0, -np.inf
            for lag in range(-4, 5):
                if lag >= 0:
                    sa, sb = seg_a[lag:], seg_b[:len(seg_b) - lag] if lag else seg_b
                else:
                    sa, sb = seg_a[:len(seg_a) + lag], seg_b[-lag:]
                if len(sa) < 10 or np.std(sa) < 1e-9 or np.std(sb) < 1e-9:
                    continue
                cc = float(np.corrcoef(sa, sb)[0, 1])
                if np.isfinite(cc) and cc > best_corr:
                    best_corr, best_lag = cc, lag
            out[lo:hi, band_idx] = best_lag
    return out


def _ssa_fill(w, L, r, iters=8):
    """SSA itérative pour série trouée (Kondrashov & Ghil) : trajectoire de
    Hankel + SVD tronquée + moyennage diagonal, en ré-imputant seulement les
    positions manquantes à chaque itération (les positions observées restent
    fixées à leur vraie valeur -- reconstruction, pas de fuite ajoutée par
    rapport à ce qui est déjà observé dans la fenêtre)."""
    n = len(w)
    L = max(4, min(L, n // 2))
    K = n - L + 1
    if K < L or r < 1:
        return w.copy()
    miss = ~np.isfinite(w)
    filled = w.copy()
    fill_val = np.nanmedian(w) if np.isfinite(np.nanmedian(w)) else 0.0
    filled[miss] = fill_val
    rr = min(r, L, K)
    for _ in range(iters):
        traj = np.lib.stride_tricks.sliding_window_view(filled, L).T  # (L, K)
        try:
            U, S, Vt = np.linalg.svd(traj, full_matrices=False)
        except np.linalg.LinAlgError:
            break
        approx = (U[:, :rr] * S[:rr]) @ Vt[:rr, :]
        recon = np.zeros(n); cnt = np.zeros(n)
        for k in range(K):
            recon[k:k + L] += approx[:, k]; cnt[k:k + L] += 1
        recon /= np.maximum(cnt, 1)
        filled = np.where(miss, recon, w)
    return filled


def _t15(ctx):
    """SSA récurrente : reconstruction par bloc à partir du contexte gauche
    (avant) et droit (arrière), rang fixé, erreur de reconstruction locale."""
    y, missing, n, c, s = ctx["interp"], ctx["missing"], ctx["n"], ctx["center"], ctx["scale"]
    out = np.zeros((n, 4))
    for lo, hi, clo, chi in _t_blocks(n, 144, 72):
        w = np.where(missing[clo:chi], np.nan, y[clo:chi])
        rec = _ssa_fill(w, L=min(48, (chi - clo) // 3), r=5)
        core = rec[lo - clo:hi - clo]
        out[lo:hi, 0] = (core - c) / s
        # variante "arrière" : reconstruction sur la fenêtre inversée
        rec_b = _ssa_fill(w[::-1], L=min(48, (chi - clo) // 3), r=5)[::-1]
        core_b = rec_b[lo - clo:hi - clo]
        out[lo:hi, 1] = (core_b - c) / s
        out[lo:hi, 2] = float(min(48, (chi - clo) // 3))
        obs_here = ~missing[lo:hi]
        if obs_here.any():
            err = float(np.mean(np.abs(core[obs_here] - y[lo:hi][obs_here]))) / s
        else:
            err = 0.0
        out[lo:hi, 3] = err
    return out


def _t16(ctx):
    """MSSA cible+donneurs (2 canaux) : projection cible sur le sous-espace commun, résidu."""
    N, y, missing, n, c, s = ctx["N"], ctx["interp"], ctx["missing"], ctx["n"], ctx["center"], ctx["scale"]
    out = np.zeros((n, 2))
    if N.shape[1] == 0:
        return out
    consensus = np.median(N, axis=1)
    for lo, hi, clo, chi in _t_blocks(n, 144, 72):
        L = min(24, (chi - clo) // 4)
        if L < 4:
            continue
        wt = np.where(missing[clo:chi], np.nan, y[clo:chi])
        wt_filled = pd.Series(wt).interpolate(limit_direction="both").ffill().bfill().to_numpy()
        wd = consensus[clo:chi]
        K = (chi - clo) - L + 1
        if K < L:
            continue
        traj_t = np.lib.stride_tricks.sliding_window_view(wt_filled, L).T
        traj_d = np.lib.stride_tricks.sliding_window_view(wd, L).T
        joint = np.vstack([traj_t, traj_d])
        try:
            U, S, Vt = np.linalg.svd(joint, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        r = min(4, len(S))
        approx = (U[:, :r] * S[:r]) @ Vt[:r, :]
        approx_t = approx[:L]
        recon = np.zeros(chi - clo); cnt = np.zeros(chi - clo)
        for k in range(K):
            recon[k:k + L] += approx_t[:, k]; cnt[k:k + L] += 1
        recon /= np.maximum(cnt, 1)
        core = recon[lo - clo:hi - clo]
        out[lo:hi, 0] = (core - c) / s
        out[lo:hi, 1] = (wt_filled[lo - clo:hi - clo] - core) / s
    return out


def _emd_sift(y, n_imf=2, max_sift=6):
    """Sifting EMD allégé (enveloppes par spline cubique)."""
    from scipy.interpolate import CubicSpline
    residual = y.copy()
    imfs = []
    n = len(y)
    x = np.arange(n)
    for _ in range(n_imf):
        h = residual.copy()
        for _ in range(max_sift):
            maxima = np.r_[False, (h[1:-1] > h[:-2]) & (h[1:-1] > h[2:]), False]
            minima = np.r_[False, (h[1:-1] < h[:-2]) & (h[1:-1] < h[2:]), False]
            if maxima.sum() < 2 or minima.sum() < 2:
                break
            try:
                upper = CubicSpline(x[maxima], h[maxima], extrapolate=True)(x)
                lower = CubicSpline(x[minima], h[minima], extrapolate=True)(x)
            except Exception:
                break
            mean_env = (upper + lower) / 2
            h = h - mean_env
        imfs.append(h)
        residual = residual - h
    while len(imfs) < n_imf:
        imfs.append(np.zeros(n))
    return imfs, residual


def _t17(ctx):
    """Valeurs des 2 premiers IMF + résidu lent (EMD allégée)."""
    y, n, c, s = ctx["interp"], ctx["n"], ctx["center"], ctx["scale"]
    try:
        imfs, residual = _emd_sift(y, n_imf=2)
    except Exception:
        imfs, residual = [np.zeros(n), np.zeros(n)], y - c
    return np.column_stack([imfs[0] / s, imfs[1] / s, (residual - c) / s])


def _bandpass(y, low, high, fs=1.0, order=3):
    from scipy.signal import butter, sosfiltfilt
    nyq = fs / 2
    lo = max(low / nyq, 1e-4); hi = min(high / nyq, 0.999)
    if lo >= hi:
        return np.zeros_like(y)
    sos = butter(order, [lo, hi], btype="band", output="sos")
    return sosfiltfilt(sos, y)


def _t18(ctx):
    """Bandes approximant une VMD (filtres passe-bande) : intra-journalier, journalier, basse fréquence."""
    y, n, s = ctx["interp"], ctx["n"], ctx["scale"]
    try:
        fast = _bandpass(y, 1 / 12, 1 / 3)
        daily = _bandpass(y, 1 / 60, 1 / 36)
        slow = _bandpass(y, 1 / 800, 1 / 200)
    except Exception:
        fast = daily = slow = np.zeros(n)
    return np.column_stack([fast / s, daily / s, slow / s])


def _t19(ctx):
    """Amplitude/phase/fréquence instantanées (Hilbert) de la bande rapide lissée."""
    from scipy.signal import hilbert
    y, n, s = ctx["interp"], ctx["n"], ctx["scale"]
    try:
        fast = _bandpass(y, 1 / 12, 1 / 3)
        analytic = hilbert(fast)
        amp = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic))
        freq = np.r_[0.0, np.diff(phase)] / (2 * np.pi)
    except Exception:
        amp, phase, freq = np.zeros(n), np.zeros(n), np.zeros(n)
    return np.column_stack([amp / s, np.sin(phase), freq])


def _t20(ctx):
    """Signature cepstrale locale par bloc + distance au bloc précédent."""
    y, n, s = ctx["interp"], ctx["n"], ctx["scale"]
    out = np.zeros((n, 2))
    prev_cep = None
    for lo, hi, clo, chi in _t_blocks(n, 144, 0):
        w = (y[clo:chi] - y[clo:chi].mean()) / s
        spec = np.fft.rfft(w * np.hanning(len(w)))
        log_mag = np.log(np.abs(spec) + 1e-6)
        cep = np.fft.irfft(log_mag)[:6]
        out[lo:hi, 0] = np.linalg.norm(cep[1:])
        if prev_cep is not None:
            out[lo:hi, 1] = np.linalg.norm(cep - prev_cep)
        prev_cep = cep
    return out


# --- T21--T30 : régimes, motifs et complexité -------------------------------

def _bocpd(y, hazard=1.0 / 200, mu0=None, kappa0=1.0, alpha0=2.0, beta0=None, max_run=250):
    """BOCPD (Adams & McKay 2007) simplifié : prédictif Student-t via prior
    Normal-Gamma conjugué, hasard constant, distribution de longueur de
    régime tronquée à `max_run`. Renvoie P(changepoint) et E[longueur],
    Var[longueur] à chaque pas."""
    n = len(y)
    mu0 = float(np.nanmean(y)) if mu0 is None else mu0
    beta0 = float(np.nanvar(y) + 1e-6) if beta0 is None else beta0
    R = np.zeros(max_run + 1); R[0] = 1.0
    mu = np.full(max_run + 1, mu0); kappa = np.full(max_run + 1, kappa0)
    alpha = np.full(max_run + 1, alpha0); beta = np.full(max_run + 1, beta0)
    pcp = np.zeros(n); erun = np.zeros(n); vrun = np.zeros(n)
    from scipy.stats import t as student_t
    for i in range(n):
        active = R > 1e-12
        scale = np.sqrt(beta[active] * (kappa[active] + 1) / (alpha[active] * kappa[active]))
        pred = student_t.pdf(y[i], df=2 * alpha[active], loc=mu[active], scale=np.maximum(scale, 1e-9))
        growth = R[active] * pred * (1 - hazard)
        cp_mass = float(np.sum(R[active] * pred * hazard))
        newR = np.zeros(max_run + 1)
        idxs = np.flatnonzero(active)
        newR[np.minimum(idxs + 1, max_run)] += growth
        newR[0] += cp_mass
        total = newR.sum()
        R_new = newR / total if total > 0 else newR
        run_lens = np.arange(max_run + 1)
        pcp[i] = R_new[0]
        erun[i] = float(np.sum(run_lens * R_new))
        vrun[i] = float(np.sum((run_lens - erun[i]) ** 2 * R_new))
        # mise à jour bayésienne des paramètres pour chaque longueur de régime (décalée +1)
        new_mu = np.zeros(max_run + 1); new_kappa = np.zeros(max_run + 1)
        new_alpha = np.zeros(max_run + 1); new_beta = np.zeros(max_run + 1)
        new_mu[0], new_kappa[0], new_alpha[0], new_beta[0] = mu0, kappa0, alpha0, beta0
        shifted = np.minimum(idxs + 1, max_run)
        new_mu[shifted] = (kappa[idxs] * mu[idxs] + y[i]) / (kappa[idxs] + 1)
        new_kappa[shifted] = kappa[idxs] + 1
        new_alpha[shifted] = alpha[idxs] + 0.5
        new_beta[shifted] = beta[idxs] + kappa[idxs] * (y[i] - mu[idxs]) ** 2 / (2 * (kappa[idxs] + 1))
        R, mu, kappa, alpha, beta = R_new, new_mu, new_kappa, new_alpha, new_beta
    return pcp, erun, vrun


def _t21(ctx):
    """BOCPD : probabilité de rupture, longueur de régime espérée + incertitude."""
    y, n = ctx["interp"], ctx["n"]
    try:
        z = (y - np.median(y)) / (ctx["scale"] + 1e-9)
        pcp, erun, vrun = _bocpd(z, max_run=min(250, n - 1))
    except Exception:
        pcp, erun, vrun = np.zeros(n), np.zeros(n), np.zeros(n)
    return np.column_stack([pcp, erun / 100.0, np.sqrt(np.maximum(vrun, 0)) / 100.0])


def _t22(ctx):
    """Segmentation pénalisée (PELT) : niveau, pente, variance et âge du segment."""
    import ruptures as rpt
    y, n, c, s = ctx["interp"], ctx["n"], ctx["center"], ctx["scale"]
    out = np.zeros((n, 4))
    try:
        algo = rpt.Pelt(model="l2", min_size=12, jump=4).fit(y.reshape(-1, 1))
        bkps = algo.predict(pen=np.log(n) * np.var(y) * 2 + 1e-6)
    except Exception:
        bkps = [n]
    prev = 0
    for b in bkps:
        seg = y[prev:b]
        if len(seg) >= 2:
            level = float(np.median(seg))
            slope = float((seg[-1] - seg[0]) / max(len(seg) - 1, 1))
            var = float(np.var(seg))
        else:
            level, slope, var = float(seg[0]) if len(seg) else c, 0.0, 0.0
        age = np.arange(b - prev)
        out[prev:b, 0] = (level - c) / s
        out[prev:b, 1] = slope / s
        out[prev:b, 2] = var / s ** 2
        out[prev:b, 3] = np.log1p(age)
        prev = b
    return out


def _cusum(z, w, k=0.5):
    roll_mean = pd.Series(z).rolling(w, min_periods=8).mean().to_numpy()
    roll_std = pd.Series(z).rolling(w, min_periods=8).std().to_numpy() + 1e-6
    roll_mean = np.nan_to_num(roll_mean); roll_std = np.nan_to_num(roll_std, nan=1.0)
    zz = (z - roll_mean) / roll_std
    n = len(z); pos = np.zeros(n); neg = np.zeros(n)
    for i in range(1, n):
        pos[i] = max(0.0, pos[i - 1] + zz[i] - k)
        neg[i] = min(0.0, neg[i - 1] + zz[i] + k)
    return pos, neg


def _t23(ctx):
    """CUSUM positif/négatif à plusieurs horizons (48, 96, 336)."""
    y, n = ctx["interp"], ctx["n"]
    cols = []
    for w in (48, 96, 336):
        pos, neg = _cusum(y, w)
        cols.extend([pos, neg])
    return np.column_stack(cols)


def _matrix_profile(y, m=48):
    import stumpy
    mp = stumpy.stump(y.astype(np.float64), m=m)
    dist = mp[:, 0].astype(float)
    idx = mp[:, 1].astype(float)
    return dist, idx, m


def _t24(ctx):
    """Matrix Profile (motif) : distance au plus proche voisin + valeur projetée depuis ce motif."""
    y, n, c, s = ctx["interp"], ctx["n"], ctx["center"], ctx["scale"]
    out = np.zeros((n, 2))
    m = 48
    if n <= m + 4:
        return out
    try:
        dist, idx, m = _matrix_profile(y, m)
    except Exception:
        return out
    nsub = len(dist)
    si = np.clip(ctx["pos"] - m // 2, 0, nsub - 1)
    nn = idx[si].astype(int)
    offset = ctx["pos"] - si
    proj_pos = np.clip(nn + offset, 0, n - 1)
    out[:, 0] = dist[si] / s
    out[:, 1] = (y[proj_pos] - c) / s
    return out


def _t25(ctx):
    """Matrix Profile (discordance) : score de caractère atypique local."""
    y, n, s = ctx["interp"], ctx["n"], ctx["scale"]
    out = np.zeros((n, 2))
    m = 48
    if n <= m + 4:
        return out
    try:
        dist, idx, m = _matrix_profile(y, m)
    except Exception:
        return out
    nsub = len(dist)
    z = (dist - np.median(dist)) / (np.std(dist) + 1e-9)
    rank = pd.Series(dist).rank(pct=True).to_numpy()
    si = np.clip(ctx["pos"] - m // 2, 0, nsub - 1)
    out[:, 0] = z[si]
    out[:, 1] = rank[si]
    return out


def _rqa_block(w, threshold_q=0.1):
    n = len(w)
    if n < 8:
        return 0.0, 0.0, 0.0
    d = np.abs(w[:, None] - w[None, :])
    eps = np.quantile(d, threshold_q)
    rec = (d <= eps).astype(int)
    np.fill_diagonal(rec, 0)
    rr = rec.mean()
    diag_lengths = []
    for k in range(-(n - 2), n - 1):
        line = np.diagonal(rec, k)
        run = 0
        for v in line:
            if v:
                run += 1
            elif run:
                diag_lengths.append(run); run = 0
        if run:
            diag_lengths.append(run)
    diag_lengths = np.asarray(diag_lengths)
    det = diag_lengths[diag_lengths >= 2].sum() / max(rec.sum(), 1)
    vert_lengths = []
    for j in range(n):
        col = rec[:, j]; run = 0
        for v in col:
            if v:
                run += 1
            elif run:
                vert_lengths.append(run); run = 0
        if run:
            vert_lengths.append(run)
    vert_lengths = np.asarray(vert_lengths)
    lam = vert_lengths[vert_lengths >= 2].sum() / max(rec.sum(), 1)
    return float(rr), float(det), float(lam)


def _t26(ctx):
    """Récurrence : taux de récurrence, déterminisme, laminarité par bloc."""
    y, n = ctx["interp"], ctx["n"]
    out = np.zeros((n, 3))
    for lo, hi, clo, chi in _t_blocks(n, 96, 32):
        w = y[clo:chi]
        if len(w) > 60:  # borne le coût O(w^2)
            idx = np.linspace(0, len(w) - 1, 60).astype(int)
            w = w[idx]
        try:
            rr, det, lam = _rqa_block(w)
        except Exception:
            rr, det, lam = 0.0, 0.0, 0.0
        out[lo:hi] = [rr, det, lam]
    return out


def _t27(ctx):
    """Entropie de permutation multi-échelle (délais 1 et 6) par bloc."""
    import antropy as ant
    y, n = ctx["interp"], ctx["n"]
    out = np.zeros((n, 2))
    for lo, hi, clo, chi in _t_blocks(n, 96, 48):
        w = y[clo:chi]
        for j, delay in enumerate((1, 6)):
            try:
                out[lo:hi, j] = ant.perm_entropy(w, order=3, delay=delay, normalize=True)
            except Exception:
                out[lo:hi, j] = 0.0
    return out


def _t28(ctx):
    """Entropie d'échantillon multi-échelle (brute + sous-échantillonnée x2) par bloc."""
    import antropy as ant
    y, n = ctx["interp"], ctx["n"]
    out = np.zeros((n, 2))
    for lo, hi, clo, chi in _t_blocks(n, 96, 48):
        w = y[clo:chi]
        try:
            out[lo:hi, 0] = ant.sample_entropy(w, order=2)
        except Exception:
            pass
        try:
            out[lo:hi, 1] = ant.sample_entropy(w[::2], order=2)
        except Exception:
            pass
    return np.nan_to_num(out)


def _hurst_rs(w):
    n = len(w)
    if n < 16:
        return 0.5
    z = w - w.mean()
    y = np.cumsum(z)
    rng = y.max() - y.min()
    std = w.std()
    if std < 1e-9 or rng <= 0:
        return 0.5
    return float(np.log(rng / std) / np.log(n))


def _t29(ctx):
    """Exposant de Hurst (R/S) + pente DFA locale par bloc."""
    import antropy as ant
    y, n = ctx["interp"], ctx["n"]
    out = np.zeros((n, 2))
    for lo, hi, clo, chi in _t_blocks(n, 96, 48):
        w = y[clo:chi]
        out[lo:hi, 0] = _hurst_rs(w)
        try:
            out[lo:hi, 1] = ant.detrended_fluctuation(w)
        except Exception:
            out[lo:hi, 1] = 0.5
    return np.nan_to_num(out, nan=0.5)


def _t30(ctx):
    """Visibility graph : degré, coefficient de clustering, closeness du point central, par fenêtre glissante."""
    import networkx as nx
    y, n = ctx["interp"], ctx["n"]
    out = np.zeros((n, 3))
    step, half = 16, 24
    for center in range(0, n, step):
        lo, hi = max(0, center - half), min(n, center + half)
        w = y[lo:hi]
        m = len(w)
        if m < 6:
            continue
        G = nx.Graph()
        G.add_nodes_from(range(m))
        for i in range(m):
            for j in range(i + 1, m):
                if j == i + 1:
                    G.add_edge(i, j); continue
                seg = w[i + 1:j]
                if seg.size == 0:
                    G.add_edge(i, j); continue
                line = w[i] + (w[j] - w[i]) * (np.arange(i + 1, j) - i) / (j - i)
                if np.all(seg < line):
                    G.add_edge(i, j)
        ci = center - lo
        deg = G.degree(ci) / max(m - 1, 1)
        clus = nx.clustering(G, ci)
        try:
            close = nx.closeness_centrality(G, ci)
        except Exception:
            close = 0.0
        out[center:min(n, center + step)] = [deg, clus, close]
    return out


# --- T31--T40 : dépendances entre courbes et graphes ------------------------

def _weighted_consensus(N, weights):
    w = np.asarray(weights, float)
    if np.sum(np.abs(w)) < 1e-12:
        return np.median(N, axis=1)
    w = w / (np.sum(np.abs(w)) + 1e-12)
    return N @ w


def _t31(ctx):
    """Voisins par corrélation partielle après retrait du 1er facteur commun ('marché')."""
    N, values, obs, n = ctx["N"], ctx["values"], ctx["observed"], ctx["n"]
    if N.shape[1] == 0:
        return np.zeros((n, 3))
    Nc = N - N.mean(0)
    try:
        _, _, Vt = np.linalg.svd(Nc[obs], full_matrices=False)
        pc1 = Vt[0]
        factor = N @ pc1
    except Exception:
        factor = np.median(N, axis=1)
    y = values[obs]; f = factor[obs]
    beta_y = float(np.dot(f - f.mean(), y - y.mean()) / (np.dot(f - f.mean(), f - f.mean()) + 1e-9))
    y_resid_full = values - factor * beta_y
    donor_resid = N - factor[:, None] * (np.dot(f - f.mean(), Nc[obs]) / (np.dot(f - f.mean(), f - f.mean()) + 1e-9))
    yr = y_resid_full[obs]
    Xr = donor_resid[obs]
    Xc = Xr - Xr.mean(0)
    denom = np.sqrt((yr - yr.mean()) @ (yr - yr.mean())) * np.sqrt((Xc ** 2).sum(0)) + 1e-9
    pcorr = (Xc.T @ (yr - yr.mean())) / denom
    consensus = _weighted_consensus(N, pcorr)
    c, s = ctx["center"], ctx["scale"]
    return np.column_stack([(consensus - c) / s, np.full(n, np.mean(np.abs(pcorr))),
                             np.full(n, np.max(np.abs(pcorr)) if len(pcorr) else 0.0)])


def _t32(ctx):
    """Agrégat des voisins conditionnels issus d'une matrice de précision sparse (Graphical Lasso)."""
    from sklearn.covariance import GraphicalLasso
    N, values, obs, n = ctx["N"], ctx["values"], ctx["observed"], ctx["n"]
    if N.shape[1] < 3:
        return np.zeros((n, 2))
    k = min(15, N.shape[1])
    corr_full = np.array([np.corrcoef(values[obs], N[obs, j])[0, 1] if np.std(N[obs, j]) > 1e-9 else 0.0
                           for j in range(N.shape[1])])
    top = np.argsort(-np.abs(np.nan_to_num(corr_full)))[:k]
    M = np.column_stack([values, N[:, top]])[obs]
    M = (M - M.mean(0)) / (M.std(0) + 1e-9)
    try:
        gl = GraphicalLasso(alpha=0.3, max_iter=100).fit(M)
        prec = gl.precision_
        w = -prec[0, 1:] / (prec[0, 0] + 1e-9)
    except Exception:
        w = np.zeros(k)
    consensus = _weighted_consensus(N[:, top], w)
    c, s = ctx["center"], ctx["scale"]
    return np.column_stack([(consensus - c) / s, np.full(n, float(np.sum(np.abs(w) > 1e-6)))])


def _t33(ctx):
    """Sélection par information mutuelle non linéaire + consensus pondéré."""
    from sklearn.feature_selection import mutual_info_regression
    N, values, obs, n = ctx["N"], ctx["values"], ctx["observed"], ctx["n"]
    if N.shape[1] == 0:
        return np.zeros((n, 2))
    try:
        mi = mutual_info_regression(N[obs], values[obs], random_state=0)
    except Exception:
        mi = np.zeros(N.shape[1])
    consensus = _weighted_consensus(N, mi)
    c, s = ctx["center"], ctx["scale"]
    return np.column_stack([(consensus - c) / s, np.full(n, float(np.max(mi) if len(mi) else 0.0))])


def _dcor(x, y):
    """Distance-correlation (Székely-Rizzo), échantillon borné pour le coût O(n^2)."""
    n = len(x)
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    A = a - a.mean(0)[None, :] - a.mean(1)[:, None] + a.mean()
    B = b - b.mean(0)[None, :] - b.mean(1)[:, None] + b.mean()
    dcov2 = (A * B).mean()
    dvarx = (A * A).mean(); dvary = (B * B).mean()
    denom = np.sqrt(max(dvarx * dvary, 0.0))
    return float(np.sqrt(max(dcov2, 0.0) / denom)) if denom > 1e-12 else 0.0


def _t34(ctx):
    """Poids fondés sur la distance-correlation (dépendances non linéaires)."""
    N, values, obs, n = ctx["N"], ctx["values"], ctx["observed"], ctx["n"]
    if N.shape[1] == 0:
        return np.zeros((n, 2))
    obs_idx = np.flatnonzero(obs)
    if len(obs_idx) > 250:
        obs_idx = np.random.RandomState(0).choice(obs_idx, 250, replace=False)
    y = values[obs_idx]
    k = min(15, N.shape[1])
    corr_full = np.array([np.corrcoef(values[obs], N[obs, j])[0, 1] if np.std(N[obs, j]) > 1e-9 else 0.0
                           for j in range(N.shape[1])])
    top = np.argsort(-np.abs(np.nan_to_num(corr_full)))[:k]
    dcors = []
    for j in top:
        try:
            dcors.append(_dcor(y, N[obs_idx, j]))
        except Exception:
            dcors.append(0.0)
    dcors = np.asarray(dcors)
    consensus = _weighted_consensus(N[:, top], dcors)
    c, s = ctx["center"], ctx["scale"]
    return np.column_stack([(consensus - c) / s, np.full(n, float(np.max(dcors) if len(dcors) else 0.0))])


def _t35(ctx):
    """Dépendance de queue (copule empirique) sup/inf cible-donneur + prédiction conditionnelle pondérée par la queue."""
    N, values, obs, n = ctx["N"], ctx["values"], ctx["observed"], ctx["n"]
    if N.shape[1] == 0:
        return np.zeros((n, 3))
    corr_full = np.array([np.corrcoef(values[obs], N[obs, j])[0, 1] if np.std(N[obs, j]) > 1e-9 else 0.0
                           for j in range(N.shape[1])])
    k = min(8, N.shape[1])
    top = np.argsort(-np.abs(np.nan_to_num(corr_full)))[:k]
    ry = pd.Series(values[obs]).rank(pct=True).to_numpy()
    lam_u, lam_l, tw = [], [], []
    q = 0.8
    for j in top:
        rd = pd.Series(N[obs, j]).rank(pct=True).to_numpy()
        upper_y, upper_d = ry > q, rd > q
        lower_y, lower_d = ry < 1 - q, rd < 1 - q
        lu = float(np.sum(upper_y & upper_d) / max(np.sum(upper_d), 1))
        ll = float(np.sum(lower_y & lower_d) / max(np.sum(lower_d), 1))
        lam_u.append(lu); lam_l.append(ll); tw.append(lu + ll)
    tw = np.asarray(tw)
    consensus = _weighted_consensus(N[:, top], tw)
    c, s = ctx["center"], ctx["scale"]
    return np.column_stack([np.full(n, np.mean(lam_u) if lam_u else 0.0),
                             np.full(n, np.mean(lam_l) if lam_l else 0.0), (consensus - c) / s])


def _t36(ctx):
    """Spread cible-donneur (cointégration Engle-Granger sur le top donneur) : z-score + correction long terme."""
    N, values, obs, n = ctx["N"], ctx["values"], ctx["observed"], ctx["n"]
    if N.shape[1] == 0:
        return np.zeros((n, 2))
    corr_full = np.array([np.corrcoef(values[obs], N[obs, j])[0, 1] if np.std(N[obs, j]) > 1e-9 else 0.0
                           for j in range(N.shape[1])])
    top = int(np.argmax(np.abs(np.nan_to_num(corr_full))))
    donor = N[:, top]
    x, y = donor[obs], values[obs]
    beta = float(np.dot(x - x.mean(), y - y.mean()) / (np.dot(x - x.mean(), x - x.mean()) + 1e-9))
    # `interp` (jamais `values`) pour avoir une valeur définie à TOUTES les
    # lignes -- sinon le spread est NaN exactement aux trous, là où la
    # feature est le plus utile.
    spread_full = ctx["interp"] - beta * donor
    spread_obs_mean = float(np.mean(spread_full[obs]))
    spread_obs_std = float(np.std(spread_full[obs]) + 1e-9)
    z = (spread_full - spread_obs_mean) / spread_obs_std
    correction = pd.Series(spread_full).rolling(96, min_periods=8, center=True).mean().to_numpy()
    correction = np.nan_to_num(correction, nan=spread_obs_mean)
    return np.column_stack([z, (correction - spread_obs_mean) / spread_obs_std])


def _granger_f(y, x, lag=2):
    """F-test de causalité de Granger allégé (OLS manuel plutôt que
    statsmodels, pour rester rapide sur de nombreux donneurs)."""
    n = len(y)
    if n < 4 * lag + 10:
        return 0.0
    Ylag = np.column_stack([y[lag - k:n - k] for k in range(1, lag + 1)])
    Xlag = np.column_stack([x[lag - k:n - k] for k in range(1, lag + 1)])
    target = y[lag:]
    X_r = np.column_stack([np.ones(len(target)), Ylag])
    X_u = np.column_stack([np.ones(len(target)), Ylag, Xlag])
    br, *_ = np.linalg.lstsq(X_r, target, rcond=None)
    bu, *_ = np.linalg.lstsq(X_u, target, rcond=None)
    rss_r = float(np.sum((target - X_r @ br) ** 2))
    rss_u = float(np.sum((target - X_u @ bu) ** 2))
    df1, df2 = lag, len(target) - X_u.shape[1]
    if rss_u < 1e-9 or df2 <= 0:
        return 0.0
    f = ((rss_r - rss_u) / df1) / (rss_u / df2)
    return float(max(f, 0.0))


def _t37(ctx):
    """Scores de causalité de Granger (donneur->cible, cible->donneur) sur les top-2 donneurs, consensus pondéré."""
    N, values, obs, n = ctx["N"], ctx["values"], ctx["observed"], ctx["n"]
    if N.shape[1] == 0:
        return np.zeros((n, 2))
    y = ctx["interp"]
    corr_full = np.array([np.corrcoef(values[obs], N[obs, j])[0, 1] if np.std(N[obs, j]) > 1e-9 else 0.0
                           for j in range(N.shape[1])])
    k = min(2, N.shape[1])
    top = np.argsort(-np.abs(np.nan_to_num(corr_full)))[:k]
    fwd, bwd = [], []
    for j in top:
        d = N[:, j]
        try:
            fwd.append(_granger_f(y, d))
            bwd.append(_granger_f(d, y))
        except Exception:
            fwd.append(0.0); bwd.append(0.0)
    fwd = np.asarray(fwd)
    consensus = _weighted_consensus(N[:, top], fwd)
    c, s = ctx["center"], ctx["scale"]
    return np.column_stack([(consensus - c) / s, np.full(n, float(np.mean(fwd) - np.mean(bwd)) if len(fwd) else 0.0)])


def _dtw_align(a, b, band=4):
    """DTW bornée en bande (petites séries journalières -> rapide)."""
    na, nb = len(a), len(b)
    D = np.full((na + 1, nb + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, na + 1):
        for j in range(max(1, i - band), min(nb, i + band) + 1):
            cost = abs(a[i - 1] - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    i, j = na, nb
    path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        choices = [D[i - 1, j], D[i, j - 1], D[i - 1, j - 1]]
        m = int(np.argmin(choices))
        if m == 0: i -= 1
        elif m == 1: j -= 1
        else: i -= 1; j -= 1
    return D[na, nb], path[::-1]


def _t38(ctx):
    """DTW journalière cible/consensus donneurs (bande +/-4) : distance + valeur donneur réalignée."""
    N, y, missing, phase, n = ctx["N"], ctx["interp"], ctx["missing"], ctx["phase"], ctx["n"]
    out = np.zeros((n, 2))
    if N.shape[1] == 0:
        return out
    consensus = np.median(N, axis=1)
    starts = np.flatnonzero(phase == 0)
    for st in starts:
        en = min(n, st + 48)
        a, b = y[st:en], consensus[st:en]
        if len(a) < 8:
            continue
        try:
            dist, path = _dtw_align(a, b, band=4)
        except Exception:
            continue
        realigned = np.array(a, copy=True)
        for i, j in path:
            realigned[i] = b[j]
        out[st:en, 0] = dist / max(len(a), 1) / (ctx["scale"] + 1e-9)
        out[st:en, 1] = (realigned - ctx["center"]) / ctx["scale"]
    return out


def _t39(ctx):
    """Distance de Wasserstein par phase de journée + mapping quantile donneur->cible."""
    from scipy.stats import wasserstein_distance
    N, values, obs, phase, n = ctx["N"], ctx["values"], ctx["observed"], ctx["phase"], ctx["n"]
    if N.shape[1] == 0:
        return np.zeros((n, 2))
    consensus = np.median(N, axis=1)
    c, s = ctx["center"], ctx["scale"]
    out = np.zeros((n, 2))
    for ph in range(48):
        rows = np.flatnonzero(phase == ph)
        if len(rows) == 0:
            continue
        obs_rows = rows[obs[rows]]
        target_dist = values[obs_rows] if len(obs_rows) >= 5 else values[obs]
        donor_dist = consensus[rows]
        try:
            wd = wasserstein_distance(target_dist, donor_dist)
        except Exception:
            wd = 0.0
        out[rows, 0] = wd / (s + 1e-9)
        if len(target_dist) >= 5:
            tq = np.sort(target_dist)
            dq = np.sort(donor_dist) if len(donor_dist) >= 2 else donor_dist
            ranks = pd.Series(consensus[rows]).rank(pct=True).to_numpy() if len(donor_dist) >= 2 else np.full(len(rows), 0.5)
            mapped = np.interp(ranks, np.linspace(0, 1, len(tq)), tq)
            out[rows, 1] = (mapped - c) / s
        else:
            out[rows, 1] = (donor_dist - c) / s
    return out


def _t40(ctx):
    """Reconstruction harmonique sur graphe donneurs (Laplacien local, style LLE) par bloc."""
    N, values, obs, n = ctx["N"], ctx["values"], ctx["observed"], ctx["n"]
    c, s = ctx["center"], ctx["scale"]
    out = np.zeros((n, 2))
    if N.shape[1] < 3:
        return out
    for lo, hi, clo, chi in _t_blocks(n, 144, 48):
        m = obs[clo:chi]
        if m.sum() < 8:
            continue
        Xd = N[clo:chi][m]
        y = values[clo:chi][m]
        k = min(10, Xd.shape[1])
        corr = np.array([np.corrcoef(y, Xd[:, j])[0, 1] if np.std(Xd[:, j]) > 1e-9 else 0.0 for j in range(Xd.shape[1])])
        top = np.argsort(-np.abs(np.nan_to_num(corr)))[:k]
        Z = Xd[:, top] - y[:, None]
        G = Z.T @ Z + 1e-3 * np.trace(Z.T @ Z + 1e-9) * np.eye(k) / max(k, 1)
        try:
            w = np.linalg.solve(G, np.ones(k))
        except np.linalg.LinAlgError:
            w = np.ones(k)
        w = w / (w.sum() + 1e-9)
        recon_block = N[lo:hi][:, top] @ w
        out[lo:hi, 0] = (recon_block - c) / s
        out[lo:hi, 1] = (recon_block - np.median(N[lo:hi], axis=1)) / s
    return out


# --- T41--T50 : complétion globale, incertitude et méta-prédictions --------
#
# Règle stricte (rappelée en tête de section) : le générateur n'est ajusté
# QUE sur les lignes observées, et sa valeur à CES MÊMES lignes est produite
# en LOOCV analytique (jamais en réutilisant la ligne courante), pour ne pas
# fabriquer un gain artificiel par fuite de cible. `_ridge_oof_basis`
# implémente ce motif une seule fois : régression ridge d'un vecteur cible
# sur une base fixe (issue des donneurs, jamais de la cible manquante), avec
# LOOCV via la diagonale de la matrice chapeau (pas de refit ligne par ligne
# -- donc peu coûteux même répété pour 10 groupes).

def _ridge_oof_basis(basis, target, obs_mask, alpha=1.0):
    """Ridge de `target` sur `basis`, ajustée aux lignes `obs_mask`
    uniquement. Renvoie (oof, fitted_all, resid_std) où `oof` est le
    meilleur estimateur sans fuite disponible à CHAQUE ligne : prédiction
    LOOCV analytique aux lignes observées (matrice chapeau), prédiction
    normale (ajustée sur toutes les lignes observées) aux lignes
    manquantes."""
    n, p = basis.shape
    fitted_all = np.full(n, float(np.nanmean(target[obs_mask])) if obs_mask.any() else 0.0)
    oof = fitted_all.copy()
    resid_std = 0.0
    n_obs = int(obs_mask.sum())
    if n_obs < p + 3 or p == 0:
        return oof, fitted_all, resid_std
    Xo, yo = basis[obs_mask], target[obs_mask]
    mu = Xo.mean(0); Xoc = Xo - mu; yo_mean = yo.mean()
    reg = alpha * max((Xoc ** 2).sum(), 1.0) / max(Xoc.shape[0], 1)
    A = Xoc.T @ Xoc + reg * np.eye(p)
    try:
        Ainv = np.linalg.pinv(A)
    except np.linalg.LinAlgError:
        return oof, fitted_all, resid_std
    beta = Ainv @ (Xoc.T @ (yo - yo_mean))
    fitted_all = (basis - mu) @ beta + yo_mean
    H = Xoc @ Ainv @ Xoc.T
    hii = np.clip(np.diag(H), 0, 0.999)
    fitted_obs = fitted_all[obs_mask]
    resid = yo - fitted_obs
    loo_resid = resid / (1 - hii)
    loo_pred = yo - loo_resid
    resid_std = float(np.std(loo_resid))
    oof = fitted_all.copy()
    oof[obs_mask] = loo_pred
    return oof, fitted_all, resid_std


def _svd_basis(N, r):
    if N.shape[1] == 0:
        return np.empty((N.shape[0], 0))
    r = min(r, N.shape[1], N.shape[0])
    U, S, Vt = np.linalg.svd(N - N.mean(0), full_matrices=False)
    return U[:, :r] * S[:r]


def _t41(ctx):
    """Complétion façon Soft-Impute : base SVD tronquée (valeurs singulières
    seuillées) du pool donneurs, cible régressée dessus en OOF strict."""
    N, values, obs, n = ctx["N"], ctx["values"], ctx["observed"], ctx["n"]
    c, s = ctx["center"], ctx["scale"]
    U, S, Vt = (np.linalg.svd(N - N.mean(0), full_matrices=False) if N.shape[1] else (None, None, None))
    if U is None:
        return np.zeros((n, 2))
    lam = np.median(S) * 0.3
    S_soft = np.clip(S - lam, 0, None)
    r = min(8, len(S_soft))
    basis = U[:, :r] * S_soft[:r]
    oof, _, resid_std = _ridge_oof_basis(basis, values, obs)
    return np.column_stack([(oof - c) / s, np.full(n, resid_std / s)])


def _t42(ctx):
    """Robust PCA (proxy) : base SVD sur donneurs winsorisés (5e/95e pct par
    donneur) + magnitude de la composante "éparse" (résidu) à chaque ligne."""
    N, values, obs, n = ctx["N"], ctx["values"], ctx["observed"], ctx["n"]
    c, s = ctx["center"], ctx["scale"]
    if N.shape[1] == 0:
        return np.zeros((n, 2))
    lo_q, hi_q = np.percentile(N, [5, 95], axis=0)
    Nw = np.clip(N, lo_q, hi_q)
    basis = _svd_basis(Nw, 6)
    oof, fitted_all, _ = _ridge_oof_basis(basis, values, obs)
    sparse = np.median(np.abs(N - Nw), axis=1)
    return np.column_stack([(oof - c) / s, sparse / s])


def _t43(ctx):
    """Scores NMF (non négatifs) du pool donneurs, cible régressée dessus en OOF strict."""
    from sklearn.decomposition import NMF
    N, values, obs, n = ctx["N"], ctx["values"], ctx["observed"], ctx["n"]
    c, s = ctx["center"], ctx["scale"]
    if N.shape[1] == 0:
        return np.zeros((n, 2))
    try:
        Npos = N - min(N.min(), 0) + 1e-3
        r = min(6, N.shape[1])
        nmf = NMF(n_components=r, init="nndsvda", max_iter=200, random_state=0)
        basis = nmf.fit_transform(Npos)
        recon_err = np.mean(np.abs(Npos - nmf.inverse_transform(basis)), axis=1)
    except Exception:
        basis, recon_err = np.zeros((n, 1)), np.zeros(n)
    oof, _, _ = _ridge_oof_basis(basis, values, obs)
    return np.column_stack([(oof - c) / s, recon_err / s])


def _t44(ctx):
    """Composantes ICA communes du pool donneurs, cible régressée dessus en OOF strict."""
    from sklearn.decomposition import FastICA
    N, values, obs, n = ctx["N"], ctx["values"], ctx["observed"], ctx["n"]
    c, s = ctx["center"], ctx["scale"]
    if N.shape[1] < 2:
        return np.zeros((n, 2))
    try:
        r = min(6, N.shape[1])
        ica = FastICA(n_components=r, max_iter=300, random_state=0, whiten="unit-variance")
        basis = ica.fit_transform(N)
        recon = ica.inverse_transform(basis)
        recon_err = np.mean(np.abs(N - recon), axis=1)
    except Exception:
        basis, recon_err = np.zeros((n, 1)), np.zeros(n)
    oof, _, _ = _ridge_oof_basis(basis, values, obs)
    return np.column_stack([(oof - c) / s, recon_err / s])


def _cp_als(tensor, r, iters=15, seed=0):
    """ALS CP allégée pour un tenseur 3-voies (jour x créneau x donneur)."""
    rng = np.random.RandomState(seed)
    dims = tensor.shape
    factors = [rng.normal(size=(d, r)) * 0.1 + 1.0 for d in dims]
    T = np.nan_to_num(tensor)  # cellules jour/créneau non couvertes -> 0 (jour final tronqué typiquement)
    for _ in range(iters):
        for mode in range(3):
            others = [m for m in range(3) if m != mode]
            unfold = np.moveaxis(T, mode, 0).reshape(dims[mode], -1)
            kr = factors[others[1]]
            other0 = factors[others[0]]
            # produit de Khatri-Rao (others[0] (x) others[1])
            KR = np.einsum('ir,jr->ijr', other0, kr).reshape(-1, r)
            gram = (factors[others[0]].T @ factors[others[0]]) * (factors[others[1]].T @ factors[others[1]])
            try:
                factors[mode] = unfold @ KR @ np.linalg.pinv(gram + 1e-6 * np.eye(r))
            except np.linalg.LinAlgError:
                pass
    return factors


def _t45(ctx):
    """Complétion CP du tenseur jour x créneau x donneur : base = interaction
    (jour,créneau) du facteur CP, cible régressée dessus en OOF strict."""
    N, values, obs, n, phase = ctx["N"], ctx["values"], ctx["observed"], ctx["n"], ctx["phase"]
    c, s = ctx["center"], ctx["scale"]
    if N.shape[1] < 3:
        return np.zeros((n, 2))
    day_id = (ctx["pos"] // 48)
    n_days = int(day_id.max()) + 1
    m = N.shape[1]
    tensor = np.full((n_days, 48, m), np.nan)
    for i in range(n):
        tensor[day_id[i], phase[i], :] = N[i]
    r = 4
    try:
        day_f, slot_f, _donor_f = _cp_als(tensor, r, iters=8)
        basis = day_f[day_id] * slot_f[phase]  # (n, r) interaction jour x créneau
    except Exception:
        basis = np.zeros((n, r))
    oof, _, _ = _ridge_oof_basis(basis, values, obs)
    day_level = np.array([np.nanmedian(N[day_id == d]) if np.any(day_id == d) else c for d in day_id])
    return np.column_stack([(oof - c) / s, (day_level - c) / s])


def _t46(ctx):
    """Tucker (HOSVD) du même tenseur jour x créneau x donneur : cœur réduit,
    cible régressée sur la projection (jour,créneau) en OOF strict."""
    N, values, obs, n, phase = ctx["N"], ctx["values"], ctx["observed"], ctx["n"], ctx["phase"]
    c, s = ctx["center"], ctx["scale"]
    if N.shape[1] < 3:
        return np.zeros((n, 2))
    day_id = ctx["pos"] // 48
    n_days = int(day_id.max()) + 1
    m = N.shape[1]
    tensor = np.zeros((n_days, 48, m))
    cnt = np.zeros((n_days, 48))
    for i in range(n):
        tensor[day_id[i], phase[i], :] = N[i]; cnt[day_id[i], phase[i]] = 1
    try:
        unfold_day = tensor.reshape(n_days, -1)
        unfold_slot = np.moveaxis(tensor, 1, 0).reshape(48, -1)
        rd, rs = min(3, n_days), min(6, 48)
        Ud = np.linalg.svd(unfold_day, full_matrices=False)[0][:, :rd]
        Us = np.linalg.svd(unfold_slot, full_matrices=False)[0][:, :rs]
        basis = np.column_stack([np.outer(Ud[day_id[i]], Us[phase[i]]).ravel() for i in range(n)])
        basis = basis.T
    except Exception:
        basis = np.zeros((n, 1))
    oof, _, _ = _ridge_oof_basis(basis, values, obs)
    return np.column_stack([(oof - c) / s, np.full(n, float(basis.shape[1]))])


def _t47(ctx):
    """Proxy allégé d'autoencodeur masqué : rang optimal choisi par
    masquage artificiel de cellules OBSERVÉES du pool donneurs (jamais de la
    cible), puis cible régressée sur la base SVD de rang choisi en OOF strict."""
    N, values, obs, n = ctx["N"], ctx["values"], ctx["observed"], ctx["n"]
    c, s = ctx["center"], ctx["scale"]
    if N.shape[1] < 3:
        return np.zeros((n, 2))
    rng = np.random.RandomState(0)
    flat = N.copy()
    mask_frac = 0.1
    hidden = rng.rand(*flat.shape) < mask_frac
    train = np.where(hidden, np.nanmean(flat), flat)
    best_r, best_err = 1, np.inf
    for r in (2, 4, 6, 8):
        r = min(r, min(flat.shape) - 1) if min(flat.shape) > 1 else 1
        try:
            U, S, Vt = np.linalg.svd(train - train.mean(0), full_matrices=False)
            recon = (U[:, :r] * S[:r]) @ Vt[:r, :] + train.mean(0)
            err = float(np.mean(np.abs(recon[hidden] - flat[hidden]))) if hidden.any() else np.inf
        except Exception:
            err = np.inf
        if err < best_err:
            best_err, best_r = err, r
    basis = _svd_basis(N, best_r)
    oof, _, _ = _ridge_oof_basis(basis, values, obs)
    return np.column_stack([(oof - c) / s, np.full(n, best_err / (s + 1e-9))])


def _t48(ctx):
    """Proxy BRITS : prédictions forward/backward pondérées par decay
    temporel (poids exponentiel décroissant avec la distance à l'ancre
    observée la plus proche), régressées en OOF strict, + désaccord."""
    y, obs, missing, n = ctx["interp"], ctx["observed"], ctx["missing"], ctx["n"]
    c, s = ctx["center"], ctx["scale"]
    dl = ctx["pos"] - ctx["li"]
    dr = ctx["ri"] - ctx["pos"]
    decay_f = np.exp(-np.clip(dl, 0, 200) / 48.0)
    decay_b = np.exp(-np.clip(dr, 0, 200) / 48.0)
    order = 6
    lags_f = np.column_stack([np.r_[np.full(k, y[0]), y[:-k]] for k in range(1, order + 1)])
    lags_b = np.column_stack([np.r_[y[k:], np.full(k, y[-1])] for k in range(1, order + 1)])
    basis_f = lags_f * decay_f[:, None]
    basis_b = lags_b * decay_b[:, None]
    oof_f, _, _ = _ridge_oof_basis(basis_f, ctx["values"], obs)
    oof_b, _, _ = _ridge_oof_basis(basis_b, ctx["values"], obs)
    return np.column_stack([(oof_f - c) / s, (oof_b - c) / s, (oof_f - oof_b) / s,
                             (decay_f - decay_b)])


def _t49(ctx):
    """Proxy SAITS : attention temporelle (fenêtre locale) et inter-courbes
    (donneurs), combinées et régressées en OOF strict, + entropies d'attention."""
    y, N, obs, n = ctx["interp"], ctx["N"], ctx["observed"], ctx["n"]
    c, s = ctx["center"], ctx["scale"]
    w = 6
    temporal = np.zeros(n); ent_t = np.zeros(n)
    for i in range(n):
        lo, hi = max(0, i - w), min(n, i + w + 1)
        idx = np.r_[np.arange(lo, i), np.arange(i + 1, hi)]
        if len(idx) == 0:
            continue
        d = np.abs(idx - i).astype(float)
        wgt = np.exp(-d / 3.0); wgt /= wgt.sum()
        temporal[i] = float(np.dot(wgt, y[idx]))
        ent_t[i] = float(-np.sum(wgt * np.log(wgt + 1e-12)) / np.log(len(wgt) + 1e-9))
    if N.shape[1] > 0:
        corr = np.array([np.corrcoef(y, N[:, j])[0, 1] if np.std(N[:, j]) > 1e-9 else 0.0 for j in range(N.shape[1])])
        wgt_c = np.abs(np.nan_to_num(corr)); wgt_c = wgt_c / (wgt_c.sum() + 1e-9)
        cross = N @ wgt_c
        ent_c = float(-np.sum(wgt_c * np.log(wgt_c + 1e-12)) / np.log(len(wgt_c) + 1e-9))
    else:
        cross, ent_c = np.full(n, c), 0.0
    basis = np.column_stack([temporal, cross])
    oof, _, _ = _ridge_oof_basis(basis, ctx["values"], obs)
    return np.column_stack([(oof - c) / s, ent_t, np.full(n, ent_c), (temporal - cross) / s])


def _t50(ctx):
    """Ensemble probabiliste : moyenne/médiane/écart-type/IQR/asymétrie des
    méta-prédictions OOF T41/T43/T44/T47 (déjà sans fuite individuellement)."""
    members = []
    for fn in (_t41, _t42, _t43, _t44, _t47):
        try:
            out = fn(ctx)
            members.append(np.asarray(out)[:, 0])  # 1re colonne = prédiction OOF
        except Exception:
            continue
    n = ctx["n"]
    if not members:
        return np.zeros((n, 5))
    M = np.column_stack(members)
    mean = M.mean(1); med = np.median(M, axis=1); std = M.std(1)
    q25, q75 = np.percentile(M, [25, 75], axis=1)
    iqr = q75 - q25
    m3 = np.mean((M - mean[:, None]) ** 3, axis=1)
    skew = m3 / (std ** 3 + 1e-9)
    return np.column_stack([mean, med, std, iqr, skew])


_T_BUILDERS = {f"T{i:02d}": globals()[f"_t{i:02d}"] for i in range(1, 51)}


def build_t_features(values, index, scaled_neighbors, missing, groups):
    """Point d'entrée de la campagne T01--T50 : construit le contexte
    partagé une seule fois puis dispatche vers chaque groupe demandé. Chaque
    groupe est enveloppé individuellement (voir note en tête de section) :
    une exception isolée retombe sur un bloc de zéros de la largeur attendue
    plutôt que de faire échouer toute la colonne."""
    ctx = _t_context(values, index, scaled_neighbors, missing)
    n = ctx["n"]
    cols = []
    for g in groups:
        builder = _T_BUILDERS.get(g)
        width = T_FEATURE_WIDTHS.get(g, 1)
        if builder is None:
            continue
        try:
            out = np.asarray(builder(ctx), float)
            if out.ndim == 1:
                out = out.reshape(-1, 1)
            if out.shape != (n, width):
                out = np.zeros((n, width))
        except Exception:
            out = np.zeros((n, width))
        cols.append(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0))
    return np.hstack(cols) if cols else np.empty((n, 0))
