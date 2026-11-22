"""Features expérimentales V13 pour l'imputation Enedis.

Le module conserve strictement le coeur V12 (40 voisins, 15 clusters,
TabPFN médian, blend interpolation/confiance) et permet d'ajouter des blocs
de features indépendants pour des ablations appariées.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from evaluate import fill_nan_with_interpolation
import feature_engineering_bench as feb


ALL_AXES = (
    "lag14", "lag456", "dow_profile", "daytype_profile", "profile_stats",
    "donor_weighted", "donor_robust", "donor_quantiles", "donor_uncertainty",
    "rolling_3", "rolling_7", "rolling_13", "neighbor_slopes",
    "calendar_harmonics", "fourier_daily",
)


def _group_profile(values, keys, stats=("median",)):
    values = np.asarray(values, dtype=float)
    observed = np.isfinite(values)
    frame = pd.DataFrame({"key": list(keys), "value": values})
    grouped = frame.loc[observed].groupby("key")["value"]
    fallback = float(np.nanmedian(values)) if observed.any() else 0.0
    out = []
    for stat in stats:
        if stat == "mad":
            mapping = grouped.apply(lambda x: np.median(np.abs(x - np.median(x))))
            default = 0.0
        else:
            mapping = getattr(grouped, stat)()
            default = fallback
        out.append(pd.Series(list(keys)).map(mapping).fillna(default).to_numpy(float))
    return np.column_stack(out)


def target_profile_features(values, index, axes):
    dt = pd.DatetimeIndex(pd.to_datetime(index))
    slot = dt.hour.to_numpy() * 2 + dt.minute.to_numpy() // 30
    cols = []
    if "dow_profile" in axes:
        cols.append(_group_profile(values, list(zip(dt.dayofweek, slot))))
    if "daytype_profile" in axes:
        daytype = (dt.dayofweek >= 5).astype(int)
        cols.append(_group_profile(values, list(zip(daytype, slot))))
    if "profile_stats" in axes:
        cols.append(_group_profile(values, slot, stats=("mean", "mad")))
    return np.hstack(cols) if cols else np.empty((len(values), 0))


def extra_periodic_features(values, axes):
    lags = []
    if "lag14" in axes:
        lags.append(48 * 14)
    if "lag456" in axes:
        lags.extend((48 * 4, 48 * 5, 48 * 6))
    if not lags:
        return np.empty((len(values), 0))
    return feb.periodic_copy_features(
        values, lags=tuple(lags), include_summary=True,
        normalize=True, include_availability=True,
    )


def donor_consensus_features(scaled, corr, axes):
    cols = []
    weights = np.clip(np.asarray(corr, float), 1e-3, None)
    weights /= weights.sum()
    weighted = scaled @ weights
    median = np.median(scaled, axis=1)
    if "donor_weighted" in axes:
        cols.append(weighted[:, None])
    if "donor_robust" in axes:
        mad = np.median(np.abs(scaled - median[:, None]), axis=1)
        cols.append(np.column_stack([median, mad]))
    if "donor_quantiles" in axes:
        q10, q25, q75, q90 = np.quantile(scaled, (.10, .25, .75, .90), axis=1)
        cols.append(np.column_stack([q10, q25, q75, q90]))
    if "donor_uncertainty" in axes:
        wstd = np.sqrt(((scaled - weighted[:, None]) ** 2) @ weights)
        agreement = 1.0 / (1.0 + wstd / (np.abs(weighted) + 1.0))
        cols.append(np.column_stack([wstd, agreement]))
    consensus = pd.Series(weighted)
    for axis, window in (("rolling_3", 3), ("rolling_7", 7), ("rolling_13", 13)):
        if axis in axes:
            mean = consensus.rolling(window, center=True, min_periods=1).mean().to_numpy()
            std = consensus.rolling(window, center=True, min_periods=1).std().fillna(0).to_numpy()
            cols.append(np.column_stack([mean, std]))
    if "neighbor_slopes" in axes:
        d1 = np.gradient(weighted)
        d2 = np.gradient(d1)
        cols.append(np.column_stack([d1, d2]))
    return np.hstack(cols) if cols else np.empty((scaled.shape[0], 0))


def calendar_harmonics(index):
    dt = pd.DatetimeIndex(pd.to_datetime(index))
    minute = dt.hour.to_numpy() * 60.0 + dt.minute.to_numpy()
    dowpos = dt.dayofweek.to_numpy() + minute / 1440.0
    cols = []
    for harmonic in (2, 3, 4):
        angle = 2 * np.pi * harmonic * minute / 1440.0
        cols.extend((np.sin(angle), np.cos(angle)))
    for harmonic in (2, 3):
        angle = 2 * np.pi * harmonic * dowpos / 7.0
        cols.extend((np.sin(angle), np.cos(angle)))
    return np.column_stack(cols)


def fourier_daily_feature(values, index, harmonics=6):
    dt = pd.DatetimeIndex(pd.to_datetime(index))
    phase = (dt.hour.to_numpy() * 60.0 + dt.minute.to_numpy()) / 1440.0
    design = [np.ones(len(values))]
    for h in range(1, harmonics + 1):
        design.extend((np.sin(2*np.pi*h*phase), np.cos(2*np.pi*h*phase)))
    X = np.column_stack(design)
    observed = np.isfinite(values)
    if observed.sum() < X.shape[1] + 2:
        return np.full((len(values), 1), np.nanmedian(values))
    coef = np.linalg.lstsq(X[observed], np.asarray(values)[observed], rcond=None)[0]
    return (X @ coef).reshape(-1, 1)


def build_extra(values, index, scaled, corr, base_time, axes):
    axes = frozenset(axes)
    cols = [base_time,
            feb.same_curve_seasonality_feature(values, index),
            feb.periodic_copy_features(values, include_summary=True,
                                       normalize=True, include_availability=True)]
    cols.extend((target_profile_features(values, index, axes),
                 extra_periodic_features(values, axes),
                 donor_consensus_features(scaled, corr, axes)))
    if "calendar_harmonics" in axes:
        cols.append(calendar_harmonics(index))
    if "fourier_daily" in axes:
        cols.append(fourier_daily_feature(values, index))
    return np.hstack([c for c in cols if c.shape[1]])


def make_v13_model(make_regressor, axes=(), k=40, n_clusters=15,
                   random_state=42, clip_nonnegative=True):
    """Pipeline V12, avec uniquement les blocs ``axes`` ajoutés."""
    def model(donor_df, dev_holed_df, dev_true_df, val_holed_df):
        pool = pd.concat([donor_df, dev_true_df], axis=1)
        pool_vals = pool.ffill().bfill().to_numpy(dtype=np.float64)
        km, donor_labels = feb.fit_shape_clusters(donor_df, n_clusters, random_state)
        if dev_true_df.shape[1]:
            dev_filled = dev_true_df.ffill().bfill()
            zdev = (dev_filled - dev_filled.mean()) / (dev_filled.std() + 1e-9)
            dev_labels = pd.Series(km.predict(zdev.T.to_numpy()), index=dev_true_df.columns)
            labels = pd.concat([donor_labels, dev_labels])
        else:
            labels = donor_labels
        positions = pd.Series(np.arange(len(labels)), index=labels.index)
        cluster_idx = {cl: positions[idx].to_numpy()
                       for cl, idx in labels.groupby(labels).groups.items()}
        tfeat = feb.time_features_full_seasonality(
            val_holed_df.index, groups=("hour", "dow", "dom", "year"))
        preds = {}
        for col in val_holed_df:
            s = val_holed_df[col]
            observed = s.notna().to_numpy(); missing = ~observed
            if not missing.any():
                preds[col] = s; continue
            obs_idx = np.flatnonzero(observed); miss_idx = np.flatnonzero(missing)
            target_obs = s.to_numpy()[obs_idx]
            interp = fill_nan_with_interpolation(s)
            cluster = feb.cluster_of(interp, km)
            same = cluster_idx.get(cluster)
            sub = pool_vals[:, same] if same is not None and len(same) >= max(2*k, 30) else pool_vals
            knn, _, scaled = feb.weighted_knn_pred(
                sub, obs_idx, miss_idx, target_obs, k, robust=True,
                adaptive_k=True, use_euclid=True)
            # Recalcule les corrélations des voisins effectivement sélectionnés
            # dans leur ordre final afin de construire un consensus pondéré.
            selected, corr = feb.adaptive_top_k_corr_euclid(
                target_obs, sub[obs_idx], k_max=k, corr_threshold=.15)
            if scaled.shape[1] != len(corr):
                corr = np.ones(scaled.shape[1])
            extra = build_extra(s.to_numpy(), val_holed_df.index, scaled, corr,
                                tfeat, axes)
            if len(obs_idx) >= 15:
                pred_model = feb.local_regression_pred(
                    make_regressor, scaled, obs_idx, miss_idx, target_obs, extra)
            else:
                pred_model = knn
            runs = feb.run_lengths_for(missing); edges = feb.edge_distance_for(missing)
            alpha = np.array([feb.continuous_alpha(runs[i], edges[i], True)
                              for i in miss_idx])
            alpha *= 1.0 - .5 * feb.periodic_agreement(s.to_numpy())[miss_idx]
            pred = alpha * interp.to_numpy()[miss_idx] + (1-alpha) * pred_model
            if clip_nonnegative:
                pred = np.maximum(pred, 0.0)
            filled = s.copy(); filled.iloc[miss_idx] = pred; preds[col] = filled
        return pd.DataFrame(preds, index=val_holed_df.index)
    return model
