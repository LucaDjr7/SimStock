"""Similarité DTW sur trajectoires d'embeddings."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

log = logging.getLogger(__name__)


def _embedding_columns(emb_daily: pd.DataFrame) -> list[str]:
    e_cols = [c for c in emb_daily.columns if c.startswith("e_")]
    if not e_cols:
        raise ValueError("Aucune colonne d'embedding e_* trouvée.")
    return e_cols


def _embedding_to_scalar_series(emb_daily: pd.DataFrame) -> pd.DataFrame:
    e_cols = _embedding_columns(emb_daily)
    df = emb_daily.copy()
    df["embedding_scalar"] = np.linalg.norm(df[e_cols].to_numpy(dtype=np.float32), axis=1)
    return df.pivot_table(index="date", columns="ticker", values="embedding_scalar", aggfunc="mean")


def _embedding_to_mean_series(emb_daily: pd.DataFrame) -> pd.DataFrame:
    e_cols = _embedding_columns(emb_daily)
    df = emb_daily.copy()
    df["embedding_scalar"] = df[e_cols].to_numpy(dtype=np.float32).mean(axis=1)
    return df.pivot_table(index="date", columns="ticker", values="embedding_scalar", aggfunc="mean")


def _embedding_to_pca_series(emb_daily: pd.DataFrame, n_components: int = 1) -> pd.DataFrame:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    e_cols = _embedding_columns(emb_daily)
    X = emb_daily[e_cols].to_numpy(dtype=np.float32)
    X = StandardScaler().fit_transform(X)
    reduced = PCA(n_components=n_components, random_state=42).fit_transform(X)
    df = emb_daily[["date", "ticker"]].copy()
    df["embedding_scalar"] = reduced[:, 0]
    return df.pivot_table(index="date", columns="ticker", values="embedding_scalar", aggfunc="mean")


def _fastdtw_distance(i: int, j: int, data_t: np.ndarray) -> float:
    from fastdtw import fastdtw

    if i == j:
        return 0.0
    return float(fastdtw(data_t[i].copy(), data_t[j].copy())[0])


def compute_dtw_matrix(series_wide: pd.DataFrame, scale: bool = True, n_jobs: int = -1) -> pd.DataFrame:
    from sklearn.preprocessing import MinMaxScaler

    df = series_wide.sort_index().ffill().bfill().dropna(axis=1, how="any")
    tickers = list(df.columns)
    n = len(tickers)
    if n < 2:
        raise ValueError("Il faut au moins 2 tickers pour calculer une matrice DTW.")

    log.info("Calcul matrice DTW : %d tickers, %d dates", n, len(df))
    if scale:
        data = MinMaxScaler().fit_transform(df)
    else:
        data = df.to_numpy(dtype=np.float64)
    data_t = data.T

    indices = [(i, j) for i in range(n) for j in range(i, n)]
    distances = Parallel(n_jobs=n_jobs)(
        delayed(_fastdtw_distance)(i, j, data_t) for i, j in tqdm(indices, desc="fastDTW")
    )

    M = np.zeros((n, n), dtype=np.float64)
    for (i, j), dist in zip(indices, distances):
        M[i, j] = dist
        M[j, i] = dist
    return pd.DataFrame(M, index=tickers, columns=tickers)


def dtw_to_similarity(
    dtw_matrix: pd.DataFrame,
    method: str = "rank",
    robust_low_q: float = 0.01,
    robust_high_q: float = 0.99,
) -> pd.DataFrame:
    """
    Convertit des distances DTW en similarités [0, 1].

    rank : distribution globale calibrée par rang, pratique pour un seuil par quantile.
    exp : exp(-d/tau), plus strict autour de 1.
    robust_minmax : min-max robuste par quantiles.
    minmax : compatibilité anciens runs, souvent trop permissif.
    """
    M = dtw_matrix.to_numpy(dtype=np.float64).copy()
    n = M.shape[0]
    idx = np.triu_indices(n, k=1)
    vals = M[idx]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("Matrice DTW invalide : aucune distance hors diagonale finie.")

    method = method.lower()
    if method == "rank":
        pair_vals = M[idx]
        order = np.argsort(pair_vals, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(order), dtype=np.float64)
        sim_vals = np.ones_like(ranks) if len(order) == 1 else 1.0 - ranks / (len(order) - 1)
        S = np.zeros_like(M, dtype=np.float64)
        S[idx] = sim_vals
        S[(idx[1], idx[0])] = sim_vals
    elif method == "exp":
        tau = float(np.median(vals[vals > 0])) if np.any(vals > 0) else 1.0
        S = np.exp(-M / max(tau, 1e-12))
    elif method == "robust_minmax":
        lo = float(np.quantile(vals, robust_low_q))
        hi = float(np.quantile(vals, robust_high_q))
        if hi <= lo:
            hi, lo = float(vals.max()), float(vals.min())
        S = np.ones_like(M) if hi <= lo else 1.0 - np.clip((M - lo) / (hi - lo), 0.0, 1.0)
    elif method == "minmax":
        mx = float(vals.max())
        S = np.ones_like(M) if mx <= 0 else 1.0 - (M / mx)
    else:
        raise ValueError("method inconnu. Choisir rank, exp, robust_minmax ou minmax.")

    np.fill_diagonal(S, 1.0)
    S = np.clip(S, 0.0, 1.0)
    return pd.DataFrame(S, index=dtw_matrix.index, columns=dtw_matrix.columns)


def build_dtw_similarity(
    emb_daily: pd.DataFrame,
    reduction: str = "pca",
    scale: bool = True,
    sim_method: str = "rank",
    n_jobs: int = -1,
) -> pd.DataFrame:
    reduction = reduction.lower()
    if reduction == "norm":
        wide = _embedding_to_scalar_series(emb_daily)
    elif reduction == "pca":
        wide = _embedding_to_pca_series(emb_daily)
    elif reduction == "mean":
        wide = _embedding_to_mean_series(emb_daily)
    else:
        raise ValueError("reduction inconnue. Choisir pca, norm ou mean.")

    dtw_mat = compute_dtw_matrix(wide, scale=scale, n_jobs=n_jobs)
    sim_mat = dtw_to_similarity(dtw_mat, method=sim_method)
    vals = sim_mat.to_numpy()[np.triu_indices_from(sim_mat.to_numpy(), k=1)]
    log.info(
        "DTW similarity prête : %d tickers | method=%s | mean=%.4f median=%.4f q99=%.4f",
        len(sim_mat),
        sim_method,
        float(vals.mean()),
        float(np.median(vals)),
        float(np.quantile(vals, 0.99)),
    )
    return sim_mat
