"""API métier de substituabilité des stocks."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from euronext_simstock.config import SubstitutionConfig

log = logging.getLogger(__name__)


@dataclass
class SimilarityResult:
    ticker_a: str
    ticker_b: str
    similarity: float
    are_substitutes: bool
    same_sector: Optional[bool] = None


class SubstitutionEngine:
    def __init__(
        self,
        similarity_matrix: pd.DataFrame,
        sector_df: Optional[pd.DataFrame] = None,
        cfg: Optional[SubstitutionConfig] = None,
    ):
        self.cfg = cfg or SubstitutionConfig()
        if not similarity_matrix.index.equals(similarity_matrix.columns):
            raise ValueError("similarity_matrix doit être carrée avec index == columns.")
        self.tickers = [str(t) for t in similarity_matrix.index]
        self._ticker_to_idx = {ticker: idx for idx, ticker in enumerate(self.tickers)}
        self._sim_matrix = similarity_matrix.to_numpy(dtype=np.float32).copy()
        np.fill_diagonal(self._sim_matrix, 1.0)

        self.sector_lookup: Dict[str, str] = {}
        if sector_df is not None and not sector_df.empty:
            self.sector_lookup = dict(zip(sector_df["ticker"].astype(str), sector_df["sector"].astype(str)))

        off = self._sim_matrix[np.triu_indices_from(self._sim_matrix, k=1)] if len(self.tickers) > 1 else np.array([1.0])
        log.info(
            "SubstitutionEngine : %d tickers | sim mean=%.3f median=%.3f",
            len(self.tickers),
            float(np.mean(off)),
            float(np.median(off)),
        )

    @classmethod
    def from_similarity_matrix(
        cls,
        sim_matrix: pd.DataFrame,
        sector_df: Optional[pd.DataFrame] = None,
        cfg: Optional[SubstitutionConfig] = None,
    ) -> "SubstitutionEngine":
        return cls(sim_matrix, sector_df=sector_df, cfg=cfg)

    @classmethod
    def from_embeddings(
        cls,
        embeddings_df: pd.DataFrame,
        sector_df: Optional[pd.DataFrame] = None,
        cfg: Optional[SubstitutionConfig] = None,
        similarity_method: str = "cosine",
    ) -> "SubstitutionEngine":
        """
        Construit un engine depuis des embeddings stock-level agrégés.

        similarity_method:
            - "cosine" : cosine rescalé dans [0, 1], conservé comme baseline rapide.
            - "l2rank" : distance L2 entre embeddings, transformée en similarité par rang global.
                         Recommandé quand le cosine brut collapse vers 1.
        """
        e_cols = [c for c in embeddings_df.columns if c.startswith("e_")]
        if not e_cols:
            raise ValueError("Aucune colonne e_* dans embeddings_df.")

        tickers = embeddings_df["ticker"].astype(str).tolist()
        emb = embeddings_df[e_cols].to_numpy(dtype=np.float32)

        method = similarity_method.lower()
        if method == "cosine":
            sim_df = cls._cosine_similarity_df(tickers, emb)
        elif method in {"l2rank", "l2_rank", "rank_l2"}:
            sim_df = cls._l2_rank_similarity_df(tickers, emb)
        else:
            raise ValueError("similarity_method doit valoir 'cosine' ou 'l2rank'.")

        return cls(sim_df, sector_df=sector_df, cfg=cfg)

    @classmethod
    def from_daily_embeddings_snapshot(
        cls,
        emb_daily: pd.DataFrame,
        sector_df: Optional[pd.DataFrame] = None,
        cfg: Optional[SubstitutionConfig] = None,
        as_of_date: Optional[str] = None,
        lookback_days: Optional[int] = None,
        aggregation: str = "last",
        similarity_method: str = "cosine",
    ) -> "SubstitutionEngine":
        """
        Construit une similarité sans fuite temporelle pour une date t :
        - garde uniquement les embeddings <= as_of_date ;
        - optionnellement limite à une fenêtre lookback_days ;
        - agrège par ticker avec 'last' ou 'mean'.
        """
        df = emb_daily.copy()
        df["date"] = pd.to_datetime(df["date"])
        if as_of_date is not None:
            as_of = pd.to_datetime(as_of_date)
            df = df[df["date"] <= as_of]
            if lookback_days is not None:
                df = df[df["date"] >= as_of - pd.Timedelta(days=int(lookback_days))]
        if df.empty:
            raise ValueError("Aucun embedding disponible pour la date/fenêtre demandée.")

        e_cols = [c for c in df.columns if c.startswith("e_")]
        aggregation = aggregation.lower()
        if aggregation == "last":
            stock_df = df.sort_values("date").groupby("ticker", sort=False)[e_cols].last().reset_index()
        elif aggregation == "mean":
            stock_df = df.groupby("ticker", sort=False)[e_cols].mean().reset_index()
        else:
            raise ValueError("aggregation doit valoir 'last' ou 'mean'.")
        return cls.from_embeddings(
            stock_df,
            sector_df=sector_df,
            cfg=cfg,
            similarity_method=similarity_method,
        )

    @staticmethod
    def _cosine_similarity_df(tickers: List[str], emb: np.ndarray) -> pd.DataFrame:
        """Cosine similarity rescalée dans [0, 1]. Baseline rapide conservée."""
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normed = emb / norms
        cos = normed @ normed.T
        sim = (cos + 1.0) / 2.0
        sim = np.clip(sim, 0.0, 1.0).astype(np.float32)
        np.fill_diagonal(sim, 1.0)
        return pd.DataFrame(sim, index=tickers, columns=tickers)

    @staticmethod
    def _l2_rank_similarity_df(tickers: List[str], emb: np.ndarray) -> pd.DataFrame:
        """
        Similarité L2-rank depuis embeddings stock-level.

        Distance L2 faible  -> similarité proche de 1.
        Distance L2 élevée -> similarité proche de 0.

        Le scoring par rang global évite le problème du cosine brut quand tous les
        embeddings partagent une forte composante commune et ont donc une cosine
        similarity presque égale à 1.
        """
        emb = np.asarray(emb, dtype=np.float32)
        n = emb.shape[0]
        if n < 2:
            sim = np.ones((n, n), dtype=np.float32)
            return pd.DataFrame(sim, index=tickers, columns=tickers)

        # Calcul vectorisé des distances L2 pairwise.
        diff = emb[:, None, :] - emb[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1)).astype(np.float64)

        iu = np.triu_indices(n, k=1)
        vals = dist[iu]

        order = np.argsort(vals, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(order), dtype=np.float64)

        if len(order) == 1:
            sim_vals = np.ones_like(ranks, dtype=np.float64)
        else:
            sim_vals = 1.0 - ranks / (len(order) - 1)

        sim = np.zeros((n, n), dtype=np.float32)
        sim[iu] = sim_vals.astype(np.float32)
        sim[(iu[1], iu[0])] = sim_vals.astype(np.float32)
        np.fill_diagonal(sim, 1.0)
        return pd.DataFrame(sim, index=tickers, columns=tickers)

    @property
    def similarity_matrix(self) -> np.ndarray:
        return self._sim_matrix

    def similarity_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._sim_matrix, index=self.tickers, columns=self.tickers)

    def similarity(self, ticker_a: str, ticker_b: str) -> float:
        ia, ib = self._idx(ticker_a), self._idx(ticker_b)
        return float(self._sim_matrix[ia, ib])

    def are_substitutes(self, ticker_a: str, ticker_b: str, threshold: Optional[float] = None) -> SimilarityResult:
        threshold = self.cfg.similarity_threshold if threshold is None else float(threshold)
        sim = self.similarity(ticker_a, ticker_b)
        same_sector = None
        if self.sector_lookup:
            sec_a = self.sector_lookup.get(ticker_a, "Unknown")
            sec_b = self.sector_lookup.get(ticker_b, "Unknown")
            same_sector = (sec_a == sec_b) and (sec_a != "Unknown")
        is_sub = sim >= threshold
        if self.cfg.same_sector_only and same_sector is False:
            is_sub = False
        return SimilarityResult(ticker_a, ticker_b, sim, is_sub, same_sector)

    def all_substitutes(
        self,
        ticker: str,
        threshold: Optional[float] = None,
        same_sector_only: Optional[bool] = None,
    ) -> List[Tuple[str, float]]:
        threshold = self.cfg.similarity_threshold if threshold is None else float(threshold)
        same_sector_only = self.cfg.same_sector_only if same_sector_only is None else bool(same_sector_only)
        idx = self._idx(ticker)
        sims = self._sim_matrix[idx]
        mask = np.ones(len(self.tickers), dtype=bool)
        mask[idx] = False
        mask &= sims >= threshold
        if same_sector_only and self.sector_lookup:
            sector = self.sector_lookup.get(ticker, "Unknown")
            sectors = np.array([self.sector_lookup.get(t, "Unknown") for t in self.tickers])
            mask &= sectors == sector
            mask &= sectors != "Unknown"
        candidate_idx = np.where(mask)[0]
        candidate_idx = candidate_idx[np.argsort(-sims[candidate_idx])]
        return [(self.tickers[i], float(sims[i])) for i in candidate_idx]

    def top_k_substitutes(
        self,
        ticker: str,
        k: Optional[int] = None,
        min_similarity: Optional[float] = None,
        same_sector_only: Optional[bool] = None,
    ) -> List[Tuple[str, float]]:
        k = self.cfg.default_top_k if k is None else int(k)
        threshold = -np.inf if min_similarity is None else float(min_similarity)
        return self.all_substitutes(ticker, threshold=threshold, same_sector_only=same_sector_only)[:k]

    def substitution_table(
        self,
        threshold: Optional[float] = None,
        same_sector_only: Optional[bool] = None,
        max_per_ticker: Optional[int] = None,
    ) -> pd.DataFrame:
        threshold = self.cfg.similarity_threshold if threshold is None else float(threshold)
        rows = []
        for ticker in self.tickers:
            subs = self.all_substitutes(ticker, threshold=threshold, same_sector_only=same_sector_only)
            if max_per_ticker is not None:
                subs = subs[: int(max_per_ticker)]
            rows.append(
                {
                    "ticker": ticker,
                    "sector": self.sector_lookup.get(ticker, "Unknown"),
                    "n_substitutes": len(subs),
                    "substitutes": [s for s, _ in subs],
                    "similarities": [round(sim, 4) for _, sim in subs],
                }
            )
        df = pd.DataFrame(rows)
        log.info(
            "Table substitution : %d tickers | mean #subs=%.2f | sans substitut=%d",
            len(df),
            float(df["n_substitutes"].mean()) if len(df) else 0.0,
            int((df["n_substitutes"] == 0).sum()) if len(df) else 0,
        )
        return df

    def substitution_table_long(
        self,
        threshold: Optional[float] = None,
        same_sector_only: Optional[bool] = None,
    ) -> pd.DataFrame:
        threshold = self.cfg.similarity_threshold if threshold is None else float(threshold)
        rows = []
        for ticker in self.tickers:
            for sub, sim in self.all_substitutes(ticker, threshold=threshold, same_sector_only=same_sector_only):
                rows.append(
                    {
                        "ticker": ticker,
                        "substitute": sub,
                        "similarity": sim,
                        "ticker_sector": self.sector_lookup.get(ticker, "Unknown"),
                        "sub_sector": self.sector_lookup.get(sub, "Unknown"),
                    }
                )
        return pd.DataFrame(rows)

    def _pair_values(self, same_sector_only: Optional[bool] = None) -> np.ndarray:
        n = len(self.tickers)
        iu = np.triu_indices(n, k=1)
        vals = self._sim_matrix[iu]
        if same_sector_only and self.sector_lookup:
            sectors = np.array([self.sector_lookup.get(t, "Unknown") for t in self.tickers])
            mask = (sectors[iu[0]] == sectors[iu[1]]) & (sectors[iu[0]] != "Unknown")
            vals = vals[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            raise ValueError("Aucune paire disponible pour calibrer le seuil.")
        return vals

    def threshold_for_avg_substitutes(self, target_avg_substitutes: float = 10, same_sector_only: Optional[bool] = None) -> float:
        n = len(self.tickers)
        if n < 2:
            raise ValueError("Il faut au moins 2 tickers.")
        target = max(0.0, min(float(target_avg_substitutes), float(n - 1)))
        keep_fraction = target / float(n - 1)
        keep_fraction = min(max(keep_fraction, 1.0 / max(1.0, n * (n - 1) / 2)), 1.0)
        vals = self._pair_values(same_sector_only=same_sector_only)
        return float(np.quantile(vals, 1.0 - keep_fraction))

    def similarity_diagnostics(self, thresholds: Optional[List[float]] = None, same_sector_only: Optional[bool] = None) -> pd.DataFrame:
        vals = self._pair_values(same_sector_only=same_sector_only)
        n = len(self.tickers)
        if thresholds is None:
            thresholds = [float(np.quantile(vals, q)) for q in [0.50, 0.75, 0.90, 0.95, 0.975, 0.99, 0.995]]
        rows = []
        for threshold in thresholds:
            pair_fraction = float((vals >= threshold).mean())
            rows.append(
                {
                    "threshold": float(threshold),
                    "pair_fraction_pct": round(pair_fraction * 100, 4),
                    "approx_avg_substitutes": round(pair_fraction * (n - 1), 2),
                    "n_pairs_above": int((vals >= threshold).sum()),
                }
            )
        return pd.DataFrame(rows)

    def substitution_groups(self, threshold: Optional[float] = None) -> List[Set[str]]:
        threshold = self.cfg.similarity_threshold if threshold is None else float(threshold)
        n = len(self.tickers)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        for i, j in np.argwhere(np.triu(self._sim_matrix >= threshold, k=1)):
            union(int(i), int(j))
        groups: Dict[int, Set[str]] = {}
        for i in range(n):
            groups.setdefault(find(i), set()).add(self.tickers[i])
        return [g for g in groups.values() if len(g) > 1]

    def mutual_substitution_cliques(self, threshold: Optional[float] = None) -> List[Set[str]]:
        import networkx as nx

        threshold = self.cfg.similarity_threshold if threshold is None else float(threshold)
        graph = nx.Graph()
        graph.add_nodes_from(self.tickers)
        for i, j in np.argwhere(np.triu(self._sim_matrix >= threshold, k=1)):
            graph.add_edge(self.tickers[int(i)], self.tickers[int(j)])
        return [set(c) for c in nx.find_cliques(graph) if len(c) >= 2]

    def compare_trades(self, trader_x: Dict[str, float], trader_y: Dict[str, float], threshold: Optional[float] = None) -> Dict:
        """Compare deux portefeuilles/ordres par matching greedy des substituts."""
        threshold = self.cfg.similarity_threshold if threshold is None else float(threshold)
        tx, ty = set(trader_x.keys()), set(trader_y.keys())
        strict = tx & ty
        x_left, y_left = tx - strict, ty - strict

        candidates = []
        for a in x_left:
            if a not in self._ticker_to_idx:
                continue
            ia = self._idx(a)
            for b in y_left:
                if b not in self._ticker_to_idx:
                    continue
                sim = float(self._sim_matrix[ia, self._idx(b)])
                if sim >= threshold:
                    candidates.append((sim, a, b))
        candidates.sort(reverse=True)

        sub_pairs = []
        x_matched, y_matched = set(), set()
        for sim, a, b in candidates:
            if a in x_matched or b in y_matched:
                continue
            sub_pairs.append((a, b, sim))
            x_matched.add(a)
            y_matched.add(b)

        total_x = sum(abs(v) for v in trader_x.values())
        explained_x = sum(abs(trader_x[t]) for t in strict) + sum(abs(trader_x[a]) for a, _, _ in sub_pairs)
        score = explained_x / total_x if total_x else 0.0
        return {
            "overlap_strict": sorted(strict),
            "overlap_substitutable": sorted(sub_pairs, key=lambda r: -r[2]),
            "unique_to_x": sorted(x_left - x_matched),
            "unique_to_y": sorted(y_left - y_matched),
            "substitutable_score": round(score, 4),
            "n_strict": len(strict),
            "n_substitutable": len(sub_pairs),
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            sim_matrix=self._sim_matrix,
            tickers=np.array(self.tickers, dtype=object),
            sectors=np.array([self.sector_lookup.get(t, "Unknown") for t in self.tickers], dtype=object),
        )
        log.info("Engine sauvegardé : %s", path)

    @classmethod
    def load(cls, path: Path, cfg: Optional[SubstitutionConfig] = None) -> "SubstitutionEngine":
        data = np.load(path, allow_pickle=True)
        tickers = [str(t) for t in data["tickers"]]
        sim_df = pd.DataFrame(data["sim_matrix"], index=tickers, columns=tickers)
        sec_df = pd.DataFrame({"ticker": tickers, "sector": [str(s) for s in data["sectors"]]})
        return cls(sim_df, sector_df=sec_df, cfg=cfg)

    def _idx(self, ticker: str) -> int:
        ticker = str(ticker)
        if ticker not in self._ticker_to_idx:
            raise KeyError(f"Ticker inconnu : {ticker}")
        return self._ticker_to_idx[ticker]
