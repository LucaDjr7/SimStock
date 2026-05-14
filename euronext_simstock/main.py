"""Pipeline de bout en bout pour Euronext-SimStock."""
from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pandas as pd
import torch

from euronext_simstock.config import (
    EMBEDDINGS_DIR,
    MODELS_DIR,
    FeatureConfig,
    ModelConfig,
    SubstitutionConfig,
    TimeWindow,
    TrainingConfig,
)
from euronext_simstock.data import (
    SectorEncoder,
    build_panel,
    download_ohlcv,
    fetch_sector_metadata,
    get_universe,
    make_dataloader,
    split_into_temporal_domains,
)
from euronext_simstock.models import SimStock
from euronext_simstock.similarity import SubstitutionEngine, build_dtw_similarity
from euronext_simstock.training import aggregate_by_stock, extract_embeddings, set_seed, train_all_domains

log = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train SimStock and build a stock substitution engine.")
    p.add_argument("--user-csv", type=Path, default=None, help="CSV univers, colonne minimale: ticker")
    p.add_argument("--sample", action="store_true", help="Utiliser l'univers sample si aucun CSV n'est fourni")
    p.add_argument("--force-refresh", action="store_true", help="Ignorer les caches Yahoo/metadata")
    p.add_argument("--save-name", type=str, default=None)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--log-level", type=str, default="INFO")

    p.add_argument("--mode", type=str, default="dtw", choices=["dtw", "cosine", "l2rank", "snapshot"], help="Méthode de construction de la similarité: dtw, cosine, l2rank ou snapshot")
    p.add_argument("--fast", action="store_true", help="Mode rapide : cosine similarity sans DTW. Alias pour --mode cosine.")
    p.add_argument("--feature-mode", type=str, default="paper", choices=["paper", "source_ma_relative"])
    p.add_argument("--dtw-reduction", type=str, default="pca", choices=["pca", "norm", "mean"])
    p.add_argument("--dtw-sim-method", type=str, default="rank", choices=["rank", "exp", "robust_minmax", "minmax"])
    p.add_argument("--dtw-n-jobs", type=int, default=-1)
    p.add_argument("--threshold", type=float, default=None, help="Seuil fixe. Si absent, on calibre via target_avg_substitutes.")
    p.add_argument("--target-avg-substitutes", type=float, default=None)
    p.add_argument("--same-sector-only", action="store_true")

    p.add_argument("--epochs-per-domain", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--negative-mode", type=str, default=None, choices=["dimension", "in_batch_hard"])
    p.add_argument("--loss-mode", type=str, default=None, choices=["softplus", "hinge"])
    p.add_argument("--resume-checkpoint", type=Path, default=None)
    p.add_argument("--skip-training", action="store_true", help="Sauter l'entraînement et charger E/hidden depuis --resume-checkpoint ou le modèle sauvegardé.")
    p.add_argument("--force-retrain", action="store_true", help="Relancer l'entraînement même si le modèle existe déjà sur disque.")

    p.add_argument("--train-start", type=str, default=None)
    p.add_argument("--train-end", type=str, default=None)
    p.add_argument("--reference-start", type=str, default=None)
    p.add_argument("--reference-end", type=str, default=None)
    p.add_argument("--test-end", type=str, default=None)
    p.add_argument("--domain-size-days", type=int, default=None)
    p.add_argument("--n-temporal-domains", type=int, default=None)
    p.add_argument("--min-history-days", type=int, default=None)
    return p


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def make_args(**overrides: Any) -> argparse.Namespace:
    """Helper pratique pour notebook : args = make_args(sample=True, epochs_per_domain=1)."""
    defaults = vars(parse_args([]))
    # argparse crée les noms en snake_case.
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _patch_configs(args: argparse.Namespace):
    tw = TimeWindow()
    fc = FeatureConfig()
    mc = ModelConfig()
    tc = TrainingConfig()
    sc = SubstitutionConfig()

    mapping = {
        "train_start": (tw, "train_start"),
        "train_end": (tw, "train_end"),
        "reference_start": (tw, "reference_start"),
        "reference_end": (tw, "reference_end"),
        "test_end": (tw, "test_end"),
        "domain_size_days": (tw, "domain_size_days"),
        "n_temporal_domains": (tw, "n_temporal_domains"),
        "min_history_days": (fc, "min_history_days"),
        "feature_mode": (fc, "feature_mode"),
        "epochs_per_domain": (tc, "epochs_per_domain"),
        "batch_size": (tc, "batch_size"),
        "num_workers": (tc, "num_workers"),
        "learning_rate": (tc, "learning_rate"),
        "save_name": (tc, "save_name"),
        "negative_mode": (mc, "negative_mode"),
        "loss_mode": (mc, "loss_mode"),
        "target_avg_substitutes": (sc, "target_avg_substitutes"),
    }
    for arg_name, (obj, attr) in mapping.items():
        if hasattr(args, arg_name):
            val = getattr(args, arg_name)
            if val is not None:
                setattr(obj, attr, val)

    if getattr(args, "same_sector_only", False):
        sc.same_sector_only = True

    # Dimensions cohérentes avec le feature engineering.
    mc.data_size = fc.data_size
    mc.noise_dim = fc.data_size
    mc.latent_dim = fc.data_size
    return tw, fc, mc, tc, sc


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda demandé mais CUDA n'est pas disponible.")
    return torch.device(device_arg)


def run_pipeline(args: Optional[argparse.Namespace] = None) -> SubstitutionEngine:
    args = args or parse_args()
    logging.basicConfig(
        level=getattr(logging, str(getattr(args, "log_level", "INFO")).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # --fast est un alias pour --mode cosine
    if getattr(args, "fast", False):
        args.mode = "cosine"

    tw, fc, mc, tc, sc = _patch_configs(args)
    set_seed(tc.seed)
    device = _resolve_device(getattr(args, "device", "auto"))

    log.info("Device: %s", device)
    log.info("TimeWindow: %s", tw)
    log.info("FeatureConfig: %s", fc)
    log.info("ModelConfig: %s", mc)
    log.info("TrainingConfig: %s", tc)

    log.info("=" * 70)
    log.info("ETAPE 1/7 — Univers")
    universe_df = get_universe(user_csv=getattr(args, "user_csv", None), sample=getattr(args, "sample", False))
    tickers = universe_df["ticker"].tolist()
    log.info("Univers: %d tickers", len(tickers))

    log.info("=" * 70)
    log.info("ETAPE 2/7 — OHLCV + secteurs")
    ohlcv = download_ohlcv(tickers, start=tw.train_start, end=tw.test_end, force_refresh=getattr(args, "force_refresh", False))
    sector_df = fetch_sector_metadata(list(ohlcv.keys()), force_refresh=getattr(args, "force_refresh", False))

    log.info("=" * 70)
    log.info("ETAPE 3/7 — Features + domaines temporels")
    sector_encoder = SectorEncoder().fit(sector_df["sector"])
    train_panel = build_panel(ohlcv, sector_df, sector_encoder, fc, start=tw.train_start, end=tw.train_end)
    reference_panel = build_panel(ohlcv, sector_df, sector_encoder, fc, start=tw.reference_start, end=tw.reference_end)
    train_domains = split_into_temporal_domains(
        train_panel,
        domain_size_days=tw.domain_size_days,
        n_domains=tw.n_temporal_domains,
    )
    train_dataloaders = [
        make_dataloader(d, fc, batch_size=tc.batch_size, shuffle=True, num_workers=tc.num_workers) for d in train_domains
    ]

    log.info("=" * 70)
    log.info("ETAPE 4/7 — Entraînement SimStock")
    model = SimStock(mc, num_sectors=sector_encoder.num_sectors, device=device).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Modèle: %.2fM paramètres", n_params / 1e6)
    save_path = MODELS_DIR / f"{tc.save_name}.pt"
    force_retrain = getattr(args, "force_retrain", False)
    resume_ckpt = getattr(args, "resume_checkpoint", None)
    skip_training = getattr(args, "skip_training", False)

    def _load_ckpt_for_inference(ckpt_path: Path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        E = ckpt["E"].to(device) if ckpt.get("E") is not None else None
        hidden = (
            (ckpt["hidden_h"].to(device), ckpt["hidden_c"].to(device))
            if ckpt.get("hidden_h") is not None
            else None
        )
        return E, hidden

    if skip_training:
        ckpt_src = resume_ckpt if (resume_ckpt is not None and Path(resume_ckpt).exists()) else save_path
        if not Path(ckpt_src).exists():
            raise RuntimeError(
                "--skip-training : aucun checkpoint disponible. "
                "Fournir --resume-checkpoint ou s'assurer que le modèle sauvegardé existe."
            )
        log.info("--skip-training : chargement E/hidden depuis %s", ckpt_src)
        E_final, hidden_final = _load_ckpt_for_inference(Path(ckpt_src))
    elif save_path.exists() and not force_retrain and resume_ckpt is None:
        log.info("Modèle existant détecté (%s) — skip training. Utiliser --force-retrain pour réentraîner.", save_path)
        E_final, hidden_final = _load_ckpt_for_inference(save_path)
    else:
        E_final, hidden_final = train_all_domains(
            train_dataloaders,
            model,
            tc,
            mc,
            device,
            save_path=save_path,
            resume_checkpoint=resume_ckpt,
        )

    log.info("=" * 70)
    log.info("ETAPE 5/7 — Embeddings référence")
    daily_path = EMBEDDINGS_DIR / f"{tc.save_name}_daily.parquet"
    partial_path = EMBEDDINGS_DIR / f"_partial_{tc.save_name}_daily.parquet"
    if daily_path.exists() and not force_retrain:
        log.info("Embeddings existants détectés — rechargement depuis disque : %s", daily_path)
        emb_daily = pd.read_parquet(daily_path)
    else:
        if partial_path.exists():
            log.info("Fichier partiel détecté — reprise depuis : %s", partial_path)
        emb_daily = extract_embeddings(
            reference_panel, model, mc, device, E_final, hidden_final,
            batch_size=tc.batch_size, feat_cfg=fc,
            partial_path=partial_path,
        )
        emb_daily.to_parquet(daily_path)
        log.info("Embeddings daily sauvegardés : %s", daily_path)
        if partial_path.exists():
            partial_path.unlink()
            log.info("Fichier partiel supprimé : %s", partial_path)

    log.info("=" * 70)
    log.info("ETAPE 6/7 — Similarité")
    mode = getattr(args, "mode", "dtw")
    if mode == "dtw":
        sim_matrix = build_dtw_similarity(
            emb_daily,
            reduction=getattr(args, "dtw_reduction", "pca"),
            sim_method=getattr(args, "dtw_sim_method", "rank"),
            n_jobs=getattr(args, "dtw_n_jobs", -1),
        )
        sim_path = EMBEDDINGS_DIR / f"{tc.save_name}_dtw_simmat.parquet"
        sim_matrix.to_parquet(sim_path)
        engine = SubstitutionEngine.from_similarity_matrix(sim_matrix, sector_df=sector_df, cfg=sc)
    elif mode == "snapshot":
        engine = SubstitutionEngine.from_daily_embeddings_snapshot(emb_daily, sector_df=sector_df, cfg=sc, aggregation="last")
    else:
        emb_stock = aggregate_by_stock(emb_daily, method="mean")
        stock_path = EMBEDDINGS_DIR / f"{tc.save_name}_stock.parquet"
        emb_stock.to_parquet(stock_path)
        engine = SubstitutionEngine.from_embeddings(emb_stock, sector_df=sector_df, cfg=sc, similarity_method=mode)
        # Sauvegarde la matrice de similarité (comme DTW) pour permettre une reprise
        sim_path = EMBEDDINGS_DIR / f"{tc.save_name}_{mode}_simmat.parquet"
        engine.similarity_dataframe().to_parquet(sim_path)
        log.info("Matrice similarité %s sauvegardée : %s", mode, sim_path)

    log.info("=" * 70)
    log.info("ETAPE 7/7 — Calibration seuil & sauvegardes")
    threshold = getattr(args, "threshold", None)
    if threshold is None:
        threshold = engine.threshold_for_avg_substitutes(
            target_avg_substitutes=sc.target_avg_substitutes,
            same_sector_only=sc.same_sector_only,
        )
        log.info("Seuil calibré pour %.1f substituts moyens: %.4f", sc.target_avg_substitutes, threshold)
    else:
        log.info("Seuil fourni: %.4f", threshold)
    sc.similarity_threshold = float(threshold)
    engine.cfg = sc

    engine_path = EMBEDDINGS_DIR / f"{tc.save_name}_engine.npz"
    engine.save(engine_path)

    diag = engine.similarity_diagnostics(same_sector_only=sc.same_sector_only)
    diag_path = EMBEDDINGS_DIR / f"{tc.save_name}_similarity_diagnostics.parquet"
    diag.to_parquet(diag_path)

    table = engine.substitution_table(threshold=threshold, same_sector_only=sc.same_sector_only, max_per_ticker=sc.max_per_ticker)
    table_path = EMBEDDINGS_DIR / f"{tc.save_name}_substitution_table.parquet"
    table.to_parquet(table_path)

    long_table = engine.substitution_table_long(threshold=threshold, same_sector_only=sc.same_sector_only)
    long_path = EMBEDDINGS_DIR / f"{tc.save_name}_substitution_long.parquet"
    long_table.to_parquet(long_path)

    run_meta = {
        "time_window": asdict(tw),
        "feature_config": asdict(fc),
        "model_config": asdict(mc),
        "training_config": asdict(tc),
        "substitution_config": asdict(sc),
        "mode": mode,
        "n_tickers_engine": len(engine.tickers),
    }
    pd.Series(run_meta, dtype=object).to_json(EMBEDDINGS_DIR / f"{tc.save_name}_run_meta.json", indent=2)

    log.info("Engine: %s", engine_path)
    log.info("Diagnostics: %s", diag_path)
    log.info("Table wide: %s", table_path)
    log.info("Table long: %s", long_path)
    return engine


def main() -> None:
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()
