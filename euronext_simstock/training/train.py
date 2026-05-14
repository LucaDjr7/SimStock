"""Boucles d'entraînement, checkpoints et extraction d'embeddings."""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from euronext_simstock.config import FeatureConfig, ModelConfig, TrainingConfig
from euronext_simstock.models import SimStock, make_noise

log = logging.getLogger(__name__)

HiddenState = Tuple[torch.Tensor, torch.Tensor]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_checkpoint(
    checkpoint_dir: Path,
    model: SimStock,
    optimizer: torch.optim.Optimizer,
    E: torch.Tensor,
    hidden: HiddenState,
    cfg_train: TrainingConfig,
    cfg_model: ModelConfig,
    domain_idx: int,
    epoch_idx: int,
    avg_loss: float,
) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "E": E.detach().cpu() if E is not None else None,
        "hidden_h": hidden[0].detach().cpu() if hidden is not None else None,
        "hidden_c": hidden[1].detach().cpu() if hidden is not None else None,
        "cfg_model": cfg_model.__dict__,
        "cfg_train": cfg_train.__dict__,
        "domain_idx": int(domain_idx),
        "epoch_idx": int(epoch_idx),
        "avg_loss": float(avg_loss),
    }
    unique_path = checkpoint_dir / f"{cfg_train.save_name}_D{domain_idx:02d}_ep{epoch_idx:02d}.pt"
    last_path = checkpoint_dir / f"{cfg_train.save_name}_last.pt"
    torch.save(payload, unique_path)
    torch.save(payload, last_path)
    log.info("Checkpoint sauvegardé : %s", last_path)


def _load_checkpoint(
    checkpoint_path: Path,
    model: SimStock,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[HiddenState], int, int]:
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    E = ckpt["E"].to(device) if ckpt.get("E") is not None else None
    hidden = None
    if ckpt.get("hidden_h") is not None and ckpt.get("hidden_c") is not None:
        hidden = (ckpt["hidden_h"].to(device), ckpt["hidden_c"].to(device))
    done_domain = int(ckpt.get("domain_idx", 0))
    done_epoch = int(ckpt.get("epoch_idx", 0))
    log.info(
        "Checkpoint chargé : %s | domaine=%d epoch=%d avg_loss=%.4f",
        checkpoint_path,
        done_domain,
        done_epoch,
        float(ckpt.get("avg_loss", float("nan"))),
    )
    return E, hidden, done_domain, done_epoch


def train_one_domain(
    dataloader: DataLoader,
    model: SimStock,
    optimizer: torch.optim.Optimizer,
    cfg_train: TrainingConfig,
    cfg_model: ModelConfig,
    device: torch.device,
    E_prev: Optional[torch.Tensor],
    hidden_prev: Optional[HiddenState],
    domain_idx: int,
    start_epoch: int = 0,
    checkpoint_dir: Optional[Path] = None,
) -> Tuple[torch.Tensor, HiddenState]:
    model.train()
    E = E_prev.detach() if E_prev is not None else None
    hidden = (hidden_prev[0].detach(), hidden_prev[1].detach()) if hidden_prev is not None else None
    z = make_noise((1, cfg_model.noise_dim), noise_type=cfg_model.noise_type, device=device)

    E_new, hidden_new = E, hidden
    for ep in range(start_epoch, cfg_train.epochs_per_domain):
        total_loss = total_triplet = total_uniformity = total_d_pos = total_d_neg = 0.0
        n_batches = 0

        # État temporel stable dans l'epoch ; update à la fin de l'epoch.
        E_epoch = E
        hidden_epoch = hidden

        pbar = tqdm(dataloader, desc=f"D{domain_idx} ep{ep + 1}/{cfg_train.epochs_per_domain}")
        for X, sector in pbar:
            X = X.to(device, non_blocking=True)
            sector = sector.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            E_new, hidden_new, loss = model(X, z, sector, E=E_epoch, hidden=hidden_epoch)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Loss non finie : {loss.item()} (domain={domain_idx}, epoch={ep + 1}, batch={n_batches + 1})"
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_train.grad_clip)
            optimizer.step()

            parts = getattr(model, "last_loss_parts", {})
            total_loss += float(loss.detach().cpu())
            total_triplet += float(parts.get("triplet", 0.0))
            total_uniformity += float(parts.get("uniformity", 0.0))
            total_d_pos += float(parts.get("d_pos", 0.0))
            total_d_neg += float(parts.get("d_neg", 0.0))
            n_batches += 1
            pbar.set_postfix(
                loss=f"{total_loss / n_batches:.4f}",
                triplet=f"{total_triplet / n_batches:.4f}",
                unif=f"{total_uniformity / n_batches:.4f}",
                dpos=f"{total_d_pos / n_batches:.3f}",
                dneg=f"{total_d_neg / n_batches:.3f}",
            )

        if n_batches == 0:
            raise RuntimeError(f"Dataloader vide pour le domaine {domain_idx}.")

        E = E_new.detach()
        hidden = (hidden_new[0].detach(), hidden_new[1].detach())
        avg_loss = total_loss / n_batches
        log.info(
            "[Domain %d epoch %d] avg_loss=%.4f | triplet=%.4f | uniformity=%.4f | d_pos=%.3f | d_neg=%.3f",
            domain_idx,
            ep + 1,
            avg_loss,
            total_triplet / n_batches,
            total_uniformity / n_batches,
            total_d_pos / n_batches,
            total_d_neg / n_batches,
        )

        if checkpoint_dir is not None:
            _save_checkpoint(
                checkpoint_dir=Path(checkpoint_dir),
                model=model,
                optimizer=optimizer,
                E=E,
                hidden=hidden,
                cfg_train=cfg_train,
                cfg_model=cfg_model,
                domain_idx=domain_idx,
                epoch_idx=ep + 1,
                avg_loss=avg_loss,
            )

    return E.detach(), (hidden[0].detach(), hidden[1].detach())


def train_all_domains(
    dataloaders: List[DataLoader],
    model: SimStock,
    cfg_train: TrainingConfig,
    cfg_model: ModelConfig,
    device: torch.device,
    save_path: Optional[Path] = None,
    checkpoint_dir: Optional[Path] = None,
    resume_checkpoint: Optional[Path] = None,
) -> Tuple[torch.Tensor, HiddenState]:
    set_seed(cfg_train.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_train.learning_rate)

    if checkpoint_dir is None and save_path is not None:
        checkpoint_dir = Path(save_path).parent / "checkpoints" / cfg_train.save_name

    E: Optional[torch.Tensor] = None
    hidden: Optional[HiddenState] = None
    start_domain = 1
    start_epoch = 0

    if resume_checkpoint is not None and Path(resume_checkpoint).exists():
        E, hidden, done_domain, done_epoch = _load_checkpoint(resume_checkpoint, model, optimizer, device)
        if done_epoch >= cfg_train.epochs_per_domain:
            start_domain = done_domain + 1
            start_epoch = 0
        else:
            start_domain = done_domain
            start_epoch = done_epoch
        log.info("Reprise depuis domaine %d, epoch index %d", start_domain, start_epoch)

    for domain_idx, dataloader in enumerate(dataloaders, start=1):
        if domain_idx < start_domain:
            log.info("Skip domaine %d/%d déjà terminé", domain_idx, len(dataloaders))
            continue
        ep0 = start_epoch if domain_idx == start_domain else 0
        log.info("==> Domaine temporel %d/%d (%d batches), start_epoch=%d", domain_idx, len(dataloaders), len(dataloader), ep0)
        E, hidden = train_one_domain(
            dataloader,
            model,
            optimizer,
            cfg_train,
            cfg_model,
            device,
            E_prev=E,
            hidden_prev=hidden,
            domain_idx=domain_idx,
            start_epoch=ep0,
            checkpoint_dir=checkpoint_dir,
        )

    if E is None or hidden is None:
        raise RuntimeError("Entraînement non exécuté : aucun état E/hidden final.")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "E": E.cpu(),
                "hidden_h": hidden[0].cpu(),
                "hidden_c": hidden[1].cpu(),
                "cfg_model": cfg_model.__dict__,
                "cfg_train": cfg_train.__dict__,
            },
            save_path,
        )
        log.info("Modèle sauvegardé : %s", save_path)

    return E, hidden


@torch.no_grad()
def extract_embeddings(
    panel,
    model: SimStock,
    cfg_model: ModelConfig,
    device: torch.device,
    E: torch.Tensor,
    hidden: HiddenState,
    batch_size: int = 512,
    feat_cfg: Optional[FeatureConfig] = None,
    partial_path: Optional[Path] = None,
    save_every_n_batches: int = 50,
) -> "pd.DataFrame":
    """Calcule un embedding CLS pour chaque ligne (ticker, date) du panel.

    Si partial_path est fourni, sauvegarde un checkpoint partiel toutes les
    save_every_n_batches itérations et reprend depuis ce fichier s'il existe.
    """
    import pandas as pd

    from euronext_simstock.data.preprocessing import panel_to_tensors

    model.eval()
    feat_cfg = feat_cfg or FeatureConfig()
    X_all, sector_all, tickers, dates = panel_to_tensors(panel, feat_cfg)
    z = make_noise((1, cfg_model.noise_dim), noise_type=cfg_model.noise_type, device=device)

    # --- Reprise depuis checkpoint partiel ---
    resume_from_row = 0
    done_df: Optional["pd.DataFrame"] = None
    if partial_path is not None:
        partial_path = Path(partial_path)
        if partial_path.exists():
            done_df = pd.read_parquet(partial_path)
            resume_from_row = len(done_df)
            log.info(
                "Reprise embeddings : %d/%d lignes déjà calculées (%s)",
                resume_from_row, len(X_all), partial_path,
            )

    def _flush(embs_so_far: list) -> None:
        """Sauvegarde partielle : done_df + nouvelles lignes."""
        stacked = np.vstack(embs_so_far)
        n_new = len(stacked)
        e_cols = [f"e_{j}" for j in range(stacked.shape[1])]
        df_new = pd.DataFrame(stacked, columns=e_cols)
        df_new["ticker"] = tickers[resume_from_row: resume_from_row + n_new]
        df_new["date"] = pd.to_datetime(dates[resume_from_row: resume_from_row + n_new])
        df_new = df_new[["ticker", "date"] + e_cols]
        combined = pd.concat([done_df, df_new], ignore_index=True) if done_df is not None else df_new
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(partial_path)
        log.info(
            "Checkpoint partiel : %d/%d lignes → %s",
            len(combined), len(X_all), partial_path,
        )

    all_embs: list = []
    batch_count = 0
    for i in tqdm(range(resume_from_row, len(X_all), batch_size), desc="Extract embeddings"):
        X_batch = X_all[i : i + batch_size].to(device, non_blocking=True)
        sector_batch = sector_all[i : i + batch_size].to(device, non_blocking=True)
        cls_emb, _ = model(X_batch, z, sector_batch, E=E, hidden=hidden, return_embedding=True)
        all_embs.append(cls_emb.cpu().numpy())
        batch_count += 1

        if partial_path is not None and batch_count % save_every_n_batches == 0:
            _flush(all_embs)

    # --- Assemblage final ---
    if not all_embs:
        # Tous les rows étaient déjà dans le fichier partiel
        if done_df is not None:
            return done_df
        raise RuntimeError("extract_embeddings : aucun embedding calculé et pas de fichier partiel.")

    embs = np.vstack(all_embs)
    n_new = len(embs)
    cols = [f"e_{i}" for i in range(embs.shape[1])]
    df_new = pd.DataFrame(embs, columns=cols)
    df_new["ticker"] = tickers[resume_from_row: resume_from_row + n_new]
    df_new["date"] = pd.to_datetime(dates[resume_from_row: resume_from_row + n_new])
    df_new = df_new[["ticker", "date"] + cols]

    if done_df is not None:
        return pd.concat([done_df, df_new], ignore_index=True)
    return df_new


def aggregate_by_stock(embeddings_df: "pd.DataFrame", method: str = "mean") -> "pd.DataFrame":
    e_cols = [c for c in embeddings_df.columns if c.startswith("e_")]
    if not e_cols:
        raise ValueError("Aucune colonne e_* dans embeddings_df.")
    method = method.lower()
    if method == "mean":
        agg = embeddings_df.groupby("ticker", sort=False)[e_cols].mean()
    elif method == "last":
        agg = embeddings_df.sort_values("date").groupby("ticker", sort=False)[e_cols].last()
    else:
        raise ValueError("method inconnu. Choisir 'mean' ou 'last'.")
    return agg.reset_index()
