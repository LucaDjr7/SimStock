"""SimStock — temporal SSL avec dimension corruption et DRAIN-style LSTM."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from euronext_simstock.models.layers import NumericalEmbedder, Sector_embedding


HiddenState = Tuple[torch.Tensor, torch.Tensor]


def uniformity_loss(z: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """Uniformity loss sur paires hors diagonale, pour limiter le collapse."""
    if z.shape[0] < 2:
        return z.new_tensor(0.0)
    sq = torch.cdist(z, z, p=2).pow(2)
    mask = ~torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    return torch.log(torch.exp(-t * sq[mask]).mean() + 1e-8)


class SimStock(nn.Module):
    TOKEN_DIM = 256
    N_HEADS = 4
    HEAD_DIM = TOKEN_DIM // N_HEADS

    def __init__(self, cfg, num_sectors: int, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.data_size = int(cfg.data_size)
        self.lambda_values = float(cfg.lambda_values)
        self.num_rnn_layer = int(cfg.num_rnn_layer)
        self.negative_mode = str(getattr(cfg, "negative_mode", "dimension")).lower()
        self.loss_mode = str(getattr(cfg, "loss_mode", "softplus")).lower()
        self.uniformity_weight = float(getattr(cfg, "uniformity_weight", 0.0))
        self.last_loss_parts = {"triplet": 0.0, "uniformity": 0.0, "total": 0.0}

        if cfg.noise_dim != self.data_size:
            raise ValueError("cfg.noise_dim doit être égal à cfg.data_size dans cette implémentation.")
        if cfg.latent_dim != cfg.noise_dim:
            raise ValueError("cfg.latent_dim doit être égal à cfg.noise_dim pour préserver les dimensions DRAIN.")
        if self.negative_mode not in {"dimension", "in_batch_hard"}:
            raise ValueError("negative_mode doit valoir 'dimension' ou 'in_batch_hard'.")
        if self.loss_mode not in {"softplus", "hinge"}:
            raise ValueError("loss_mode doit valoir 'softplus' ou 'hinge'.")

        # DRAIN-style generator g_phi
        self.init_lin_h = nn.Linear(cfg.noise_dim, cfg.latent_dim)
        self.init_lin_c = nn.Linear(cfg.noise_dim, cfg.latent_dim)
        self.init_input = nn.Linear(cfg.noise_dim, cfg.latent_dim)
        self.rnn = nn.LSTM(cfg.latent_dim, cfg.latent_dim, self.num_rnn_layer)

        e_dim = self._weights_total_dim()
        self.lin_transform_down = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, e_dim),
        )
        self.lin_transform_up = nn.Sequential(
            nn.Linear(e_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
        )

        # Static metadata
        self.sector_emb = Sector_embedding(cfg.sector_emb, num_sectors)
        self.sector_projection = nn.Linear(cfg.sector_emb, self.data_size)

        # Feature tokenizer
        self.numerical_embedder = NumericalEmbedder(self.TOKEN_DIM, self.data_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.TOKEN_DIM) * 0.02)

        self.scale = self.HEAD_DIM ** -0.5
        self.dropout = nn.Dropout(0.05)
        self.out_norm = nn.LayerNorm(self.TOKEN_DIM)

    def _weights_total_dim(self) -> int:
        nt = self.data_size + 1
        td = self.TOKEN_DIM
        return (3 * nt * td) + (td * td) + td

    def _build_attn_weights(self, E: torch.Tensor):
        nt = self.data_size + 1
        td = self.TOKEN_DIM
        if E.dim() != 2 or E.shape[0] != 1:
            raise ValueError(f"E doit avoir la shape (1, e_dim), reçu {tuple(E.shape)}")
        offset = 0
        q_w = E[:, offset : offset + nt * td].reshape(nt, td)
        offset += nt * td
        k_w = E[:, offset : offset + nt * td].reshape(nt, td)
        offset += nt * td
        v_w = E[:, offset : offset + nt * td].reshape(nt, td)
        offset += nt * td
        out_w = E[:, offset : offset + td * td].reshape(td, td)
        offset += td * td
        out_b = E[:, offset : offset + td].reshape(1, td)
        return q_w, k_w, v_w, out_w, out_b

    def _initial_hidden(self, z: torch.Tensor) -> HiddenState:
        init_h = [torch.tanh(self.init_lin_h(z)) for _ in range(self.num_rnn_layer)]
        init_c = [torch.tanh(self.init_lin_c(z)) for _ in range(self.num_rnn_layer)]
        return torch.stack(init_h, dim=0), torch.stack(init_c, dim=0)

    def generate_dynamic_params(
        self,
        z: torch.Tensor,
        E: Optional[torch.Tensor] = None,
        hidden: Optional[HiddenState] = None,
    ) -> Tuple[torch.Tensor, HiddenState]:
        """Génère E_new et hidden_new depuis l'état temporel précédent."""
        if (E is None) != (hidden is None):
            raise ValueError("E et hidden doivent être tous deux None ou tous deux fournis.")
        if E is None:
            hidden = self._initial_hidden(z)
            inputs = torch.tanh(self.init_input(z))
        else:
            inputs = self.lin_transform_up(E)
        out, hidden_new = self.rnn(inputs.unsqueeze(0), hidden)
        E_new = self.lin_transform_down(out.squeeze(0))
        return E_new, hidden_new

    def _tokenize(self, X: torch.Tensor) -> torch.Tensor:
        tokens = self.numerical_embedder(X)
        cls = repeat(self.cls_token, "1 1 d -> b 1 d", b=X.shape[0])
        return torch.cat((cls, tokens), dim=1)

    def augment_positive(self, tokens: torch.Tensor, lam: float) -> torch.Tensor:
        idx = torch.randperm(tokens.shape[-1], device=tokens.device)
        return lam * tokens + (1.0 - lam) * tokens[:, :, idx]

    def augment_negative_dimension(self, tokens: torch.Tensor, lam: float) -> torch.Tensor:
        idx = torch.randperm(tokens.shape[-1], device=tokens.device)
        return (1.0 - lam) * tokens + lam * tokens[:, :, idx]

    def augment_positive_features(self, X: torch.Tensor, lam: float) -> torch.Tensor:
        """Positive: gentle mix on raw 25-d features (before tokenisation)."""
        idx = torch.randperm(X.shape[-1], device=X.device)
        return lam * X + (1.0 - lam) * X[:, idx]

    def augment_negative_features(self, X: torch.Tensor, lam: float) -> torch.Tensor:
        """Negative: heavy dimension corruption on raw 25-d features (before tokenisation)."""
        idx = torch.randperm(X.shape[-1], device=X.device)
        return (1.0 - lam) * X + lam * X[:, idx]

    def _attn_forward(self, tokens: torch.Tensor, weights) -> Tuple[torch.Tensor, torch.Tensor]:
        q_w, k_w, v_w, out_w, out_b = weights
        q = torch.einsum("bnd,nd->bnd", tokens, q_w)
        k = torch.einsum("bnd,nd->bnd", tokens, k_w)
        v = torch.einsum("bnd,nd->bnd", tokens, v_w)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.N_HEADS), (q, k, v))
        sim = torch.einsum("b h i d, b h j d -> b h i j", q * self.scale, k)
        attn = self.dropout(sim.softmax(dim=-1))
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = torch.einsum("b n i, i j -> b n j", out, out_w) + out_b
        out = self.out_norm(out + tokens)
        return out[:, 0, :], attn

    def _triplet_from_distances(self, d_pos: torch.Tensor, d_neg: torch.Tensor) -> torch.Tensor:
        raw = d_pos - d_neg + float(self.cfg.triplet_margin)
        if self.loss_mode == "hinge":
            return torch.relu(raw).mean()
        return F.softplus(raw).mean()

    def forward(
        self,
        X: torch.Tensor,
        z: torch.Tensor,
        sector: torch.Tensor,
        E: Optional[torch.Tensor] = None,
        hidden: Optional[HiddenState] = None,
        return_embedding: bool = False,
    ):
        E_new, hidden_new = self.generate_dynamic_params(z=z, E=E, hidden=hidden)
        weights = self._build_attn_weights(E_new)

        sector_vec = self.sector_projection(self.sector_emb(sector))
        X = X + sector_vec
        tokens_anchor = self._tokenize(X)

        if return_embedding:
            cls_emb, attn = self._attn_forward(tokens_anchor, weights)
            return cls_emb, attn

        # Positive: augmentation sur features 25-d avant tokenisation
        X_pos = self.augment_positive_features(X, self.lambda_values)
        tokens_pos = self._tokenize(X_pos)
        anchor_emb, _ = self._attn_forward(tokens_anchor, weights)
        pos_emb, _ = self._attn_forward(tokens_pos, weights)

        anchor_n = F.normalize(anchor_emb, p=2, dim=-1)
        pos_n = F.normalize(pos_emb, p=2, dim=-1)
        d_pos = torch.norm(anchor_n - pos_n, p=2, dim=1)

        if self.negative_mode == "dimension" or X.shape[0] < 2:
            # Négatif par corruption de dimension sur features 25-d
            X_neg = self.augment_negative_features(X, self.lambda_values)
            tokens_neg = self._tokenize(X_neg)
            neg_emb, _ = self._attn_forward(tokens_neg, weights)
            neg_n = F.normalize(neg_emb, p=2, dim=-1)
        else:
            # Négatifs intra-batch: hard mining sur les embeddings anchor
            dist_matrix = torch.cdist(anchor_n, anchor_n, p=2)
            eye = torch.eye(dist_matrix.shape[0], dtype=torch.bool, device=dist_matrix.device)
            dist_matrix = dist_matrix.masked_fill(eye, float("inf"))
            hard_neg_idx = torch.argmin(dist_matrix, dim=1)
            neg_n = anchor_n[hard_neg_idx]

        d_neg = torch.norm(anchor_n - neg_n, p=2, dim=1)
        triplet = self._triplet_from_distances(d_pos, d_neg)
        unif = uniformity_loss(anchor_n)
        loss = triplet + self.uniformity_weight * unif

        self.last_loss_parts = {
            "triplet": float(triplet.detach().cpu()),
            "uniformity": float(unif.detach().cpu()),
            "total": float(loss.detach().cpu()),
            "d_pos": float(d_pos.mean().detach().cpu()),
            "d_neg": float(d_neg.mean().detach().cpu()),
        }
        return E_new, hidden_new, loss
