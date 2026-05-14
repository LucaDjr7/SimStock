"""Couches utilitaires de SimStock."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Sector_embedding(nn.Module):
    """Embedding secteur. L'ID 0 = Unknown avec vecteur initialisé à zéro."""

    def __init__(self, emb_dim: int, num_sectors: int):
        super().__init__()
        if num_sectors < 1:
            raise ValueError("num_sectors doit être >= 1")
        self.embedding = nn.Embedding(num_sectors, emb_dim, padding_idx=0)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.embedding.weight[0].zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2 and x.size(1) == 1:
            x = x.squeeze(1)
        return self.embedding(x.long())


class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class NumericalEmbedder(nn.Module):
    """Tokenizer numérique : x_j -> x_j * W_j + b_j."""

    def __init__(self, dim: int, num_numerical_types: int):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim) * 0.02)
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.weights + self.biases


def feed_forward(dim: int, mult: int = 4, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim),
    )


def make_noise(shape: tuple, noise_type: str = "Gaussian", device: torch.device | str = "cpu") -> torch.Tensor:
    if noise_type == "Gaussian":
        return torch.randn(shape, device=device)
    if noise_type == "Uniform":
        return torch.rand(shape, device=device) * 2.0 - 1.0
    raise ValueError(f"noise_type inconnu : {noise_type}")
