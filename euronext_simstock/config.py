"""
Configuration centrale du pipeline Euronext-SimStock.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = ARTIFACTS_DIR / "data"
CACHE_DIR = ARTIFACTS_DIR / "cache"
MODELS_DIR = ARTIFACTS_DIR / "models"
EMBEDDINGS_DIR = ARTIFACTS_DIR / "embeddings"
OUTPUT_DIR = ARTIFACTS_DIR / "output"

for _d in (DATA_DIR, CACHE_DIR, MODELS_DIR, EMBEDDINGS_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Univers Euronext — suffixes Yahoo Finance
# ---------------------------------------------------------------------------
EURONEXT_SUFFIXES: Dict[str, str] = {
    ".PA": "Euronext Paris",
    ".AS": "Euronext Amsterdam",
    ".BR": "Euronext Brussels",
    ".LS": "Euronext Lisbon",
    ".IR": "Euronext Dublin",
    ".OL": "Euronext Oslo / Oslo Børs",
    ".MI": "Borsa Italiana / Euronext Milan",
}

EURONEXT_SUFFIXES_ENABLED: List[str] = list(EURONEXT_SUFFIXES.keys())


# ---------------------------------------------------------------------------
# Fenêtres temporelles
# ---------------------------------------------------------------------------
@dataclass
class TimeWindow:
    """
    Découpage chronologique recommandé :
    - train : apprend les représentations temporelles ;
    - reference : construit les similarités utilisables ensuite ;
    - test : période hors-échantillon pour évaluer les choix.
    """

    train_start: str = "2019-01-01"
    train_end: str = "2023-12-31"
    reference_start: str = "2024-01-01"
    reference_end: str = "2024-12-31"
    test_start: str = "2025-01-01"
    test_end: str = "2025-12-31"

    # Fenêtres temporelles D_1...D_T : 126 jours boursiers ~= 6 mois.
    domain_size_days: int = 126
    n_temporal_domains: Optional[int] = None


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
@dataclass
class FeatureConfig:
    """
    feature_mode="paper" : calcule d'abord les 5 z-features décrites dans le papier,
    puis applique les moyennes mobiles temporelles [5, 10, 15, 20, 25].

    feature_mode="source_ma_relative" : conserve l'ancienne variante MA(C)/C_t - 1,
    utile uniquement pour reproduire tes runs précédents.
    """

    ma_windows: List[int] = field(default_factory=lambda: [5, 10, 15, 20, 25])
    n_price_features: int = 5
    min_history_days: int = 200
    feature_mode: str = "paper"

    @property
    def data_size(self) -> int:
        return self.n_price_features * len(self.ma_windows)


# ---------------------------------------------------------------------------
# Hyperparamètres modèle
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    data_size: int = 25
    noise_dim: int = 25
    latent_dim: int = 25
    hidden_dim: int = 128
    sector_emb: int = 256
    num_rnn_layer: int = 1
    lambda_values: float = 0.7
    noise_type: str = "Gaussian"
    triplet_margin: float = 0.4

    # "in_batch_hard" = négatifs intra-batch (hard mining), résout le collapse d_neg→2.
    # "dimension" = corruption de dimension post-tokenisation (mode original).
    negative_mode: str = "in_batch_hard"

    # "softplus" stabilise l'optimisation ; "hinge" correspond au max(0, ...).
    loss_mode: str = "softplus"

    # Peut rendre la loss totale négative : toujours lire triplet/uniformity séparément.
    uniformity_weight: float = 0.01


# ---------------------------------------------------------------------------
# Hyperparamètres entraînement
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    batch_size: int = 512
    learning_rate: float = 1e-3
    epochs_per_domain: int = 5
    grad_clip: float = 1.0
    seed: int = 42
    save_name: str = "euronext_simstock_v1"
    num_workers: int = 0


# ---------------------------------------------------------------------------
# Substituabilité
# ---------------------------------------------------------------------------
@dataclass
class SubstitutionConfig:
    similarity_threshold: float = 0.85
    same_sector_only: bool = False
    default_top_k: int = 5
    target_avg_substitutes: int = 10
    max_per_ticker: int = 20
