from __future__ import annotations

import logging

from euronext_simstock.config import EMBEDDINGS_DIR, SubstitutionConfig, TrainingConfig
from euronext_simstock.similarity import SubstitutionEngine

logging.basicConfig(level="INFO", format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    tc = TrainingConfig()
    engine_path = EMBEDDINGS_DIR / f"{tc.save_name}_engine.npz"
    engine = SubstitutionEngine.load(engine_path, cfg=SubstitutionConfig())
    print(f"Engine chargé: {len(engine.tickers)} tickers")

    ticker = engine.tickers[0]
    print(f"\nSubstituts de {ticker}:")
    for sub, sim in engine.all_substitutes(ticker, threshold=engine.cfg.similarity_threshold)[:20]:
        print(f"  {sub:12s} sim={sim:.4f} sector={engine.sector_lookup.get(sub, 'Unknown')}")

    print("\nDiagnostics seuils:")
    print(engine.similarity_diagnostics().to_string(index=False))

    if len(engine.tickers) >= 4:
        trader_x = {engine.tickers[0]: 100.0, engine.tickers[1]: 50.0}
        trader_y = {engine.tickers[0]: 100.0, engine.tickers[2]: 50.0}
        print("\nComparaison trades:")
        print(engine.compare_trades(trader_x, trader_y, threshold=engine.cfg.similarity_threshold))


if __name__ == "__main__":
    main()
