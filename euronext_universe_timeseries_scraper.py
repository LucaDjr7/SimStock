"""
euronext_universe_timeseries_scraper.py
=======================================

Builds a dated Euronext equity universe for 2019-01-01 -> 2025-12-31.

Outputs
-------
1) euronext_universe_timeseries.csv
   One row per available trading date and ticker. This is the file to use
   when you need a dated / survivorship-bias-aware universe.

2) euronext_universe_for_load_user_universe.csv
   One row per ticker, compatible with the current load_user_universe(csv_path)
   function from universe.py.

3) euronext_universe_summary.csv
   One row per ticker with first/last trade dates and a simple delisting flag.

4) euronext_current_universe.csv
   Snapshot scraped from Euronext Live's public stock directory.

5) euronext_delisting_notices.csv, optional
   Best-effort scrape of Euronext cash notices containing delisting keywords.

Important limits
----------------
- The public Euronext stock directory is a current snapshot, not a historical
  security master. Therefore, old delisted names are recovered only if they can
  be found through Euronext cash notices and/or still resolve on Yahoo Finance.
- For a legally complete historical Euronext security master, use Euronext's
  licensed reference data / corporate actions products. This script is designed
  for a reproducible public-data workflow, not as a certified reference-data feed.
- The script keeps a normal requests.Session and handles cookies and rate limits.
  It does not bypass CAPTCHA or anti-bot challenges. If a challenge appears, it
  stops and tells you which URL failed.

Usage
-----
python euronext_universe_timeseries_scraper.py \
  --start 2019-01-01 \
  --end 2025-12-31 \
  --out-dir data/euronext_universe \
  --include-notices \
  --max-workers 4

Then, with your existing universe.py:

from pathlib import Path
from euronext_simstock.universe import load_user_universe
universe = load_user_universe(
    Path("data/euronext_universe/euronext_universe_for_load_user_universe.csv")
)

If you need the universe as of a specific historical date, use:

from euronext_universe_timeseries_scraper import load_universe_as_of
universe_2021 = load_universe_as_of(
    "data/euronext_universe/euronext_universe_timeseries.csv",
    as_of="2021-12-31",
)
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import html
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import quote, urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "yfinance is required. Install with: pip install yfinance requests beautifulsoup4 tqdm pandas pyarrow"
    ) from exc


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("euronext-universe")


# ---------------------------------------------------------------------------
# Euronext / Yahoo mappings
# ---------------------------------------------------------------------------

# MIC groups reused by Euronext Live's public DataTables endpoint.
# These groups cover the main/current equity directories, including Growth,
# Access and equivalent local markets where Euronext exposes them.
MARKET_GROUPS: dict[str, dict[str, Any]] = {
    "amsterdam": {
        "mics": ["XAMS", "TNLA"],
        "exchange_suffix": ".AS",
        "label": "Euronext Amsterdam",
    },
    "brussels": {
        "mics": ["XBRU", "ALXB", "MLXB", "TNLB", "ENXB"],
        "exchange_suffix": ".BR",
        "label": "Euronext Brussels",
    },
    "dublin": {
        "mics": ["XESM", "XMSM", "XATL"],
        "exchange_suffix": ".IR",
        "label": "Euronext Dublin",
    },
    "lisbon": {
        "mics": ["XLIS", "ALXL", "ENXL"],
        "exchange_suffix": ".LS",
        "label": "Euronext Lisbon",
    },
    "milan": {
        "mics": ["MTAA", "EXGM", "MTAH", "MIVX", "BGEM", "ETLX"],
        "exchange_suffix": ".MI",
        "label": "Euronext Milan",
    },
    "oslo": {
        "mics": ["XOSL", "MERK", "XOAS"],
        "exchange_suffix": ".OL",
        "label": "Oslo Børs",
    },
    "paris": {
        "mics": ["XPAR", "ALXP", "XMLI"],
        "exchange_suffix": ".PA",
        "label": "Euronext Paris",
    },
}

MIC_TO_MARKET: dict[str, str] = {
    mic: market for market, cfg in MARKET_GROUPS.items() for mic in cfg["mics"]
}
MIC_TO_SUFFIX: dict[str, str] = {
    mic: cfg["exchange_suffix"] for _, cfg in MARKET_GROUPS.items() for mic in cfg["mics"]
}
MIC_TO_LABEL: dict[str, str] = {
    mic: cfg["label"] for _, cfg in MARKET_GROUPS.items() for mic in cfg["mics"]
}
KNOWN_SUFFIXES = {cfg["exchange_suffix"] for cfg in MARKET_GROUPS.values()}

ISIN_RE = re.compile(r"\b[A-Z]{2}[A-Z0-9]{9}[0-9]\b")
ADN_RE = re.compile(r"/product/equities/([A-Z]{2}[A-Z0-9]{9}[0-9]-[A-Z0-9]+)", re.I)
NOTICE_ID_RE = re.compile(r"\b[A-Z]{2,6}_\d{8}_\d{4,6}_[A-Z]{2,4}\b")


@dataclass(frozen=True)
class RequestConfig:
    timeout: int = 30
    min_delay: float = 0.25
    max_retries: int = 5
    backoff_factor: float = 1.2


# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------


def make_session(cfg: RequestConfig = RequestConfig()) -> requests.Session:
    """Create a requests session with browser-like headers, retries and cookies."""
    session = requests.Session()
    retry = Retry(
        total=cfg.max_retries,
        connect=cfg.max_retries,
        read=cfg.max_retries,
        status=cfg.max_retries,
        backoff_factor=cfg.backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,fr;q=0.8",
            "Connection": "keep-alive",
        }
    )
    return session


def assert_not_challenge(resp: requests.Response, url: str) -> None:
    """
    Detect probable anti-bot pages without false-positive blocking normal HTML pages.

    This deliberately does not bypass CAPTCHA. It only makes the error message more
    precise and avoids treating generic words such as "cloudflare" or
    "enable javascript" as sufficient proof of a challenge on a 200 page.
    """
    text = resp.text[:10000].lower() if resp.text else ""

    if resp.status_code in {401, 403, 429}:
        raise RuntimeError(
            f"HTTP {resp.status_code} on {url}. This is probably an access block, "
            "rate limit, or anti-bot response, not a missing Internet connection. "
            "The script intentionally does not bypass CAPTCHA. Use --current-csv "
            "with a manually exported/current universe CSV, or retry later with "
            "a higher --min-delay and lower --max-workers."
        )

    hard_tokens = (
        "captcha",
        "recaptcha",
        "hcaptcha",
        "cf-chl",
        "are you a human",
        "access denied",
        "unusual traffic",
    )
    found = [tok for tok in hard_tokens if tok in text]
    if found:
        raise RuntimeError(
            f"Possible anti-bot/CAPTCHA page on {url}; found tokens: {found}. "
            "The script intentionally does not bypass CAPTCHA. Use --current-csv "
            "with a manually exported/current universe CSV, or retry later with "
            "a higher --min-delay and lower --max-workers."
        )


# ---------------------------------------------------------------------------
# HTML / parsing helpers
# ---------------------------------------------------------------------------


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    s = html.unescape(str(value))
    s = BeautifulSoup(s, "html.parser").get_text(" ", strip=True)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_attr(html_value: Any, attr_name: str) -> str:
    if html_value is None:
        return ""
    soup = BeautifulSoup(str(html_value), "html.parser")
    node = soup.find(attrs={attr_name: True})
    return str(node.get(attr_name, "")).strip() if node else ""


def extract_href(html_value: Any) -> str:
    if html_value is None:
        return ""
    soup = BeautifulSoup(str(html_value), "html.parser")
    a = soup.find("a", href=True)
    return str(a["href"]).strip() if a else ""


def extract_isin(values: Iterable[Any]) -> str:
    joined = " ".join(clean_text(v) for v in values)
    match = ISIN_RE.search(joined)
    return match.group(0).upper() if match else ""


def extract_adn_from_href(href: str) -> tuple[str, str]:
    """Return (adn, mic) from a Euronext product URL."""
    match = ADN_RE.search(href or "")
    if not match:
        return "", ""
    adn = match.group(1).upper()
    mic = adn.rsplit("-", 1)[-1].upper()
    return adn, mic


def yahoo_ticker_from_euronext_symbol(symbol: str, exchange_suffix: str) -> str:
    """Convert an Euronext symbol to the Yahoo symbol style expected by yfinance."""
    sym = (symbol or "").strip().upper()
    if not sym:
        return ""
    if any(sym.endswith(suf) for suf in KNOWN_SUFFIXES):
        return sym

    # Clean common display artefacts but keep characters Yahoo sometimes accepts.
    sym = sym.replace("/", "-")
    sym = re.sub(r"\s+", "-", sym)
    sym = re.sub(r"[^A-Z0-9.\-]", "", sym)
    if not sym:
        return ""
    return f"{sym}{exchange_suffix}"


def normalize_date(value: Any) -> Optional[pd.Timestamp]:
    if value is None or str(value).strip() == "":
        return None
    try:
        return pd.to_datetime(str(value), dayfirst=True, errors="coerce")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Euronext current stock directory
# ---------------------------------------------------------------------------

# The DataTables JSON endpoint used by the public directory can return an empty
# payload depending on the session / current site version. The CSV export is
# more stable and is the preferred source for the current directory.
EURONEXT_STOCK_DOWNLOAD_URL = (
    "https://live.euronext.com/en/pd_es/data/stocks/download"
    "?mics=dm_all_stock"
    "&initialLetter="
    "&fe_type=csv"
    "&fe_decimal_separator=."
    "&fe_date_format=d%2Fm%2FY"
)

# Keep the old JSON endpoint as a fallback only.
def euronext_stock_endpoint(mics: list[str]) -> str:
    return "https://live.euronext.com/en/pd_es/data/stocks?mics=" + quote(",".join(mics), safe="")


def datatable_payload(start: int = 0, length: int = 100, draw: int = 3) -> dict[str, Any]:
    return {
        "draw": draw,
        "columns[0][data]": 0,
        "columns[0][name]": "",
        "search[value]": "",
        "search[regex]": "false",
        "args[initialLetter]": "",
        "iDisplayLength": length,
        "iDisplayStart": start,
        "sSortDir_0": "asc",
        "sSortField": "name",
    }


def post_json(session: requests.Session, url: str, payload: dict[str, Any], cfg: RequestConfig) -> dict[str, Any]:
    headers = {
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Origin": "https://live.euronext.com",
        "Referer": "https://live.euronext.com/en/products/equities/list",
        "X-Requested-With": "XMLHttpRequest",
    }
    resp = session.post(url, data=payload, headers=headers, timeout=cfg.timeout)
    assert_not_challenge(resp, url)
    resp.raise_for_status()
    try:
        return resp.json()
    except Exception as exc:
        raise RuntimeError(f"Unexpected non-JSON response from {url}: {resp.text[:300]}") from exc


def parse_stock_row(row: list[Any], market_key: str) -> dict[str, Any]:
    """Parse one row from Euronext's legacy aaData array."""
    cfg = MARKET_GROUPS[market_key]

    link_cell = row[1] if len(row) > 1 else ""
    href = extract_href(link_cell)
    full_url = urljoin("https://live.euronext.com", href) if href else ""
    adn, mic_from_url = extract_adn_from_href(href)

    isin = extract_isin(row)
    if not isin and adn:
        isin = adn.split("-", 1)[0]

    raw_symbol = clean_text(row[3]) if len(row) > 3 else ""
    market_html = row[4] if len(row) > 4 else ""
    market_text = clean_text(market_html)

    mic = mic_from_url or next(iter(cfg["mics"]))
    exchange_suffix = MIC_TO_SUFFIX.get(mic, cfg["exchange_suffix"])
    market_label = MIC_TO_LABEL.get(mic, cfg["label"])

    name = extract_attr(link_cell, "data-order") or clean_text(link_cell)
    ticker = yahoo_ticker_from_euronext_symbol(raw_symbol, exchange_suffix)

    return {
        "ticker": ticker,
        "exchange_suffix": exchange_suffix,
        "name": name,
        "sector": "",
        "isin": isin,
        "euronext_symbol": raw_symbol,
        "mic": mic,
        "market_key": market_key,
        "market": market_label,
        "market_display": market_text,
        "euronext_adn": adn,
        "url": full_url,
        "source": "euronext_current_directory_json",
        "source_date": pd.Timestamp.utcnow().date().isoformat(),
    }


def _decode_response_bytes(content: bytes) -> str:
    """Decode Euronext CSV bytes robustly."""
    for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            return content.decode(enc)
        except UnicodeDecodeError:
            continue
    return content.decode("latin-1", errors="replace")


def _read_exported_csv(text: str) -> pd.DataFrame:
    """Read the Euronext CSV export despite delimiter changes."""
    import io

    # Remove empty preamble lines sometimes present in downloaded CSV files.
    text = "\n".join(line for line in text.splitlines() if line.strip())
    if not text.strip():
        return pd.DataFrame()

    attempts = [
        {"sep": None, "engine": "python"},
        {"sep": ";"},
        {"sep": ","},
        {"sep": "\t"},
    ]
    last_exc: Exception | None = None
    for kwargs in attempts:
        try:
            df = pd.read_csv(io.StringIO(text), **kwargs)
            if df.shape[1] >= 2:
                df.columns = [str(c).strip() for c in df.columns]
                return df
        except Exception as exc:  # pragma: no cover - depends on remote format
            last_exc = exc
    raise RuntimeError(f"Could not parse Euronext CSV export. Last error: {last_exc}")


def _norm_col(name: str) -> str:
    import unicodedata

    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def _first_present(row: pd.Series, colmap: dict[str, str], candidates: list[str]) -> str:
    for cand in candidates:
        col = colmap.get(cand)
        if col is None:
            continue
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip():
            return clean_text(val)
    return ""


def _infer_suffix_from_text(*values: Any) -> tuple[str, str, str]:
    """
    Infer (exchange_suffix, market_key, market_label) from any row text.

    The CSV export may expose only a market label, not the MIC. We only need the
    Yahoo suffix for the rest of this project, so location-level inference is
    sufficient.
    """
    import unicodedata

    raw = " ".join(clean_text(v) for v in values if v is not None)
    text = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii").lower()

    patterns = [
        ("amsterdam", ".AS", "amsterdam", "Euronext Amsterdam"),
        ("brussels", ".BR", "brussels", "Euronext Brussels"),
        ("bruxelles", ".BR", "brussels", "Euronext Brussels"),
        ("dublin", ".IR", "dublin", "Euronext Dublin"),
        ("lisbon", ".LS", "lisbon", "Euronext Lisbon"),
        ("lisboa", ".LS", "lisbon", "Euronext Lisbon"),
        ("milan", ".MI", "milan", "Euronext Milan"),
        ("milano", ".MI", "milan", "Euronext Milan"),
        ("eurotlx", ".MI", "milan", "Euronext Milan"),
        ("oslo", ".OL", "oslo", "Oslo Børs"),
        ("bors", ".OL", "oslo", "Oslo Børs"),
        ("paris", ".PA", "paris", "Euronext Paris"),
    ]
    for token, suffix, key, label in patterns:
        if token in text:
            return suffix, key, label

    # If the CSV contains a MIC, use it.
    for mic, suffix in MIC_TO_SUFFIX.items():
        if mic.lower() in text:
            label = MIC_TO_LABEL.get(mic, "")
            key = next((k for k, cfg in MARKET_GROUPS.items() if mic in cfg["mics"]), "")
            return suffix, key, label

    return "", "", ""


def _parse_download_row(row: pd.Series, colmap: dict[str, str]) -> dict[str, Any]:
    name = _first_present(row, colmap, ["name", "instrument_name", "product_name", "company_name"])
    isin = _first_present(row, colmap, ["isin", "isin_code"])
    symbol = _first_present(row, colmap, ["symbol", "ticker", "code", "trading_symbol"])
    mic = _first_present(row, colmap, ["mic", "mic_code", "market_identifier_code"])
    market_display = _first_present(row, colmap, ["market", "trading_location", "location", "exchange", "market_name"])
    sector = _first_present(row, colmap, ["sector", "industry", "supersector", "subsector"])

    # Fallback: sometimes ISIN is embedded in a link or unnamed column.
    if not isin:
        isin = extract_isin([row.get(c, "") for c in row.index])

    href = ""
    for c in row.index:
        maybe_href = extract_href(row.get(c, ""))
        if maybe_href:
            href = maybe_href
            break
    full_url = urljoin("https://live.euronext.com", href) if href else ""
    adn, mic_from_url = extract_adn_from_href(href)
    mic = (mic_from_url or mic or "").upper()

    suffix = MIC_TO_SUFFIX.get(mic, "") if mic else ""
    market_key = next((k for k, cfg in MARKET_GROUPS.items() if mic in cfg["mics"]), "") if mic else ""
    market_label = MIC_TO_LABEL.get(mic, "") if mic else ""

    if not suffix:
        suffix, market_key, market_label = _infer_suffix_from_text(market_display, mic, " ".join(map(str, row.values)))

    ticker = yahoo_ticker_from_euronext_symbol(symbol, suffix) if suffix else ""

    return {
        "ticker": ticker,
        "exchange_suffix": suffix,
        "name": name,
        "sector": sector,
        "isin": isin,
        "euronext_symbol": symbol,
        "mic": mic or (next(iter(MARKET_GROUPS[market_key]["mics"])) if market_key else ""),
        "market_key": market_key,
        "market": market_label,
        "market_display": market_display,
        "euronext_adn": adn,
        "url": full_url,
        "source": "euronext_current_directory_csv_export",
        "source_date": pd.Timestamp.utcnow().date().isoformat(),
    }


def scrape_current_euronext_universe_csv_export(
    session: Optional[requests.Session] = None,
    cfg: RequestConfig = RequestConfig(),
) -> pd.DataFrame:
    """Download the current Euronext stock directory through the public CSV export."""
    session = session or make_session(cfg)

    seed_url = "https://live.euronext.com/en/products/equities/list"
    seed = session.get(seed_url, timeout=cfg.timeout)
    assert_not_challenge(seed, seed_url)
    time.sleep(cfg.min_delay)

    headers = {
        "Accept": "text/csv,application/csv,application/octet-stream,*/*;q=0.8",
        "Referer": seed_url,
        "X-Requested-With": "XMLHttpRequest",
    }
    log.info("Downloading Euronext current directory CSV export")
    resp = session.get(EURONEXT_STOCK_DOWNLOAD_URL, headers=headers, timeout=cfg.timeout)
    assert_not_challenge(resp, EURONEXT_STOCK_DOWNLOAD_URL)
    resp.raise_for_status()

    text = _decode_response_bytes(resp.content)
    if "<html" in text[:500].lower() and "name isin symbol market" not in text.lower():
        raise RuntimeError(
            "Euronext CSV export returned HTML instead of CSV. "
            f"First 300 chars: {text[:300]!r}"
        )

    raw = _read_exported_csv(text)
    if raw.empty:
        return pd.DataFrame()

    colmap = {_norm_col(c): c for c in raw.columns}
    rows = []
    for _, row in raw.iterrows():
        parsed = _parse_download_row(row, colmap)
        if parsed["ticker"] and parsed["isin"] and parsed["exchange_suffix"]:
            rows.append(parsed)

    df = pd.DataFrame(rows)
    if df.empty:
        # Keep a useful debug file in memory through the error message.
        raise RuntimeError(
            "CSV export was downloaded but no rows could be parsed. "
            f"Columns seen: {list(raw.columns)}. First rows: {raw.head(3).to_dict(orient='records')}"
        )

    return (
        df.sort_values(["ticker", "isin", "mic"])
        .drop_duplicates(subset=["ticker", "isin", "exchange_suffix"])
        .reset_index(drop=True)
    )


def scrape_current_euronext_universe_json_fallback(
    session: Optional[requests.Session] = None,
    cfg: RequestConfig = RequestConfig(),
    page_size: int = 100,
) -> pd.DataFrame:
    """Legacy JSON/DataTables fallback. Kept for backwards compatibility."""
    session = session or make_session(cfg)

    seed_url = "https://live.euronext.com/en/products/equities/list"
    seed = session.get(seed_url, timeout=cfg.timeout)
    assert_not_challenge(seed, seed_url)
    time.sleep(cfg.min_delay)

    all_rows: list[dict[str, Any]] = []
    totals_seen: dict[str, int] = {}
    for market_key, market_cfg in MARKET_GROUPS.items():
        url = euronext_stock_endpoint(market_cfg["mics"])
        log.info("Scraping Euronext current directory JSON fallback: %s", market_cfg["label"])

        first = post_json(session, url, datatable_payload(0, page_size), cfg)
        total = int(first.get("iTotalDisplayRecords") or first.get("recordsTotal") or 0)
        totals_seen[market_key] = total
        starts = list(range(0, max(total, 1), page_size))

        for i, start in enumerate(starts):
            data = first if i == 0 else post_json(session, url, datatable_payload(start, page_size), cfg)
            aa_data = data.get("aaData") or data.get("data") or []
            for row in aa_data:
                if not isinstance(row, list):
                    continue
                parsed = parse_stock_row(row, market_key)
                if parsed["ticker"] and parsed["isin"]:
                    all_rows.append(parsed)
            time.sleep(cfg.min_delay)

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise RuntimeError(
            "No instruments scraped from Euronext current directory JSON fallback. "
            f"Record totals reported by endpoint: {totals_seen}"
        )

    return (
        df.sort_values(["ticker", "isin", "mic"])
        .drop_duplicates(subset=["ticker", "isin", "mic"])
        .reset_index(drop=True)
    )


def scrape_current_euronext_universe(
    session: Optional[requests.Session] = None,
    cfg: RequestConfig = RequestConfig(),
    page_size: int = 100,
) -> pd.DataFrame:
    """
    Scrape/download the current Euronext stock directory for all markets.

    Preferred path: CSV export URL. Fallback: legacy JSON DataTables endpoint.
    """
    session = session or make_session(cfg)
    try:
        df = scrape_current_euronext_universe_csv_export(session=session, cfg=cfg)
        if not df.empty:
            log.info("Downloaded %s instruments from Euronext CSV export", len(df))
            return df
    except Exception as exc:
        log.warning("CSV export failed, trying JSON fallback. Reason: %s", exc)

    return scrape_current_euronext_universe_json_fallback(session=session, cfg=cfg, page_size=page_size)


# ---------------------------------------------------------------------------
# Optional: delisting notices
# ---------------------------------------------------------------------------


def parse_cash_notice_page(html_text: str) -> list[dict[str, Any]]:
    """Best-effort parser for Euronext cash notice list pages."""
    soup = BeautifulSoup(html_text, "html.parser")
    text = soup.get_text("\n", strip=True)
    notice_ids = list(NOTICE_ID_RE.finditer(text))
    notices: list[dict[str, Any]] = []

    for idx, match in enumerate(notice_ids):
        start = match.start()
        end = notice_ids[idx + 1].start() if idx + 1 < len(notice_ids) else len(text)
        chunk = text[start:end]
        chunk_norm = re.sub(r"\s+", " ", chunk).strip()
        if "delisting" not in chunk_norm.lower():
            continue
        if "shares" not in chunk_norm.lower() and "equities" not in chunk_norm.lower():
            # Avoid structured products, certificates, warrants etc. unless the page text says shares/equities.
            continue

        dates = re.findall(r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b", chunk_norm)
        isins = sorted(set(ISIN_RE.findall(chunk_norm)))
        notices.append(
            {
                "notice_id": match.group(0),
                "publication_date": dates[0] if dates else "",
                "effective_date": dates[1] if len(dates) > 1 else "",
                "isin_candidates": ";".join(isins),
                "notice_text": chunk_norm[:1000],
                "source": "euronext_cash_notices",
            }
        )
    return notices


def scrape_delisting_notices(
    session: Optional[requests.Session] = None,
    cfg: RequestConfig = RequestConfig(),
    start: str = "2019-01-01",
    end: str = "2025-12-31",
    max_pages: int = 1500,
    page_size: int = 50,
) -> pd.DataFrame:
    """Scrape cash notices and keep share/equity delisting notices in the target window."""
    session = session or make_session(cfg)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    rows: list[dict[str, Any]] = []
    base = "https://live.euronext.com/en/resources/notices-corporate-actions/cash-notices"
    log.info("Scraping Euronext cash notices for delisting keywords; max_pages=%s", max_pages)

    for page in tqdm(range(max_pages), desc="Cash notices"):
        url = f"{base}?alias=1&page={page}&pageSize={page_size}"
        resp = session.get(url, timeout=cfg.timeout)
        assert_not_challenge(resp, url)
        if resp.status_code >= 400:
            break
        parsed = parse_cash_notice_page(resp.text)
        if parsed:
            rows.extend(parsed)

        # Stop if the page appears to contain dates older than the requested start window.
        all_dates = re.findall(r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b", resp.text)
        parsed_dates = pd.to_datetime(all_dates, dayfirst=True, errors="coerce")
        parsed_dates = parsed_dates[~pd.isna(parsed_dates)]
        if len(parsed_dates) and parsed_dates.min() < start_ts - pd.Timedelta(days=30):
            # Pages are typically reverse chronological. Keep a little buffer.
            break
        time.sleep(cfg.min_delay)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for col in ["publication_date", "effective_date"]:
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
    mask = df["publication_date"].between(start_ts, end_ts) | df["effective_date"].between(start_ts, end_ts)
    df = df[mask].drop_duplicates(subset=["notice_id"]).reset_index(drop=True)
    return df


def yahoo_search(session: requests.Session, query: str, cfg: RequestConfig) -> list[dict[str, Any]]:
    """Unofficial Yahoo search endpoint used only for ISIN/name resolution fallback."""
    if not query:
        return []
    url = "https://query1.finance.yahoo.com/v1/finance/search"
    params = {"q": query, "quotesCount": 10, "newsCount": 0}
    headers = {"Accept": "application/json, text/plain, */*", "User-Agent": session.headers.get("User-Agent", "")}
    resp = session.get(url, params=params, headers=headers, timeout=cfg.timeout)
    if resp.status_code >= 400:
        return []
    try:
        data = resp.json()
    except Exception:
        return []
    return data.get("quotes") or []


def candidates_from_notices(
    notices: pd.DataFrame,
    session: Optional[requests.Session] = None,
    cfg: RequestConfig = RequestConfig(),
) -> pd.DataFrame:
    """Try to resolve delisted notice ISINs/names to Yahoo tickers."""
    if notices.empty:
        return pd.DataFrame()
    session = session or make_session(cfg)
    rows: list[dict[str, Any]] = []

    for _, notice in tqdm(notices.iterrows(), total=len(notices), desc="Resolve notices"):
        queries: list[str] = []
        for isin in str(notice.get("isin_candidates", "")).split(";"):
            if ISIN_RE.fullmatch(isin or ""):
                queries.append(isin)
        # Fallback on notice text, but keep it short.
        if not queries:
            text = str(notice.get("notice_text", ""))
            queries.append(text[:120])

        for query in queries:
            for q in yahoo_search(session, query, cfg):
                symbol = str(q.get("symbol", "")).upper().strip()
                if not symbol or not any(symbol.endswith(suf) for suf in KNOWN_SUFFIXES):
                    continue
                suffix = next(suf for suf in KNOWN_SUFFIXES if symbol.endswith(suf))
                rows.append(
                    {
                        "ticker": symbol,
                        "exchange_suffix": suffix,
                        "name": q.get("shortname") or q.get("longname") or "",
                        "sector": "",
                        "isin": query if ISIN_RE.fullmatch(query) else "",
                        "euronext_symbol": symbol[: -len(suffix)],
                        "mic": "",
                        "market_key": "",
                        "market": "",
                        "market_display": "",
                        "euronext_adn": "",
                        "url": "",
                        "source": "euronext_delisting_notice_yahoo_resolved",
                        "source_date": pd.Timestamp.utcnow().date().isoformat(),
                        "notice_id": notice.get("notice_id", ""),
                    }
                )
            time.sleep(cfg.min_delay)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates(subset=["ticker"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Yahoo historical dates -> dated universe
# ---------------------------------------------------------------------------


def fetch_yahoo_dates(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily Yahoo history for one ticker and return available dates."""
    # yfinance end is exclusive, so add one day.
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.download(
        ticker,
        start=start,
        end=end_exclusive,
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame(columns=["date"])

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    date_col = "Date" if "Date" in df.columns else "date"
    close_col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
    if close_col is None:
        return pd.DataFrame(columns=["date"])
    out = df[[date_col, close_col]].rename(columns={date_col: "date", close_col: "close"})
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out = out.dropna(subset=["close"]).drop_duplicates(subset=["date"])
    return out[["date"]]


def build_timeseries_universe(
    candidates: pd.DataFrame,
    start: str,
    end: str,
    out_path: Path,
    max_workers: int = 4,
    min_observations: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build and stream the dated ticker universe to CSV.

    Returns
    -------
    summary_df, load_user_df
    """
    metadata_cols = [
        "ticker",
        "exchange_suffix",
        "name",
        "sector",
        "isin",
        "euronext_symbol",
        "mic",
        "market",
        "source",
    ]
    for col in metadata_cols:
        if col not in candidates.columns:
            candidates[col] = ""

    candidates = (
        candidates[candidates["ticker"].astype(str).str.len() > 0]
        .sort_values(["ticker", "source"])
        .drop_duplicates(subset=["ticker"], keep="first")
        .reset_index(drop=True)
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    fieldnames = ["date"] + metadata_cols
    summary_rows: list[dict[str, Any]] = []
    load_rows: list[dict[str, Any]] = []

    def job(row_dict: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
        ticker = row_dict["ticker"]
        try:
            dates = fetch_yahoo_dates(ticker, start, end)
        except Exception as exc:
            log.warning("Yahoo failed for %s: %s", ticker, exc)
            dates = pd.DataFrame(columns=["date"])
        return row_dict, dates

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(job, row._asdict()) for row in candidates[metadata_cols].itertuples(index=False)]
            for fut in tqdm(cf.as_completed(futures), total=len(futures), desc="Yahoo history"):
                meta, dates = fut.result()
                ticker = meta["ticker"]
                n_obs = len(dates)
                if n_obs < min_observations:
                    summary_rows.append(
                        {
                            **meta,
                            "n_days": n_obs,
                            "first_trade_date": "",
                            "last_trade_date": "",
                            "alive_at_period_end": False,
                            "inferred_delisted_before_period_end": False,
                            "status": "no_or_insufficient_yahoo_history",
                        }
                    )
                    continue

                first_trade = pd.to_datetime(dates["date"]).min().date()
                last_trade = pd.to_datetime(dates["date"]).max().date()
                alive_at_end_threshold = (pd.Timestamp(end) - pd.Timedelta(days=21)).date()
                alive_at_period_end = last_trade >= alive_at_end_threshold
                inferred_delisted = not alive_at_period_end

                for d in dates["date"]:
                    writer.writerow({"date": d, **meta})

                summary_rows.append(
                    {
                        **meta,
                        "n_days": n_obs,
                        "first_trade_date": first_trade,
                        "last_trade_date": last_trade,
                        "alive_at_period_end": alive_at_period_end,
                        "inferred_delisted_before_period_end": inferred_delisted,
                        "status": "ok",
                    }
                )
                load_rows.append(
                    {
                        "ticker": ticker,
                        "exchange_suffix": meta.get("exchange_suffix", ""),
                        "name": meta.get("name", ""),
                        "sector": meta.get("sector", ""),
                    }
                )

    summary_df = pd.DataFrame(summary_rows).sort_values(["status", "ticker"]).reset_index(drop=True)
    load_user_df = pd.DataFrame(load_rows).drop_duplicates(subset=["ticker"]).sort_values("ticker").reset_index(drop=True)
    return summary_df, load_user_df


# ---------------------------------------------------------------------------
# Historical loader helper for your backtest / universe.py workflow
# ---------------------------------------------------------------------------


def load_universe_as_of(timeseries_csv: str | Path, as_of: str) -> pd.DataFrame:
    """
    Load the ticker universe that had a Yahoo/Euronext trading observation on or before `as_of`.

    This is useful because the provided load_user_universe(csv_path) function drops duplicates
    by ticker and therefore cannot preserve the date dimension by itself.
    """
    df = pd.read_csv(timeseries_csv, parse_dates=["date"])
    as_of_ts = pd.Timestamp(as_of)
    df = df[df["date"] <= as_of_ts].copy()
    if df.empty:
        return pd.DataFrame(columns=["ticker", "exchange_suffix", "name", "sector"])

    # Keep tickers that have traded recently enough before the as_of date.
    # This avoids retaining securities that stopped trading years before as_of.
    recent_threshold = as_of_ts - pd.Timedelta(days=31)
    last_seen = df.groupby("ticker", as_index=False)["date"].max()
    alive = last_seen[last_seen["date"] >= recent_threshold]["ticker"]
    out = df[df["ticker"].isin(alive)].sort_values("date").drop_duplicates("ticker", keep="last")
    keep_cols = [c for c in ["ticker", "exchange_suffix", "name", "sector"] if c in out.columns]
    return out[keep_cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(args: argparse.Namespace) -> None:
    cfg = RequestConfig(timeout=args.timeout, min_delay=args.min_delay, max_retries=args.max_retries)
    session = make_session(cfg)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    current_path = out_dir / "euronext_current_universe.csv"
    notices_path = out_dir / "euronext_delisting_notices.csv"
    candidates_path = out_dir / "euronext_candidate_universe.csv"
    timeseries_path = out_dir / "euronext_universe_timeseries.csv"
    load_path = out_dir / "euronext_universe_for_load_user_universe.csv"
    summary_path = out_dir / "euronext_universe_summary.csv"

    if args.current_csv and Path(args.current_csv).exists():
        current = pd.read_csv(args.current_csv)
        log.info("Loaded current universe from %s", args.current_csv)
    elif current_path.exists() and not args.force:
        current = pd.read_csv(current_path)
        log.info("Loaded cached current universe from %s", current_path)
    else:
        current = scrape_current_euronext_universe(session=session, cfg=cfg, page_size=args.page_size)
        current.to_csv(current_path, index=False)
        log.info("Saved %s rows to %s", len(current), current_path)

    candidate_frames = [current]

    if args.include_notices:
        if notices_path.exists() and not args.force:
            notices = pd.read_csv(notices_path)
            log.info("Loaded cached notices from %s", notices_path)
        else:
            notices = scrape_delisting_notices(
                session=session,
                cfg=cfg,
                start=args.start,
                end=args.end,
                max_pages=args.max_notice_pages,
                page_size=args.notice_page_size,
            )
            notices.to_csv(notices_path, index=False)
            log.info("Saved %s delisting notices to %s", len(notices), notices_path)

        resolved = candidates_from_notices(notices, session=session, cfg=cfg) if not notices.empty else pd.DataFrame()
        if not resolved.empty:
            candidate_frames.append(resolved)
            log.info("Resolved %s notice-based Yahoo ticker candidates", len(resolved))

    candidates = pd.concat(candidate_frames, ignore_index=True, sort=False)
    candidates = candidates.drop_duplicates(subset=["ticker"], keep="first").sort_values("ticker").reset_index(drop=True)
    candidates.to_csv(candidates_path, index=False)
    log.info("Candidate universe: %s tickers saved to %s", len(candidates), candidates_path)

    summary, load_user = build_timeseries_universe(
        candidates=candidates,
        start=args.start,
        end=args.end,
        out_path=timeseries_path,
        max_workers=args.max_workers,
        min_observations=args.min_observations,
    )
    summary.to_csv(summary_path, index=False)
    load_user.to_csv(load_path, index=False)

    log.info("Saved dated universe to %s", timeseries_path)
    log.info("Saved load_user_universe-compatible file to %s", load_path)
    log.info("Saved summary to %s", summary_path)
    if not summary.empty:
        ok = int((summary["status"] == "ok").sum())
        delisted = int(summary.get("inferred_delisted_before_period_end", pd.Series(dtype=bool)).fillna(False).sum())
        log.info("Summary: %s tickers with history; %s inferred delisted before %s", ok, delisted, args.end)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a dated Euronext ticker universe from public sources.")
    parser.add_argument("--start", default="2019-01-01", help="Start date, YYYY-MM-DD")
    parser.add_argument("--end", default="2025-12-31", help="End date, YYYY-MM-DD")
    parser.add_argument("--out-dir", default="data/euronext_universe", help="Output directory")
    parser.add_argument("--include-notices", action="store_true", help="Also scrape Euronext cash delisting notices")
    parser.add_argument("--current-csv", default="", help="Optional pre-scraped current universe CSV")
    parser.add_argument("--force", action="store_true", help="Ignore cached CSVs and scrape again")
    parser.add_argument("--page-size", type=int, default=100, help="Euronext DataTables page size")
    parser.add_argument("--notice-page-size", type=int, default=50, help="Cash notice page size")
    parser.add_argument("--max-notice-pages", type=int, default=1500, help="Max cash notice pages to crawl")
    parser.add_argument("--max-workers", type=int, default=4, help="Concurrent Yahoo downloads")
    parser.add_argument("--min-observations", type=int, default=1, help="Minimum Yahoo observations to keep a ticker")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds")
    parser.add_argument("--min-delay", type=float, default=0.25, help="Minimum delay between Euronext/Yahoo helper requests")
    parser.add_argument("--max-retries", type=int, default=5, help="HTTP retries")
    return parser


if __name__ == "__main__":
    run_pipeline(build_arg_parser().parse_args())
