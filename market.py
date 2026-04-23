import requests
import time
import json
import pandas as pd
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

GAMMA = "https://gamma-api.polymarket.com"
CLOB  = "https://clob.polymarket.com"

# ─── Étape 1 : Récupération des marchés par slug (parallélisée) ───────────────
#
# L'API Gamma n'expose pas les marchés btc-updown-5m via la pagination classique.
# La seule méthode fiable est de requêter chaque slug exact : btc-updown-5m-{ts}
# On parallélise avec ThreadPoolExecutor pour compenser le volume (~26 000 slugs).

def _fetch_slug(slug):
    """Requête un slug et retourne les marchés trouvés (liste vide sinon)."""
    try:
        resp = requests.get(
            f"{GAMMA}/events",
            params={"slug": slug},
            timeout=10,
        )
        if resp.status_code != 200 or not resp.json():
            return []

        event = resp.json()[0]
        if not event.get("endDate"):
            return []

        results = []
        for market in event.get("markets", []):
            try:
                outcomes = json.loads(market.get("outcomes", "[]"))
                prices   = json.loads(market.get("outcomePrices", "[]"))
                tokens   = json.loads(market.get("clobTokenIds", "[]"))
            except json.JSONDecodeError:
                continue

            results.append({
                "slug":         slug,
                "title":        event.get("title", ""),
                "start_ts":     event.get("startDate"),
                "end_ts":       event.get("endDate"),
                "outcomes":     outcomes,
                "final_prices": prices,
                "token_ids":    tokens,
                "winner":       market.get("winner", ""),
            })
        return results

    except requests.RequestException:
        return []


def fetch_all_btc_5min_markets(months=3, max_workers=20):
    """
    Génère tous les slugs btc-updown-5m sur la période et les requête en parallèle.
    ~26 000 slugs / 20 workers ≈ 3-5 minutes au lieu de 25 min en séquentiel.
    """
    now_ts   = int(datetime.now(tz=timezone.utc).timestamp())
    # Arrondi au slot de 5 min le plus proche
    now_ts   = (now_ts // 300) * 300
    start_ts = now_ts - months * 30 * 24 * 3600

    slugs = [f"btc-updown-5m-{ts}" for ts in range(start_ts, now_ts, 300)]
    total = len(slugs)

    logger.info(f"Recherche de {total} slugs | {months} mois | {max_workers} workers")
    logger.info(f"Période : {datetime.fromtimestamp(start_ts, tz=timezone.utc)} → {datetime.fromtimestamp(now_ts, tz=timezone.utc)}")

    markets  = []
    done     = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_slug, slug): slug for slug in slugs}

        for future in as_completed(futures):
            result = future.result()
            if result:
                markets.extend(result)

            done += 1
            if done % 500 == 0 or done == total:
                logger.info(f"Progression : {done}/{total} slugs | {len(markets)} marchés trouvés")\

    logger.info(f"Récupération terminée : {len(markets)} marchés BTC 5-min")
    return markets


# ─── Étape 2 : Historique de prix ─────────────────────────────────────────────

def fetch_price_history(token_id, fidelity=1):
    """
    Récupère l'évolution des cotes pour un token donné.
    Retourne [] en cas d'erreur (sans lever d'exception).
    """
    if not token_id:
        return []
    try:
        resp = requests.get(
            f"{CLOB}/prices-history",
            params={"market": token_id, "interval": "max", "fidelity": fidelity},
            timeout=15,
        )
        if resp.status_code != 200:
            return []
        return resp.json().get("history", [])
    except requests.RequestException:
        return []


# ─── Étape 3 : Construction du dataset de backtest ────────────────────────────

def _process_single_market(m):
    """Traite un marché et retourne la ligne de résultat (ou None si invalide)."""
    tokens   = m.get("token_ids", [])
    outcomes = m.get("outcomes", [])
    winner   = m.get("winner", "")

    if len(tokens) < 2 or len(outcomes) < 2:
        return None

    up_idx   = next((i for i, o in enumerate(outcomes) if "up" in str(o).lower()), 0)
    down_idx = 1 - up_idx

    up_token   = tokens[up_idx]   if up_idx < len(tokens) else None
    down_token = tokens[down_idx] if down_idx < len(tokens) else None

    up_history = fetch_price_history(up_token)

    entry_up_price = up_history[0]["p"] if up_history else None

    if winner and up_token and winner.lower() == up_token.lower():
        result = "UP"
    elif winner and down_token and winner.lower() == down_token.lower():
        result = "DOWN"
    else:
        result = "UNKNOWN"

    return {
        "slug":           m["slug"],
        "start_ts":       m["start_ts"],
        "end_ts":         m["end_ts"],
        "entry_up_price": entry_up_price,
        "result":         result,
        "up_token_id":    up_token,
        "price_history":  up_history,
    }


def build_backtest_dataset(markets, max_workers=10):
    """
    Assemble le dataset final.
    Les requêtes CLOB sont parallélisées avec ThreadPoolExecutor.
    """
    rows  = []
    total = len(markets)
    logger.info(f"build_backtest_dataset | {total} marchés | {max_workers} workers")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_single_market, m): i for i, m in enumerate(markets)}

        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                rows.append(result)

            if done % 100 == 0 or done == total:
                logger.info(f"Progression : {done}/{total} | {len(rows)} lignes valides")

    logger.info(f"build_backtest_dataset terminé : {len(rows)} lignes")
    return pd.DataFrame(rows)


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    markets = fetch_all_btc_5min_markets(months=3)
    logger.info(f"Trouvé {len(markets)} marchés BTC 5-min")

    if len(markets) == 0:
        logger.error("Aucun marché trouvé — vérifiez l'API ou les filtres")
    else:
        df = build_backtest_dataset(markets, max_workers=10)

        if df.empty:
            logger.error("DataFrame vide après build_backtest_dataset")
        else:
            output_path = "btc_5min_polymarket_backtest.parquet"
            df.to_parquet(output_path, index=False)
            logger.info(f"Parquet exporté : {output_path} ({len(df)} lignes)")
            print(df[["slug", "start_ts", "entry_up_price", "result"]].head(20))