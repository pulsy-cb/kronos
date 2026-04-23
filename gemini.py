import requests
import json
import time
import numpy as np
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _iso_to_unix(iso_str):
    if not iso_str:
        return None
    try:
        iso_str = iso_str.replace(" ", "T").split("+")[0].rstrip("Z") + "Z"
        return int(datetime.fromisoformat(iso_str.replace("Z", "+00:00")).timestamp())
    except Exception:
        return None


def _get_session(dt):
    if dt is None:
        return None
    h = dt.hour
    if 0 <= h < 8:  return "asia"
    if 8 <= h < 16: return "london"
    return "ny"


def _extract_fee_schedule(market):
    schedule = market.get("feeSchedule")
    return {
        "fees_enabled":       market.get("feesEnabled"),
        "fee_type":           market.get("feeType"),
        "taker_fee_rate":     float(schedule["rate"])       if isinstance(schedule, dict) and "rate"       in schedule else None,
        "maker_rebate_rate":  float(schedule["rebateRate"]) if isinstance(schedule, dict) and "rebateRate" in schedule else None,
        "fee_exponent":       schedule.get("exponent")      if isinstance(schedule, dict) else None,
        "fee_taker_only":     schedule.get("takerOnly")     if isinstance(schedule, dict) else None,
        "maker_base_fee_bps": market.get("makerBaseFee"),
        "taker_base_fee_bps": market.get("takerBaseFee"),
    }


# ─── Diagnostic CLOB ──────────────────────────────────────────────────────────

def diagnose_clob(token_id):
    """
    Test rapide pour comprendre pourquoi le CLOB ne répond pas.
    À appeler une fois au démarrage.
    """
    logger.info(f"[DIAG] Test CLOB avec token : {token_id[:20]}...")
    try:
        resp = requests.get(
            f"{CLOB}/prices-history",
            params={"market": token_id, "interval": "max", "fidelity": 1},
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        logger.info(f"[DIAG] Status        : {resp.status_code}")
        logger.info(f"[DIAG] Headers       : {dict(resp.headers)}")
        logger.info(f"[DIAG] Body (200 car): {resp.text[:200]}")

        if resp.status_code == 200:
            data = resp.json()
            history = data.get("history", [])
            logger.info(f"[DIAG] Ticks retournés : {len(history)}")
            if history:
                logger.info(f"[DIAG] Premier tick : {history[0]}")
                logger.info(f"[DIAG] Dernier tick : {history[-1]}")
            else:
                logger.warning("[DIAG] history[] vide malgré status 200 → token sans activité ou géoblocage soft")
        elif resp.status_code == 403:
            logger.error("[DIAG] 403 Forbidden → géoblocage IP ou token requis")
        elif resp.status_code == 429:
            logger.error("[DIAG] 429 Too Many Requests → rate limit")
        else:
            logger.error(f"[DIAG] Erreur inattendue : {resp.status_code}")

    except requests.exceptions.ConnectionError as e:
        logger.error(f"[DIAG] ConnectionError → serveur injoignable depuis cette IP : {e}")
    except requests.exceptions.Timeout:
        logger.error("[DIAG] Timeout → CLOB trop lent ou bloqué")
    except Exception as e:
        logger.error(f"[DIAG] Exception inattendue : {e}")


# ─── Étape 1 : Fetch slugs ────────────────────────────────────────────────────

def _fetch_slug(slug):
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

            fee_info = _extract_fee_schedule(market)

            results.append({
                "slug":               slug,
                "condition_id":       market.get("conditionId", ""),
                "question_id":        market.get("questionID", ""),
                "event_start_ts":     _iso_to_unix(market.get("eventStartTime")),
                "event_end_ts":       _iso_to_unix(market.get("endDate")),
                "closed_ts":          _iso_to_unix(market.get("closedTime")),
                "uma_end_ts":         _iso_to_unix(market.get("umaEndDate")),
                "outcomes":           outcomes,
                "final_prices":       prices,
                "token_ids":          tokens,
                "winner":             market.get("winner", ""),
                "volume_usdc":        float(market.get("volumeClob") or market.get("volume") or 0),
                "liquidity_usdc":     float(market.get("liquidityClob") or market.get("liquidity") or 0),
                "spread_raw":         float(market.get("spread") or 0),
                "best_bid":           float(market.get("bestBid") or 0),
                "best_ask":           float(market.get("bestAsk") or 0),
                "last_trade_price":   float(market.get("lastTradePrice") or 0),
                "order_book_enabled": market.get("enableOrderBook", False),
                "order_min_size":     market.get("orderMinSize"),
                "order_tick_size":    market.get("orderPriceMinTickSize"),
                **fee_info,
            })
        return results

    except requests.RequestException:
        return []


def fetch_all_btc_5min_markets(months=3, max_workers=20):
    now_ts   = (int(datetime.now(tz=timezone.utc).timestamp()) // 300) * 300
    start_ts = now_ts - months * 30 * 24 * 3600
    slugs    = [f"btc-updown-5m-{ts}" for ts in range(start_ts, now_ts, 300)]
    total    = len(slugs)

    logger.info(f"Recherche de {total} slugs | {months} mois | {max_workers} workers")

    markets, done = [], 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_slug, slug): slug for slug in slugs}
        for future in as_completed(futures):
            result = future.result()
            if result:
                markets.extend(result)
            done += 1
            if done % 500 == 0 or done == total:
                logger.info(f"Progression : {done}/{total} slugs | {len(markets)} marchés bruts")

    logger.info(f"Récupération terminée : {len(markets)} marchés BTC 5-min bruts")
    return markets


# ─── Étape 2 : Historique de prix avec retry ──────────────────────────────────

def fetch_price_history(token_id, fidelity=1, max_retries=3, base_delay=2.0):
    """
    Récupère l'historique avec retry exponentiel.
    Logue le status exact pour diagnostiquer les blocages serveur.
    """
    if not token_id:
        return []

    headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}

    for attempt in range(max_retries):
        try:
            resp = requests.get(
                f"{CLOB}/prices-history",
                params={"market": token_id, "interval": "max", "fidelity": fidelity},
                timeout=15,
                headers=headers,
            )

            if resp.status_code == 200:
                history = resp.json().get("history", [])
                return history  # peut être [] si marché sans activité

            elif resp.status_code == 429:
                # Rate limit → attendre plus longtemps
                wait = base_delay * (3 ** attempt)
                logger.warning(f"[CLOB] 429 rate limit sur {token_id[:16]}… retry dans {wait:.1f}s")
                time.sleep(wait)

            elif resp.status_code == 403:
                logger.error(f"[CLOB] 403 Forbidden → géoblocage probable (token: {token_id[:16]}…)")
                return []  # Inutile de retry

            else:
                logger.warning(f"[CLOB] Status {resp.status_code} pour {token_id[:16]}… (attempt {attempt+1})")
                time.sleep(base_delay * (2 ** attempt))

        except requests.exceptions.Timeout:
            logger.warning(f"[CLOB] Timeout {token_id[:16]}… (attempt {attempt+1}/{max_retries})")
            time.sleep(base_delay * (2 ** attempt))

        except requests.exceptions.ConnectionError as e:
            logger.error(f"[CLOB] ConnectionError {token_id[:16]}… : {e}")
            return []  # Connexion impossible, inutile de retry

        except Exception as e:
            logger.warning(f"[CLOB] Exception {token_id[:16]}… : {e}")
            time.sleep(base_delay)

    logger.warning(f"[CLOB] Abandonné après {max_retries} tentatives : {token_id[:16]}…")
    return []


# ─── Étape 3 : Features prix ──────────────────────────────────────────────────

def extract_price_features(history, event_start_ts, event_end_ts):
    if not history:
        return None

    history = sorted(history, key=lambda x: x["t"])
    prices  = [h["p"] for h in history]
    times   = [h["t"] for h in history]

    price_open_up = None
    if event_start_ts:
        for h in history:
            if h["t"] >= event_start_ts:
                price_open_up = h["p"]
                break
    price_open_up = price_open_up if price_open_up is not None else prices[0]

    price_5min_before_up = None
    if event_start_ts:
        candidates = [h for h in history if h["t"] <= event_start_ts - 300]
        if candidates:
            price_5min_before_up = candidates[-1]["p"]

    price_close_up = prices[-1]
    if event_end_ts:
        candidates = [h for h in history if h["t"] <= event_end_ts]
        if candidates:
            price_close_up = candidates[-1]["p"]

    n_ticks      = len(prices)
    price_min_up = min(prices)
    price_max_up = max(prices)
    diffs        = [abs(prices[i+1] - prices[i]) for i in range(n_ticks - 1)]
    volatility   = float(np.std(diffs)) if diffs else 0.0
    trading_duration_s = times[-1] - times[0] if n_ticks > 1 else 0

    return {
        "price_open_up":           price_open_up,
        "price_close_up":          price_close_up,
        "price_5min_before_up":    price_5min_before_up,
        "price_min_up":            price_min_up,
        "price_max_up":            price_max_up,
        "price_open_down":         round(1 - price_open_up, 4),
        "price_close_down":        round(1 - price_close_up, 4),
        "price_5min_before_down":  round(1 - price_5min_before_up, 4) if price_5min_before_up is not None else None,
        "price_min_down":          round(1 - price_max_up, 4),
        "price_max_down":          round(1 - price_min_up, 4),
        "price_range":             round(price_max_up - price_min_up, 4),
        "sentiment_shift":         round(price_close_up - price_open_up, 4),
        "volatility":              round(volatility, 6),
        "n_ticks":                 n_ticks,
        "trading_duration_s":      trading_duration_s,
        "polymarket_bias":         round(price_open_up - 0.5, 4),
        "full_history":            json.dumps(history),
    }


# ─── Étape 4 : Build dataset ───────────────────────────────────────────────────

def _process_single_market(m):
    tokens   = m.get("token_ids", [])
    outcomes = m.get("outcomes", [])

    if len(tokens) < 2 or len(outcomes) < 2:
        return None, "invalid_structure"

    up_idx     = next((i for i, o in enumerate(outcomes) if "up" in str(o).lower()), 0)
    down_idx   = 1 - up_idx
    up_token   = tokens[up_idx]   if up_idx   < len(tokens) else None
    down_token = tokens[down_idx] if down_idx < len(tokens) else None

    up_history = fetch_price_history(up_token)

    pf = extract_price_features(up_history, m.get("event_start_ts"), m.get("event_end_ts"))
    if pf is None:
        return None, "no_history"

    final_prices            = m.get("final_prices", [])
    result_up_final_price   = None
    result_down_final_price = None

    if final_prices and len(final_prices) > up_idx:
        try:
            result_up_final_price   = float(final_prices[up_idx])
            result_down_final_price = float(final_prices[down_idx]) if len(final_prices) > down_idx else None
            result = "UP" if result_up_final_price >= 0.99 else "DOWN"
        except (ValueError, TypeError):
            result = "UNKNOWN"
    else:
        result = "UNKNOWN"

    slot_ts = m.get("event_start_ts")
    slot_dt = datetime.fromtimestamp(slot_ts, tz=timezone.utc) if slot_ts else None

    row = {
        "slug":                    m["slug"],
        "condition_id":            m.get("condition_id", ""),
        "question_id":             m.get("question_id", ""),
        "up_token_id":             up_token,
        "down_token_id":           down_token,
        "event_start_ts":          slot_ts,
        "event_end_ts":            m.get("event_end_ts"),
        "closed_ts":               m.get("closed_ts"),
        "uma_end_ts":              m.get("uma_end_ts"),
        "hour_utc":                slot_dt.hour      if slot_dt else None,
        "day_of_week":             slot_dt.weekday() if slot_dt else None,
        "session":                 _get_session(slot_dt),
        "result":                  result,
        "result_up_final_price":   result_up_final_price,
        "result_down_final_price": result_down_final_price,
        **pf,
        "volume_usdc":             m.get("volume_usdc", 0),
        "liquidity_usdc":          m.get("liquidity_usdc", 0),
        "spread_raw":              m.get("spread_raw"),
        "best_bid":                m.get("best_bid"),
        "best_ask":                m.get("best_ask"),
        "last_trade_price":        m.get("last_trade_price"),
        "order_book_enabled":      m.get("order_book_enabled"),
        "order_min_size":          m.get("order_min_size"),
        "order_tick_size":         m.get("order_tick_size"),
        "fees_enabled":            m.get("fees_enabled"),
        "fee_type":                m.get("fee_type"),
        "taker_fee_rate":          m.get("taker_fee_rate"),
        "maker_rebate_rate":       m.get("maker_rebate_rate"),
        "fee_exponent":            m.get("fee_exponent"),
        "fee_taker_only":          m.get("fee_taker_only"),
        "maker_base_fee_bps":      m.get("maker_base_fee_bps"),
        "taker_base_fee_bps":      m.get("taker_base_fee_bps"),
    }
    return row, "ok"


def build_backtest_dataset(markets, max_workers=10):
    rows    = []
    skipped = {"no_history": 0, "invalid_structure": 0, "other": 0}
    total   = len(markets)
    logger.info(f"build_backtest_dataset | {total} marchés bruts | {max_workers} workers")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_single_market, m): i for i, m in enumerate(markets)}
        for done, future in enumerate(as_completed(futures), 1):
            result, reason = future.result()
            if result:
                rows.append(result)
            else:
                skipped[reason if reason in skipped else "other"] += 1

            if done % 100 == 0 or done == total:
                logger.info(
                    f"{done}/{total} | {len(rows)} valides | "
                    f"skip: no_history={skipped['no_history']} "
                    f"invalid={skipped['invalid_structure']}"
                )

    logger.info(
        f"Terminé : {len(rows)} valides | "
        f"no_history={skipped['no_history']} | "
        f"invalid={skipped['invalid_structure']}"
    )
    return pd.DataFrame(rows)


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    markets = fetch_all_btc_5min_markets(months=3)

    if not markets:
        logger.error("Aucun marché trouvé")
    else:
        # ── Diagnostic CLOB avant de lancer le build ──
        # Prend le premier marché avec des tokens et teste le CLOB directement
        test_token = None
        for m in markets:
            tokens = m.get("token_ids", [])
            if tokens:
                test_token = tokens[0]
                break

        if test_token:
            logger.info("=== DIAGNOSTIC CLOB ===")
            diagnose_clob(test_token)
            logger.info("=== FIN DIAGNOSTIC ===")
        
        df = build_backtest_dataset(markets, max_workers=10)

        if df.empty:
            logger.error(
                "DataFrame vide. Causes possibles :\n"
                "  1. Géoblocage IP sur clob.polymarket.com (vérifier les logs [DIAG])\n"
                "  2. Rate limiting (réduire max_workers ou ajouter des délais)\n"
                "  3. Tous les marchés récupérés sont sans activité (volume=0)"
            )
        else:
            df_lite = df.drop(columns=["full_history"], errors="ignore")
            df_lite.to_parquet("btc_5min_markets.parquet", index=False)
            logger.info(f"btc_5min_markets.parquet → {len(df_lite)} lignes")

            df[["slug", "up_token_id", "full_history"]].to_parquet(
                "btc_5min_histories.parquet", index=False
            )

            print(f"\n{'='*50}")
            print(f"Marchés stockés         : {len(df)}")
            print(f"UP   : {(df['result']=='UP').sum()} ({(df['result']=='UP').mean()*100:.1f}%)")
            print(f"DOWN : {(df['result']=='DOWN').sum()} ({(df['result']=='DOWN').mean()*100:.1f}%)")
            print(f"UNKNOWN : {(df['result']=='UNKNOWN').sum()}")
            print(f"Avec frais renseignés   : {df['taker_fee_rate'].notna().sum()}")
            print(f"{'='*50}")

            cols = [
                "slug", "result",
                "price_open_up", "price_open_down",
                "price_close_up", "price_close_down",
                "sentiment_shift", "volatility",
                "volume_usdc", "session",
                "taker_fee_rate", "fees_enabled",
            ]
            print(df[cols].head(10).to_string(index=False))