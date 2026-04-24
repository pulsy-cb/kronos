"""
Kronos Polymarket Backtest Engine
==================================
Backtest de trading Polymarket 5-min avec prédictions Kronos.
Utilise les données historiques Polymarket pour le settlement réel.
"""

import json
import re
import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
import plotly.utils
from datetime import datetime, timezone
from model.kronos import set_inference_seed


def _to_native(obj):
    """Convert numpy types to native Python for JSON."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and (obj != obj):  # NaN
        return None
    return obj


def _extract_ts_from_slug(slug: str) -> int | None:
    """Extract Unix timestamp from slug like 'btc-updown-5m-1769104200'."""
    m = re.search(r'-([0-9]{10,})$', slug)
    if m:
        return int(m.group(1))
    return None


def _parse_ts(val) -> int | None:
    """Parse a timestamp value (int, float, str) into Unix seconds."""
    if val is None or val != val:  # None / NaNan
        return None
    if isinstance(val, (int, float)):
        return int(val)
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


class PolymarketBacktestSession:
    """Session de backtest Polymarket 5-min."""

    def __init__(self, df_price, df_polymarket, predictor, params):
        self.df_price = df_price  # Donnees de prix (M1 ou M5)
        self.df_polymarket = df_polymarket  # Donnees Polymarket historiques
        self.predictor = predictor
        self.params = params
        self.batch_size = params.get("batch_size", 16)
        self.cancelled = False
        self.progress = 0.0
        self.status = "pending"
        self.error_message = None
        self.current_step = 0
        self.total_steps = 0
        self.results = None

    def run(self):
        """Point d'entree principal."""
        try:
            self.status = "running"
            self._run_polymarket_backtest()
            if not self.cancelled:
                self.status = "completed"
        except Exception as e:
            self.status = "error"
            self.error_message = str(e)
            import traceback
            print("[PM Backtest FATAL]", traceback.format_exc())

    def cancel(self):
        self.cancelled = True

    def _run_polymarket_backtest(self):
        """Execute le backtest Polymarket."""
        set_inference_seed()

        df_price = self.df_price.copy()
        df_pm = self.df_polymarket.copy()
        params = self.params

        # Parametres de prediction
        lookback = params["lookback"]
        pred_len = params["pred_len"]
        step_size = params.get("step_size", 1)
        batch_size = params.get("batch_size", 16)
        signal_threshold = params.get("signal_threshold", 0.0003)
        temperature = params.get("temperature", 0.1)
        top_p = params.get("top_p", 1.0)
        sample_count = params.get("sample_count", 1)

        # Parametres Polymarket
        bet_amount = params.get("bet_amount", 1.0)
        initial_capital = params.get("initial_capital", 10000.0)
        fee_pct = params.get("fee_pct", 0.02)  # 2% de frais Polymarket

        # ── Normaliser timestamps de prix ─────────────────────
        # load_data_file garantit deja timestamps ou l'index est gere. On s'assure juste que
        # c'est bien un DatetimeIndex/tz-naive ou avec tz, et on cree une colonne _unix_ts
        # pour les comparaisons robustes.

        # Si timestamps n'existe pas, prendre l'index
        if "timestamps" not in df_price.columns:
            if "timestamp" in df_price.columns:
                df_price["timestamps"] = pd.to_datetime(df_price["timestamp"])
            elif df_price.index.name == "timestamp" or df_price.index.name is not None:
                df_price = df_price.reset_index()
                if "timestamp" in df_price.columns:
                    df_price["timestamps"] = pd.to_datetime(df_price["timestamp"])
            else:
                raise ValueError("Impossible de trouver la colonne timestamps dans les donnees de prix")

        # Coercer timestamps en datetime
        ts = df_price["timestamps"]
        if pd.api.types.is_numeric_dtype(ts):
            df_price["timestamps"] = pd.to_datetime(ts, unit="s")
        else:
            df_price["timestamps"] = pd.to_datetime(ts)

        # Cree une colonne unix pour comparaison rapide (tz-naive pour eviter les conflits)
        def _to_unix(dt_val):
            if pd.isna(dt_val):
                return np.nan
            if hasattr(dt_val, "tzinfo") and dt_val.tzinfo is not None:
                dt_val = dt_val.tz_localize(None) if hasattr(dt_val, "tz_localize") else dt_val.replace(tzinfo=None)
            return int(dt_val.timestamp())

        df_price["_unix_ts"] = df_price["timestamps"].apply(_to_unix)
        df_price = df_price.dropna(subset=["_unix_ts"]).reset_index(drop=True)
        # Ordonner par temps croissant
        df_price = df_price.sort_values("_unix_ts", ignore_index=True)

        price_ts_list = df_price["_unix_ts"].values.tolist()
        price_min_ts = price_ts_list[0]
        price_max_ts = price_ts_list[-1]

        print(f"[PM Backtest] Price range: {price_min_ts} to {price_max_ts} ({len(df_price)} rows)")

        # ── Normaliser Polymarket data ──────────────────────────────
        # Supporte deux formats:
        # 1. Format recent: event_start_ts (int), price_open_up/float, price_open_down/float, result/str
        # 2. Format ancien: slug avec timestamp dedans, entry_up_price, start_ts iso

        if "event_start_ts" in df_pm.columns:
            df_pm["_event_ts"] = df_pm["event_start_ts"].astype(int)
        elif "start_ts" in df_pm.columns:
            # start_ts peut etre int ou ISO string
            st = df_pm["start_ts"]
            if pd.api.types.is_numeric_dtype(st):
                df_pm["_event_ts"] = st.astype(int)
            else:
                df_pm["_event_ts"] = pd.to_datetime(st).apply(lambda x: int(x.timestamp()))
        elif "slug" in df_pm.columns:
            df_pm["_event_ts"] = df_pm["slug"].apply(_extract_ts_from_slug)
        else:
            raise ValueError("Donnees Polymarket sans colonne 'event_start_ts', 'start_ts' ni 'slug'")

        df_pm = df_pm.dropna(subset=["_event_ts"]).copy()
        df_pm["_event_ts"] = df_pm["_event_ts"].astype(int)

        # ── Filtrer par date si specifie ──────────────────────────
        start_date = params.get("start_date")
        end_date = params.get("end_date")
        print(f"[PM Backtest] ========== PARAMS RECUS ==========")
        print(f"[PM Backtest] start_date = {start_date!r}")
        print(f"[PM Backtest] end_date   = {end_date!r}")
        print(f"[PM Backtest] ==================================")

        if start_date or end_date:
            if start_date:
                start_ts_filter = int(pd.to_datetime(start_date).timestamp())
                # On ne garde que les PM dont l'ouverture est apres start_date
                df_pm = df_pm[df_pm["_event_ts"] >= start_ts_filter].reset_index(drop=True)
                # Pour les prix, on prend tout ce qui couvre la periode (besoin de lookback avant)
                # Donc on ne filtre pas les prix trop agressivement, juste apres start
                df_price = df_price[df_price["_unix_ts"] >= start_ts_filter - lookback * 300].reset_index(drop=True)
                price_ts_list = df_price["_unix_ts"].values.tolist()
                if not price_ts_list:
                    raise ValueError("Aucune donnee de prix disponible apres le filtre start_date")
            if end_date:
                end_ts_filter = int(pd.to_datetime(end_date).timestamp())
                df_pm = df_pm[df_pm["_event_ts"] <= end_ts_filter].reset_index(drop=True)
                df_price = df_price[df_price["_unix_ts"] <= end_ts_filter + lookback * 300].reset_index(drop=True)
                price_ts_list = df_price["_unix_ts"].values.tolist()
                if not price_ts_list:
                    raise ValueError("Aucune donnee de prix disponible apres le filtre end_date")

        pm_min_ts = int(df_pm["_event_ts"].min()) if len(df_pm) else None
        pm_max_ts = int(df_pm["_event_ts"].max()) if len(df_pm) else None
        print(f"[PM Backtest] PM events: {len(df_pm)}  range: {pm_min_ts} .. {pm_max_ts}")

        if len(df_price) < lookback + pred_len:
            raise ValueError(f"Donnees insuffisantes: besoin {lookback + pred_len}, obtenu {len(df_price)}")

        if df_pm.empty:
            raise ValueError("Aucune donnee Polymarket dans la periode selectionnee")

        # ── Creer un index pour lookup rapide des marches Polymarket ──
        pm_markets = {}
        for _, row in df_pm.iterrows():
            ts = int(row["_event_ts"])
            pm_markets[ts] = row

        # ── Generer les fenetres de backtest ──
        window_starts = sorted(pm_markets.keys())
        self.total_steps = len(window_starts)
        self.current_step = 0

        print(f"[PM Backtest] Window starts: {len(window_starts)} (from {window_starts[0]} to {window_starts[-1]})")

        # Colonnes requises
        required_cols = ["open", "high", "low", "close"]
        if "volume" in df_price.columns:
            required_cols.append("volume")

        # Capital et tracking
        capital = initial_capital
        total_wagered = 0.0
        total_pnl = 0.0
        wins = 0
        losses = 0

        bets = []  # Historique des bets
        equity_curve = []  # Courbe de capital

        # ── Preparer les fenetres ──
        windows_data = []
        # Pour lookup rapide, on a price_ts_list et on peut chercher par binary search
        # On cherche l'index dans df_price ou _unix_ts est <= window_ts (le plus proche avant)
        for i, window_ts in enumerate(window_starts):
            if i % step_size != 0:
                continue
            if window_ts not in pm_markets:
                continue

            # Chercher l'index i_price tel que price_ts_list[i_price] <= window_ts < price_ts_list[i_price + 1]
            # (plus reche candle avant le debut du marche)
            import bisect
            pos = bisect.bisect_right(price_ts_list, window_ts) - 1
            if pos < 0:
                continue

            idx = pos
            if idx < lookback - 1:
                continue

            x_df = df_price.iloc[idx - lookback + 1:idx + 1][required_cols].copy().reset_index(drop=True)
            x_timestamp = df_price.iloc[idx - lookback + 1:idx + 1]["timestamps"].reset_index(drop=True)

            # Timestamp cible pour prediction (pred_len pas de 5 minutes apres event_start_ts)
            y_dt = pd.Timestamp(window_ts, unit="s", tz="UTC")
            y_timestamp = pd.Series(
                [y_dt + pd.Timedelta(seconds=300) * (i + 1) for i in range(pred_len)],
                name="timestamps"
            )

            if isinstance(x_timestamp, pd.DatetimeIndex):
                x_timestamp = pd.Series(x_timestamp, name="timestamps")

            windows_data.append({
                "window_ts": window_ts,
                "window_dt": y_dt - pd.Timedelta(seconds=300),  # correspond à idx
                "pm_market": pm_markets[window_ts],
                "x_df": x_df,
                "x_timestamp": x_timestamp,
                "y_timestamp": y_timestamp,
                "idx": idx,
            })

        self.total_steps = len(windows_data)
        print(f"[PM Backtest] Total windows to process: {self.total_steps} (step_size={step_size})")

        if self.total_steps == 0:
            print(f"[PM Backtest WARN] 0 fenetres. Verifier chevauchement timestamps prix/PM.")
            print(f"[PM Backtest]  -> Price ts range: {price_ts_list[0]} .. {price_ts_list[-1]}")
            print(f"[PM Backtest]  -> PM ts range:    {window_starts[0]} .. {window_starts[-1]}")
            print(f"[PM Backtest]  -> Est-ce que PM ts sont DANS la range des prix ?")
            # Afficher combien de PM ts tombent dans la range
            in_range = [ts for ts in window_starts if price_ts_list[0] <= ts <= price_ts_list[-1]]
            print(f"[PM Backtest]  -> PM ts dans prix range: {len(in_range)}")

        # ── Prediction en batch ──
        all_predictions = []
        for batch_start in range(0, len(windows_data), batch_size):
            if self.cancelled:
                self.status = "cancelled"
                return

            batch_end = min(batch_start + batch_size, len(windows_data))
            batch = windows_data[batch_start:batch_end]

            try:
                with torch.inference_mode():
                    set_inference_seed()
                    pred_dfs = self.predictor.predict_batch(
                        df_list=[w["x_df"] for w in batch],
                        x_timestamp_list=[w["x_timestamp"] for w in batch],
                        y_timestamp_list=[w["y_timestamp"] for w in batch],
                        pred_len=pred_len,
                        T=temperature,
                        top_p=top_p,
                        sample_count=sample_count,
                        sample_logits=False,
                        verbose=False,
                    )
                    for i, pred_df in enumerate(pred_dfs):
                        all_predictions.append({
                            "window_ts": batch[i]["window_ts"],
                            "window_dt": batch[i]["window_dt"],
                            "pm_market": batch[i]["pm_market"],
                            "pred_df": pred_df,
                            "x_df": batch[i]["x_df"],
                        })
                    self.current_step = batch_end
                    self.progress = self.current_step / self.total_steps if self.total_steps else 0
            except Exception as e:
                print(f"[PM Backtest] Batch prediction failed: {e}")
                import traceback
                traceback.print_exc()
                continue

        # ── Traiter les predictions et simuler les trades ──
        for pred_data in all_predictions:
            window_ts = pred_data["window_ts"]
            window_dt = pred_data["window_dt"]
            pm_market = pred_data["pm_market"]
            pred_df = pred_data["pred_df"]
            x_df = pred_data["x_df"]

            current_close = float(x_df["close"].iloc[-1])
            predicted_close = float(pred_df["close"].iloc[-1])
            predicted_return = (predicted_close - current_close) / current_close

            if predicted_return > signal_threshold:
                direction = "UP"
            elif predicted_return < -signal_threshold:
                direction = "DOWN"
            else:
                direction = "FLAT"

            # Enregistrer le point d'equite
            equity_curve.append({
                "timestamp": pd.Timestamp(window_ts, unit="s", tz="UTC").isoformat(),
                "capital": capital,
                "total_wagered": total_wagered,
                "total_pnl": total_pnl,
            })

            if direction == "FLAT":
                continue

            # ── Recuperer les cotes Polymarket ──
            # Format recent: price_open_up / price_open_down (float, 0-1)
            if "price_open_up" in pm_market and "price_open_down" in pm_market:
                up_price = pm_market.get("price_open_up")
                down_price = pm_market.get("price_open_down")
            elif "entry_up_price" in pm_market:
                # Format ancien: up_price deduit de entry_up_price, down = 1 - up
                up_price_raw = pm_market.get("entry_up_price")
                if up_price_raw is None or up_price_raw != up_price_raw:
                    continue
                up_price = float(up_price_raw)
                down_price = 1.0 - up_price
            else:
                continue

            if up_price is None or down_price is None or up_price <= 0 or down_price <= 0:
                continue

            if direction == "UP":
                buy_price = up_price
                pm_side = "Up"
            else:
                buy_price = down_price
                pm_side = "Down"

            if buy_price <= 0.01:
                continue

            shares = bet_amount / buy_price

            total_wagered += bet_amount

            # ── Settlement ──
            result_direction = pm_market.get("result")
            if result_direction is None or result_direction not in ("UP", "DOWN"):
                continue

            won = (direction == result_direction)

            if won:
                gross_payout = shares
                fee = gross_payout * fee_pct
                net_payout = gross_payout - fee
                pnl = net_payout - bet_amount
                wins += 1
            else:
                gross_payout = 0
                fee = 0
                net_payout = 0
                pnl = -bet_amount
                losses += 1

            total_pnl += pnl
            capital += pnl

            bets.append(_to_native({
                "timestamp": pd.Timestamp(window_ts, unit="s", tz="UTC").isoformat(),
                "window_ts": window_ts,
                "direction": direction,
                "pm_side": pm_side,
                "buy_price": round(buy_price, 4),
                "bet_amount": bet_amount,
                "shares": round(shares, 4),
                "predicted_return": round(predicted_return, 6),
                "predicted_close": round(predicted_close, 2),
                "actual_result": result_direction,
                "won": won,
                "gross_payout": round(gross_payout, 2),
                "fee": round(fee, 2),
                "net_payout": round(net_payout, 2),
                "pnl": round(pnl, 2),
            }))

            equity_curve.append({
                "timestamp": pd.Timestamp(window_ts, unit="s", tz="UTC").isoformat(),
                "capital": capital,
                "total_wagered": total_wagered,
                "total_pnl": total_pnl,
            })

        # ── Calculer les metriques ────────────────────────────────
        total_bets = len(bets)
        win_rate = (wins / total_bets * 100) if total_bets > 0 else 0
        roi = (total_pnl / total_wagered * 100) if total_wagered > 0 else 0

        first_price = float(df_price["close"].iloc[0])
        last_price = float(df_price["close"].iloc[-1])
        bh_return = ((last_price - first_price) / first_price) * 100

        metrics = _to_native({
            "total_bets": total_bets,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": round(win_rate, 2),
            "total_wagered_eur": round(total_wagered, 2),
            "total_pnl_eur": round(total_pnl, 2),
            "roi_pct": round(roi, 2),
            "initial_capital": round(initial_capital, 2),
            "final_capital": round(capital, 2),
            "buy_hold_return_pct": round(bh_return, 2),
            "avg_bet_size": round(bet_amount, 2),
            "total_fees": round(sum(b["fee"] for b in bets), 2),
        })

        price_data = _prepare_price_data(df_price)

        self.results = {
            "params": params,
            "metrics": metrics,
            "bets": bets,
            "equity_curve": equity_curve,
            "price_data": price_data,
        }


def _prepare_price_data(df):
    """Prepare les donnees de prix pour le chart."""
    data = []
    for _, row in df.iterrows():
        ts = row["timestamps"]
        ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        data.append({
            "timestamp": ts_str,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]) if "volume" in row else 0,
        })
    return data


def create_equity_chart(results):
    """Cree la courbe d'equite."""
    equity = results["equity_curve"]
    if not equity:
        fig = go.Figure()
        fig.update_layout(title="Equity Curve (no data)", height=400)
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    timestamps = [e["timestamp"] for e in equity]
    capital_values = [e["capital"] for e in equity]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=capital_values, name="Capital",
        line=dict(color="#667eea", width=2),
    ))

    initial = results["metrics"]["initial_capital"]
    fig.add_trace(go.Scatter(
        x=timestamps, y=[initial] * len(timestamps), name="Initial Capital",
        line=dict(color="#a0aec0", width=1.5, dash="dash"),
    ))

    fig.update_layout(
        title="Equity Curve", xaxis_title="Time", yaxis_title="Capital (EUR)",
        template="plotly_white", height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def create_drawdown_chart(results):
    """Cree le chart de drawdown."""
    equity = results["equity_curve"]
    if not equity:
        fig = go.Figure()
        fig.update_layout(title="Drawdown (no data)", height=250)
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    capital_values = [e["capital"] for e in equity]
    timestamps = [e["timestamp"] for e in equity]

    peak = capital_values[0]
    drawdowns = []
    for val in capital_values:
        if val > peak:
            peak = val
        dd = (peak - val) / peak * 100 if peak > 0 else 0
        drawdowns.append(-dd)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=drawdowns, fill="tozeroy", fillcolor="rgba(239, 68, 68, 0.3)",
        line=dict(color="#e53e3e", width=1), name="Drawdown",
    ))
    fig.update_layout(
        title="Drawdown", xaxis_title="Time", yaxis_title="Drawdown (%)",
        template="plotly_white", height=250,
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def create_bets_chart(results):
    """Cree le chart des bets avec wins/losses."""
    bets = results["bets"]
    if not bets:
        fig = go.Figure()
        fig.update_layout(title="Bets (no data)", height=300)
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    wins = [b for b in bets if b["won"]]
    losses = [b for b in bets if not b["won"]]

    fig = go.Figure()
    if wins:
        fig.add_trace(go.Scatter(
            x=[b["timestamp"] for b in wins], y=[b["pnl"] for b in wins],
            mode="markers", marker=dict(symbol="triangle-up", size=12, color="#48bb78"), name="Win",
        ))
    if losses:
        fig.add_trace(go.Scatter(
            x=[b["timestamp"] for b in losses], y=[b["pnl"] for b in losses],
            mode="markers", marker=dict(symbol="triangle-down", size=12, color="#f56565"), name="Loss",
        ))
    fig.update_layout(
        title="Bets P&L", xaxis_title="Time", yaxis_title="P&L (EUR)",
        template="plotly_white", height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
