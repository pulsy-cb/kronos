"""
Kronos Polymarket Backtest Engine
==================================
Backtest de trading Polymarket 5-min avec prédictions Kronos.
Utilise les données historiques Polymarket pour le settlement réel.
"""

import json
import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
import plotly.utils
from datetime import datetime, timezone
from model.kronos import set_inference_seed


class PolymarketBacktestSession:
    """Session de backtest Polymarket 5-min."""

    def __init__(self, df_price, df_polymarket, predictor, params):
        self.df_price = df_price  # Données de prix (M1 ou M5)
        self.df_polymarket = df_polymarket  # Données Polymarket historiques
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
        """Point d'entrée principal."""
        try:
            self.status = "running"
            self._run_polymarket_backtest()
            if not self.cancelled:
                self.status = "completed"
        except Exception as e:
            self.status = "error"
            self.error_message = str(e)

    def cancel(self):
        self.cancelled = True

    def _run_polymarket_backtest(self):
        """Exécute le backtest Polymarket."""
        set_inference_seed()
        
        df_price = self.df_price
        df_pm = self.df_polymarket
        params = self.params
        
        # Paramètres de prédiction
        lookback = params["lookback"]
        pred_len = params["pred_len"]
        step_size = params.get("step_size", 1)
        signal_threshold = params.get("signal_threshold", 0.0003)
        temperature = params.get("temperature", 0.1)
        top_p = params.get("top_p", 1.0)
        sample_count = params.get("sample_count", 1)
        
        # Paramètres Polymarket
        bet_amount = params.get("bet_amount", 1.0)
        initial_capital = params.get("initial_capital", 10000.0)
        fee_pct = params.get("fee_pct", 0.02)  # 2% de frais Polymarket
        
        # Filtrer par date si spécifié
        start_date = params.get("start_date")
        end_date = params.get("end_date")
        
        if start_date or end_date:
            if "timestamps" in df_price.columns:
                if start_date:
                    start_dt = pd.to_datetime(start_date)
                    if df_price["timestamps"].dtype == "object":
                        df_price["timestamps"] = pd.to_datetime(df_price["timestamps"])
                    df_price = df_price[df_price["timestamps"] >= start_dt]
                if end_date:
                    end_dt = pd.to_datetime(end_date)
                    if df_price["timestamps"].dtype == "object":
                        df_price["timestamps"] = pd.to_datetime(df_price["timestamps"])
                    df_price = df_price[df_price["timestamps"] <= end_dt]
            df_price = df_price.reset_index(drop=True)
            
            # Filtrer aussi Polymarket
            if "event_start_ts" in df_pm.columns:
                if start_date:
                    start_ts = int(pd.to_datetime(start_date).timestamp())
                    df_pm = df_pm[df_pm["event_start_ts"] >= float(start_ts)]
                if end_date:
                    end_ts = int(pd.to_datetime(end_date).timestamp())
                    df_pm = df_pm[df_pm["event_start_ts"] <= float(end_ts)]
        
        if len(df_price) < lookback + pred_len:
            raise ValueError(f"Données insuffisantes: besoin {lookback + pred_len}, obtenu {len(df_price)}")
        
        if df_pm.empty:
            raise ValueError("Aucune donnée Polymarket dans la période sélectionnée")
        
        # Mapper les timestamps Polymarket aux données de prix
        # Polymarket: event_start_ts en Unix seconds
        # Prix: timestamps en datetime
        
        # Créer un index pour lookup rapide des marchés Polymarket
        pm_markets = {}
        for _, row in df_pm.iterrows():
            ts = int(float(row["event_start_ts"]))
            pm_markets[ts] = row
        
        # Générer les fenêtres de backtest (alignées sur 5-min comme Polymarket)
        # Utiliser les timestamps Polymarket directement pour garantir l'alignement
        pm_timestamps = sorted(df_pm["event_start_ts"].astype(int).unique())
        
        print(f"[PM Backtest] PM timestamps range: {pm_timestamps[0]} to {pm_timestamps[-1]}")
        
        # window_starts = les timestamps Polymarket directement
        window_starts = pm_timestamps
        
        self.total_steps = len(window_starts)
        self.current_step = 0
        
        # Debug info
        print(f"[PM Backtest] Price data: {len(df_price)} rows, {df_price.index[0]} to {df_price.index[-1]}")
        print(f"[PM Backtest] Polymarket data: {len(df_pm)} markets")
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
        
        # Boucle de backtest
        for window_ts in window_starts:
            if self.cancelled:
                self.status = "cancelled"
                return
            
            self.current_step += 1
            self.progress = self.current_step / self.total_steps
            
            # Vérifier si on a un marché Polymarket pour cette fenêtre
            if window_ts not in pm_markets:
                if self.current_step <= 10:
                    print(f"[PM Backtest] No PM market for window_ts={window_ts}")
                continue
            
            pm_market = pm_markets[window_ts]
            
            if self.current_step <= 10:
                print(f"[PM Backtest] Found PM market at window_ts={window_ts}: {pm_market.get('slug', 'N/A')}")
            
            # Trouver l'index dans df_price correspondant au début de la fenêtre
            # window_ts est en secondes Unix, df_price index est datetime UTC
            window_dt = pd.Timestamp(window_ts, unit="s", tz="UTC")
            
            # Utiliser l'index directement (plus efficace)
            price_index = df_price.index
            
            # Trouver la candle la plus proche avant ou égale à window_dt
            try:
                # get_indexer donne l'index le plus proche
                loc = price_index.get_indexer([window_dt], method="pad")[0]
                if loc < 0:
                    continue
                idx = price_index[loc:loc+1][0]
                idx = df_price.index.get_loc(idx)
            except Exception:
                # Fallback: search manually
                mask = price_index <= window_dt
                if not mask.any():
                    continue
                idx = mask[mask].index[-1]
            
            # Vérifier qu'on a assez de données pour lookback
            if idx < lookback - 1:
                continue
            
            # Données pour la prédiction
            x_df = df_price.iloc[idx - lookback + 1:idx + 1][required_cols].copy()
            x_timestamp = df_price.iloc[idx - lookback + 1:idx + 1]["timestamps"]
            
            # Timestamp futur pour la prédiction (fin de la fenêtre 5-min)
            y_timestamp = pd.Series([window_dt + pd.Timedelta(seconds=300)], name="timestamps")
            
            if isinstance(x_timestamp, pd.DatetimeIndex):
                x_timestamp = pd.Series(x_timestamp, name="timestamps")
            
            # Prédiction Kronos
            try:
                with torch.inference_mode():
                    set_inference_seed()
                    pred_df = self.predictor.predict(
                        df=x_df,
                        x_timestamp=x_timestamp,
                        y_timestamp=y_timestamp,
                        pred_len=pred_len,
                        T=temperature,
                        top_p=top_p,
                        sample_count=sample_count,
                        sample_logits=False,
                        verbose=False,
                    )
            except Exception as e:
                continue
            
            # Calculer la direction prédite
            current_close = float(x_df["close"].iloc[-1])
            predicted_close = float(pred_df["close"].iloc[-1])
            predicted_return = (predicted_close - current_close) / current_close
            
            # Déterminer la direction
            if predicted_return > signal_threshold:
                direction = "UP"
            elif predicted_return < -signal_threshold:
                direction = "DOWN"
            else:
                direction = "FLAT"
            
            # Debug: log first 10 predictions
            if self.current_step <= 10:
                print(f"[PM Backtest] Step {self.current_step}: pred_return={predicted_return:.6f}, threshold={signal_threshold}, direction={direction}")
            
            # Enregistrer le point d'équité
            equity_curve.append({
                "timestamp": window_dt.isoformat(),
                "capital": capital,
                "total_wagered": total_wagered,
                "total_pnl": total_pnl,
            })
            
            # Si FLAT, pas de bet
            if direction == "FLAT":
                continue
            
            # Récupérer les cotes Polymarket pour ce marché
            up_price = pm_market.get("price_open_up")
            down_price = pm_market.get("price_open_down")
            
            if up_price is None or down_price is None or up_price <= 0 or down_price <= 0:
                continue
            
            # Choisir le côté à acheter
            if direction == "UP":
                buy_price = up_price
                pm_side = "Up"
                result_key = "result_up_final_price"
            else:
                buy_price = down_price
                pm_side = "Down"
                result_key = "result_down_final_price"
            
            if buy_price <= 0.01:
                continue
            
            # Calculer les shares et payout potentiel
            shares = bet_amount / buy_price
            potential_payout = shares  # 1€ par share si win
            
            # Capital investi
            total_wagered += bet_amount
            
            # Settlement: utiliser le résultat réel Polymarket
            result_final = pm_market.get(result_key)
            result_direction = pm_market.get("result")
            
            if result_final is None:
                continue
            
            # Déterminer si on a gagné
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
            
            # Enregistrer le bet
            bets.append({
                "timestamp": window_dt.isoformat(),
                "window_ts": window_ts,
                "direction": direction,
                "pm_side": pm_side,
                "buy_price": round(buy_price, 4),
                "bet_amount": bet_amount,
                "shares": round(shares, 4),
                "predicted_return": round(predicted_return, 6),
                "predicted_close": round(predicted_close, 2),
                "actual_result": result_direction,
                "result_final_price": result_final,
                "won": won,
                "gross_payout": round(gross_payout, 2),
                "fee": round(fee, 2),
                "net_payout": round(net_payout, 2),
                "pnl": round(pnl, 2),
            })
            
            # Mettre à jour l'équité après le bet
            equity_curve.append({
                "timestamp": window_dt.isoformat(),
                "capital": capital,
                "total_wagered": total_wagered,
                "total_pnl": total_pnl,
            })
        
        # Calculer les métriques
        total_bets = len(bets)
        win_rate = (wins / total_bets * 100) if total_bets > 0 else 0
        roi = (total_pnl / total_wagered * 100) if total_wagered > 0 else 0
        
        # Buy & hold benchmark (juste pour référence)
        first_price = float(df_price["close"].iloc[0])
        last_price = float(df_price["close"].iloc[-1])
        bh_return = ((last_price - first_price) / first_price) * 100
        
        metrics = {
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
        }
        
        # Préparer les données de prix pour le chart
        price_data = _prepare_price_data(df_price)
        
        self.results = {
            "params": params,
            "metrics": metrics,
            "bets": bets,
            "equity_curve": equity_curve,
            "price_data": price_data,
        }


def _prepare_price_data(df):
    """Prépare les données de prix pour le chart."""
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
    """Crée la courbe d'équité."""
    equity = results["equity_curve"]
    
    if not equity:
        fig = go.Figure()
        fig.update_layout(title="Equity Curve (no data)", height=400)
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    timestamps = [e["timestamp"] for e in equity]
    capital_values = [e["capital"] for e in equity]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=capital_values,
        name="Capital",
        line=dict(color="#667eea", width=2),
    ))
    
    # Ajouter ligne capital initial
    initial = results["metrics"]["initial_capital"]
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=[initial] * len(timestamps),
        name="Initial Capital",
        line=dict(color="#a0aec0", width=1.5, dash="dash"),
    ))
    
    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Time",
        yaxis_title="Capital (EUR)",
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def create_drawdown_chart(results):
    """Crée le chart de drawdown."""
    equity = results["equity_curve"]
    
    if not equity:
        fig = go.Figure()
        fig.update_layout(title="Drawdown (no data)", height=250)
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    capital_values = [e["capital"] for e in equity]
    timestamps = [e["timestamp"] for e in equity]
    
    # Calculer drawdown
    peak = capital_values[0]
    drawdowns = []
    for val in capital_values:
        if val > peak:
            peak = val
        dd = (peak - val) / peak * 100 if peak > 0 else 0
        drawdowns.append(-dd)  # Négatif pour affichage vers le bas
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=drawdowns,
        fill="tozeroy",
        fillcolor="rgba(239, 68, 68, 0.3)",
        line=dict(color="#e53e3e", width=1),
        name="Drawdown",
    ))
    
    fig.update_layout(
        title="Drawdown",
        xaxis_title="Time",
        yaxis_title="Drawdown (%)",
        template="plotly_white",
        height=250,
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def create_bets_chart(results):
    """Crée le chart des bets avec wins/losses."""
    bets = results["bets"]
    
    if not bets:
        fig = go.Figure()
        fig.update_layout(title="Bets (no data)", height=300)
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Séparer wins et losses
    wins = [b for b in bets if b["won"]]
    losses = [b for b in bets if not b["won"]]
    
    fig = go.Figure()
    
    # Wins
    if wins:
        fig.add_trace(go.Scatter(
            x=[b["timestamp"] for b in wins],
            y=[b["pnl"] for b in wins],
            mode="markers",
            marker=dict(symbol="triangle-up", size=12, color="#48bb78"),
            name="Win",
        ))
    
    # Losses
    if losses:
        fig.add_trace(go.Scatter(
            x=[b["timestamp"] for b in losses],
            y=[b["pnl"] for b in losses],
            mode="markers",
            marker=dict(symbol="triangle-down", size=12, color="#f56565"),
            name="Loss",
        ))
    
    fig.update_layout(
        title="Bets P&L",
        xaxis_title="Time",
        yaxis_title="P&L (EUR)",
        template="plotly_white",
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
