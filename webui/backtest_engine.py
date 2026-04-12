"""
Kronos Backtest Engine — single-asset rolling backtest using Kronos predictions.
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.utils


class BacktestSession:
    """Stateful backtest session that runs in a background thread."""

    def __init__(self, df, predictor, params):
        self.df = df
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
        """Main entry point — runs the full backtest pipeline."""
        try:
            self.status = "running"
            self._run_rolling_backtest()
            if not self.cancelled:
                self.status = "completed"
        except Exception as e:
            self.status = "error"
            self.error_message = str(e)

    def cancel(self):
        self.cancelled = True

    # ------------------------------------------------------------------
    # Rolling prediction + signal generation
    # ------------------------------------------------------------------

    def _run_rolling_backtest(self):
        df = self.df
        params = self.params
        lookback = params["lookback"]
        pred_len = params["pred_len"]
        step_size = params["step_size"]
        signal_threshold = params["signal_threshold"]
        exit_threshold = params.get("exit_threshold", signal_threshold)
        initial_capital = params["initial_capital"]
        commission_per_trade = params.get("commission_per_trade", 0.07)
        temperature = params["temperature"]
        top_p = params["top_p"]
        sample_count = params["sample_count"]
        start_date = params.get("start_date")
        end_date = params.get("end_date")

        # Filter data by date range if specified
        if start_date or end_date:
            if "timestamps" in df.columns:
                if start_date:
                    start_dt = pd.to_datetime(start_date)
                    df = df[df["timestamps"] >= start_dt]
                if end_date:
                    end_dt = pd.to_datetime(end_date)
                    df = df[df["timestamps"] <= end_dt]
            df = df.reset_index(drop=True)
            if len(df) == 0:
                raise ValueError("No data in the selected date range")

        required_data = lookback + pred_len
        if len(df) < required_data:
            raise ValueError(
                f"Need at least {required_data} rows in selected period, got {len(df)}"
            )

        max_start = len(df) - required_data
        starts = list(range(0, max_start + 1, step_size))

        required_cols = ["open", "high", "low", "close"]
        if "volume" in df.columns:
            required_cols.append("volume")

        # ---------------------------------------------------------------
        # Phase 1: Prepare all input windows for batched prediction
        # ---------------------------------------------------------------
        all_x_dfs = []
        all_x_timestamps = []
        all_y_timestamps = []
        all_current_closes = []
        all_current_timestamps = []

        for idx in starts:
            x_df = df.iloc[idx : idx + lookback][required_cols].copy()
            x_timestamp = df.iloc[idx : idx + lookback]["timestamps"]
            y_timestamp = df.iloc[idx + lookback : idx + lookback + pred_len][
                "timestamps"
            ]

            if isinstance(x_timestamp, pd.DatetimeIndex):
                x_timestamp = pd.Series(x_timestamp, name="timestamps")
            if isinstance(y_timestamp, pd.DatetimeIndex):
                y_timestamp = pd.Series(y_timestamp, name="timestamps")

            current_close = float(df.iloc[idx + lookback - 1]["close"])
            current_timestamp = df.iloc[idx + lookback - 1]["timestamps"]

            all_x_dfs.append(x_df)
            all_x_timestamps.append(x_timestamp)
            all_y_timestamps.append(y_timestamp)
            all_current_closes.append(current_close)
            all_current_timestamps.append(current_timestamp)

        total_windows = len(starts)
        self.total_steps = total_windows
        self.current_step = 0

        # ---------------------------------------------------------------
        # Phase 2: Batched prediction using predict_batch
        # ---------------------------------------------------------------
        all_predictions = [None] * total_windows
        batch_size = self.batch_size
        use_batch = hasattr(self.predictor, 'predict_batch')

        if use_batch:
            batch_idx = 0
            while batch_idx < total_windows:
                if self.cancelled:
                    self.status = "cancelled"
                    return

                end_idx = min(batch_idx + batch_size, total_windows)
                batch_dfs = all_x_dfs[batch_idx:end_idx]
                batch_x_ts = all_x_timestamps[batch_idx:end_idx]
                batch_y_ts = all_y_timestamps[batch_idx:end_idx]

                pred_dfs = self.predictor.predict_batch(
                    df_list=batch_dfs,
                    x_timestamp_list=batch_x_ts,
                    y_timestamp_list=batch_y_ts,
                    pred_len=pred_len,
                    T=temperature,
                    top_p=top_p,
                    sample_count=sample_count,
                    verbose=False,
                )

                for i, pred_df in enumerate(pred_dfs):
                    all_predictions[batch_idx + i] = pred_df

                batch_idx = end_idx
                self.current_step = end_idx
                self.progress = end_idx / total_windows
        else:
            # Fallback: sequential prediction (original behavior)
            for step_i in range(total_windows):
                if self.cancelled:
                    self.status = "cancelled"
                    return

                pred_df = self.predictor.predict(
                    df=all_x_dfs[step_i],
                    x_timestamp=all_x_timestamps[step_i],
                    y_timestamp=all_y_timestamps[step_i],
                    pred_len=pred_len,
                    T=temperature,
                    top_p=top_p,
                    sample_count=sample_count,
                )
                all_predictions[step_i] = pred_df

                self.current_step = step_i + 1
                self.progress = self.current_step / total_windows

        # ---------------------------------------------------------------
        # Phase 3: Generate signals from predictions
        # ---------------------------------------------------------------
        signals = []

        for step_i in range(total_windows):
            pred_df = all_predictions[step_i]
            current_close = all_current_closes[step_i]
            predicted_end_close = float(pred_df.iloc[-1]["close"])
            predicted_return = (predicted_end_close - current_close) / current_close
            current_timestamp = all_current_timestamps[step_i]

            if predicted_return > signal_threshold:
                signal_type = "long"
            elif predicted_return < -exit_threshold:
                signal_type = "exit"
            else:
                signal_type = "neutral"

            signals.append(
                {
                    "timestamp": (
                        current_timestamp.isoformat()
                        if hasattr(current_timestamp, "isoformat")
                        else str(current_timestamp)
                    ),
                    "signal": signal_type,
                    "predicted_return": predicted_return,
                    "current_close": current_close,
                    "predicted_close": predicted_end_close,
                }
            )

        # Simulate trades
        trade_results = self._simulate_trades(
            df, signals, initial_capital,
            commission_per_trade=params.get("commission_per_trade", 0.07),
            stop_loss_pct=params.get("stop_loss_pct", 0.0),
            take_profit_pct=params.get("take_profit_pct", 0.0),
            max_hold_bars=params.get("max_hold_bars", 0),
        )

        # Calculate metrics
        metrics = self._calculate_metrics(
            trade_results, initial_capital, df
        )

        self.results = {
            "params": params,
            "metrics": metrics,
            "equity_curve": trade_results["equity_curve"],
            "buy_hold_curve": trade_results["buy_hold_curve"],
            "drawdown_series": trade_results["drawdown_series"],
            "signals": signals,
            "trades": trade_results["trades"],
            "price_data": self._prepare_price_data(df),
        }

    # ------------------------------------------------------------------
    # Trade simulation
    # ------------------------------------------------------------------

    def _simulate_trades(self, df, signals, initial_capital, commission_per_trade=0.07,
                          stop_loss_pct=0.0, take_profit_pct=0.0, max_hold_bars=0):
        # commission_per_trade = total round-trip commission (split 50/50 on buy/sell)
        commission_per_side = commission_per_trade / 2.0
        capital = initial_capital
        position = 0.0
        entry_price = 0.0
        entry_time = None
        entry_bar_idx = 0
        in_position = False
        peak_capital = initial_capital

        equity_curve = []
        drawdown_series = []
        trades = []

        signal_map = {s["timestamp"]: s for s in signals}

        for bar_idx, (_, row) in enumerate(df.iterrows()):
            ts = row["timestamps"]
            ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            close = float(row["close"])
            high = float(row["high"])
            low = float(row["low"])

            signal = signal_map.get(ts_str)

            # --- Check exit conditions when in position ---
            if in_position:
                exit_reason = None

                # Stop-loss: check against bar low (intraday)
                if stop_loss_pct > 0:
                    low_return = (low - entry_price) / entry_price
                    if low_return <= -stop_loss_pct:
                        exit_reason = "stop_loss"
                        # Use stop price, not close
                        close = entry_price * (1 - stop_loss_pct)

                # Take-profit: check against bar high (intraday)
                if take_profit_pct > 0 and exit_reason is None:
                    high_return = (high - entry_price) / entry_price
                    if high_return >= take_profit_pct:
                        exit_reason = "take_profit"
                        close = entry_price * (1 + take_profit_pct)

                # Max hold period
                if max_hold_bars > 0 and exit_reason is None:
                    bars_held = bar_idx - entry_bar_idx
                    if bars_held >= max_hold_bars:
                        exit_reason = "max_hold"

                # Signal-based exit (EXIT signal from model)
                if exit_reason is None and signal is not None and signal["signal"] == "exit":
                    exit_reason = "signal"

                # Execute exit
                if exit_reason is not None:
                    gross = position * close
                    capital = gross - commission_per_side
                    pnl = (close - entry_price) * position - commission_per_trade
                    return_pct = (close - entry_price) / entry_price
                    trades.append(
                        {
                            "type": "sell",
                            "timestamp": ts_str,
                            "price": close,
                            "shares": position,
                            "pnl": pnl,
                            "return_pct": return_pct,
                            "entry_price": entry_price,
                            "entry_time": entry_time,
                            "exit_reason": exit_reason,
                        }
                    )
                    position = 0.0
                    in_position = False

            # --- Check entry conditions when NOT in position ---
            if not in_position and signal is not None and signal["signal"] == "long":
                investable = capital - commission_per_side
                position = investable / close
                entry_price = close
                entry_time = ts_str
                entry_bar_idx = bar_idx
                capital = 0.0
                in_position = True
                trades.append(
                    {
                        "type": "buy",
                        "timestamp": ts_str,
                        "price": close,
                        "shares": position,
                    }
                )

            # Portfolio value
            portfolio_value = capital + (position * close if in_position else 0.0)
            equity_curve.append(
                {"timestamp": ts_str, "portfolio_value": portfolio_value}
            )

            # Drawdown
            peak_capital = max(peak_capital, portfolio_value)
            drawdown = (
                (peak_capital - portfolio_value) / peak_capital
                if peak_capital > 0
                else 0.0
            )
            drawdown_series.append(
                {"timestamp": ts_str, "drawdown": drawdown}
            )

        # Force-close any open position at the end
        if in_position:
            close = float(df.iloc[-1]["close"])
            ts = df.iloc[-1]["timestamps"]
            ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            gross = position * close
            capital = gross - commission_per_side
            pnl = (close - entry_price) * position - commission_per_trade
            return_pct = (close - entry_price) / entry_price
            trades.append(
                {
                    "type": "sell",
                    "timestamp": ts_str,
                    "price": close,
                    "shares": position,
                    "pnl": pnl,
                    "return_pct": return_pct,
                    "entry_price": entry_price,
                    "entry_time": entry_time,
                    "exit_reason": "forced_close",
                }
            )
            position = 0.0
            in_position = False

        # Buy-and-hold benchmark
        first_close = float(df.iloc[0]["close"])
        buy_hold_curve = []
        for _, row in df.iterrows():
            ts = row["timestamps"]
            ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            bh_value = initial_capital * (float(row["close"]) / first_close)
            buy_hold_curve.append(
                {"timestamp": ts_str, "portfolio_value": bh_value}
            )

        return {
            "equity_curve": equity_curve,
            "buy_hold_curve": buy_hold_curve,
            "drawdown_series": drawdown_series,
            "trades": trades,
        }

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _calculate_metrics(self, trade_results, initial_capital, df):
        equity = trade_results["equity_curve"]
        trades = trade_results["trades"]
        drawdown = trade_results["drawdown_series"]

        final_value = equity[-1]["portfolio_value"] if equity else initial_capital
        total_return = (final_value - initial_capital) / initial_capital

        # Daily returns for Sharpe
        returns = []
        for i in range(1, len(equity)):
            prev = equity[i - 1]["portfolio_value"]
            curr = equity[i]["portfolio_value"]
            if prev > 0:
                returns.append((curr - prev) / prev)

        sharpe = 0.0
        if len(returns) > 1 and np.std(returns) > 0:
            # Detect bar frequency to annualize correctly
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))

        # Max drawdown
        max_dd = max((d["drawdown"] for d in drawdown), default=0.0)

        # Win rate
        sell_trades = [t for t in trades if t["type"] == "sell"]
        wins = [t for t in sell_trades if t["pnl"] > 0]
        win_rate = len(wins) / len(sell_trades) if sell_trades else 0.0

        # Profit factor
        gross_profit = sum(t["pnl"] for t in sell_trades if t["pnl"] > 0)
        gross_loss = abs(sum(t["pnl"] for t in sell_trades if t["pnl"] < 0))
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else float("inf")
        )

        # Buy & hold return
        first_close = float(df.iloc[0]["close"])
        last_close = float(df.iloc[-1]["close"])
        bh_return = (last_close - first_close) / first_close

        # Average trade return
        avg_trade_return = (
            float(np.mean([t["return_pct"] for t in sell_trades]))
            if sell_trades
            else 0.0
        )

        return {
            "total_return_pct": round(total_return * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "win_rate_pct": round(win_rate * 100, 1),
            "num_trades": len(sell_trades),
            "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "Inf",
            "buy_hold_return_pct": round(bh_return * 100, 2),
            "final_value": round(final_value, 2),
            "avg_trade_return_pct": round(avg_trade_return * 100, 2),
        }

    # ------------------------------------------------------------------
    # Price data for chart
    # ------------------------------------------------------------------

    def _prepare_price_data(self, df):
        data = []
        for _, row in df.iterrows():
            ts = row["timestamps"]
            ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            data.append(
                {
                    "timestamp": ts_str,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]) if "volume" in row else 0,
                }
            )
        return data


# ======================================================================
# Chart generation functions
# ======================================================================


def create_equity_chart(results):
    """Equity curve + buy-and-hold benchmark."""
    equity = results["equity_curve"]
    buy_hold = results["buy_hold_curve"]

    timestamps = [e["timestamp"] for e in equity]
    strategy_values = [e["portfolio_value"] for e in equity]
    bh_values = [b["portfolio_value"] for b in buy_hold]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=strategy_values,
            name="Strategy",
            line=dict(color="#667eea", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=bh_values,
            name="Buy & Hold",
            line=dict(color="#a0aec0", width=1.5, dash="dash"),
        )
    )

    # Mark trade entry/exit on equity curve
    for t in results["trades"]:
        if t["type"] == "buy":
            fig.add_trace(
                go.Scatter(
                    x=[t["timestamp"]],
                    y=[
                        next(
                            (
                                e["portfolio_value"]
                                for e in equity
                                if e["timestamp"] == t["timestamp"]
                            ),
                            0,
                        )
                    ],
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=12, color="#48bb78"),
                    name="Buy",
                    showlegend=False,
                )
            )
        elif t["type"] == "sell":
            fig.add_trace(
                go.Scatter(
                    x=[t["timestamp"]],
                    y=[
                        next(
                            (
                                e["portfolio_value"]
                                for e in equity
                                if e["timestamp"] == t["timestamp"]
                            ),
                            0,
                        )
                    ],
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=12, color="#f56565"),
                    name="Sell",
                    showlegend=False,
                )
            )

    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Time",
        yaxis_title="Portfolio Value",
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def create_drawdown_chart(results):
    """Drawdown area chart (inverted)."""
    dd = results["drawdown_series"]
    timestamps = [d["timestamp"] for d in dd]
    values = [-d["drawdown"] * 100 for d in dd]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=values,
            fill="tozeroy",
            fillcolor="rgba(239, 68, 68, 0.3)",
            line=dict(color="#e53e3e", width=1),
            name="Drawdown",
        )
    )
    fig.update_layout(
        title="Drawdown",
        xaxis_title="Time",
        yaxis_title="Drawdown (%)",
        template="plotly_white",
        height=250,
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def create_price_trades_chart(results):
    """Candlestick chart with trade markers overlaid."""
    price_data = results["price_data"]
    trades = results["trades"]

    timestamps = [p["timestamp"] for p in price_data]
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=timestamps,
            open=[p["open"] for p in price_data],
            high=[p["high"] for p in price_data],
            low=[p["low"] for p in price_data],
            close=[p["close"] for p in price_data],
            name="Price",
            increasing_line_color="#26A69A",
            decreasing_line_color="#EF5350",
        )
    )

    # Buy markers
    buy_trades = [t for t in trades if t["type"] == "buy"]
    if buy_trades:
        fig.add_trace(
            go.Scatter(
                x=[t["timestamp"] for t in buy_trades],
                y=[t["price"] for t in buy_trades],
                mode="markers",
                marker=dict(symbol="triangle-up", size=14, color="#48bb78"),
                name="Buy",
            )
        )

    # Sell markers
    sell_trades = [t for t in trades if t["type"] == "sell"]
    if sell_trades:
        fig.add_trace(
            go.Scatter(
                x=[t["timestamp"] for t in sell_trades],
                y=[t["price"] for t in sell_trades],
                mode="markers",
                marker=dict(symbol="triangle-down", size=14, color="#f56565"),
                name="Sell",
            )
        )

    fig.update_layout(
        title="Price Chart with Trade Signals",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_white",
        height=450,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)