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
        initial_capital = params["initial_capital"]
        transaction_cost_pct = params["transaction_cost_pct"]
        temperature = params["temperature"]
        top_p = params["top_p"]
        sample_count = params["sample_count"]

        required_data = lookback + pred_len
        if len(df) < required_data:
            raise ValueError(
                f"Need at least {required_data} rows, got {len(df)}"
            )

        max_start = len(df) - required_data
        starts = list(range(0, max_start + 1, step_size))
        self.total_steps = len(starts)

        required_cols = ["open", "high", "low", "close"]
        if "volume" in df.columns:
            required_cols.append("volume")

        signals = []

        for step_i, idx in enumerate(starts):
            if self.cancelled:
                self.status = "cancelled"
                return

            self.current_step = step_i + 1
            self.progress = self.current_step / self.total_steps

            # Extract lookback window
            x_df = df.iloc[idx : idx + lookback][required_cols].copy()
            x_timestamp = df.iloc[idx : idx + lookback]["timestamps"]

            # Build future timestamps
            y_timestamp = df.iloc[idx + lookback : idx + lookback + pred_len][
                "timestamps"
            ]

            if isinstance(x_timestamp, pd.DatetimeIndex):
                x_timestamp = pd.Series(x_timestamp, name="timestamps")
            if isinstance(y_timestamp, pd.DatetimeIndex):
                y_timestamp = pd.Series(y_timestamp, name="timestamps")

            # Run prediction
            pred_df = self.predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=temperature,
                top_p=top_p,
                sample_count=sample_count,
            )

            # Generate signal
            current_close = float(df.iloc[idx + lookback - 1]["close"])
            predicted_end_close = float(pred_df.iloc[-1]["close"])
            predicted_return = (
                predicted_end_close - current_close
            ) / current_close

            current_timestamp = df.iloc[idx + lookback - 1]["timestamps"]

            if predicted_return > signal_threshold:
                signal_type = "long"
            elif predicted_return < -signal_threshold:
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
            df, signals, initial_capital, transaction_cost_pct
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

    def _simulate_trades(self, df, signals, initial_capital, transaction_cost_pct):
        capital = initial_capital
        position = 0.0
        entry_price = 0.0
        entry_time = None
        in_position = False
        peak_capital = initial_capital

        equity_curve = []
        drawdown_series = []
        trades = []

        signal_map = {s["timestamp"]: s for s in signals}

        for _, row in df.iterrows():
            ts = row["timestamps"]
            ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            close = float(row["close"])

            signal = signal_map.get(ts_str)

            # Check for trade actions
            if signal is not None:
                if not in_position and signal["signal"] == "long":
                    # Enter long
                    cost = capital * transaction_cost_pct
                    investable = capital - cost
                    position = investable / close
                    entry_price = close
                    entry_time = ts_str
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

                elif in_position and signal["signal"] == "exit":
                    # Exit position
                    gross = position * close
                    cost = gross * transaction_cost_pct
                    capital = gross - cost
                    pnl = (close - entry_price) * position - (
                        entry_price * position * transaction_cost_pct
                    ) - (close * position * transaction_cost_pct)
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
                        }
                    )
                    position = 0.0
                    in_position = False

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
            cost = gross * transaction_cost_pct
            capital = gross - cost
            pnl = (close - entry_price) * position - (
                entry_price * position * transaction_cost_pct
            ) - (close * position * transaction_cost_pct)
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
                    "forced_close": True,
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