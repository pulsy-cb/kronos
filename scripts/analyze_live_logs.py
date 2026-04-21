#!/usr/bin/env python3
"""
Analyse poussée des logs live Kronos.
Usage: python scripts/analyze_live_logs.py [log_file_or_pattern]

Examples:
  python scripts/analyze_live_logs.py webui/logs/live_XAUUSD_M1_2026-04-21.jsonl
  python scripts/analyze_live_logs.py webui/logs/live_*_M1_2026-04-21.jsonl
  python scripts/analyze_live_logs.py                          # analyse tous les logs
"""

import json
import sys
import glob
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# ── Data loading ──────────────────────────────────────────────────────────

def load_events(filepath):
    """Load all JSONL events from a file."""
    events = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def load_all_logs(pattern=None):
    """Load events from one or more log files."""
    if pattern is None:
        pattern = "webui/logs/live_*.jsonl"

    files = sorted(glob.glob(pattern))
    if not files:
        print(f"Aucun fichier trouvé pour: {pattern}")
        sys.exit(1)

    all_events = []
    for fp in files:
        fname = os.path.basename(fp)
        parts = fname.replace("live_", "").replace(".jsonl", "").split("_")
        # live_XAUUSD_M1_2026-04-21.jsonl -> symbol=XAUUSD, tf=M1, date=2026-04-21
        symbol = parts[0] if len(parts) >= 1 else "?"
        tf = parts[1] if len(parts) >= 2 else "?"
        date = parts[2] if len(parts) >= 3 else "?"

        events = load_events(fp)
        for e in events:
            e["_file"] = fname
            e["_symbol"] = symbol
            e["_timeframe"] = tf
            e["_date"] = date
        all_events.extend(events)

    print(f"Chargé {len(all_events)} événements depuis {len(files)} fichier(s)")
    for fp in files:
        print(f"  - {os.path.basename(fp)}")
    print()
    return all_events


# ── Parsing ────────────────────────────────────────────────────────────────

def parse_trades(events):
    """
    Match trade open/close pairs by ticket.
    Returns list of trade dicts with computed fields.
    """
    opens = {}  # ticket -> event
    trades = []

    for e in events:
        if e.get("type") != "trade":
            continue
        ticket = e.get("ticket")
        if ticket is None:
            continue
        if e.get("action") == "open":
            opens[ticket] = e
        elif e.get("action") == "close":
            open_ev = opens.pop(ticket, None)
            if open_ev is None:
                # orphan close, skip
                continue
            close_ev = e
            pnl = close_ev.get("pnl", 0.0)
            direction = close_ev.get("direction", open_ev.get("direction", "?"))
            duration_min = 0
            try:
                t_open = datetime.fromisoformat(open_ev["timestamp"])
                t_close = datetime.fromisoformat(close_ev["timestamp"])
                duration_min = (t_close - t_open).total_seconds() / 60.0
            except (ValueError, KeyError):
                pass

            trades.append({
                "ticket": ticket,
                "model_key": open_ev.get("model_key", "?"),
                "model_name": open_ev.get("model_name", "?"),
                "symbol": open_ev.get("_symbol", "?"),
                "session_id": open_ev.get("session_id", "?"),
                "direction": direction,
                "entry_price": open_ev.get("price", 0),
                "exit_price": close_ev.get("price", 0),
                "volume": open_ev.get("volume", 0),
                "sl": open_ev.get("sl"),
                "tp": open_ev.get("tp"),
                "pnl": pnl,
                "reason": close_ev.get("reason", "?"),
                "duration_min": round(duration_min, 2),
                "open_ts": open_ev.get("timestamp", ""),
                "close_ts": close_ev.get("timestamp", ""),
            })

    return trades


def parse_signals(events):
    """Extract all signal events."""
    signals = []
    for e in events:
        if e.get("type") == "signal":
            signals.append(e)
    return signals


def parse_equity(events):
    """Extract equity curve per model."""
    equity_by_model = defaultdict(list)
    for e in events:
        if e.get("type") == "equity":
            mk = e.get("model_key", "?")
            equity_by_model[mk].append({
                "timestamp": e.get("timestamp", ""),
                "equity": e.get("equity", 0),
                "balance": e.get("balance", 0),
            })
    return dict(equity_by_model)


# ── Stats computation ───────────────────────────────────────────────────────

def compute_model_stats(trades, signals, model_key):
    """Compute comprehensive stats for a single model."""
    model_trades = [t for t in trades if t["model_key"] == model_key]
    model_signals = [s for s in signals if s.get("model_key") == model_key]

    if not model_trades and not model_signals:
        return None

    stats = {}

    # ── Trade stats ──
    if model_trades:
        pnls = [t["pnl"] for t in model_trades]
        wins = [t for t in model_trades if t["pnl"] > 0]
        losses = [t for t in model_trades if t["pnl"] < 0]
        breakeven = [t for t in model_trades if t["pnl"] == 0]

        total_pnl = sum(pnls)
        avg_pnl = total_pnl / len(model_trades) if model_trades else 0

        # Win rate
        stats["total_trades"] = len(model_trades)
        stats["wins"] = len(wins)
        stats["losses"] = len(losses)
        stats["breakeven"] = len(breakeven)
        stats["win_rate"] = len(wins) / len(model_trades) * 100 if model_trades else 0

        # PnL
        stats["total_pnl"] = round(total_pnl, 2)
        stats["avg_pnl"] = round(avg_pnl, 4)
        stats["best_trade"] = round(max(pnls), 2) if pnls else 0
        stats["worst_trade"] = round(min(pnls), 2) if pnls else 0
        stats["median_pnl"] = round(sorted(pnls)[len(pnls)//2], 2) if pnls else 0

        # Profit factor
        gross_profit = sum(t["pnl"] for t in wins) if wins else 0
        gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 0
        stats["profit_factor"] = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0
        stats["gross_profit"] = round(gross_profit, 2)
        stats["gross_loss"] = round(gross_loss, 2)

        # Avg win / avg loss
        stats["avg_win"] = round(gross_profit / len(wins), 4) if wins else 0
        stats["avg_loss"] = round(gross_loss / len(losses), 4) if losses else 0

        # Expectancy
        stats["expectancy"] = round(
            (stats["win_rate"]/100) * stats["avg_win"] - (1 - stats["win_rate"]/100) * stats["avg_loss"],
            4
        ) if stats["avg_win"] or stats["avg_loss"] else 0

        # Direction breakdown
        longs = [t for t in model_trades if t["direction"] == "long"]
        shorts = [t for t in model_trades if t["direction"] == "short"]
        stats["long_trades"] = len(longs)
        stats["short_trades"] = len(shorts)
        stats["long_wins"] = len([t for t in longs if t["pnl"] > 0])
        stats["short_wins"] = len([t for t in shorts if t["pnl"] > 0])
        stats["long_win_rate"] = round(stats["long_wins"] / len(longs) * 100, 1) if longs else 0
        stats["short_win_rate"] = round(stats["short_wins"] / len(shorts) * 100, 1) if shorts else 0
        stats["long_pnl"] = round(sum(t["pnl"] for t in longs), 2)
        stats["short_pnl"] = round(sum(t["pnl"] for t in shorts), 2)

        # Close reason breakdown
        reason_counts = defaultdict(int)
        reason_pnl = defaultdict(float)
        for t in model_trades:
            reason_counts[t["reason"]] += 1
            reason_pnl[t["reason"]] += t["pnl"]
        stats["close_reasons"] = dict(reason_counts)
        stats["close_reason_pnl"] = {k: round(v, 2) for k, v in reason_pnl.items()}

        # Duration stats
        durations = [t["duration_min"] for t in model_trades]
        stats["avg_duration_min"] = round(sum(durations) / len(durations), 2) if durations else 0
        stats["max_duration_min"] = round(max(durations), 2) if durations else 0
        stats["min_duration_min"] = round(min(durations), 2) if durations else 0

        # Consecutive wins/losses (max streak)
        max_win_streak = 0
        max_loss_streak = 0
        cur_win = 0
        cur_loss = 0
        for t in model_trades:
            if t["pnl"] > 0:
                cur_win += 1
                cur_loss = 0
                max_win_streak = max(max_win_streak, cur_win)
            elif t["pnl"] < 0:
                cur_loss += 1
                cur_win = 0
                max_loss_streak = max(max_loss_streak, cur_loss)
            else:
                cur_win = 0
                cur_loss = 0
        stats["max_win_streak"] = max_win_streak
        stats["max_loss_streak"] = max_loss_streak

        # Drawdown (peak-to-trough)
        cumulative = []
        running = 0
        for t in model_trades:
            running += t["pnl"]
            cumulative.append(running)
        peak = 0
        max_dd = 0
        for c in cumulative:
            peak = max(peak, c)
            dd = peak - c
            max_dd = max(max_dd, dd)
        stats["max_drawdown"] = round(max_dd, 2)
        stats["final_cumulative_pnl"] = round(cumulative[-1], 2) if cumulative else 0

        # Risk-reward per trade
        sl_distances = []
        tp_distances = []
        for t in model_trades:
            if t["sl"] and t["tp"] and t["entry_price"]:
                if t["direction"] == "long":
                    sl_dist = t["entry_price"] - t["sl"]
                    tp_dist = t["tp"] - t["entry_price"]
                else:
                    sl_dist = t["sl"] - t["entry_price"]
                    tp_dist = t["entry_price"] - t["tp"]
                if sl_dist > 0:
                    sl_distances.append(sl_dist)
                    tp_distances.append(tp_dist)
        if sl_distances:
            avg_rr = sum(tp/sl for tp, sl in zip(tp_distances, sl_distances)) / len(sl_distances)
            stats["avg_risk_reward"] = round(avg_rr, 2)
        else:
            stats["avg_risk_reward"] = None

    # ── Signal stats ──
    if model_signals:
        signal_dirs = defaultdict(int)
        for s in model_signals:
            signal_dirs[s.get("signal", "?")] += 1

        stats["total_signals"] = len(model_signals)
        stats["signal_directions"] = dict(signal_dirs)

        # Predicted return stats
        pred_returns = [s.get("predicted_return", 0) for s in model_signals if s.get("predicted_return") is not None]
        if pred_returns:
            stats["avg_predicted_return"] = round(sum(pred_returns) / len(pred_returns), 6)
            stats["max_predicted_return"] = round(max(pred_returns), 6)
            stats["min_predicted_return"] = round(min(pred_returns), 6)
            stats["std_predicted_return"] = round(
                (sum((r - sum(pred_returns)/len(pred_returns))**2 for r in pred_returns) / len(pred_returns)) ** 0.5,
                6
            )

        # Signal-to-trade ratio
        if model_trades:
            stats["signal_to_trade_ratio"] = round(len(model_signals) / len(model_trades), 2)

    # ── Equity curve ──
    equity_events = [e for e in events_global if e.get("type") == "equity" and e.get("model_key") == model_key]
    if equity_events:
        equities = [e.get("equity", 0) for e in equity_events]
        stats["starting_equity"] = equities[0] if equities else 0
        stats["ending_equity"] = equities[-1] if equities else 0
        stats["equity_change"] = round(stats["ending_equity"] - stats["starting_equity"], 2) if equities else 0
        stats["peak_equity"] = round(max(equities), 2)
        stats["min_equity"] = round(min(equities), 2)

    return stats


# ── Display ─────────────────────────────────────────────────────────────────

def print_separator(char="─", length=60):
    print(char * length)


def print_model_report(model_key, model_name, stats, symbol="?", timeframe="?"):
    if stats is None:
        return

    print_separator("═")
    print(f"  {model_name}  ({model_key})")
    print(f"  Symbole: {symbol}  |  Timeframe: {timeframe}")
    print_separator("═")

    # ── Trades ──
    if "total_trades" in stats:
        print()
        print("  ┌─ TRADES ──────────────────────────────────────┐")
        print(f"  │  Total trades:      {stats['total_trades']:>6}                    │")
        print(f"  │  Wins:               {stats['wins']:>6}                    │")
        print(f"  │  Losses:             {stats['losses']:>6}                    │")
        print(f"  │  Breakeven:          {stats['breakeven']:>6}                    │")
        print(f"  │  Win rate:           {stats['win_rate']:>6.1f}%                  │")
        print( "  └───────────────────────────────────────────────┘")

        print()
        print("  ┌─ PnL ─────────────────────────────────────────┐")
        print(f"  │  Total PnL:       {stats['total_pnl']:>10.2f}                    │")
        print(f"  │  Avg PnL/trade:   {stats['avg_pnl']:>10.4f}                    │")
        print(f"  │  Median PnL:      {stats['median_pnl']:>10.2f}                    │")
        print(f"  │  Best trade:      {stats['best_trade']:>10.2f}                    │")
        print(f"  │  Worst trade:     {stats['worst_trade']:>10.2f}                    │")
        print(f"  │  Gross profit:    {stats['gross_profit']:>10.2f}                    │")
        print(f"  │  Gross loss:       {stats['gross_loss']:>10.2f}                    │")
        pf = stats['profit_factor']
        pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"
        print(f"  │  Profit factor:    {pf_str:>10}                    │")
        print(f"  │  Expectancy:       {stats['expectancy']:>10.4f}                    │")
        print( "  └───────────────────────────────────────────────┘")

        print()
        print("  ┌─ DIRECTIONS ───────────────────────────────────┐")
        print(f"  │  Long  trades:    {stats['long_trades']:>5}   PnL: {stats['long_pnl']:>8.2f}       │")
        print(f"  │  Short trades:    {stats['short_trades']:>5}   PnL: {stats['short_pnl']:>8.2f}       │")
        print(f"  │  Long  win rate:  {stats['long_win_rate']:>5.1f}%                      │")
        print(f"  │  Short win rate:  {stats['short_win_rate']:>5.1f}%                      │")
        print( "  └───────────────────────────────────────────────┘")

        print()
        print("  ┌─ CLOSE REASONS ───────────────────────────────┐")
        for reason, count in sorted(stats["close_reasons"].items()):
            pnl = stats["close_reason_pnl"].get(reason, 0)
            print(f"  │  {reason:<14} {count:>4}x  PnL: {pnl:>8.2f}         │")
        print( "  └───────────────────────────────────────────────┘")

        print()
        print("  ┌─ DURATION ─────────────────────────────────────┐")
        print(f"  │  Avg duration:    {stats['avg_duration_min']:>8.2f} min              │")
        print(f"  │  Max duration:    {stats['max_duration_min']:>8.2f} min              │")
        print(f"  │  Min duration:    {stats['min_duration_min']:>8.2f} min              │")
        print( "  └───────────────────────────────────────────────┘")

        print()
        print("  ┌─ STREAKS & DRAWDOWN ───────────────────────────┐")
        print(f"  │  Max win streak:    {stats['max_win_streak']:>4}                       │")
        print(f"  │  Max loss streak:   {stats['max_loss_streak']:>4}                       │")
        print(f"  │  Max drawdown:    {stats['max_drawdown']:>8.2f}                       │")
        print(f"  │  Cumulative PnL: {stats['final_cumulative_pnl']:>8.2f}                       │")
        print( "  └───────────────────────────────────────────────┘")

        if stats.get("avg_risk_reward") is not None:
            print()
            print("  ┌─ RISK/REWARD ──────────────────────────────────┐")
            print(f"  │  Avg Risk:Reward:   {stats['avg_risk_reward']:>6.2f}:1                   │")
            print( "  └───────────────────────────────────────────────┘")

    # ── Signals ──
    if "total_signals" in stats:
        print()
        print("  ┌─ SIGNALS ─────────────────────────────────────┐")
        print(f"  │  Total signals:     {stats['total_signals']:>6}                    │")
        for direction, count in sorted(stats["signal_directions"].items()):
            pct = count / stats["total_signals"] * 100
            bar = "█" * int(pct / 2)
            print(f"  │  {direction:<10}  {count:>4}  ({pct:>5.1f}%) {bar:<20} │")
        if "avg_predicted_return" in stats:
            print(f"  │  Avg pred return: {stats['avg_predicted_return']:>10.6f}               │")
            print(f"  │  Max pred return: {stats['max_predicted_return']:>10.6f}               │")
            print(f"  │  Min pred return: {stats['min_predicted_return']:>10.6f}               │")
            print(f"  │  Std pred return: {stats['std_predicted_return']:>10.6f}               │")
        if "signal_to_trade_ratio" in stats:
            print(f"  │  Signal/trade:      {stats['signal_to_trade_ratio']:>6.1f}:1                  │")
        print( "  └───────────────────────────────────────────────┘")

    # ── Equity ──
    if "starting_equity" in stats:
        print()
        print("  ┌─ EQUITY ───────────────────────────────────────┐")
        print(f"  │  Starting equity:  {stats['starting_equity']:>10.2f}                 │")
        print(f"  │  Ending equity:    {stats['ending_equity']:>10.2f}                 │")
        print(f"  │  Change:           {stats['equity_change']:>+10.2f}                 │")
        print(f"  │  Peak equity:      {stats['peak_equity']:>10.2f}                 │")
        print(f"  │  Min equity:       {stats['min_equity']:>10.2f}                 │")
        if stats["starting_equity"] > 0:
            ret_pct = stats["equity_change"] / stats["starting_equity"] * 100
            print(f"  │  Return:            {ret_pct:>+9.2f}%                 │")
        print( "  └───────────────────────────────────────────────┘")

    print()


def print_comparison_table(trades, signals):
    """Print a compact comparison table across all models."""
    model_keys = sorted(set(t["model_key"] for t in trades) | set(s.get("model_key") for s in signals))
    if not model_keys:
        return

    model_names = {}
    for t in trades:
        model_names[t["model_key"]] = t["model_name"]
    for s in signals:
        model_names[s.get("model_key")] = s.get("model_name", s.get("model_key"))

    print()
    print_separator("═")
    print("  COMPARAISON DES MODELES")
    print_separator("═")
    print()

    header = f"  {'Modèle':<25} {'Trades':>6} {'Win%':>7} {'PnL':>9} {'Avg':>8} {'PF':>6} {'MaxDD':>8} {'StreakW':>8} {'StreakL':>8}"
    print(header)
    print(f"  {'─'*25} {'─'*6} {'─'*7} {'─'*9} {'─'*8} {'─'*6} {'─'*8} {'─'*8} {'─'*8}")

    for mk in model_keys:
        model_trades = [t for t in trades if t["model_key"] == mk]
        stats = compute_model_stats(trades, signals, mk)
        if stats is None:
            continue
        name = model_names.get(mk, mk)[:24]
        n = stats.get("total_trades", 0)
        wr = stats.get("win_rate", 0)
        pnl = stats.get("total_pnl", 0)
        avg = stats.get("avg_pnl", 0)
        pf = stats.get("profit_factor", 0)
        pf_s = f"{pf:.1f}" if pf != float("inf") else "∞"
        mdd = stats.get("max_drawdown", 0)
        sw = stats.get("max_win_streak", 0)
        sl = stats.get("max_loss_streak", 0)

        print(f"  {name:<25} {n:>6} {wr:>6.1f}% {pnl:>+9.2f} {avg:>+8.4f} {pf_s:>6} {mdd:>8.2f} {sw:>8} {sl:>8}")

    print()


def print_trade_log(trades, model_key=None):
    """Print detailed trade log, optionally filtered by model."""
    filtered = trades if model_key is None else [t for t in trades if t["model_key"] == model_key]
    if not filtered:
        return

    print()
    print_separator("─")
    label = f"  TRADE LOG — {model_key}" if model_key else "  TRADE LOG (ALL)"
    print(label)
    print_separator("─")
    print(f"  {'#':>3} {'Dir':>5} {'Entry':>10} {'Exit':>10} {'PnL':>8} {'Reason':>12} {'Dur(min)':>8} {'Model':>16}")
    print(f"  {'─'*3} {'─'*5} {'─'*10} {'─'*10} {'─'*8} {'─'*12} {'─'*8} {'─'*16}")

    for i, t in enumerate(filtered, 1):
        print(f"  {i:>3} {t['direction']:>5} {t['entry_price']:>10.2f} {t['exit_price']:>10.2f} "
              f"{t['pnl']:>+8.2f} {t['reason']:>12} {t['duration_min']:>8.2f} {t['model_key']:>16}")


# ── Main ────────────────────────────────────────────────────────────────────

events_global = []

def main():
    global events_global

    pattern = sys.argv[1] if len(sys.argv) > 1 else "webui/logs/live_*.jsonl"

    events = load_all_logs(pattern)
    events_global = events

    trades = parse_trades(events)
    signals = parse_signals(events)

    # Detect unique models & symbols
    model_keys = sorted(set(t["model_key"] for t in trades) | set(s.get("model_key") for s in signals))
    model_names = {}
    for t in trades:
        model_names[t["model_key"]] = t["model_name"]
    for s in signals:
        model_names[s.get("model_key")] = s.get("model_name", s.get("model_key"))

    symbols = sorted(set(t.get("symbol", "?") for t in trades) | set(s.get("_symbol", "?") for s in signals))
    timeframes = sorted(set(s.get("_timeframe", "?") for s in signals) | set(t.get("_timeframe", "?") for t in trades))

    print(f"Modèles détectés: {model_keys}")
    print(f"Symboles: {symbols}")
    print(f"Timeframes: {timeframes}")
    print(f"Trades reconstitués: {len(trades)}")
    print(f"Signaux: {len(signals)}")
    print()

    # ── Per-model detailed reports ──
    for mk in model_keys:
        name = model_names.get(mk, mk)
        stats = compute_model_stats(trades, signals, mk)
        if stats:
            sym = symbols[0] if symbols else "?"
            tf = timeframes[0] if timeframes else "?"
            print_model_report(mk, name, stats, symbol=sym, timeframe=tf)

    # ── Comparison table ──
    print_comparison_table(trades, signals)

    # ── Trade logs per model ──
    for mk in model_keys:
        print_trade_log(trades, model_key=mk)

    # ── Per-symbol breakdown if multiple symbols ──
    if len(symbols) > 1:
        print()
        print_separator("═")
        print("  ANALYSE PAR SYMBOLE")
        print_separator("═")
        for sym in symbols:
            sym_trades = [t for t in trades if t.get("symbol") == sym]
            sym_signals = [s for s in signals if s.get("_symbol") == sym]
            print(f"\n  ─── {sym} ───")
            for mk in model_keys:
                mk_trades = [t for t in sym_trades if t["model_key"] == mk]
                mk_signals = [s for s in sym_signals if s.get("model_key") == mk]
                stats = compute_model_stats(mk_trades, mk_signals, mk)
                if stats:
                    print_model_report(mk, model_names.get(mk, mk), stats, symbol=sym)


if __name__ == "__main__":
    main()