"""
Analyse de logs backtest JSONL pour identifier patterns et sessions difficiles.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import statistics

def parse_logs(filepath):
    """Parse un fichier JSONL et retourne les evenements par session."""
    sessions = defaultdict(lambda: {
        "signals": [],
        "trades": [],
        "equity": [],
        "info": {}
    })
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = ev.get("session_id", "unknown")
            t = ev.get("type")
            if t == "session_start":
                sessions[sid]["info"] = ev
            elif t == "signal":
                sessions[sid]["signals"].append(ev)
            elif t == "trade":
                sessions[sid]["trades"].append(ev)
            elif t == "equity":
                sessions[sid]["equity"].append(ev)
    return sessions

def extract_hour(ts):
    try:
        if isinstance(ts, str) and len(ts) >= 13:
            return int(ts[11:13])
    except Exception:
        pass
    return None

def analyze_by_hour(trades):
    """Analyse des trades par heure."""
    hours = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0, "count": 0})
    for t in trades:
        if t.get("action") != "close":
            continue
        h = extract_hour(t.get("timestamp"))
        if h is None:
            continue
        pnl = t.get("pnl", 0)
        hours[h]["count"] += 1
        hours[h]["pnl"] += pnl
        if pnl > 0:
            hours[h]["wins"] += 1
        else:
            hours[h]["losses"] += 1
    return hours

def analyze_by_day(trades):
    """Analyse des trades par jour."""
    days = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0, "count": 0})
    for t in trades:
        if t.get("action") != "close":
            continue
        d = extract_day(t.get("timestamp"))
        if d is None:
            continue
        pnl = t.get("pnl", 0)
        days[d]["count"] += 1
        days[d]["pnl"] += pnl
        if pnl > 0:
            days[d]["wins"] += 1
        else:
            days[d]["losses"] += 1
    return days

def analyze_signals_by_hour(signals):
    hours = defaultdict(int)
    for s in signals:
        h = extract_hour(s.get("timestamp"))
        if h is not None:
            hours[h] += 1
    return hours

def extract_day(ts):
    try:
        if isinstance(ts, str) and len(ts) >= 10:
            return ts[:10]
    except Exception:
        pass
    return None

def analyze_session(session_id, data):
    info = data["info"]
    signals = data["signals"]
    trades = data["trades"]
    equity = data["equity"]

    closed_trades = [t for t in trades if t.get("action") == "close"]
    pnls = [t.get("pnl", 0) for t in closed_trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_trades = len(closed_trades)
    win_rate = len(wins) / total_trades * 100 if total_trades else 0
    total_pnl = sum(pnls)
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0
    profit_factor = abs(sum(wins) / sum(losses)) if sum(losses) != 0 else float('inf')

    # Evolution equity
    max_dd = 0
    peak = 0
    final_equity = None
    for e in equity:
        eq = e.get("equity", 0)
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd
        final_equity = eq

    # Analyse par heure
    hour_stats = analyze_by_hour(trades)
    day_stats = analyze_by_day(trades)
    signal_hours = analyze_signals_by_hour(signals)

    return {
        "session_id": session_id,
        "model_key": info.get("model_key", "?"),
        "symbol": info.get("symbol", "?"),
        "timeframe": info.get("timeframe", "?"),
        "date": info.get("timestamp", "")[:10] if isinstance(info.get("timestamp"), str) else "?",
        "total_signals": len(signals),
        "total_trades": total_trades,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
        "final_equity": final_equity,
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "hour_stats": hour_stats,
        "day_stats": day_stats,
        "signal_hours": signal_hours,
    }

def print_hour_table(hour_stats, total_signals_by_hour=None):
    print(f"{'Heure':<8} {'Trades':<8} {'Wins':<8} {'Losses':<8} {'WR%':<8} {'PnL':<12} {'Avg PnL':<10}")
    print("-" * 70)
    for h in sorted(hour_stats.keys()):
        s = hour_stats[h]
        wr = s["wins"] / s["count"] * 100 if s["count"] else 0
        avg = s["pnl"] / s["count"] if s["count"] else 0
        sig = total_signals_by_hour.get(h, "") if total_signals_by_hour else ""
        sig_str = f" (sig:{sig})" if sig else ""
        print(f"{h:02d}h     {s['count']:<8} {s['wins']:<8} {s['losses']:<8} {wr:<8.1f} {s['pnl']:<12.2f} {avg:<10.2f}{sig_str}")

def print_day_table(day_stats):
    print(f"{'Jour':<12} {'Trades':<8} {'Wins':<8} {'Losses':<8} {'WR%':<8} {'PnL':<12} {'Avg PnL':<10}")
    print("-" * 70)
    for d in sorted(day_stats.keys()):
        s = day_stats[d]
        wr = s["wins"] / s["count"] * 100 if s["count"] else 0
        avg = s["pnl"] / s["count"] if s["count"] else 0
        print(f"{d:<12} {s['count']:<8} {s['wins']:<8} {s['losses']:<8} {wr:<8.1f} {s['pnl']:<12.2f} {avg:<10.2f}")

def main():
    files = [
        Path("webui/logs/webui_logs_backtest_kronos-mini_XAUUSD_M1_2025-10-01.jsonl"),
        Path("webui/logs/backtest_kronos-base_XAUUSD_M1_2026-03-02.jsonl"),
        Path("webui/logs/backtest_kronos-base_EURUSD_M1_2026-03-02.jsonl"),
        Path("webui/logs/backtest_kronos-mini_EURUSD_M1_2026-03-04.jsonl"),
    ]
    all_sessions = []
    for fp in files:
        if not fp.exists():
            print(f"Fichier non trouve: {fp}")
            continue
        print(f"\n{'='*70}")
        print(f"Analyse de: {fp.name}")
        print(f"{'='*70}")
        sessions = parse_logs(fp)
        print(f"Sessions trouvees: {len(sessions)}")

        results = []
        for sid, data in sessions.items():
            r = analyze_session(sid, data)
            results.append(r)
            all_sessions.append(r)

        for r in results:
            print(f"\n--- Session {r['session_id']} ---")
            print(f"Modele: {r['model_key']}, Symbole: {r['symbol']}, Periode: {r['date']}")
            print(f"Trades: {r['total_trades']}, WR: {r['win_rate']:.1f}%, PnL: {r['total_pnl']:.2f}, DD: {r['max_drawdown']:.2f}")
            print(f"Avg Win: {r['avg_win']:.2f}, Avg Loss: {r['avg_loss']:.2f}, PF: {r['profit_factor']:.2f}")

            print(f"\n--- Analyse par heure ---")
            print_hour_table(r["hour_stats"], r["signal_hours"])

            print(f"\n--- Analyse par jour (top 10 pires) ---")
            worst_days = sorted(r["day_stats"].items(), key=lambda x: x[1]["pnl"])[:10]
            print(f"{'Jour':<12} {'Trades':<8} {'Wins':<8} {'Losses':<8} {'WR%':<8} {'PnL':<12}")
            print("-" * 60)
            for d, s in worst_days:
                wr = s["wins"] / s["count"] * 100 if s["count"] else 0
                print(f"{d:<12} {s['count']:<8} {s['wins']:<8} {s['losses']:<8} {wr:<8.1f} {s['pnl']:<12.2f}")

            # Identifier les patterns
            print(f"\n--- Insights ---")
            bad_hours = [(h, s) for h, s in r["hour_stats"].items() if s["count"] >= 10 and s["wins"] / s["count"] < 0.45]
            if bad_hours:
                bad_hours.sort(key=lambda x: x[1]["pnl"])
                print("Heures a eviter (WR < 45% et >=10 trades):")
                for h, s in bad_hours[:5]:
                    wr = s["wins"] / s["count"] * 100
                    print(f"  {h:02d}h: WR={wr:.1f}%, PnL={s['pnl']:.2f}, {s['count']} trades")
            else:
                print("Aucune heure avec WR < 45% (>=10 trades).")

    # TABLEAU COMPARATIF GLOBAL
    print(f"\n{'='*70}")
    print("TABLEAU COMPARATIF GLOBAL XAUUSD vs EURUSD")
    print(f"{'='*70}")

    group_by = {}
    for s in all_sessions:
        key = (s["model_key"], s["symbol"])
        if key not in group_by:
            group_by[key] = []
        group_by[key].append(s)

    print(f"\n{'Modele':<20} {'Symbole':<10} {'Trades':<10} {'WR%':<10} {'PnL':<12} {'DD':<12} {'PF':<8}")
    print("-" * 80)
    for (model, symbol), arr in sorted(group_by.items()):
        total_pnl = sum(s["total_pnl"] for s in arr)
        total_trades = sum(s["total_trades"] for s in arr)
        total_wins = sum(s["winning_trades"] for s in arr)
        total_dd = sum(s["max_drawdown"] for s in arr)
        wr_list = [s["win_rate"] for s in arr if s["total_trades"] > 0]
        avg_wr = statistics.mean(wr_list) if wr_list else 0
        global_wr = total_wins / total_trades * 100 if total_trades else 0
        pf_list = [s["profit_factor"] for s in arr if s["profit_factor"] != float('inf')]
        avg_pf = statistics.mean(pf_list) if pf_list else 0
        print(f"{model:<20} {symbol:<10} {total_trades:<10} {avg_wr:<10.1f} {total_pnl:<12.2f} {total_dd:<12.2f} {avg_pf:<8.2f}")

    # HEURES A EVITER - COMPARATIF
    print(f"\n{'='*70}")
    print("HEURES A EVITER PAR PAIRE")
    print(f"{'='*70}")
    for (model, symbol), arr in sorted(group_by.items()):
        print(f"\n{model} - {symbol}:")
        global_hours = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0, "count": 0})
        for s in arr:
            for h, stat in s["hour_stats"].items():
                global_hours[h]["count"] += stat["count"]
                global_hours[h]["wins"] += stat["wins"]
                global_hours[h]["losses"] += stat["losses"]
                global_hours[h]["pnl"] += stat["pnl"]

        bad = [(h, s) for h, s in global_hours.items() if s["count"] >= 10 and s["wins"] / s["count"] < 0.45]
        if bad:
            bad.sort(key=lambda x: x[1]["pnl"])
            print(f"{'Heure':<8} {'Trades':<8} {'WR%':<8} {'PnL':<12}")
            print("-" * 40)
            for h, s in bad[:8]:
                wr = s["wins"] / s["count"] * 100
                print(f"{h:02d}h     {s['count']:<8} {wr:<8.1f} {s['pnl']:<12.2f}")
        else:
            print("  Aucune heure critique (WR < 45%)")

if __name__ == "__main__":
    main()
