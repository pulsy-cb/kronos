#!/usr/bin/env python3
"""
Analyse de la degradation temporelle des modeles Kronos.
Verifie si les decisions regressent au fil du temps (fatigue du modele).

Usage:
  python scripts/analyze_temporal_decay.py webui/logs/live_XAUUSD_M1_2026-04-21.jsonl
  python scripts/analyze_temporal_decay.py webui/logs/live_XAUUSD_M1_2026-04-14.jsonl
  python scripts/analyze_temporal_decay.py webui/logs/live_XAUUSD_M1_*.jsonl
"""

import json
import sys
import glob
import os
from collections import defaultdict
from datetime import datetime, timedelta

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


# ── Data loading ──────────────────────────────────────────────────────────

def load_events(filepath):
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


def load_and_tag(filepath):
    fname = os.path.basename(filepath)
    parts = fname.replace("live_", "").replace(".jsonl", "").split("_")
    symbol = parts[0] if len(parts) >= 1 else "?"
    tf = parts[1] if len(parts) >= 2 else "?"
    date = parts[2] if len(parts) >= 3 else "?"

    events = load_events(filepath)
    for e in events:
        e["_file"] = fname
        e["_symbol"] = symbol
        e["_timeframe"] = tf
        e["_date"] = date
    return events, date, symbol


def parse_trades(events):
    opens = {}
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
                continue
            close_ev = e
            pnl = close_ev.get("pnl", 0.0)
            direction = close_ev.get("direction", open_ev.get("direction", "?"))
            try:
                t_open = datetime.fromisoformat(open_ev["timestamp"])
                t_close = datetime.fromisoformat(close_ev["timestamp"])
                duration_min = (t_close - t_open).total_seconds() / 60.0
                close_hour = t_close
            except (ValueError, KeyError):
                duration_min = 0
                close_hour = None

            trades.append({
                "ticket": ticket,
                "model_key": open_ev.get("model_key", "?"),
                "model_name": open_ev.get("model_name", "?"),
                "symbol": open_ev.get("_symbol", "?"),
                "direction": direction,
                "entry_price": open_ev.get("price", 0),
                "exit_price": close_ev.get("price", 0),
                "volume": open_ev.get("volume", 0),
                "pnl": pnl,
                "reason": close_ev.get("reason", "?"),
                "duration_min": round(duration_min, 2),
                "close_time": close_hour,
                "close_ts": close_ev.get("timestamp", ""),
            })
    return trades


# ── Temporal analysis ───────────────────────────────────────────────────────

def analyze_hourly_decay(trades, model_key, date_str):
    """Split trades by hour, compute stats per hour."""
    model_trades = [t for t in trades if t["model_key"] == model_key and t["close_time"] is not None]
    if not model_trades:
        return None

    # Group by hour
    hourly = defaultdict(list)
    for t in model_trades:
        hour = t["close_time"].hour
        hourly[hour].append(t)

    results = {}
    for hour in sorted(hourly.keys()):
        ht = hourly[hour]
        pnls = [t["pnl"] for t in ht]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        results[hour] = {
            "count": len(ht),
            "pnl": round(sum(pnls), 2),
            "avg_pnl": round(sum(pnls) / len(pnls), 4) if pnls else 0,
            "win_rate": round(len(wins) / len(ht) * 100, 1) if ht else 0,
            "best": round(max(pnls), 2) if pnls else 0,
            "worst": round(min(pnls), 2) if pnls else 0,
            "wins": len(wins),
            "losses": len(losses),
        }
    return results


def analyze_rolling_window(trades, model_key, window=10):
    """Compute rolling win rate and PnL over a sliding window of N trades."""
    model_trades = sorted(
        [t for t in trades if t["model_key"] == model_key and t["close_time"] is not None],
        key=lambda t: t["close_time"]
    )
    if len(model_trades) < window:
        return []

    rolling = []
    for i in range(len(model_trades) - window + 1):
        batch = model_trades[i:i+window]
        pnls = [t["pnl"] for t in batch]
        wins = [p for p in pnls if p > 0]
        rolling.append({
            "start_idx": i + 1,
            "end_idx": i + window,
            "start_time": batch[0]["close_time"].strftime("%H:%M"),
            "end_time": batch[-1]["close_time"].strftime("%H:%M"),
            "win_rate": round(len(wins) / window * 100, 1),
            "total_pnl": round(sum(pnls), 2),
            "avg_pnl": round(sum(pnls) / window, 4),
        })
    return rolling


def analyze_halves(trades, model_key):
    """Compare first half vs second half of trading session."""
    model_trades = sorted(
        [t for t in trades if t["model_key"] == model_key and t["close_time"] is not None],
        key=lambda t: t["close_time"]
    )
    if len(model_trades) < 4:
        return None

    mid = len(model_trades) // 2
    first = model_trades[:mid]
    second = model_trades[mid:]

    def half_stats(half_trades):
        pnls = [t["pnl"] for t in half_trades]
        wins = [p for p in pnls if p > 0]
        return {
            "count": len(half_trades),
            "pnl": round(sum(pnls), 2),
            "avg_pnl": round(sum(pnls) / len(pnls), 4) if pnls else 0,
            "win_rate": round(len(wins) / len(half_trades) * 100, 1),
            "best": round(max(pnls), 2),
            "worst": round(min(pnls), 2),
        }

    return {
        "first_half": half_stats(first),
        "second_half": half_stats(second),
        "first_time": f"{first[0]['close_time'].strftime('%H:%M')} - {first[-1]['close_time'].strftime('%H:%M')}",
        "second_time": f"{second[0]['close_time'].strftime('%H:%M')} - {second[-1]['close_time'].strftime('%H:%M')}",
    }


def analyze_quarters(trades, model_key):
    """Split session into 4 quarters for finer analysis."""
    model_trades = sorted(
        [t for t in trades if t["model_key"] == model_key and t["close_time"] is not None],
        key=lambda t: t["close_time"]
    )
    if len(model_trades) < 8:
        return None

    q_size = len(model_trades) // 4
    quarters = []
    for i in range(4):
        start = i * q_size
        end = (i + 1) * q_size if i < 3 else len(model_trades)
        qt = model_trades[start:end]
        pnls = [t["pnl"] for t in qt]
        wins = [p for p in pnls if p > 0]
        quarters.append({
            "label": f"Q{i+1}",
            "trades": len(qt),
            "time_range": f"{qt[0]['close_time'].strftime('%H:%M')}-{qt[-1]['close_time'].strftime('%H:%M')}",
            "pnl": round(sum(pnls), 2),
            "avg_pnl": round(sum(pnls) / len(pnls), 4),
            "win_rate": round(len(wins) / len(qt) * 100, 1),
        })
    return quarters


def analyze_consecutive_loss_streaks(trades, model_key):
    """Find where loss streaks cluster in time."""
    model_trades = sorted(
        [t for t in trades if t["model_key"] == model_key and t["close_time"] is not None],
        key=lambda t: t["close_time"]
    )
    streaks = []
    current_streak = []
    for t in model_trades:
        if t["pnl"] < 0:
            current_streak.append(t)
        else:
            if len(current_streak) >= 3:
                streaks.append({
                    "length": len(current_streak),
                    "time_start": current_streak[0]["close_time"].strftime("%H:%M"),
                    "time_end": current_streak[-1]["close_time"].strftime("%H:%M"),
                    "pnl": round(sum(t["pnl"] for t in current_streak), 2),
                })
            current_streak = []
    # Check final streak
    if len(current_streak) >= 3:
        streaks.append({
            "length": len(current_streak),
            "time_start": current_streak[0]["close_time"].strftime("%H:%M"),
            "time_end": current_streak[-1]["close_time"].strftime("%H:%M"),
            "pnl": round(sum(t["pnl"] for t in current_streak), 2),
        })
    return streaks


# ── Display ─────────────────────────────────────────────────────────────────

def print_hourly_table(hourly, model_name, date_str):
    if not hourly:
        return
    print(f"\n  === DEGRADATION HORAIRE — {model_name} — {date_str} ===\n")
    print(f"  {'Heure':>6} {'Trades':>6} {'Win%':>7} {'PnL':>9} {'AvgPnL':>9} {'Best':>7} {'Worst':>7}")
    print(f"  {'------':>6} {'------':>6} {'-----':>7} {'---------':>9} {'-------':>9} {'-----':>7} {'------':>7}")

    for hour in sorted(hourly.keys()):
        h = hourly[hour]
        print(f"  {hour:>6}h {h['count']:>6} {h['win_rate']:>6.1f}% {h['pnl']:>+9.2f} {h['avg_pnl']:>+9.4f} {h['best']:>+7.2f} {h['worst']:>+7.2f}")

    # Trend indicator
    hours = sorted(hourly.keys())
    if len(hours) >= 2:
        first_3_avg = []
        last_3_avg = []
        for h in hours[:3]:
            first_3_avg.append(hourly[h]["win_rate"])
        for h in hours[-3:]:
            last_3_avg.append(hourly[h]["win_rate"])

        early_wr = sum(first_3_avg) / len(first_3_avg)
        late_wr = sum(last_3_avg) / len(last_3_avg)

        early_pnl = sum(hourly[h]["pnl"] for h in hours[:3])
        late_pnl = sum(hourly[h]["pnl"] for h in hours[-3:])

        print()
        if late_wr < early_wr - 10:
            print(f"  >>> DEGRADATION CONFIRMEE: Win rate debut={early_wr:.1f}% -> fin={late_wr:.1f}% (delta={late_wr-early_wr:+.1f}%)")
        elif late_wr > early_wr + 10:
            print(f"  >>> AMELIORATION: Win rate debut={early_wr:.1f}% -> fin={late_wr:.1f}% (delta={late_wr-early_wr:+.1f}%)")
        else:
            print(f"  >>> STABLE: Win rate debut={early_wr:.1f}% -> fin={late_wr:.1f}% (delta={late_wr-early_wr:+.1f}%)")

        if late_pnl < early_pnl:
            print(f"  >>> PnL debut={early_pnl:+.2f}$ -> fin={late_pnl:+.2f}$ (degradation={early_pnl-late_pnl:+.2f}$)")
        else:
            print(f"  >>> PnL debut={early_pnl:+.2f}$ -> fin={late_pnl:+.2f}$ (amelioration={late_pnl-early_pnl:+.2f}$)")


def print_rolling_table(rolling, model_name, window):
    if not rolling:
        return
    print(f"\n  === ROLLING WIN RATE (fenetre={window} trades) — {model_name} ===\n")
    print(f"  {'Trades':>12} {'Plage horaire':>16} {'Win%':>7} {'PnL cumul':>10} {'AvgPnL':>9}")
    print(f"  {'----------':>12} {'--------------':>16} {'-----':>7} {'---------':>10} {'-------':>9}")

    # Sample every N rows to avoid wall of text
    step = max(1, len(rolling) // 25)
    shown = set()
    for i, r in enumerate(rolling):
        if i % step == 0 or i == len(rolling) - 1 or r["win_rate"] <= 20 or r["win_rate"] >= 80:
            key = (r["start_idx"], r["end_idx"])
            if key not in shown:
                shown.add(key)
                print(f"  #{r['start_idx']:>4}-#{r['end_idx']:<4} {r['start_time']:>5}-{r['end_time']:<5} {r['win_rate']:>6.1f}% {r['total_pnl']:>+10.2f} {r['avg_pnl']:>+9.4f}")

    # Find lowest and highest rolling win rates
    min_wr = min(rolling, key=lambda r: r["win_rate"])
    max_wr = max(rolling, key=lambda r: r["win_rate"])
    print()
    print(f"  Pire fenetre:  #{min_wr['start_idx']}-#{min_wr['end_idx']} ({min_wr['start_time']}-{min_wr['end_time']}) Win={min_wr['win_rate']}% PnL={min_wr['total_pnl']:+.2f}$")
    print(f"  Meilleure fenetre: #{max_wr['start_idx']}-#{max_wr['end_idx']} ({max_wr['start_time']}-{max_wr['end_time']}) Win={max_wr['win_rate']}% PnL={max_wr['total_pnl']:+.2f}$")


def print_halves_comparison(halves, model_name, date_str):
    if not halves:
        return
    print(f"\n  === PREMIERE MOITIE vs DEUXIEME MOITIE — {model_name} — {date_str} ===\n")
    f = halves["first_half"]
    s = halves["second_half"]

    print(f"  {'Metrique':<18} {'1ere moitie':>14} {'2eme moitie':>14} {'Delta':>10}")
    print(f"  {'--------':<18} {'-----------':>14} {'-----------':>14} {'-----':>10}")
    print(f"  {'Periode':<18} {halves['first_time']:>14} {halves['second_time']:>14} {'':>10}")
    print(f"  {'Trades':<18} {f['count']:>14} {s['count']:>14} {s['count']-f['count']:>+10}")
    print(f"  {'PnL total':<18} {f['pnl']:>+14.2f} {s['pnl']:>+14.2f} {s['pnl']-f['pnl']:>+10.2f}")
    print(f"  {'PnL moyen/trade':<18} {f['avg_pnl']:>+14.4f} {s['avg_pnl']:>+14.4f} {s['avg_pnl']-f['avg_pnl']:>+10.4f}")
    print(f"  {'Win rate':<18} {f['win_rate']:>13.1f}% {s['win_rate']:>13.1f}% {s['win_rate']-f['win_rate']:>+9.1f}%")
    print(f"  {'Meilleur trade':<18} {f['best']:>+14.2f} {s['best']:>+14.2f}")
    print(f"  {'Pire trade':<18} {f['worst']:>+14.2f} {s['worst']:>+14.2f}")

    # Verdict
    print()
    if s["win_rate"] < f["win_rate"] - 8 and s["avg_pnl"] < f["avg_pnl"]:
        print(f"  *** DEGRADATION NETTE en 2eme moitie: -{f['win_rate']-s['win_rate']:.1f}% win rate, PnL moyen {s['avg_pnl']-f['avg_pnl']:+.4f}")
    elif s["win_rate"] > f["win_rate"] + 8:
        print(f"  *** AMELIORATION en 2eme moitie: +{s['win_rate']-f['win_rate']:.1f}% win rate")
    else:
        print(f"  *** PAS DE DEGRADATION SIGNIFICATIVE (delta win rate={s['win_rate']-f['win_rate']:+.1f}%)")


def print_quarters(quarters, model_name, date_str):
    if not quarters:
        return
    print(f"\n  === ANALYSE PAR QUARTS — {model_name} — {date_str} ===\n")
    print(f"  {'Quart':>5} {'Trades':>6} {'Plage':>14} {'Win%':>7} {'PnL':>9} {'AvgPnL':>9}")
    print(f"  {'-----':>5} {'------':>6} {'----':>14} {'-----':>7} {'---------':>9} {'-------':>9}")

    for q in quarters:
        print(f"  {q['label']:>5} {q['trades']:>6} {q['time_range']:>14} {q['win_rate']:>6.1f}% {q['pnl']:>+9.2f} {q['avg_pnl']:>+9.4f}")

    # Trend from Q1 to Q4
    wrs = [q["win_rate"] for q in quarters]
    if len(wrs) >= 2 and wrs[-1] < wrs[0] - 10:
        print(f"\n  >>> TENDANCE BAISSIERE: Q1={wrs[0]:.1f}% -> Q4={wrs[-1]:.1f}%")
    elif len(wrs) >= 2 and wrs[-1] > wrs[0] + 10:
        print(f"\n  >>> TENDANCE HAUSSIERE: Q1={wrs[0]:.1f}% -> Q4={wrs[-1]:.1f}%")
    else:
        print(f"\n  >>> PAS DE TENDANCE CLAIRE: Q1={wrs[0]:.1f}% -> Q4={wrs[-1]:.1f}%")


def print_loss_streaks(streaks, model_name, date_str):
    if not streaks:
        print(f"\n  Aucune serie de 3+ pertes consecutives pour {model_name}")
        return
    print(f"\n  === SERIES DE PERTES (3+) — {model_name} — {date_str} ===\n")
    print(f"  {'Debut':>6} {'Fin':>6} {'Longueur':>9} {'PnL total':>10}")
    print(f"  {'-----':>6} {'---':>6} {'--------':>9} {'---------':>10}")
    for s in streaks:
        print(f"  {s['time_start']:>6} {s['time_end']:>6} {s['length']:>9} {s['pnl']:>+10.2f}")


def print_cross_day_summary(all_results):
    """Print a summary comparing temporal decay across all days."""
    print("\n")
    print("=" * 70)
    print("  RESUME CROSS-JOUR — LA DEGRADATION EST-ELLE SYSTEMATIQUE ?")
    print("=" * 70)

    for model_key in sorted(all_results.keys()):
        print(f"\n  {model_key}:")
        print(f"  {'Date':<12} {'Q1 WR%':>8} {'Q2 WR%':>8} {'Q3 WR%':>8} {'Q4 WR%':>8} {'1er PnL':>9} {'2eme PnL':>9} {'Verdict':>12}")
        print(f"  {'----':<12} {'------':>8} {'------':>8} {'------':>8} {'------':>8} {'-------':>9} {'-------':>9} {'-------':>12}")

        for date_str, data in sorted(all_results[model_key].items()):
            q = data.get("quarters")
            h = data.get("halves")
            if not q or not h:
                continue

            q_wrs = [q[i]["win_rate"] if i < len(q) else 0 for i in range(4)]
            f_pnl = h["first_half"]["pnl"]
            s_pnl = h["second_half"]["pnl"]

            # Determine verdict
            if q_wrs[-1] < q_wrs[0] - 10 and s_pnl < f_pnl:
                verdict = "DEGRADATION"
            elif q_wrs[-1] > q_wrs[0] + 10:
                verdict = "AMELIORATION"
            else:
                verdict = "STABLE"

            print(f"  {date_str:<12} {q_wrs[0]:>7.1f}% {q_wrs[1]:>7.1f}% {q_wrs[2]:>7.1f}% {q_wrs[3]:>7.1f}% {f_pnl:>+9.2f} {s_pnl:>+9.2f} {verdict:>12}")

        # Count how many days show degradation
        days_degradation = sum(
            1 for date_str, data in all_results[model_key].items()
            if data.get("halves") and data["halves"]["second_half"]["win_rate"] < data["halves"]["first_half"]["win_rate"] - 8
            and data["halves"]["second_half"]["avg_pnl"] < data["halves"]["first_half"]["avg_pnl"]
        )
        days_total = sum(
            1 for date_str, data in all_results[model_key].items()
            if data.get("halves") is not None
        )

        if days_total > 0:
            pct = days_degradation / days_total * 100
            print(f"\n  >>> {model_key}: DEGRADATION dans {days_degradation}/{days_total} jours ({pct:.0f}%)")
            if pct >= 60:
                print(f"  >>> CONCLUSION: DEGRADATION CONFIRMEE pour {model_key} — ce n'est pas une impression!")
            elif pct >= 40:
                print(f"  >>> CONCLUSION: DEGRADATION PARTIELLE pour {model_key} — tendancielle mais pas systematique")
            else:
                print(f"  >>> CONCLUSION: PAS DE DEGRADATION SYSTEMATIQUE pour {model_key}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else "webui/logs/live_XAUUSD_M1_*.jsonl"

    files = sorted(glob.glob(pattern))
    if not files:
        print(f"Aucun fichier pour: {pattern}")
        sys.exit(1)

    print(f"Analyse de degradation temporelle sur {len(files)} fichier(s)")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    all_results = defaultdict(dict)  # model_key -> {date: {halves, quarters, ...}}

    for filepath in files:
        events, date_str, symbol = load_and_tag(filepath)
        trades = parse_trades(events)

        model_keys = sorted(set(t["model_key"] for t in trades))

        print(f"\n{'='*70}")
        print(f"  DATE: {date_str} | {symbol} | {len(trades)} trades")
        print(f"{'='*70}")

        for mk in model_keys:
            model_trades = [t for t in trades if t["model_key"] == mk]
            if len(model_trades) < 4:
                print(f"\n  {mk}: seulement {len(model_trades)} trades — skip")
                continue

            model_name = model_trades[0].get("model_name", mk)

            # Hourly decay
            hourly = analyze_hourly_decay(trades, mk, date_str)
            print_hourly_table(hourly, model_name, date_str)

            # Halves
            halves = analyze_halves(trades, mk)
            print_halves_comparison(halves, model_name, date_str)

            # Quarters
            quarters = analyze_quarters(trades, mk)
            print_quarters(quarters, model_name, date_str)

            # Loss streaks
            streaks = analyze_consecutive_loss_streaks(trades, mk)
            print_loss_streaks(streaks, model_name, date_str)

            # Rolling window
            rolling = analyze_rolling_window(trades, mk, window=10)
            print_rolling_table(rolling, model_name, window=10)

            # Store for cross-day summary
            all_results[mk][date_str] = {
                "halves": halves,
                "quarters": quarters,
                "hourly": hourly,
            }

    # Cross-day summary
    if len(files) > 1:
        print_cross_day_summary(dict(all_results))


if __name__ == "__main__":
    main()