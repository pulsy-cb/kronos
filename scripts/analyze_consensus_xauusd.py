"""
Analyse de consensus entre kronos-base et kronos-mini sur XAUUSD (periode 03/2026).
Objectif: voir si les trades ou les deux modeles sont d'accord ont un meilleur WR.
"""
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def parse_logs(filepath, model_key_filter=None):
    """Parse un fichier JSONL et retourne signaux et trades par session."""
    sessions = defaultdict(lambda: {"signals": [], "trades": []})
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if model_key_filter and ev.get("model_key") != model_key_filter:
                continue
            sid = ev.get("session_id", "unknown")
            t = ev.get("type")
            if t == "signal":
                sessions[sid]["signals"].append(ev)
            elif t == "trade":
                sessions[sid]["trades"].append(ev)
    return sessions

def build_signal_trade_map(signals, trades, period_start="2026-03-01", period_end="2026-04-01"):
    """
    Associe chaque signal d'entree au PnL du trade resultant.
    Retourne un dict: {timestamp_iso: {"signal": "long"/"short"/"neutral", "pnl": float or None}}
    """
    filtered_signals = []
    for s in signals:
        ts = s.get("timestamp", "")
        if isinstance(ts, str) and len(ts) >= 10:
            if period_start <= ts[:10] < period_end:
                filtered_signals.append(s)

    filtered_signals.sort(key=lambda x: x.get("timestamp", ""))
    trades_sorted = sorted(trades, key=lambda x: x.get("timestamp", ""))

    opens = [t for t in trades_sorted if t.get("action") == "open"]
    closes = [t for t in trades_sorted if t.get("action") == "close"]

    open_queue = opens[:]
    trade_pnls = []

    for c in closes:
        for i, o in enumerate(open_queue):
            if o.get("direction") == c.get("direction"):
                entry_ts = o.get("timestamp", c.get("timestamp"))
                trade_pnls.append({
                    "entry_ts": entry_ts,
                    "direction": o.get("direction"),
                    "pnl": c.get("pnl", 0),
                    "reason": c.get("reason", "unknown"),
                    "close_ts": c.get("timestamp"),
                })
                open_queue.pop(i)
                break

    signal_map = {}
    sig_idx = 0
    for tp in sorted(trade_pnls, key=lambda x: x["entry_ts"]):
        last_entry_signal = None
        last_entry_ts = None
        while sig_idx < len(filtered_signals):
            s = filtered_signals[sig_idx]
            s_ts = s.get("timestamp", "")
            if s_ts > tp["entry_ts"]:
                break
            sig = s.get("signal", "neutral")
            if sig in ("long", "short"):
                last_entry_signal = sig
                last_entry_ts = s_ts
            sig_idx += 1

        if last_entry_ts:
            if last_entry_ts not in signal_map:
                signal_map[last_entry_ts] = {
                    "signal": last_entry_signal,
                    "pnl": tp["pnl"],
                    "direction": tp["direction"],
                    "reason": tp["reason"],
                }

    for s in filtered_signals:
        ts = s.get("timestamp", "")
        sig = s.get("signal", "neutral")
        if ts not in signal_map:
            signal_map[ts] = {
                "signal": sig,
                "pnl": None,
                "direction": None,
                "reason": None,
            }

    return signal_map

def direction_from_signal(signal):
    if signal in ("long",):
        return "long"
    elif signal in ("short",):
        return "short"
    else:
        return "neutral"

def extract_hour(ts):
    try:
        if isinstance(ts, str) and len(ts) >= 13:
            return int(ts[11:13])
    except Exception:
        pass
    return None

def main():
    base_file = Path("webui/logs/backtest_kronos-base_XAUUSD_M1_2026-03-02.jsonl")
    mini_file = Path("webui/logs/webui_logs_backtest_kronos-mini_XAUUSD_M1_2025-10-01.jsonl")

    print("=" * 75)
    print("ANALYSE CONSENSUS BASE vs MINI sur XAUUSD (Mars 2026)")
    print("=" * 75)

    base_sessions = parse_logs(base_file, "kronos-base")
    mini_sessions = parse_logs(mini_file, "kronos-mini")

    base_data = list(base_sessions.values())[0] if base_sessions else {"signals": [], "trades": []}
    mini_data = list(mini_sessions.values())[0] if mini_sessions else {"signals": [], "trades": []}

    base_map = build_signal_trade_map(base_data["signals"], base_data["trades"])
    mini_map = build_signal_trade_map(mini_data["signals"], mini_data["trades"])

    common_ts = sorted(set(base_map.keys()) & set(mini_map.keys()))
    print(f"\nTimestamps communs en Mars 2026: {len(common_ts)}")
    print(f"Trades du base mappes: {sum(1 for v in base_map.values() if v['pnl'] is not None)}")
    print(f"Trades du mini mappes: {sum(1 for v in mini_map.values() if v['pnl'] is not None)}")

    # Analyse par categorie de consensus (base comme reference)
    categories = {
        "both_long": {"count": 0, "trades": 0, "wins": 0, "pnl": 0.0, "hour_stats": defaultdict(lambda: {"count": 0, "trades": 0, "wins": 0, "pnl": 0.0})},
        "both_short": {"count": 0, "trades": 0, "wins": 0, "pnl": 0.0, "hour_stats": defaultdict(lambda: {"count": 0, "trades": 0, "wins": 0, "pnl": 0.0})},
        "both_neutral": {"count": 0, "trades": 0, "wins": 0, "pnl": 0.0, "hour_stats": defaultdict(lambda: {"count": 0, "trades": 0, "wins": 0, "pnl": 0.0})},
        "base_long_mini_neutral": {"count": 0, "trades": 0, "wins": 0, "pnl": 0.0, "hour_stats": defaultdict(lambda: {"count": 0, "trades": 0, "wins": 0, "pnl": 0.0})},
        "base_neutral_mini_long": {"count": 0, "trades": 0, "wins": 0, "pnl": 0.0, "hour_stats": defaultdict(lambda: {"count": 0, "trades": 0, "wins": 0, "pnl": 0.0})},
        "other": {"count": 0, "trades": 0, "wins": 0, "pnl": 0.0, "hour_stats": defaultdict(lambda: {"count": 0, "trades": 0, "wins": 0, "pnl": 0.0})},
    }

    for ts in common_ts:
        b = base_map[ts]
        m = mini_map[ts]
        b_dir = direction_from_signal(b["signal"])
        m_dir = direction_from_signal(m["signal"])

        if b_dir == "long" and m_dir == "long":
            cat = "both_long"
        elif b_dir == "short" and m_dir == "short":
            cat = "both_short"
        elif b_dir == "neutral" and m_dir == "neutral":
            cat = "both_neutral"
        elif b_dir == "long" and m_dir == "neutral":
            cat = "base_long_mini_neutral"
        elif b_dir == "neutral" and m_dir == "long":
            cat = "base_neutral_mini_long"
        else:
            cat = "other"

        h = extract_hour(ts)
        categories[cat]["count"] += 1
        categories[cat]["hour_stats"][h]["count"] += 1

        # On prend le PnL du base comme reference
        if b["pnl"] is not None:
            categories[cat]["trades"] += 1
            categories[cat]["pnl"] += b["pnl"]
            categories[cat]["hour_stats"][h]["trades"] += 1
            categories[cat]["hour_stats"][h]["pnl"] += b["pnl"]
            if b["pnl"] > 0:
                categories[cat]["wins"] += 1
                categories[cat]["hour_stats"][h]["wins"] += 1

    # Stats globales
    base_total_trades = sum(1 for v in base_map.values() if v["pnl"] is not None)
    base_total_wins = sum(1 for v in base_map.values() if v["pnl"] is not None and v["pnl"] > 0)
    base_total_pnl = sum(v["pnl"] for v in base_map.values() if v["pnl"] is not None)
    base_wr = base_total_wins / base_total_trades * 100 if base_total_trades else 0

    print(f"\n{'='*75}")
    print("METRIQUES GLOBALES DU BASE (Mars 2026)")
    print(f"{'='*75}")
    print(f"Trades total: {base_total_trades}, Wins: {base_total_wins}, WR: {base_wr:.1f}%, PnL: {base_total_pnl:.2f}")

    print(f"\n{'='*75}")
    print("COMPARATIF CONSENSUS (Base comme reference)")
    print(f"{'='*75}")
    print(f"{'Categorie':<30} {'Signaux':<10} {'Trades':<10} {'Wins':<10} {'WR%':<10} {'PnL':<12}")
    print("-" * 75)

    total_consensus_trades = 0
    total_consensus_wins = 0
    total_consensus_pnl = 0.0

    for cat_key, cat_label in [
        ("both_long", "Both Long (consensus)"),
        ("both_short", "Both Short (consensus)"),
        ("base_long_mini_neutral", "Base Long, Mini Neutre"),
        ("base_neutral_mini_long", "Base Neutre, Mini Long"),
        ("both_neutral", "Both Neutral"),
        ("other", "Autre"),
    ]:
        c = categories[cat_key]
        wr = c["wins"] / c["trades"] * 100 if c["trades"] else 0
        print(f"{cat_label:<30} {c['count']:<10} {c['trades']:<10} {c['wins']:<10} {wr:<10.1f} {c['pnl']:<12.2f}")
        if cat_key in ("both_long", "both_short"):
            total_consensus_trades += c["trades"]
            total_consensus_wins += c["wins"]
            total_consensus_pnl += c["pnl"]

    consensus_wr = total_consensus_wins / total_consensus_trades * 100 if total_consensus_trades else 0

    print(f"\n{'='*75}")
    print("RESUME STRICT CONSENSUS (Both Long + Both Short)")
    print(f"{'='*75}")
    print(f"Trades en consensus:           {total_consensus_trades}")
    print(f"Wins en consensus:             {total_consensus_wins}")
    print(f"Win Rate consensus:            {consensus_wr:.1f}%")
    print(f"PnL total consensus:           {total_consensus_pnl:.2f}")
    print(f"\nBase WR global:                {base_wr:.1f}%")
    print(f"Amelioration WR consensus:     {consensus_wr - base_wr:+.1f} points")

    # Analyse par heure
    print(f"\n{'='*75}")
    print("COMPARATIF HORAIRE: Both Long vs Base Long/Mini Neutre")
    print(f"{'='*75}")
    print(f"{'Heure':<8} {'BL-Trades':<12} {'BL-WR%':<10} {'BL-PnL':<12} {'BLN-Trades':<12} {'BLN-WR%':<10} {'BLN-PnL':<12} {'Delta':<10}")
    print("-" * 90)

    bl = categories["both_long"]["hour_stats"]
    bln = categories["base_long_mini_neutral"]["hour_stats"]

    all_hours = sorted(set(bl.keys()) | set(bln.keys()))
    for h in all_hours:
        s_bl = bl.get(h, {"count": 0, "trades": 0, "wins": 0, "pnl": 0.0})
        s_bln = bln.get(h, {"count": 0, "trades": 0, "wins": 0, "pnl": 0.0})
        wr_bl = s_bl["wins"] / s_bl["trades"] * 100 if s_bl["trades"] else 0
        wr_bln = s_bln["wins"] / s_bln["trades"] * 100 if s_bln["trades"] else 0
        delta = wr_bln - wr_bl
        print(f"{h:02d}h     {s_bl['trades']:<12} {wr_bl:<10.1f} {s_bl['pnl']:<12.2f} {s_bln['trades']:<12} {wr_bln:<10.1f} {s_bln['pnl']:<12.2f} {delta:+.1f}")

    # Calcul des totaux par heure
    print(f"\n{'='*75}")
    print("HEURES A EVITER / PRIVILEGIER (Consensus vs Solo)")
    print(f"{'='*75}")

    print("\nHeures ou le consensus est particulierement mauvais (BL WR < 40%):")
    for h in sorted(bl.keys()):
        s = bl[h]
        if s["trades"] >= 10:
            wr = s["wins"] / s["trades"] * 100
            if wr < 40:
                print(f"  {h:02d}h: WR={wr:.1f}%, PnL={s['pnl']:.2f}, {s['trades']} trades")

    print("\nHeures ou Base Long / Mini Neutre est meilleur (BLN WR > 45%):")
    for h in sorted(bln.keys()):
        s = bln[h]
        if s["trades"] >= 5:
            wr = s["wins"] / s["trades"] * 100
            if wr > 45:
                print(f"  {h:02d}h: WR={wr:.1f}%, PnL={s['pnl']:.2f}, {s['trades']} trades")

    # Conclusion
    print(f"\n{'='*75}")
    print("CONCLUSION")
    print(f"{'='*75}")

    solo_trades = categories["base_long_mini_neutral"]["trades"]
    solo_wins = categories["base_long_mini_neutral"]["wins"]
    solo_pnl = categories["base_long_mini_neutral"]["pnl"]
    solo_wr = solo_wins / solo_trades * 100 if solo_trades else 0

    print(f"Base Long + Mini Neutre (Base 'solo'):")
    print(f"  Trades: {solo_trades}, Wins: {solo_wins}, WR: {solo_wr:.1f}%, PnL: {solo_pnl:.2f}")
    print(f"\nBoth Long (Consensus):")
    print(f"  Trades: {total_consensus_trades}, Wins: {total_consensus_wins}, WR: {consensus_wr:.1f}%, PnL: {total_consensus_pnl:.2f}")
    print(f"\nVerdict: Le consensus {'ameliore' if consensus_wr > solo_wr else 'degrade'} le WR de {abs(consensus_wr - solo_wr):.1f} points.")
    print(f"         Le consensus transforme un PnL {'positif' if solo_pnl > 0 else 'negatif'} en un PnL {'positif' if total_consensus_pnl > 0 else 'negatif'}.")

if __name__ == "__main__":
    main()
