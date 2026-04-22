#!/usr/bin/env python3
"""
Analyse croisee des signaux de modeles en parallelle.
Explore comment le desaccord/accord entre modeles peut etre exploite.

Usage: python scripts/analyze_model_disagreement.py [log_file]
"""

import json
import sys
import glob
import os
from collections import defaultdict, Counter
from datetime import datetime, timedelta

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


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


def parse_trades(events):
    opens_by_ticket = {}
    orphan_opens = []
    closes_no_ticket = []
    trades = []

    for e in events:
        if e.get("type") != "trade":
            continue
        ticket = e.get("ticket")
        action = e.get("action")
        if ticket is not None:
            if action == "open":
                opens_by_ticket[ticket] = e
            elif action == "close":
                open_ev = opens_by_ticket.pop(ticket, None)
                if open_ev is None:
                    continue
                pnl = e.get("pnl", 0.0)
                direction = e.get("direction", open_ev.get("direction", "?"))
                duration_min = 0
                close_hour = None
                try:
                    t_open = datetime.fromisoformat(open_ev["timestamp"])
                    t_close = datetime.fromisoformat(e["timestamp"])
                    duration_min = (t_close - t_open).total_seconds() / 60.0
                    close_hour = t_close
                except (ValueError, KeyError):
                    pass
                trades.append({
                    "ticket": ticket,
                    "model_key": open_ev.get("model_key", "?"),
                    "direction": direction,
                    "entry_price": open_ev.get("price", 0),
                    "exit_price": e.get("price", 0),
                    "pnl": pnl,
                    "reason": e.get("reason", "?"),
                    "duration_min": round(duration_min, 2),
                    "open_ts": open_ev.get("timestamp", ""),
                    "close_ts": e.get("timestamp", ""),
                    "close_time": close_hour,
                })
        else:
            if action == "open":
                orphan_opens.append(e)
            elif action == "close":
                closes_no_ticket.append(e)

    for close_ev in closes_no_ticket:
        mk = close_ev.get("model_key", "?")
        sid = close_ev.get("session_id", "?")
        for i, open_ev in enumerate(orphan_opens):
            if open_ev.get("model_key", "?") == mk and open_ev.get("session_id", "?") == sid:
                orphan_opens.pop(i)
                pnl = close_ev.get("pnl", 0.0)
                direction = close_ev.get("direction", open_ev.get("direction", "?"))
                duration_min = 0
                close_hour = None
                try:
                    t_open = datetime.fromisoformat(open_ev["timestamp"])
                    t_close = datetime.fromisoformat(close_ev["timestamp"])
                    duration_min = (t_close - t_open).total_seconds() / 60.0
                    close_hour = t_close
                except (ValueError, KeyError):
                    pass
                trades.append({
                    "ticket": None,
                    "model_key": mk,
                    "direction": direction,
                    "entry_price": open_ev.get("price", 0),
                    "exit_price": close_ev.get("price", 0),
                    "pnl": pnl,
                    "reason": close_ev.get("reason", "?"),
                    "duration_min": round(duration_min, 2),
                    "open_ts": open_ev.get("timestamp", ""),
                    "close_ts": close_ev.get("timestamp", ""),
                    "close_time": close_hour,
                })
                break

    return trades


def match_signals_by_time(signals_a, signals_b, tolerance_seconds=60):
    """
    Match signals from two models by timestamp within a tolerance window.
    Returns list of (sig_a, sig_b) pairs.
    """
    matched = []
    used_b = set()

    # Sort by timestamp
    sa = sorted(signals_a, key=lambda s: s["timestamp"])
    sb = sorted(signals_b, key=lambda s: s["timestamp"])

    for sig_a in sa:
        try:
            ta = datetime.fromisoformat(sig_a["timestamp"])
        except ValueError:
            continue

        best_match = None
        best_dt = timedelta(seconds=tolerance_seconds + 1)

        for j, sig_b in enumerate(sb):
            if j in used_b:
                continue
            try:
                tb = datetime.fromisoformat(sig_b["timestamp"])
            except ValueError:
                continue

            dt = abs(ta - tb)
            if dt <= timedelta(seconds=tolerance_seconds) and dt < best_dt:
                best_match = (j, sig_b)
                best_dt = dt

        if best_match:
            j, sig_b = best_match
            matched.append((sig_a, sig_b, best_dt.total_seconds()))
            used_b.add(j)

    return matched


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else "webui/logs/live_XAUUSD_M1_*.jsonl"
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"Aucun fichier pour: {pattern}")
        sys.exit(1)

    for filepath in files:
        fname = os.path.basename(filepath)
        date_part = fname.replace("live_", "").replace(".jsonl", "").split("_")[-1]
        print(f"\n{'='*70}")
        print(f"  FICHIER: {fname}")
        print(f"{'='*70}")

        events = load_events(filepath)
        signals = [e for e in events if e.get("type") == "signal"]
        trades = parse_trades(events)

        # Group signals by model
        model_signals = defaultdict(list)
        for s in signals:
            model_signals[s["model_key"]].append(s)

        model_keys = sorted(model_signals.keys())
        print(f"Modeles: {model_keys}")
        for mk in model_keys:
            sigs = model_signals[mk]
            dirs = Counter(s.get("signal", "?") for s in sigs)
            print(f"  {mk}: {len(sigs)} signals — {dict(dirs)}")

        # Find models that ran in parallel (overlap in time)
        model_time_ranges = {}
        for mk in model_keys:
            sigs = model_signals[mk]
            if len(sigs) >= 2:
                try:
                    t_start = datetime.fromisoformat(sigs[0]["timestamp"])
                    t_end = datetime.fromisoformat(sigs[-1]["timestamp"])
                    model_time_ranges[mk] = (t_start, t_end)
                except ValueError:
                    pass

        # Pair all model combinations
        parallel_pairs = []
        for i, mk_a in enumerate(model_keys):
            for mk_b in model_keys[i+1:]:
                if mk_a in model_time_ranges and mk_b in model_time_ranges:
                    a_start, a_end = model_time_ranges[mk_a]
                    b_start, b_end = model_time_ranges[mk_b]
                    overlap_start = max(a_start, b_start)
                    overlap_end = min(a_end, b_end)
                    if overlap_end > overlap_start:
                        parallel_pairs.append((mk_a, mk_b, overlap_start, overlap_end))

        if not parallel_pairs:
            print("\n  Aucune paire de modeles en parallelle.")
            continue

        for mk_a, mk_b, overlap_start, overlap_end in parallel_pairs:
            print(f"\n  {'─'*60}")
            print(f"  PAIRE: {mk_a} vs {mk_b}")
            print(f"  Overlap: {overlap_start.strftime('%H:%M')} - {overlap_end.strftime('%H:%M')}")
            print(f"  {'─'*60}")

            # Filter signals to overlap period
            sigs_a = [s for s in model_signals[mk_a]
                      if overlap_start.isoformat() <= s["timestamp"] <= overlap_end.isoformat()]
            sigs_b = [s for s in model_signals[mk_b]
                      if overlap_start.isoformat() <= s["timestamp"] <= overlap_end.isoformat()]

            if not sigs_a or not sigs_b:
                print("  Pas de signaux dans la periode overlap")
                continue

            # Match signals by time (60s tolerance)
            matched = match_signals_by_time(sigs_a, sigs_b, tolerance_seconds=60)
            print(f"\n  Signaux apparies (tolerance 60s): {len(matched)}")

            if not matched:
                continue

            # ── Agreement analysis ───────────────────────────────────
            agreement_stats = {
                "both_long": 0, "both_short": 0, "both_neutral": 0,
                "a_long_b_neutral": 0, "a_short_b_neutral": 0,
                "a_neutral_b_long": 0, "a_neutral_b_short": 0,
                "disagree_dir": 0,  # one long, other short
                "both_exit": 0,
            }

            matched_details = []
            for sig_a, sig_b, dt_sec in matched:
                sa = sig_a.get("signal", "neutral")
                sb = sig_b.get("signal", "neutral")
                pra = sig_a.get("predicted_return", 0)
                prb = sig_b.get("predicted_return", 0)

                # Classify the pair
                if sa == "long" and sb == "long":
                    category = "both_long"
                elif sa == "short" and sb == "short":
                    category = "both_short"
                elif sa == "neutral" and sb == "neutral":
                    category = "both_neutral"
                elif sa == "exit" and sb == "exit":
                    category = "both_exit"
                elif sa == "long" and sb == "short" or sa == "short" and sb == "long":
                    category = "disagree_dir"
                elif sa in ("long", "short") and sb == "neutral":
                    category = f"a_{sa}_b_neutral"
                elif sa == "neutral" and sb in ("long", "short"):
                    category = f"a_neutral_b_{sb}"
                elif sa == "exit" or sb == "exit":
                    # One says exit, other says something
                    category = "one_exit"
                else:
                    category = "other"

                if category in agreement_stats:
                    agreement_stats[category] += 1
                else:
                    agreement_stats[category] = agreement_stats.get(category, 0) + 1

                matched_details.append({
                    "timestamp": sig_a["timestamp"],
                    "dt_sec": round(dt_sec, 1),
                    "sig_a": sa,
                    "sig_b": sb,
                    "pr_a": pra,
                    "pr_b": prb,
                    "category": category,
                })

            # Print agreement breakdown
            total = len(matched)
            agree = agreement_stats["both_long"] + agreement_stats["both_short"] + agreement_stats["both_neutral"]
            disagree = agreement_stats["disagree_dir"]

            print(f"\n  ┌─ ACCORD / DESACCORD ──────────────────────────────┐")
            print(f"  │  Total paires:           {total:>6}                     │")
            print(f"  │  Les deux d'accord:      {agree:>6} ({agree/total*100:.1f}%)               │")
            print(f"  │    both long:            {agreement_stats['both_long']:>6}                     │")
            print(f"  │    both short:           {agreement_stats['both_short']:>6}                     │")
            print(f"  │    both neutral:         {agreement_stats['both_neutral']:>6}                     │")
            print(f"  │  Direction opposee:      {disagree:>6} ({disagree/total*100:.1f}%)               │")
            print(f"  │  Un actif, autre neutre:                                │")
            for cat in sorted(agreement_stats.keys()):
                if "neutral" in cat:
                    print(f"  │    {cat:<22} {agreement_stats[cat]:>6}                     │")
            print(f"  └─────────────────────────────────────────────────────┘")

            # ── Cross-reference with trades ───────────────────────────
            # For each model, find trades that happened after each signal type
            trades_by_model = defaultdict(list)
            for t in trades:
                trades_by_model[t["model_key"]].append(t)

            # Map: signal timestamp -> nearest trade
            def find_trade_after_signal(sig_ts, model_trades, max_lag_sec=120):
                """Find the first trade opened after a signal, within max_lag_sec."""
                try:
                    t_sig = datetime.fromisoformat(sig_ts)
                except ValueError:
                    return None

                for t in model_trades:
                    try:
                        t_open = datetime.fromisoformat(t["open_ts"])
                        dt = (t_open - t_sig).total_seconds()
                        if 0 <= dt <= max_lag_sec:
                            return t
                    except (ValueError, KeyError):
                        pass
                return None

            # ── STRATEGY ANALYSIS ───────────────────────────────────────
            print(f"\n  ┌─ ANALYSE STRATEGIQUE ─────────────────────────────┐")

            # Strategy 1: Trade ONLY when both agree on direction
            both_agree_trades_a = []
            both_agree_trades_b = []
            for md in matched_details:
                if md["category"] in ("both_long", "both_short"):
                    for mk, trade_list in [("a", trades_by_model[mk_a]), ("b", trades_by_model[mk_b])]:
                        t = find_trade_after_signal(md["timestamp"], trade_list)
                        if t:
                            if mk == "a":
                                both_agree_trades_a.append(t)
                            else:
                                both_agree_trades_b.append(t)

            # Strategy 2: Trade when one is active, other is neutral (confident move)
            one_active_trades = []
            for md in matched_details:
                if "neutral" in md["category"]:
                    # The active model's signal
                    active_mk = mk_a if md["sig_a"] in ("long", "short") else mk_b
                    t = find_trade_after_signal(md["timestamp"], trades_by_model[active_mk])
                    if t:
                        one_active_trades.append((md["category"], t))

            # Strategy 3: Skip when they disagree on direction
            disagree_trades = []
            for md in matched_details:
                if md["category"] == "disagree_dir":
                    for mk, trade_list in [("a", trades_by_model[mk_a]), ("b", trades_by_model[mk_b])]:
                        t = find_trade_after_signal(md["timestamp"], trade_list)
                        if t:
                            disagree_trades.append((mk, t))

            # Print results for each strategy
            def print_trade_stats(label, trade_list):
                if not trade_list:
                    print(f"  │  {label:<40} 0 trades  │")
                    return
                pnls = [t["pnl"] for t in trade_list]
                wins = [p for p in pnls if p > 0]
                wr = len(wins) / len(pnls) * 100 if pnls else 0
                total_pnl = sum(pnls)
                avg = total_pnl / len(pnls) if pnls else 0
                print(f"  │  {label:<20} {len(pnls):>4} trades  WR={wr:>5.1f}%  PnL={total_pnl:>+8.2f}  Avg={avg:>+7.4f} │")

            print_trade_stats(f"Both agree ({mk_a})", both_agree_trades_a)
            print_trade_stats(f"Both agree ({mk_b})", both_agree_trades_b)

            # One active, other neutral breakdown
            by_cat = defaultdict(list)
            for cat, t in one_active_trades:
                by_cat[cat].append(t)
            for cat in sorted(by_cat.keys()):
                print_trade_stats(f"One active: {cat}", by_cat[cat])

            # Disagreement trades
            by_mk = defaultdict(list)
            for mk, t in disagree_trades:
                by_mk[mk].append(t)
            for mk in sorted(by_mk.keys()):
                print_trade_stats(f"Disagree dir ({mk})", by_mk[mk])

            print(f"  └─────────────────────────────────────────────────────┘")

            # ── Predicted return magnitude analysis ────────────────────
            print(f"\n  ┌─ MAGNITUDE DES PREDICTIONS ────────────────────────┐")
            for md in matched_details[:50]:  # Show sample
                if md["category"] not in ("both_neutral",):
                    pass  # skip neutrals for readability

            # Average predicted return by category
            pr_by_cat = defaultdict(list)
            for md in matched_details:
                pr_by_cat[md["category"]].append((md["pr_a"], md["pr_b"]))

            for cat in sorted(pr_by_cat.keys()):
                pairs = pr_by_cat[cat]
                avg_a = sum(p[0] for p in pairs) / len(pairs)
                avg_b = sum(p[1] for p in pairs) / len(pairs)
                # Also show std
                std_a = (sum((p[0] - avg_a)**2 for p in pairs) / len(pairs)) ** 0.5
                std_b = (sum((p[1] - avg_b)**2 for p in pairs) / len(pairs)) ** 0.5
                print(f"  │  {cat:<22} avg_a={avg_a:+.5f} std_a={std_a:.5f}  avg_b={avg_b:+.5f} std_b={std_b:.5f} │")
            print(f"  └─────────────────────────────────────────────────────┘")

            # ── Time gap between signals ───────────────────────────────
            print(f"\n  ┌─ ECART TEMPOREL ENTRE SIGNAUX ──────────────────────┐")
            dt_vals = [md["dt_sec"] for md in matched_details]
            if dt_vals:
                avg_dt = sum(dt_vals) / len(dt_vals)
                max_dt = max(dt_vals)
                min_dt = min(dt_vals)
                under_1s = sum(1 for d in dt_vals if d <= 1.0)
                under_5s = sum(1 for d in dt_vals if d <= 5.0)
                under_30s = sum(1 for d in dt_vals if d <= 30.0)
                print(f"  │  Avg ecart:    {avg_dt:>6.1f}s                           │")
                print(f"  │  Min ecart:    {min_dt:>6.1f}s                           │")
                print(f"  │  Max ecart:    {max_dt:>6.1f}s                           │")
                print(f"  │  < 1s:         {under_1s:>4} ({under_1s/len(dt_vals)*100:.1f}%)                   │")
                print(f"  │  < 5s:         {under_5s:>4} ({under_5s/len(dt_vals)*100:.1f}%)                   │")
                print(f"  │  < 30s:        {under_30s:>4} ({under_30s/len(dt_vals)*100:.1f}%)                   │")
            print(f"  └─────────────────────────────────────────────────────┘")

            # ── Detailed signal-by-signal for active signals ────────────
            active_pairs = [md for md in matched_details if md["category"] not in ("both_neutral",)]
            if active_pairs:
                print(f"\n  ┌─ DETAIL DES SIGNAUX ACTIFS (non-neutral) ──────────┐")
                print(f"  │  {'Heure':>6} {'A':>6} {'B':>6} {'Cat':>20} {'PR_A':>9} {'PR_B':>9} │")
                print(f"  │  {'-----':>6} {'-----':>6} {'-----':>6} {'---':>20} {'-------':>9} {'-------':>9} │")
                for md in active_pairs[:40]:
                    ts = md["timestamp"][11:16] if len(md["timestamp"]) > 16 else md["timestamp"]
                    print(f"  │  {ts:>6} {md['sig_a']:>6} {md['sig_b']:>6} {md['category']:>20} {md['pr_a']:>+9.5f} {md['pr_b']:>+9.5f} │")
                if len(active_pairs) > 40:
                    print(f"  │  ... et {len(active_pairs)-40} autres                                    │")
                print(f"  └─────────────────────────────────────────────────────┘")

            # ── CONSENSUS SIGNAL QUALITY ────────────────────────────────
            # For signals where both agree on direction, check if the trade won
            print(f"\n  ┌─ QUALITE DU CONSENSUS ─────────────────────────────┐")
            for direction in ["both_long", "both_short"]:
                dir_signals = [md for md in matched_details if md["category"] == direction]
                if not dir_signals:
                    continue

                # Check trades from BOTH models after these consensus signals
                consensus_trades = []
                for md in dir_signals:
                    for mk in [mk_a, mk_b]:
                        t = find_trade_after_signal(md["timestamp"], trades_by_model[mk], max_lag_sec=120)
                        if t:
                            consensus_trades.append(t)

                if consensus_trades:
                    pnls = [t["pnl"] for t in consensus_trades]
                    wins = [p for p in pnls if p > 0]
                    wr = len(wins) / len(pnls) * 100
                    print(f"  │  {direction:<15} {len(pnls):>4} trades  WR={wr:>5.1f}%  PnL={sum(pnls):>+8.2f} │")

            # For signals where one is active and other neutral
            for cat in ["a_long_b_neutral", "a_short_b_neutral", "a_neutral_b_long", "a_neutral_b_short"]:
                cat_signals = [md for md in matched_details if md["category"] == cat]
                if not cat_signals:
                    continue

                # The active model
                active_mk = mk_a if "a_" in cat.split("_")[0] + "_" + cat.split("_")[1] else mk_b
                if cat.startswith("a_") and cat.endswith("_neutral"):
                    active_mk = mk_a
                elif cat.startswith("a_neutral_b_"):
                    active_mk = mk_b
                else:
                    continue

                cat_trades = []
                for md in cat_signals:
                    t = find_trade_after_signal(md["timestamp"], trades_by_model[active_mk], max_lag_sec=120)
                    if t:
                        cat_trades.append(t)

                if cat_trades:
                    pnls = [t["pnl"] for t in cat_trades]
                    wins = [p for p in pnls if p > 0]
                    wr = len(wins) / len(pnls) * 100 if pnls else 0
                    print(f"  │  {cat:<22} {len(pnls):>4} trades  WR={wr:>5.1f}%  PnL={sum(pnls):>+8.2f} │")

            # Direction disagreement
            disagree_signals = [md for md in matched_details if md["category"] == "disagree_dir"]
            if disagree_signals:
                dis_trades = []
                for md in disagree_signals:
                    for mk in [mk_a, mk_b]:
                        t = find_trade_after_signal(md["timestamp"], trades_by_model[mk], max_lag_sec=120)
                        if t:
                            dis_trades.append(t)
                if dis_trades:
                    pnls = [t["pnl"] for t in dis_trades]
                    wins = [p for p in pnls if p > 0]
                    wr = len(wins) / len(pnls) * 100 if pnls else 0
                    print(f"  │  {'disagree_dir':<22} {len(pnls):>4} trades  WR={wr:>5.1f}%  PnL={sum(pnls):>+8.2f} │")

            print(f"  └─────────────────────────────────────────────────────┘")

            # ── EXIT SIGNAL ANALYSIS ───────────────────────────────────
            print(f"\n  ┌─ SIGNAUX DE SORTIE ───────────────────────────────┐")
            exit_pairs = [md for md in matched_details if md["sig_a"] == "exit" or md["sig_b"] == "exit"]
            both_exit = [md for md in matched_details if md["sig_a"] == "exit" and md["sig_b"] == "exit"]
            one_exit = [md for md in exit_pairs if md not in both_exit]
            print(f"  │  Both exit:     {len(both_exit):>4}                                │")
            print(f"  │  One exit only: {len(one_exit):>4}                                │")

            # Check if exit signals led to profitable closes
            if exit_pairs:
                exit_trades = []
                for md in exit_pairs:
                    for mk in [mk_a, mk_b]:
                        t = find_trade_after_signal(md["timestamp"], trades_by_model[mk], max_lag_sec=180)
                        if t and t["reason"] == "signal_exit":
                            exit_trades.append(t)
                if exit_trades:
                    pnls = [t["pnl"] for t in exit_trades]
                    wins = [p for p in pnls if p > 0]
                    wr = len(wins) / len(pnls) * 100 if pnls else 0
                    print(f"  │  Exit trades:   {len(pnls):>4}  WR={wr:>5.1f}%  PnL={sum(pnls):>+8.2f} │")
            print(f"  └─────────────────────────────────────────────────────┘")

            # ── KEY INSIGHT ─────────────────────────────────────────────
            print(f"\n  ┌─ INSIGHTS CLES ───────────────────────────────────┐")
            # Compare win rate: consensus vs solo vs disagreement
            all_trades_a = trades_by_model[mk_a]
            all_trades_b = trades_by_model[mk_b]

            for mk, all_t in [(mk_a, all_trades_a), (mk_b, all_trades_b)]:
                if all_t:
                    pnls = [t["pnl"] for t in all_t]
                    wins = [p for p in pnls if p > 0]
                    wr = len(wins) / len(pnls) * 100 if pnls else 0
                    print(f"  │  {mk:<22} solo WR={wr:>5.1f}%  ({len(pnls)} trades)     │")

            # Consensus trades (from matched_details where both agree)
            consensus_all_trades = []
            for md in matched_details:
                if md["category"] in ("both_long", "both_short"):
                    for mk in [mk_a, mk_b]:
                        t = find_trade_after_signal(md["timestamp"], trades_by_model[mk], max_lag_sec=120)
                        if t:
                            consensus_all_trades.append(t)
            if consensus_all_trades:
                pnls = [t["pnl"] for t in consensus_all_trades]
                wins = [p for p in pnls if p > 0]
                wr = len(wins) / len(pnls) * 100 if pnls else 0
                print(f"  │  {'CONSENSUS (both agree)':<22} WR={wr:>5.1f}%  ({len(pnls)} trades)     │")

            # One active, other neutral
            solo_active_trades = []
            for md in matched_details:
                if "neutral" in md["category"]:
                    active_mk = mk_a if md["sig_a"] in ("long", "short") else mk_b
                    t = find_trade_after_signal(md["timestamp"], trades_by_model[active_mk], max_lag_sec=120)
                    if t:
                        solo_active_trades.append(t)
            if solo_active_trades:
                pnls = [t["pnl"] for t in solo_active_trades]
                wins = [p for p in pnls if p > 0]
                wr = len(wins) / len(pnls) * 100 if pnls else 0
                print(f"  │  {'SOLO (other neutral)':<22} WR={wr:>5.1f}%  ({len(pnls)} trades)     │")

            print(f"  └─────────────────────────────────────────────────────┘")


if __name__ == "__main__":
    main()