#!/usr/bin/env python3
"""
Rapport quotidien Polymarket Paper Trading.
Analyse les logs et envoie un résumé Telegram.
"""

import os, sys, glob, re
from datetime import datetime, timezone

KRONOS_DIR = "/root/kronos"
LOGS_DIR = os.path.join(KRONOS_DIR, "logs")


def tail(filepath, n=600):
    try:
        with open(filepath) as f:
            return f.readlines()[-n:]
    except Exception:
        return []


def parse_log(filepath):
    """Extrait le dernier SUMMARY et les 5 derniers SETTLED."""
    lines = tail(filepath, 700)
    summary = None
    recent_settled = []
    cycle = None
    for line in reversed(lines):
        if summary is None and "SUMMARY" in line:
            m = re.search(
                r'Bets:\s*(\d+).*Wins:\s*(\d+).*Accuracy:\s*([\d.]+)%.*P&L:\s*([\-+\d.]+).*ROI:\s*([\-+\d.]+)%',
                line,
            )
            if m:
                summary = {
                    "bets": int(m.group(1)),
                    "wins": int(m.group(2)),
                    "accuracy": float(m.group(3)),
                    "pnl": float(m.group(4)),
                    "roi": float(m.group(5)),
                }
        if len(recent_settled) < 5 and "SETTLED" in line and "[PM] Settled" not in line:
            m = re.search(
                r'SETTLED\s*\|\s*([\+\-])\s*(Up|Down).*€([\-+\d.]+).*Actual:\s*(UP|DOWN)',
                line,
            )
            if m:
                recent_settled.append(
                    {
                        "icon": m.group(1),
                        "side": m.group(2),
                        "profit": float(m.group(3)),
                        "actual": m.group(4),
                    }
                )
        if cycle is None and "Cycle" in line:
            m = re.search(r'\[Cycle\s*(\d+)\]', line)
            if m:
                cycle = int(m.group(1))
        if summary and len(recent_settled) >= 5 and cycle is not None:
            break
    return summary, list(reversed(recent_settled)), cycle


def get_latest_logs():
    """Retourne le fichier le plus récent pour chaque (model, paire, timeframe)."""
    log_files = glob.glob(os.path.join(LOGS_DIR, "kronos_*.log"))
    groups = {}
    for f in log_files:
        name = os.path.basename(f)
        # kronos_xaumodel-mini_BTCUSDT_M5_20260424_003409.log
        parts = name.replace("kronos_", "").replace(".log", "").split("_")
        if len(parts) >= 5 and "M5" in parts[2]:
            key = tuple(parts[:3])
            groups.setdefault(key, []).append((f, os.path.getmtime(f)))
    return {k: max(v, key=lambda x: x[1])[0] for k, v in groups.items()}


def main():
    latest = get_latest_logs()

    expected = [
        ("xaumodel-mini", "BTCUSDT", "M5"),
        ("xaumodel-mini", "ETHUSDT", "M5"),
        ("xaumodel-local", "BTCUSDT", "M5"),
        ("xaumodel-local", "ETHUSDT", "M5"),
    ]

    active_count = sum(1 for k in expected if k in latest)

    report_lines = [
        "🤖 *Rapport Polymarket Paper Trading*",
        f"📅 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"🖥️ Processus actifs : {active_count}/4",
        "",
    ]

    if not latest:
        report_lines.append("⚠️ Aucun fichier log trouvé.")
        print("\n".join(report_lines))
        return

    for key in expected:
        filepath = latest.get(key)
        model, symbol, tf = key

        if not filepath:
            report_lines.append(f"*{symbol}* | modèle `{model}` | TF {tf}")
            report_lines.append("  ⏳ En attente de données… bot probablement mort.")
            report_lines.append("")
            continue

        summary, settled, cycle = parse_log(filepath)

        report_lines.append(f"*{symbol}* | modèle `{model}` | TF {tf}")
        if summary:
            report_lines.append(
                f"  📊 Bets: `{summary['bets']}` | Wins: `{summary['wins']}` | "
                f"Accuracy: `{summary['accuracy']:.1f}%` | "
                f"P&L: `{summary['pnl']:+.2f}€` | ROI: `{summary['roi']:+.1f}%` | "
                f"Cycle `{cycle or '?'}`"
            )
        else:
            report_lines.append("  ⏳ En cours de démarrage…")

        if settled:
            report_lines.append("  🔴 Derniers trades:")
            for s in settled:
                icon = "✅" if s["icon"] == "+" else "❌"
                report_lines.append(
                    f"    {icon} {s['side']} → Actual: {s['actual']} | {s['profit']:+.2f}€"
                )
        report_lines.append("")

    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
