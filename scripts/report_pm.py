#!/usr/bin/env python3
"""
Rapport quotidien Polymarket Paper Trading.
Analye les logs et envoie un résumé Telegram.
"""

import os, sys, glob, re, json, subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path

KRONOS_DIR = "/root/kronos"
LOGS_DIR = os.path.join(KRONOS_DIR, "logs")

def tail_log(filepath, n=200):
    try:
        with open(filepath) as f:
            lines = f.readlines()
        return lines[-n:]
    except Exception:
        return []

def extract_summary(lines):
    """Extrait le dernier SUMMARY et les derniers SETTLED."""
    summary = {}
    recent_settled = []
    for line in reversed(lines):
        if "SUMMARY" in line and not summary:
            # 2026-04-23 23:55:30 | INFO    |   SUMMARY | Bets: 301 | Wins: 157 | Accuracy: 52.2% | P&L: +12.06€ | ROI: +4.0%
            m = re.search(r'Bets:\s*(\d+).*Wins:\s*(\d+).*Accuracy:\s*([\d.]+)%.*P&L:\s*([\-+\d.]+)€.*ROI:\s*([\-+\d.]+)%', line)
            if m:
                summary = {
                    "bets": int(m.group(1)),
                    "wins": int(m.group(2)),
                    "accuracy": float(m.group(3)),
                    "pnl": float(m.group(4)),
                    "roi": float(m.group(5)),
                }
        if "SETTLED" in line and len(recent_settled) < 5:
            # 2026-04-24 00:20:02 | INFO    |   SETTLED | + Up   | €+0.98 | Entry 78224.01 -> Exit 78360.27 | Actual: UP
            m = re.search(r'SETTLED \|\s*([\+\-])\s*(\w+).*€([\-+\d.]+).*Actual:\s*(\w+)', line)
            if m:
                recent_settled.append({
                    "icon": m.group(1),
                    "side": m.group(2),
                    "profit": float(m.group(3)),
                    "actual": m.group(4),
                })
        if summary and len(recent_settled) >= 5:
            break
    return summary, list(reversed(recent_settled))

def main():
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y%m%d")

    # Find all active log files
    log_files = glob.glob(os.path.join(LOGS_DIR, "kronos_*.log"))
    log_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)

    report_lines = [
        f"🤖 *Rapport Polymarket Paper Trading*",
        f"📅 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
    ]

    if not log_files:
        report_lines.append("⚠️ Aucun fichier log trouvé.")
        print("\n".join(report_lines))
        return

    for log_file in log_files[:4]:  # Max 4 bots
        filename = os.path.basename(log_file)
        # kronos_xaumodel-mini_BTCUSDT_M5_20260424_003305.log
        parts = filename.replace("kronos_", "").replace(".log", "").split("_")
        if len(parts) >= 4:
            model = parts[0]
            symbol = parts[1]
            tf = parts[2]
        else:
            model, symbol, tf = "?", "?", "?"

        lines = tail_log(log_file, 300)
        summary, settled = extract_summary(lines)

        report_lines.append(f"*{symbol}* | modèle `{model}` | TF {tf}")
        if summary:
            report_lines.append(
                f"  📊 Bets: `{summary['bets']}` | Wins: `{summary['wins']}` | "
                f"Accuracy: `{summary['accuracy']:.1f}%` | "
                f"P&L: `{summary['pnl']:+.2f}€` | ROI: `{summary['roi']:+.1f}%`"
            )
        else:
            report_lines.append("  ⏳ En cours de démarrage...")

        if settled:
            report_lines.append("  🔴 Derniers trades:")
            for s in settled:
                icon = "✅" if s["icon"] == "+" else "❌"
                report_lines.append(f"    {icon} {s['side']} → Actual: {s['actual']} | {s['profit']:+.2f}€")
        report_lines.append("")

    message = "\n".join(report_lines)
    print(message)

    # Send to Telegram if hermes gateway is available
    # The cron will auto-deliver back to this chat since we use 'origin'

if __name__ == "__main__":
    main()
