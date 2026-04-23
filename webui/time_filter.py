"""
Time filter pour le trading live et backtest.
Filtre les signaux selon le symbole et l'heure.
"""

from datetime import datetime, time

# Plages horaires par symbole (UTC)
# Format: liste de tuples (heure_debut, heure_fin) autorisées
# Si vide = pas de restriction
# Si non defini = pas de restriction
TIME_RULES = {
    # XAUUSD: exclure 16h-21h, 12h-14h, 09h30-11h
    # Donc garder: 02h-09h30, 11h-12h, 14h-16h, 21h-23h59
    "XAUUSD": [
        (time(2, 0), time(9, 30)),
        (time(11, 0), time(12, 0)),
        (time(14, 0), time(16, 0)),
        (time(21, 0), time(23, 59)),
    ],
    # Forex (EURUSD, GBPUSD, etc.): 07h30-19h uniquement
    # On utilise un pattern par defaut pour les paires forex non XAUUSD
    "FOREX": [
        (time(7, 30), time(19, 0)),
    ],
}


def is_trading_allowed(timestamp, symbol, custom_rules=None):
    """
    Verifie si le trading est autorise a un timestamp donne pour un symbole.

    Args:
        timestamp: str (ISO) ou datetime
        symbol: str, ex: "XAUUSD", "EURUSD", "EURUSD.i"
        custom_rules: dict optionnel de regles personnalisees

    Returns:
        bool: True si le trading est autorise
    """
    # Normalise le symbole (retire suffixes MT5)
    sym = symbol.upper()
    for suffix in [".I", ".M", ".C", ".R", ".ECN", ".PRO", ".STD", "M", "C"]:
        if sym.endswith(suffix):
            sym = sym[:-len(suffix)]

    rules = custom_rules or TIME_RULES

    # Cherche les regles specifiques au symbole, sinon utilise FOREX par defaut
    symbol_rules = rules.get(sym)
    if symbol_rules is None:
        # Si c'est une paire forex (6 chars, 3 lettres + 3 lettres)
        if len(sym) == 6 and sym[:3].isalpha() and sym[3:].isalpha():
            symbol_rules = rules.get("FOREX", [])
        else:
            symbol_rules = rules.get("DEFAULT", [])

    # Pas de regles = toujours autorise
    if not symbol_rules:
        return True

    # Extrait l'heure du timestamp
    if isinstance(timestamp, str):
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    elif isinstance(timestamp, datetime):
        dt = timestamp
    else:
        return True

    t = dt.time()

    # Verifie si dans une plage autorisee
    for start, end in symbol_rules:
        if start <= t <= end:
            return True
    return False


def get_allowed_hours(symbol, custom_rules=None):
    """Retourne la liste des plages horaires autorisees pour un symbole."""
    sym = symbol.upper()
    for suffix in [".I", ".M", ".C", ".R", ".ECN", ".PRO", ".STD", "M", "C"]:
        if sym.endswith(suffix):
            sym = sym[:-len(suffix)]

    rules = custom_rules or TIME_RULES

    symbol_rules = rules.get(sym)
    if symbol_rules is None:
        if len(sym) == 6 and sym[:3].isalpha() and sym[3:].isalpha():
            symbol_rules = rules.get("FOREX", [])
        else:
            symbol_rules = rules.get("DEFAULT", [])

    return symbol_rules


def format_rules(symbol, custom_rules=None):
    """Formate les regles horaires en string lisible."""
    rules = get_allowed_hours(symbol, custom_rules)
    if not rules:
        return "24/7 (pas de restriction)"
    parts = []
    for start, end in rules:
        parts.append(f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}")
    return ", ".join(parts)
