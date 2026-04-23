"""
Diagnostic script for tensor dimension mismatch in live trading.
Run this before starting a live session to verify data and model compatibility.
"""
import torch
import pandas as pd
from datetime import datetime, timedelta

def diagnose_live_setup(predictor, feed, symbol, timeframe, lookback, pred_len):
    """Diagnostique la compatibilite entre le modele et les donnees live."""
    print("=" * 60)
    print("DIAGNOSTIC LIVE TRADING")
    print("=" * 60)
    print(f"Symbole: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Lookback: {lookback}")
    print(f"Pred_len: {pred_len}")
    print()

    # 1. Verifier la connexion broker
    print("--- 1. Connexion broker ---")
    if not feed.is_connected():
        print("ERREUR: Broker non connecte!")
        return False
    print("OK: Broker connecte")
    print()

    # 2. Recuperer les dernieres bougies
    print("--- 2. Donnees historiques ---")
    df, err = feed.get_latest_candles(symbol, timeframe, lookback + 10, as_kronos=True)
    if err or df is None:
        print(f"ERREUR: Impossible de recuperer les donnees: {err}")
        return False
    print(f"OK: {len(df)} bougies recuperees")
    print(f"Colonnes: {list(df.columns)}")
    print(f"Types: {df.dtypes.to_dict()}")
    print()

    # 3. Verifier les colonnes requises
    print("--- 3. Colonnes requises ---")
    required_cols = ["open", "high", "low", "close", "volume", "amount", "timestamps"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERREUR: Colonnes manquantes: {missing}")
        return False
    print("OK: Toutes les colonnes presentes")
    print()

    # 4. Verifier les NaN
    print("--- 4. Qualite des donnees ---")
    price_cols = ["open", "high", "low", "close"]
    nan_count = df[price_cols].isnull().sum().sum()
    if nan_count > 0:
        print(f"ATTENTION: {nan_count} valeurs NaN dans les prix")
    else:
        print("OK: Pas de NaN dans les prix")
    print()

    # 5. Verifier la longueur des donnees
    print("--- 5. Longueur des donnees ---")
    df_trimmed = df.iloc[-lookback:].reset_index(drop=True)
    if len(df_trimmed) < lookback:
        print(f"ERREUR: Donnees insuffisantes: {len(df_trimmed)} < {lookback}")
        return False
    print(f"OK: {len(df_trimmed)} bougies apres trim (lookback={lookback})")
    print()

    # 6. Verifier la compatibilite modele
    print("--- 6. Compatibilite modele ---")
    x_df = df_trimmed[["open", "high", "low", "close", "volume", "amount"]]
    x_timestamp = df_trimmed["timestamps"]
    last_ts = df_trimmed["timestamps"].iloc[-1]
    tf_seconds = {"M1": 60, "M5": 300, "M15": 900, "H1": 3600, "H4": 14400, "D1": 86400}.get(timeframe, 300)
    y_timestamp = pd.Series([last_ts + pd.Timedelta(seconds=tf_seconds) * (i + 1) for i in range(pred_len)])

    print(f"x_df shape: {x_df.shape}")
    print(f"x_timestamp length: {len(x_timestamp)}")
    print(f"y_timestamp length: {len(y_timestamp)}")

    # Verifier d_in du modele
    model_d_in = getattr(predictor.tokenizer, 'd_in', None)
    data_features = x_df.shape[1]
    print(f"Modele d_in: {model_d_in}")
    print(f"Features dans les donnees: {data_features}")
    if model_d_in and model_d_in != data_features:
        print(f"ERREUR: Mismatch d_in! Modele attend {model_d_in}, donnees ont {data_features}")
        return False
    print("OK: Dimensions compatibles")
    print()

    # 7. Test prediction unique
    print("--- 7. Test prediction ---")
    try:
        with torch.inference_mode():
            pred_df = predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                sample_count=1,
                sample_logits=False,
                verbose=False,
            )
        print(f"OK: Prediction reussie")
        print(f"  Shape prediction: {pred_df.shape}")
        print(f"  Colonnes: {list(pred_df.columns)}")
    except Exception as e:
        print(f"ERREUR: Prediction a echoue: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    print("=" * 60)
    print("DIAGNOSTIC TERMINE: Tout est OK")
    print("=" * 60)
    return True


if __name__ == "__main__":
    import sys
    # Usage: python diagnose_live.py <symbol> <timeframe> <lookback> <pred_len>
    # Ex: python diagnose_live.py XAUUSD M5 120 1
    symbol = sys.argv[1] if len(sys.argv) > 1 else "XAUUSD"
    timeframe = sys.argv[2] if len(sys.argv) > 2 else "M5"
    lookback = int(sys.argv[3]) if len(sys.argv) > 3 else 120
    pred_len = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    print("Pour utiliser ce diagnostic, importez-le dans votre script live:")
    print("  from scripts.diagnose_live import diagnose_live_setup")
    print("  diagnose_live_setup(predictor, feed, symbol, timeframe, lookback, pred_len)")
