import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import crypto_zone_screener as czs

_cache = {}
_lock = threading.Lock()
CACHE_TTL = {"15m": 60, "1h": 120, "4h": 300, "1d": 600, "1w": 900}
HIGHER = {"15m": "1h", "1h": "4h", "4h": "1d", "1d": "1w"}
WORKERS = 4

def get_candles(symbol, interval, limit):
    key = (symbol, interval)
    now = time.time()
    with _lock:
        item = _cache.get(key)
        if item and now - item["ts"] < CACHE_TTL.get(interval, 120):
            return item["df"]
    df = czs.fetch_ohlcv(symbol, interval, limit)
    if df is not None:
        with _lock:
            _cache[key] = {"ts": time.time(), "df": df}
    return df

def _mtf_note(symbol, interval, args, direction):
    h = HIGHER.get(interval)
    if not h or direction == 0:
        return "—"
    df = get_candles(symbol, h, args.limit)
    if df is None or len(df) < 80:
        return "—"
    df = czs.add_indicators(df)
    accum = czs.find_zones(df, "accum_score", args.accum_threshold, args.min_zone)
    distrib = czs.find_zones(df, "distrib_score", args.distrib_threshold, args.min_zone)
    last = len(df) - 1
    a = czs.get_zone_state(accum, last, args.zone_age)
    d = czs.get_zone_state(distrib, last, args.zone_age)
    sig = czs.choose_signal(df, a, d)
    hdir = 1 if sig["signal"] == "ЛОНГ" else (-1 if sig["signal"] == "ШОРТ" else 0)
    if hdir == direction:
        return "подтв."
    if hdir == 0:
        return "нейтр."
    return "конфл."

def _analyze_one(item, args):
    try:
        symbol = item["symbol"]
        df = get_candles(symbol, args.interval, args.limit)
        if df is None or len(df) < 80:
            return None
        df = czs.add_indicators(df)
        accum = czs.find_zones(df, "accum_score", args.accum_threshold, args.min_zone)
        distrib = czs.find_zones(df, "distrib_score", args.distrib_threshold, args.min_zone)
        last = len(df) - 1
        a = czs.get_zone_state(accum, last, args.zone_age)
        d = czs.get_zone_state(distrib, last, args.zone_age)
        sig = czs.choose_signal(df, a, d)
        z = sig["zone"]
        direction = 1 if sig["signal"] == "ЛОНГ" else (-1 if sig["signal"] == "ШОРТ" else 0)
        return {
            "symbol": symbol, "signal": sig["signal"], "phase": sig["phase"],
            "zone_state": sig["zone_state"],
            "zone_len": z["length"] if z is not None else 0,
            "score": round(sig["score"], 3),
            "price": float(df["close"].iloc[-1]),
            "chg24h": float(item.get("chg24h", 0)),
            "ret20_pct": float(df["close"].pct_change(20).iloc[-1]) * 100,
            "vol_z": float(df["vol_z"].iloc[-1]),
            "cmf": float(df["cmf"].iloc[-1]),
            "taker_ratio": float(df["taker_ratio"].iloc[-1]),
            "rsi": float(df["rsi"].iloc[-1]),
            "zone_range": f"{z['price_low']:.2f}-{z['price_high']:.2f}" if z is not None else "-",
            "reasons": sig["reasons"],
            "mtf": _mtf_note(symbol, args.interval, args, direction),
        }
    except Exception:
        return None

def run_scan(args):
    top = czs.get_top_symbols(args.top, args.min_quote)
    if not top:
        return pd.DataFrame()
    rows = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(_analyze_one, it, args) for it in top]
        for f in as_completed(futs):
            r = f.result()
            if r:
                rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    warnings = []
    btc = get_candles("BTCUSDT", args.interval, args.limit)
    if btc is not None:
        br = btc["close"].pct_change().tail(100).reset_index(drop=True)
        def corr(sym):
            d2 = get_candles(sym, args.interval, args.limit)
            if d2 is None:
                return np.nan
            sr = d2["close"].pct_change().tail(100).reset_index(drop=True)
            if len(sr) != len(br):
                return np.nan
            return float(sr.corr(br))
        df["btc_corr"] = df["symbol"].map(corr).round(2)
        hi_corr = df["btc_corr"].fillna(0) > 0.7
        n_long = int(((df["signal"] == "ЛОНГ") & hi_corr).sum())
        n_short = int(((df["signal"] == "ШОРТ") & hi_corr).sum())
        if n_long >= 3:
            warnings.append(f"концентрация: {n_long} ЛОНГ с корреляцией>0.7 к BTC")
        if n_short >= 3:
            warnings.append(f"концентрация: {n_short} ШОРТ с корреляцией>0.7 к BTC")
    else:
        df["btc_corr"] = np.nan
    df.attrs["warnings"] = warnings
    df = df.sort_values(by=["score", "vol_z"], ascending=False).reset_index(drop=True)
    return df

