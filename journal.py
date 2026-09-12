
import os
from datetime import datetime
import pandas as pd
import crypto_zone_screener as czs

JOURNAL_FILE = "signals_journal.csv"
EVAL_HORIZON = 20
COOLDOWN_H = 24

COLS = ["time", "symbol", "interval", "signal", "phase", "zone_state", "score",
        "ai_verdict", "ai_conf", "mtf", "price", "stop", "target",
        "prob", "council", "ob", "status", "result", "profit_pct", "eval_time"]

def _load():
    if os.path.exists(JOURNAL_FILE):
        try:
            df = pd.read_csv(JOURNAL_FILE)
            for c in COLS:
                if c not in df.columns:
                    df[c] = ""
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=COLS)

def _save(df):
    df.to_csv(JOURNAL_FILE, index=False, encoding="utf-8-sig")

def record_signals(df, interval, ts):
    if df is None or df.empty:
        return 0
    j = _load()
    rows = []
    for _, r in df.iterrows():
        if r["signal"] not in ("ЛОНГ", "ШОРТ"):
            continue
        if not j.empty:
            open_same = j[(j.symbol == r["symbol"]) & (j.signal == r["signal"]) & (j.status == "open")]
            if not open_same.empty:
                continue
            recent = j[(j.symbol == r["symbol"]) & (j.signal == r["signal"]) &
                       (pd.to_datetime(j["time"], errors="coerce") >
                        pd.to_datetime(ts) - pd.Timedelta(hours=COOLDOWN_H))]
            if not recent.empty:
                continue
        price = float(r["price"])
        try:
            lo, hi = [float(x) for x in str(r["zone_range"]).split("-")]
        except Exception:
            lo, hi = price * 0.97, price * 1.03
        if r["signal"] == "ЛОНГ":
            stop, target = lo * 0.99, hi + (hi - lo)
        else:
            stop, target = hi * 1.01, lo - (hi - lo)
        rows.append({
            "time": ts, "symbol": r["symbol"], "interval": interval,
            "signal": r["signal"], "phase": r["phase"], "zone_state": r["zone_state"],
            "score": r["score"], "ai_verdict": r.get("ai_verdict", ""),
            "ai_conf": r.get("ai_conf", ""), "mtf": r.get("mtf", ""), "prob": r.get("prob", ""), "council": r.get("council", ""), "ob": r.get("ob", ""),
            "price": price, "stop": round(stop, 8), "target": round(target, 8),
            "status": "open", "result": "", "profit_pct": "", "eval_time": "",
        })
    if rows:
        j = pd.concat([j, pd.DataFrame(rows)], ignore_index=True)
        _save(j)
    return len(rows)

def evaluate_open():
    j = _load()
    if j.empty:
        return 0
    opened = j[j.status == "open"]
    if opened.empty:
        return 0
    cache = {}
    changed = 0
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for idx, r in opened.iterrows():
        sym, interval = r["symbol"], r["interval"]
        if sym not in cache:
            cache[sym] = czs.fetch_ohlcv(sym, interval, 200)
        df = cache[sym]
        if df is None or df.empty:
            continue
        t0 = pd.to_datetime(r["time"])
        fut = df[df["dt"] > t0]
        if len(fut) < EVAL_HORIZON:
            continue
        entry = float(r["price"])
        stop = float(r["stop"])
        target = float(r["target"])
        result, profit = None, None
        for _, c in fut.head(EVAL_HORIZON).iterrows():
            if r["signal"] == "ЛОНГ":
                if c["low"] <= stop:
                    result, profit = "stop", (stop - entry) / entry * 100; break
                if c["high"] >= target:
                    result, profit = "target", (target - entry) / entry * 100; break
            else:
                if c["high"] >= stop:
                    result, profit = "stop", (entry - stop) / entry * 100; break
                if c["low"] <= target:
                    result, profit = "target", (entry - target) / entry * 100; break
        if result is None:
            close = float(fut.iloc[EVAL_HORIZON - 1]["close"])
            profit = (close - entry) / entry * 100 if r["signal"] == "ЛОНГ" else (entry - close) / entry * 100
            result = "timeout"
        j.loc[idx, "status"] = "closed"
        j.loc[idx, "result"] = result
        j.loc[idx, "profit_pct"] = round(profit, 2)
        j.loc[idx, "eval_time"] = now
        changed += 1
    if changed:
        _save(j)
    return changed

def calibration():
    j = _load()
    if j.empty:
        return "Журнал пуст.\nКалибровка появится после первых сканов."
    closed = j[j.status == "closed"]
    if closed.empty:
        n_open = int((j.status == "open").sum())
        return f"Открытых сигналов: {n_open}, размеченных исходов пока нет."
    total = len(closed)
    wins = int((closed.profit_pct.astype(float) > 0).sum())
    lines = [f"=== Калибровка сигналов ===",
             f"Исходов: {total} | выигрышных: {wins} ({wins/total*100:.0f}%) | "
             f"средний P/L: {closed.profit_pct.astype(float).mean():.2f}%"]
    for key in ["signal", "phase", "ai_verdict", "mtf"]:
        if key not in closed.columns:
            continue
        vals = closed[closed[key].notna() & (closed[key].astype(str) != "")]
        if vals.empty:
            continue
        lines.append(f"\nПо категории «{key}»:")
        for name, s in vals.groupby(key)["profit_pct"]:
            s = s.astype(float); n = len(s)
            lines.append(f"   {name}: n={n}, hit-rate={(s>0).mean()*100:.0f}%, средняя={s.mean():.2f}%")
    lines.append("\nПо типам исхода:")
    for name, cnt in closed["result"].value_counts().items():
        lines.append(f"   {name}: {cnt}")
    lines.append("\nПравило: hit-rate >55% при n>=15 — рабочая категория.")
    return "\n".join(lines)

