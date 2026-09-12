#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
brain.py — итоговый вероятностный вердикт и самообучение.

Слой 1 (эвристика): голоса инструментов (зоны, AI-агенты, консилиум графика,
стакан, Delta-RSI v2, MTF) сводятся в согласованный счёт s и эвристическую
вероятность успеха.
Слой 2 (обучение): логистическая регрессия на исходах signals_journal.csv
(признаки записываются в момент сигнала). До MIN_SAMPLES исходов работает
только эвристика; далее модель переобучается автоматически в мониторинге
(каждые +5 новых исходов) или кнопкой «Обучить».
Модель: model_brain.json.
"""
import os
import json
import math
from datetime import datetime

import numpy as np
import pandas as pd

import crypto_zone_screener as czs

MODEL_FILE = "model_brain.json"
FEATURES = ["score", "ai_conf", "council", "ob", "mtf_enc", "prob0"]
MIN_SAMPLES = 30

TOOLS_WEIGHTS = {"zones": 0.30, "agents": 0.20, "council": 0.25,
                 "ob": 0.10, "drsi": 0.10, "mtf": 0.05}


def _num(r, k, d=0.0):
    try:
        v = float(r.get(k, d))
        return v if v == v else d
    except Exception:
        return d


def _mtf_enc(v):
    return {"подтв.": 1.0, "нейтр.": 0.0, "конфл.": -1.0}.get(str(v), 0.0)


def _load_model():
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _save_model(m):
    with open(MODEL_FILE, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)


def heuristic_prob(s):
    """Эвристическая вероятность из согласованного счёта s (шкала ~-2..+2)."""
    return float(np.clip(50 + 40 * np.tanh(s / 1.5), 5, 95))


def collect_features(row, args):
    """Снимок голосов всех инструментов по строке таблицы.
    Возвращает (feat, breakdown, s, prob0)."""
    sym = row["symbol"]
    direction = 1 if row["signal"] == "ЛОНГ" else (-1 if row["signal"] == "ШОРТ" else 0)
    br = []

    sz = float(np.clip((_num(row, "score") - 3.0) / 3.0, -1, 1)) * direction
    br.append(("Зоны (сила и фаза)", sz))

    sa = float(np.clip(_num(row, "ai_conf") * 2 - 1, -1, 1)) if direction else 0.0
    br.append(("AI-агенты", sa))

    sc = 0.0
    try:
        import fast_scan
        import chart_ai
        df = fast_scan.get_candles(sym, args.interval, args.limit)
        if df is not None and len(df) >= 120:
            df = czs.add_indicators(df)
            res = chart_ai.council(df.tail(220).reset_index(drop=True), sym)
            sc = float(np.clip(res["score"] * 3, -1, 1)) * direction
        br.append(("Консилиум графика", sc))
    except Exception:
        br.append(("Консилиум графика", 0.0))

    so = 0.0
    try:
        import orderbook_analyzer as oba
        ob = oba.intent(sym)
        so = 1.0 if ob["dir"] == direction else (-1.0 if ob["dir"] == -direction else 0.0)
        br.append(("Стакан (намерение)", so))
    except Exception:
        br.append(("Стакан (намерение)", 0.0))

    sd = 0.0
    try:
        import fast_scan
        import drsi_v2
        df2 = fast_scan.get_candles(sym, args.interval, args.limit)
        if df2 is not None and len(df2) >= 120:
            z, sig, r2 = drsi_v2.compute_drsi_v2(df2)
            ev = drsi_v2.events(df2, z, r2)
            if bool(ev["long"].iloc[-1]) and direction == 1:
                sd = 1.0
            elif bool(ev["short"].iloc[-1]) and direction == -1:
                sd = 1.0
            elif bool(ev["long"].iloc[-1]) or bool(ev["short"].iloc[-1]):
                sd = -0.5
            else:
                sd = float(np.clip(float(z.iloc[-1]) / 2, -0.5, 0.5)) * direction
        br.append(("Delta-RSI v2", sd))
    except Exception:
        br.append(("Delta-RSI v2", 0.0))

    sm = _mtf_enc(row.get("mtf", ""))
    br.append(("MTF (старший ТФ)", sm))

    w = TOOLS_WEIGHTS
    s = (w["zones"] * sz + w["agents"] * sa + w["council"] * sc +
         w["ob"] * so + w["drsi"] * sd + w["mtf"] * sm)
    s = float(np.clip(s / sum(w.values()), -1, 1)) * 2.0
    prob0 = heuristic_prob(s * 1.5)

    feat = {
        "score": _num(row, "score"),
        "ai_conf": _num(row, "ai_conf"),
        "council": sc,
        "ob": so,
        "mtf_enc": sm,
        "prob0": prob0,
    }
    return feat, br, s, prob0


def predict(feat):
    """Вероятность успеха: обученная модель либо эвристика.
    Возвращает (p, used_model)."""
    m = _load_model()
    x = np.array([feat[k] for k in FEATURES], dtype=float)
    if m and m.get("n", 0) >= MIN_SAMPLES:
        z = (x - np.array(m["mean"])) / (np.array(m["std"]) + 1e-9)
        p = 100.0 / (1.0 + math.exp(-(float(np.dot(np.array(m["w"]), z)) + m["b"])))
        return float(np.clip(p, 3, 97)), True
    return float(feat["prob0"]), False


def enrich_dataframe(df, args, cap=25):
    """Вероятности для сигнальных строк; совместимо с pandas 3.x
    (числа не вставляются в строковую колонку)."""
    df = df.copy()
    prob_map = {}
    council_map = {}
    sig = df[df["signal"].isin(["ЛОНГ", "ШОРТ"])].head(cap)
    for idx, row in sig.iterrows():
        try:
            feat, br, s, prob0 = collect_features(row, args)
            p, used = predict(feat)
            prob_map[idx] = f"{p:.0f}"
            council_map[idx] = float(feat["council"])
        except Exception as e:
            print("row fail:", row.get("symbol"), type(e).__name__, e)
    df["prob"] = [prob_map.get(i, "") for i in df.index]
    df["council"] = [council_map.get(i, np.nan) for i in df.index]
    return df


def _journal_closed():
    import journal
    j = journal._load()
    if j.empty:
        return pd.DataFrame()
    return j[j.status == "closed"].copy()


def _x_from_row(r):
    prob0 = _num(r, "prob")
    if prob0 == 0.0:
        prob0 = 50.0
    return np.array([
        _num(r, "score"), _num(r, "ai_conf"), _num(r, "council"),
        _num(r, "ob"), _mtf_enc(r.get("mtf", "")), prob0], dtype=float)


def train():
    """Обучение логистической регрессии на исходах журнала."""
    c = _journal_closed()
    if len(c) < MIN_SAMPLES:
        return (f"Обучение отложено: размеченных исходов {len(c)} < {MIN_SAMPLES}. "
                f"Вердикт остаётся эвристическим.")
    X = np.array([_x_from_row(r) for _, r in c.iterrows()])
    y = (c.profit_pct.astype(float) > 0).to_numpy().astype(float)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-9
    Z = (X - mean) / std
    w = np.zeros(Z.shape[1])
    b = 0.0
    lr = 0.05
    for _ in range(400):
        p = 1.0 / (1.0 + np.exp(-(Z @ w + b)))
        g = p - y
        w -= lr * (Z.T @ g) / len(y) + 1e-4 * w
        b -= lr * float(g.mean())
    p = 1.0 / (1.0 + np.exp(-(Z @ w + b)))
    acc = float(((p > 0.5) == (y > 0.5)).mean())
    _save_model({"w": w.tolist(), "b": float(b), "mean": mean.tolist(),
                 "std": std.tolist(), "n": int(len(y)),
                 "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 "in_sample_acc": acc})
    lines = [f"Модель обучена на {len(y)} исходах, in-sample точность {acc * 100:.1f}%",
             "Веса признаков (знак = направление пользы):"]
    for name, wi in zip(FEATURES, w):
        lines.append(f"   {name}: {wi:+.3f}")
    lines.append("Калибровка (прогноз против факта):")
    dfp = pd.DataFrame({"p": p * 100.0, "y": y})
    dfp["bucket"] = pd.cut(dfp.p, [0, 40, 50, 60, 100])
    for bk, g in dfp.groupby("bucket", observed=True):
        if len(g):
            lines.append(f"   прогноз {bk}: факт {g.y.mean() * 100:.0f}% (n={len(g)})")
    lines.append("In-sample точность оптимистична; рабочая оценка — строки калибровки.")
    return "\n".join(lines)


def maybe_train():
    """Автопереобучение: вызывается в мониторинге."""
    c = _journal_closed()
    n = len(c)
    m = _load_model()
    n0 = m.get("n", 0) if m else 0
    if n >= MIN_SAMPLES and n - n0 >= 5:
        return train()
    return None


def verdict_report(row, args):
    """Полный вердикт по строке: голоса инструментов и итоговая вероятность."""
    feat, br, s, prob0 = collect_features(row, args)
    p, used = predict(feat)
    lines = [f"=== Итоговый вердикт: {row['symbol']} | {row['signal']} ===", ""]
    for name, v in br:
        bar = "+" * int(max(0.0, v) * 10) or ("-" * int(max(0.0, -v) * 10) or "0")
        lines.append(f"{name:24s}: {v:+.2f}  {bar}")
    lines.append("")
    lines.append(f"Согласованный счёт инструментов: {s:+.2f} (шкала -2..+2)")
    lines.append(f"Эвристическая вероятность: {prob0:.0f}%")
    src = ("обученная модель по исходам журнала"
           if used else "эвристика: исходов журнала ещё < 30")
    lines.append(f"ИТОГОВАЯ вероятность достижения цели: {p:.0f}%  [{src}]")
    lines.append("")
    if p >= 60:
        lines.append("Фильтр: РАССМАТРИВАТЬ (вход по плану, стоп за границей зоны).")
    elif p >= 45:
        lines.append("Фильтр: НАБЛЮДЕНИЕ (ждать объёмного слома границы зоны).")
    else:
        lines.append("Фильтр: ПРОПУСТИТЬ (инструменты не согласованы).")
    lines.append("")
    lines.append("Вероятность = доля исторически похожих сигналов, дошедших до цели,")
    lines.append("а не гарантия по конкретной сделке.")
    return "\n".join(lines)