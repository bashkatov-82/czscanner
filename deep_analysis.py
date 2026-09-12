#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Глубокий анализ с оценкой надёжности сигнала.

Шесть слоёв проверки: MTF, режим рынка (BTC), ликвидность,
волатильность (ATR), walk-forward бэктест, bootstrap-ДИ.
На выходе — интегральная оценка надёжности 0..1 и вердикт.
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime

import crypto_zone_screener as czs

HIGHER_TF = {"15m": "1h", "1h": "4h", "4h": "1d", "1d": "1w"}


def atr_pct(df, period=14):
    """ATR14 в процентах от цены."""
    h, l, c = df["high"], df["low"], df["close"]
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return float(atr.iloc[-1] / c.iloc[-1] * 100)


def _dir_of(sig):
    if sig == "ЛОНГ":
        return 1
    if sig == "ШОРТ":
        return -1
    return 0


def _state_of(df, args):
    """Сигнал и списки зон для DataFrame с уже посчитанными индикаторами."""
    accum = czs.find_zones(df, "accum_score", args.accum_threshold, args.min_zone)
    distrib = czs.find_zones(df, "distrib_score", args.distrib_threshold, args.min_zone)
    last = len(df) - 1
    a = czs.get_zone_state(accum, last, args.zone_age)
    d = czs.get_zone_state(distrib, last, args.zone_age)
    sig = czs.choose_signal(df, a, d)
    return sig, accum, distrib


def market_regime(interval, limit, args):
    """Режим рынка по BTC: risk-on / risk-off / нейтральный."""
    df = czs.fetch_ohlcv("BTCUSDT", interval, limit)
    if df is None or len(df) < 80:
        return "неизвестен", 0
    df = czs.add_indicators(df)
    sig, _, _ = _state_of(df, args)
    btc_dir = _dir_of(sig["signal"])
    slope = float(df["sma50"].iloc[-1] - df["sma50"].iloc[-10])
    cmf = float(df["cmf"].iloc[-1])
    if btc_dir == 1 or (slope > 0 and cmf > 0.02):
        return "risk-on", btc_dir
    if btc_dir == -1 or (slope < 0 and cmf < -0.02):
        return "risk-off", btc_dir
    return "нейтральный", btc_dir


def liquidity_usd(symbol):
    data = czs.get_json(f"{czs.BASE_URL}/api/v3/ticker/24hr", params={"symbol": symbol})
    if not data:
        return 0.0
    return czs.safe_float(data.get("quoteVolume"), 0.0)


def bootstrap_ci(values, n=1000, ci=0.95):
    """95% доверительный интервал средней прибыли бутстрепом."""
    if len(values) < 3:
        return None
    rng = np.random.default_rng(42)
    arr = np.asarray(values, dtype=float)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n)]
    lo, hi = np.percentile(means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(lo), float(hi)


def walk_forward(symbol, df, zones, direction, stop, take, hold):
    """Калибровка на первых 60% зон, оценка на отложенных 40%."""
    if not zones or direction == 0:
        return {}, {}, []
    n = len(zones)
    split = max(1, int(n * 0.6))
    side = "long" if direction == 1 else "short"
    calib_tr = czs.backtest_direction(symbol, df, zones[:split], side, stop, take, hold)
    valid_tr = czs.backtest_direction(symbol, df, zones[split:], side, stop, take, hold)
    cs = czs.backtest_stats(calib_tr)
    vs = czs.backtest_stats(valid_tr)
    profits = [] if valid_tr.empty else valid_tr["profit_pct"].tolist()
    return cs, vs, profits


def _journal(row):
    path = os.path.join(os.getcwd(), "deep_reports.csv")
    df = pd.DataFrame([row])
    if os.path.exists(path):
        df.to_csv(path, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(path, index=False, encoding="utf-8-sig")


def analyze_symbol_deep(symbol, interval, limit, args):
    """Полный глубокий анализ одной пары. Возвращает список строк отчёта."""
    lines = [f"=== Глубокий анализ {symbol} | ТФ {interval} ==="]

    df = czs.fetch_ohlcv(symbol, interval, limit)
    if df is None or len(df) < 80:
        return lines + ["Недостаточно данных для анализа."]

    df = czs.add_indicators(df)
    sig, accum, distrib = _state_of(df, args)
    direction = _dir_of(sig["signal"])

    lines.append(
        f"Базовый сигнал: {sig['signal']} | фаза: {sig['phase']} | "
        f"зона: {sig['zone_state']} | сила: {sig['score']:.1f}"
    )

    if direction == 0:
        lines.append("Направленного сигнала нет — глубокий анализ не применим.")
        return lines

    # ---- 1) MTF ----
    htf = HIGHER_TF.get(interval)
    mtf_score, mtf_note = 0.5, "нет старшего ТФ для проверки"
    if htf:
        hdf = czs.fetch_ohlcv(symbol, htf, limit)
        if hdf is not None and len(hdf) >= 80:
            hdf = czs.add_indicators(hdf)
            hsig, _, _ = _state_of(hdf, args)
            htf_dir = _dir_of(hsig["signal"])
            if htf_dir == direction:
                mtf_score, mtf_note = 1.0, f"старший ТФ ({htf}) подтверждает: {hsig['signal']}"
            elif htf_dir == 0:
                mtf_score, mtf_note = 0.6, f"старший ТФ ({htf}) нейтрален: {hsig['signal']}"
            else:
                mtf_score, mtf_note = 0.0, f"КОНФЛИКТ со старшим ТФ ({htf}): {hsig['signal']}"
    lines.append(f"1) MTF: {mtf_note}")

    # ---- 2) Режим рынка ----
    regime, _ = market_regime(interval, limit, args)
    regime_ok = True
    reg_note = f"режим рынка (BTC): {regime}"
    if symbol != "BTCUSDT":
        if direction == -1 and regime == "risk-on":
            regime_ok = False
            reg_note += " — шорт альта против risk-on: надёжность снижена"
        if direction == 1 and regime == "risk-off":
            regime_ok = False
            reg_note += " — лонг альта против risk-off: надёжность снижена"
    lines.append(f"2) {reg_note}")

    # ---- 3) Ликвидность ----
    liq = liquidity_usd(symbol)
    if liq >= 50e6:
        liq_score, liq_note = 1.0, f"достаточная (${liq / 1e6:.0f}M за 24ч)"
    elif liq >= 10e6:
        liq_score, liq_note = 0.6, f"умеренная (${liq / 1e6:.0f}M за 24ч)"
    else:
        liq_score, liq_note = 0.2, f"НИЗКАЯ (${liq / 1e6:.1f}M за 24ч) — риск слиппэджа"
    lines.append(f"3) Ликвидность: {liq_note}")

    # ---- 4) Волатильность и ATR-стопы ----
    ap = atr_pct(df)
    stop = round(max(1.5, 1.5 * ap), 2)
    take = round(stop * 2, 2)
    vol_note = f"ATR14 = {ap:.2f}% цены; ATR-стоп ≈ {stop}% / тейк ≈ {take}%"
    if ap > 6:
        vol_note += " — ЭКСТРЕМАЛЬНАЯ волатильность: уменьшить размер позиции"
    vol_score = 1.0 if ap <= 3 else (0.7 if ap <= 6 else 0.4)
    lines.append(f"4) Волатильность: {vol_note}")

    # ---- 5) Walk-forward ----
    zones = accum if direction == 1 else distrib
    cs, vs, profits = walk_forward(symbol, df, zones, direction, stop, take, args.hold)
    sample_ok = vs.get("Сделок", 0) >= 5
    lines.append(
        f"5) Walk-forward: калибровка {cs.get('Сделок', 0)} сделок "
        f"(винрейт {cs.get('Винрейт %', 0):.0f}%); ВАЛИДАЦИЯ {vs.get('Сделок', 0)} сделок "
        f"(винрейт {vs.get('Винрейт %', 0):.0f}%, средняя {vs.get('Средняя %', 0):.2f}%)"
    )
    if not sample_ok:
        lines.append("   Выборка валидации < 5 сделок: статистическая надёжность низкая.")

    # ---- 6) Bootstrap ДИ ----
    ci = bootstrap_ci(profits) if profits else None
    ci_ok = False
    if ci:
        ci_ok = ci[0] > 0
        lines.append(f"6) Bootstrap 95% ДИ средней прибыли на валидации: [{ci[0]:.2f}%; {ci[1]:.2f}%]")
        if not ci_ok:
            lines.append("   Интервал включает ноль: edge статистически не отличим от случайности.")
    else:
        lines.append("6) ДИ: недостаточно сделок для оценки.")

    # ---- Интегральная надёжность ----
    bt_score = min(1.0, vs.get("Винрейт %", 0) / 60) if sample_ok else 0.0
    score = (0.20 * mtf_score
             + 0.20 * (1.0 if regime_ok else 0.3)
             + 0.15 * liq_score
             + 0.15 * vol_score
             + 0.20 * bt_score
             + 0.10 * (1.0 if ci_ok else 0.0))
    grade = "ВЫСОКАЯ" if score >= 0.7 else ("СРЕДНЯЯ" if score >= 0.5 else "НИЗКАЯ")

    lines.append("-" * 70)
    lines.append(f"НАДЁЖНОСТЬ: {score:.2f} / 1.00 — уровень {grade}")
    lines.append("Правило: ВЫСОКАЯ + сработавший триггер = рассматривать;")
    lines.append("СРЕДНЯЯ = список наблюдения; НИЗКАЯ = игнорировать.")

    try:
        _journal({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol, "interval": interval, "signal": sig["signal"],
            "zone_state": sig["zone_state"], "score_zone": round(sig["score"], 2),
            "mtf": mtf_note, "regime": regime, "liquidity_mln": round(liq / 1e6, 1),
            "atr_pct": round(ap, 2), "valid_trades": vs.get("Сделок", 0),
            "valid_winrate": round(vs.get("Винрейт %", 0), 1),
            "ci_lo": round(ci[0], 2) if ci else "", "ci_hi": round(ci[1], 2) if ci else "",
            "reliability": round(score, 2), "grade": grade,
        })
    except Exception:
        pass

    return lines