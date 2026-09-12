#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Delta-RSI v2 — уточнённый порт индикатора Delta-RSI Oscillator (c) tbiktag (MPL-2.0).
Отличия от v1: logit(RSI) перед дифференцированием; центрированный базис и
взвешенный МНК (производная = a1, меньше краевого шума); z-score нормировка D-RSI;
гистерезис и подтверждение сигнала; фильтр качества по R2; опциональный тренд-гейт.
"""
import numpy as np
import pandas as pd


def _rsi(close, length):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    rs = gain / (loss + 1e-12)
    return 100 - 100 / (1 + rs)


def _logit(rsi_series):
    x = np.clip(rsi_series.to_numpy(dtype=float) / 100.0, 1e-3, 1 - 1e-3)
    return np.log(x / (1 - x))


def compute_drsi_v2(df, rsi_len=21, window=21, degree=2, signal_len=9, halflife=None):
    """Возвращает (zscore D-RSI, сигнальная EMA, R2 аппроксимации)."""
    n = len(df)
    L = _logit(_rsi(df["close"], rsi_len))
    hl = halflife or max(3.0, window / 3.0)

    z = np.arange(window) - (window - 1)          # текущая свеча: z = 0
    V = np.vander(z, N=degree + 1, increasing=True)
    w = 0.5 ** ((window - 1 - np.arange(window)) / hl)   # вес 1 у текущей свечи
    sw = np.sqrt(w)
    Vw = V * sw[:, None]

    drsi = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    for t in range(window - 1, n):
        seg = L[t - window + 1:t + 1]
        if np.isnan(seg).any():
            continue
        yw = seg * sw
        coef, *_ = np.linalg.lstsq(Vw, yw, rcond=None)
        yhat = V @ coef
        sse = float(np.sum(w * (seg - yhat) ** 2))
        mu = float(np.average(seg, weights=w))
        sst = float(np.sum(w * (seg - mu) ** 2))
        drsi[t] = coef[1]                 # dP/dz в z=0 = производная на текущей свече
        r2[t] = 1 - sse / sst if sst > 1e-12 else np.nan

    drsi = pd.Series(drsi, index=df.index)
    r2 = pd.Series(r2, index=df.index)
    sd = drsi.rolling(100, min_periods=30).std()
    zscore = drsi / (sd + 1e-12)
    signal = zscore.ewm(span=signal_len, adjust=False).mean()
    return zscore, signal, r2


def events(df, zscore, r2, h=0.25, r2_min=0.5, use_trend_gate=False, ema_len=200):
    """Подтверждённые события с гистерезисом: входы и выходы."""
    n = len(df)
    zv = zscore.to_numpy()
    r2v = r2.to_numpy()
    close = df["close"].to_numpy()
    ema = df["close"].ewm(span=ema_len, adjust=False).mean().to_numpy()

    long_ev = np.zeros(n, bool)
    short_ev = np.zeros(n, bool)
    exit_long = np.zeros(n, bool)
    exit_short = np.zeros(n, bool)

    for t in range(2, n):
        fit_ok = (r2v[t] == r2v[t]) and (r2v[t] >= r2_min)
        gate_l = (not use_trend_gate) or (close[t] > ema[t])
        gate_s = (not use_trend_gate) or (close[t] < ema[t])

        raw_up = (zv[t - 1] > h) and (zv[t - 2] <= h)      # пересечение +h свечу назад
        raw_dn = (zv[t - 1] < -h) and (zv[t - 2] >= -h)
        raw_xd = (zv[t - 1] < 0) and (zv[t - 2] >= 0)      # пересечение нуля вниз
        raw_xu = (zv[t - 1] > 0) and (zv[t - 2] <= 0)

        if raw_up and zv[t] > h and fit_ok and gate_l:
            long_ev[t] = True
        if raw_dn and zv[t] < -h and fit_ok and gate_s:
            short_ev[t] = True
        if raw_xd and zv[t] < 0:
            exit_long[t] = True
        if raw_xu and zv[t] > 0:
            exit_short[t] = True

    idx = df.index
    return {"long": pd.Series(long_ev, index=idx),
            "short": pd.Series(short_ev, index=idx),
            "exit_long": pd.Series(exit_long, index=idx),
            "exit_short": pd.Series(exit_short, index=idx)}


class DrsiV2Analyst:
    """Член консилиума: Delta-RSI v2 (формат chart_ai: вердикт, пояснения, метки)."""
    name = "Аналитик Delta-RSI v2"

    def __init__(self, rsi_len=21, window=21, degree=2, signal_len=9,
                 h=0.25, r2_min=0.5, use_trend_gate=False, lookback_marks=60):
        self.p = dict(rsi_len=rsi_len, window=window, degree=degree,
                      signal_len=signal_len, h=h, r2_min=r2_min,
                      use_trend_gate=use_trend_gate)
        self.lookback_marks = lookback_marks

    def analyze(self, df, symbol=None):
        z, sig, r2 = compute_drsi_v2(df, self.p["rsi_len"], self.p["window"],
                                     self.p["degree"], self.p["signal_len"])
        ev = events(df, z, r2, self.p["h"], self.p["r2_min"], self.p["use_trend_gate"])

        notes, marks = [], []
        zv = float(z.iloc[-1])
        sv = float(sig.iloc[-1])
        r2v = float(r2.iloc[-1])
        long_now = bool(ev["long"].iloc[-1])
        short_now = bool(ev["short"].iloc[-1])

        verdict = 0.0
        if long_now:
            verdict += 0.7
            notes.append("D-RSI v2: подтверждённый сигнал покупки")
        if short_now:
            verdict -= 0.7
            notes.append("D-RSI v2: подтверждённый сигнал продажи")
        if not long_now and not short_now:
            base = 0.25 * max(-1.0, min(1.0, zv / 2))
            base += 0.15 if zv > float(z.iloc[-2]) else -0.15
            base += 0.10 if zv > sv else -0.10
            verdict = max(-0.6, min(0.6, base))
            notes.append("подтверждённого входа нет; учтён фон z-D-RSI")
        notes.append(f"z-D-RSI={zv:.2f}, сигнальная={sv:.2f}, R2={r2v:.2f}")
        if r2v == r2v and r2v < self.p["r2_min"]:
            notes.append("R2 ниже порога: аппроксимация ненадёжна, события глушатся")

        n = len(df)
        start = max(0, n - self.lookback_marks)
        for i in range(start, n):
            if bool(ev["long"].iloc[i]):
                marks.append({"kind": "point", "x": i, "y": df["low"].iloc[i],
                              "marker": "*", "color": "#00c853"})
            if bool(ev["short"].iloc[i]):
                marks.append({"kind": "point", "x": i, "y": df["high"].iloc[i],
                              "marker": "*", "color": "#ff1744"})
        return max(-1.0, min(1.0, verdict)), notes, marks


if __name__ == "__main__":
    import crypto_zone_screener as czs
    df = czs.fetch_ohlcv("BTCUSDT", "4h", 400)
    v, notes, marks = DrsiV2Analyst().analyze(df)
    print(f"Вердикт D-RSI v2: {v:+.2f}")
    for t in notes:
        print("  —", t)
    print(f"Меток: {len(marks)}")