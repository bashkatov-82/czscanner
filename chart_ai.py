#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Консилиум ИИ-аналитиков графика: 7 специалистов + координатор.
Новое: Аналитик разворотов (слом структуры) и Аналитик стакана (живые стенки).
"""
import numpy as np
import pandas as pd

import crypto_zone_screener as czs


def _swing_points(df, w=5):
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    n = len(df)
    highs, lows = [], []
    for i in range(w, n - w):
        if h[i] >= np.max(h[i - w:i + w + 1]):
            highs.append(i)
        if l[i] <= np.min(l[i - w:i + w + 1]):
            lows.append(i)
    return highs, lows


class StructureAnalyst:
    name = "Аналитик структуры"

    def analyze(self, df, symbol=None):
        highs, lows = _swing_points(df, 5)
        notes, marks = [], []
        verdict = 0.0
        if len(highs) >= 2 and len(lows) >= 2:
            sh1, sh2 = highs[-2], highs[-1]
            sl1, sl2 = lows[-2], lows[-1]
            hh = df["high"].iloc[sh2] > df["high"].iloc[sh1]
            hl = df["low"].iloc[sl2] > df["low"].iloc[sl1]
            if hh and hl:
                verdict = 0.7; notes.append("восходящая структура: HH/HL")
            elif not hh and not hl:
                verdict = -0.7; notes.append("нисходящая структура: LH/LL")
            elif hh and not hl:
                verdict = 0.2; notes.append("слом структуры: HH при LL (переходная фаза)")
            else:
                verdict = -0.2; notes.append("структура: LH при HL (борьба/диапазон)")
        else:
            notes.append("мало swing-точек для оценки структуры")
        for i in highs[-6:]:
            marks.append({"kind": "point", "x": i, "y": df["high"].iloc[i], "marker": "v", "color": "#c62828"})
        for i in lows[-6:]:
            marks.append({"kind": "point", "x": i, "y": df["low"].iloc[i], "marker": "^", "color": "#007a3d"})
        return verdict, notes, marks


class DivergenceAnalyst:
    name = "Аналитик дивергенций"

    def analyze(self, df, symbol=None):
        notes, marks = [], []
        verdict = 0.0
        if "rsi" not in df.columns or "obv" not in df.columns:
            return 0.0, ["нет RSI/OBV для проверки дивергенций"], []
        highs, lows = _swing_points(df, 5)
        if len(lows) >= 2:
            i1, i2 = lows[-2], lows[-1]
            if df["low"].iloc[i2] < df["low"].iloc[i1] and (
                    df["rsi"].iloc[i2] > df["rsi"].iloc[i1] or df["obv"].iloc[i2] > df["obv"].iloc[i1]):
                verdict += 0.6
                notes.append("бычья дивергенция по минимуму (цена ниже, RSI/OBV выше)")
                marks.append({"kind": "link", "x1": i1, "y1": df["low"].iloc[i1],
                              "x2": i2, "y2": df["low"].iloc[i2], "color": "#007a3d"})
        if len(highs) >= 2:
            i1, i2 = highs[-2], highs[-1]
            if df["high"].iloc[i2] > df["high"].iloc[i1] and (
                    df["rsi"].iloc[i2] < df["rsi"].iloc[i1] or df["obv"].iloc[i2] < df["obv"].iloc[i1]):
                verdict -= 0.6
                notes.append("медвежья дивергенция по максимуму (цена выше, RSI/OBV ниже)")
                marks.append({"kind": "link", "x1": i1, "y1": df["high"].iloc[i1],
                              "x2": i2, "y2": df["high"].iloc[i2], "color": "#c62828"})
        if not notes:
            notes.append("дивергенций на последних swing-точках не найдено")
        return max(-1.0, min(1.0, verdict)), notes, marks


class ReversalAnalyst:
    """Разворот тренда: слом последней структуры + подтверждение объёмом."""
    name = "Аналитик разворотов"

    def analyze(self, df, symbol=None):
        highs, lows = _swing_points(df, 5)
        notes, marks = [], []
        verdict = 0.0
        if len(highs) < 2 or len(lows) < 2:
            return 0.0, ["недостаточно данных для разворота"], []

        hh = df["high"].iloc[highs[-1]] > df["high"].iloc[highs[-2]]
        hl = df["low"].iloc[lows[-1]] > df["low"].iloc[lows[-2]]
        lh = df["high"].iloc[highs[-1]] < df["high"].iloc[highs[-2]]
        ll = df["low"].iloc[lows[-1]] < df["low"].iloc[lows[-2]]

        close = df["close"].iloc[-1]
        last_hl = df["low"].iloc[lows[-1]]
        last_lh = df["high"].iloc[highs[-1]]
        vol_z = float(df["vol_z"].iloc[-1]) if "vol_z" in df.columns else 0.0

        if hh and hl and close < last_hl:
            verdict = -0.8
            notes.append(f"разворот бык->медведь: закрытие ниже последнего higher low {last_hl:.6g}")
            marks.append({"kind": "line", "y": last_hl, "color": "#c62828", "style": "-.",
                          "text": "СЛОМ СТРУКТУРЫ"})
            if vol_z > 1:
                notes.append("слом на повышенном объёме — подтверждение")
        elif lh and ll and close > last_lh:
            verdict = 0.8
            notes.append(f"разворот медведь->бык: закрытие выше последнего lower high {last_lh:.6g}")
            marks.append({"kind": "line", "y": last_lh, "color": "#007a3d", "style": "-.",
                          "text": "СЛОМ СТРУКТУРЫ"})
            if vol_z > 1:
                notes.append("слом на повышенном объёме — подтверждение")
        else:
            trend = "восходящий" if (hh and hl) else ("нисходящий" if (lh and ll) else "боковик")
            notes.append(f"разворота нет: текущая структура — {trend}")
        return verdict, notes, marks


class OrderbookAnalyst:
    """Живой стакан: стенки и дисбаланс (работает только при переданном symbol)."""
    name = "Аналитик стакана"

    def analyze(self, df, symbol=None):
        if not symbol:
            return 0.0, ["символ не передан — стакан не анализируется"], []
        try:
            import orderbook_analyzer as oba
        except Exception:
            return 0.0, ["модуль orderbook_analyzer.py не найден"], []
        res = oba.intent(symbol)
        notes = [res["note"]]
        marks = []
        for w in res["bid_walls"][:1]:
            notes.append(f"bid-стенка {w['price']:.6g} (${w['usd'] / 1000:.0f}K, {w['dist_pct']:.2f}% от цены)")
            marks.append({"kind": "line", "y": w["price"], "color": "#00c853", "style": "-.",
                          "text": f"BID WALL {w['price']:.6g}"})
        for w in res["ask_walls"][:1]:
            notes.append(f"ask-стенка {w['price']:.6g} (${w['usd'] / 1000:.0f}K, {w['dist_pct']:.2f}% от цены)")
            marks.append({"kind": "line", "y": w["price"], "color": "#ff1744", "style": "-.",
                          "text": f"ASK WALL {w['price']:.6g}"})
        for e in res["events"][-3:]:
            notes.append(f"стенка {e['type']}: {e['side']} @ {e['price']:.6g}")
        return res["dir"] * 0.8, notes, marks


class CandleAnalyst:
    name = "Аналитик свечных паттернов"

    def analyze(self, df, symbol=None):
        notes, marks = [], []
        verdict = 0.0
        tail = df.tail(3).reset_index(drop=True)
        base_idx = len(df) - len(tail)
        for k, row in tail.iterrows():
            i = base_idx + k
            o, h, l, c = row["open"], row["high"], row["low"], row["close"]
            body = abs(c - o)
            rng = h - l
            if rng <= 0:
                continue
            up_w = h - max(o, c)
            dn_w = min(o, c) - l
            if body <= 0.25 * rng:
                notes.append(f"доджи на свече #{i} — неопределённость")
                marks.append({"kind": "point", "x": i, "y": h, "marker": "o", "color": "#b26a00"})
            elif dn_w >= 2 * body and up_w <= 0.3 * body:
                verdict += 0.4
                notes.append(f"молот/пин-бар на свече #{i}")
                marks.append({"kind": "point", "x": i, "y": l, "marker": "d", "color": "#007a3d"})
            elif up_w >= 2 * body and dn_w <= 0.3 * body:
                verdict -= 0.4
                notes.append(f"падающая звезда на свече #{i}")
                marks.append({"kind": "point", "x": i, "y": h, "marker": "d", "color": "#c62828"})
        if len(df) >= 2:
            p, c2 = df.iloc[-2], df.iloc[-1]
            if p["close"] < p["open"] and c2["close"] > c2["open"] and c2["close"] > p["open"] and c2["open"] < p["close"]:
                verdict += 0.5
                notes.append("бычье поглощение на последней свече")
                marks.append({"kind": "point", "x": len(df) - 1, "y": c2["low"], "marker": "d", "color": "#007a3d"})
            if p["close"] > p["open"] and c2["close"] < c2["open"] and c2["close"] < p["open"] and c2["open"] > p["close"]:
                verdict -= 0.5
                notes.append("медвежье поглощение на последней свече")
                marks.append({"kind": "point", "x": len(df) - 1, "y": c2["high"], "marker": "d", "color": "#c62828"})
        verdict = max(-1.0, min(1.0, verdict))
        if not notes:
            notes.append("значимых свечных паттернов нет")
        return verdict, notes, marks


class LevelsAnalyst:
    name = "Аналитик уровней и Volume Profile"

    def analyze(self, df, symbol=None):
        notes, marks = [], []
        verdict = 0.0
        centers, vols = czs.calculate_volume_profile(df, 40)
        if len(vols) == 0:
            return 0.0, ["нет данных профиля"], []
        order = np.argsort(vols)[::-1]
        poc = float(centers[order[0]])
        price = float(df["close"].iloc[-1])
        notes.append(f"POC ≈ {poc:.6g}")
        marks.append({"kind": "line", "y": poc, "color": "#ffb300", "style": "--"})
        for idx in order[1:3]:
            if vols[idx] > 0.5 * vols[order[0]]:
                notes.append(f"объёмный узел ≈ {centers[idx]:.6g}")
                marks.append({"kind": "line", "y": float(centers[idx]), "color": "#90a4ae", "style": ":"})
        if price > poc:
            verdict += 0.3; notes.append("цена выше POC — контроль у покупателей")
        else:
            verdict -= 0.3; notes.append("цена ниже POC — контроль у продавцов")
        return max(-1.0, min(1.0, verdict)), notes, marks


class MomentumAnalyst:
    name = "Аналитик моментума"

    def analyze(self, df, symbol=None):
        notes, marks = [], []
        verdict = 0.0
        close = df["close"]
        if "rsi" in df.columns:
            rsi = df["rsi"]
        else:
            delta = close.diff()
            gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
            loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
            rsi = 100 - 100 / (1 + gain / (loss + 1e-12))
        rsi_now = float(rsi.iloc[-1])
        roc = float(close.pct_change(10).iloc[-1]) * 100
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        hist = macd - macd.ewm(span=9, adjust=False).mean()
        hist_up = float(hist.iloc[-1]) > float(hist.iloc[-2])
        if rsi_now > 70:
            verdict -= 0.3; notes.append(f"RSI {rsi_now:.0f} — перекупленность")
        elif rsi_now < 30:
            verdict += 0.3; notes.append(f"RSI {rsi_now:.0f} — перепроданность")
        elif rsi_now > 50:
            verdict += 0.2; notes.append(f"RSI {rsi_now:.0f} — бычья половина")
        else:
            verdict -= 0.2; notes.append(f"RSI {rsi_now:.0f} — медвежья половина")
        if hist_up:
            verdict += 0.2; notes.append("гистограмма MACD растёт")
        else:
            verdict -= 0.2; notes.append("гистограмма MACD снижается")
        if roc > 10:
            verdict -= 0.2; notes.append(f"перегон +{roc:.1f}% за 10 свечей")
        elif roc < -10:
            verdict += 0.2; notes.append(f"перегон {roc:.1f}% за 10 свечей")
        return max(-1.0, min(1.0, verdict)), notes, marks


ANALYSTS = [StructureAnalyst(), DivergenceAnalyst(), ReversalAnalyst(),
            OrderbookAnalyst(), CandleAnalyst(), LevelsAnalyst(), MomentumAnalyst()]
WEIGHTS = [0.25, 0.20, 0.15, 0.15, 0.10, 0.10, 0.05]


def council(df, symbol=None):
    rows, marks = [], []
    score = 0.0
    for a, w in zip(ANALYSTS, WEIGHTS):
        v, notes, mk = a.analyze(df, symbol)
        score += v * w
        rows.append((a.name, v, notes))
        marks.extend(mk)
    if score > 0.35:
        consensus = "выраженный бычий перевес"
    elif score > 0.10:
        consensus = "умеренно бычий"
    elif score < -0.35:
        consensus = "выраженный медвежий перевес"
    elif score < -0.10:
        consensus = "умеренно медвежий"
    else:
        consensus = "нейтрально / диапазон"
    return {"score": score, "consensus": consensus, "rows": rows, "marks": marks}


def draw_marks(ax, df, symbol=None):
    res = council(df, symbol)
    for m in res["marks"]:
        k = m["kind"]
        if k == "point":
            ax.scatter([m["x"]], [m["y"]], marker=m["marker"], color=m["color"], s=40, zorder=5)
        elif k == "line":
            ax.axhline(m["y"], color=m["color"], ls=m.get("style", "--"), lw=1.0, alpha=0.8)
            if m.get("text"):
                ax.annotate(m["text"], xy=(len(df) - 1, m["y"]), fontsize=8,
                            color=m["color"], va="bottom", ha="right")
        elif k == "link":
            ax.plot([m["x1"], m["x2"]], [m["y1"], m["y2"]],
                    color=m["color"], lw=1.2, ls="--", alpha=0.9)
    return res


def report_text(symbol, res):
    lines = [f"=== Консилиум ИИ-аналитиков графика: {symbol} ==="]
    for name, v, notes in res["rows"]:
        side = "бычий" if v > 0.05 else ("медвежий" if v < -0.05 else "нейтральный")
        lines.append(f"\n{name}: {v:+.2f} ({side})")
        for n in notes:
            lines.append(f"   — {n}")
    lines.append("\n" + "-" * 60)
    lines.append(f"Координатор: консенсус {res['score']:+.2f} — {res['consensus']}")
    lines.append("Веса: структура 25%, дивергенции 20%, развороты 15%, стакан 15%,")
    lines.append("свечи 10%, уровни 10%, моментум 5%.")
    lines.append("Стакан — живые данные: стенки могут измениться за секунды.")
    return "\n".join(lines)
try:
    import drsi_v2
    ANALYSTS.append(drsi_v2.DrsiV2Analyst())
    WEIGHTS.append(0.10)
    _wsum = sum(WEIGHTS)
    WEIGHTS = [w / _wsum for w in WEIGHTS]
except Exception:
    pass
